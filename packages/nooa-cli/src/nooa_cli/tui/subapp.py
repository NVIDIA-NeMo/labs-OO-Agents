# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable in-app subview primitives for the terminal TUI.

A subview is not a nested prompt_toolkit Application. It is a lightweight
state/render/key object hosted by the single long-lived TUIApplication so
terminal ownership, resize handling, mouse input, and focus remain centralized.

Subview conventions:

- ``q`` closes the subview.
- ``Esc`` is contextual inside the subview (clear/cancel/back), not the generic
  close key.
- The host owns resize handling; subviews render to the supplied width/height and
  must not launch their own prompt_toolkit Application or terminal-size poller.
"""

from __future__ import annotations

import textwrap
import unicodedata
from collections.abc import Callable
from typing import Literal, Protocol
from urllib.parse import urlsplit

SubviewKeyResult = Literal["handled", "close", "ignored"]


class InAppSubview(Protocol):
    """A modal/browseable view hosted inside the main TUIApplication."""

    title: str

    def render(self, width: int, height: int) -> str:
        """Render this view as an ANSI/plain terminal frame."""

    def handle_key(self, action: str, value: str = "") -> SubviewKeyResult:
        """Handle a semantic key action from the host application."""

    def on_open(self) -> None:
        """Called after the view becomes active."""

    def on_close(self) -> None:
        """Called just before the view is removed."""


def normalize_key_result(result: SubviewKeyResult | bool | None) -> SubviewKeyResult:
    """Accept legacy bool-ish handlers while new views return explicit results."""
    if result == "close":
        return "close"
    if result == "ignored" or result is False:
        return "ignored"
    return "handled"


def _safe_terminal_text(value: str) -> str:
    """Strip terminal controls and bidi formatting from user/server text."""
    return "".join(
        character
        for character in value
        if character in "\n\t"
        or (
            ord(character) >= 32
            and not 127 <= ord(character) <= 159
            and unicodedata.category(character) != "Cf"
        )
    )


def _safe_http_url(value: str | None) -> str | None:
    """Return a terminal-safe HTTP(S) URL, or ``None`` for an unsafe target."""
    if not value:
        return None
    safe = _safe_terminal_text(value).strip()
    if any(character.isspace() for character in safe):
        return None
    parsed = urlsplit(safe)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return safe


def _terminal_hyperlink(label: str, url: str) -> str:
    """Build an OSC 8 link wrapped as prompt_toolkit zero-width escapes."""
    start = f"\x01\x1b]8;;{url}\x07\x02"
    end = "\x01\x1b]8;;\x07\x02"
    return start + _safe_terminal_text(label) + end


class TextPromptView:
    """Single-line text input hosted by the existing TUI application."""

    # Prompt views do not consume mouse events. Leaving terminal mouse mode off
    # lets users use native selection/copy while the modal is open.
    mouse_support = False

    _TEXT_ACTIONS = {
        "quit": "q",
        "resume": "r",
        "slash": "/",
        "j": "j",
        "k": "k",
    }

    def __init__(
        self,
        title: str,
        message: str,
        *,
        default: str = "",
        masked: bool = False,
        link_url: str | None = None,
        copy_handler: Callable[[str], bool] | None = None,
    ) -> None:
        self.title = title
        self.message = message
        self.masked = masked
        self._buffer = default
        self._scroll = 0
        self._max_scroll = 0
        self._page_size = 1
        self.value: str | None = None
        self.link_url = _safe_http_url(link_url)
        self._copy_handler = copy_handler
        self._copy_status = ""

    def render(self, width: int, height: int) -> str:
        width = max(int(width), 40)
        height = max(int(height), 4)
        header = f" {self.title} ".ljust(width, "─")[:width]
        message_lines: list[str] = []
        safe_message = _safe_terminal_text(self.message)
        for paragraph in safe_message.splitlines():
            message_lines.extend(textwrap.wrap(paragraph, width=width) or [""])
        if self.link_url:
            message_lines.extend(
                ("", _terminal_hyperlink("Open authorization URL", self.link_url))
            )
        if self._copy_status:
            footer_text = f" {self._copy_status}  Enter submit  Esc cancel "
        elif self.link_url:
            footer_text = " ↑/↓ scroll  Ctrl+Y copy URL  Enter submit  Esc cancel "
        else:
            footer_text = " ↑/↓ scroll  Enter submit  Esc cancel "
        footer = footer_text.ljust(width, "─")[:width]
        body_height = max(height - 2, 0)
        message_height = max(body_height - 2, 0)
        self._page_size = max(message_height, 1)
        self._max_scroll = max(len(message_lines) - message_height, 0)
        self._scroll = min(max(self._scroll, 0), self._max_scroll)
        visible = message_lines[self._scroll : self._scroll + message_height]
        visible.extend("" for _ in range(max(message_height - len(visible), 0)))
        displayed = "•" * len(self._buffer) if self.masked else _safe_terminal_text(self._buffer)
        visible.extend(("", "> " + displayed))
        return "\n".join([header, *visible, footer])

    def handle_key(self, action: str, value: str = "") -> SubviewKeyResult:
        if action == "enter":
            self.value = self._buffer.strip()
            return "close"
        if action == "escape":
            self.value = None
            return "close"
        if action == "copy" and self.link_url:
            copied = False
            if self._copy_handler is not None:
                try:
                    copied = bool(self._copy_handler(self.link_url))
                except Exception:
                    copied = False
            self._copy_status = "URL copied" if copied else "URL copy unavailable"
            return "handled"
        if action == "backspace":
            self._buffer = self._buffer[:-1]
            return "handled"
        if action == "text":
            self._buffer += value
            return "handled"
        if action in ("down", "scroll_down"):
            self._scroll = min(self._scroll + 1, self._max_scroll)
            return "handled"
        if action in ("up", "scroll_up"):
            self._scroll = max(self._scroll - 1, 0)
            return "handled"
        if action == "page_down":
            self._scroll = min(self._scroll + self._page_size, self._max_scroll)
            return "handled"
        if action == "page_up":
            self._scroll = max(self._scroll - self._page_size, 0)
            return "handled"
        if action == "home":
            self._scroll = 0
            return "handled"
        if action == "end":
            self._scroll = self._max_scroll
            return "handled"
        mapped = self._TEXT_ACTIONS.get(action)
        if mapped is not None:
            self._buffer += mapped
            return "handled"
        return "handled"

    def on_open(self) -> None:
        pass

    def on_close(self) -> None:
        pass


class SensitiveTextPromptView(TextPromptView):
    """Backward-compatible masked prompt used for OAuth material."""

    def __init__(
        self,
        title: str,
        message: str,
        *,
        link_url: str | None = None,
        copy_handler: Callable[[str], bool] | None = None,
    ) -> None:
        super().__init__(
            title,
            message,
            masked=True,
            link_url=link_url,
            copy_handler=copy_handler,
        )


class ChoicePromptView:
    """Searchable single-choice prompt hosted by the existing TUI."""

    _TEXT_ACTIONS = TextPromptView._TEXT_ACTIONS

    def __init__(self, title: str, message: str, options: list[str]) -> None:
        if not options:
            raise ValueError("ChoicePromptView requires at least one option")
        self.title = title
        self.message = message
        self.options = list(dict.fromkeys(options))
        self.query = ""
        self.cursor = 0
        self.scroll = 0
        self._page_size = 1
        self.value: str | None = None

    def _matches(self) -> list[str]:
        needle = self.query.casefold()
        if not needle:
            return self.options
        return [option for option in self.options if needle in option.casefold()]

    def render(self, width: int, height: int) -> str:
        width = max(int(width), 40)
        height = max(int(height), 6)
        header = f" {self.title} ".ljust(width, "─")[:width]
        footer = " Type to filter  ↑/↓ select  Enter choose  Esc cancel ".ljust(width, "─")[:width]
        message = _safe_terminal_text(self.message)
        message_lines = textwrap.wrap(message, width=width) or [""]
        available = max(height - len(message_lines) - 3, 1)
        self._page_size = available
        matches = self._matches()
        if matches:
            self.cursor = min(max(self.cursor, 0), len(matches) - 1)
            if self.cursor < self.scroll:
                self.scroll = self.cursor
            elif self.cursor >= self.scroll + available:
                self.scroll = self.cursor - available + 1
        else:
            self.cursor = 0
            self.scroll = 0
        rows = []
        for index, option in enumerate(matches[self.scroll : self.scroll + available], self.scroll):
            marker = "❯" if index == self.cursor else " "
            rows.append(f"{marker} {_safe_terminal_text(option)}"[:width])
        if not rows:
            rows = ["  (no matching models)"]
        rows.extend("" for _ in range(max(available - len(rows), 0)))
        return "\n".join(
            [header, *message_lines, f"> {_safe_terminal_text(self.query)}", *rows, footer]
        )

    def handle_key(self, action: str, value: str = "") -> SubviewKeyResult:
        matches = self._matches()
        if action == "enter":
            self.value = matches[self.cursor] if matches else None
            return "close" if self.value is not None else "handled"
        if action == "escape":
            self.value = None
            return "close"
        if action == "backspace":
            self.query = self.query[:-1]
            self.cursor = self.scroll = 0
            return "handled"
        if action == "text":
            self.query += value
            self.cursor = self.scroll = 0
            return "handled"
        if action in ("down", "scroll_down", "tab") and matches:
            self.cursor = min(self.cursor + 1, len(matches) - 1)
            return "handled"
        if action in ("up", "scroll_up") and matches:
            self.cursor = max(self.cursor - 1, 0)
            return "handled"
        if action == "page_down" and matches:
            self.cursor = min(self.cursor + self._page_size, len(matches) - 1)
            return "handled"
        if action == "page_up" and matches:
            self.cursor = max(self.cursor - self._page_size, 0)
            return "handled"
        if action == "home":
            self.cursor = 0
            return "handled"
        if action == "end" and matches:
            self.cursor = len(matches) - 1
            return "handled"
        mapped = self._TEXT_ACTIONS.get(action)
        if mapped is not None:
            self.query += mapped
            self.cursor = self.scroll = 0
        return "handled"

    def on_open(self) -> None:
        pass

    def on_close(self) -> None:
        pass
