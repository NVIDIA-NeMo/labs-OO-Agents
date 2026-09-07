# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Container-based input overlays hosted like the /resume and /todos browsers.

These prompts are real prompt_toolkit containers: a ``BufferControl`` owns the
cursor and editing natively, so typing, pasting, cursor movement, and the
open/close repaints behave exactly like the fullscreen browsers the rest of
the app already uses. They replace the legacy text-rendered prompt views,
which projected ANSI into a cursorless control and emulated editing by hand —
the two behaviors users saw as the OAuth cursor fight.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable
from typing import Any

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.formatted_text import AnyFormattedText
from prompt_toolkit.layout import HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.processors import BeforeInput, PasswordProcessor

from .subapp import SubviewKeyResult
from .terminal_safety import safe_hyperlink_target


def _safe_terminal_text(value: str) -> str:
    """Strip terminal controls and bidi/formatting characters from server text.

    ``sanitize_live_text`` removes ANSI; this additionally removes Unicode Cf
    (bidi override, joiners) so untrusted messages cannot reorder the prompt.
    """
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


# Letters the host application binds as navigation actions inside subviews.
# An input overlay treats them as ordinary text so typing still works.
_TEXT_ACTIONS = {
    "quit": "q",
    "resume": "r",
    "slash": "/",
    "j": "j",
    "k": "k",
}

_LINK_CLOSE = ("[ZeroWidthEscape]", "\x1b]8;;\x1b\\")


def _link_fragments(url: str | None) -> list[tuple[str, str]]:
    """Return visible, clickable OSC-8 fragments for a safe URL."""
    target = safe_hyperlink_target(url)
    if target is None:
        return [("class:fullscreen-browser.muted", "(URL unavailable — see scrollback)")]
    return [
        _LINK_CLOSE,
        ("[ZeroWidthEscape]", f"\x1b]8;;{target}\x1b\\"),
        ("", target),
        _LINK_CLOSE,
    ]


class PromptOverlay:
    """Single-line input overlay with an optional clickable URL.

    Hosted by ``TUIApplication.open_subview`` exactly like the container-based
    browsers: ``container`` swaps in as the layout root, the real ``Buffer``
    owns the cursor, and the atomic teardown restores the composer with one
    final frame on close.
    """

    # Leave terminal mouse mode off so native selection/copy still works.
    mouse_support = False
    pending_input: str | None = None

    def __init__(
        self,
        app: Any,
        title: str,
        message: str,
        *,
        default: str = "",
        masked: bool = False,
        link_url: str | None = None,
        copy_handler: Callable[[str], bool] | None = None,
    ) -> None:
        self.app = app
        self.title = title
        self.message = _safe_terminal_text(message)
        self.link_url = link_url
        self._copy_handler = copy_handler
        self._copy_status = ""
        self.value: str | None = None

        self.buffer = Buffer(multiline=False)
        if default:
            self.buffer.insert_text(default)
        # PasswordProcessor first masks the buffer text; BeforeInput then
        # prepends the visible prompt marker so it is not masked.
        processors = [PasswordProcessor(char="•")] if masked else []
        processors.append(BeforeInput("❯ ", style="class:prompt"))
        self.input_control = BufferControl(self.buffer, input_processors=processors)
        self.input_window = Window(self.input_control, height=1, style="class:input-area")

        self.title_control = FormattedTextControl(
            lambda: [("class:fullscreen-browser.title", f" {self.title} ")]
        )
        self.body_control = FormattedTextControl(self._body_text)
        self.body_window = Window(
            self.body_control, wrap_lines=True, height=Dimension(min=1, weight=1)
        )
        self.footer_control = FormattedTextControl(self._footer_text)

        self.container = HSplit(
            [
                Window(self.title_control, height=1),
                Window(char="─", height=1, style="class:fullscreen-browser.separator"),
                self.body_window,
                Window(char=" ", height=1),
                self.input_window,
                Window(char="─", height=1, style="class:fullscreen-browser.separator"),
                Window(self.footer_control, height=1),
            ]
        )

    def _body_text(self) -> AnyFormattedText:
        fragments: list[tuple[str, str]] = []
        for line in self.message.splitlines() or [""]:
            fragments.append(("", line))
            fragments.append(("", "\n"))
        if self.link_url is not None:
            fragments.append(("", "\n"))
            fragments.extend(_link_fragments(self.link_url))
        else:
            fragments.pop()
        return fragments

    def _footer_text(self) -> AnyFormattedText:
        parts = " Enter submit   Esc cancel"
        if self.link_url:
            parts = " Ctrl+Y copy URL" + parts
        if self._copy_status:
            parts = f" {self._copy_status}" + parts
        # Close any open OSC-8 link state before painting the footer. The
        # alternating class forces prompt_toolkit to consider the first cell
        # changed each frame so the close escape is re-emitted on repaints.
        marker = f"class:native-hyperlink-boundary-{self.app.render_counter & 1}"
        return [
            ("[ZeroWidthEscape]", "\x1b]8;;\x1b\\"),
            (f"class:fullscreen-browser.footer {marker}", " "),
            ("class:fullscreen-browser.footer", parts),
        ]

    def on_open(self) -> None:
        try:
            self.app.layout.focus(self.input_window)
        except Exception:
            pass

    def on_close(self) -> None:
        pass

    def render(self, width: int, height: int) -> str:  # pragma: no cover - container host
        """Container-based views never render through the ANSI subview path."""
        return ""

    def handle_key(self, action: str, value: str = "") -> SubviewKeyResult:
        buffer = self.buffer
        if action == "enter":
            self.value = buffer.text.strip()
            return "close"
        if action == "escape":
            self.value = None
            return "close"
        if action == "copy" and self.link_url is not None:
            copied = False
            if self._copy_handler is not None:
                try:
                    copied = bool(self._copy_handler(str(self.link_url)))
                except Exception:
                    copied = False
            self._copy_status = "URL copied" if copied else "URL copy unavailable"
            return "handled"
        if action == "text":
            buffer.insert_text(value)
            return "handled"
        if action == "backspace":
            buffer.delete_before_cursor(1)
            return "handled"
        if action == "left":
            buffer.cursor_left()
            return "handled"
        if action == "right":
            buffer.cursor_right()
            return "handled"
        if action == "home":
            buffer.cursor_position = 0
            return "handled"
        if action == "end":
            buffer.cursor_position = len(buffer.text)
            return "handled"
        if action == "space":
            buffer.insert_text(" ")
            return "handled"
        mapped = _TEXT_ACTIONS.get(action)
        if mapped is not None:
            buffer.insert_text(mapped)
            return "handled"
        return "handled"


class ChoiceOverlay:
    """Searchable single-choice overlay on the container host pattern."""

    mouse_support = False
    pending_input: str | None = None

    def __init__(self, app: Any, title: str, message: str, options: list[str]) -> None:
        if not options:
            raise ValueError("ChoiceOverlay requires at least one option")
        self.app = app
        self.title = title
        self.message = _safe_terminal_text(message)
        self.options = list(dict.fromkeys(options))
        self.value: str | None = None
        self.cursor = 0
        self._offset = 0
        self._page = 1

        self.buffer = Buffer(multiline=False)
        self.input_control = BufferControl(
            self.buffer, input_processors=[BeforeInput("❯ ", style="class:prompt")]
        )
        self.input_window = Window(self.input_control, height=1, style="class:input-area")

        self.title_control = FormattedTextControl(
            lambda: [("class:fullscreen-browser.title", f" {self.title} ")]
        )
        self.list_control = FormattedTextControl(self._list_text)
        self.list_window = Window(
            self.list_control, wrap_lines=False, height=Dimension(min=1, weight=1)
        )
        self.footer_control = FormattedTextControl(
            lambda: [
                (
                    "class:fullscreen-browser.footer",
                    " Type to filter   ↑/↓ select   Enter choose   Esc cancel",
                )
            ]
        )

        self.container = HSplit(
            [
                Window(self.title_control, height=1),
                Window(char="─", height=1, style="class:fullscreen-browser.separator"),
                Window(
                    FormattedTextControl(lambda: [("", self.message)]),
                    wrap_lines=True,
                    height=Dimension(min=1, max=3),
                ),
                self.list_window,
                Window(char=" ", height=1),
                self.input_window,
                Window(char="─", height=1, style="class:fullscreen-browser.separator"),
                Window(self.footer_control, height=1),
            ]
        )

    def _matches(self) -> list[str]:
        needle = self.buffer.text.casefold()
        if not needle:
            return self.options
        return [option for option in self.options if needle in option.casefold()]

    def _list_text(self) -> AnyFormattedText:
        matches = self._matches()
        try:
            size = self.app.output.get_size()
            page = max(int(size.rows) - 6, 1)
        except Exception:
            page = 20
        self._page = page
        if not matches:
            self.cursor = 0
            self._offset = 0
            return [("class:fullscreen-browser.muted", "  (no matching options)")]
        self.cursor = min(max(self.cursor, 0), len(matches) - 1)
        if self.cursor < self._offset:
            self._offset = self.cursor
        elif self.cursor >= self._offset + page:
            self._offset = self.cursor - page + 1
        fragments: list[tuple[str, str]] = []
        for index in range(self._offset, min(self._offset + page, len(matches))):
            marker = "❯ " if index == self.cursor else "  "
            style = "class:fullscreen-browser.selected" if index == self.cursor else ""
            fragments.append((style, f"{marker}{matches[index]}"))
            fragments.append(("", "\n"))
        if fragments:
            fragments.pop()
        return fragments

    def on_open(self) -> None:
        try:
            self.app.layout.focus(self.input_window)
        except Exception:
            pass

    def on_close(self) -> None:
        pass

    def render(self, width: int, height: int) -> str:  # pragma: no cover - container host
        return ""

    def handle_key(self, action: str, value: str = "") -> SubviewKeyResult:
        if action == "enter":
            matches = self._matches()
            if matches:
                self.value = matches[self.cursor]
                return "close"
            return "handled"
        if action == "escape":
            self.value = None
            return "close"
        if action == "text":
            self.buffer.insert_text(value)
            return "handled"
        if action == "backspace":
            self.buffer.delete_before_cursor(1)
            return "handled"
        if action == "space":
            self.buffer.insert_text(" ")
            return "handled"
        if action in ("down", "scroll_down"):
            matches = self._matches()
            if matches:
                self.cursor = min(self.cursor + 1, len(matches) - 1)
            return "handled"
        if action in ("up", "scroll_up"):
            self.cursor = max(self.cursor - 1, 0)
            return "handled"
        if action == "home":
            self.cursor = 0
            return "handled"
        if action == "end":
            self.cursor = max(len(self._matches()) - 1, 0)
            return "handled"
        mapped = _TEXT_ACTIONS.get(action)
        if mapped is not None:
            self.buffer.insert_text(mapped)
            return "handled"
        return "handled"
