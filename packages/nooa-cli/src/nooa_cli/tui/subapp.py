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
from typing import Literal, Protocol

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


class SensitiveTextPromptView:
    """Single-line masked input hosted by the existing TUI application."""

    _TEXT_ACTIONS = {
        "quit": "q",
        "resume": "r",
        "slash": "/",
        "j": "j",
        "k": "k",
    }

    def __init__(self, title: str, message: str) -> None:
        self.title = title
        self.message = message
        self._buffer = ""
        self._scroll = 0
        self._max_scroll = 0
        self._page_size = 1
        self.value: str | None = None

    def render(self, width: int, height: int) -> str:
        width = max(int(width), 40)
        height = max(int(height), 4)
        header = f" {self.title} ".ljust(width, "─")[:width]
        footer = " ↑/↓ scroll  Enter submit  Esc cancel ".ljust(width, "─")[:width]
        message_lines: list[str] = []
        safe_message = "".join(
            character
            for character in self.message
            if character in "\n\t"
            or (
                ord(character) >= 32
                and not 127 <= ord(character) <= 159
                and unicodedata.category(character) != "Cf"
            )
        )
        for paragraph in safe_message.splitlines():
            message_lines.extend(textwrap.wrap(paragraph, width=width) or [""])
        body_height = max(height - 2, 0)
        message_height = max(body_height - 2, 0)
        self._page_size = max(message_height, 1)
        self._max_scroll = max(len(message_lines) - message_height, 0)
        self._scroll = min(max(self._scroll, 0), self._max_scroll)
        visible = message_lines[self._scroll : self._scroll + message_height]
        visible.extend("" for _ in range(max(message_height - len(visible), 0)))
        visible.extend(("", "> " + ("•" * len(self._buffer))))
        return "\n".join([header, *visible, footer])

    def handle_key(self, action: str, value: str = "") -> SubviewKeyResult:
        if action == "enter":
            self.value = self._buffer.strip()
            return "close"
        if action == "escape":
            self.value = None
            return "close"
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
