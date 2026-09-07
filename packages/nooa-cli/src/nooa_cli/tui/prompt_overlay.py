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

from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.formatted_text import AnyFormattedText
from prompt_toolkit.layout import HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.processors import BeforeInput, PasswordProcessor

from .subapp import SubviewKeyResult
from .terminal_safety import safe_hyperlink_target, strip_format_controls

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


def _safe_web_link(value: str | None) -> str | None:
    """Return a strict HTTP(S) target for a link rendered as clickable.

    ``safe_hyperlink_target`` also accepts ``file://`` URLs, but the only
    callers here render MCP-provided authorization URLs: require HTTP(S)
    with a host, and reject Unicode format (Cf) characters outright.
    """
    target = safe_hyperlink_target(value)
    if target is None or strip_format_controls(target) != target:
        return None
    parsed = urlsplit(target)
    # ``netloc`` can be a bare port (``https://:443`` -> netloc=":443") while
    # ``hostname`` is None; require an actual host before rendering a link.
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    return target


def _link_fragments(url: str | None) -> list[tuple[str, str]]:
    """Return visible, clickable OSC-8 fragments for a safe URL."""
    target = _safe_web_link(url)
    if target is None:
        return [("class:fullscreen-browser.muted", "(URL unavailable — use Ctrl+Y to copy)")]
    return [
        _LINK_CLOSE,
        ("[ZeroWidthEscape]", f"\x1b]8;;{target}\x1b\\"),
        ("", target),
        _LINK_CLOSE,
    ]


def _separator() -> Window:
    return Window(char="─", height=1, style="class:fullscreen-browser.separator")


def _hyperlink_boundary_fragments(app: Any, parts: str) -> list[tuple[str, str]]:
    """Return footer fragments that close any open OSC-8 link state.

    prompt_toolkit only re-emits zero-width escapes for changed cells, so
    the first footer cell alternates an invisible class on every frame
    (via the app's render counter) to guarantee the close escape is
    re-emitted on each repaint.
    """
    marker = f"class:native-hyperlink-boundary-{app.render_counter & 1}"
    return [
        ("[ZeroWidthEscape]", "\x1b]8;;\x1b\\"),
        (f"class:fullscreen-browser.footer {marker}", " "),
        ("class:fullscreen-browser.footer", parts),
    ]


class _InputOverlayBase:
    """Shared chrome for container-hosted input overlays.

    Subclasses set ``value`` on completion and define their ``body_control``
    and ``footer_control``; this base owns the buffer, input window, title,
    chrome, focus, and the shared text-handling actions.
    """

    # Leave terminal mouse mode off so native selection/copy still works.
    mouse_support = False
    pending_input: str | None = None

    def __init__(
        self,
        app: Any,
        title: str,
        message: str,
        body_control: FormattedTextControl,
        footer_control: FormattedTextControl,
        *,
        default: str = "",
        masked: bool = False,
    ) -> None:
        self.app = app
        self.title = title
        self.message = strip_format_controls(message)

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

        self.container = HSplit(
            [
                Window(self.title_control, height=1),
                _separator(),
                Window(body_control, wrap_lines=True, height=Dimension(min=1, weight=1)),
                Window(char=" ", height=1),
                self.input_window,
                _separator(),
                Window(footer_control, height=1),
            ]
        )

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

    def _insert_text(self, value: str) -> None:
        # A bracketed paste may embed line breaks the single-line buffer
        # cannot edit; collapse each line boundary to one space while
        # preserving ordinary spaces (including a bare space keypress).
        self.buffer.insert_text(" ".join(value.splitlines()))

    def _handle_edit_action(self, action: str, value: str = "") -> bool:
        """Apply shared editing actions; return True when ``action`` matched."""
        buffer = self.buffer
        if action == "text":
            self._insert_text(value)
            return True
        if action == "backspace":
            buffer.delete_before_cursor(1)
            return True
        if action == "left":
            buffer.cursor_left()
            return True
        if action == "right":
            buffer.cursor_right()
            return True
        if action == "home":
            buffer.cursor_position = 0
            return True
        if action == "end":
            buffer.cursor_position = len(buffer.text)
            return True
        if action == "space":
            self._insert_text(" ")
            return True
        mapped = _TEXT_ACTIONS.get(action)
        if mapped is not None:
            self._insert_text(mapped)
            return True
        return False


class PromptOverlay(_InputOverlayBase):
    """Single-line input overlay with an optional clickable URL.

    Hosted by ``TUIApplication.open_subview`` exactly like the container-based
    browsers: ``container`` swaps in as the layout root, the real ``Buffer``
    owns the cursor, and the atomic teardown restores the composer with one
    final frame on close.
    """

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
        self.link_url = link_url
        self._copy_handler = copy_handler
        self._copy_status = ""
        self.value: str | None = None
        super().__init__(
            app,
            title,
            message,
            FormattedTextControl(self._body_text),
            FormattedTextControl(self._footer_text),
            default=default,
            masked=masked,
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
        return _hyperlink_boundary_fragments(self.app, parts)

    def handle_key(self, action: str, value: str = "") -> SubviewKeyResult:
        if action == "enter":
            self.value = self.buffer.text.strip()
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
        if self._handle_edit_action(action, value):
            return "handled"
        return "handled"


class ChoiceOverlay(_InputOverlayBase):
    """Searchable single-choice overlay on the container host pattern."""

    def __init__(self, app: Any, title: str, message: str, options: list[str]) -> None:
        if not options:
            raise ValueError("ChoiceOverlay requires at least one option")
        self.options = list(dict.fromkeys(options))
        self.value: str | None = None
        self.cursor = 0
        self._offset = 0
        self._page = 1
        super().__init__(
            app,
            title,
            message,
            FormattedTextControl(self._list_text),
            FormattedTextControl(
                lambda: [
                    (
                        "class:fullscreen-browser.footer",
                        " Type to filter   ↑/↓ select   Enter choose   Esc cancel",
                    )
                ]
            ),
        )

    def _matches(self) -> list[str]:
        needle = self.buffer.text.casefold()
        if not needle:
            return self.options
        return [option for option in self.options if needle in option.casefold()]

    def _clamped_matches(self) -> list[str]:
        """Return current matches with the cursor clamped into range.

        prompt_toolkit processes a burst of queued keys before the next
        repaint, so ``enter`` can arrive before the render-time clamp in
        ``_list_text`` ever runs. Filtering here keeps selection in range.
        """
        matches = self._matches()
        if matches:
            self.cursor = min(max(self.cursor, 0), len(matches) - 1)
        return matches

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

    def handle_key(self, action: str, value: str = "") -> SubviewKeyResult:
        if action == "enter":
            matches = self._clamped_matches()
            if matches:
                self.value = matches[self.cursor]
                return "close"
            return "handled"
        if action == "escape":
            self.value = None
            return "close"
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
        if self._handle_edit_action(action, value):
            self._clamped_matches()
            return "handled"
        return "handled"
