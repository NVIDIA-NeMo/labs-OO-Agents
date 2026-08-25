# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single long-lived ``prompt_toolkit.Application`` owning the whole TUI.

This is the "Plan C" rewrite: one Application that holds output
scrollback, the type-ahead queue region, the input buffer, and the
status line. No ``patch_stdout`` and no per-turn ``prompt_async`` —
so no handoff race that drops the first keystroke after the agent
finishes.

Grown incrementally against the failing tests in
``tests/cli/test_tui_app_behavior.py``. Each method exists because a
behaviour test needed it.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import logging
import os
import re
import shutil
import subprocess
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, AnyFormattedText
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.key_binding.bindings.scroll import scroll_page_down, scroll_page_up
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import (
    ConditionalContainer,
    DynamicContainer,
    HSplit,
    Layout,
    VSplit,
    Window,
)
from prompt_toolkit.layout.containers import WindowAlign
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl, UIContent
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.layout.menus import CompletionsMenuControl
from prompt_toolkit.layout.processors import BeforeInput
from prompt_toolkit.layout.screen import Char, Screen, WritePosition
from prompt_toolkit.mouse_events import (
    MouseButton,
    MouseEvent,
    MouseEventType,
    MouseModifier,
)
from prompt_toolkit.selection import SelectionType

from nooa_cli.interactive.state import (
    AgentLifecycle,
    CancellationState,
    InteractiveAgent,
)

from .agent_controller import AgentController
from .completer import expand_mentions
from .fullscreen_transcript import FullscreenTranscriptModel
from .host_services import TUIHostServices
from .input_handler import _set_completions_sync, create_prompt_style
from .resize_reflow import (
    TRANSCRIPT_REFLOW_DEBOUNCE_SECONDS,
    ResizeReplayRequest,
    TranscriptResizeState,
)
from .subapp import InAppSubview, normalize_key_result
from .terminal_safety import (
    normalize_transcript_block,
    project_prompt_toolkit_ansi,
    sanitize_live_text,
    sanitize_transcript_ansi,
    strip_safe_ansi,
)

logger = logging.getLogger(__name__)


class _ResizeAwareApplication(Application[Any]):
    """Let native replay fold SIGWINCH into its semantic resize transaction."""

    def __init__(
        self,
        *args: Any,
        defer_resize_redraw: Callable[[], bool],
        resize_redraw_is_deferred: Callable[[], bool],
        **kwargs: Any,
    ) -> None:
        self._defer_resize_redraw = defer_resize_redraw
        self._resize_redraw_is_deferred = resize_redraw_is_deferred
        super().__init__(*args, **kwargs)

    def _on_resize(self) -> None:
        if self._defer_resize_redraw():
            return
        self._run_with_deferred_flush(super()._on_resize)

    def _redraw(self, render_as_done: bool = False) -> None:
        if not render_as_done and self._resize_redraw_is_deferred():
            return
        super()._redraw(render_as_done=render_as_done)

    def redraw_after_deferred_resize(self) -> None:
        """Run prompt_toolkit's normal erase/redraw after resize settles."""
        self._run_with_deferred_flush(super()._on_resize)

    def _run_with_deferred_flush(self, callback: Callable[[], Any]) -> Any:
        """Buffer a prompt_toolkit terminal mutation and publish it once."""
        output = self.output
        physical_flush = output.flush

        def defer_flush() -> None:
            return None

        output.flush = defer_flush  # type: ignore[method-assign]
        try:
            result = callback()
        except BaseException:
            output.flush = physical_flush  # type: ignore[method-assign]
            try:
                physical_flush()
            except Exception:
                # Preserve the callback failure; callers cannot recover from a
                # secondary flush error more usefully than from its root cause.
                pass
            raise
        output.flush = physical_flush  # type: ignore[method-assign]
        physical_flush()
        return result

    def run_atomic_native_replay(self, replay: Callable[[Any], bool]) -> bool:
        """Erase, replay, and redraw through one physical output flush.

        ``run_in_terminal`` flushes its erase before invoking the callback. For
        a semantic transcript replay, this exposes an empty/intermediate frame
        before the rebuilt transcript and live region. This transaction runs
        synchronously on the UI loop and buffers every renderer flush before
        publishing the complete terminal update once.
        """
        output = self.output

        def update() -> bool:
            self.renderer.erase()
            replayed = replay(output)
            self.renderer.reset()
            self._request_absolute_cursor_position()
            # Bypass this class's deferred-redraw guard: this is the one final
            # live-region frame belonging to the resize transaction.
            super(_ResizeAwareApplication, self)._redraw()
            return replayed

        return bool(self._run_with_deferred_flush(update))


def _is_raw_mouse_report(data: str) -> bool:
    """Return whether raw input is a supported terminal mouse report."""
    return (
        data.startswith("\x1b[M")
        or data.startswith("\x1b[<")
        or re.fullmatch(r"\x1b\[\d+(?:;\d+){2}[Mm]", data) is not None
    )


class DispatcherExit(Exception):
    """Raised by handle() to signal the dispatcher should exit.

    Used by test harnesses. In the real TUI, exit is triggered by
    /exit → external task cancellation, not by the LLM.
    """


class _GraphemeWindow(Window):
    """Window that installs extended graphemes as atomic terminal cells.

    prompt_toolkit normally expands formatted text by code point. That gives
    flags, ZWJ emoji, modifiers, and keycaps the wrong width and can clip half
    a valid grapheme at the viewport edge. The transcript model already wraps
    on extended-grapheme boundaries; this final screen projection preserves
    those boundaries and supplies the terminal cluster width to the renderer.
    """

    def _copy_body(
        self,
        ui_content: UIContent,
        new_screen: Screen,
        write_position: WritePosition,
        move_x: int,
        width: int,
        vertical_scroll: int = 0,
        horizontal_scroll: int = 0,
        wrap_lines: bool = False,
        highlight_lines: bool = False,
        vertical_scroll_2: int = 0,
        always_hide_cursor: bool = False,
        has_focus: bool = False,
        align: WindowAlign = WindowAlign.LEFT,
        get_line_prefix: Callable[[int, int], AnyFormattedText] | None = None,
    ) -> tuple[dict[int, tuple[int, int]], dict[tuple[int, int], tuple[int, int]]]:
        mappings = super()._copy_body(
            ui_content,
            new_screen,
            write_position,
            move_x,
            width,
            vertical_scroll,
            horizontal_scroll,
            wrap_lines,
            highlight_lines,
            vertical_scroll_2,
            always_hide_cursor,
            has_focus,
            align,
            get_line_prefix,
        )
        visible_lines, _ = mappings
        grapheme_coordinates: dict[tuple[int, int], tuple[int, int]] = {}
        xpos = write_position.xpos + move_x
        ypos = write_position.ypos
        for screen_y, (line_number, _column) in visible_lines.items():
            if screen_y < 0 or screen_y >= write_position.height:
                continue
            row = new_screen.data_buffer[ypos + screen_y]
            for x in range(xpos, xpos + width):
                row[x] = Char()
            fragments = ui_content.get_line(line_number)
            styled_chars: list[tuple[str, str]] = []
            raw_at_offset: dict[int, str] = {}
            for style, text, *_rest in fragments:
                if "[ZeroWidthEscape]" in style:
                    raw_at_offset[len(styled_chars)] = raw_at_offset.get(
                        len(styled_chars), ""
                    ) + str(text)
                else:
                    styled_chars.extend((style, char) for char in text)

            # The superclass positions raw escapes using code-point widths. We
            # replace its coordinates because this window projects extended
            # graphemes atomically and can therefore use different cell widths.
            escape_row = new_screen.zero_width_escapes[ypos + screen_y]
            for screen_x in range(xpos, xpos + width + 1):
                escape_row.pop(screen_x, None)
            x = xpos
            logical_cell = 0
            for start, stop, cells in FullscreenTranscriptModel._grapheme_spans(styled_chars):
                cluster = "".join(char for _style, char in styled_chars[start:stop])
                if cluster == "\n":
                    break
                cells = max(0, cells)
                if cells > width:
                    # Keep source/export text intact, but never emit a glyph
                    # the physical terminal cannot fit in this viewport. A
                    # one-cell ellipsis keeps prompt_toolkit and the terminal's
                    # cursor model in agreement.
                    cluster = "…"
                    cells = 1
                elif x + cells > xpos + width:
                    break
                atom = Char(cluster, styled_chars[start][0] if start < stop else "")
                atom.width = cells
                if x < xpos + width:
                    if sequence := raw_at_offset.get(start):
                        escape_row[x] += sequence
                    row[x] = atom
                    for continuation in range(cells):
                        grapheme_coordinates[line_number, logical_cell + continuation] = (
                            ypos + screen_y,
                            x + continuation,
                        )
                    for continuation in range(1, cells):
                        row[x + continuation] = Char("")
                x += cells
                logical_cell += cells

        return visible_lines, grapheme_coordinates


class _FullscreenTranscriptControl(FormattedTextControl):
    """Transcript control owning wheel navigation and drag selection."""

    def __init__(
        self,
        *args: Any,
        scroll_callback: Callable[[int], None],
        mouse_navigation_enabled: Callable[[], bool],
        selection_callback: Callable[[str, int, int], None],
        link_callback: Callable[[int, int], bool],
        code_action_at: Callable[[int, int], str | None],
        copy_code_callback: Callable[[str], None],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._scroll_callback = scroll_callback
        self._mouse_navigation_enabled = mouse_navigation_enabled
        self._selection_callback = selection_callback
        self._link_callback = link_callback
        self._code_action_at = code_action_at
        self._copy_code_callback = copy_code_callback
        self._pressed_code_payload: str | None = None
        self._render_width: int | None = None
        self._render_height: int | None = None
        self._formatted_geometry: tuple[int, int | None] | None = None
        self._dragging = False
        self._drag_moved = False
        self._drag_position = (0, 0)
        self._autoscroll_direction = 0
        self._autoscroll_timer: asyncio.TimerHandle | None = None

    @property
    def render_size(self) -> tuple[int, int] | None:
        if self._render_width is None or self._render_height is None:
            return None
        return self._render_width, self._render_height

    def create_content(self, width: int, height: int | None):
        # ``preferred_height`` asks for content with ``height=None`` before the
        # concrete window height is allocated. Keep that measurement useful,
        # but do not let its render-counter cache poison the paint that follows
        # when bottom chrome changed this frame.
        self._render_width = max(1, width)
        if height is not None:
            self._render_height = max(1, height)
        geometry = (self._render_width, self._render_height)
        if geometry != self._formatted_geometry:
            self._fragment_cache.clear()
            self._formatted_geometry = geometry
        content = super().create_content(width, height)

        def get_line(index: int):
            line = content.get_line(index)
            if any(text for _style, text, *_rest in line):
                return line
            # prompt_toolkit only installs coordinate mappings for painted
            # cells. A visually blank space keeps logical blank transcript
            # rows mouse-addressable without changing model/exported text.
            return [("", " ")]

        return UIContent(
            get_line=get_line,
            line_count=content.line_count,
            cursor_position=content.cursor_position,
            menu_position=content.menu_position,
            show_cursor=content.show_cursor,
        )

    def mouse_handler(self, mouse_event: MouseEvent):
        # Terminals commonly reserve Option/Alt (or Shift in tmux) to bypass
        # mouse reporting and perform native selection. If such a modified
        # event is reported anyway, do not mutate application selection state.
        if (
            MouseModifier.ALT in mouse_event.modifiers
            or MouseModifier.SHIFT in mouse_event.modifiers
        ):
            # Stop any application drag/autoscroll without clearing the visible
            # selection; the modified gesture belongs to the terminal.
            self.cancel_drag()
            self._pressed_code_payload = None
            return NotImplemented
        if not self._mouse_navigation_enabled():
            self.cancel_drag()
            self._pressed_code_payload = None
            return NotImplemented

        delta = _fullscreen_wheel_delta(mouse_event)
        if delta is not None:
            self._scroll_callback(delta)
            return None

        x = mouse_event.position.x
        y = mouse_event.position.y
        if (
            mouse_event.event_type is MouseEventType.MOUSE_DOWN
            and mouse_event.button is MouseButton.LEFT
        ):
            self.cancel_drag()
            payload = self._code_action_at(x, y)
            if payload is not None:
                self._pressed_code_payload = payload
                return None
            self._pressed_code_payload = None
            self._dragging = True
            self._drag_position = (x, y)
            self._selection_callback("start", x, y)
            return None
        if (
            mouse_event.event_type is MouseEventType.MOUSE_MOVE
            and self._pressed_code_payload is not None
        ):
            if self._code_action_at(x, y) != self._pressed_code_payload:
                self._pressed_code_payload = None
            return None
        if (
            mouse_event.event_type is MouseEventType.MOUSE_UP
            and self._pressed_code_payload is not None
        ):
            payload = self._pressed_code_payload
            self._pressed_code_payload = None
            if self._code_action_at(x, y) == payload:
                self._copy_code_callback(payload)
            return None
        if mouse_event.event_type is MouseEventType.MOUSE_MOVE and self._dragging:
            # tmux cannot forward a release that happens outside its pane.  In
            # all-motion mode, the first event after the pointer re-enters is a
            # no-button move; treat that as the missing release instead of
            # leaving the drag lease and edge autoscroll stranded.
            if mouse_event.button is MouseButton.NONE:
                self._finish_drag(x, y, moved=True)
                return None
            self._drag_moved = True
            self._drag_position = (x, y)
            direction = 0
            if self._render_height is not None:
                if self._render_height == 1:
                    direction = 0
                elif self._render_height < 4:
                    distance_from_top = y
                    distance_from_bottom = self._render_height - 1 - y
                    if distance_from_top < distance_from_bottom:
                        direction = -1
                    elif distance_from_bottom < distance_from_top:
                        direction = 1
                elif y < 2:
                    direction = -1
                elif y >= self._render_height - 2:
                    direction = 1
            self._set_autoscroll(direction)
            self._selection_callback("extend", x, y)
            return None
        if mouse_event.event_type is MouseEventType.MOUSE_UP and self._dragging:
            moved = self._drag_moved or self._drag_position != (x, y)
            self._finish_drag(x, y, moved=moved)
            if not moved:
                self._link_callback(x, y)
            return None
        if mouse_event.event_type is MouseEventType.MOUSE_UP:
            self._pressed_code_payload = None
        return super().mouse_handler(mouse_event)

    @property
    def dragging(self) -> bool:
        """Whether this control currently owns an application selection drag."""
        return self._dragging

    def handle_external_mouse(self, mouse_event: MouseEvent, *, below: bool) -> bool:
        """Handle move/release routed to chrome beyond this control's window."""
        if not self._dragging:
            return False
        if (
            MouseModifier.ALT in mouse_event.modifiers
            or MouseModifier.SHIFT in mouse_event.modifiers
        ):
            self.cancel_drag()
            return False
        if mouse_event.event_type not in (
            MouseEventType.MOUSE_MOVE,
            MouseEventType.MOUSE_UP,
        ):
            return False
        width = max(1, self._render_width or 1)
        height = max(1, self._render_height or 1)
        # Coordinates are local to whichever chrome child received the event.
        # Once a downward drag leaves the transcript, clamp to its true lower-
        # right boundary instead of interpreting a right-side child's local x.
        x = width - 1 if below else 0
        y = height - 1 if below else 0
        self._drag_moved = True
        self._drag_position = (x, y)
        if mouse_event.event_type is MouseEventType.MOUSE_MOVE:
            if mouse_event.button is MouseButton.NONE:
                self._finish_drag(x, y, moved=True)
            else:
                self._set_autoscroll(1 if below else -1)
                self._selection_callback("extend", x, y)
        else:
            self._finish_drag(x, y, moved=True)
        return True

    def _finish_drag(self, x: int, y: int, *, moved: bool) -> None:
        """Resolve an owned drag at the latest observable pointer position."""
        self._dragging = False
        self._drag_moved = False
        self._drag_position = (x, y)
        self._set_autoscroll(0)
        self._selection_callback("finish" if moved else "cancel", x, y)

    def cancel_drag(self) -> None:
        """Cancel an active drag and any stationary edge autoscroll."""
        self._dragging = False
        self._drag_moved = False
        self._set_autoscroll(0)

    def _set_autoscroll(self, direction: int, *, delay: float = 0.35) -> None:
        if direction == self._autoscroll_direction and self._autoscroll_timer is not None:
            return
        self._autoscroll_direction = direction
        if self._autoscroll_timer is not None:
            self._autoscroll_timer.cancel()
            self._autoscroll_timer = None
        if not direction or not self._dragging:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._autoscroll_timer = loop.call_later(delay, self._autoscroll_tick)

    def _autoscroll_tick(self) -> None:
        self._autoscroll_timer = None
        if not self._dragging or not self._autoscroll_direction:
            return
        self._scroll_callback(self._autoscroll_direction)
        self._selection_callback("extend", *self._drag_position)
        self._set_autoscroll(self._autoscroll_direction, delay=0.12)


class _ComposerBuffer(Buffer):
    """Input buffer with conventional selection-aware editing semantics."""

    def insert_text(
        self,
        data: str,
        overwrite: bool = False,
        move_cursor: bool = True,
        fire_event: bool = True,
    ) -> None:
        if self.selection_state is not None:
            self.cut_selection()
        super().insert_text(
            data,
            overwrite=overwrite,
            move_cursor=move_cursor,
            fire_event=fire_event,
        )

    def delete(self, count: int = 1) -> str:
        if self.selection_state is not None:
            return self.cut_selection().text
        return super().delete(count=count)

    def delete_before_cursor(self, count: int = 1) -> str:
        if self.selection_state is not None:
            return self.cut_selection().text
        return super().delete_before_cursor(count=count)


def _fullscreen_wheel_delta(mouse_event: MouseEvent) -> int | None:
    """Map one prompt_toolkit wheel report to transcript visual rows."""
    return {
        MouseEventType.SCROLL_UP: -3,
        MouseEventType.SCROLL_DOWN: 3,
    }.get(mouse_event.event_type)


class _TranscriptGestureControlMixin:
    """Forward transcript-wide pointer gestures received by adjacent controls."""

    def __init__(
        self,
        *args: Any,
        transcript_drag_callback: Callable[[MouseEvent], bool] | None = None,
        transcript_scroll_callback: Callable[[int], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._transcript_drag_callback = transcript_drag_callback
        self._transcript_scroll_callback = transcript_scroll_callback

    def _forward_transcript_gesture(self, mouse_event: MouseEvent) -> tuple[bool, Any]:
        """Return whether the shared policy handled the event and its result."""
        if (
            MouseModifier.ALT in mouse_event.modifiers
            or MouseModifier.SHIFT in mouse_event.modifiers
        ):
            if self._transcript_drag_callback is not None:
                self._transcript_drag_callback(mouse_event)
            return True, NotImplemented
        if self._transcript_drag_callback is not None and self._transcript_drag_callback(
            mouse_event
        ):
            return True, None
        delta = _fullscreen_wheel_delta(mouse_event)
        if delta is not None and self._transcript_scroll_callback is not None:
            self._transcript_scroll_callback(delta)
            return True, None
        return False, None


class _FullscreenDragBoundaryControl(_TranscriptGestureControlMixin, FormattedTextControl):
    """Chrome control that hands pointer gestures back to the transcript.

    prompt_toolkit routes mouse events to the window under the pointer. Without
    this handoff, releasing over bottom chrome strands the transcript control's
    drag lease, and wheel gestures over chrome never reach transcript scrolling.
    """

    def mouse_handler(self, mouse_event: MouseEvent):
        handled, result = self._forward_transcript_gesture(mouse_event)
        if handled:
            return result
        return super().mouse_handler(mouse_event)


class _ComposerWindow(Window):
    """Input window whose wheel viewport can detach from the edit cursor."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._manual_wheel_scroll = False
        self.always_hide_cursor = Condition(self.manual_cursor_is_offscreen)

    def reset_manual_wheel_scroll(self) -> None:
        """Resume prompt_toolkit's normal keep-the-cursor-visible behavior."""
        self._manual_wheel_scroll = False

    def scroll_without_moving_cursor(self, delta: int) -> None:
        """Move the wrapped viewport by visual rows, leaving the buffer cursor alone."""
        info = self.render_info
        if info is None or not delta:
            return
        self._manual_wheel_scroll = True
        step = 1 if delta > 0 else -1
        for _ in range(abs(delta)):
            if step < 0:
                if self.vertical_scroll_2 > 0:
                    self.vertical_scroll_2 -= 1
                elif self.vertical_scroll > 0:
                    self.vertical_scroll -= 1
                    self.vertical_scroll_2 = info.get_height_for_line(self.vertical_scroll) - 1
            elif self._manual_rows_below(info) > info.window_height:
                line_height = info.get_height_for_line(self.vertical_scroll)
                if self.vertical_scroll_2 + 1 < line_height:
                    self.vertical_scroll_2 += 1
                elif self.vertical_scroll + 1 < info.content_height:
                    self.vertical_scroll += 1
                    self.vertical_scroll_2 = 0

    def _manual_rows_below(self, info: Any) -> int:
        """Return visual rows from the current manual viewport through EOF."""
        return (
            info.get_height_for_line(self.vertical_scroll)
            - self.vertical_scroll_2
            + sum(
                info.get_height_for_line(line_number)
                for line_number in range(self.vertical_scroll + 1, info.content_height)
            )
        )

    def _scroll(self, ui_content: UIContent, width: int, height: int) -> None:
        if not self._manual_wheel_scroll:
            super()._scroll(ui_content, width, height)
            return
        self.vertical_scroll = min(max(0, self.vertical_scroll), max(0, ui_content.line_count - 1))
        line_height = ui_content.get_height_for_line(
            self.vertical_scroll, width, self.get_line_prefix
        )
        self.vertical_scroll_2 = min(max(0, self.vertical_scroll_2), line_height - 1)

    def manual_cursor_is_offscreen(self) -> bool:
        """Return whether detached wheel scrolling currently hides the edit cursor."""
        info = self.render_info
        if not self._manual_wheel_scroll or info is None:
            return False
        cursor = info.ui_content.cursor_position
        rows_before_cursor = sum(
            info.get_height_for_line(line_number) for line_number in range(cursor.y)
        )
        cursor_rows_in_line = info.ui_content.get_height_for_line(
            cursor.y,
            info.window_width,
            self.get_line_prefix,
            slice_stop=cursor.x,
        )
        cursor_visual_row = rows_before_cursor + max(0, cursor_rows_in_line - 1)
        rows_before_viewport = (
            sum(
                info.get_height_for_line(line_number) for line_number in range(self.vertical_scroll)
            )
            + self.vertical_scroll_2
        )
        return not (
            rows_before_viewport <= cursor_visual_row < rows_before_viewport + info.window_height
        )


class _NativeSelectionBufferControl(_TranscriptGestureControlMixin, BufferControl):
    """Composer control that hands transcript-wide pointer gestures to their owner."""

    def __init__(
        self,
        *args: Any,
        transcript_drag_callback: Callable[[MouseEvent], bool] | None = None,
        transcript_scroll_callback: Callable[[int], None] | None = None,
        selection_copy_callback: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            transcript_drag_callback=transcript_drag_callback,
            transcript_scroll_callback=transcript_scroll_callback,
            **kwargs,
        )
        self._selection_copy_callback = selection_copy_callback

    def mouse_handler(self, mouse_event: MouseEvent):
        handled, result = self._forward_transcript_gesture(mouse_event)
        if handled:
            return result
        result = super().mouse_handler(mouse_event)
        if (
            mouse_event.event_type is MouseEventType.MOUSE_UP
            and self.buffer.selection_state is not None
            and self._selection_copy_callback is not None
        ):
            # macOS terminals consume Command-C themselves, but cannot see a
            # prompt_toolkit-owned selection. Mirror a completed mouse
            # selection to the system clipboard so Command-C has the expected
            # result without deleting or hiding the selected composer text.
            _document, clipboard_data = self.buffer.document.cut_selection()
            if clipboard_data.text:
                self._selection_copy_callback(clipboard_data.text)
        return result


class _NativeSelectionCompletionsMenuControl(
    _TranscriptGestureControlMixin, CompletionsMenuControl
):
    """Completion menu that preserves native selection and transcript gestures."""

    def mouse_handler(self, mouse_event: MouseEvent):
        handled, result = self._forward_transcript_gesture(mouse_event)
        if handled:
            return result
        return super().mouse_handler(mouse_event)


def _native_hyperlink_boundary(
    fragments: list[tuple[str, str]], *, render_counter: int
) -> list[tuple[str, str]]:
    """Close transcript OSC-8 state before painting an adjacent chrome row."""
    boundary = [("[ZeroWidthEscape]", "\x1b]8;;\x1b\\")]
    for index, (style, text) in enumerate(fragments):
        if not text:
            boundary.append((style, text))
            continue
        # prompt_toolkit emits zero-width escapes only when their cell is
        # repainted. Alternate one invisible class on the first chrome cell so
        # every frame emits the close, without invalidating the whole row.
        marker = f"class:native-hyperlink-boundary-{render_counter & 1}"
        boundary.append((f"{style} {marker}".strip(), text[:1]))
        if text[1:]:
            boundary.append((style, text[1:]))
        boundary.extend(fragments[index + 1 :])
        break
    return boundary


class _ReturnToTailControl(_TranscriptGestureControlMixin, FormattedTextControl):
    """One-row fullscreen affordance for resuming live transcript output."""

    def __init__(
        self,
        callback: Callable[[], None],
        *,
        has_new_agent_message: Callable[[], bool] = lambda: False,
        transcript_drag_callback: Callable[[MouseEvent], bool] | None = None,
        transcript_scroll_callback: Callable[[int], None] | None = None,
        render_counter: Callable[[], int] = lambda: 0,
    ) -> None:
        def _formatted_text() -> list[tuple[str, str]]:
            notice_visible = has_new_agent_message()
            notice = " New agent message   " if notice_visible else " " * 21
            fragments = [
                ("class:return-to-tail" if notice_visible else "", notice),
                ("class:return-to-tail", "↓ Return to bottom (Ctrl+End)"),
            ]
            return _native_hyperlink_boundary(
                fragments,
                render_counter=render_counter(),
            )

        super().__init__(
            _formatted_text,
            focusable=False,
            show_cursor=False,
            transcript_drag_callback=transcript_drag_callback,
            transcript_scroll_callback=transcript_scroll_callback,
        )
        self._callback = callback

    def mouse_handler(self, mouse_event: MouseEvent):
        handled, result = self._forward_transcript_gesture(mouse_event)
        if handled:
            return result
        if mouse_event.button is not MouseButton.LEFT:
            return NotImplemented
        if mouse_event.event_type is MouseEventType.MOUSE_DOWN:
            return None
        if mouse_event.event_type is MouseEventType.MOUSE_UP:
            self._callback()
            return None
        return NotImplemented


class _SubviewControl(FormattedTextControl):
    """Formatted subview content with position-aware wheel dispatch."""

    def __init__(
        self,
        *args: Any,
        mouse_callback: Callable[[str, int, int], bool],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._mouse_callback = mouse_callback

    def mouse_handler(self, mouse_event: MouseEvent):
        if (
            MouseModifier.ALT in mouse_event.modifiers
            or MouseModifier.SHIFT in mouse_event.modifiers
        ):
            return NotImplemented
        action = {
            MouseEventType.SCROLL_UP: "scroll_up",
            MouseEventType.SCROLL_DOWN: "scroll_down",
        }.get(mouse_event.event_type)
        if action is not None and self._mouse_callback(
            action, mouse_event.position.x, mouse_event.position.y
        ):
            return None
        return super().mouse_handler(mouse_event)


def _strip_ansi(text: str) -> str:
    return strip_safe_ansi(text)


def terminal_cols(default: int = 120, minimum: int = 20) -> int:
    """Live terminal column count, clamped to [``minimum``, ∞).

    Wrapped so every caller gets the same fallback behaviour
    (``(120, 24)`` on stat failure) and clamp. Used by the status-rule
    renderer here and by the block-rendering helpers in ``Session`` so
    rich text (user-message bars, full-width rules) spans the live
    width and doesn't hardcode 120.
    """
    try:
        return max(shutil.get_terminal_size((default, 24)).columns, minimum)
    except Exception:
        return default


def format_session_rule(cols: int, label: str = "") -> list[tuple[str, str]]:
    """Build the formatted-text fragments for the session rule.

    The rule is rendered at ``cols - 1`` so it never occupies the terminal's
    final column. A full-bleed line forces a cursor wrap to the next row on most
    terminals; when ``run_in_terminal`` (``emit_block``) repaints this
    non-full-screen app after a SIGWINCH resize, prompt_toolkit's erase is sized
    to the pre-resize frame and can't reclaim that wrapped cell — leaving a stale
    rule of the old width plus a blank gap line in the scrollback (the
    "resize clutter" bug). Reserving the last column keeps the rule on one row so
    the erase stays correct across resizes.
    """
    from rich.cells import cell_len, set_cell_size

    width = max(cols - 1, 1)
    label = sanitize_live_text(label).replace("\n", " ")
    label_width = cell_len(label)
    if label:
        # Clamp the whole line (fill + space + label) to ``width`` so an
        # over-long label can't push the rule into the final column and
        # re-introduce the resize residue. The dash branch needs room for at
        # least one dash plus the separating space, i.e. ``len(label) <=
        # width - 2``; otherwise (including ``len(label) == width - 1``, where
        # ``max(..., 1)`` would silently round the fill back up and re-bleed to
        # the final column) fall straight through to truncation, keeping the
        # label text.
        if label_width > width - 2:
            # Keep grapheme clusters intact. Reversing code points to preserve
            # the tail can detach combining marks and ZWJ emoji sequences.
            return [("class:rule.label", set_cell_size(label, width).rstrip())]
        dashes = width - label_width - 1  # >= 1 by the guard above
        return [
            ("class:rule", "─" * dashes + " "),
            ("class:rule.label", label),
        ]
    return [("class:rule", "─" * width)]


PROMPT_MARKER = "❯ "
_CTRL_C_EXIT_WINDOW_SECONDS = 2.0
_MIN_INTERRUPT_STATUS_SECONDS = 0.75
_TRANSCRIPT_CLEAR_SEQUENCE = "\x1b[r\x1b[0m\x1b[H\x1b[2J\x1b[3J\x1b[H"
_FULLSCREEN_TRANSCRIPT_MAX_RECORDS = 10_000
_FULLSCREEN_TRANSCRIPT_MAX_BYTES = 16 * 1024 * 1024


@dataclass
class TranscriptBlock:
    source: str
    replay: Callable[[], str] | None = None
    event_id: str | None = None
    tags: frozenset[str] = frozenset()
    keep: bool = False
    transcript_epoch: int = 0
    transcript_record_id: int | None = None
    resident_bytes: int = 0
    replay_cache: dict[int, str] = field(default_factory=dict)
    fullscreen_rendered: str | None = None
    code_copy_actions: dict[str, str] = field(default_factory=dict)
    agent_message: bool = False


@dataclass(frozen=True)
class _ResizeReplayQueueItem:
    request: ResizeReplayRequest
    transcript_blocks: tuple[TranscriptBlock, ...]
    transcript_epoch: int


@dataclass(frozen=True, slots=True, eq=False)
class _PendingInputHandoff:
    """One admitted submission awaiting transcript commit or withdrawal."""

    text: str


@dataclass(frozen=True)
class _ClearTranscriptQueueItem:
    transcript_epoch: int


def _coalesce_string_into_queue(inq: Any, text: str) -> None:
    """Push *text* onto *inq*, merging into the trailing item if it's a string.

    UX policy, lifted out of ``submit_message``: when a user types
    multiple lines in quick succession (Enter, type more, Enter), we
    want one composite multi-line item — not N tiny items the agent
    handles one-by-one. The trailing queued item is the merge target
    only if it's a ``str``; non-string items (anything a producer puts
    that isn't a typed message) are preserved unchanged.
    """
    tail = inq.pop_last()
    if isinstance(tail, str):
        inq.put(f"{tail}\n{text}")
        return
    if tail is not None:
        inq.put(tail)
    inq.put(text)


async def _stop_litellm_worker() -> None:
    """Stop litellm's global logging worker so its tasks don't outlive the loop."""
    try:
        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

        await GLOBAL_LOGGING_WORKER.stop()
    except Exception:
        pass


def _short_exception_message(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return text[:160]


class _CallbackScheduler:
    """Small UIScheduler adapter around a thread-safe callback marshal."""

    def __init__(self, schedule: Callable[[Callable[[], None]], None]) -> None:
        self._schedule = schedule

    def schedule(self, callback: Callable[[], None]) -> None:
        self._schedule(callback)


@dataclass(frozen=True, slots=True)
class _ClipboardResult:
    success: bool
    transport: str = ""
    reason: str = "clipboard unavailable"


class TUIApplication:
    """Owns a single, long-lived ``prompt_toolkit.Application`` for the TUI."""

    def __init__(
        self,
        *,
        agent: InteractiveAgent | None = None,
        host_services: TUIHostServices | None = None,
        on_command: Callable[[str], Awaitable[None] | None] | None = None,
        on_cancel_command: Callable[[], bool] | None = None,
        on_bang: Callable[[str], Awaitable[None] | None] | None = None,
        on_output: Callable[[Any], Awaitable[None] | None] | None = None,
        on_agent_activity: Callable[[], None] | None = None,
        completer: Completer | None = None,
        session_label: Callable[[], str] | None = None,
        config: Any = None,
        full_screen: bool | None = None,
        display_mode: Any = None,
        submission_guard: Callable[[], str | None] | None = None,
    ) -> None:
        """
        Args:
            agent: Host-neutral direct state/control boundary for the
                currently rendered agent.
            on_command: Called with the raw slash text (e.g. ``"/help"``)
                whenever the user submits one. Session wires this to its
                CommandRegistry. If omitted, commands still land in
                ``commands_dispatched()`` for introspection but nothing
                runs.
            on_cancel_command: Synchronously request cancellation of the
                active slash command. Returns whether a command accepted the
                request. Bare Esc falls back to interrupting the agent when false.
            on_bang: Called with the bang body (e.g. ``"echo hi"`` for
                ``!echo hi``). Session wires this to run_in_terminal +
                bash. If omitted, bang commands are only recorded in
                ``last_bang_command()``.
            on_output: Called with structured ``Output`` values emitted by
                dispatcher-level behavior. Session wires this to
                ``frontend.render(...)``.
            completer: Optional prompt_toolkit ``Completer`` for Tab
                completion. When omitted no completion is offered.
            full_screen: Deprecated compatibility boolean. True selects
                ``native-replay`` and false selects ``native`` when
                ``display_mode`` is omitted.
            display_mode: Resolved restart-only display mode. Fullscreen uses
                one alternate-screen Application-owned transcript renderer.
            submission_guard: Returns an actionable error when plain agent-bound
                input must be rejected. Slash and bang commands remain available.

        The per-message echo ("queued → accepted" transition, user-bar
        render, SessionUserMessage log) is wired on the agent's
        ``_user_messages_in`` Channel via ``set_on_get``. The
        dispatcher itself doesn't call back — that would double-fire
        the echo when the agent dequeues a message mid-turn.
        """
        from .config import DisplayMode, TUIConfig, resolve_display_mode

        mode_config: dict[str, Any] = {}
        if display_mode is not None:
            mode_config["display_mode"] = display_mode
        if full_screen is not None:
            mode_config["full_screen"] = full_screen
        resolved_display_mode = resolve_display_mode(TUIConfig(**mode_config))

        self.display_mode = resolved_display_mode
        self._is_fullscreen = resolved_display_mode is DisplayMode.FULLSCREEN
        self._agent = agent
        self._host_services = host_services or TUIHostServices()
        self._on_command = on_command
        self._on_cancel_command = on_cancel_command
        self._on_bang = on_bang
        self._on_output = on_output
        self._on_agent_activity = on_agent_activity
        self._session_label_fn: Callable[[], str] | None = session_label
        self._config = config
        self._submission_guard = submission_guard
        self._ctrl_c_exit_armed = False
        self._ctrl_c_exit_timer: asyncio.TimerHandle | None = None
        self._exit_hint_text = ""
        self._interrupting_agent_turn = False
        self._interrupt_status_acknowledged = False
        self._interrupt_status_started_at: float | None = None
        self._interrupt_status_clear_timer: asyncio.TimerHandle | None = None
        self._transient_status_text = ""
        self._transient_status_style = "class:status"
        self._transient_status_timer: asyncio.TimerHandle | None = None
        self._clipboard_task: asyncio.Task[None] | None = None
        self._link_task: asyncio.Task[None] | None = None

        self._agent_controller = AgentController(
            _CallbackScheduler(self._schedule_agent_callback),
            self._on_agent_change,
        )

        # Compatibility attribute used by the existing replay implementation.
        self.full_screen = resolved_display_mode is DisplayMode.NATIVE_REPLAY

        # ``output_buffer`` is the ANSI-stripped logical transcript used by
        # tests and printable-transcript callers. Source-bearing blocks below
        # are the single retained representation used for terminal replay.
        self.output_buffer = Buffer(read_only=False)
        self._status_region_occupied = False
        self._fullscreen_transcript = FullscreenTranscriptModel(
            show_trailing_blank=lambda: not self._status_region_occupied
        )
        # Fullscreen requests mouse reporting immediately so ordinary drag and
        # wheel gestures reach prompt_toolkit rather than terminal scrollback.
        # Option/Alt-drag can bypass reporting in supporting terminals; F6 is
        # the reliable escape hatch that disables application mouse handling.
        self._fullscreen_mouse_navigation = self._is_fullscreen
        self._has_unseen_agent_message = False
        # Retained transcript replay units. On resize we intentionally clear
        # the visible screen + terminal scrollback, then rewrite these blocks.
        self._transcript_blocks: list[TranscriptBlock] = []
        self._fullscreen_transcript_bytes = 0
        self._fullscreen_semantic_replay_count = 0
        self._transcript_epoch = 0
        self._next_transcript_record_id = 0
        self._untagged_replay_tail = 200

        # In-app subview host. These are modal views inside the single
        # prompt_toolkit Application, so resize/input remain owned by one app.
        self._active_subview: InAppSubview | None = None
        self._active_subview_done: asyncio.Future[None] | None = None
        self._subview_control: FormattedTextControl | None = None

        # Input window: where user keystrokes land. A caller (Session)
        # passes the real CommandRegistry-backed completer; otherwise
        # Tab produces no suggestions.
        from prompt_toolkit.completion import DummyCompleter

        self._completer = completer or DummyCompleter()
        self.input_buffer = _ComposerBuffer(
            multiline=True,
            completer=self._completer,
            complete_while_typing=False,
            accept_handler=self._accept_handler,
        )
        self.input_buffer.on_text_changed += self._on_input_text_changed
        self.input_buffer.on_cursor_position_changed += self._on_input_cursor_position_changed

        # History — a plain list of submitted strings and a cursor that
        # tracks Up/Down navigation. Simpler than prompt_toolkit's async
        # InMemoryHistory machinery, which requires juggling working_lines
        # and _load_history_task to survive Buffer.reset().
        self._history: list[str] = []
        self._history_cursor: int | None = None

        # Command routing. Slash (/foo) items are appended to
        # ``_commands_dispatched``; bang (!foo) items set
        # ``_last_bang_command`` and (in production) run via
        # ``run_in_terminal``. Tests read both via the accessor methods.
        self._commands_dispatched: list[str] = []
        self._last_bang_command: str | None = None
        # Set by _run_callback on the sync-error path; read by
        # _drain_next to bail out of a pathological "every queued
        # command raises" loop instead of dumping N stack traces.
        self._last_sync_callback_raised: bool = False

        # Status line fields — surfaced via status_text().
        self._session_label: str = ""
        self._spinner_frame: str = "⠋"
        self._spinner_frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        self._spinner_task: asyncio.Task | None = None
        self._command_status_text: str = ""
        self._command_queue_texts: list[str] = []
        # Admitted text remains visible here until its accepted transcript
        # block is committed. A worker may dequeue before the next paint.
        self._pending_input_handoff: list[_PendingInputHandoff] = []
        self._llm_probe_status_text: str = ""

        self._prompt_processor = BeforeInput(PROMPT_MARKER, style="class:prompt")
        # Many-producer, single-consumer path for transcript content:
        # emit_block() enqueues one ANSI chunk; a single background task
        # (started in run_async) drains the queue in order and writes
        # each chunk via run_in_terminal → sys.__stdout__. Everything
        # that used to have its own scheduling (patch_stdout proxy,
        # direct run_in_terminal in _render_message, etc.) now funnels
        # through this one queue — no races.
        self._block_queue: (
            asyncio.Queue[TranscriptBlock | _ResizeReplayQueueItem | _ClearTranscriptQueueItem]
            | None
        ) = None
        self._consumer_task: asyncio.Task | None = None
        # Diagnostic/test counter for fullscreen clear+rewrite replays. Resize
        # state distinguishes transient geometry observations from the width
        # that actually rebuilt scrollback.
        self._fullscreen_invalidate_count = 0
        self._resize_reflow = TranscriptResizeState()
        self._resize_replay_timer: asyncio.TimerHandle | None = None
        self._fullscreen_rebuild_timer: asyncio.TimerHandle | None = None
        self._fullscreen_rebuild_generation = 0
        self._resize_replay_schedule_generation = 0
        self._queued_resize_replay_generation: int | None = None
        self._resize_replay_failure_generation: int | None = None
        # Resize-aware modes suppress prompt_toolkit's immediate SIGWINCH
        # redraw so transient geometry collapses into one settled frame.
        self._resize_redraw_deferred = False
        # A real height shrink can compress the non-full-screen live region
        # below its preferred height.  prompt_toolkit erases using a cursor
        # offset captured before SIGWINCH, so one rebuild is required after the
        # normal layout fits again even though transcript wrapping did not
        # change.
        self._height_compaction_needs_replay = False
        self._resize_replays_enabled = False
        self._replay_columns_override: int | None = None
        # Captured in run_async; used by emit_block for thread-safe
        # enqueue without calling the deprecated asyncio.get_event_loop().
        self._loop: asyncio.AbstractEventLoop | None = None
        # Set by run_async's stdout/stderr forwarder install; called in
        # the finally to restore the real streams.
        self._uninstall_stream_capture: Callable[[], None] | None = None

        kb = self._build_key_bindings()

        # Fullscreen owns transcript cells inside the Application. Compatibility
        # modes keep their historical native-scrollback output path exactly.
        self._output_window = (
            _GraphemeWindow(
                _FullscreenTranscriptControl(
                    lambda: self._fullscreen_transcript.formatted_text(
                        width=self._transcript_viewport_size()[0],
                        height=self._transcript_viewport_size()[1],
                        render_counter=self._app.render_counter,
                    ),
                    focusable=False,
                    show_cursor=False,
                    scroll_callback=self._scroll_fullscreen_transcript,
                    mouse_navigation_enabled=lambda: self._fullscreen_mouse_navigation,
                    selection_callback=self._handle_fullscreen_selection,
                    link_callback=self._open_fullscreen_link_at,
                    code_action_at=self._fullscreen_code_action_at,
                    copy_code_callback=self._start_fullscreen_selection_copy,
                ),
                wrap_lines=False,
                # The model virtualizes formatted content to exactly the visible
                # rows.  prompt_toolkit therefore renders from row zero rather
                # than rescanning/skipping the full retained document.
                get_vertical_scroll=lambda _window: 0,
                always_hide_cursor=True,
            )
            if self._is_fullscreen
            else None
        )

        # Queue chrome is a pure projection of runtime state plus the short
        # admission→transcript visibility handoff.
        def _queue_pending() -> list[str]:
            return self._pending_input_display()

        def _queue_formatted():
            rows = []
            command_queue = list(self._command_queue_texts)
            if command_queue:
                noun = "command" if len(command_queue) == 1 else "commands"
                rows.append(f"│ {len(command_queue)} {noun} queued")
                for index, text in enumerate(command_queue):
                    branch = "└─" if index == len(command_queue) - 1 else "├─"
                    rows.append(f"{branch} {sanitize_live_text(text)}")
            for text in _queue_pending():
                for line in sanitize_live_text(str(text)).split("\n"):
                    rows.append(f"│ {line}")
            if not rows:
                return []
            fragments = [("class:queue", "\n".join(rows))]
            if self._is_fullscreen:
                return _native_hyperlink_boundary(
                    fragments, render_counter=self._app.render_counter
                )
            return fragments

        queue_window = ConditionalContainer(
            Window(
                _FullscreenDragBoundaryControl(
                    _queue_formatted,
                    focusable=False,
                    transcript_drag_callback=self._handle_fullscreen_drag_over_bottom_chrome,
                    transcript_scroll_callback=(
                        self._scroll_fullscreen_transcript if self._is_fullscreen else None
                    ),
                ),
                wrap_lines=True,
                dont_extend_height=True,
            ),
            filter=Condition(lambda: bool(_queue_pending()) or bool(self._command_queue_texts)),
        )

        input_style = "class:input-area"
        input_window = _ComposerWindow(
            _NativeSelectionBufferControl(
                self.input_buffer,
                input_processors=[self._prompt_processor],
                transcript_drag_callback=(
                    self._handle_fullscreen_drag_over_bottom_chrome if self._is_fullscreen else None
                ),
                transcript_scroll_callback=(
                    self._scroll_fullscreen_input_or_transcript if self._is_fullscreen else None
                ),
                selection_copy_callback=(
                    self._copy_input_selection if self._is_fullscreen else None
                ),
            ),
            wrap_lines=True,
            height=(Dimension(min=1, max=10) if self._is_fullscreen else Dimension(min=1)),
            dont_extend_height=True,
            style=input_style,
        )
        self._input_window = input_window
        optional_row = Dimension(min=0, preferred=1, max=1)

        def _input_padding_window() -> Window:
            return Window(
                _FullscreenDragBoundaryControl(
                    lambda: [("", " ")],
                    focusable=False,
                    transcript_drag_callback=self._handle_fullscreen_drag_over_bottom_chrome,
                    transcript_scroll_callback=(
                        self._scroll_fullscreen_transcript if self._is_fullscreen else None
                    ),
                ),
                height=optional_row,
                style=input_style,
            )

        self._input_container = HSplit(
            [
                _input_padding_window(),
                input_window,
                _input_padding_window(),
            ],
            style=input_style,
            # The children above have a one-row aggregate minimum.  Keep a
            # blank final fallback as a belt-and-suspenders guard for terminal
            # implementations that briefly report zero rows.
            window_too_small=Window(),
        )

        # Status line at the bottom — shows spinner + session label.
        def _status_formatted():
            fragments: list[tuple[str, str]] = []
            rows = self._status_rows(include_transient=not self._is_fullscreen)
            for index, row in enumerate(rows):
                if index:
                    fragments.append(("class:status", "   " if self._is_fullscreen else "\n\n"))
                fragments.extend((style, sanitize_live_text(text)) for style, text in row)
            if self._is_fullscreen:
                return _native_hyperlink_boundary(
                    fragments, render_counter=self._app.render_counter
                )
            return fragments

        def _status_height() -> Dimension:
            if self._is_fullscreen:
                return Dimension(min=0, max=1, preferred=1)
            lines = self.status_text().splitlines()
            height = max(1, len(lines))
            # Status is useful chrome, not a reason to replace the entire UI
            # with prompt_toolkit's emergency "Window too small" window.
            return Dimension(min=0, max=height, preferred=height)

        self._status_control = _FullscreenDragBoundaryControl(
            _status_formatted,
            focusable=False,
            transcript_drag_callback=self._handle_fullscreen_drag_over_bottom_chrome,
            transcript_scroll_callback=(
                self._scroll_fullscreen_transcript if self._is_fullscreen else None
            ),
        )
        status_window = Window(
            self._status_control,
            height=_status_height,
        )

        self._transient_status_container = ConditionalContainer(
            Window(
                _FullscreenDragBoundaryControl(
                    lambda: _native_hyperlink_boundary(
                        [
                            ("class:status", " "),
                            (
                                self._transient_status_style,
                                sanitize_live_text(self._transient_status_text),
                            ),
                        ],
                        render_counter=self._app.render_counter,
                    ),
                    focusable=False,
                    transcript_drag_callback=self._handle_fullscreen_drag_over_bottom_chrome,
                    transcript_scroll_callback=self._scroll_fullscreen_transcript,
                ),
                height=1,
                dont_extend_width=True,
            ),
            filter=Condition(lambda: self._is_fullscreen and bool(self._transient_status_text)),
        )

        self._return_to_tail_control = _ReturnToTailControl(
            self._jump_fullscreen_to_tail,
            has_new_agent_message=lambda: self._has_unseen_agent_message,
            transcript_drag_callback=self._handle_fullscreen_drag_over_bottom_chrome,
            transcript_scroll_callback=self._scroll_fullscreen_transcript,
            render_counter=lambda: self._app.render_counter,
        )
        self._return_to_tail_container = ConditionalContainer(
            Window(
                self._return_to_tail_control,
                height=1,
                dont_extend_width=True,
            ),
            filter=Condition(
                lambda: (
                    self._is_fullscreen and not self._fullscreen_transcript.viewport.follows_tail
                )
            ),
        )

        # Session rule: right above the input, always visible. Shows the
        # session name + short uuid + context-usage label, right-aligned
        # on a horizontal rule. Built from formatted text (not a Rich
        # Rule) so it re-measures with the live terminal width.
        def _session_rule_formatted():
            label = self._session_label_fn() if self._session_label_fn is not None else ""
            current = self._read_terminal_size()
            columns = current[0] if current is not None else terminal_cols(minimum=1)
            fragments = format_session_rule(columns, label)
            if self._is_fullscreen:
                return _native_hyperlink_boundary(
                    fragments, render_counter=self._app.render_counter
                )
            return fragments

        session_rule = Window(
            _FullscreenDragBoundaryControl(
                _session_rule_formatted,
                focusable=False,
                transcript_drag_callback=self._handle_fullscreen_drag_over_bottom_chrome,
                transcript_scroll_callback=(
                    self._scroll_fullscreen_transcript if self._is_fullscreen else None
                ),
            ),
            height=optional_row,
        )

        # Completion menu as a real layout region below the input.
        # Shrinks to the number of completions (with a 12-row cap) so the
        # HSplit doesn't inflate it with blank space when there are only
        # 1–4 matches. The stock ``CompletionsMenu`` wraps the control in
        # a Window with ``Dimension(min=1, max=12)`` and no preferred
        # size — HSplit then gives it the max height, leading to ugly
        # gaps below the completions when the list is short. We use
        # ``CompletionsMenuControl`` directly so we can set a dynamic
        # ``preferred`` height based on the actual completion count.
        _COMPLETION_MAX = 12

        def _completions_height() -> Dimension:
            state = self.input_buffer.complete_state
            n = len(state.completions) if state is not None else 0
            exact = min(n, _COMPLETION_MAX)
            # Exact, not a range. Dimension(min=1, max=12) lets HSplit
            # inflate the window when extra space is available — which
            # is exactly what causes growing blank gaps between the
            # prompt and the menu as completions narrow.
            # A large completion set consumes only rows left after the input's
            # one-row minimum; it can shrink all the way to zero instead of
            # forcing HSplit's "Window too small" fallback.
            return Dimension(min=0, max=exact, preferred=exact)

        completions_window = ConditionalContainer(
            Window(
                content=_NativeSelectionCompletionsMenuControl(
                    transcript_drag_callback=self._handle_fullscreen_drag_over_bottom_chrome,
                    transcript_scroll_callback=(
                        self._scroll_fullscreen_transcript if self._is_fullscreen else None
                    ),
                ),
                width=Dimension(min=8),
                height=_completions_height,
                dont_extend_height=True,
                right_margins=(
                    [] if self._is_fullscreen else [ScrollbarMargin(display_arrows=True)]
                ),
            ),
            filter=Condition(
                lambda: (
                    self.input_buffer.complete_state is not None
                    and bool(self.input_buffer.complete_state.completions)
                )
            ),
        )

        # Active bottom region (top → bottom):
        #   status (spinner + optional badges)
        #   queued command/type-ahead lines
        #   session rule — always visible while at the transcript tail
        #   input composer (one padding row above and below the input)
        #   completions (only while completing)
        if self._is_fullscreen:
            self._status_region_container = ConditionalContainer(
                VSplit(
                    [
                        status_window,
                        self._transient_status_container,
                        self._return_to_tail_container,
                    ],
                ),
                filter=Condition(
                    lambda: (
                        self._status_region_occupied
                        or not self._fullscreen_transcript.viewport.follows_tail
                    )
                ),
            )
            status_region = self._status_region_container
        else:
            self._status_region_container = None
            status_region = status_window
        main_children = [
            status_region,
            queue_window,
            session_rule,
            self._input_container,
            completions_window,
        ]
        if self._output_window is not None:
            main_children.insert(0, self._output_window)
        main_container = HSplit(main_children, window_too_small=Window())
        self._main_container = main_container

        def _subview_formatted():
            view = self._active_subview
            if view is None:
                return ANSI("")
            try:
                size = self._app.output.get_size()
                width, height = int(size.columns), int(size.rows)
            except Exception:
                width, height = terminal_cols(minimum=80), 24
            # Subviews intentionally return SGR-colored ANSI, but their rows
            # also contain session names, model output, server text, and other
            # untrusted data.  Apply the same allowlist as transcript blocks
            # before prompt_toolkit explodes it into screen cells.
            return ANSI(
                project_prompt_toolkit_ansi(sanitize_transcript_ansi(view.render(width, height)))
            )

        self._subview_control = _SubviewControl(
            _subview_formatted,
            mouse_callback=self._subview_mouse,
            focusable=True,
            show_cursor=False,
        )
        subview_window = Window(
            self._subview_control,
            wrap_lines=False,
            always_hide_cursor=True,
            # Compact prompt views render only their bounded line count, while
            # explorer views still render a full terminal-sized frame.
            dont_extend_height=True,
        )

        def _root_container():
            return subview_window if self._active_subview is not None else main_container

        def _subview_mouse_enabled() -> bool:
            view = self._active_subview
            if view is None:
                return self._is_fullscreen and self._fullscreen_mouse_navigation
            if self._is_fullscreen and not self._fullscreen_mouse_navigation:
                return False
            return bool(getattr(view, "mouse_support", True))

        self._app = _ResizeAwareApplication(
            layout=Layout(
                DynamicContainer(_root_container),
                focused_element=input_window,
            ),
            key_bindings=kb,
            style=create_prompt_style(),
            full_screen=self._is_fullscreen,
            before_render=self._before_render,
            after_render=self._after_render,
            # When the Application exits (e.g. /exit), erase the live
            # region so the final screen is just the committed
            # scrollback. Otherwise the empty ❯ from the input line
            # gets a final redraw right before exit and appears as a
            # ghost prompt above '❯ /exit' in the transcript.
            erase_when_done=True,
            mouse_support=Condition(_subview_mouse_enabled),
            # SIGWINCH already invalidates the app; the fallback poll creates a
            # delayed second redraw (~0.5–0.75s later), which is visible in
            # fullscreen subviews after terminal resize.
            terminal_size_polling_interval=None,
            defer_resize_redraw=self._defer_prompt_toolkit_resize_redraw,
            resize_redraw_is_deferred=self._prompt_toolkit_resize_redraw_is_deferred,
        )
        # Bare Escape passes through both the VT100 prefix parser and the key
        # binding prefix matcher. Their one-second defaults make interruption
        # feel broken; terminal-generated Meta sequences arrive as one read, so
        # a short ambiguity window preserves them without delaying Esc.
        self._app.ttimeoutlen = 0.05
        self._app.timeoutlen = 0.05

    def observe_agent(self) -> None:
        """Observe the configured agent for this application run.

        Construction remains side-effect free so composition failures cannot
        leak a subscription. ``run_async`` invokes this inside its teardown
        guard, and composition roots may invoke it explicitly once guarded.
        """
        if self._agent is not None and self._agent_controller.state is None:
            self._agent_controller.observe(self._agent)

    def refresh_style(self) -> None:
        """Apply the current palette to chrome and retained fullscreen output."""
        self._app.style = create_prompt_style()
        if self._is_fullscreen and self._fullscreen_semantic_replay_count:
            # Width caches also contain resolved theme colors. Theme changes
            # must force each semantic callback to render again at this width.
            for block in self._transcript_blocks:
                block.replay_cache.clear()
            self._rebuild_fullscreen_transcript()
        else:
            self._app.invalidate()

    async def open_event_explorer(self, event_manager: Any) -> None:
        """Open the event explorer as an in-app subview."""
        from .event_explorer import EventExplorerView

        await self.open_subview(EventExplorerView(event_manager))

    async def open_session_explorer(self) -> None:
        """Open the session explorer as an in-app subview."""
        from .session_explorer import SessionExplorerView

        await self.open_subview(SessionExplorerView())

    async def open_activity_overlay(self, outputs: list[Any]) -> None:
        """Open the activity snapshot as an in-app subview."""
        from .activity_overlay import ActivityOverlayView

        await self.open_subview(ActivityOverlayView(outputs))

    @staticmethod
    def _validate_clipboard_text(text: str) -> _ClipboardResult | None:
        if not text:
            return _ClipboardResult(False, reason="selection is empty")
        if len(text.encode("utf-8")) > 100_000:
            return _ClipboardResult(False, reason="selection exceeds 100 KB")
        return None

    async def _copy_to_local_clipboard_async(self, text: str) -> _ClipboardResult | None:
        """Try a cancellable local clipboard helper without blocking the UI loop."""
        invalid = self._validate_clipboard_text(text)
        if invalid is not None:
            return invalid
        command = self._local_clipboard_command()
        if command is None:
            return None
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError:
            return None
        try:
            await asyncio.wait_for(process.communicate(text.encode("utf-8")), timeout=2)
        except asyncio.CancelledError:
            await self._terminate_clipboard_process(process)
            raise
        except TimeoutError:
            await self._terminate_clipboard_process(process)
            return None
        except OSError:
            await self._terminate_clipboard_process(process)
            return None
        return _ClipboardResult(True, transport="local") if process.returncode == 0 else None

    @staticmethod
    async def _terminate_clipboard_process(
        process: asyncio.subprocess.Process,
    ) -> None:
        if process.returncode is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        try:
            await process.wait()
        except ProcessLookupError:
            pass

    @staticmethod
    def _local_clipboard_command() -> tuple[str, ...] | None:
        # Graphical Linux clipboard clients can be installed even when their
        # corresponding display is unavailable. In particular, displayless sbx
        # environments expose an ``xclip`` compatibility shim that exits 0 for
        # text without copying it. Do not treat that false success as a local
        # clipboard write; fall through to OSC 52 instead.
        local_commands = [
            ("pbcopy", ()),
        ]
        display_disabled = os.environ.get("SBX_NO_DISPLAY") == "1"
        if not display_disabled and os.environ.get("WAYLAND_DISPLAY"):
            local_commands.append(("wl-copy", ()))
        if not display_disabled and os.environ.get("DISPLAY"):
            local_commands.extend(
                (
                    ("xclip", ("-selection", "clipboard")),
                    ("xsel", ("--clipboard", "--input")),
                )
            )
        for executable, arguments in local_commands:
            path = shutil.which(executable)
            if path is not None:
                return (path, *arguments)
        return None

    def _copy_to_local_clipboard_result(self, text: str) -> _ClipboardResult | None:
        """Try one available platform clipboard command, otherwise return None."""
        invalid = self._validate_clipboard_text(text)
        if invalid is not None:
            return invalid
        command = self._local_clipboard_command()
        if command is None:
            return None
        try:
            subprocess.run(
                list(command),
                input=text.encode("utf-8"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=2,
            )
            return _ClipboardResult(True, transport="local")
        except (OSError, subprocess.SubprocessError):
            return None

    def _copy_to_osc52_result(self, text: str) -> _ClipboardResult:
        invalid = self._validate_clipboard_text(text)
        if invalid is not None:
            return invalid
        try:
            payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
            self._app.output.write_raw(f"\x1b]52;c;{payload}\x07")
            self._app.output.flush()
            return _ClipboardResult(True, transport="osc52")
        except Exception as exc:
            return _ClipboardResult(False, reason=_short_exception_message(exc))

    def _copy_to_clipboard_result(self, text: str) -> _ClipboardResult:
        """Copy text locally when possible, with OSC 52 for remote terminals."""
        invalid = self._validate_clipboard_text(text)
        if invalid is not None:
            return invalid
        remote = bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"))
        local = None if remote else self._copy_to_local_clipboard_result(text)
        return local if local is not None else self._copy_to_osc52_result(text)

    def _copy_to_clipboard(self, text: str) -> bool:
        """Compatibility callback used by in-app sensitive prompts."""
        return self._copy_to_clipboard_result(text).success

    async def prompt_sensitive(
        self, title: str, message: str, *, link_url: str | None = None
    ) -> str:
        """Collect masked text without launching a nested terminal application."""
        from .subapp import SensitiveTextPromptView

        view = SensitiveTextPromptView(
            title,
            message,
            link_url=link_url,
            copy_handler=self._copy_to_clipboard,
        )
        await self.open_subview(view)
        return view.value or ""

    async def prompt_text(self, title: str, message: str, default: str = "") -> str:
        """Collect ordinary text in a reusable in-app modal view."""
        from .subapp import TextPromptView

        view = TextPromptView(title, message, default=default)
        await self.open_subview(view)
        return view.value or ""

    async def prompt_choice(self, title: str, message: str, options: list[str]) -> str:
        """Collect one searchable choice in a reusable in-app modal view."""
        from .subapp import ChoicePromptView

        view = ChoicePromptView(title, message, options)
        await self.open_subview(view)
        return view.value or ""

    async def open_job_explorer(self) -> None:
        """Open the job explorer as an in-app subview."""
        from .job_explorer import JobExplorerView

        state = self._agent_controller.state
        jobs = () if state is None else state.workspace.jobs
        await self.open_subview(JobExplorerView(jobs))

    async def open_todo_explorer(self) -> None:
        """Open the host-provided todo explorer view."""
        if self._host_services.open_todo_view is None:
            raise RuntimeError("Todo explorer is not available for this session.")
        view = self._host_services.open_todo_view()
        if inspect.isawaitable(view):
            view = await view
        await self.open_subview(view)

    async def open_memory_explorer(self) -> None:
        """Open the host-provided memory explorer view."""
        if self._host_services.open_memory_view is None:
            raise RuntimeError("Memory is not enabled for this agent (see /memory).")
        view = self._host_services.open_memory_view()
        if inspect.isawaitable(view):
            view = await view
        await self.open_subview(view)

    def _cancel_fullscreen_drag(self) -> None:
        """Cancel renderer selection when transcript mouse ownership is lost."""
        self._fullscreen_transcript.clear_selection()
        control = self._output_window.content if self._output_window else None
        if isinstance(control, _FullscreenTranscriptControl):
            control.cancel_drag()

    async def open_subview(self, view: InAppSubview) -> None:
        """Open *view* inside the existing prompt_toolkit Application.

        This is the reusable seam for future ToDo, Sessions, Jobs, Artifacts,
        and similar browse/edit/comment panes. It deliberately does not launch
        a nested Application; the host owns focus, key dispatch, resize, mouse,
        and restoration to the main prompt. Host-level convention: ``q`` closes
        the subview; ``Esc`` is reserved for contextual clear/cancel/back inside
        the active view.
        """
        if self._active_subview_done is not None and not self._active_subview_done.done():
            return
        self._cancel_fullscreen_drag()
        self._active_subview = view
        loop = asyncio.get_running_loop()
        self._active_subview_done = loop.create_future()
        view.on_open()
        if self._subview_control is not None:
            self._app.layout.focus(self._subview_control)
        if self._app.is_running:
            self._app.invalidate()
        try:
            await self._active_subview_done
        finally:
            active = self._active_subview
            if active is not None:
                active.on_close()
            self._active_subview = None
            self._active_subview_done = None
            try:
                self._app.layout.focus(self._input_window)
            except Exception:
                pass
            if self._is_fullscreen and self._resize_reflow.has_pending_replay:
                # The resize callback observes geometry while a modal subview is
                # visible, but semantic transcript work stays deferred until the
                # main view returns. Rebuild before invalidating so the first
                # restored frame already has the settled projection.
                self._rebuild_fullscreen_transcript()
            if self._app.is_running:
                self._app.invalidate()
            if not self._is_fullscreen and self._resize_reflow.has_pending_replay:
                self._schedule_resize_replay()

    def _prefill_input(self, text: str) -> None:
        self.input_buffer.text = text
        self.input_buffer.cursor_position = len(text)
        try:
            self.input_buffer.cancel_completion()
        except Exception:
            pass

    def prefill_input(self, text: str, *, overwrite: bool = False) -> bool:
        """Place text in the command buffer without submitting it.

        Command results arrive asynchronously, so the default preserves text
        the user has already started typing. Returns whether the prefill was
        applied.
        """
        if self.input_buffer.text and not overwrite:
            return False
        self._prefill_input(text)
        if self._app.is_running:
            self._app.invalidate()
        return True

    def _close_subview(self) -> None:
        done = self._active_subview_done
        if done is not None and not done.done():
            done.get_loop().call_soon_threadsafe(done.set_result, None)
        else:
            active = self._active_subview
            if active is not None:
                active.on_close()
            self._active_subview = None
        if self._app.is_running:
            self._app.invalidate()

    @property
    def active_subview(self) -> InAppSubview | None:
        """Currently hosted in-app subview, if any."""
        return self._active_subview

    @property
    def _event_explorer_model(self) -> Any | None:
        """Compatibility accessor for tests while /events moves to subviews."""
        view = self._active_subview
        return getattr(view, "model", None)

    def _subview_key(self, event, action: str, value: str = "") -> bool:
        view = self._active_subview
        if view is None:
            return False
        result = normalize_key_result(view.handle_key(action, value))
        if result == "close":
            pending_input = getattr(view, "pending_input", None)
            self._close_subview()
            if pending_input:
                self._prefill_input(str(pending_input))
        elif result == "ignored":
            return False
        if self._app.is_running:
            self._app.invalidate()
        return True

    def _subview_mouse(self, action: str, x: int, y: int) -> bool:
        """Dispatch a mouse action, preserving its position for pane routing."""
        view = self._active_subview
        if view is None:
            return False
        handler = getattr(view, "handle_mouse", None)
        if handler is None:
            return self._subview_key(None, action)
        result = normalize_key_result(handler(action, x, y))
        if result == "ignored":
            return False
        if result == "close":
            self._close_subview()
        if self._app.is_running:
            self._app.invalidate()
        return True

    def _transcript_viewport_size(self) -> tuple[int, int]:
        """Return the rendered fullscreen transcript geometry when available."""
        window = self._output_window
        control = None if window is None else window.content
        if isinstance(control, _FullscreenTranscriptControl):
            current = control.render_size
            if current is not None:
                # During a debounced resize this control can still expose the
                # previous frame's width. Never project wider than the physical
                # terminal or prompt_toolkit will clip the overflow before the
                # next frame can update ``render_size``.
                try:
                    physical_width = max(1, int(self._app.output.get_size().columns))
                except Exception:
                    physical_width = current[0]
                return min(current[0], physical_width), current[1]
        try:
            size = self._app.output.get_size()
            return max(1, int(size.columns)), max(1, int(size.rows) - 6)
        except Exception:
            return terminal_cols(minimum=1), 18

    def _fullscreen_input_overflows(self) -> bool:
        """Return whether the capped composer currently has hidden visual rows."""
        if not self._is_fullscreen:
            return False
        info = self._input_window.render_info
        if info is None or info.window_height <= 0:
            return False
        return (
            sum(info.get_height_for_line(line_number) for line_number in range(info.content_height))
            > info.window_height
        )

    def _scroll_fullscreen_input_or_transcript(self, delta: int) -> None:
        """Wheel the capped composer when overflowing, otherwise the transcript."""
        if not self._fullscreen_input_overflows():
            self._scroll_fullscreen_transcript(delta)
            return
        self._input_window.scroll_without_moving_cursor(delta)
        if self._app.is_running:
            self._app.invalidate()

    def _scroll_fullscreen_transcript(self, delta: int) -> None:
        if not self._is_fullscreen:
            return
        width, height = self._transcript_viewport_size()
        self._fullscreen_transcript.scroll_visual_lines(delta, width=width, height=height)
        if self._fullscreen_transcript.viewport.follows_tail:
            self._has_unseen_agent_message = False
        if self._app.is_running:
            self._app.invalidate()

    def _jump_fullscreen_to_tail(self) -> None:
        """Resume following live output from a key binding or mouse click."""
        if not self._is_fullscreen:
            return
        self._fullscreen_transcript.jump_to_tail()
        self._has_unseen_agent_message = False
        if self._app.is_running:
            self._app.invalidate()

    def _handle_fullscreen_drag_over_bottom_chrome(self, mouse_event: MouseEvent) -> bool:
        """Resolve an active transcript drag when the pointer enters bottom chrome."""
        control = self._output_window.content if self._output_window else None
        if not isinstance(control, _FullscreenTranscriptControl) or not control.dragging:
            return False
        return control.handle_external_mouse(mouse_event, below=True)

    def _open_fullscreen_link_at(self, x: int, y: int) -> bool:
        """Open a safe hyperlink under a click without affecting drag-copy."""
        if not self._is_fullscreen:
            return False
        width, height = self._transcript_viewport_size()
        url = self._fullscreen_transcript.hyperlink_at(x=x, y=y, width=width, height=height)
        if url is None:
            return False

        # A browser on an SSH host is not the user's browser. Reuse the
        # remote-aware clipboard path (OSC 52 fallback) so the URL reaches the
        # terminal that received the click without launching anything remotely.
        if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
            self._start_fullscreen_selection_copy(url)
            return True

        # Opening a URL is irreversible. Ignore duplicate clicks while one
        # launch is in flight rather than cancelling an await whose subprocess
        # may already have accepted the request.
        if self._link_task is not None and not self._link_task.done():
            return True

        async def open_link() -> None:
            task = asyncio.current_task()
            try:
                opened = await self._open_local_url(url)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("failed to open transcript hyperlink", exc_info=True)
                opened = False
            finally:
                if self._link_task is task:
                    self._link_task = None
            if not opened:
                # A local machine without a browser helper still gets a useful,
                # explicit result instead of a swallowed click.
                self._start_fullscreen_selection_copy(url)

        self._link_task = asyncio.create_task(open_link())
        return True

    @staticmethod
    def _browser_open_command(url: str) -> tuple[str, ...] | None:
        """Return a direct-argv browser opener, avoiding shell interpretation."""
        candidates = (
            ("open", (url,)),
            ("xdg-open", (url,)),
            ("wslview", (url,)),
            ("gio", ("open", url)),
            ("rundll32.exe", ("url.dll,FileProtocolHandler", url)),
        )
        for executable, arguments in candidates:
            path = shutil.which(executable)
            if path is not None:
                return (path, *arguments)
        return None

    async def _open_local_url(self, url: str) -> bool:
        """Launch a validated URL with a cancellable local helper process."""
        command = self._browser_open_command(url)
        if command is None:
            return False
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except OSError:
            return False
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.CancelledError:
            await self._terminate_clipboard_process(process)
            raise
        except TimeoutError:
            # Browser launchers can remain attached after successfully handing
            # the URL off. Do not kill the helper or report a false failure.
            return True
        return process.returncode == 0

    def _fullscreen_code_action_at(self, x: int, y: int) -> str | None:
        """Resolve a code-copy action at one visible transcript cell."""
        width, height = self._transcript_viewport_size()
        return self._fullscreen_transcript.copy_action_at(
            x=x,
            y=y,
            width=width,
            height=height,
        )

    def _handle_fullscreen_selection(self, action: str, x: int, y: int) -> None:
        """Apply one mouse-selection transition and copy on button release."""
        width, height = self._transcript_viewport_size()
        if action == "cancel":
            self._fullscreen_transcript.clear_selection()
        elif action == "start":
            self._fullscreen_transcript.begin_selection(x=x, y=y, width=width, height=height)
        else:
            self._fullscreen_transcript.update_selection(x=x, y=y, width=width, height=height)
        if action == "finish":
            text = self._fullscreen_transcript.selected_text()
            # The mouse gesture is complete once the button is released. Capture
            # its payload, then remove the visual selection immediately so every
            # release target behaves alike while clipboard I/O runs asynchronously.
            self._fullscreen_transcript.clear_selection()
            if text:
                self._start_fullscreen_selection_copy(text)
        if self._app.is_running:
            self._app.invalidate()

    def _copy_input_selection(self, text: str) -> None:
        """Mirror a prompt_toolkit composer selection to app and system clipboards."""
        self._app.clipboard.set_text(text)
        self._start_fullscreen_selection_copy(text)

    def _start_fullscreen_selection_copy(self, text: str) -> None:
        """Copy without blocking prompt_toolkit's event loop on local helpers."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self._report_fullscreen_copy(text, self._copy_to_clipboard_result(text))
            return
        previous = self._clipboard_task
        if previous is not None:
            previous.cancel()
        self._clipboard_task = asyncio.create_task(
            self._copy_fullscreen_selection(text, previous=previous)
        )

    async def _copy_fullscreen_selection(
        self, text: str, *, previous: asyncio.Task[None] | None = None
    ) -> None:
        task = asyncio.current_task()
        try:
            if previous is not None:
                try:
                    await previous
                except asyncio.CancelledError:
                    current = asyncio.current_task()
                    if current is not None and current.cancelling():
                        raise
            remote = bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"))
            result = None if remote else await self._copy_to_local_clipboard_async(text)
            if result is None:
                result = self._copy_to_osc52_result(text)
            self._report_fullscreen_copy(text, result)
        finally:
            if self._clipboard_task is task:
                self._clipboard_task = None

    def _report_fullscreen_copy(self, text: str, result: _ClipboardResult) -> None:
        if result.success:
            count = len(text)
            noun = "character" if count == 1 else "characters"
            self._show_transient_status(f"Copied {count} {noun}", style="class:return-to-tail")
        else:
            self._show_transient_status(
                f"Copy failed: {result.reason}. Try Option/Alt-drag, or press F6 for native selection."
            )

    def _show_transient_status(
        self,
        text: str,
        *,
        seconds: float = 3.0,
        style: str = "class:status",
    ) -> None:
        self._transient_status_text = text
        self._transient_status_style = style
        if self._transient_status_timer is not None:
            self._transient_status_timer.cancel()
            self._transient_status_timer = None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            self._transient_status_timer = loop.call_later(seconds, self._clear_transient_status)
        if self._app.is_running:
            self._app.invalidate()

    def _clear_transient_status(self) -> None:
        self._transient_status_timer = None
        self._transient_status_text = ""
        self._transient_status_style = "class:status"
        if self._app.is_running:
            self._app.invalidate()

    # ── key bindings --------------------------------------------------

    def _build_key_bindings(self):  # returns KeyBindingsBase (union of KB + merged)
        from .input_handler import create_key_bindings as _legacy_kb

        # The legacy bindings handle Enter (accept_handler dispatches to
        # our _accept_handler), Alt+Enter / Ctrl+J (newline), Tab (via
        # default bindings), and the slash/bang auto-trigger that re-opens
        # the completion menu as the user types a command.
        legacy = _legacy_kb(vi_mode=False)

        kb = KeyBindings()
        subview_active = Condition(lambda: self._active_subview is not None)
        subview_inactive = ~subview_active
        input_selection_active = (
            Condition(lambda: self.input_buffer.selection_state is not None) & subview_inactive
        )
        fullscreen_transcript = Condition(lambda: self._is_fullscreen) & subview_inactive

        @kb.add("pageup", filter=fullscreen_transcript, eager=True)
        def _(event):
            if self._fullscreen_input_overflows():
                self._input_window.reset_manual_wheel_scroll()
                scroll_page_up(event)
            else:
                _, height = self._transcript_viewport_size()
                self._scroll_fullscreen_transcript(-height)

        @kb.add("pagedown", filter=fullscreen_transcript, eager=True)
        def _(event):
            if self._fullscreen_input_overflows():
                self._input_window.reset_manual_wheel_scroll()
                scroll_page_down(event)
            else:
                _, height = self._transcript_viewport_size()
                self._scroll_fullscreen_transcript(height)

        @kb.add(Keys.ControlHome, filter=fullscreen_transcript, eager=True)
        def _(event):
            width, _ = self._transcript_viewport_size()
            self._fullscreen_transcript.jump_to_start(width=width)
            event.app.invalidate()

        @kb.add(Keys.ControlEnd, filter=fullscreen_transcript, eager=True)
        def _(event):
            self._jump_fullscreen_to_tail()

        @kb.add("f6", filter=Condition(lambda: self._is_fullscreen), eager=True)
        def _(event):
            self._fullscreen_mouse_navigation = not self._fullscreen_mouse_navigation
            if not self._fullscreen_mouse_navigation:
                self._fullscreen_transcript.clear_selection()
                if isinstance(
                    self._output_window.content if self._output_window else None,
                    _FullscreenTranscriptControl,
                ):
                    self._output_window.content.cancel_drag()
            self._command_status_text = (
                "Mouse navigation enabled (Option/Alt-drag where supported; F6 otherwise)"
                if self._fullscreen_mouse_navigation
                else "Native terminal selection enabled (F6 to restore app mouse)"
            )
            event.app.invalidate()

        @kb.add(Keys.Any, filter=subview_active, eager=True)
        def _(event):
            # Drop parsed mouse events and any raw mouse CSI bytes that slip
            # through — subviews disable mouse_support so these would otherwise
            # be appended verbatim to text buffers (e.g. API-key prompt).
            for kp in event.key_sequence:
                if kp.key in (Keys.Vt100MouseEvent, Keys.WindowsMouseEvent):
                    return
            data = event.data or ""
            if _is_raw_mouse_report(data):
                return
            self._subview_key(event, "text", data)

        @kb.add(Keys.BracketedPaste, filter=subview_active, eager=True)
        def _(event):
            self._subview_key(event, "text", event.data)

        @kb.add("escape", filter=subview_active, eager=True)
        def _(event):
            self._subview_key(event, "escape")

        @kb.add("q", filter=subview_active, eager=True)
        def _(event):
            self._subview_key(event, "quit")

        @kb.add("r", filter=subview_active, eager=True)
        def _(event):
            self._subview_key(event, "resume")

        @kb.add("enter", filter=subview_active, eager=True)
        def _(event):
            self._subview_key(event, "enter")

        @kb.add("/", filter=subview_active, eager=True)
        def _(event):
            self._subview_key(event, "slash")

        @kb.add("backspace", filter=subview_active, eager=True)
        @kb.add("c-h", filter=subview_active, eager=True)
        def _(event):
            self._subview_key(event, "backspace")

        @kb.add("c-y", filter=subview_active, eager=True)
        def _(event):
            self._subview_key(event, "copy")

        @kb.add("f2", filter=subview_active, eager=True)
        def _(event):
            self._subview_key(event, "native_selection")

        @kb.add("tab", filter=subview_active, eager=True)
        def _(event):
            self._subview_key(event, "tab")

        @kb.add("down", filter=subview_active, eager=True)
        def _(event):
            self._subview_key(event, "down")

        @kb.add("j", filter=subview_active, eager=True)
        def _(event):
            self._subview_key(event, "j")

        @kb.add("up", filter=subview_active, eager=True)
        def _(event):
            self._subview_key(event, "up")

        @kb.add("k", filter=subview_active, eager=True)
        def _(event):
            self._subview_key(event, "k")

        @kb.add("pagedown", filter=subview_active, eager=True)
        def _(event):
            self._subview_key(event, "page_down")

        @kb.add("pageup", filter=subview_active, eager=True)
        def _(event):
            self._subview_key(event, "page_up")

        @kb.add("home", filter=subview_active, eager=True)
        def _(event):
            self._subview_key(event, "home")

        @kb.add("end", filter=subview_active, eager=True)
        def _(event):
            self._subview_key(event, "end")

        @kb.add(Keys.ScrollDown, filter=subview_active, eager=True)
        def _(event):
            self._subview_key(event, "scroll_down")

        @kb.add(Keys.ScrollUp, filter=subview_active, eager=True)
        def _(event):
            self._subview_key(event, "scroll_up")

        @kb.add("c-c", filter=subview_active, eager=True)
        def _(event):
            self._close_subview()

        def _extend_input_selection(event, move: Callable[[int], None]) -> None:
            buffer = event.current_buffer
            if buffer.selection_state is None:
                buffer.start_selection(selection_type=SelectionType.CHARACTERS)
            move(event.arg)

        @kb.add("s-left", filter=subview_inactive, eager=True)
        def _(event):
            _extend_input_selection(event, event.current_buffer.cursor_left)

        @kb.add("s-right", filter=subview_inactive, eager=True)
        def _(event):
            _extend_input_selection(event, event.current_buffer.cursor_right)

        @kb.add("s-up", filter=subview_inactive, eager=True)
        def _(event):
            _extend_input_selection(event, event.current_buffer.cursor_up)

        @kb.add("s-down", filter=subview_inactive, eager=True)
        def _(event):
            _extend_input_selection(event, event.current_buffer.cursor_down)

        @kb.add("c-x", filter=input_selection_active, eager=True)
        def _(event):
            # Retain the text in prompt_toolkit's clipboard before deleting it.
            # This leaves an in-app recovery path if the system copy later fails.
            _document, clipboard_data = self.input_buffer.document.cut_selection()
            if clipboard_data.text:
                event.app.clipboard.set_data(clipboard_data)
                self._start_fullscreen_selection_copy(clipboard_data.text)
                self.input_buffer.cut_selection()

        @kb.add("c-c", filter=subview_inactive, eager=True)
        def _(event):
            # Ctrl-C advances one destructive step per press: discard a draft,
            # then interrupt active work, then confirm exit. Never combine a
            # composer clear with cancellation, so a draft is recoverably cheap
            # to abandon even while an agent is running.
            if event.current_buffer.text:
                event.current_buffer.reset()
                self._history_cursor = None
                self._clear_ctrl_c_exit()
                return

            # After the clear/interrupt step, an armed press exits even if
            # cancellation cleanup has not acknowledged yet.
            if self._ctrl_c_exit_armed:
                self._clear_ctrl_c_exit()
                event.app.exit()
                return

            # Slash commands and agent work can overlap. Attempt both instead
            # of short-circuiting so one Ctrl-C interrupts every active turn
            # owner before the next press becomes the exit step.
            command_cancelled = self.request_command_cancel()
            agent_cancelled = self.request_agent_cancel(source="ctrl-c")
            if command_cancelled or agent_cancelled:
                self._arm_ctrl_c_exit()
                return

            # At an idle prompt, retain the established double-Ctrl-C safety
            # gesture.
            self._arm_ctrl_c_exit()

        @kb.add("c-d", filter=subview_inactive)
        def _(event):
            event.app.exit()

        @kb.add("tab", filter=subview_inactive)
        def _(event):
            # Standard Tab: open the menu if closed, advance to the
            # next option if already open. start_completion doesn't
            # advance on repeat presses — complete_next does both.
            buf = event.current_buffer
            if buf.complete_state is None:
                _set_completions_sync(buf)
                if buf.complete_state is not None and buf.complete_state.completions:
                    buf.complete_next()
            else:
                buf.complete_next()

        @kb.add("s-tab", filter=subview_inactive)
        def _(event):
            buf = event.current_buffer
            if buf.complete_state is not None:
                buf.complete_previous()

        # Empty-buffer Up: queue pop wins over history — matches the
        # pre-rewrite typeahead UX (pop the last thing you typed while
        # the agent was working so you can edit it). In the forever-loop
        # model we pop from the agent's user_messages queue; items
        # already consumed by the agent can't be edited.
        empty_buffer = Condition(lambda: self.input_buffer.text == "") & subview_inactive

        def _pop_last_queued() -> str | None:
            if self._agent_controller.state is None:
                return None
            return self._agent_controller.withdraw_pending_input()

        @kb.add("up", filter=empty_buffer)
        def _(event):
            popped = _pop_last_queued()
            if popped is not None:
                self.complete_pending_input_handoff(popped)
                self.input_buffer.text = popped
                self.input_buffer.cursor_position = len(popped)
                return
            self._history_navigate(-1)

        @kb.add("down", filter=empty_buffer)
        def _(event):
            self._history_navigate(+1)

        # Absorb escape-prefixed Meta/Option input as one non-interrupting
        # gesture. prompt_toolkit otherwise falls back from an unmatched pair
        # such as Option-[ (ESC + "[") to the bare-Escape binding below, which
        # accidentally cancels the active turn. More-specific bindings (for
        # example Alt+Enter and Emacs Meta navigation) still win.
        @kb.add("escape", Keys.Any, filter=subview_inactive)
        def _(event):
            data = event.data
            if data and data != "\x1b":
                event.current_buffer.insert_text(data)

        # A standalone Esc is emitted only after prompt_toolkit's VT parser has
        # allowed time for a possible escape-prefixed key to arrive.
        @kb.add("escape", filter=subview_inactive)
        def _(event):
            if self.request_command_cancel():
                return
            self.request_agent_cancel(source="escape")

        # Merge so our bindings (C-c with is_thinking awareness, Tab
        # trigger, Esc cancel, empty-buffer Up/Down for queue+history)
        # override the legacy bindings for the same keys, while legacy
        # still provides Enter → accept_handler, Alt+Enter newline, and
        # the slash auto-trigger characters.
        return merge_key_bindings([legacy, kb])

    # ── submission pipeline -------------------------------------------

    def _accept_handler(self, buffer: Buffer) -> bool:
        """prompt_toolkit accept_handler — invoked by ``validate_and_handle()``.

        Slash/bang commands dispatch immediately. Plain text is pushed
        onto ``agent.user_messages`` via ``submit_message`` — the agent's
        forever-loop ``handle()`` picks it up when it calls
        ``self.get_next_input(...)``.

        Returning False tells prompt_toolkit to reset the buffer (clear
        the text, don't keep it as the working-lines tip).
        """
        text = buffer.text
        if not text.strip():
            return False
        self._jump_fullscreen_to_tail()
        if not self._history or self._history[-1] != text:
            self._history.append(text)
        self._history_cursor = None

        if text.startswith("/"):
            self._commands_dispatched.append(text)
            self._run_callback(self._on_command, text)
            return False
        if text.startswith("!"):
            body = text[1:].strip()
            self._last_bang_command = body
            self._run_callback(self._on_bang, body)
            return False

        state = self._agent_controller.state
        mention_base = None if state is None else state.working_directory
        self.submit_message(expand_mentions(text, base_dir=mention_base))
        return False

    def _run_callback(
        self,
        cb: Callable[[str], Awaitable[None] | None] | None,
        arg: str,
    ) -> asyncio.Task | None:
        """Invoke one user callback; return the scheduled Task or None.

        Used by every "call an out-of-band function from the TUI" site
        (``on_command``, ``on_bang``).

        - Synchronous callback (``None``, a regular function, or one
          that raised): returns ``None``. Errors are surfaced into the
          scrollback so an unhandled exception doesn't vanish into
          asyncio's default handler. Sets
          ``self._last_sync_callback_raised = True`` so callers that
          loop (``_drain_next``) can stop after a failure.
        - Coroutine callback: scheduled as a Task and returned. The
          caller can ``add_done_callback`` on it to chain follow-up
          work (e.g. ``_drain_next`` for queued commands). Errors
          inside the coroutine are surfaced via a done-callback
          installed here.
        """
        self._last_sync_callback_raised = False
        if cb is None:
            return None
        try:
            result = cb(arg)
        except BaseException as exc:
            self.emit_block(f"[callback error] {type(exc).__name__}: {exc}\n")
            self._last_sync_callback_raised = True
            return None
        if not asyncio.iscoroutine(result):
            return None
        task = asyncio.ensure_future(result)

        def _report(t: asyncio.Task) -> None:
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                self.emit_block(f"[callback error] {type(exc).__name__}: {exc}\n")

        task.add_done_callback(_report)
        return task

    def _ensure_spinner_task(self) -> None:
        """Start a background task cycling the spinner frame while live work
        needs animation. Invalidates the app each tick so the status line
        redraws; exits when the animated statuses clear."""
        if self._spinner_task is not None and not self._spinner_task.done():
            return

        async def _animate() -> None:
            i = 0
            try:
                while self.is_thinking() or self._llm_probe_status_text:
                    self._spinner_frame = self._spinner_frames[i % len(self._spinner_frames)]
                    if self._app.is_running:
                        self._app.invalidate()
                    i += 1
                    await asyncio.sleep(0.08)
            finally:
                # Paint once after the agent stops so "thinking…" clears.
                if self._app.is_running:
                    self._app.invalidate()

        # Agent snapshots may arrive synchronously during construction,
        # before run_async() establishes the application owner loop.  In that
        # case the initial render will start the spinner after startup.
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        self._spinner_task = loop.create_task(_animate())

    def _history_navigate(self, direction: int) -> None:
        """Move the history cursor by ``direction`` (-1=older, +1=newer)."""
        if not self._history:
            return
        if self._history_cursor is None:
            if direction < 0:
                self._history_cursor = len(self._history) - 1
            else:
                return
        else:
            new = self._history_cursor + direction
            if new < 0 or new >= len(self._history):
                return
            self._history_cursor = new
        self.input_buffer.text = self._history[self._history_cursor]
        self.input_buffer.cursor_position = len(self.input_buffer.text)

    def _pending_input_display(self) -> list[str]:
        """Return runtime queue text plus admissions not yet reflected by it."""
        state = self._agent_controller.state
        pending = [] if state is None else list(state.pending_inputs)
        handoff = [item.text for item in self._pending_input_handoff]
        if not handoff:
            return pending
        if not pending:
            return handoff

        # Runtime coalesces new admissions into its final queue item. Replace
        # the represented suffix with the individual handoff rows, while
        # retaining any runtime-owned prefix and every earlier queue item.
        tail = pending[-1]
        for represented in range(len(handoff), 0, -1):
            combined = "\n".join(handoff[:represented])
            if tail == combined:
                return pending[:-1] + handoff
            suffix = f"\n{combined}"
            if tail.endswith(suffix):
                return pending[:-1] + [tail[: -len(suffix)]] + handoff
        return pending + handoff

    def submit_message(self, user_message: str) -> None:
        """Submit text through the current interactive agent."""
        if self._agent_controller.state is None:
            return
        if self._submission_guard is not None:
            try:
                problem = self._submission_guard()
            except Exception:
                logger.debug("TUI submission guard failed", exc_info=True)
                problem = None
            if problem:
                self.emit_block(f"\x1b[31m{problem}\x1b[0m\n")
                return
        # Register visibility before admission: another loop may consume and
        # queue the accepted transcript echo before ``submit`` returns.
        handoff = _PendingInputHandoff(user_message)
        self._pending_input_handoff.append(handoff)
        try:
            accepted = self._agent_controller.submit(user_message)
        except Exception:
            self._discard_pending_input_handoff(handoff)
            logger.debug("TUI message submission failed", exc_info=True)
            self.emit_block("\x1b[31mMessage rejected.\x1b[0m\n")
            return
        if not accepted:
            self._discard_pending_input_handoff(handoff)
            self.emit_block("\x1b[31mMessage rejected.\x1b[0m\n")
            return
        if self._app.is_running:
            self._app.invalidate()

    def _discard_pending_input_handoff(self, handoff: _PendingInputHandoff) -> None:
        """Discard one exact optimistic admission without disturbing duplicates."""
        for index, candidate in enumerate(self._pending_input_handoff):
            if candidate is handoff:
                self._pending_input_handoff.pop(index)
                break

    def complete_pending_input_handoff(self, text: str) -> None:
        """Retire the submissions represented by one consumed queue item."""
        combined = ""
        consumed = 0
        for handoff in self._pending_input_handoff:
            combined = f"{combined}\n{handoff.text}" if consumed else handoff.text
            consumed += 1
            if combined == text or text.endswith(f"\n{combined}"):
                del self._pending_input_handoff[:consumed]
                break
        else:
            # A callback can arrive after another consumer has advanced the
            # queue. Retire the matching admission without dropping older rows.
            for index, handoff in enumerate(self._pending_input_handoff):
                if handoff.text == text or text.endswith(f"\n{handoff.text}"):
                    del self._pending_input_handoff[index]
                    break
        if self._app.is_running:
            self._app.invalidate()

    def clear_pending_input_handoffs(self) -> None:
        """Discard optimistic queue rows after the runtime queue is flushed."""
        self._pending_input_handoff.clear()
        if self._app.is_running:
            self._app.invalidate()

    def _schedule_agent_callback(self, callback: Callable[[], None]) -> None:
        """Marshal agent observation delivery onto the prompt-toolkit owner loop."""
        loop = self._loop
        if loop is None or not loop.is_running():
            callback()
            return
        try:
            on_ui_loop = asyncio.get_running_loop() is loop
        except RuntimeError:
            on_ui_loop = False
        if on_ui_loop:
            callback()
        else:
            loop.call_soon_threadsafe(callback)

    def _on_agent_change(self, state: Any) -> None:
        # Agent implementations may publish from worker threads. Keep timer and
        # prompt-toolkit mutations on the Application's owner loop even if a
        # caller bypasses the controller's normal scheduler boundary.
        loop = self._loop
        if loop is not None and loop.is_running():
            try:
                on_ui_loop = asyncio.get_running_loop() is loop
            except RuntimeError:
                on_ui_loop = False
            if not on_ui_loop:
                loop.call_soon_threadsafe(self._on_agent_change, state)
                return

        # A pre-interrupt observation may already be queued when cancellation is
        # admitted. Only teardown may acknowledge the optimistic status; the
        # minimum display interval keeps a fast cancellation from clearing it
        # before prompt_toolkit can paint even one frame.
        if state is None:
            self._acknowledge_agent_interrupt()
        elif state is not None:
            self._retire_acknowledged_interrupt_for_new_turn(state)
        app = getattr(self, "_app", None)
        if app is not None and app.is_running:
            app.invalidate()
        self._ensure_spinner_task()

    def runtime_notification_received(self) -> None:
        """Refresh native chrome after the host dequeues runtime work."""
        self._on_dispatcher_dequeued()

    def runtime_state_changed(self) -> None:
        """Marshal a host-runtime state change onto the UI owner loop."""
        loop = self._loop
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(self._refresh_runner_state)
        else:
            self._refresh_runner_state()

    def _refresh_runner_state(self) -> None:
        # This hook runs at the runtime transition site, while observation
        # delivery may still be coalesced behind other UI work. Read the
        # immutable source snapshot so a queued replacement turn cannot inherit
        # the prior turn's delayed interrupt label even for one frame.
        agent = self._agent
        if agent is not None:
            self._retire_acknowledged_interrupt_for_new_turn(agent.state)
        if self._app.is_running:
            self._app.invalidate()
        self._ensure_spinner_task()

    def _retire_acknowledged_interrupt_for_new_turn(self, state: Any) -> None:
        if (
            self._interrupting_agent_turn
            and self._interrupt_status_acknowledged
            and state.lifecycle in {AgentLifecycle.THINKING, AgentLifecycle.WAITING}
            and state.workspace.cancellation is CancellationState.NONE
        ):
            # A queued message has started a new turn. Never let the previous
            # turn's minimum display interval label this fresh work as stopping.
            self._clear_agent_interrupt_status()

    def runtime_cancelled(self) -> None:
        """Acknowledge completed cancellation and render its transcript marker."""
        self._schedule_agent_callback(self._acknowledge_agent_interrupt)
        self.emit_block("\x1b[33m✗ Interrupted agent turn.\x1b[0m\n")

    def _acknowledge_agent_interrupt(self) -> None:
        """Clear interrupt feedback only after it had time to reach a frame."""
        if not self._interrupting_agent_turn:
            return
        self._interrupt_status_acknowledged = True
        started_at = self._interrupt_status_started_at
        remaining = (
            0.0
            if started_at is None
            else _MIN_INTERRUPT_STATUS_SECONDS - (time.monotonic() - started_at)
        )
        if remaining <= 0:
            self._clear_agent_interrupt_status()
        elif self._interrupt_status_clear_timer is None:
            loop = self._loop
            if loop is None or not loop.is_running():
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    self._clear_agent_interrupt_status()
                    return
            self._interrupt_status_clear_timer = loop.call_later(
                remaining, self._clear_agent_interrupt_status
            )

    def _clear_agent_interrupt_status(self) -> None:
        self._interrupt_status_clear_timer = None
        self._interrupt_status_started_at = None
        self._interrupt_status_acknowledged = False
        self._interrupting_agent_turn = False
        if self._app.is_running:
            self._app.invalidate()

    def invalidate(self) -> None:
        """Thread-safe repaint hook for composition-root-owned policies."""
        loop = self._loop

        def _invalidate() -> None:
            if self._app.is_running:
                self._app.invalidate()

        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(_invalidate)
        else:
            _invalidate()

    def request_agent_cancel(self, *, source: str = "escape") -> bool:
        """Request user-visible cancellation through the interactive agent boundary."""
        if source in {"swap", "session"}:
            raise ValueError("host transitions must cancel through their lifecycle owner")
        if self._agent_controller.state is None:
            return False
        accepted = self._agent_controller.interrupt()
        if accepted:
            # The runtime observation callback may arrive on another loop. Paint
            # acknowledgement immediately so the key press never appears lost.
            if self._interrupt_status_clear_timer is not None:
                self._interrupt_status_clear_timer.cancel()
                self._interrupt_status_clear_timer = None
            self._interrupting_agent_turn = True
            self._interrupt_status_acknowledged = False
            self._interrupt_status_started_at = time.monotonic()
            if self._app.is_running:
                self._app.invalidate()
            if self._on_agent_activity is not None:
                self._on_agent_activity()
        return accepted

    def _resume_input_cursor_following(self) -> None:
        """End a mouse-wheel viewport lease after the user edits or moves."""
        input_window = getattr(self, "_input_window", None)
        if isinstance(input_window, _ComposerWindow):
            input_window.reset_manual_wheel_scroll()

    def _on_input_text_changed(self, _buffer: Buffer) -> None:
        """Resume cursor following and cancel any pending double-Ctrl-C gesture."""
        self._resume_input_cursor_following()
        self._clear_ctrl_c_exit()

    def _on_input_cursor_position_changed(self, _buffer: Buffer) -> None:
        """Bring the edit cursor back onscreen after keyboard or pointer movement."""
        self._resume_input_cursor_following()

    def request_command_cancel(self) -> bool:
        """Ask the host to cancel its active slash command, if any."""
        callback = self._on_cancel_command
        if callback is None:
            return False
        try:
            return bool(callback())
        except Exception:
            logger.debug("Slash-command cancellation callback failed", exc_info=True)
            return False

    def _arm_ctrl_c_exit(self) -> None:
        """Require a second Ctrl-C shortly after the first before exiting."""
        self._clear_ctrl_c_exit()
        self._ctrl_c_exit_armed = True
        self._exit_hint_text = "Press Ctrl+C again to exit"
        self._ctrl_c_exit_timer = asyncio.get_running_loop().call_later(
            _CTRL_C_EXIT_WINDOW_SECONDS,
            self._clear_ctrl_c_exit,
        )
        if self._app.is_running:
            self._app.invalidate()

    def _clear_ctrl_c_exit(self) -> None:
        """Disarm exit confirmation and remove its transient status hint."""
        timer = self._ctrl_c_exit_timer
        self._ctrl_c_exit_timer = None
        if timer is not None:
            timer.cancel()
        changed = self._ctrl_c_exit_armed or bool(self._exit_hint_text)
        self._ctrl_c_exit_armed = False
        self._exit_hint_text = ""
        app = getattr(self, "_app", None)
        if changed and app is not None and app.is_running:
            app.invalidate()

    def _on_dispatcher_dequeued(self) -> None:
        """React to a just-dequeued item: redraw queue pane, restart spinner.

        Without this, the queue pane can show stale contents until the
        next event happens to trigger a redraw (spinner tick, user key,
        scrollback write). And the spinner animation task exits when
        ``is_thinking()`` was False between turns — a new turn wants
        it running again.
        """
        ui_loop = self._loop
        try:
            on_ui_loop = asyncio.get_running_loop() is ui_loop
        except RuntimeError:
            on_ui_loop = False
        if ui_loop is not None and not on_ui_loop:
            ui_loop.call_soon_threadsafe(self._on_dispatcher_dequeued)
            return
        if self._app.is_running:
            self._app.invalidate()
        self._ensure_spinner_task()

    # ── output pipeline -----------------------------------------------

    def clear_transcript(self) -> None:
        """Clear live transcript buffers and fullscreen resize replay retention."""
        loop = self._loop
        if loop is not None:
            try:
                on_ui_loop = asyncio.get_running_loop() is loop
            except RuntimeError:
                on_ui_loop = False
            if not on_ui_loop:
                try:
                    loop.call_soon_threadsafe(self._clear_transcript_on_ui_loop)
                except RuntimeError:
                    # The UI loop has already closed; there is no live prompt
                    # buffer left to update.
                    pass
                return
        self._clear_transcript_on_ui_loop()

    def _clear_transcript_on_ui_loop(self) -> None:
        self._transcript_epoch += 1
        self._transcript_blocks.clear()
        self._fullscreen_transcript_bytes = 0
        self._fullscreen_semantic_replay_count = 0
        if self._is_fullscreen:
            self._fullscreen_transcript.clear()
            self._has_unseen_agent_message = False
            self._app.invalidate()
            return
        self.output_buffer.set_document(Document(""), bypass_readonly=True)
        queue = self._block_queue
        if queue is not None:
            queue.put_nowait(_ClearTranscriptQueueItem(self._transcript_epoch))

    def _on_stray_output(self, content: str, disposition: str) -> None:
        """Forward stray stdout/stderr diagnostics to the host boundary."""
        if self._host_services.record_stray_output is not None:
            try:
                self._host_services.record_stray_output(content, disposition)
            except Exception:
                logger.debug("stray-output recorder failed", exc_info=True)

    def emit_block(
        self,
        text: str,
        replay: Callable[[], str] | None = None,
        *,
        event_id: str | None = None,
        tags: set[str] | frozenset[str] | None = None,
        keep: bool = False,
        code_copy_actions: dict[str, str] | None = None,
        agent_message: bool = False,
    ) -> None:
        """Enqueue one ANSI-bearing block for the transcript.

        This is the ONE public contract for writing to the transcript:
        all producers (activity lines, code cells, agent markdown,
        interrupt notices, user echo) call this. A single consumer
        task drains the queue and writes each block in FIFO order via
        ``run_in_terminal`` → ``sys.__stdout__``. No races.

        Thread-safe: retention, prompt_toolkit state, and terminal-queue
        insertion are committed together on the UI loop. This keeps the replay
        snapshot order identical to the terminal write order.
        """
        if not text:
            return

        block = TranscriptBlock(
            source=text,
            replay=replay,
            event_id=str(event_id) if event_id is not None else None,
            tags=frozenset(str(t) for t in (tags or ())),
            keep=keep,
            code_copy_actions=dict(code_copy_actions or {}),
            agent_message=agent_message,
        )

        # Before the consumer is up (pre-run_async) we're single-threaded
        # by construction — safe to touch the buffer directly + emit to
        # stdout. After run_async, route everything via the loop.
        loop = self._loop
        if self._block_queue is None or loop is None:
            rendered = self._render_transcript_source(block.source)
            evicted = self._retain_transcript_block(block, rendered)
            if self._is_fullscreen:
                self._fullscreen_transcript.append(
                    rendered,
                    record_id=block.transcript_record_id,
                    copy_actions=block.code_copy_actions,
                )
                self._fullscreen_transcript.evict_prefix(evicted)
                self._note_unseen_agent_message(block)
                self._app.invalidate()
                return
            self._append_stripped_to_buffer(rendered)
            import sys as _sys

            try:
                _sys.stdout.write(rendered)
                _sys.stdout.flush()
            except Exception:
                pass
            return

        # On-thread fast path: mutate buffer directly so tests that
        # inspect ``output_buffer.text`` right after a call see the
        # update without waiting for a loop tick.
        try:
            on_thread = asyncio.get_running_loop() is loop
        except RuntimeError:
            on_thread = False
        if on_thread:
            self._enqueue_transcript_block(block)
            return

        # Off-thread: use one callback so retention order and queue order cannot
        # diverge, and replay cannot observe a block before the queue does.
        try:
            loop.call_soon_threadsafe(self._enqueue_transcript_block, block)
        except RuntimeError:
            # Teardown won the race with a late producer. Fullscreen must never
            # leak application content onto the restored primary screen.
            if self._is_fullscreen:
                return
            rendered = self._render_replay_source(block.source)
            import sys as _sys

            out = _sys.__stdout__
            if out is not None:
                try:
                    out.write(rendered)
                    out.flush()
                except Exception:
                    pass

    @staticmethod
    def _fullscreen_block_resident_bytes(block: TranscriptBlock, rendered: str) -> int:
        """Conservatively charge all retained textual renderer representations.

        The source and replay result remain live on ``TranscriptBlock``.  The
        model retains safe ANSI and plain text, and at most two projection plus
        two formatted-width caches.  Charging those bounded copies up front
        keeps replay expansion and resize caches inside the advertised budget;
        Python container overhead is deliberately outside this byte contract.
        """
        source_bytes = len(block.source.encode("utf-8"))
        copy_action_bytes = sum(
            len(action_id.encode("utf-8")) + len(payload.encode("utf-8"))
            for action_id, payload in block.code_copy_actions.items()
        )
        replay_cache_bytes = sum(
            len(source.encode("utf-8")) for source in block.replay_cache.values()
        )
        rendered_bytes = len(rendered.encode("utf-8"))
        plain_bytes = len(_strip_ansi(rendered).encode("utf-8"))
        return (
            source_bytes
            + copy_action_bytes
            + replay_cache_bytes
            + (2 * rendered_bytes)
            + (5 * plain_bytes)
        )

    def _retain_transcript_block(self, block: TranscriptBlock, rendered: str) -> int:
        block.transcript_epoch = self._transcript_epoch
        if block.transcript_record_id is None:
            block.transcript_record_id = self._next_transcript_record_id
            self._next_transcript_record_id += 1
        self._transcript_blocks.append(block)
        if self._is_fullscreen:
            block.fullscreen_rendered = rendered
            block.resident_bytes = self._fullscreen_block_resident_bytes(block, rendered)
            self._fullscreen_transcript_bytes += block.resident_bytes
            if block.replay is not None:
                self._fullscreen_semantic_replay_count += 1
            evicted = 0
            retained_bytes = self._fullscreen_transcript_bytes
            while evicted < len(self._transcript_blocks) and (
                len(self._transcript_blocks) - evicted > _FULLSCREEN_TRANSCRIPT_MAX_RECORDS
                or retained_bytes > _FULLSCREEN_TRANSCRIPT_MAX_BYTES
            ):
                retained_bytes -= self._transcript_blocks[evicted].resident_bytes
                evicted += 1
            if evicted:
                self._fullscreen_semantic_replay_count -= sum(
                    block.replay is not None for block in self._transcript_blocks[:evicted]
                )
                del self._transcript_blocks[:evicted]
                self._fullscreen_transcript_bytes = retained_bytes
            return evicted
        # Native replay retains its existing bounded untagged tail (plus
        # tagged/kept blocks).
        if not block.keep and block.event_id is None and not block.tags:
            self._trim_untagged_transcript_tail()
        return 0

    def _trim_untagged_transcript_tail(self) -> None:
        """Bound source retention even when no resize replay has run yet."""
        untagged_indexes = [
            index
            for index, block in enumerate(self._transcript_blocks)
            if not block.keep and block.event_id is None and not block.tags
        ]
        excess = len(untagged_indexes) - self._untagged_replay_tail
        if excess <= 0:
            return
        discard = set(untagged_indexes[:excess])
        self._transcript_blocks = [
            block for index, block in enumerate(self._transcript_blocks) if index not in discard
        ]

    def _enqueue_transcript_block(self, block: TranscriptBlock) -> None:
        queue = self._block_queue
        # An off-thread callback accepted before teardown can run after the
        # ordered queue has retired and prompt_toolkit has restored the primary
        # screen. Fullscreen output is renderer-owned, so discard that stale
        # callback before mutating retained/view state or touching stdout.
        if queue is None and self._is_fullscreen:
            return

        rendered = self._render_transcript_source(block.source)
        evicted = self._retain_transcript_block(block, rendered)
        if self._is_fullscreen:
            self._fullscreen_transcript.append(
                rendered,
                record_id=block.transcript_record_id,
                copy_actions=block.code_copy_actions,
            )
            self._fullscreen_transcript.evict_prefix(evicted)
            self._note_unseen_agent_message(block)
            self._app.invalidate()
            return
        self._append_stripped_to_buffer(rendered)
        if queue is not None:
            queue.put_nowait(block)
            return

        # A call_soon_threadsafe callback can outlive the queue during
        # teardown. The terminal is no longer owned by prompt_toolkit then, so
        # deliver the already-retained block directly instead of dropping it.
        import sys as _sys

        out = _sys.__stdout__
        if out is not None:
            try:
                out.write(rendered)
                out.flush()
            except Exception:
                pass

    def _note_unseen_agent_message(self, block: TranscriptBlock) -> None:
        """Show a notice when agent prose arrives outside the anchored viewport."""
        if self._fullscreen_transcript.viewport.follows_tail:
            self._has_unseen_agent_message = False
        elif block.agent_message:
            self._has_unseen_agent_message = True

    def _append_stripped_to_buffer(self, text: str) -> None:
        """Append the ANSI-stripped transcript text to ``output_buffer``.

        Runs on the event loop thread (either because ``emit_block``
        scheduled it via ``call_soon_threadsafe`` or because we're still
        in the pre-consumer, single-threaded bootstrap phase).
        """
        stripped = _strip_ansi(text)
        existing = self.output_buffer.text
        appended = stripped if not existing or existing.endswith("\n") else "\n" + stripped
        joined = existing + appended
        self.output_buffer.document = Document(text=joined, cursor_position=len(joined))

    # ── surface the harness (and real callers) rely on ----------------

    @property
    def is_running(self) -> bool:
        return self._app.is_running

    def close_agent_observation(self) -> None:
        """Stop presentation delivery without owning or stopping the agent."""
        self._agent_controller.close()

    async def run_async(self) -> None:
        # Capture the loop once so emit_block can enqueue safely from
        # any thread without calling the deprecated get_event_loop().
        self._loop = asyncio.get_running_loop()
        self._block_queue = asyncio.Queue()
        self._resize_replays_enabled = self.full_screen or self._is_fullscreen
        self._consumer_task = None
        self._uninstall_stream_capture = None
        try:
            self._consumer_task = asyncio.ensure_future(self._consume_blocks())

            # Route stray sys.stdout / sys.stderr writes (aiohttp warnings,
            # litellm noise, stray prints) into the scrollback instead of
            # letting them corrupt prompt_toolkit's paint. Must install here
            # — before the first agent cell runs and before the framework
            # wraps sys.stdout with ContextVarStream — so agent-cell stdout
            # capture layers on top and still works unchanged.
            from .stream_forwarder import install_stray_stream_capture

            self._uninstall_stream_capture = install_stray_stream_capture(
                self.emit_block, on_stray=self._on_stray_output
            )
            self.observe_agent()
            # set_exception_handler=False keeps the handler Session installed
            # (_loud_handler) active for the whole app lifetime. Otherwise
            # prompt_toolkit replaces it with its own, which prints "Exception
            # None\nPress ENTER to continue..." for non-exception asyncio
            # contexts (e.g. "Task was destroyed but it is pending!") and
            # swallows every other diagnostic field.
            await self._app.run_async(set_exception_handler=False)
        finally:
            # No resize callback or queued replay may clear the terminal after
            # prompt_toolkit gives up ownership of it.
            self._resize_replays_enabled = False
            self._cancel_resize_replay_work()
            self._clear_ctrl_c_exit()
            if self._interrupt_status_clear_timer is not None:
                self._interrupt_status_clear_timer.cancel()
                self._interrupt_status_clear_timer = None
            self._interrupting_agent_turn = False
            self._interrupt_status_acknowledged = False
            self._interrupt_status_started_at = None
            if self._transient_status_timer is not None:
                self._transient_status_timer.cancel()
                self._transient_status_timer = None
            self._transient_status_text = ""
            self._transient_status_style = "class:status"
            clipboard_task = self._clipboard_task
            if clipboard_task is not None:
                clipboard_task.cancel()
                try:
                    await clipboard_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.debug("clipboard task failed during teardown", exc_info=True)
                if self._clipboard_task is clipboard_task:
                    self._clipboard_task = None
            link_task = self._link_task
            if link_task is not None:
                link_task.cancel()
                try:
                    await link_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.debug("link task failed during teardown", exc_info=True)
                if self._link_task is link_task:
                    self._link_task = None
            self._cancel_fullscreen_drag()
            # Restore sys.stdout / sys.stderr FIRST so any post-exit
            # prints from teardown code (spinner cleanup, snapshot save,
            # goodbye message) go straight to the real terminal rather
            # than back into the dying block queue.
            uninstall = getattr(self, "_uninstall_stream_capture", None)
            if uninstall is not None:
                try:
                    uninstall()
                except Exception:
                    pass
                self._uninstall_stream_capture = None

            # Detach before the first teardown await.  Runtime events released
            # while policy/host shutdown yields are then generation-filtered and
            # cannot mutate renderer state after prompt-toolkit has exited.
            try:
                self.close_agent_observation()
            except BaseException:
                logger.exception("agent observation teardown failed")

            # The composition root owns agent/policy lifecycle. Quiesce every
            # producer that can still call ``emit_block`` before retiring the
            # sole ordered consumer, so final in-flight output joins the FIFO.
            if self._host_services.before_output_drain is not None:
                try:
                    await self._host_services.before_output_drain()
                except Exception:
                    logger.debug("output producer quiescence failed", exc_info=True)

            # Let the single consumer finish ordinary blocks queued during
            # teardown (e.g. 'Goodbye! Stay vibing.' from /exit). Once the
            # prompt_toolkit app has exited, run_in_terminal executes its
            # callable directly, so preserving the FIFO is both safe and less
            # lossy than racing a manual drain against an in-flight consumer.
            import sys as _sys

            q = self._block_queue
            if q is not None and self._consumer_task is not None:
                await asyncio.sleep(0)
                try:
                    await asyncio.wait_for(q.join(), timeout=1.0)
                except TimeoutError:
                    logger.debug("timed out draining TUI output during teardown")
            if self._consumer_task is not None:
                self._consumer_task.cancel()
                try:
                    await self._consumer_task
                except asyncio.CancelledError:
                    pass
                except BaseException:
                    pass
            # Later UI-loop cleanup callbacks bypass the retired queue and use
            # emit_block's direct-output path. Keep the local q for the final
            # fallback drain below.
            self._block_queue = None

            # A timed-out or exceptionally stopped consumer can leave queued
            # ordinary blocks behind. Flush those directly, but deliberately
            # discard resize barriers now that replay is disabled.
            if q is not None:
                while not q.empty():
                    try:
                        item = q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    try:
                        if (
                            isinstance(item, TranscriptBlock)
                            and item.transcript_epoch == self._transcript_epoch
                        ):
                            out = _sys.__stdout__
                            if out is not None:
                                try:
                                    out.write(self._render_replay_source(item.source))
                                    out.flush()
                                except Exception:
                                    pass
                    finally:
                        q.task_done()
            if self._spinner_task is not None and not self._spinner_task.done():
                self._spinner_task.cancel()
                # Await so the spinner's finally block runs (invalidate())
                # and asyncio doesn't emit "Task was destroyed" on loop
                # close. CancelledError on a cancelled task is expected.
                try:
                    await self._spinner_task
                except (asyncio.CancelledError, BaseException):
                    pass
            self._consumer_task = None
            self._spinner_task = None
            self._queued_resize_replay_generation = None
            self._replay_columns_override = None
            self._loop = None

    async def _consume_blocks(self) -> None:
        """Drain ``_block_queue`` forever; write each block above the
        prompt via ``run_in_terminal`` → ``sys.__stdout__``.

        One consumer, FIFO order, no races. Writing to ``__stdout__``
        (not ``sys.stdout``) bypasses the framework's ContextVarStream
        wrapper so ``self.message()`` content never gets captured as
        cell stdout.
        """
        import sys as _sys

        from prompt_toolkit.application import run_in_terminal

        assert self._block_queue is not None
        while True:
            item = await self._block_queue.get()

            try:
                if isinstance(item, TranscriptBlock):

                    def _write(block: TranscriptBlock = item) -> None:
                        if block.transcript_epoch != self._transcript_epoch:
                            return
                        out = _sys.__stdout__
                        if out is not None:
                            # A block may have waited behind another terminal
                            # operation while the pane narrowed. Enforce the
                            # physical-width invariant at the actual write,
                            # not only when the item entered the FIFO.
                            out.write(self._render_replay_source(block.source))
                            out.flush()

                    await run_in_terminal(_write)
                elif isinstance(item, _ResizeReplayQueueItem):
                    await self._consume_resize_replay(item)
                else:
                    await self._consume_clear_transcript(item)
            except asyncio.CancelledError:
                raise
            except Exception:
                # Best-effort — a single failed write shouldn't wedge
                # the consumer. Fall through and pick up the next block.
                continue
            finally:
                self._block_queue.task_done()

    def output_columns(self, minimum: int = 20) -> int:
        """Current transcript render width, including an active resize replay."""
        if self._replay_columns_override is not None:
            return max(int(self._replay_columns_override), minimum)
        try:
            return max(int(self._app.output.get_size().columns), minimum)
        except Exception:
            return terminal_cols(minimum=minimum)

    def transcript_columns(self) -> int:
        """Safe printable width for native-scrollback blocks.

        Direct terminal output runs with autowrap enabled.  Reserving the last
        physical column avoids the delayed-wrap ambiguity where a subsequent
        newline can create an extra row and move prompt_toolkit's live-region
        origin.  Unlike ``output_columns``, this never inflates a genuinely
        narrow terminal to an arbitrary rendering minimum.
        """
        if self._replay_columns_override is not None:
            physical = int(self._replay_columns_override)
        else:
            current = self._read_terminal_size()
            physical = current[0] if current is not None else terminal_cols(minimum=1)
        return max(physical - 1, 1)

    def _render_transcript_source(self, source: str) -> str:
        """Normalize a block for its selected renderer ownership model."""
        if self._is_fullscreen:
            # prompt_toolkit owns wrapping and reflow in alternate-screen mode.
            return normalize_transcript_block(source)
        return self._render_replay_source(source)

    def _render_replay_source(self, source: str) -> str:
        """Normalize source text at the native terminal's current safe width."""
        return normalize_transcript_block(source, columns=self.transcript_columns())

    def _defer_prompt_toolkit_resize_redraw(self) -> bool:
        """Fold native-replay width changes into the settled transcript replay.

        prompt_toolkit normally erases and redraws immediately on SIGWINCH. In
        native-replay mode that live-region paint is followed by our debounced
        clear, semantic transcript replay, and another live-region paint. Width
        changes can safely skip the first paint: the atomic native replay
        performs the final redraw. Row-only changes stay immediate.
        """
        if not self._resize_replays_enabled:
            return False
        current = self._read_terminal_size()
        if current is None:
            return False
        previous = self._resize_reflow.observed_size

        if self._is_fullscreen:
            # Alternate-screen mode owns every cell, so even a row-only resize
            # can repaint the whole viewport. Hold SIGWINCH paints until the
            # geometry settles rather than exposing each transient tmux size.
            if previous == current:
                return self._fullscreen_rebuild_timer is not None
            self._resize_reflow.observe(current)
            if previous is None or self._active_subview is not None:
                return False
            self._resize_redraw_deferred = True
            self._schedule_fullscreen_rebuild()
            return True

        if not self.full_screen:
            return False
        if self._app._running_in_terminal:
            # Never let SIGWINCH erase an external editor/shell, including an
            # unchanged duplicate signal. Observe changed geometry so a width
            # replay remains pending; row-only changes use the handoff's final
            # prompt_toolkit redraw after ownership returns.
            if previous != current:
                self._observe_terminal_size(current)
            self._resize_redraw_deferred = True
            return True
        if previous == current:
            # Coalesce duplicate signals while a width transaction is pending.
            # Once settled, still let prompt_toolkit repaint its live region
            # when an unchanged pane is exposed again.
            return self._resize_reflow.has_pending_replay

        self._observe_terminal_size(current)
        if previous is None or self._active_subview is not None:
            return False
        if previous[0] != current[0] or self._resize_reflow.has_pending_replay:
            self._resize_redraw_deferred = True
            return True
        return False

    def _prompt_toolkit_resize_redraw_is_deferred(self) -> bool:
        """Keep a pending resize hidden until its terminal transaction can run."""
        if not self._resize_redraw_deferred or self._active_subview is not None:
            return False
        if self.full_screen and self._resize_replays_enabled:
            if self._resize_reflow.has_pending_replay:
                if (
                    self._resize_replay_timer is None
                    and self._queued_resize_replay_generation is None
                ):
                    # ``run_in_terminal`` suppresses prompt_toolkit redraws while
                    # an external program owns the terminal. Its final redraw
                    # reaches this hook after ownership returns; resume the
                    # deferred width replay then.
                    self._schedule_resize_replay()
                return True
            # A row-only SIGWINCH needed no semantic replay. Let the handoff's
            # own final redraw publish the new layout exactly once.
            self._resize_redraw_deferred = False
            return False
        return True

    def _finish_deferred_resize_redraw(self) -> None:
        if not self._resize_redraw_deferred or not self._resize_replays_enabled:
            return
        self._resize_redraw_deferred = False
        self._app.redraw_after_deferred_resize()

    def _before_render(self, _app) -> None:
        """Observe frame-local status and terminal geometry before rendering."""
        if self._is_fullscreen:
            self._status_region_occupied = bool(self._status_rows())
        current = self._read_terminal_size()
        if current is None:
            return
        if self._is_fullscreen:
            previous = self._resize_reflow.observed_size
            self._resize_reflow.observe(current)
            if previous is not None and previous[0] != current[0]:
                self._schedule_fullscreen_rebuild()
            return
        if self.full_screen:
            self._observe_terminal_size(current)

    def _after_render(self, app) -> None:
        """Never leave terminal-native OSC-8 state open beyond one rendered frame."""
        if self._is_fullscreen:
            try:
                app.output.write_raw("\x1b]8;;\x1b\\")
                app.output.flush()
            except Exception:
                logger.debug("failed to close native hyperlink state", exc_info=True)

    def _read_terminal_size(self) -> tuple[int, int] | None:
        try:
            size = self._app.output.get_size()
            return int(size.columns), int(size.rows)
        except Exception:
            return None

    def _observe_terminal_size(
        self,
        size: tuple[int, int],
    ) -> None:
        previous_size = self._resize_reflow.observed_size
        observation = self._resize_reflow.observe(size)
        recovery_requested = False
        if self._active_subview is None:
            compressed = self._main_layout_is_compressed(size)
            rows_changed = previous_size is not None and previous_size[1] != size[1]
            if rows_changed and compressed:
                self._height_compaction_needs_replay = True
            elif self._height_compaction_needs_replay and not compressed:
                recovery_requested = self._resize_reflow.request_replay()
                self._height_compaction_needs_replay = False
        if observation.changed:
            self._resize_replay_failure_generation = None
        replay_is_queued = self._queued_resize_replay_generation == self._resize_reflow.generation
        replay_failed = self._resize_replay_failure_generation == self._resize_reflow.generation
        should_schedule = (
            recovery_requested
            or observation.should_debounce
            or (
                self._resize_reflow.has_pending_replay
                and self._resize_replay_timer is None
                and not replay_is_queued
                and not replay_failed
            )
        )
        if self._active_subview is None and should_schedule:
            self._schedule_resize_replay()

    def _schedule_fullscreen_rebuild(self) -> None:
        """Coalesce fullscreen redraws until resize input settles."""
        loop = self._loop
        if loop is None or loop.is_closed():
            if self._fullscreen_semantic_replay_count:
                self._rebuild_fullscreen_transcript()
            else:
                # Static records project at render time; preserve the O(1)
                # synchronous fallback used before the application loop starts.
                self._cancel_fullscreen_drag()
                self._mark_fullscreen_resize_replayed()
                self._fullscreen_invalidate_count += 1
                self._app.invalidate()
            return
        if self._fullscreen_rebuild_timer is not None:
            self._fullscreen_rebuild_timer.cancel()
        self._fullscreen_rebuild_generation += 1
        generation = self._fullscreen_rebuild_generation
        self._fullscreen_rebuild_timer = loop.call_later(
            TRANSCRIPT_REFLOW_DEBOUNCE_SECONDS,
            self._run_scheduled_fullscreen_rebuild,
            generation,
        )

    def _run_scheduled_fullscreen_rebuild(self, generation: int) -> None:
        if generation != self._fullscreen_rebuild_generation:
            return
        self._fullscreen_rebuild_timer = None
        if not self._resize_replays_enabled:
            return
        if self._fullscreen_semantic_replay_count:
            self._rebuild_fullscreen_transcript()
        else:
            # Static records are projected by the fullscreen model at render
            # time, so only the settled prompt_toolkit frame is required.
            self._cancel_fullscreen_drag()
            self._mark_fullscreen_resize_replayed()
            self._fullscreen_invalidate_count += 1
            self._finish_fullscreen_resize_redraw()

    def _mark_fullscreen_resize_replayed(self) -> None:
        request = self._resize_reflow.prepare_replay()
        if request is not None and self._resize_reflow.is_current(request):
            self._resize_reflow.mark_replayed(request)

    def _finish_fullscreen_resize_redraw(self) -> None:
        if self._resize_redraw_deferred:
            self._resize_redraw_deferred = False
            self._app.redraw_after_deferred_resize()
        else:
            self._app.invalidate()

    def _rebuild_fullscreen_transcript(self) -> None:
        """Refresh width-sensitive semantic blocks from a one-width cache."""
        self._cancel_fullscreen_drag()
        width = self.transcript_columns()
        chunks: list[str] = []
        retained_bytes = 0
        for block in self._transcript_blocks:
            if block.replay is None:
                rendered = block.fullscreen_rendered
                if rendered is None:
                    rendered = self._render_transcript_source(block.source)
            else:
                source = block.replay_cache.get(width)
                if source is None:
                    try:
                        source = block.replay()
                    except Exception:
                        source = block.source
                    block.replay_cache = {width: source}
                rendered = self._render_transcript_source(source)
            block.fullscreen_rendered = rendered
            block.resident_bytes = self._fullscreen_block_resident_bytes(block, rendered)
            retained_bytes += block.resident_bytes
            chunks.append(rendered)
        evicted = 0
        while evicted < len(self._transcript_blocks) and (
            len(self._transcript_blocks) - evicted > _FULLSCREEN_TRANSCRIPT_MAX_RECORDS
            or retained_bytes > _FULLSCREEN_TRANSCRIPT_MAX_BYTES
        ):
            retained_bytes -= self._transcript_blocks[evicted].resident_bytes
            evicted += 1
        if evicted:
            self._fullscreen_semantic_replay_count -= sum(
                block.replay is not None for block in self._transcript_blocks[:evicted]
            )
            del self._transcript_blocks[:evicted]
            del chunks[:evicted]
        self._fullscreen_transcript_bytes = retained_bytes
        self._fullscreen_transcript.replace(
            chunks,
            record_ids=[
                block.transcript_record_id if block.transcript_record_id is not None else index
                for index, block in enumerate(self._transcript_blocks)
            ],
            copy_actions=[block.code_copy_actions for block in self._transcript_blocks],
        )
        self._mark_fullscreen_resize_replayed()
        self._fullscreen_invalidate_count += 1
        self._finish_fullscreen_resize_redraw()

    def _main_layout_is_compressed(self, size: tuple[int, int]) -> bool:
        """Return whether optional main-view rows cannot all fit."""
        if self._active_subview is not None:
            return False
        columns, rows = size
        try:
            preferred = self._main_container.preferred_height(columns, rows).preferred
        except Exception:
            # The fixed normal chrome is status + rule + padded input.  This
            # fallback is only for a broken third-party dimension callback.
            preferred = 5
        return preferred > rows

    def _schedule_resize_replay(self) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        if self._resize_replay_timer is not None:
            self._resize_replay_timer.cancel()
        self._resize_replay_schedule_generation += 1
        schedule_generation = self._resize_replay_schedule_generation
        self._resize_replay_timer = loop.call_later(
            TRANSCRIPT_REFLOW_DEBOUNCE_SECONDS,
            self._start_resize_replay,
            schedule_generation,
        )

    def _start_resize_replay(
        self,
        schedule_generation: int,
    ) -> None:
        if schedule_generation != self._resize_replay_schedule_generation:
            return
        self._resize_replay_timer = None
        if not self._resize_replays_enabled:
            return
        if self._app._running_in_terminal:
            # An external editor/shell currently owns the terminal. Leave the
            # request pending; prompt_toolkit's handoff redraw will schedule it
            # through ``_prompt_toolkit_resize_redraw_is_deferred``.
            return
        current = self._read_terminal_size()
        if current is None:
            # Keep the pending width. Restore prompt_toolkit rendering so a
            # later successful frame can observe and schedule it again.
            self._finish_deferred_resize_redraw()
            return
        observation = self._resize_reflow.observe(current)
        if observation.should_debounce:
            self._schedule_resize_replay()
            return

        if self._active_subview is not None:
            return

        request = self._resize_reflow.prepare_replay()
        if request is None:
            self._finish_deferred_resize_redraw()
            return

        queue = self._block_queue
        if queue is None:
            return
        if self._queued_resize_replay_generation is not None:
            # Keep at most one replay barrier in the FIFO. If this one is stale,
            # its consumer schedules the latest pending width after removing it.
            return

        # Replay is a barrier in the same FIFO as ordinary terminal writes. The
        # snapshot is taken at the barrier's exact queue position, so later
        # blocks are written once after the rebuilt prefix instead of appearing
        self._prune_transcript_blocks_for_active_events()
        self._queued_resize_replay_generation = request.generation
        queue.put_nowait(
            _ResizeReplayQueueItem(
                request=request,
                transcript_blocks=tuple(self._transcript_blocks),
                transcript_epoch=self._transcript_epoch,
            )
        )

    async def _consume_clear_transcript(self, item: _ClearTranscriptQueueItem) -> None:
        from prompt_toolkit.application import run_in_terminal

        await run_in_terminal(lambda: self._clear_terminal_if_current(item))

    def _clear_terminal_if_current(self, item: _ClearTranscriptQueueItem) -> bool:
        if not self._resize_replays_enabled or item.transcript_epoch != self._transcript_epoch:
            return False
        import sys as _sys

        out = _sys.__stdout__
        if out is None:
            return False
        try:
            out.write(_TRANSCRIPT_CLEAR_SEQUENCE)
            out.flush()
            self._height_compaction_needs_replay = False
            return True
        except Exception:
            return False

    async def _consume_resize_replay(self, item: _ResizeReplayQueueItem) -> None:
        schedule_latest_pending = False
        try:
            if (
                not self._resize_replays_enabled
                or item.transcript_epoch != self._transcript_epoch
                or not self._resize_reflow.is_current(item.request)
            ):
                schedule_latest_pending = self._resize_reflow.has_pending_replay
                return
            if self._active_subview is not None:
                return
            if self._app._running_in_terminal:
                # A terminal handoff can begin after this barrier was queued.
                # Never erase/replay over an external editor or shell; leave the
                # request pending for prompt_toolkit's handoff redraw to resume.
                schedule_latest_pending = self._resize_reflow.has_pending_replay
                return
            if not item.transcript_blocks and not item.request.required:
                self._resize_reflow.mark_replayed(item.request)
                self._resize_replay_failure_generation = None
                self._finish_deferred_resize_redraw()
                return

            self._replay_columns_override = item.request.width
            transaction_completed = False
            try:
                replayed = self._app.run_atomic_native_replay(
                    lambda output: self._replay_queue_item_if_current(item, output=output)
                )
                transaction_completed = True
            except asyncio.CancelledError:
                raise
            except Exception:
                replayed = False
            finally:
                self._replay_columns_override = None
                if transaction_completed:
                    self._resize_redraw_deferred = False
                else:
                    # Recover the live region if an output implementation
                    # failed before the transaction's final flush.
                    self._finish_deferred_resize_redraw()

            if replayed:
                # Commit only after the single physical flush succeeds. A
                # failed flush therefore remains eligible for the bounded
                # retry below instead of silently accepting a missing replay.
                self._resize_reflow.mark_replayed(item.request)
                self._resize_replay_failure_generation = None
                self._height_compaction_needs_replay = False
                self._fullscreen_invalidate_count += 1
                # Verify geometry once afterward. A genuinely newer width
                # starts its own transaction; an unchanged width is a no-op.
                current = self._read_terminal_size()
                if current is not None:
                    self._observe_terminal_size(current)
            elif (
                item.transcript_epoch != self._transcript_epoch
                or not self._resize_reflow.is_current(item.request)
            ):
                schedule_latest_pending = self._resize_reflow.has_pending_replay
            elif self._resize_replay_failure_generation != item.request.generation:
                # One automatic retry recovers a transient stdout failure but
                # cannot spin forever on a permanently broken terminal.
                self._resize_replay_failure_generation = item.request.generation
                schedule_latest_pending = self._resize_reflow.has_pending_replay
        finally:
            if self._queued_resize_replay_generation == item.request.generation:
                self._queued_resize_replay_generation = None
            if (
                schedule_latest_pending
                and self._resize_replays_enabled
                and self._active_subview is None
                and self._resize_replay_timer is None
            ):
                self._schedule_resize_replay()

    def _replay_queue_item_if_current(
        self, item: _ResizeReplayQueueItem, *, output: Any | None = None
    ) -> bool:
        if (
            not self._resize_replays_enabled
            or self._active_subview is not None
            or item.transcript_epoch != self._transcript_epoch
            or not self._resize_reflow.is_current(item.request)
            or not self._resize_output_width_is_current(item)
        ):
            return False
        return self._replay_fullscreen_transcript(
            item.transcript_blocks,
            clear_even_if_empty=item.request.required,
            still_current=lambda: (
                self._resize_replays_enabled
                and self._active_subview is None
                and item.transcript_epoch == self._transcript_epoch
                and self._resize_reflow.is_current(item.request)
                and self._resize_output_width_is_current(item)
            ),
            output=output,
            flush=False,
            count_invalidation=False,
        )

    def _resize_output_width_is_current(self, item: _ResizeReplayQueueItem) -> bool:
        current = self._read_terminal_size()
        return current is not None and current[0] == item.request.width

    def _cancel_resize_replay_work(self) -> None:
        self._resize_replay_schedule_generation += 1
        if self._resize_replay_timer is not None:
            self._resize_replay_timer.cancel()
            self._resize_replay_timer = None
        self._resize_redraw_deferred = False
        self._fullscreen_rebuild_generation += 1
        if self._fullscreen_rebuild_timer is not None:
            self._fullscreen_rebuild_timer.cancel()
            self._fullscreen_rebuild_timer = None

    @staticmethod
    def _tag_range(tag: str) -> tuple[int, int] | None:
        try:
            if ".." in tag:
                a, b = tag.split("..", 1)
                return int(a), int(b)
            n = int(tag)
            return n, n
        except Exception:
            return None

    def _active_replay_identity(self) -> tuple[set[str], list[tuple[int, int]]]:
        if self._host_services.replay_identity is None:
            return set(), []
        try:
            active_ids, ranges = self._host_services.replay_identity()
        except Exception:
            logger.debug("replay identity provider failed", exc_info=True)
            return set(), []
        return set(active_ids), list(ranges)

    def _tag_is_active(self, tag: str, active_ranges: list[tuple[int, int]]) -> bool:
        rng = self._tag_range(tag)
        if rng is None:
            return False
        start, end = rng
        return any(
            start <= active_end and end >= active_start
            for active_start, active_end in active_ranges
        )

    def _prune_transcript_blocks_for_active_events(self) -> None:
        if not self._transcript_blocks:
            return
        active_ids, active_ranges = self._active_replay_identity()
        has_active_identity = bool(active_ids or active_ranges)
        keep_indexes: set[int] = set()
        untagged_indexes: list[int] = []
        for i, block in enumerate(self._transcript_blocks):
            if block.keep:
                keep_indexes.add(i)
            elif block.event_id is not None:
                if not has_active_identity or block.event_id in active_ids:
                    keep_indexes.add(i)
            elif block.tags:
                if not has_active_identity or any(
                    self._tag_is_active(tag, active_ranges) for tag in block.tags
                ):
                    keep_indexes.add(i)
            else:
                untagged_indexes.append(i)
        keep_indexes.update(untagged_indexes[-self._untagged_replay_tail :])
        self._transcript_blocks = [
            block for i, block in enumerate(self._transcript_blocks) if i in keep_indexes
        ]

    def _replay_fullscreen_transcript(
        self,
        transcript_blocks: tuple[TranscriptBlock, ...] | None = None,
        *,
        clear_even_if_empty: bool = False,
        still_current: Callable[[], bool] | None = None,
        output: Any | None = None,
        flush: bool = True,
        count_invalidation: bool = True,
    ) -> bool:
        if not self.full_screen:
            return False
        if output is None:
            import sys as _sys

            output = _sys.__stdout__
        if output is None:
            return False
        if transcript_blocks is None:
            self._prune_transcript_blocks_for_active_events()
            transcript_blocks = tuple(self._transcript_blocks)
        if not transcript_blocks and not clear_even_if_empty:
            return False
        chunks: list[str] = []
        for block in transcript_blocks:
            try:
                # Semantic callbacks and retained sources cross the same
                # trust/width boundary at replay time. Re-normalizing sources
                # is what makes arbitrary diagnostics and stream
                # output safe after a narrower resize too.
                source = block.replay() if block.replay is not None else block.source
            except Exception:
                source = block.source
            chunks.append(self._render_replay_source(source))
        if still_current is not None and not still_current():
            return False
        try:
            # Reset scroll region + style state before clearing. Homing again
            # after the purge gives prompt_toolkit a stable origin for its live
            # input/status redraw at the end of the atomic transaction.
            write = getattr(output, "write_raw", None)
            if write is None:
                write = output.write
            write(_TRANSCRIPT_CLEAR_SEQUENCE)
            write("".join(chunks))
            if flush:
                output.flush()
            if count_invalidation:
                self._fullscreen_invalidate_count += 1
            return True
        except Exception:
            return False

    def exit(self) -> None:
        if self._app.is_running:
            self._app.exit()

    def prompt_char_visible(self) -> bool:
        """True once the prompt-marker processor is attached to the input."""
        return self._prompt_processor is not None

    def input_cursor_position(self) -> int:
        """Current cursor position within the input buffer (0-indexed)."""
        return self.input_buffer.cursor_position

    def is_thinking(self) -> bool:
        """True while the dispatcher is inside ``agent.handle()``.

        Tracked by the ``_in_respond`` flag rather than the user_messages
        queue's waiter count: the agent can ``await self.user_messages.get()``
        mid-turn (clarification flow), and during that wait the queue has
        a waiter — but the agent is genuinely thinking, not idle. The
        flag captures the dispatcher → handle() boundary directly.
        """
        state = self._agent_controller.state
        return state is not None and state.lifecycle is AgentLifecycle.THINKING

    def commands_dispatched(self) -> list[str]:
        """Slash commands the user has submitted, in order."""
        return list(self._commands_dispatched)

    def last_bang_command(self) -> str | None:
        """Most recent ``!shell-command`` the user submitted, or None."""
        return self._last_bang_command

    def completion_candidates(self) -> list[str]:
        """Completion candidates currently offered for the input buffer text.

        Returns each candidate as the *full* replacement string (i.e. what
        the buffer would contain if that candidate were applied) — so a
        Completion(text='/help', start_position=-3) against buffer '/he'
        reads back as '/help', not '/he/help'.
        """
        from prompt_toolkit.completion import CompleteEvent

        doc = self.input_buffer.document
        before = doc.text_before_cursor
        result = []
        for c in self._completer.get_completions(doc, CompleteEvent()):
            prefix = before[: c.start_position] if c.start_position < 0 else before
            result.append(prefix + c.text)
        return result

    def set_command_status(self, text: str) -> None:
        """Set transient command lifecycle text in the dynamic status area."""
        self._command_status_text = text
        app = getattr(self, "_app", None)
        if app is not None and app.is_running:
            app.invalidate()

    def set_command_queue(self, commands: list[str]) -> None:
        """Set queued command text shown in the dynamic queue area."""
        self._command_queue_texts = list(commands)
        app = getattr(self, "_app", None)
        if app is not None and app.is_running:
            app.invalidate()

    def set_llm_probe_status(self, text: str) -> None:
        """Set transient LLM startup probe text in the dynamic status area."""
        self._llm_probe_status_text = text
        if text:
            self._ensure_spinner_task()
        self.invalidate()

    def _status_rows(self, *, include_transient: bool = True) -> list[list[tuple[str, str]]]:
        """Return dynamic status rows as independently styled fragments."""
        rows: list[list[tuple[str, str]]] = []
        state = self._agent_controller.state
        if self._agent_controller.failure is not None:
            rows.append([("class:status", "Agent observation disconnected.")])
        if self._interrupting_agent_turn or (
            state is not None and state.workspace.cancellation is CancellationState.REQUESTED
        ):
            rows.append([("class:status", "Interrupting agent turn")])
        elif self.is_thinking():
            rows.append([("class:status", f"{self._spinner_frame} thinking...")])
        if self._llm_probe_status_text:
            rows.append([("class:status", f"{self._spinner_frame} {self._llm_probe_status_text}")])
        auxiliary_status = ""
        if self._host_services.auxiliary_status is not None:
            try:
                auxiliary_status = self._host_services.auxiliary_status()
            except Exception:
                logger.debug("auxiliary status callback failed", exc_info=True)
        if auxiliary_status:
            rows.append([("class:status", auxiliary_status)])
        if include_transient and self._transient_status_text:
            rows.append([(self._transient_status_style, self._transient_status_text)])
        if self._command_status_text:
            rows.append([("class:status", self._command_status_text)])
        if self._exit_hint_text:
            rows.append([("class:status", self._exit_hint_text)])
        if self._session_label:
            label = f"[{self._session_label}]"
            if rows:
                rows[-1].append(("class:status", f"   {label}"))
            else:
                rows.append([("class:status", label)])
        return rows

    def status_text(self) -> str:
        """Plain-text projection of the dynamic status rows."""
        return "\n\n".join("".join(text for _style, text in row) for row in self._status_rows())

    def set_session_label(self, label: str) -> None:
        """Set the bracketed label shown on the right of the status line."""
        self._session_label = label
