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
import datetime
import logging
import re
import shutil
import subprocess
import threading
from collections.abc import Awaitable, Callable
from concurrent.futures import Future as ConcurrentFuture
from dataclasses import dataclass
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout import ConditionalContainer, DynamicContainer, HSplit, Layout, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.layout.menus import CompletionsMenuControl
from prompt_toolkit.layout.processors import BeforeInput
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType

from .completer import expand_mentions
from .input_handler import create_prompt_style
from .subapp import InAppSubview, normalize_key_result

logger = logging.getLogger(__name__)


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
        action = {
            MouseEventType.SCROLL_UP: "scroll_up",
            MouseEventType.SCROLL_DOWN: "scroll_down",
        }.get(mouse_event.event_type)
        if action is not None and self._mouse_callback(
            action, mouse_event.position.x, mouse_event.position.y
        ):
            return None
        return super().mouse_handler(mouse_event)


# CSI + OSC stripper for the plain-text view of output_buffer. We keep
# the original ANSI in _output_ansi; tests that assert on buffer text
# don't want escape sequences in their comparisons.
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07]*\x07")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


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
    width = max(cols - 1, 1)
    if label:
        # Clamp the whole line (fill + space + label) to ``width`` so an
        # over-long label can't push the rule into the final column and
        # re-introduce the resize residue. The dash branch needs room for at
        # least one dash plus the separating space, i.e. ``len(label) <=
        # width - 2``; otherwise (including ``len(label) == width - 1``, where
        # ``max(..., 1)`` would silently round the fill back up and re-bleed to
        # the final column) fall straight through to truncation, keeping the
        # label tail (the context-usage / id that matters most).
        if len(label) > width - 2:
            return [("class:rule.label", label[-width:])]
        dashes = width - len(label) - 1  # >= 1 by the guard above
        return [
            ("class:rule", "─" * dashes + " "),
            ("class:rule.label", label),
        ]
    return [("class:rule", "─" * width)]


PROMPT_MARKER = "❯ "
_CTRL_C_EXIT_WINDOW_SECONDS = 2.0


@dataclass
class ReplayBlock:
    raw: str
    replay: Callable[[], str] | None = None
    event_id: str | None = None
    tags: frozenset[str] = frozenset()
    keep: bool = False


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


class TUIApplication:
    """Owns a single, long-lived ``prompt_toolkit.Application`` for the TUI."""

    def __init__(
        self,
        agent: Any = None,
        *,
        on_command: Callable[[str], Awaitable[None] | None] | None = None,
        on_cancel_command: Callable[[], bool] | None = None,
        on_bang: Callable[[str], Awaitable[None] | None] | None = None,
        on_output: Callable[[Any], Awaitable[None] | None] | None = None,
        completer: Completer | None = None,
        session_label: Callable[[], str] | None = None,
        config: Any = None,
        full_screen: bool = True,
        submission_guard: Callable[[], str | None] | None = None,
    ) -> None:
        """
        Args:
            agent: Object with an async ``handle()`` method that pumps
                input queues (see ``BaseTUIAgent.handle``).
            on_command: Called with the raw slash text (e.g. ``"/help"``)
                whenever the user submits one. Session wires this to its
                CommandRegistry. If omitted, commands still land in
                ``commands_dispatched()`` for introspection but nothing
                runs.
            on_cancel_command: Synchronously request cancellation of the
                active slash command. Returns whether a command accepted the
                request. Esc falls back to interrupting the agent when false.
            on_bang: Called with the bang body (e.g. ``"echo hi"`` for
                ``!echo hi``). Session wires this to run_in_terminal +
                bash. If omitted, bang commands are only recorded in
                ``last_bang_command()``.
            on_output: Called with structured ``Output`` values emitted by
                dispatcher-level behavior. Session wires this to
                ``frontend.render(...)``.
            completer: Optional prompt_toolkit ``Completer`` for Tab
                completion. When omitted no completion is offered.
            full_screen: Native-scrollback mode. Transcript output is written to
                the terminal scrollback; resize only redraws prompt_toolkit's
                live input/status region.
            submission_guard: Returns an actionable error when plain agent-bound
                input must be rejected. Slash and bang commands remain available.

        The per-message echo ("queued → accepted" transition, user-bar
        render, SessionUserMessage log) is wired on the agent's
        ``_user_messages_in`` Channel via ``set_on_get``. The
        dispatcher itself doesn't call back — that would double-fire
        the echo when the agent dequeues a message mid-turn.
        """
        self.agent = agent
        self._on_command = on_command
        self._on_cancel_command = on_cancel_command
        self._on_bang = on_bang
        self._on_output = on_output
        self._session_label_fn: Callable[[], str] | None = session_label
        self._config = config
        self._submission_guard = submission_guard
        self._ctrl_c_exit_armed = False
        self._ctrl_c_exit_timer: asyncio.TimerHandle | None = None
        self._exit_hint_text = ""

        # Register a QueueManager notify callback to also start the
        # dispatcher when items arrive on non-user channels while idle.
        # Without this, slash commands or spawned jobs that produce output
        # before any user message is typed would queue items indefinitely.
        # The callback may fire from the agent-loop thread (spawned jobs),
        # so schedule _ensure_dispatcher_task on the UI loop to avoid
        # attaching dispatcher state to the wrong event loop.
        if agent is not None:
            qm = getattr(agent, "queue_manager", None)
            if qm is not None:
                qm.set_notify_callback(self._on_queue_notify)
        self.full_screen = full_screen

        # Output scrollback. Two parallel stores:
        #   * ``output_buffer`` — plain-text view, used by tests and by
        #     callers that want a printable transcript. ANSI stripped.
        #   * ``_output_ansi`` — raw ANSI chunks rendered into the live
        #     TUI via FormattedTextControl so Rich styling survives.
        self.output_buffer = Buffer(read_only=False)
        self._output_ansi: list[str] = []
        # Retained transcript replay units. On resize we intentionally clear
        # the visible screen + terminal scrollback, then rewrite these blocks.
        self._replay_blocks: list[ReplayBlock] = []
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
        self.input_buffer = Buffer(
            multiline=True,
            completer=self._completer,
            complete_while_typing=False,
            accept_handler=self._accept_handler,
        )
        self.input_buffer.on_text_changed += self._on_input_text_changed

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

        self._prompt_processor = BeforeInput(PROMPT_MARKER, style="class:prompt")
        self._agent_task: asyncio.Future | None = None
        self._agent_thread_future: ConcurrentFuture | None = None
        self._agent_loop_task: asyncio.Task | None = None
        self._agent_loop: asyncio.AbstractEventLoop | None = None
        self._agent_sentinel: asyncio.Task | None = None
        self._agent_thread: threading.Thread | None = None
        self._agent_loop_ready = threading.Event()
        self._agent_loop_stopped = threading.Event()
        # True while the dispatcher is inside an ``agent.handle()`` call.
        # ``is_thinking()`` checks this flag rather than the user_messages
        # queue's waiter count — counting waiters conflated "dispatcher
        # idle between turns" with "agent mid-turn awaiting clarification
        # via await self.user_messages.get()". The latter case had the
        # spinner stop while the agent was genuinely working.
        self._in_respond: bool = False
        self._agent_cancel_requested: bool = False
        self._keep_going_tasks: set[asyncio.Task[Any]] = set()
        self._keep_going_generation: int = 0

        # Single producer-many-consumers path for transcript content:
        # emit_block() enqueues one ANSI chunk; a single background task
        # (started in run_async) drains the queue in order and writes
        # each chunk via run_in_terminal → sys.__stdout__. Everything
        # that used to have its own scheduling (patch_stdout proxy,
        # direct run_in_terminal in _render_message, etc.) now funnels
        # through this one queue — no races.
        self._block_queue: asyncio.Queue[str] | None = None
        self._consumer_task: asyncio.Task | None = None
        # Diagnostic/test counter for fullscreen clear+rewrite replays.
        self._fullscreen_invalidate_count = 0
        self._last_rendered_size: tuple[int, int] | None = None
        self._replay_columns_override: int | None = None
        # Captured in run_async; used by emit_block for thread-safe
        # enqueue without calling the deprecated asyncio.get_event_loop().
        self._loop: asyncio.AbstractEventLoop | None = None
        # Set by run_async's stdout/stderr forwarder install; called in
        # the finally to restore the real streams.
        self._uninstall_stream_capture: Callable[[], None] | None = None

        # Let the FakeAgent (or real agent) push output into our scrollback
        # without knowing about our internals.
        if hasattr(agent, "emit"):
            agent.emit = self.emit_block  # type: ignore[attr-defined]

        kb = self._build_key_bindings()

        # Transcript output lives in terminal scrollback. In full_screen mode we
        # still use native scrollback during steady state, but on resize we clear
        # the visible screen and replay the retained transcript at the new width.
        self._output_window = None

        # Queue window: shown whenever the agent has unconsumed messages
        # in its user_messages input queue. Mirrors the pre-rewrite
        # ``│ foo`` visual. Reads the hidden Channel
        # (``_user_messages_in``) — the public ``user_messages`` is the
        # LLM-facing OutputQueue facade and doesn't expose snapshot.
        def _queue_pending() -> list[str]:
            if self.agent is None:
                return []
            q = getattr(self.agent, "_user_messages_in", None)
            if q is None:
                return []
            return q.snapshot()

        def _queue_formatted():
            rows = []
            command_queue = list(self._command_queue_texts)
            if command_queue:
                noun = "command" if len(command_queue) == 1 else "commands"
                rows.append(f"│ {len(command_queue)} {noun} queued")
                for index, text in enumerate(command_queue):
                    branch = "└─" if index == len(command_queue) - 1 else "├─"
                    rows.append(f"{branch} {text}")
            for text in _queue_pending():
                for line in str(text).split("\n"):
                    rows.append(f"│ {line}")
            if not rows:
                return []
            return [("class:queue", "\n".join(rows))]

        queue_window = ConditionalContainer(
            Window(
                FormattedTextControl(_queue_formatted, focusable=False),
                dont_extend_height=True,
            ),
            filter=Condition(lambda: bool(_queue_pending()) or bool(self._command_queue_texts)),
        )

        input_style = "class:input-area"
        input_window = Window(
            BufferControl(
                self.input_buffer,
                input_processors=[self._prompt_processor],
            ),
            wrap_lines=True,
            height=Dimension(min=1),
            dont_extend_height=True,
            style=input_style,
        )
        self._input_window = input_window
        self._input_container = HSplit(
            [
                Window(height=1, char=" ", style=input_style),
                input_window,
                Window(height=1, char=" ", style=input_style),
            ],
            style=input_style,
        )

        # Status line at the bottom — shows spinner + session label.
        def _status_formatted():
            text = self.status_text()
            return [("class:status", text)] if text else []

        def _status_height() -> Dimension:
            lines = self.status_text().splitlines()
            height = max(1, len(lines))
            return Dimension(min=height, max=height, preferred=height)

        status_window = Window(
            FormattedTextControl(_status_formatted, focusable=False),
            height=_status_height,
        )

        # Session rule: right above the input, always visible. Shows the
        # session name + short uuid + context-usage label, right-aligned
        # on a horizontal rule. Built from formatted text (not a Rich
        # Rule) so it re-measures with the live terminal width.
        def _session_rule_formatted():
            label = self._session_label_fn() if self._session_label_fn is not None else ""
            return format_session_rule(terminal_cols(minimum=20), label)

        session_rule = Window(
            FormattedTextControl(_session_rule_formatted, focusable=False), height=1
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
            exact = min(max(n, 1), _COMPLETION_MAX)
            # Exact, not a range. Dimension(min=1, max=12) lets HSplit
            # inflate the window when extra space is available — which
            # is exactly what causes growing blank gaps between the
            # prompt and the menu as completions narrow.
            return Dimension(min=exact, max=exact, preferred=exact)

        completions_window = ConditionalContainer(
            Window(
                content=CompletionsMenuControl(),
                width=Dimension(min=8),
                height=_completions_height,
                dont_extend_height=True,
                right_margins=[ScrollbarMargin(display_arrows=True)],
            ),
            filter=Condition(lambda: self.input_buffer.complete_state is not None),
        )

        # Active bottom region (top → bottom):
        #   status (spinner + optional badges)
        #   queued command/type-ahead lines
        #   session rule — always visible while at the transcript tail
        #   input composer (one padding row above and below the input)
        #   completions (only while completing)
        main_container = HSplit(
            [
                status_window,
                queue_window,
                session_rule,
                self._input_container,
                completions_window,
            ],
        )

        def _subview_formatted():
            view = self._active_subview
            if view is None:
                return ANSI("")
            try:
                size = self._app.output.get_size()
                width, height = int(size.columns), int(size.rows)
            except Exception:
                width, height = terminal_cols(minimum=80), 24
            return ANSI(view.render(width, height))

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
            return view is not None and bool(getattr(view, "mouse_support", True))

        self._app = Application(
            layout=Layout(
                DynamicContainer(_root_container),
                focused_element=input_window,
            ),
            key_bindings=kb,
            style=create_prompt_style(),
            full_screen=False,
            before_render=self._before_render,
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
        )

    def refresh_style(self) -> None:
        """Apply the current TUI palette to the live prompt-toolkit app."""
        self._app.style = create_prompt_style()
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

    def _copy_to_clipboard(self, text: str) -> bool:
        """Copy explicit user-requested text through the host or terminal."""
        if not text or len(text.encode("utf-8")) > 100_000:
            return False

        # Native macOS copy is the most reliable path for a locally hosted TUI.
        pbcopy = shutil.which("pbcopy")
        if pbcopy:
            try:
                subprocess.run(
                    [pbcopy],
                    input=text.encode("utf-8"),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                    timeout=2,
                )
                return True
            except (OSError, subprocess.SubprocessError):
                pass

        # OSC 52 asks the user's terminal to set its clipboard, which also
        # works across SSH when the terminal permits clipboard requests.
        try:
            payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
            self._app.output.write_raw(f"\x1b]52;c;{payload}\x07")
            self._app.output.flush()
            return True
        except Exception:
            return False

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

        qm = getattr(self.agent, "queue_manager", None) if self.agent else None
        await self.open_subview(JobExplorerView(qm))

    async def open_todo_explorer(self) -> None:
        """Open the todo explorer as an in-app subview."""
        from .todo_explorer import TodoExplorerView

        todo_mgr = getattr(self.agent, "todo", None) if self.agent else None
        await self.open_subview(TodoExplorerView(todo_mgr))

    async def open_memory_explorer(self) -> None:
        """Open the memory explorer as an in-app subview."""
        from .memory_explorer import (
            MemoryExplorerView,
            build_memory_rows,
            last_reflection_summary,
        )

        agent = self.agent
        memory_skill = getattr(agent, "memory", None) if agent else None
        manager = memory_skill._mgr if memory_skill is not None else None
        if manager is None:
            raise RuntimeError("Memory is not enabled for this agent (see /memory).")

        rows, reflection_line = await self.agent_run_async(
            lambda: (build_memory_rows(agent, manager), last_reflection_summary(manager))
        )

        def _forget(memory_id: str) -> None:
            self.agent_run(lambda: manager.forget(memory_id))

        def _mark_done(memory_id: str) -> None:
            self.agent_run(lambda: manager.update(memory_id, status="done"))

        await self.open_subview(
            MemoryExplorerView(
                rows,
                forget=_forget,
                mark_done=_mark_done,
                last_reflection=reflection_line,
            )
        )

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
            if self._app.is_running:
                self._app.invalidate()

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

        @kb.add("c-c", filter=subview_inactive)
        def _(event):
            # The second C-c in the confirmation window exits through the
            # normal Application path; Session.run() then performs its full
            # snapshot/close/terminal-restoration cleanup in ``finally``.
            if self._ctrl_c_exit_armed:
                self._clear_ctrl_c_exit()
                event.app.exit()
                return

            # The first C-c always clears the composer. While an agent is
            # running it also requests cancellation; at an idle prompt it arms
            # the existing second-C-c-to-exit confirmation.
            event.current_buffer.reset()
            self._history_cursor = None
            if self.request_agent_cancel(source="ctrl-c"):
                self._arm_ctrl_c_exit()
                return
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
                buf.start_completion(select_first=True)
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
            if self.agent is None:
                return None
            q = getattr(self.agent, "_user_messages_in", None)
            if q is None:
                return None
            item = q.pop_last()
            return None if item is None else str(item)

        @kb.add("up", filter=empty_buffer)
        def _(event):
            popped = _pop_last_queued()
            if popped is not None:
                self.input_buffer.text = popped
                self.input_buffer.cursor_position = len(popped)
                return
            self._history_navigate(-1)

        @kb.add("down", filter=empty_buffer)
        def _(event):
            self._history_navigate(+1)

        # Esc: soft-cancel the agent while preserving the queue. Any
        # messages already submitted during the turn are delivered as
        # the next respond() via the done-callback.
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

        mention_base = getattr(self.agent, "cwd", None) if self.agent is not None else None
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
        """Start a background task cycling the spinner frame while the
        agent is thinking. Invalidates the app each tick so the status
        line redraws; exits when ``is_thinking()`` becomes False."""
        if self._spinner_task is not None and not self._spinner_task.done():
            return

        async def _animate() -> None:
            i = 0
            try:
                while self.is_thinking():
                    self._spinner_frame = self._spinner_frames[i % len(self._spinner_frames)]
                    if self._app.is_running:
                        self._app.invalidate()
                    i += 1
                    await asyncio.sleep(0.08)
            finally:
                # Paint once after the agent stops so "thinking…" clears.
                if self._app.is_running:
                    self._app.invalidate()

        self._spinner_task = asyncio.ensure_future(_animate())

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

    async def swap_agent(self, new_agent: Any) -> None:
        """Hot-swap the agent the dispatcher drives (slash-inception).

        The dispatcher loop binds ``agent = self.agent`` once per loop and
        then calls ``agent.handle()`` each turn, so a bare ``self.agent =
        new`` reassignment does NOT redirect a *running* loop. This method
        performs the supported swap: acknowledge-cancel the current dispatcher,
        re-point ``self.agent``, and lazy-restart it bound to the new agent.

        The new agent is expected to already share the OLD agent's live
        channels (``queue_manager`` / ``_user_messages_in`` / readers) and
        ``_render_message`` callback, so input routing + output rendering are
        unchanged. Any messages already queued on the shared
        ``_user_messages_in`` (e.g. an inception seed prompt) are drained by
        the fresh dispatcher and delivered to ``new_agent.handle``.

        Call via ``await app.agent_run_async(lambda: app.swap_agent(new))`` so
        cancellation is acknowledged before the new dispatcher starts.
        """
        self._swapping = True
        old_task = self._agent_task
        old_thread_future = self._agent_thread_future
        await self.cancel_agent_turn(source="swap")

        # In the two-loop TUI, the UI-loop wrapper future's done callback can
        # lag behind the acknowledged agent-loop cancellation. Detach the old
        # wrapper before re-arming so a stale callback cannot block or clobber
        # the dispatcher bound to the new agent.
        if old_task is self._agent_task:
            self._agent_task = None
        if old_thread_future is self._agent_thread_future:
            self._agent_thread_future = None
        self._agent_cancel_requested = False
        self._agent_loop_task = None
        self._swapping = False
        self.agent = new_agent

        # Re-arm a fresh dispatcher bound to the new agent if there's pending
        # input (the inception seed prompt is already queued by the caller).
        q = getattr(new_agent, "_user_messages_in", None)
        if q is not None and q.qsize() > 0:
            self._ensure_dispatcher_task()

    def submit_message(self, user_message: str) -> None:
        """Treat ``user_message`` as if the user just typed and submitted it.

        Pushes the text onto the agent's user_messages Channel and
        lazy-starts the dispatcher task. The user-bar echo + session
        bookkeeping fire later — on the Channel's ``on_get`` hook,
        when the message is actually dequeued — so a message typed
        while the agent is working only shows in the queue pane until
        its turn comes up. The hook fires identically whether the
        dispatcher or agent code pulls the item.

        Consecutive submissions that land while the dispatcher is
        still busy are *merged* into the trailing queue item with a
        ``\\n`` separator. That preserves the old "type, Enter, type,
        Enter → one message" UX where the user is composing a
        multi-line thought across several Enters.

        Public so ``Session`` can submit a message programmatically —
        e.g. a slash command that returns an ``agent_message``
        (``/compact``) and wants to drive the same path as a typed
        message.
        """
        if self.agent is None:
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
        inq = getattr(self.agent, "_user_messages_in", None)
        if inq is None:
            self.emit_block("[submit_message dropped] agent has no user_messages queue\n")
            return
        _coalesce_string_into_queue(inq, user_message)
        self._ensure_dispatcher_task()

    def _ensure_dispatcher_task(self, *, start_with_race: bool = False) -> None:
        """Start the per-turn dispatcher if not already live.

        The dispatcher is the outer loop around ``agent.handle()``:
        it reads the result's ``kind`` and waits on the appropriate
        queue before calling ``handle()`` again with the new
        ``(queue_name, item)`` notification.

        Lazy-started: on session load there's no task. The first user
        message triggers submit_message → put onto user_messages →
        _ensure_dispatcher_task → dispatcher loops until exit or cancel.
        """
        if self.agent is None:
            return
        if self._agent_task is not None and not self._agent_task.done():
            return
        if self._agent_cancel_requested:
            return
        if self._loop is None:
            # Unit tests call submit_message() without running the prompt_toolkit
            # app. In that mode there is no UI loop to protect, so preserve the
            # old same-loop dispatcher semantics.
            self._agent_task = asyncio.ensure_future(
                self._dispatcher_loop(start_with_race=start_with_race)
            )
            self._agent_loop_task = self._agent_task
        else:
            agent_loop = self._ensure_agent_loop()

            async def _run_dispatcher() -> None:
                self._agent_loop_task = asyncio.current_task()
                try:
                    await self._dispatcher_loop(start_with_race=start_with_race)
                finally:
                    self._agent_loop_task = None

            self._agent_thread_future = asyncio.run_coroutine_threadsafe(
                _run_dispatcher(), agent_loop
            )
            self._agent_task = asyncio.wrap_future(self._agent_thread_future)
        self._agent_task.add_done_callback(self._on_agent_done)
        self._ensure_spinner_task()

    def _ensure_agent_loop(self) -> asyncio.AbstractEventLoop:
        """Return the dedicated event loop used for agent dispatch.

        prompt_toolkit stays on the main/UI loop. Agent turns run here so
        synchronous work inside ``agent.handle()`` cannot freeze input or
        spinner redraws.

        A sentinel task keeps the loop alive even when the dispatcher exits
        (gl-212). Without it, run_forever() returns when the last task
        finishes, closing the loop and invalidating BashSession pipes.
        """
        if self._agent_loop is not None and self._agent_loop.is_running():
            return self._agent_loop

        self._agent_loop_ready.clear()
        self._agent_loop_stopped.clear()
        loop = asyncio.new_event_loop()
        self._agent_loop = loop

        def _run() -> None:
            asyncio.set_event_loop(loop)

            def _suppress_litellm_task_destroyed(
                loop_: asyncio.AbstractEventLoop, context: dict
            ) -> None:
                msg = context.get("message", "")
                if "Task was destroyed" in msg:
                    task = context.get("task")
                    if task is not None and "LoggingWorker" in repr(task):
                        return
                loop_.default_exception_handler(context)

            loop.set_exception_handler(_suppress_litellm_task_destroyed)

            # gl-212 sentinel: keeps run_forever() alive across dispatcher restarts.
            async def _sentinel():
                await asyncio.Future()

            self._agent_sentinel = loop.create_task(_sentinel(), name="agent-loop-sentinel")
            self._agent_loop_ready.set()
            try:
                loop.run_forever()
            finally:
                try:
                    pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                    for task in pending:
                        task.cancel()
                    if pending:
                        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    loop.run_until_complete(loop.shutdown_asyncgens())
                finally:
                    loop.close()
                    self._agent_loop_stopped.set()

        self._agent_thread = threading.Thread(
            target=_run,
            name="nemo-tui-agent-loop",
            daemon=True,
        )
        self._agent_thread.start()
        if not self._agent_loop_ready.wait(timeout=5.0):
            raise RuntimeError("Agent event loop thread failed to start within 5s")
        return loop

    async def _stop_agent_loop(self) -> None:
        """Stop the dedicated agent loop without blocking prompt_toolkit."""
        loop = self._agent_loop
        thread = self._agent_thread
        if loop is None:
            return
        if self._agent_thread_future is not None and not self._agent_thread_future.done():
            self._agent_thread_future.cancel()
        # Cancel the sentinel so run_forever() can exit (gl-212).
        # Use call_soon_threadsafe: sentinel lives on the agent loop thread.
        if self._agent_sentinel is not None:
            loop.call_soon_threadsafe(self._agent_sentinel.cancel)
            self._agent_sentinel = None
        # Gracefully stop litellm's global logging worker so its tasks are
        # cancelled before we tear down the loop (prevents "Task was destroyed
        # but it is pending!" warnings from orphaned _worker_loop tasks).
        try:
            fut = asyncio.run_coroutine_threadsafe(_stop_litellm_worker(), loop)
            await asyncio.wait_for(asyncio.wrap_future(fut), timeout=2.0)
        except Exception:
            pass
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread.is_alive():
            await asyncio.to_thread(thread.join, 5.0)
            if thread.is_alive():
                logger.warning("Agent loop thread did not stop within 5s — abandoning")
        self._agent_loop = None
        self._agent_thread = None
        self._agent_thread_future = None

    async def _dispatcher_loop(self, *, start_with_race: bool = False) -> None:
        """Drive ``agent.handle()`` turn-by-turn until DispatcherExit or cancellation.

        The "queued → accepted" echo (user-bar render, SessionUserMessage
        log) is fired by the user_messages channel's ``on_get`` hook —
        installed by ``Session`` — not from this dispatcher loop. That
        way the echo is symmetric: it fires both when the dispatcher
        takes the next user_messages item AND when the agent dequeues
        one mid-turn via ``await self.user_messages.get()``. Calling
        the hook here would double-render the dispatcher case.

        ``QueueManager.race()`` returns ``list[(name, item)]`` —
        currently always length 1, but the list shape is the contract
        so future ``deliver=`` modes can return more items per call
        without changing the dispatcher's loop body.
        """
        from nooa.runtime.channels import Channel

        from .output import StopReasonOutput

        agent = self.agent
        assert agent is not None

        user_messages_in: Channel = agent._user_messages_in
        qm = agent.queue_manager

        if start_with_race:
            notification = await self._next_race_notification(qm)
            if not notification:
                return
        else:
            # Wait for the first user message (already queued by submit_message
            # that started us). qsize()>0 → get() returns immediately.
            item = await user_messages_in.get()
            # The user's on_get hook may synchronously enqueue host-owned
            # instructions (such as the one-shot session-title request). Drain
            # them into this notification so housekeeping shares the normal
            # user turn instead of triggering another LLM call.
            notification = self._drain_pending_queue_items(qm, [("user_messages", item)])
        self._in_respond = True
        self._on_dispatcher_dequeued()

        while True:
            self._in_respond = True
            try:
                # Reflection owns only idle windows. Any channel notification
                # that is about to start a turn interrupts it first.
                runner = self._reflection_runner()
                if runner is not None:
                    await runner.interrupt()
                result = await agent.handle(notification)
            except DispatcherExit:
                return
            finally:
                self._in_respond = False

            result_explanation = getattr(result, "explanation", "")
            logger.info(
                "[DISPATCHER] handle() returned kind=%r explanation=%r",
                result.kind,
                result_explanation,
            )

            reflection_deferred = False
            keep_going_enabled, keep_going_model = self._keep_going_state(agent)
            if keep_going_enabled and str(result.kind) == "DONE":
                if keep_going_model:
                    reflection_deferred = True
                    generation = self._keep_going_generation
                    task = asyncio.create_task(
                        self._run_keep_going_audit(
                            agent,
                            result,
                            keep_going_model,
                            generation,
                        ),
                        name="keep-going-audit",
                    )
                    self._keep_going_tasks.add(task)
                    task.add_done_callback(self._keep_going_tasks.discard)
                    logger.info("[DISPATCHER] keep-going audit scheduled")
                else:
                    logger.warning("keep-going enabled without keep_going_model; audit skipped")
                    await self._emit_structured_output(
                        StopReasonOutput(
                            "KEEP_GOING",
                            "disabled: configure a model with /keep-going model <model-id>",
                        )
                    )

            if not reflection_deferred:
                self._schedule_reflection(agent)

            if result_explanation:
                await self._emit_structured_output(
                    StopReasonOutput(result.kind, result_explanation)
                )

            # Every stop reason uses the same dispatcher wake path: race all
            # declared channels/events and re-enter with the first arrival.
            # ``kind`` explains why the agent stopped; it does not select a
            # different queue primitive.
            running = qm.running_handles()
            if running:
                now = datetime.datetime.now().strftime("%H:%M:%S")
                lines = "".join(f"  ⠿ {h.label}\n" for h in running)
                self.emit_block(
                    f"\x1b[2m{now} waiting — {len(running)} job(s) running:\n{lines}\x1b[0m"
                )

            # Race all channels for the next item(s). Returns
            # [(name, item)] for queue-mode winners or [] for
            # event-triggered wakes (events already in the prompt).
            try:
                items = await qm.race()
            except ValueError:
                return
            # Show which job(s) fired.
            if running and items:
                now = datetime.datetime.now().strftime("%H:%M:%S")
                fired_names = {name for name, _ in items}
                fired = [h for h in running if h.name in fired_names]
                for h in fired:
                    self.emit_block(f"\x1b[32m  ✓ {h.label} — {now}\x1b[0m\n")

            # Drain all pending items across all channels, grouped by name.
            pending = self._drain_pending_queue_items(qm, items)
            if "user_messages" in pending or "slash_commands" in pending:
                self._invalidate_keep_going_audits()
            self._in_respond = True
            self._on_dispatcher_dequeued()
            notification = pending

    async def _emit_structured_output(self, output: Any) -> None:
        """Emit dispatcher-owned structured output through the frontend path.

        Unit tests and minimal harnesses may construct ``TUIApplication``
        without a frontend callback; in that case keep the terminal fallback so
        existing same-loop dispatcher tests can inspect ``emit_block`` output.
        """
        if self._on_output is not None:
            result = self._on_output(output)
            if result is not None:
                await result
            return
        display_text = getattr(output, "display_text", None)
        if callable(display_text):
            self.emit_block(f"\x1b[2m{display_text()}\x1b[0m\n")

    def _keep_going_state(self, agent: Any) -> tuple[bool, str | None]:
        vars_obj = getattr(agent, "vars", None)
        if vars_obj is not None and "tui_keep_going" in vars_obj:
            enabled = bool(vars_obj.get("tui_keep_going"))
        else:
            enabled = bool(
                self._config is not None
                and getattr(getattr(self._config, "tui", None), "keep_going", False)
            )

        if vars_obj is not None and "tui_keep_going_model" in vars_obj:
            value = vars_obj.get("tui_keep_going_model")
        else:
            value = getattr(getattr(self._config, "tui", None), "keep_going_model", None)
        model = str(value).strip() if value is not None else ""
        if not model or model.lower() == "none":
            model = ""
        return enabled, model or None

    def _invalidate_keep_going_audits(self) -> None:
        """Make in-flight audits stale and cancel them on their owning loop."""
        self._keep_going_generation += 1
        loop = self._agent_loop
        for task in list(self._keep_going_tasks):
            try:
                task_loop = task.get_loop()
            except RuntimeError:
                task_loop = None
            if loop is not None and loop.is_running() and task_loop is loop:
                loop.call_soon_threadsafe(task.cancel)
            else:
                task.cancel()

    async def _run_keep_going_audit(
        self, agent: Any, result: Any, model: str, generation: int
    ) -> None:
        """Audit DONE in the background and queue a current continuation."""
        from .output import StopReasonOutput

        try:
            from .keep_going import build_keep_going_prompt

            keep_going_prompt = await build_keep_going_prompt(agent, result, model=model)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if generation != self._keep_going_generation:
                return
            message = _short_exception_message(exc)
            logger.warning("keep-going audit failed: %s", message)
            await self._emit_structured_output(
                StopReasonOutput(
                    "KEEP_GOING",
                    f"judge failed: {message}; use /keep-going off or /keep-going model <model-id>",
                )
            )
            self._schedule_reflection(agent)
            return
        if generation != self._keep_going_generation:
            return
        if not keep_going_prompt:
            self._schedule_reflection(agent)
            return
        enabled, current_model = self._keep_going_state(agent)
        if not enabled or current_model != model:
            self._schedule_reflection(agent)
            return

        prompt_text = getattr(keep_going_prompt, "prompt", keep_going_prompt)
        display_reason = getattr(
            keep_going_prompt,
            "display_reason",
            "continuing unfinished work",
        )
        if display_reason:
            await self._emit_structured_output(StopReasonOutput("KEEP_GOING", str(display_reason)))

        system_messages_in = getattr(agent, "_system_messages_in", None)
        if system_messages_in is not None:
            system_messages_in.put(str(prompt_text))
            logger.info("[DISPATCHER] keep-going queued system continuation prompt")
            return

        user_messages_in = getattr(agent, "_user_messages_in", None)
        if user_messages_in is not None:
            user_messages_in.put(str(prompt_text))
            logger.info("[DISPATCHER] keep-going queued fallback user continuation prompt")
            return
        self._schedule_reflection(agent)

    def _reflection_runner(self) -> Any | None:
        """Return the configured runner and wire its thread-safe repaint hook."""
        agent = self.agent
        runner = getattr(agent, "_tui_reflection_runner", None) if agent is not None else None
        if runner is not None:
            runner.invalidate = self._invalidate_for_reflection
        return runner

    def _schedule_reflection(self, agent: Any) -> None:
        runner = getattr(agent, "_tui_reflection_runner", None)
        if runner is not None:
            runner.invalidate = self._invalidate_for_reflection
            runner.on_response_done()

    def _invalidate_for_reflection(self) -> None:
        loop = self._loop

        def _invalidate() -> None:
            if self._app.is_running:
                self._app.invalidate()

        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(_invalidate)
        else:
            _invalidate()

    async def interrupt_reflection(self, *, teardown: bool = False) -> None:
        """Stop idle reflection on its owning loop, optionally detaching it."""
        runner = self._reflection_runner()
        if runner is None:
            return

        async def _stop() -> None:
            await runner.interrupt()
            if teardown:
                runner.teardown()

        await self.agent_run_async(_stop)

    async def _next_race_notification(self, qm: Any) -> dict[str, list]:
        """Wait for QueueManager output and drain all pending queue items."""
        try:
            items = await qm.race()
        except ValueError:
            return {}
        return self._drain_pending_queue_items(qm, items)

    def _on_queue_notify(self) -> None:
        """QueueManager notify callback: (re)start the dispatcher on any put.

        Registered via ``QueueManager.set_notify_callback``. May fire from
        the agent-loop thread (spawned jobs), so marshal onto the UI loop
        before touching dispatcher state to avoid attaching it to the wrong
        event loop.

        Fires for every channel put (user/system messages included), but
        ``_ensure_dispatcher_task`` is idempotent — it early-returns when a
        dispatcher is already running — so an extra call when a turn is
        already in flight is a harmless no-op.
        """
        ui_loop = self._loop
        if ui_loop is not None and ui_loop.is_running():
            ui_loop.call_soon_threadsafe(lambda: self._ensure_dispatcher_task(start_with_race=True))
        else:
            self._ensure_dispatcher_task(start_with_race=True)

    def _drain_pending_queue_items(self, qm: Any, items: list[tuple[str, Any]]) -> dict[str, list]:
        """Group race winners and already-buffered queue items by channel name."""
        pending: dict[str, list] = {}
        for name, value in items:
            pending.setdefault(name, []).append(value)
        for ch in qm.channels().values():
            if ch.mode == "queue":
                for val in ch.drain():
                    pending.setdefault(ch.name, []).append(val)
        return pending

    def _restart_dispatcher_if_pending(self) -> None:
        """Restart the dispatcher if user messages or non-user work is pending."""
        q = getattr(self.agent, "_user_messages_in", None) if self.agent else None
        if q is not None and q.qsize() > 0:
            self._ensure_dispatcher_task()
        elif self._has_pending_or_running_non_user_work():
            self._ensure_dispatcher_task(start_with_race=True)

    def _has_pending_or_running_non_user_work(self) -> bool:
        """True when post-cancel background channel work should wake the agent."""
        agent = self.agent
        qm = getattr(agent, "queue_manager", None) if agent is not None else None
        if qm is None:
            return False
        if qm.running_handles():
            return True
        for name, ch in qm.channels().items():
            if name == "user_messages":
                continue
            if ch.mode == "queue" and not ch.is_empty():
                return True
        return False

    def request_agent_cancel(self, *, source: str = "escape") -> bool:
        """Request cancellation of the current agent turn.

        Esc/Ctrl-C cancel the active agent turn only. They deliberately do
        not cancel ``QueueManager.spawn`` jobs: those are background work and
        remain controlled by ``queue_manager.cancel(...)``/``shutdown()``.

        The TUI runs agent turns on a dedicated event loop in production. Do
        not cancel the UI-loop ``wrap_future`` proxy directly; that proxy can
        look done before the real agent-loop task has observed cancellation
        and finished cleanup. Cancelling the source task keeps the status in
        ``cancelling...`` until the done callback receives acknowledgement.
        """
        task = self._agent_task
        if task is None or task.done():
            return False
        if self._agent_cancel_requested:
            return source != "ctrl-c"
        if not self._in_respond and source not in {"swap", "session"}:
            return False

        self._agent_cancel_requested = True
        self._invalidate_keep_going_audits()
        if self._app.is_running:
            self._app.invalidate()
        self._ensure_spinner_task()

        loop_task = self._agent_loop_task
        if loop_task is not None:
            try:
                target_loop = loop_task.get_loop()
            except RuntimeError:
                target_loop = None
            if target_loop is not None and target_loop.is_running():

                def _cancel_loop_task() -> None:
                    if not loop_task.done():
                        loop_task.cancel()

                target_loop.call_soon_threadsafe(_cancel_loop_task)
            else:
                loop_task.cancel()
        elif self._agent_thread_future is not None and not self._agent_thread_future.done():
            # The dispatcher coroutine may be running before its task pointer is
            # published back to the UI thread. Cancel the source future rather
            # than the UI wrapper so cancellation is still delivered to the
            # agent loop and acknowledged through _on_agent_done.
            self._agent_thread_future.cancel()
        else:
            task.cancel()
        return True

    def _on_input_text_changed(self, _buffer: Buffer) -> None:
        """Typing after an exit warning cancels the double-Ctrl-C gesture."""
        self._clear_ctrl_c_exit()

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

    async def cancel_agent_turn(self, *, source: str = "escape") -> bool:
        """Request turn cancellation and await the dispatcher acknowledgement."""
        task = self._agent_task
        if task is None or task.done():
            return False
        if not self.request_agent_cancel(source=source):
            return False
        try:
            current_loop = asyncio.get_running_loop()
            current_task = asyncio.current_task()
            task_loop = task.get_loop() if hasattr(task, "get_loop") else None
            loop_task = self._agent_loop_task
            loop_task_loop = loop_task.get_loop() if loop_task is not None else None

            if task_loop is current_loop and task is not current_task:
                await task
            elif (
                loop_task is not None
                and loop_task is not current_task
                and loop_task_loop is current_loop
            ):
                await loop_task
            elif self._agent_thread_future is not None:
                await asyncio.wrap_future(self._agent_thread_future)
            elif task is not current_task:
                while not task.done():
                    await asyncio.sleep(0)
        except asyncio.CancelledError:
            pass
        return True

    async def shutdown_agent_queue_manager(
        self, *, agent: Any | None = None, flush: bool = False
    ) -> None:
        """Shutdown an agent's QueueManager on the loop that owns spawned jobs."""
        target = self.agent if agent is None else agent
        qm = getattr(target, "queue_manager", None) if target is not None else None
        if qm is None:
            return

        async def _shutdown_and_flush() -> None:
            await qm.shutdown()
            if flush:
                for name in qm.names():
                    ch = qm.get_channel(name)
                    if ch.mode == "queue":
                        ch.flush()

        loop = self._agent_loop
        if loop is not None and loop.is_running():
            await self.agent_run_async(_shutdown_and_flush)
        else:
            await _shutdown_and_flush()

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

    def _on_agent_done(self, task: asyncio.Task) -> None:
        """Fired when the dispatcher exits (STOP, error, or cancellation).

        On cancellation OR exception with messages still pending we
        lazy-restart so neither soft-cancel (Esc) nor a crashed turn
        strands queued input — the user can keep typing and the next
        message wakes the dispatcher again. STOP exits cleanly.
        """
        if task is not self._agent_task:
            return
        if task.cancelled():
            self._agent_cancel_requested = False
            self._agent_loop_task = None
            was_swapping = getattr(self, "_swapping", False)
            was_session_transition = getattr(self, "_session_transitioning", False)
            suppress_cancel_restart = was_swapping or was_session_transition
            if was_swapping:
                self._swapping = False
            if not suppress_cancel_restart:
                self.emit_block("\x1b[33m✗ Interrupted agent turn.\x1b[0m\n")
            if self._app.is_running:
                self._app.invalidate()
            q = getattr(self.agent, "_user_messages_in", None) if self.agent else None
            if not suppress_cancel_restart:
                if q is not None and q.qsize() > 0:
                    self._ensure_dispatcher_task()
                elif self._has_pending_or_running_non_user_work():
                    self._ensure_dispatcher_task(start_with_race=True)
            return
        self._agent_cancel_requested = False
        self._agent_loop_task = None
        exc = task.exception()
        if exc is not None:
            self.emit_block(f"Agent error: {exc}\n")
        self._restart_dispatcher_if_pending()

    # ── agent_run: dispatch to agent thread ----------------------------

    def agent_run(self, fn):
        """Run *fn* on the agent-loop thread, blocking the caller until done.

        Use for any operation that mutates agent state from outside the
        dispatcher loop (slash commands, shutdown, renderer attach/detach).
        If no agent loop is running (unit tests, pre-startup), executes
        inline on the caller's thread.

        *fn* is a zero-arg callable. If it returns a coroutine, the
        coroutine is awaited on the agent loop. Otherwise the callable
        itself is scheduled on the agent loop via ``call_soon_threadsafe``.

        Returns whatever *fn* (or the coroutine) returns. Propagates
        exceptions. Times out after 30 seconds.
        """
        loop = self._agent_loop
        if loop is None or not loop.is_running():
            # No agent loop (tests or pre-startup) — run inline.
            result = fn()
            if asyncio.iscoroutine(result):
                raise TypeError("agent_run() with a coroutine requires a running agent loop")
            return result

        # Guard: if already on the agent loop, run inline to avoid deadlock.
        # (agent_run blocks the caller waiting for the loop to execute the
        # wrapper — if the caller IS the loop, that's a guaranteed hang.)
        import threading

        if getattr(loop, "_thread_id", None) == threading.current_thread().ident:
            result = fn()
            if asyncio.iscoroutine(result):
                raise TypeError(
                    "agent_run() called from agent loop with a coroutine — "
                    "use 'await' directly instead"
                )
            return result

        # Schedule on agent loop. Use an async wrapper so both sync and
        # async callables go through run_coroutine_threadsafe uniformly.
        async def _wrapper():
            result = fn()
            if asyncio.iscoroutine(result):
                return await result
            return result

        future = asyncio.run_coroutine_threadsafe(_wrapper(), loop)
        try:
            return future.result(timeout=30)
        except TimeoutError:
            future.cancel()
            raise

    async def agent_run_async(self, fn):
        """Like ``agent_run``, but awaitable — never blocks the calling loop.

        Slash commands run as tasks on the prompt_toolkit UI loop. Calling the
        blocking ``agent_run`` from there freezes the UI loop (``future.result``)
        while the agent loop is busy — starving the block-queue consumer
        (``self.message()`` output stops appearing) and wedging prompt_toolkit's
        redraw (status-bar / input flicker). Awaiting the cross-loop future
        instead yields control so the UI keeps painting and draining output.

        Falls back to inline execution when there is no separate agent loop
        (tests / pre-startup) or when already on the agent loop.
        """
        loop = self._agent_loop
        if loop is None or not loop.is_running():
            result = fn()
            if asyncio.iscoroutine(result):
                return await result
            return result

        import threading

        if getattr(loop, "_thread_id", None) == threading.current_thread().ident:
            result = fn()
            if asyncio.iscoroutine(result):
                return await result
            return result

        async def _wrapper():
            result = fn()
            if asyncio.iscoroutine(result):
                return await result
            return result

        future = asyncio.run_coroutine_threadsafe(_wrapper(), loop)
        try:
            return await asyncio.wait_for(asyncio.wrap_future(future), timeout=30)
        except TimeoutError:
            future.cancel()
            raise

    # ── output pipeline -----------------------------------------------

    def clear_transcript(self) -> None:
        """Clear live transcript buffers and fullscreen resize replay retention."""
        self._output_ansi.clear()
        self._replay_blocks.clear()
        self.output_buffer.set_document(Document(""), bypass_readonly=True)

    def _on_stray_output(self, content: str, disposition: str) -> None:
        """Record stray stdout/stderr intercepts as hidden runtime events."""
        agent = self.agent
        if agent is None:
            return
        em = getattr(agent, "event_manager", None)
        if em is None:
            return
        try:
            from nooa.events import DebugTrace

            em.add(DebugTrace(content=f"[stray:{disposition}] {content[:200]}"))
        except Exception:
            pass

    def emit_block(
        self,
        text: str,
        replay: Callable[[], str] | None = None,
        *,
        event_id: str | None = None,
        tags: set[str] | frozenset[str] | None = None,
        keep: bool = False,
    ) -> None:
        """Enqueue one ANSI-bearing block for the transcript.

        This is the ONE public contract for writing to the transcript:
        all producers (activity lines, code cells, agent markdown,
        interrupt notices, user echo) call this. A single consumer
        task drains the queue and writes each block in FIFO order via
        ``run_in_terminal`` → ``sys.__stdout__``. No races.

        Thread-safe: any mutation of prompt_toolkit state (``Buffer``
        document, which fires callbacks synchronously) is routed through
        ``call_soon_threadsafe`` so it always runs on the loop thread.
        ``list.append`` on ``_output_ansi`` is already GIL-safe.
        """
        if not text:
            return

        # GIL-safe: list.append is atomic. Reading from off-thread is fine.
        self._output_ansi.append(text)
        self._replay_blocks.append(
            ReplayBlock(
                raw=text,
                replay=replay,
                event_id=str(event_id) if event_id is not None else None,
                tags=frozenset(str(t) for t in (tags or ())),
                keep=keep,
            )
        )

        # Before the consumer is up (pre-run_async) we're single-threaded
        # by construction — safe to touch the buffer directly + emit to
        # stdout. After run_async, route everything via the loop.
        if self._block_queue is None or self._loop is None:
            self._append_stripped_to_buffer(text)
            import sys as _sys

            try:
                _sys.stdout.write(text)
                _sys.stdout.flush()
            except Exception:
                pass
            return

        # On-thread fast path: mutate buffer directly so tests that
        # inspect ``output_buffer.text`` right after a call see the
        # update without waiting for a loop tick.
        try:
            on_thread = asyncio.get_running_loop() is self._loop
        except RuntimeError:
            on_thread = False
        if on_thread:
            self._append_stripped_to_buffer(text)
            self._block_queue.put_nowait(text)
            return

        # Off-thread: route everything through the loop so off-thread
        # producers (e.g. background workers, OTEL exporter) never touch
        # prompt_toolkit state directly.
        self._loop.call_soon_threadsafe(self._append_stripped_to_buffer, text)
        self._loop.call_soon_threadsafe(self._block_queue.put_nowait, text)

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

    async def run_async(self) -> None:
        # Capture the loop once so emit_block can enqueue safely from
        # any thread without calling the deprecated get_event_loop().
        self._loop = asyncio.get_running_loop()
        self._block_queue = asyncio.Queue()
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
        try:
            # set_exception_handler=False keeps the handler Session installed
            # (_loud_handler) active for the whole app lifetime. Otherwise
            # prompt_toolkit replaces it with its own, which prints "Exception
            # None\nPress ENTER to continue..." for non-exception asyncio
            # contexts (e.g. "Task was destroyed but it is pending!") and
            # swallows every other diagnostic field.
            await self._app.run_async(set_exception_handler=False)
        finally:
            self._clear_ctrl_c_exit()
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

            try:
                await self.interrupt_reflection(teardown=True)
            except Exception:
                logger.debug("reflection shutdown during TUI teardown failed", exc_info=True)

            try:
                await self.shutdown_agent_queue_manager()
            except Exception:
                logger.debug("queue manager shutdown during TUI teardown failed", exc_info=True)

            # Drain any blocks queued during teardown (e.g. 'Goodbye!
            # Stay vibing.' from /exit) straight to the real stdout —
            # the consumer task's run_in_terminal no longer works once
            # the app has exited.
            import sys as _sys

            q = self._block_queue
            if q is not None:
                while not q.empty():
                    try:
                        chunk = q.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    out = _sys.__stdout__
                    if out is not None:
                        try:
                            out.write(chunk)
                            out.flush()
                        except Exception:
                            pass
            if self._consumer_task is not None:
                self._consumer_task.cancel()
                try:
                    await self._consumer_task
                except asyncio.CancelledError:
                    pass
                except BaseException:
                    pass
            if self._spinner_task is not None and not self._spinner_task.done():
                self._spinner_task.cancel()
                # Await so the spinner's finally block runs (invalidate())
                # and asyncio doesn't emit "Task was destroyed" on loop
                # close. CancelledError on a cancelled task is expected.
                try:
                    await self._spinner_task
                except (asyncio.CancelledError, BaseException):
                    pass
            await self._stop_agent_loop()
            self._consumer_task = None
            self._spinner_task = None
            self._block_queue = None
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
            text = await self._block_queue.get()

            def _write(t: str = text) -> None:
                out = _sys.__stdout__
                if out is not None:
                    out.write(t)
                    out.flush()

            try:
                await run_in_terminal(_write)
            except asyncio.CancelledError:
                raise
            except Exception:
                # Best-effort — a single failed write shouldn't wedge
                # the consumer. Fall through and pick up the next block.
                continue

    def output_columns(self, minimum: int = 20) -> int:
        """Current transcript render width, including forced resize replays."""
        if self._replay_columns_override is not None:
            return max(int(self._replay_columns_override), minimum)
        try:
            return max(int(self._app.output.get_size().columns), minimum)
        except Exception:
            return terminal_cols(minimum=minimum)

    def _before_render(self, _app) -> None:
        """Replay retained transcript when fullscreen mode sees a terminal resize."""
        if not self.full_screen:
            return
        try:
            size = self._app.output.get_size()
            current = (int(size.columns), int(size.rows))
        except Exception:
            return
        if self._last_rendered_size is None:
            self._last_rendered_size = current
            return
        if current != self._last_rendered_size:
            self._last_rendered_size = current
            if self._active_subview is not None:
                return
            self._replay_fullscreen_transcript()

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
        em = getattr(self.agent, "event_manager", None)
        if em is None or not hasattr(em, "items"):
            return set(), []
        active_ids: set[str] = set()
        ranges: list[tuple[int, int]] = []
        try:
            items = list(em.items())
        except Exception:
            return active_ids, ranges
        for tag, event in items:
            rng = self._tag_range(str(tag))
            if rng is not None:
                ranges.append(rng)
            event_id = getattr(event, "id", None)
            if event_id is not None:
                active_ids.add(str(event_id))
        return active_ids, ranges

    def _tag_is_active(self, tag: str, active_ranges: list[tuple[int, int]]) -> bool:
        rng = self._tag_range(tag)
        if rng is None:
            return False
        start, end = rng
        return any(
            start <= active_end and end >= active_start
            for active_start, active_end in active_ranges
        )

    def _prune_replay_blocks_for_active_events(self) -> None:
        if not self._replay_blocks:
            return
        active_ids, active_ranges = self._active_replay_identity()
        if not active_ids and not active_ranges:
            return
        keep_indexes: set[int] = set()
        untagged_indexes: list[int] = []
        for i, block in enumerate(self._replay_blocks):
            if block.keep:
                keep_indexes.add(i)
            elif block.event_id is not None:
                if block.event_id in active_ids:
                    keep_indexes.add(i)
            elif block.tags:
                if any(self._tag_is_active(tag, active_ranges) for tag in block.tags):
                    keep_indexes.add(i)
            else:
                untagged_indexes.append(i)
        keep_indexes.update(untagged_indexes[-self._untagged_replay_tail :])
        self._replay_blocks = [
            block for i, block in enumerate(self._replay_blocks) if i in keep_indexes
        ]

    def _replay_fullscreen_transcript(self) -> None:
        if not self.full_screen or not self._replay_blocks:
            return
        import sys as _sys

        out = _sys.__stdout__
        if out is None:
            return
        self._prune_replay_blocks_for_active_events()
        chunks: list[str] = []
        for block in self._replay_blocks:
            if block.replay is None:
                chunks.append(block.raw)
                continue
            try:
                chunks.append(block.replay())
            except Exception:
                chunks.append(block.raw)
        try:
            out.write("\x1b[H\x1b[2J\x1b[3J")
            out.write("".join(chunks))
            out.flush()
            self._fullscreen_invalidate_count += 1
        except Exception:
            return

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
        if self._agent_cancel_requested:
            return True
        if self._agent_task is None or self._agent_task.done():
            return False
        return self._in_respond

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

    def status_text(self) -> str:
        """One-line status area text.

        Shows ``<spinner> thinking...`` while the agent is working,
        ``<spinner> cancelling agent turn...`` while cancellation is awaiting
        acknowledgement, and a bracketed session label when one is set. Example::

            ⠋ thinking...    [session-abc]

        """
        lines: list[str] = []
        if self._agent_cancel_requested:
            lines.append(f"{self._spinner_frame} cancelling agent turn...")
        elif self.is_thinking():
            lines.append(f"{self._spinner_frame} thinking...")
        reflection_frame = ""
        runner = self._reflection_runner()
        if runner is not None:
            reflection_frame = runner.indicator_frame()
        if reflection_frame:
            lines.append(reflection_frame)
        if self._command_status_text:
            lines.append(self._command_status_text)
        if self._exit_hint_text:
            lines.append(self._exit_hint_text)
        if self._session_label:
            if lines:
                lines[-1] = f"{lines[-1]}   [{self._session_label}]"
            else:
                lines.append(f"[{self._session_label}]")
        return "\n\n".join(lines)

    def set_session_label(self, label: str) -> None:
        """Set the bracketed label shown on the right of the status line."""
        self._session_label = label

    def handle_resize(self, cols: int, rows: int) -> None:
        """Hint the app to re-layout for a new terminal size.

        prompt_toolkit handles real resize events internally; this hook
        exists for tests (and callers that want to force a redraw after
        a non-SIGWINCH layout change).
        """
        if self.full_screen:
            size = (int(cols), int(rows))
            if self._last_rendered_size != size:
                self._last_rendered_size = size
                if self._active_subview is None:
                    self._replay_columns_override = int(cols)
                    try:
                        self._replay_fullscreen_transcript()
                    finally:
                        self._replay_columns_override = None
        if self._app.is_running:
            self._app.invalidate()
