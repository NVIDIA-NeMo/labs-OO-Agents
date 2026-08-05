# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session — the REPL loop that glues a Frontend to an Agent.

``Session`` is frontend-agnostic: it reads input via ``frontend.get_input()``,
routes commands through ``CommandHandler``, and renders every output through
``frontend.render()``.  ``TerminalFrontend`` is the concrete frontend
implementation.

All *behavior* lives here — event subscription, streaming state, show_python
decisions.  Frontends are pure rendering.
"""

import asyncio
import inspect
import io
import logging
import re
import shlex
import sys
import traceback
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console as RichConsole
from rich.text import Text

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from nooa import Agent
    from nooa.tools.shell_tools import ShellTools

    from .agent_event_renderer import AgentEventRenderer
    from .commands import CommandRegistry
    from .config import Config
    from .frontend import Frontend
    from .session_manager import SessionManager
    from .tui_application import TUIApplication


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hex_to_ansi256(hex_color: str) -> int:
    """Convert a ``#rrggbb`` hex string to the nearest xterm-256 index.

    Used when we render ANSI directly (e.g. the user-message bar) and
    can't rely on Rich's width/wrap logic to emit correctly-padded
    terminal output.
    """
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

    # 6x6x6 color cube starting at index 16.
    def _q(v: int) -> int:
        # 0,95,135,175,215,255 — standard xterm cube steps.
        if v < 48:
            return 0
        if v < 115:
            return 1
        return (v - 35) // 40

    return 16 + 36 * _q(r) + 6 * _q(g) + _q(b)


def _effective_slash_output_to_agent(agent: Any, slash_result: Any) -> bool:
    """Return fresh output routing for a slash result.

    The command registry can be stale across hot reloads or long-lived TUI
    processes. Prefer the currently attached Skill metadata when it can be
    found; fall back to the SlashCommandResult flag for built-in/text commands.
    """
    command = str(getattr(slash_result, "command", "")).lower()
    if command:
        try:
            from nooa.skill import Skill, get_slash_commands

            for attr_name in dir(agent):
                if attr_name.startswith("_"):
                    continue
                try:
                    obj = getattr(agent, attr_name)
                except Exception:
                    continue
                if not isinstance(obj, Skill):
                    continue
                for meta, _method in get_slash_commands(obj):
                    if meta.name.lower() == command:
                        return bool(getattr(meta, "output_to_agent", True))
        except Exception:
            logger.debug("Failed to resolve fresh slash command metadata", exc_info=True)
    return bool(getattr(slash_result, "output_to_agent", True))


def _build_user_bar(text: str, app: "TUIApplication", colors: dict) -> str:
    """Build a full-width highlighted user-message bar as raw ANSI.

    Bypasses Rich because reconciling Rich's wrap/crop/overflow logic
    with manual ``ljust`` padding is brittle across Rich versions and
    terminal emulators — direct CSI emission always renders the full
    width-spanning highlighted row the spec asks for.

    Each input line becomes one bar row:
      ``ESC[fg;bg m{prefix}{line}{padding}{ ESC[0m}\\n``
    where the first row carries the ``❯`` prompt glyph and
    continuation rows start flush-left.
    """
    # Prefer prompt_toolkit's live width — ``run_in_terminal`` will use
    # this number when writing above the prompt. Falls back to the
    # terminal_cols helper if the app output can't report.
    cols: int
    try:
        cols = app.output_columns(minimum=20)
    except Exception:
        try:
            cols = app._app.output.get_size().columns  # type: ignore[attr-defined]
        except Exception:
            from .tui_application import terminal_cols

            cols = terminal_cols(minimum=40)
    cols = max(cols, 20)

    fg = _hex_to_ansi256(colors["text"])
    bg = _hex_to_ansi256(colors["surface2"])
    on = f"\x1b[38;5;{fg};48;5;{bg}m"
    off = "\x1b[0m"

    rows: list[str] = []
    for i, line in enumerate(text.split("\n")):
        shown = f" ❯ {line} " if i == 0 else f" {line} "
        # Clamp to cols so an overlong line becomes multiple bar rows
        # rather than wrapping chaotically at the terminal's edge.
        while len(shown) > cols:
            rows.append(f"{on}{shown[:cols]}{off}")
            shown = shown[cols:]
        rows.append(f"{on}{shown.ljust(cols)}{off}")
    return "\n".join(rows) + "\n"


class _EmitStream:
    """A ``Console.file`` target that batches writes into one ``emit_block``
    per ``flush()`` — or one per ``hold()`` span.

    Rich's ``Console.print`` flushes at the end; without buffering each
    stylised chunk (many per print call) would enqueue a separate block
    and pay the ``run_in_terminal`` hop. Batching collapses them into
    one atomic scrollback block.

    A multi-output command (e.g. ``/activity`` → table + code block) calls
    ``frontend.render()`` once per output, each ending in a ``flush()``. That
    is several ``emit_block``s → several ``run_in_terminal`` hops, and the
    thinking-spinner's ``invalidate()`` (~12/s) repaints the prompt region
    between them — visible flicker. ``hold()`` defers flushing so the whole
    command renders as ONE block / ONE hop.
    """

    def __init__(
        self,
        emit: Callable[..., None],
        replay_width: Callable[[], int] | None = None,
        clear: Callable[[], None] | None = None,
    ) -> None:
        self.supports_semantic_replay = True
        self._emit = emit
        self._replay_width = replay_width
        self._clear = clear
        self._buf: list[str] = []
        self._held = 0

    def write(self, text: str) -> int:
        if text:
            self._buf.append(text)
        return len(text)

    def flush(self) -> None:
        if self._held:
            # Defer until the hold span releases — keep the buffer intact so
            # subsequent renders append to the same block.
            return
        if not self._buf:
            return
        chunk = "".join(self._buf)
        self._buf.clear()
        if chunk:
            self._emit(chunk)

    def clear_transcript(self) -> None:
        self._buf.clear()
        if self._clear is not None:
            self._clear()

    def replay_width(self, default: int = 80) -> int:
        if self._replay_width is None:
            return default
        try:
            return int(self._replay_width())
        except Exception:
            return default

    def emit_with_replay(self, text: str, replay: Callable[[], str]) -> None:
        if self._buf:
            chunk = "".join(self._buf)
            self._buf.clear()
            if chunk:
                self._emit(chunk)
        if text:
            self._emit(text, replay=replay)

    @contextmanager
    def hold(self):
        """Defer ``flush()`` for the duration of the span, emitting once on exit.

        Re-entrant: nested holds only release on the outermost exit.
        """
        self._held += 1
        try:
            yield
        finally:
            self._held -= 1
            if self._held == 0:
                self.flush()

    def isatty(self) -> bool:
        return True


def _short_model_name(full_name: str) -> str:
    """Convert a full model registry key to a short display name.

    Examples:
        "claude-sonnet-4-5" → "sonnet-4-5"
        "gpt-4o" → "gpt-4o"
    """
    part = full_name.split("/")[-1]
    part = part.replace("bedrock-claude-", "").replace("bedrock-", "").replace("claude-", "")
    part = re.sub(r"-v\d+$", "", part)
    return part


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class Session:
    """Frontend-agnostic REPL loop.

    Owns all behavior: agent event subscription, streaming state, show_python
    decisions.  The frontend is a pure rendering surface.

    Args:
        frontend: Any object implementing the ``Frontend`` protocol.
        agent: A NOOA agent with a ``handle(notification)`` async method.
        config: Loaded ``Config`` (holds tui / agent sub-configs).
        registry: Pre-built ``CommandRegistry``.
        session_manager: Optional ``SessionManager`` for persisting turns.
    """

    def __init__(
        self,
        frontend: "Frontend",
        agent: "Agent",
        config: "Config",
        registry: "CommandRegistry",
        session_manager: "SessionManager | None" = None,
        initial_outputs: list[Any] | None = None,
    ) -> None:
        from .commands import CommandHandler

        self.frontend = frontend
        self.agent = agent
        self.config = config
        self.registry = registry
        self._handler = CommandHandler(registry=registry, frontend=frontend)
        self._session_manager = session_manager
        from .toolbar import ToolbarRegistry

        self._toolbar = ToolbarRegistry()
        self._initial_outputs = list(initial_outputs or [])
        self._first_message: str | None = None  # first user turn (for auto-naming)

        # Streaming state shared with the AgentEventRenderer: the
        # tool_call_id → code map that pairs a preview with its matching
        # ``PythonOutput`` event.
        self._pending_code: dict[str, str] = {}
        self._background_tasks: set[asyncio.Task] = set()  # fire-and-forget tasks
        self._naming_futures: set = set()  # concurrent.futures.Future from agent-loop dispatch
        self._command_runner = None

        # Populated at the start of ``run()``; referenced by the handler
        # methods (``_on_command``, ``_on_user_message_ui``, ``_loud_handler``,
        # etc.) so they can live as real methods instead of 240 lines of
        # nested closures inside ``run()``.
        self._app: TUIApplication | None = None
        self._renderer: AgentEventRenderer | None = None
        self._unsub_activity: Callable[[], None] | None = None
        self._emit_console: RichConsole | None = None
        self._loud_handler_reentrant: bool = False
        # Own ShellTools for bang (!) commands — avoids cross-loop issues
        # when the agent's shell was created on a different event loop.
        self._bang_shell: ShellTools | None = None

    @property
    def show_python(self) -> bool:
        """Whether to display full Python code execution panels."""
        return self.config.tui.show_python

    @show_python.setter
    def show_python(self, value: bool) -> None:
        self.config.tui.show_python = value

    @property
    def show_diffs(self) -> bool:
        """Whether to display semantic file-edit diffs inline."""
        return self.config.tui.show_diffs

    @show_diffs.setter
    def show_diffs(self, value: bool) -> None:
        self.config.tui.show_diffs = value

    @property
    def session_id(self) -> str | None:
        return self._session_manager.session_id if self._session_manager else None

    def _context_usage_label(self) -> str:
        """Compact ``"ctx N%"`` label from the most recent ContextWindowStats.

        The percentage is the provider-reported prompt-token count over the
        USABLE context window (model window minus the output-token reserve) —
        so 100% means "the next call at the current completion budget will be
        rejected", not "the window is byte-full". Until the first response
        returns usage (``prompt_tokens`` is None) we show the placeholder
        ``"ctx —"`` rather than a local estimate.
        """
        stats = getattr(self.agent, "context_stats", None)
        if stats is None:
            return "ctx —"
        util = stats.overall_utilization  # prompt / (window - output reserve)
        if util is None:
            return "ctx —"
        return f"ctx {util * 100:.0f}%"

    # ------------------------------------------------------------------
    # Exit diagnostics
    # ------------------------------------------------------------------

    def _dump_exit_diagnostics(self) -> None:
        """Print pending tasks, threads, and subprocesses on exit for debugging hangs."""
        import sys
        import threading

        lines: list[str] = []

        # Pending asyncio tasks
        try:
            pending = [t for t in asyncio.all_tasks() if not t.done()]
            if pending:
                lines.append(f"Pending asyncio tasks ({len(pending)}):")
                for t in pending:
                    coro = t.get_coro()
                    name = getattr(coro, "__qualname__", str(coro))
                    lines.append(f"  - {t.get_name()}: {name}")
        except RuntimeError:
            pass

        # Non-daemon threads still alive
        alive = [
            t
            for t in threading.enumerate()
            if t.is_alive() and not t.daemon and t != threading.main_thread()
        ]
        if alive:
            lines.append(f"Live non-daemon threads ({len(alive)}):")
            for t in alive:
                lines.append(f"  - {t.name} (ident={t.ident})")

        # Background tasks tracked by this session
        bg = [t for t in self._background_tasks if not t.done()]
        if bg:
            lines.append(f"Background session tasks ({len(bg)}):")
            for t in bg:
                lines.append(f"  - {t.get_name()}")

        if lines:
            sys.stderr.write("\n\033[2m[exit diagnostics]\n")
            for line in lines:
                sys.stderr.write(f"  {line}\n")
            sys.stderr.write("\033[0m\n")
            sys.stderr.flush()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Drive the single long-lived ``TUIApplication`` REPL.

        One Application owns the whole terminal: output scrolls above
        the live prompt region via a single-consumer block queue, so
        producers (command dispatch, agent events, messages) render in
        strict submission order and there's no handoff race that drops
        the first keystroke between turns.

        The handler logic lives in instance methods (``_on_command``,
        ``_on_user_message_ui``, ``_emit_text``, ``_loud_handler``, …);
        ``run`` is just the wiring.
        """
        from .agent_event_renderer import AgentEventRenderer
        from .input_handler import SlashCommandCompleter
        from .output import TextOutput
        from .theme import CATPPUCCIN_THEME
        from .tui_application import TUIApplication

        # Save terminal attributes so we can restore them on exit, even
        # if prompt_toolkit crashes without resetting the terminal.
        self._saved_termios = None
        try:
            import termios

            if sys.stdin.isatty():
                self._saved_termios = termios.tcgetattr(sys.stdin.fileno())
        except (ImportError, OSError):
            pass

        # The block-rendering Rich Console: re-used across ``_emit_text``
        # calls with width reset per-call to track live terminal width.
        self._emit_console = RichConsole(
            file=io.StringIO(),
            force_terminal=True,
            color_system="256",
            width=120,
            theme=CATPPUCCIN_THEME,
        )

        self._app = TUIApplication(
            agent=self.agent,
            on_command=self._on_command,
            on_bang=self._on_bang,
            on_output=self._on_app_output,
            completer=SlashCommandCompleter(self.registry),
            session_label=self._session_label,
            config=self.config,
            full_screen=self.config.tui.full_screen,
            submission_guard=self._llm_submission_error,
        )
        bind_app = getattr(self.frontend, "bind_app", None)
        if callable(bind_app):
            bind_app(self._app)
        # Wire agent_run into all commands so they can dispatch mutations
        # to the agent thread via self.agent_run(fn). agent_run_async is the
        # awaitable variant — commands running on the UI loop use it so they
        # never block the loop (and stall message() output / spin the prompt).
        self._handler._agent_run_async = self._app.agent_run_async
        for cmd in self.registry.commands():
            cmd._agent_run = self._app.agent_run
            cmd._agent_run_async = self._app.agent_run_async
        # Wire the user-bar render + SessionUserMessage log on the channel's
        # on_get hook so the echo fires when the dispatcher (or agent
        # code mid-turn) actually dequeues the message — symmetric across
        # both consumer paths, which is why the hook lives on the queue
        # and not on the dispatcher loop. self.agent is typed as Agent;
        # the queue is on BaseTUIAgent. getattr matches the existing
        # convention in tui_application.py for the same lookup.
        queue = getattr(self.agent, "_user_messages_in", None)
        if queue is not None:

            def _on_user_message_hook(text: str) -> None:
                # DB writes run here on the agent loop thread (the on_get
                # caller), keeping all sqlite access on one thread.
                user_event_id = None
                user_tags = None
                if self._session_manager is not None:
                    recorded = self._session_manager.record_user(text)
                    if isinstance(recorded, tuple) and len(recorded) == 2:
                        tag, event = recorded
                        user_tags = {str(tag)}
                        event_id = getattr(event, "id", None)
                        if event_id is not None:
                            user_event_id = str(event_id)
                # UI rendering must happen on the UI loop.
                app = self._app
                loop = getattr(app, "_loop", None) if app is not None else None
                if loop is None:
                    self._on_user_message_ui(text, event_id=user_event_id, tags=user_tags)
                    return
                try:
                    on_ui_loop = asyncio.get_running_loop() is loop
                except RuntimeError:
                    on_ui_loop = False
                if on_ui_loop:
                    self._on_user_message_ui(text, event_id=user_event_id, tags=user_tags)
                else:
                    loop.call_soon_threadsafe(
                        lambda: self._on_user_message_ui(
                            text,
                            event_id=user_event_id,
                            tags=user_tags,
                        )
                    )

            queue.set_on_get(_on_user_message_hook)

        # Swap the frontend's Rich Console for one that writes through
        # our block queue, so slash-command output (e.g. /help tables)
        # lands in scrollback instead of clobbering the live prompt.
        tui_console = getattr(self.frontend, "console", None)
        if tui_console is not None and hasattr(tui_console, "replace_console"):
            tui_console.replace_console(
                RichConsole(
                    file=_EmitStream(
                        self._app.emit_block,
                        replay_width=lambda app=self._app: app.output_columns(minimum=40),
                        clear=self._app.clear_transcript,
                    ),  # type: ignore[arg-type]
                    force_terminal=True,
                    color_system="256",
                    width=120,
                    theme=CATPPUCCIN_THEME,
                )
            )

            for output in self._initial_outputs:
                await self.frontend.render(output)
            self._initial_outputs.clear()

        self._renderer = AgentEventRenderer(
            agent=self.agent,
            emit_text=self._emit_text,
            show_python=lambda: self.show_python,
            show_diffs=lambda: self.show_diffs,
            pending_code=self._pending_code,
            colors=self._colors,
        )
        # Replace Python's default asyncio exception handler with one
        # that surfaces every swallowed task exception into the TUI.
        # Without this, any coroutine we schedule (spinner, commands,
        # background bookkeeping) that raises vanishes into logging and
        # the user sees "nothing happened".
        self._startup_loop = asyncio.get_running_loop()
        self._prev_exception_handler = self._startup_loop.get_exception_handler()
        self._startup_loop.set_exception_handler(self._loud_handler)

        # Subscribe inside the try so any exception between attach and
        # ``app.run_async`` completion still fires ``renderer.detach``
        # in the finally.
        try:
            self._renderer.attach()
            # Event-driven activity tracking: LLMCallStart/LLMCallEnd "on"
            # hooks feed get_activity() (and /activity) without inferring
            # model-wait state from cell boundaries.
            try:
                from nooa.runtime.debug_handler import attach_activity_tracking

                em = getattr(self.agent, "event_manager", None)
                if em is not None:
                    self._unsub_activity = attach_activity_tracking(em)
            except Exception:
                logger.debug("attach_activity_tracking failed", exc_info=True)
            auto_connect_mcp = getattr(self.registry, "auto_connect_mcp", None)
            if callable(auto_connect_mcp):
                auto_connect_result = auto_connect_mcp()
                if asyncio.iscoroutine(auto_connect_result):
                    self._fire_and_forget(auto_connect_result)
            await self._app.run_async()
        except (KeyboardInterrupt, EOFError):
            await self.frontend.render(
                TextOutput("Interrupted by the user. Exiting TUI...", "warning")
            )
        finally:
            # Order matters:
            # 1. Detach the renderer FIRST. Clears agent._render_message
            #    so any post-shutdown self.message() call (e.g. from
            #    save_snapshot) doesn't write through a dead emit_text.
            # 2. Cancel AND AWAIT fire-and-forget tasks so they don't
            #    trip "Task was destroyed but it is pending" warnings
            #    when the loop closes.
            # 3. Diagnostics, frontend close, snapshot, session close.
            self._renderer.detach()
            if self._unsub_activity is not None:
                try:
                    self._unsub_activity()
                except Exception:
                    logger.debug("detach activity tracking failed", exc_info=True)
                self._unsub_activity = None
            # Shut down spawned jobs before cancelling background tasks
            # so generator finally blocks run cleanly.
            qm = getattr(self.agent, "queue_manager", None)
            if qm is not None:
                app = getattr(self, "_app", None)
                if app is not None:
                    await app.shutdown_agent_queue_manager(agent=self.agent)
                else:
                    await qm.shutdown()
            await self._cancel_background_tasks()
            if self._bang_shell is not None:
                await self._bang_shell.close()
                self._bang_shell = None
            self._dump_exit_diagnostics()
            self.frontend.close()
            if self._session_manager is not None:
                storage = getattr(self.agent, "_storage", None)
                if storage is not None and hasattr(storage, "save_snapshot"):
                    try:
                        app = getattr(self, "_app", None)
                        if app is not None:
                            await app.agent_run_async(lambda: storage.save_snapshot(self.agent))
                        else:
                            # No agent loop — safe to call inline (single-threaded).
                            storage.save_snapshot(self.agent)
                    except Exception:
                        # A failed shutdown snapshot silently loses ALL durable
                        # state (vars, todos, ...) on the next resume — surface
                        # it at WARNING, not DEBUG, so it can't hide.
                        logger.warning("save_snapshot on shutdown failed", exc_info=True)
            # Coding agents own long-lived shell and LLM resources. Close them
            # after the final snapshot but before releasing session storage.
            close_agent = getattr(type(self.agent), "close", None)
            if callable(close_agent):
                try:
                    close_result = close_agent(self.agent)
                    if inspect.isawaitable(close_result):
                        await close_result
                except Exception:
                    logger.warning("agent shutdown failed", exc_info=True)
            if self._session_manager is not None:
                self._session_manager.close()
            # Restore the previous exception handler so we don't pin
            # this Session or route unrelated failures through a torn-down TUI.
            if self._startup_loop is not None:
                self._startup_loop.set_exception_handler(self._prev_exception_handler)
            self._restore_terminal()
            self._print_exit_message()

    async def _on_app_output(self, output: Any) -> None:
        """Render structured output emitted by ``TUIApplication``.

        The dispatcher can run on the dedicated agent loop thread, while
        frontend rendering belongs on the UI/startup loop. This bridge keeps
        the app frontend-agnostic and preserves one structured output path for
        terminal and non-terminal frontends.
        """
        loop = self._startup_loop
        if loop is None:
            return

        async def _render() -> None:
            await self.frontend.render(output)

        try:
            on_ui_loop = asyncio.get_running_loop() is loop
        except RuntimeError:
            on_ui_loop = False
        if on_ui_loop:
            await _render()
            return

        future = asyncio.run_coroutine_threadsafe(_render(), loop)
        await asyncio.wrap_future(future)

    def _restore_terminal(self) -> None:
        """Best-effort restoration of terminal state on exit.

        prompt_toolkit normally restores the terminal, but on crashes,
        signals, or unhandled exceptions the terminal can be left in raw
        mode with echo disabled. We saved termios attrs at startup and
        restore them here as a safety net.
        """
        try:
            import termios

            if self._saved_termios is not None and sys.stdin.isatty():
                termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, self._saved_termios)
                return
        except (ImportError, OSError):
            pass
        # Fallback: if termios isn't available or failed, shell out to stty
        try:
            import subprocess

            if sys.stdin.isatty():
                subprocess.run(["stty", "sane"], stdin=sys.stdin, check=False)
        except Exception:
            pass

    def _print_exit_message(self) -> None:
        """Print the parting line to stderr with session id + name.

        Runs after session_manager.close() so the persisted session is
        what the user sees referenced. Goes through ``sys.stderr``
        (not the frontend) because the prompt_toolkit Application has
        already cleaned up its terminal state and the frontend is
        closed.
        """
        sm = self._session_manager
        if sm is not None:
            short = (sm.session_id or "")[:8]
            name = sm.name
            if name and short:
                tag = f"{name} [{short}]"
            elif short:
                tag = f"[{short}]"
            else:
                tag = ""
        else:
            tag = ""
        suffix = f" — {tag}" if tag else ""
        sys.stderr.write(f"\n\x1b[2mGoodbye! Stay vibing.{suffix}\x1b[0m\n")
        if sm is not None and short:
            sys.stderr.write(
                f"\x1b[2mResume this session with \x1b[0m\x1b[1mnooa tui -c {short}\x1b[0m\n"
            )

    # ------------------------------------------------------------------
    # Handlers — driven by TUIApplication, called in run()'s event loop.
    # Each closes over ``self._app`` / ``self._renderer`` / ``self._emit_console``
    # set up in ``run()``. Don't call any of these before ``run()`` has
    # started; calling an assertion-gated attribute access ("``self._app``
    # is None") raises a descriptive error.
    # ------------------------------------------------------------------

    @property
    def _colors(self) -> dict[str, str]:
        """Theme colour table — reads the current theme so ``/theme`` has
        effect without restarting the app."""
        from .theme import COLORS

        return COLORS

    def _render_to_ansi(self, renderable: Any) -> str:
        """Render a Rich renderable using the current terminal width."""
        assert self._emit_console is not None
        from .tui_application import terminal_cols

        try:
            width = self._app.output_columns(minimum=40)  # type: ignore[union-attr]
        except Exception:
            width = terminal_cols(minimum=40)
        self._emit_console.width = max(int(width), 40)
        buf = io.StringIO()
        self._emit_console.file = buf
        self._emit_console.print(renderable)
        return buf.getvalue()

    def _emit_text(
        self,
        renderable: Any,
        *,
        event_id: str | None = None,
        tags: set[str] | frozenset[str] | None = None,
        keep: bool = False,
    ) -> None:
        """Render a Rich renderable → ANSI → enqueue to the block queue.

        In fullscreen mode, also retain a replay callback that re-renders the
        same semantic Rich/Markdown object at the current width on resize.
        """
        assert self._app is not None

        rendered = self._render_to_ansi(renderable)
        full_screen = bool(
            getattr(getattr(getattr(self, "config", None), "tui", None), "full_screen", False)
            is True
        )
        replay = (lambda r=renderable: self._render_to_ansi(r)) if full_screen else None
        if replay is None:
            self._app.emit_block(rendered, event_id=event_id, tags=tags, keep=keep)
        else:
            self._app.emit_block(rendered, replay=replay, event_id=event_id, tags=tags, keep=keep)

    def _get_command_runner(self):
        """Return the TUI-local command runner, creating it lazily for tests."""
        runner = getattr(self, "_command_runner", None)
        if runner is None:
            from .command_runner import CommandRunner

            frontend = getattr(self, "frontend", None)
            render = getattr(frontend, "render", None)
            if not callable(render):

                async def render(_output):
                    return None

            app = getattr(self, "_app", None)
            set_dynamic_status = getattr(app, "set_command_status", None)
            set_dynamic_queue = getattr(app, "set_command_queue", None)
            runner = CommandRunner(
                render,
                set_dynamic_status=set_dynamic_status if callable(set_dynamic_status) else None,
                set_dynamic_queue=set_dynamic_queue if callable(set_dynamic_queue) else None,
            )
            self._command_runner = runner
        return runner

    async def _on_command(self, text: str) -> None:
        """Handle one slash command through the TUI-local command runner."""

        async def _work():
            return await self._run_command(text)

        await self._get_command_runner().run(kind="slash", text=text, work=_work)

    async def _run_command(self, text: str) -> Callable[[], Awaitable[None]] | None:
        """Run one slash command body and return any post-done render callback."""
        assert self._app is not None

        result = await self._handler.handle(text, render_outputs=False)
        if result.new_session_manager is not None:
            # Suppress normal cancelled-turn UX/restart while the command runner
            # moves storage/session state; the post-swap render is the user-visible
            # result for /clear, /session new, and /session resume.
            self._app._session_transitioning = True
            try:
                # Cancel the running agent turn so it doesn't keep working
                # in the stale session after /clear or /session new.
                await self._app.cancel_agent_turn(source="session")
                await self._swap_session_manager(result.new_session_manager)
                post_swap = getattr(result, "post_session_swap", None)
                if post_swap is not None:
                    extra_outputs = await self._app.agent_run_async(post_swap)
                    if extra_outputs:
                        result.outputs.extend(extra_outputs)
                self._first_message = None
            finally:
                self._app._session_transitioning = False
        if (
            result.compact_done
            and self._first_message
            and self._session_manager
            and not self._session_manager.user_named
        ):
            self._name_session_on_agent_loop(self._first_message)

        async def _render_result_outputs() -> None:
            frontend = getattr(self, "frontend", None)
            render = getattr(frontend, "render", None)
            if not callable(render):
                return

            from .commands import render_command_outputs

            await render_command_outputs(frontend, result.outputs)

        if result.exit:

            async def _render_outputs_then_exit() -> None:
                await _render_result_outputs()
                self._app.exit()

            return _render_outputs_then_exit

        if result.slash_result is None and result.agent_message is None:
            return _render_result_outputs
        if result.slash_result is not None:
            # slash-inception: a SwapAgentRequest asks us to hot-swap the agent
            # the dispatcher drives. The skill (running on the UI loop, with no
            # app handle) built the new agent and shared the old agent's live
            # channels; we hold the app, so we do the actual swap here.
            # Identify the request structurally (duck typing on its fields)
            # rather than by class name — the inception skill lives in another
            # package, and a string __name__ check breaks silently on a rename
            # or a same-named class from elsewhere.
            _swap_req = getattr(result.slash_result, "value", None)
            if (
                _swap_req is not None
                and hasattr(_swap_req, "new_agent")
                and hasattr(_swap_req, "seed_prompt")
            ):
                from .output import AgentMessage

                async def _render_and_swap() -> None:
                    _text = str(result.slash_result)
                    if _text:
                        await self.frontend.render(AgentMessage(_text, show_rule=False))
                    assert self._app is not None
                    _new = _swap_req.new_agent
                    # Queue the seed prompt on the SHARED user-messages channel, then
                    # swap+restart the dispatcher onto the new agent (on the agent loop).
                    _q = getattr(_new, "_user_messages_in", None)
                    if _q is not None:
                        _q.put(_swap_req.seed_prompt)
                    await self._app.agent_run_async(lambda: self._app.swap_agent(_new))

                return _render_and_swap

            # Show slash-command output after the durable done marker. Skill slash
            # commands often return Markdown (tables, lists), so render via the
            # frontend instead of dumping raw text through emit_block.
            async def _render_slash_output() -> None:
                await _render_result_outputs()
                text = str(result.slash_result)
                if text:
                    from .output import AgentMessage

                    await self.frontend.render(AgentMessage(text, show_rule=False))

            if not _effective_slash_output_to_agent(self.agent, result.slash_result):
                # User-only command (e.g. a read-only /mcp list): the human sees
                # the output above, but it is NOT fed to the agent — no queue
                # put, no submitted message, no agent turn spent.
                return _render_slash_output
            # Post the full SlashCommandResult to the slash_commands queue
            # (agent can access .value for the raw Python object). This put
            # also wakes the dispatcher (qm.race() includes the slash_commands
            # queue), so the slash command must NOT also be submitted as a user
            # message — doing both delivers the same command twice (once on
            # each queue).
            #
            # Strict routing: a slash command result ONLY ever travels the
            # slash_commands channel — never user_messages. Every real TUI
            # agent (BaseTUIAgent) creates that channel in __init__, so this
            # is always present in normal use. If self.agent is something else
            # (a plain Agent, a partially-initialized object, a test double),
            # there is no slash-capable destination: warn loudly to scrollback
            # and drop the result rather than smuggling it through the
            # user-message path (where it would masquerade as something the
            # human typed).
            slash_ch = getattr(self.agent, "_slash_commands_in", None)
            if slash_ch is not None:
                slash_ch.put(result.slash_result)
            else:
                self._emit_text(
                    Text(
                        f"⚠ agent has no slash_commands channel — dropping /{result.slash_result.command}",
                        style="yellow",
                    )
                )
            return _render_slash_output
        elif result.agent_message is not None:
            # Slash-command-generated agent turn — feed through the same
            # path as a typed message so the user bar, session bookkeeping,
            # and agent dispatch stay consistent.
            self._app.submit_message(result.agent_message)
            return _render_result_outputs

    async def _on_bang(self, body: str) -> None:
        """Dispatch a ``!shell-command`` body through the command runner."""
        text = "!" + body

        async def _work():
            return await self._handle_bang(text, defer_render=True)

        await self._get_command_runner().run(kind="bang", text=text, work=_work)

    def _session_label(self) -> str:
        """Render configured toolbar items on the rule above the input."""
        from .toolbar import ToolbarContext

        manager = self._session_manager
        shell = getattr(self.agent, "shell", None)
        cwd = Path(getattr(shell, "cwd", self.config.agent.working_dir)).resolve()
        return self._toolbar.render(
            self.config.tui.toolbar_items,
            ToolbarContext(
                model=self.config.tui.default_model,
                working_directory=cwd,
                context_usage=self._context_usage_label(),
                session_id=manager.session_id if manager is not None else None,
                session_title=manager.name if manager is not None else None,
                agent=self.agent,
            ),
        )

    def _llm_submission_error(self) -> str | None:
        """Explain why agent-bound input is disabled while LLM config is broken."""
        health = getattr(self.registry, "blocking_llm_health", None)
        if health is None or getattr(health, "blocking", False) is not True:
            return None
        lines = [
            "Cannot send this message because the configured LLM is unavailable.",
        ]
        if health.error_message:
            lines.append(health.error_message)
        if health.fix_hint:
            lines.extend(("", health.fix_hint))
        lines.extend(
            (
                "",
                "Slash commands and !shell commands still work.",
                "After fixing the registry, use /model <name> to validate and enable the model.",
            )
        )
        return "\n".join(lines)

    def _on_user_message_ui(
        self,
        text: str,
        *,
        event_id: str | None = None,
        tags: set[str] | frozenset[str] | None = None,
    ) -> None:
        """Render the user's submitted text as a full-width grey bar and
        reset per-turn renderer state.

        DB bookkeeping (record_user) is handled by the on_get hook on
        the agent loop thread; this method only does UI work.
        """
        assert self._renderer is not None and self._app is not None

        if self._first_message is None:
            self._first_message = text
            if (
                self._session_manager is not None
                and not self._session_manager.user_named
                and not (self._session_manager.name or "").strip()
            ):
                self._name_session_on_agent_loop(text)

        bar = _build_user_bar(text, self._app, self._colors)
        full_screen = bool(
            getattr(getattr(getattr(self, "config", None), "tui", None), "full_screen", False)
            is True
        )
        replay = (
            (lambda t=text: _build_user_bar(t, self._app, self._colors)) if full_screen else None
        )
        emit_kwargs = {}
        if event_id is not None:
            emit_kwargs["event_id"] = event_id
        if tags:
            emit_kwargs["tags"] = tags
        if replay is None:
            self._app.emit_block(bar, **emit_kwargs)
        else:
            self._app.emit_block(bar, replay=replay, **emit_kwargs)
        self._renderer.reset_turn()

    def _loud_handler(self, _loop: asyncio.AbstractEventLoop, context: dict) -> None:
        """asyncio exception handler that surfaces every swallowed task
        exception into the scrollback instead of Python's logging.

        Guards against re-entry: if ``emit_block`` itself raises, the
        asyncio loop would call us back with the new exception, yielding
        unbounded recursion. On re-entry we fall back to a bare stderr
        write.
        """
        assert self._app is not None

        msg = context.get("message", "")
        exc = context.get("exception")
        # --- gl-212 diagnostics: enrich "bound to a different event loop" ---
        if exc is not None and "bound to a different event loop" in str(exc):
            startup_loop = getattr(self, "_startup_loop", None)
            diag_lines = [
                "[gl-212] asyncio.Lock bound to a different event loop — diagnostic dump:",
                f"  handler loop: id={id(_loop):#x}",
                f"  startup loop: id={id(startup_loop):#x} (same={startup_loop is _loop})",
                f"  exception: {exc!r}",
            ]
            # Inspect known Lock instances on the agent
            try:
                shell = getattr(self.agent, "shell", None)
                if shell is not None:
                    bash = getattr(shell, "_session", None)
                    if bash is not None:
                        lock = getattr(bash, "_lock", None)
                        diag_lines.append(f"  BashSession._lock: {lock!r}")
                        if lock is not None and hasattr(lock, "_loop"):
                            diag_lines.append(
                                f"    lock._loop: id={id(lock._loop):#x} (same={lock._loop is _loop})"
                            )
            except Exception as e:
                diag_lines.append(f"  [BashSession inspection failed: {e}]")
            try:
                actor = getattr(self.agent, "_actor", None)
                if actor is not None:
                    gen_lock = getattr(actor, "_generation_lock", None)
                    diag_lines.append(f"  Actor._generation_lock: {gen_lock!r}")
                    if gen_lock is not None and hasattr(gen_lock, "_loop"):
                        diag_lines.append(
                            f"    lock._loop: id={id(gen_lock._loop):#x} (same={gen_lock._loop is _loop})"
                        )
            except Exception as e:
                diag_lines.append(f"  [Actor inspection failed: {e}]")
            diag_lines.append("  Suggestion: restart the TUI. File details on gl#212.")
            diag_lines.append("")
            diag_msg = "\n".join(diag_lines)
            if self._loud_handler_reentrant:
                err = sys.__stderr__
                if err is not None:
                    err.write(diag_msg)
                    err.flush()
            else:
                self._loud_handler_reentrant = True
                try:
                    self._app.emit_block(diag_msg)
                except Exception:
                    # emit_block failed (plausible during degraded loop state) —
                    # fall back to stderr so the diagnostic isn't lost.
                    err = sys.__stderr__
                    if err is not None:
                        err.write(diag_msg)
                        err.flush()
                finally:
                    self._loud_handler_reentrant = False
            # Fall through to normal handler for the full traceback
        # litellm's LiteLLMAiohttpTransport recreates its cached aiohttp
        # ClientSession on error-recovery / loop-mismatch / session-closed
        # paths without awaiting close() on the old one. When GC reaps the
        # orphan, aiohttp's finalizer fires this warning via
        # call_exception_handler. It's upstream noise, not our bug, and
        # drowns real diagnostics — drop it before formatting.
        if msg == "Unclosed client session" or msg == "Unclosed connector":
            return
        task = context.get("task")
        # litellm's LoggingWorker._worker_loop tasks get orphaned when the
        # global singleton detects an event loop change and drops the old
        # _worker_task reference without cancelling it. Harmless upstream noise.
        if "Task was destroyed" in msg and task is not None and "LoggingWorker" in repr(task):
            return
        future = context.get("future")
        source_tb = context.get("source_traceback")
        line = f"[asyncio] {msg}"
        if exc is not None:
            line += f" — {type(exc).__name__}: {exc}"
            if hasattr(exc, "__traceback__"):
                line += "\n" + "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                )
        # Non-exception contexts (e.g. "Task was destroyed but it is pending!")
        # carry the offending task/future and — when PYTHONASYNCIODEBUG=1 — a
        # source_traceback pointing at where it was created.
        if task is not None:
            line += f"\n  task={task!r}"
        if future is not None and future is not task:
            line += f"\n  future={future!r}"
        if source_tb:
            line += "\n  source_traceback (where task was created, enable PYTHONASYNCIODEBUG=1):\n"
            line += "".join(traceback.format_list(source_tb))
        # Other keys asyncio sometimes supplies (handle, protocol, transport,
        # socket, peername, client_session, ...). 2000 chars per field covers
        # aiohttp ``ClientSession`` reprs (~1.1KB — connector/base_url/auth
        # all land in the middle) without letting a pathologically huge
        # transport (SSL state, large buffers) swamp the scrollback.
        from nooa.agentdoc import truncating_pformat

        _known = ("message", "exception", "task", "future", "source_traceback")
        for k, v in context.items():
            if k in _known:
                continue
            line += f"\n  {k}={truncating_pformat(v, max_chars=2000)}"
        line += "\n"

        if self._loud_handler_reentrant:
            err = sys.__stderr__
            if err is not None:
                try:
                    err.write(line)
                    err.flush()
                except Exception:
                    pass
            return
        self._loud_handler_reentrant = True
        try:
            self._app.emit_block(line)
        except Exception as inner:
            err = sys.__stderr__
            if err is not None:
                try:
                    err.write(f"[loud_handler fallback] {inner}\n{line}")
                    err.flush()
                except Exception:
                    pass
        finally:
            self._loud_handler_reentrant = False

    def _fire_and_forget(self, coro) -> asyncio.Task:
        """Schedule a coroutine as a tracked background task.

        Tracked means: cancelled at shutdown by ``_cancel_background_tasks``,
        and self-removing from the set when it finishes so the set
        doesn't leak references to completed tasks.
        """
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    async def _cancel_background_tasks(self) -> None:
        """Cancel and await pending fire-and-forget tasks.

        The set is populated by auto-naming and post-compact renaming;
        tasks that finish remove themselves via ``discard``. At shutdown
        anything still in the set is stale. We cancel then ``gather``
        so the cancellation actually propagates before the loop closes —
        without the await asyncio emits "Task was destroyed but it is
        pending" on some orderings.
        """
        pending = [t for t in self._background_tasks if not t.done()]
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._background_tasks.clear()
        # Cancel any naming futures dispatched to the agent loop.
        naming = getattr(self, "_naming_futures", None)
        if naming:
            for f in list(naming):
                f.cancel()
            naming.clear()

    # ------------------------------------------------------------------
    # Session manager swap (triggered by /session new)
    # ------------------------------------------------------------------

    async def _swap_session_manager(self, new_sm: "SessionManager") -> None:
        """Close the current session and switch to *new_sm*."""
        app = getattr(self, "_app", None)
        interrupt_reflection = getattr(app, "interrupt_reflection", None)
        if callable(interrupt_reflection):
            await interrupt_reflection()

        # Shut down spawned jobs and flush all queue channels so stale
        # items from the old session don't leak into the new one.
        qm = getattr(self.agent, "queue_manager", None)
        if qm is not None:
            if app is not None:
                await app.shutdown_agent_queue_manager(agent=self.agent, flush=True)
            else:
                await qm.shutdown()
                for name in qm.names():
                    ch = qm.get_channel(name)
                    if ch.mode == "queue":
                        ch.flush()
        if self._session_manager is not None:
            # Save snapshot before closing so /clear, /session new, and
            # /session resume don't lose the current session's self.v/todo.
            storage = getattr(self.agent, "_storage", None)
            if storage is not None and hasattr(storage, "save_snapshot"):
                try:
                    app = getattr(self, "_app", None)
                    if app is not None:
                        await app.agent_run_async(lambda: storage.save_snapshot(self.agent))
                    else:
                        storage.save_snapshot(self.agent)
                except Exception:
                    logger.debug("save_snapshot on session swap failed", exc_info=True)
            self._session_manager.close()
        self._session_manager = new_sm
        # Point the agent at the new storage AND repoint the agent's
        # stable EventManager at the new backend. The set_backend call
        # is what keeps subscribers (e.g. AgentEventRenderer) alive
        # across the swap.
        if hasattr(self.agent, "_storage"):

            def _do_swap():
                self.agent._storage = new_sm._storage
                self.agent.event_manager.set_backend(new_sm._storage.event_backend)
                self.agent._session_manager = new_sm

                from .bootstrap import configure_tui_memory

                try:
                    configure_tui_memory(
                        self.agent,
                        self.config,
                        agent_db=new_sm.agent_db_path,
                        session_id=new_sm.session_id,
                    )
                except Exception:
                    logger.warning("memory reconfiguration on session swap failed", exc_info=True)

            if app is not None:
                await app.agent_run_async(_do_swap)
            else:
                _do_swap()
        # Propagate to registry and all command instances so /session export etc. use new ID.
        self.registry.session_manager = new_sm
        for cmd in self.registry.commands():
            cmd.session_manager = new_sm
        # Start a fresh trace for the new session so it gets its own .jsonl file.
        # Use the first 8 chars of the SQLite session UUID to correlate trace↔storage.
        try:
            from nooa.tracing import set_session

            from .session_manager import _make_trace_session_name

            set_session(_make_trace_session_name(new_sm.session_id or ""))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Session auto-naming
    # ------------------------------------------------------------------

    def _name_session_on_agent_loop(self, first_message: str) -> None:
        """Dispatch auto-naming to the agent loop to avoid cross-thread sqlite access."""
        agent_loop = getattr(self._app, "_agent_loop", None)
        if isinstance(agent_loop, asyncio.AbstractEventLoop) and agent_loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                self._auto_name_session(first_message), agent_loop
            )
            # Track for cancellation on shutdown. concurrent.futures.Future isn't
            # awaitable so we keep a separate set from _background_tasks.
            self._naming_futures.add(future)
            future.add_done_callback(self._naming_futures.discard)
        else:
            # No agent loop available — fall back to local scheduling.
            self._fire_and_forget(self._auto_name_session(first_message))

    async def _auto_name_session(self, first_message: str) -> None:
        """Generate and persist a session name from the first user message."""
        if self._session_manager is None or self._session_manager.user_named:
            return
        try:
            name = await self.agent.name_session(first_message[:400])  # type: ignore[union-attr]
            name = str(name).strip().strip('"').strip("'")[:60]
            if name and not self._session_manager.user_named:
                self._session_manager.rename(name, user_named=False)
        except Exception:
            # Fallback: first few words of the message
            words = first_message.split()
            fallback = " ".join(words[:5])[:60]
            if fallback:
                self._session_manager.rename(fallback, user_named=False)

    # ------------------------------------------------------------------
    # Bang (!) command routing
    # ------------------------------------------------------------------

    async def _handle_bang(self, user_input: str, *, defer_render: bool = False):
        from .output import BashOutput, TextOutput

        cmd = user_input[1:].strip()
        if not cmd:
            return

        # !commands → run through shell (not recorded as conversation turns)
        if not hasattr(self.agent, "shell"):
            output = TextOutput(
                "Direct bash commands (!) require an agent with shell support.",
                "warning",
            )
            if defer_render:

                async def _render_warning() -> None:
                    await self.frontend.render(output)

                return _render_warning
            await self.frontend.render(output)
            return

        try:
            shell = await self._get_bang_shell()
            result = await shell.run(cmd)
            if result:
                output = BashOutput(
                    stdout=result.stdout or "",
                    stderr=result.stderr or "",
                    return_code=result.returncode,
                )
                if defer_render:

                    async def _render_output() -> None:
                        await self.frontend.render(output)

                    return _render_output
                await self.frontend.render(output)
        except Exception as e:
            if defer_render:
                raise
            await self.frontend.render(TextOutput(f"Bash error: {e}", "error"))

    async def _get_bang_shell(self) -> "ShellTools":
        """Return (lazily creating) a ShellTools owned by the TUI for bang commands.

        The agent's ShellTools may have its asyncio.Lock bound to a different
        event loop.  This dedicated instance is created on the TUI's loop,
        avoiding "attached to a different loop" errors.

        Syncs cwd from the agent's shell on each call so the two stay
        in step when the agent changes directory during a session.
        """
        from nooa.tools.shell_tools import ShellTools

        if self._bang_shell is None:
            cwd = self.agent.shell.cwd if hasattr(self.agent, "shell") else "."
            self._bang_shell = ShellTools(cwd=cwd)
        elif hasattr(self.agent, "shell"):
            agent_cwd = str(self.agent.shell.cwd)
            if str(self._bang_shell.cwd) != agent_cwd:
                await self._bang_shell.run(f"cd {shlex.quote(agent_cwd)}")
        return self._bang_shell
