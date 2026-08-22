# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frontend protocol and TerminalFrontend implementation.

A ``Frontend`` is a **pure rendering surface** — it renders Output objects,
reads user input, and shows/hides a thinking indicator.  It has NO behavior:
no event subscription, no show_python decisions, no streaming state.

All behavior lives in ``Session``.
"""

import asyncio
import io
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .output import (
    AgentMessage,
    BashOutput,
    ClearScreen,
    CodeExecution,
    CommandStatus,
    DiffOutput,
    HelpOutput,
    HistoryReplay,
    Output,
    RichOutput,
    StartupInfo,
    StopReasonOutput,
    TableOutput,
    TextOutput,
    Thinking,
)
from .theme import COLORS

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from .config import Config


def render_history_replay_to_ansi(output: HistoryReplay, width: int) -> str:
    """Render resumed conversation history to one ANSI block at *width*."""
    from rich.console import Console as RichConsole
    from rich.markdown import Markdown
    from rich.rule import Rule
    from rich.text import Text

    dim = COLORS["overlay1"]
    user_color = COLORS["subtext1"]

    render_width = max(int(width), 1)
    buf = io.StringIO()
    # Rich ignores ``width=`` for force_terminal consoles when it can read the
    # real terminal size.  Supplying COLUMNS pins the semantic replay width.
    bc = RichConsole(
        file=buf,
        width=render_width,
        highlight=False,
        force_terminal=True,
        _environ={"COLUMNS": str(render_width), "LINES": "25"},
    )

    if output.show_header:
        omit_note = ""
        if output.omitted_count:
            omit_note = f" ({output.omitted_count} earlier turns omitted)"
        bc.print(
            Rule(
                title=f"[{dim}]session {output.session_id} — history{omit_note}[/]",
                style=COLORS["surface1"],
            )
        )
    for turn in output.turns:
        if turn.role == "user":
            # Apply the style at the print boundary so Rich extends the
            # background through padding on every explicit or wrapped row.
            bc.print(
                Text(f" You: {turn.content}"),
                style=f"{user_color} on {COLORS['surface0']}",
                justify="left",
                overflow="fold",
            )
        else:
            bc.print(Text("OO:", style=f"bold {dim}"))
            bc.print(Markdown(turn.content), style=dim)
        bc.print()
    if output.show_footer:
        bc.print(Rule(style=COLORS["surface1"]))

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Frontend(Protocol):
    """Everything a Session needs from its rendering layer.

    ``TerminalFrontend`` is the concrete implementation of this protocol.
    Commands and Session code should ONLY interact with frontends through
    this protocol — never reach into frontend-specific attributes.

    Frontends are PURE RENDERING — no event subscription, no streaming state,
    no behavioral decisions.
    """

    async def render(self, output: Output) -> None:
        """Render one structured output object."""
        ...

    def batch_render(self) -> "AbstractContextManager[None]":
        """Context manager grouping multiple ``render()`` calls into one block.

        Terminal frontends coalesce the enclosed output into a single
        scrollback write (one ``run_in_terminal`` hop) so the live prompt /
        spinner doesn't repaint between a command's outputs. Non-batching
        frontends may return a no-op context (e.g. ``contextlib.nullcontext``).
        """
        ...

    async def get_input(
        self,
        prompt: str,
        completions: list[str] | None = None,
        default: str = "",
        bottom_toolbar: object = None,
    ) -> str:
        """Read one line of user input.

        Args:
            prompt: The text to display before the cursor.
            completions: Optional list of completion candidates (e.g. model names).
                         When provided the frontend may offer autocomplete.
            default: Pre-fill the input buffer with this text.
            bottom_toolbar: Optional callable for a status bar (e.g. spinner).
        """
        ...

    async def prompt_sensitive(
        self, title: str, message: str, *, link_url: str | None = None
    ) -> str:
        """Collect a masked one-line value without adding it to input history."""
        ...

    async def prompt_text(self, title: str, message: str, default: str = "") -> str:
        """Collect unmasked one-line text in the active host UI."""
        ...

    async def prompt_choice(self, title: str, message: str, options: list[str]) -> str:
        """Collect one value from a searchable list in the active host UI."""
        ...

    async def start_thinking(self, message: str = "thinking...") -> None:
        """Show a loading/thinking indicator."""
        ...

    async def stop_thinking(self) -> None:
        """Hide the loading/thinking indicator."""
        ...

    @property
    def is_connected(self) -> bool:
        """Whether the frontend is still connected (always True for terminal)."""
        ...

    async def open_editor(
        self, filename: str, content: str, language: str = "plaintext"
    ) -> str | None:
        """Open an editor for the given file content. Returns edited content or None."""
        ...

    def close(self) -> None:
        """Clean up frontend resources."""
        ...


# ---------------------------------------------------------------------------
# TerminalFrontend
# ---------------------------------------------------------------------------


FULLSCREEN_EDITOR_POLICY = (
    "External editors are unavailable in fullscreen mode because they require terminal "
    "handoff. Restart with --display-mode native or native-replay, then run /edit again."
)


class ExternalEditorUnavailableError(RuntimeError):
    """Raised when a display mode cannot safely hand the terminal to an editor."""


class TerminalFrontend:
    """Frontend backed by Rich (output) + prompt_toolkit (input).

    Pure rendering — no event subscription or behavioral state.
    """

    def __init__(self, config: "Config") -> None:
        from .console import TUIConsole

        self._config = config
        self._console = TUIConsole()
        self._input_handler = None  # initialised after registry is ready
        self._app = None
        self._renderers = self._build_renderer_map()

    # ------------------------------------------------------------------
    # Input handler initialisation (needs the command registry)
    # ------------------------------------------------------------------

    def bind_app(self, app: Any) -> None:
        """Bind the live TUIApplication for in-app modal views."""
        self._app = app

    async def open_event_explorer(self, event_manager: Any) -> None:
        if self._app is None:
            raise RuntimeError("TUI application is not ready.")
        await self._app.open_event_explorer(event_manager)

    async def open_session_explorer(self) -> None:
        if self._app is None:
            raise RuntimeError("TUI application is not ready.")
        await self._app.open_session_explorer()

    async def open_job_explorer(self) -> None:
        if self._app is None:
            raise RuntimeError("TUI application is not ready.")
        await self._app.open_job_explorer()

    async def open_todo_explorer(self) -> None:
        if self._app is None:
            raise RuntimeError("TUI application is not ready.")
        await self._app.open_todo_explorer()

    async def open_memory_explorer(self) -> None:
        if self._app is None:
            raise RuntimeError("TUI application is not ready.")
        await self._app.open_memory_explorer()

    async def open_activity_overlay(self, outputs: list[Output]) -> None:
        if self._app is None:
            raise RuntimeError("TUI application is not ready.")
        await self._app.open_activity_overlay(outputs)

    def init_input(self, registry) -> None:
        """Wire up the prompt_toolkit input handler with slash completions."""
        from .input_handler import TUIInputHandler

        self._input_handler = TUIInputHandler(
            registry=registry,
            vi_mode=self._config.tui.vi_mode,
        )

    @property
    def is_connected(self) -> bool:
        """Terminal is always connected."""
        return True

    def close(self) -> None:
        """Clean up terminal state."""
        pass

    # ------------------------------------------------------------------
    # Frontend protocol implementation
    # ------------------------------------------------------------------

    async def render(self, output: Output) -> None:  # type: ignore[override]
        """Dispatch *output* to the appropriate Rich rendering call."""
        handler = self._renderers.get(type(output))
        if handler is not None:
            # The live Console is long-lived, but terminal width is not.  Keep
            # every structured render inside the native-scrollback safe width
            # instead of retaining the constructor's arbitrary 120 columns.
            stream = getattr(self._console.console, "file", None)
            replay_width = getattr(stream, "replay_width", None)
            if callable(replay_width):
                self._console.console.width = max(
                    int(replay_width(self._console.console.width or 80)), 1
                )
            handler(output)

    def batch_render(self):
        """Coalesce the enclosed ``render()`` calls into one scrollback block.

        When the Rich console writes through a ``_EmitStream`` (the live TUI),
        this holds its flush so a multi-output command renders as a single
        ``emit_block`` / ``run_in_terminal`` hop — no prompt repaint between
        outputs. Falls back to a no-op context for plain consoles (tests).
        """
        from contextlib import nullcontext

        stream = getattr(self._console.console, "file", None)
        hold = getattr(stream, "hold", None)
        return hold() if callable(hold) else nullcontext()

    def _build_renderer_map(self) -> "dict[type, Callable[[Any], None]]":
        """Build the {Output subclass → handler} dispatch table once at
        construction time. Cheaper and clearer than an isinstance ladder;
        also surfaces unregistered Output kinds as silent no-ops (same
        behaviour as the old default-fallthrough)."""
        return {
            TextOutput: self._render_text,
            TableOutput: self._render_table,
            HelpOutput: self._render_help,
            AgentMessage: self._render_agent_message,
            CodeExecution: self._render_code_execution,
            StartupInfo: self._render_startup,
            ClearScreen: self._render_clear,
            Thinking: self._render_thinking,
            StopReasonOutput: self._render_stop_reason,
            BashOutput: self._render_bash,
            CommandStatus: self._render_command_status,
            RichOutput: self._render_rich,
            DiffOutput: self._render_diff,
            HistoryReplay: self._render_history_replay,
        }

    def _render_help(self, output: HelpOutput) -> None:
        self._console.print_help(output.commands)

    def _render_agent_message(self, output: AgentMessage) -> None:
        self._console.print_agent(
            output.content, show_rule=output.show_rule, soft_wrap=output.soft_wrap
        )

    def _render_clear(self, _output: ClearScreen) -> None:
        stream = getattr(self._console.console, "file", None)
        clear_transcript = getattr(stream, "clear_transcript", None)
        if callable(clear_transcript):
            clear_transcript()
        else:
            self._console.console.clear()

    def _render_thinking(self, output: Thinking) -> None:
        if output.active:
            self._console.start_spinner(output.message)
        else:
            self._console.stop_spinner()

    def _render_stop_reason(self, output: StopReasonOutput) -> None:
        from rich.markup import escape

        self._console.console.print(f"[{COLORS['overlay2']}]{escape(output.display_text())}[/]")

    async def get_input(
        self,
        prompt: str,
        completions: list[str] | None = None,
        default: str = "",
        bottom_toolbar: object = None,
    ) -> str:
        """Read user input from the terminal.

        When *completions* are provided, a temporary PromptSession with
        WordCompleter is used when a prompt supplies a fixed choice list.
        Otherwise the main session (with slash-command completion) is used.
        """
        if completions:
            from prompt_toolkit import PromptSession
            from prompt_toolkit.completion import WordCompleter

            session = PromptSession(
                completer=WordCompleter(completions, ignore_case=True),
                complete_while_typing=True,
            )
            return (await session.prompt_async(prompt)).strip()

        if self._input_handler:
            return await self._input_handler.get_input(
                prompt, default=default, bottom_toolbar=bottom_toolbar
            )

        # Fallback: plain input via executor
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: self._console.console.input(f"[user]{prompt}[/user]").strip()
        )

    async def prompt_sensitive(
        self, title: str, message: str, *, link_url: str | None = None
    ) -> str:
        """Collect OAuth material in the live application's masked modal view."""
        if self._app is not None:
            return await self._app.prompt_sensitive(title, message, link_url=link_url)

        from prompt_toolkit import PromptSession

        self._console.console.print(message)
        if link_url:
            self._console.console.print(link_url, markup=False)
        session = PromptSession(is_password=True)
        return (await session.prompt_async(f"{title}: ")).strip()

    async def prompt_text(self, title: str, message: str, default: str = "") -> str:
        """Collect ordinary text without launching a nested live application."""
        if self._app is not None:
            return await self._app.prompt_text(title, message, default)
        self._console.console.print(message)
        return await self.get_input(f"{title}: ", default=default)

    async def prompt_choice(self, title: str, message: str, options: list[str]) -> str:
        """Collect a searchable choice without launching a nested live application."""
        if self._app is not None:
            return await self._app.prompt_choice(title, message, options)
        self._console.console.print(message)
        return await self.get_input(f"{title}: ", completions=options)

    async def start_thinking(self, message: str = "thinking...") -> None:
        self._console.start_spinner(message)

    async def stop_thinking(self) -> None:
        self._console.stop_spinner()

    # ------------------------------------------------------------------
    # Internal rendering helpers
    # ------------------------------------------------------------------

    def _render_text(self, output: TextOutput) -> None:
        lvl = output.level
        if lvl == "error":
            self._console.print_error(output.content)
        elif lvl == "warning":
            self._console.print_warning(output.content)
        elif lvl == "success":
            self._console.print_success(output.content)
        elif lvl == "status":
            self._console.print_status(output.content)
        else:
            self._console.print_info(output.content)

    def _render_table(self, output: TableOutput) -> None:
        self._console.print_table(
            output.title, output.columns, output.rows, show_header=output.show_header
        )
        if output.footer:
            self._console.print_status(output.footer)

    def _render_bash(self, output: BashOutput) -> None:
        if output.stdout:
            # Ensure text ends with newline so _EmitStream produces exactly
            # ONE flush (one emit_block call). Two separate flushes (text
            # sans \n, then bare \n) create two run_in_terminal hops with
            # a prompt repaint between them that overwrites single-line
            # output. See issue #156.
            text = output.stdout if output.stdout.endswith("\n") else output.stdout + "\n"
            self._console.console.print(text, end="")
        if output.stderr:
            self._console.print_error(output.stderr)

    def _render_command_status(self, output: CommandStatus) -> None:
        prefix = "/" if output.kind == "slash" else "!"
        text = output.text if output.text.startswith(("/", "!")) else f"{prefix}{output.text}"
        label = f"[{output.id}] {text}"
        if output.state == "queued":
            self._console.print_status(f"○ queued {label}")
        elif output.state == "running":
            self._console.print_status(f"▶ running {label}")
        elif output.state == "done":
            from rich.markup import escape

            self._console.print_success(f"[command]{escape(text)}[/command]")
        elif output.state == "cancelled":
            self._console.print_warning(f"■ cancelled {label}")
        else:
            suffix = f": {output.error}" if output.error else ""
            self._console.print_error(f"✗ failed {label}{suffix}")

    def _render_rich(self, output: RichOutput) -> None:
        """Terminal fallback for rich visual output — show summary text."""
        label = f"[bold]{output.kind}[/bold]"
        if output.title:
            label = f"[bold]{output.kind}: {output.title}[/bold]"
        if output.fallback_text:
            self._console.print_info(f"{label} — {output.fallback_text}")
        else:
            self._console.print_info(
                f"{label} (rich rendering available in the web terminal — run `nooa-term`)"
            )

    def _render_diff(self, output: DiffOutput) -> None:
        """Render a unified diff with syntax highlighting."""
        from rich.rule import Rule
        from rich.syntax import Syntax

        if not output.diff.strip():
            return
        label = f"diff — {output.filename}" if output.filename else "diff"
        self._console.console.print(
            Rule(title=f"[bold]{label}[/bold]", style=COLORS["surface2"], align="left")
        )
        self._console.console.print(Syntax(output.diff, "diff", theme="monokai", word_wrap=True))

    async def open_editor(
        self, filename: str, content: str, language: str = "plaintext"
    ) -> str | None:
        """Open *content* in $EDITOR via a temp file and return the edited text."""
        import os
        import subprocess
        import tempfile
        from pathlib import Path

        # Alternate-screen mode has one terminal owner for its entire lifetime.
        # Do not even import the handoff primitive on this policy path.
        from .config import DisplayMode

        if getattr(self._app, "display_mode", None) is DisplayMode.FULLSCREEN:
            raise ExternalEditorUnavailableError(FULLSCREEN_EDITOR_POLICY)

        from prompt_toolkit.application import run_in_terminal

        suffix = Path(filename).suffix or ".txt"
        editor = os.environ.get("EDITOR", os.environ.get("VISUAL", "vi"))

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        try:
            os.close(tmp_fd)
            Path(tmp_path).write_text(content)
            self._console.stop_spinner()
            await run_in_terminal(lambda: subprocess.run([editor, tmp_path]), in_executor=True)
            return Path(tmp_path).read_text(errors="replace")
        except Exception:
            return None
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def _render_history_replay(self, output: HistoryReplay) -> None:
        """Render past conversation turns in a dimmed style.

        Uses a buffered console to pre-render all output, then writes one block.
        When the live TUI has redirected the console through ``_EmitStream``,
        attach a semantic replay callback so fullscreen resize replay re-renders
        resumed history at the new width instead of reusing stale ANSI.
        """
        stream = self._console.console.file
        replay_width = getattr(stream, "replay_width", None)
        if callable(replay_width):
            width = replay_width(self._console.console.width or 80)
        else:
            width = self._console.console.width or 80

        rendered = render_history_replay_to_ansi(output, width)
        emit_with_replay = getattr(stream, "emit_with_replay", None)
        if getattr(stream, "supports_semantic_replay", False) is True and callable(
            emit_with_replay
        ):
            emit_with_replay(
                rendered,
                lambda o=output, s=stream, w=width: render_history_replay_to_ansi(
                    o, s.replay_width(w) if callable(getattr(s, "replay_width", None)) else w
                ),
            )
            return

        # Single write to the real terminal
        stream.write(rendered)
        stream.flush()

    def _render_startup(self, info: StartupInfo) -> None:
        from rich.rule import Rule
        from rich.table import Table

        green = COLORS["green"]
        subtext = COLORS["subtext1"]
        overlay = COLORS["overlay1"]
        sapphire = COLORS["sapphire"]
        peach = COLORS["peach"]

        llm_status = getattr(info, "llm_status", "ready" if info.llm_ready else "unavailable")
        if llm_status in {"ready", "checking"}:
            title = f"[bold {COLORS['mauve']}]NOOA ready[/]"
        else:
            title = f"[bold {COLORS['red']}]NOOA — LLM unavailable[/]"
        self._console.console.print(Rule(title=title, style=COLORS["surface2"]))

        table = Table.grid(padding=(0, 2))
        table.add_column(style=f"bold {sapphire}", no_wrap=True)
        table.add_column(style=subtext)

        table.add_row(
            "model",
            f"[bold {green}]{info.short_model}[/] [{overlay}]({info.model})[/]",
        )
        table.add_row("working dir", info.working_dir)

        if info.history_policy and info.history_limit:
            table.add_row(
                "history",
                f"{info.history_policy}  limit {info.history_limit:,} tokens",
            )

        if info.tracing_enabled:
            trace_val = (
                f"[{peach}]{info.trace_dir}[/]"
                if info.trace_dir
                else f"[{peach}]OTLP auto-probe[/]"
            )
            table.add_row("tracing", trace_val)

        if info.custom_agent:
            table.add_row("agent", f"[{green}]{info.custom_agent}[/]")

        keybinds = ("vi mode  " if info.vi_mode else "") + (
            "Tab: complete  |  ↑↓: history  |  Ctrl+U: clear  |  Shift+Enter: newline  |  "
            "Esc: interrupt  |  Ctrl+C ×2: exit"
        )
        table.add_row("keys", f"[{overlay}]{keybinds}[/]")
        table.add_row(
            "commands",
            f"[{sapphire}]/help[/][{overlay}] for all  ·  [/][{sapphire}]/exit[/][{overlay}] to quit[/]",
        )
        table.add_row("!cmd", f"[{overlay}]run a bash command (e.g. !ls -la)[/]")

        self._console.console.print(table)
        self._console.console.print()

    def _render_code_execution(self, execution: CodeExecution) -> None:
        """Render a code execution panel (reasoning + code + output)."""

        from rich.rule import Rule
        from rich.syntax import Syntax
        from rich.text import Text

        try:
            from nooa.context_blocks import ResultStatus as _RS

            _NO_RETURN = None
            try:
                from nooa.events import (
                    _NO_RETURN as _NO_RETURN,  # type: ignore[assignment]
                )
            except ImportError:
                pass
        except ImportError:
            _RS = None
            _NO_RETURN = None

        self._console.stop_spinner()

        elements: list = []

        if execution.code:
            highlight = {execution.highlight_line} if execution.highlight_line is not None else None
            # A snippet (start_line > 1) is numbered from its real file offset, so
            # leading whitespace lines must be preserved or the gutter numbers and
            # highlighted line drift. Only full-cell blocks get the cosmetic strip.
            code = execution.code.rstrip() if execution.start_line > 1 else execution.code.strip()
            elements.append(
                Syntax(
                    code,
                    "python",
                    theme="monokai",
                    line_numbers=True,
                    word_wrap=True,
                    start_line=execution.start_line,
                    highlight_lines=highlight,
                )
            )

        output_parts: list = []
        if execution.stdout:
            output_parts.append(Text(execution.stdout, style=COLORS["text"]))
        if execution.stderr:
            output_parts.append(Text(execution.stderr, style=COLORS["peach"]))
        if execution.error:
            output_parts.append(Text(execution.error, style=f"bold {COLORS['red']}"))
        if execution.value is not None:
            if _NO_RETURN is None or execution.value is not _NO_RETURN:
                val = execution.value if isinstance(execution.value, str) else repr(execution.value)
                output_parts.append(Text(f"=> {val}", style=f"bold {COLORS['green']}"))

        if output_parts:
            elements.append(Text(""))
            elements.append(Rule(style=COLORS["surface1"]))
            for part in output_parts:
                elements.append(part)

        if elements:
            self._console.console.print(
                Rule(title="[python]Python[/python]", style=COLORS["surface2"], align="left")
            )
            for el in elements:
                self._console.console.print(el)
            sys.stdout.flush()
            sys.stderr.flush()

    # ------------------------------------------------------------------
    # Pass-through accessors (used by Session)
    # ------------------------------------------------------------------

    @property
    def console(self):
        return self._console

    @property
    def raw_console(self):
        return self._console.console
