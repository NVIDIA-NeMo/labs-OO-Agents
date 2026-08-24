# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for plumbing inside ``Session`` and ``TUIConsole``.

These cover seams the post-landing test-coverage review flagged:

- G4: ``_EmitStream`` must coalesce one ``Console.print`` into one
  ``emit_block`` call (not one per chunk).
- G5: ``TUIConsole.replace_console`` swaps the underlying Rich console.
- G6: ``Session._cancel_background_tasks`` cancels and awaits every
  tracked task and leaves the set empty.

The style is direct construction with no prompt_toolkit / no harness —
these components don't need an ``Application`` to be testable.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from nooa_cli.tui.console import TUIConsole
from nooa_cli.tui.session import _EmitStream
from rich.console import Console
from rich.table import Table


def test_session_emission_uses_only_resolved_display_mode_for_replay(monkeypatch) -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock

    import nooa_cli.tui.session as session_module
    from nooa_cli.tui.config import DisplayMode
    from nooa_cli.tui.session import Session

    for mode in DisplayMode:
        app = SimpleNamespace(
            display_mode=mode,
            transcript_columns=lambda: 80,
            emit_block=Mock(),
            complete_pending_input_handoff=Mock(),
            color_depth=8,
        )
        session = Session.__new__(Session)
        session._app = app
        session._renderer = Mock()
        # The deprecated boolean must not override the already-resolved mode.
        session.config = SimpleNamespace(tui=SimpleNamespace(full_screen=True))

        session._emit_text("rich output")
        rich_kwargs = app.emit_block.call_args.kwargs
        assert ("replay" in rich_kwargs) is (mode is DisplayMode.FULLSCREEN)

        app.emit_block.reset_mock()
        monkeypatch.setattr(session_module, "_build_user_bar", lambda *_args: "BAR")
        session._on_user_message_ui("hello")
        bar_kwargs = app.emit_block.call_args.kwargs
        assert ("replay" in bar_kwargs) is (mode is DisplayMode.FULLSCREEN)
        app.complete_pending_input_handoff.assert_called_once_with("hello")


def test_stale_user_message_ui_callback_is_ignored_after_session_swap() -> None:
    from unittest.mock import Mock

    from nooa_cli.tui.session import Session

    session = Session.__new__(Session)
    session._session_generation = 2
    session._app = Mock()
    session._renderer = Mock()

    session._on_user_message_ui("old-session", session_generation=1)

    session._app.emit_block.assert_not_called()
    session._app.complete_pending_input_handoff.assert_not_called()
    session._renderer.reset_turn.assert_not_called()


def test_session_semantic_render_uses_isolated_consoles_across_threads(
    monkeypatch,
) -> None:
    """Resize replay and live rendering must not share mutable Console state."""
    from types import SimpleNamespace

    import nooa_cli.tui.session as session_module
    from nooa_cli.tui.session import Session

    first_inside_print = threading.Event()
    release_first = threading.Event()
    second_finished = threading.Event()

    class BlockingConsole:
        def __init__(self, *, file, **_kwargs) -> None:
            self.file = file

        def print(self, renderable) -> None:
            if renderable == "first":
                first_inside_print.set()
                assert release_first.wait(timeout=1)
            self.file.write(renderable)

    monkeypatch.setattr(session_module, "RichConsole", BlockingConsole)
    session = Session.__new__(Session)
    session._app = SimpleNamespace(transcript_columns=lambda: 79)
    rendered: dict[str, str] = {}

    first = threading.Thread(
        target=lambda: rendered.setdefault("first", session._render_to_ansi("first"))
    )

    def render_second() -> None:
        rendered["second"] = session._render_to_ansi("second")
        second_finished.set()

    second = threading.Thread(target=render_second)
    first.start()
    assert first_inside_print.wait(timeout=1)
    second.start()
    assert second_finished.wait(timeout=1)
    second.join(timeout=1)
    assert second.is_alive() is False
    release_first.set()
    first.join(timeout=1)

    assert first.is_alive() is False
    assert rendered == {"first": "first", "second": "second"}


def test_session_markdown_render_preserves_copyable_line_boundaries() -> None:
    """self.message() prose must not inherit Rich's width padding or hard wraps."""
    from types import SimpleNamespace

    from nooa_cli.tui.session import Session
    from nooa_cli.tui.terminal_safety import strip_safe_ansi
    from rich.markdown import Markdown

    message = (
        "I left a brief closing note that the interaction design and implementation "
        "need a more fundamental rethink. The local diagnostic mouse fix remains "
        "uncommitted and was not pushed."
    )
    session = Session.__new__(Session)
    session._app = SimpleNamespace(transcript_columns=lambda: 60)

    rendered = strip_safe_ansi(session._render_to_ansi(Markdown(message)))

    assert rendered == f"{message}\n"


def test_emit_stream_coalesces_one_print_into_one_emit_call() -> None:
    """Rich writes many small chunks per ``Console.print``. ``_EmitStream``
    must buffer until the trailing ``flush()`` and call ``emit`` exactly
    once — otherwise each styled span pays a ``run_in_terminal`` hop."""
    calls: list[str] = []
    stream = _EmitStream(calls.append)

    table = Table(title="test")
    table.add_column("a")
    table.add_column("b")
    table.add_row("1", "2")
    table.add_row("3", "4")

    # Emulate what the redirected Rich console does: file=stream, print, flush.
    Console(file=stream, force_terminal=True, color_system="256", width=80).print(table)

    assert len(calls) == 1, f"expected 1 emit call, got {len(calls)}"
    # And the coalesced block has the full table content.
    assert "test" in calls[0]
    assert "a" in calls[0] and "b" in calls[0]


def test_emit_stream_marks_agent_message_scope_without_changing_other_emits() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def emit(text: str, **kwargs: object) -> None:
        calls.append((text, kwargs))

    stream = _EmitStream(emit)
    stream.write("activity")
    stream.flush()
    with stream.agent_message():
        stream.write("agent reply")
        stream.flush()

    assert calls == [
        ("activity", {}),
        ("agent reply", {"agent_message": True}),
    ]


def test_emit_stream_preserves_agent_message_marker_across_held_flush() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    def emit(text: str, **kwargs: object) -> None:
        calls.append((text, kwargs))

    stream = _EmitStream(emit)
    with stream.hold():
        with stream.agent_message():
            stream.write("agent reply")
            stream.flush()

    assert calls == [("agent reply", {"agent_message": True})]


def test_emit_stream_empty_flush_is_noop() -> None:
    """Flushing an empty buffer doesn't emit a stray empty block."""
    calls: list[str] = []
    stream = _EmitStream(calls.append)
    stream.flush()
    assert calls == []


def test_emit_stream_multiple_prints_produce_multiple_emits() -> None:
    """Each ``Console.print`` flushes at the end → one emit per print."""
    calls: list[str] = []
    stream = _EmitStream(calls.append)
    c = Console(file=stream, force_terminal=True, color_system="256", width=80)
    c.print("first")
    c.print("second")
    assert len(calls) == 2


def test_terminal_frontend_marks_agent_message_blocks() -> None:
    from types import SimpleNamespace

    from nooa_cli.tui.frontend import TerminalFrontend
    from nooa_cli.tui.output import AgentMessage

    calls: list[tuple[str, dict[str, object]]] = []

    def emit(text: str, **kwargs: object) -> None:
        calls.append((text, kwargs))

    frontend = TerminalFrontend(SimpleNamespace(tui=SimpleNamespace(vi_mode=False)))
    frontend.console.replace_console(
        Console(
            file=_EmitStream(emit),
            force_terminal=True,
            color_system="256",
            width=80,
        )
    )

    frontend._render_agent_message(AgentMessage("hello", show_rule=False))

    assert len(calls) == 1
    assert calls[0][1] == {"agent_message": True}


def test_tui_console_replace_console_swaps_underlying_console() -> None:
    """``replace_console`` is the public seam Session uses to redirect
    frontend output. A replaced console is what ``print_agent`` /
    ``print_table`` / etc. end up writing to."""
    tui = TUIConsole()
    original = tui.console

    # A StringIO-backed console so we can inspect what was written.
    import io as _io

    captured_file = _io.StringIO()
    replacement = Console(file=captured_file, force_terminal=True, color_system="256", width=80)
    tui.replace_console(replacement)

    assert tui.console is replacement
    assert tui.console is not original

    # A call that routes through ``tui.console`` should land in our file.
    tui.console.print("hello-from-replacement")
    assert "hello-from-replacement" in captured_file.getvalue()


async def test_session_cancel_background_tasks_cancels_and_clears() -> None:
    """``_cancel_background_tasks`` cancels every tracked task, awaits
    their cancellation, and empties the set."""
    from nooa_cli.tui.session import Session

    # Minimal Session stub — we only touch ``_background_tasks`` and
    # ``_cancel_background_tasks``. Avoids the real Session's config/agent
    # construction dance.
    session = Session.__new__(Session)
    session._background_tasks = set()

    async def _long_running() -> None:
        await asyncio.sleep(60)

    t1 = asyncio.create_task(_long_running())
    t2 = asyncio.create_task(_long_running())
    session._background_tasks.update({t1, t2})

    await session._cancel_background_tasks()

    assert session._background_tasks == set()
    assert t1.cancelled() and t2.cancelled()


async def test_session_cancel_background_tasks_is_safe_when_empty() -> None:
    """No tracked tasks → the helper returns immediately; no errors."""
    from nooa_cli.tui.session import Session

    session = Session.__new__(Session)
    session._background_tasks = set()

    await session._cancel_background_tasks()
    assert session._background_tasks == set()


def test_print_exit_message_includes_name_and_short_hash(capsys) -> None:
    """When the session has a name and id, the exit message tags both
    in the form ``name [first8charsofhash]`` — same shape as the
    session-label rendered above the input bar."""
    from unittest.mock import Mock

    from nooa_cli.tui.session import Session

    session = Session.__new__(Session)
    session._session_manager = Mock()
    session._session_manager.session_id = "abc1234567890def"
    session._session_manager.name = "my-debug-run"

    session._print_exit_message()
    err = capsys.readouterr().err
    assert "Goodbye! Stay vibing." in err
    assert "my-debug-run [abc12345]" in err


def test_print_exit_message_neutralizes_controls_in_session_name(capsys) -> None:
    from unittest.mock import Mock

    from nooa_cli.tui.session import Session

    session = Session.__new__(Session)
    session._app = None
    session._session_manager = Mock(
        session_id="abc1234567890def",
        name="unsafe\x1b[2J\r\x07",
    )

    session._print_exit_message()

    err = capsys.readouterr().err
    assert "\x1b[2J" not in err
    assert "\x07" not in err
    assert r"unsafe\x1b[2J\r\x07" in err


def test_print_exit_message_short_hash_only_when_name_missing(capsys) -> None:
    """Unnamed sessions still get the bracketed short-hash tag."""
    from unittest.mock import Mock

    from nooa_cli.tui.session import Session

    session = Session.__new__(Session)
    session._session_manager = Mock()
    session._session_manager.session_id = "deadbeefcafebabe"
    session._session_manager.name = None

    session._print_exit_message()
    err = capsys.readouterr().err
    assert "Goodbye! Stay vibing. — [deadbeef]" in err


def test_print_exit_message_no_session_manager(capsys) -> None:
    """Sessions without a session_manager still get the goodbye line,
    just without a tag."""
    from nooa_cli.tui.session import Session

    session = Session.__new__(Session)
    session._session_manager = None

    session._print_exit_message()
    err = capsys.readouterr().err
    assert "Goodbye! Stay vibing." in err
    # Strip ANSI before checking for absence of a bracketed session tag.
    import re

    plain = re.sub(r"\x1b\[[0-9;]*m", "", err)
    assert "[" not in plain  # no bracketed tag


async def test_session_on_user_message_fires_when_dispatcher_dequeues() -> None:
    """Regression guard for the original bug: the user-bar echo wiring
    used to assign to ``self._app.on_user_message``, an attribute that
    nothing reads. The fix installs a hook on
    the channel's ``on_get`` hook, so the echo fires when the
    dispatcher dequeues the message.
    """
    from unittest.mock import Mock, patch

    from nooa_cli.tui.session import Session

    from nooa.runtime.channels import Channel

    session = Session.__new__(Session)
    session._renderer = Mock()
    session._app = Mock()
    session._app.color_depth = 8
    session._session_manager = Mock()
    session._session_manager.user_named = True  # skip auto-name path
    session._session_title_requested = False
    # _colors is a read-only property that reads the global theme; no setup needed.

    queue: Channel[str] = Channel("user_messages", "queue")

    # The real hook does record_user (DB) then routes to _on_user_message_ui (UI).
    def _combined_hook(text: str) -> None:
        session._session_manager.record_user(text)
        session._on_user_message_ui(text)

    queue.set_on_get(_combined_hook)

    queue.put("hi from dispatcher")
    with patch("nooa_cli.tui.session._build_user_bar", return_value="BAR"):
        item = await queue.get()
    assert item == "hi from dispatcher"

    session._session_manager.record_user.assert_called_once_with("hi from dispatcher")
    session._renderer.reset_turn.assert_called_once()
    session._app.emit_block.assert_called_once_with("BAR")


async def test_session_on_user_message_fires_for_mid_turn_dequeue() -> None:
    """Symmetry: ``on_get`` lives on the queue (not the dispatcher loop)
    precisely so the echo fires when the agent drains mid-turn via
    ``await self.user_messages.get()`` — not just when the dispatcher
    dequeues. If a future refactor moves the call into the dispatcher
    loop, the mid-turn drain path silently drops user-bar / SessionUserMessage.
    """
    from unittest.mock import Mock, patch

    from nooa_cli.tui.session import Session

    from nooa.runtime.channels import Channel

    session = Session.__new__(Session)
    session._renderer = Mock()
    session._app = Mock()
    session._app.color_depth = 8
    session._session_manager = Mock()
    session._session_manager.user_named = True
    session._session_title_requested = False
    # _colors is a read-only property that reads the global theme; no setup needed.

    inq: Channel[str] = Channel("user_messages", "queue")

    # The real hook does record_user (DB) then routes to _on_user_message_ui (UI).
    def _combined_hook(text: str) -> None:
        session._session_manager.record_user(text)
        session._on_user_message_ui(text)

    inq.set_on_get(_combined_hook)
    # Mid-turn drain goes through the read facade, the same surface
    # the LLM uses.
    reader = inq.reader

    inq.put("clarification")
    with patch("nooa_cli.tui.session._build_user_bar", return_value="BAR"):
        item = await reader.get()
    assert item == "clarification"

    # Must fire exactly once — symmetric with the dispatcher path.
    session._session_manager.record_user.assert_called_once_with("clarification")
    session._renderer.reset_turn.assert_called_once()
    session._app.emit_block.assert_called_once_with("BAR")


async def test_session_cancel_background_tasks_skips_done_tasks() -> None:
    """Tasks that already finished aren't cancelled (a no-op), but the
    set is still cleared."""
    from nooa_cli.tui.session import Session

    session = Session.__new__(Session)
    session._background_tasks = set()

    async def _noop() -> None:
        return

    t = asyncio.create_task(_noop())
    await t  # task completes before cancel_background_tasks runs
    session._background_tasks.add(t)

    await session._cancel_background_tasks()
    assert session._background_tasks == set()
    assert not t.cancelled()  # done tasks aren't flipped to cancelled


async def test_on_command_clear_cancels_agent_task() -> None:
    """``/clear`` while the agent is mid-turn must cancel ``_agent_task``
    so the old turn doesn't keep running in the stale session."""
    from unittest.mock import AsyncMock, MagicMock

    from nooa_cli.tui.commands import CommandResult
    from nooa_cli.tui.output import ClearScreen
    from nooa_cli.tui.session import Session

    session = Session.__new__(Session)
    session._session_title_requested = True
    session._background_tasks = set()
    session._emit_console = None

    agent = MagicMock()
    agent._storage = MagicMock()
    agent.event_manager = MagicMock()
    agent.event_manager.set_backend = MagicMock()
    agent.queue_manager = MagicMock()
    agent.queue_manager.shutdown = AsyncMock()
    agent.queue_manager._channels = {}
    agent.queue_manager.names = MagicMock(return_value=[])
    session.agent = agent

    registry = MagicMock()
    registry.commands = MagicMock(return_value=[])
    session.registry = registry
    session._session_manager = MagicMock()
    session._session_manager.close = MagicMock()

    # Simulate a running agent task
    async def _fake_agent_work():
        await asyncio.sleep(999)

    fake_task = asyncio.ensure_future(_fake_agent_work())

    app = MagicMock()
    app._agent_task = fake_task

    async def _cancel_agent_turn(*, source: str = "session") -> bool:
        assert source == "session"
        fake_task.cancel()
        try:
            await fake_task
        except asyncio.CancelledError:
            pass
        return True

    async def _cancel_for_transition() -> bool:
        return await _cancel_agent_turn(source="session")

    runner = MagicMock()
    runner.cancel_for_transition = AsyncMock(side_effect=_cancel_for_transition)
    session._local_agent_runner = runner
    session._app = app
    session._emit_text = MagicMock()

    # Make _handler.handle return a result with new_session_manager
    new_sm = MagicMock()
    new_sm.session_id = "new-session-id"
    new_sm._storage = MagicMock()

    fake_result = CommandResult(success=True, outputs=[ClearScreen()])
    fake_result.new_session_manager = new_sm

    handler = MagicMock()
    handler.handle = AsyncMock(return_value=fake_result)
    session._handler = handler

    # Mock _swap_session_manager so it doesn't try real session operations
    session._swap_session_manager = AsyncMock()

    await session._on_command("/clear")

    runner.cancel_for_transition.assert_awaited_once_with()
    assert fake_task.cancelled(), (
        f"_agent_task not cancelled; done={fake_task.done()}, cancelled={fake_task.cancelled()}"
    )
    assert session._session_title_requested is False
    session._swap_session_manager.assert_awaited_once_with(new_sm)


async def test_on_command_clear_without_running_task() -> None:
    """``/clear`` when no agent task is running must not crash."""
    from unittest.mock import AsyncMock, MagicMock

    from nooa_cli.tui.commands import CommandResult
    from nooa_cli.tui.output import ClearScreen
    from nooa_cli.tui.session import Session

    session = Session.__new__(Session)
    session._session_title_requested = True
    session._background_tasks = set()

    agent = MagicMock()
    agent._storage = MagicMock()
    agent.event_manager = MagicMock()
    agent.event_manager.set_backend = MagicMock()
    agent.queue_manager = MagicMock()
    agent.queue_manager.shutdown = AsyncMock()
    agent.queue_manager._channels = {}
    agent.queue_manager.names = MagicMock(return_value=[])
    session.agent = agent

    registry = MagicMock()
    registry.commands = MagicMock(return_value=[])
    session.registry = registry
    session._session_manager = MagicMock()
    session._session_manager.close = MagicMock()

    app = MagicMock()
    app._agent_task = None  # no running task
    runner = MagicMock()
    runner.cancel_for_transition = AsyncMock(return_value=False)
    session._local_agent_runner = runner
    session._app = app
    session._emit_text = MagicMock()

    new_sm = MagicMock()
    new_sm.session_id = "new-session-id"
    new_sm._storage = MagicMock()

    fake_result = CommandResult(success=True, outputs=[ClearScreen()])
    fake_result.new_session_manager = new_sm

    handler = MagicMock()
    handler.handle = AsyncMock(return_value=fake_result)
    session._handler = handler

    # Mock _swap_session_manager so it doesn't try real session operations
    session._swap_session_manager = AsyncMock()

    # Must not raise
    await session._on_command("/clear")
    runner.cancel_for_transition.assert_awaited_once_with()
    assert session._session_title_requested is False


async def test_run_command_runs_post_session_swap_on_agent_loop() -> None:
    """Session-owned post-swap mutations must go through agent_run_async."""
    from unittest.mock import AsyncMock, MagicMock

    from nooa_cli.tui.commands import CommandResult
    from nooa_cli.tui.output import ClearScreen, TextOutput
    from nooa_cli.tui.session import Session

    session = Session.__new__(Session)
    session._session_title_requested = True
    session.agent = MagicMock()
    session.registry = MagicMock()
    session.registry.commands = MagicMock(return_value=[])
    session._session_manager = MagicMock()
    session._emit_text = MagicMock()

    calls: list[str] = []

    async def _agent_run_async(fn):
        calls.append("agent_run_async")
        value = fn()
        if hasattr(value, "__await__"):
            value = await value
        return value

    app = MagicMock()
    runner = MagicMock()
    runner.cancel_for_transition = AsyncMock(return_value=True)
    session._local_agent_runner = runner
    runner.run_async = AsyncMock(side_effect=_agent_run_async)
    session._app = app

    new_sm = MagicMock()
    fake_result = CommandResult(success=True, outputs=[ClearScreen()])
    fake_result.new_session_manager = new_sm

    async def _post_swap():
        calls.append("post_swap")
        return [TextOutput("post swap done", "status")]

    fake_result.post_session_swap = _post_swap
    handler = MagicMock()
    handler.handle = AsyncMock(return_value=fake_result)
    session._handler = handler
    session._swap_session_manager = AsyncMock()

    render_callback = await session._run_command("/clear")

    runner.cancel_for_transition.assert_awaited_once_with()
    session._swap_session_manager.assert_awaited_once_with(new_sm)
    runner.run_async.assert_awaited_once()
    assert calls == ["agent_run_async", "post_swap"]
    assert any(getattr(o, "content", None) == "post swap done" for o in fake_result.outputs)
    assert render_callback is not None


async def test_run_command_marks_session_transition_while_cancelling() -> None:
    """Session commands suppress cancelled-turn UX until the swap completes."""
    from unittest.mock import AsyncMock, MagicMock

    from nooa_cli.tui.commands import CommandResult
    from nooa_cli.tui.output import ClearScreen
    from nooa_cli.tui.session import Session

    session = Session.__new__(Session)
    session._session_title_requested = True
    session.agent = MagicMock()
    session.registry = MagicMock()
    session.registry.commands = MagicMock(return_value=[])
    session._session_manager = MagicMock()
    session._emit_text = MagicMock()

    app = MagicMock()
    app._session_transitioning = False

    async def _cancel_agent_turn() -> bool:
        assert app._session_transitioning is True
        return True

    async def _swap_session_manager(_new_sm) -> None:
        assert app._session_transitioning is True

    runner = MagicMock()
    runner.cancel_for_transition = AsyncMock(side_effect=_cancel_agent_turn)
    session._local_agent_runner = runner
    session._app = app

    new_sm = MagicMock()
    fake_result = CommandResult(success=True, outputs=[ClearScreen()])
    fake_result.new_session_manager = new_sm

    handler = MagicMock()
    handler.handle = AsyncMock(return_value=fake_result)
    session._handler = handler
    session._swap_session_manager = AsyncMock(side_effect=_swap_session_manager)

    await session._run_command("/clear")

    assert app._session_transitioning is False
    runner.cancel_for_transition.assert_awaited_once_with()
    session._swap_session_manager.assert_awaited_once_with(new_sm)


async def test_on_command_slash_result_posts_to_queue_without_double_submit() -> None:
    """A slash command returning a result must be delivered to the agent
    exactly once. ``_on_command`` posts the ``SlashCommandResult`` to the
    ``slash_commands`` queue, which already wakes the dispatcher — it must
    NOT also re-submit the same text as a user message (that delivered the
    command twice: once on each queue)."""
    from unittest.mock import AsyncMock, MagicMock

    from nooa_cli.tui.commands import CommandResult
    from nooa_cli.tui.session import Session

    from nooa.slash_dispatch import SlashCommandResult

    session = Session.__new__(Session)
    session._session_title_requested = False
    session._session_manager = None

    slash_ch = MagicMock()
    agent = MagicMock()
    agent._slash_commands_in = slash_ch
    session.agent = agent
    agent_runner = MagicMock()

    def submit_slash_result(result):
        slash_ch.put(result)
        return True

    agent_runner.submit_slash_result.side_effect = submit_slash_result
    session._local_agent_runner = agent_runner

    app = MagicMock()
    app._agent_task = None
    session._app = app
    session._emit_text = MagicMock()
    frontend = MagicMock()
    frontend.render = AsyncMock()
    session.frontend = frontend

    sr = SlashCommandResult(command="status", args="", value={"ok": True}, text="status: ok")
    fake_result = CommandResult(success=True, outputs=[])
    fake_result.slash_result = sr

    handler = MagicMock()
    handler.handle = AsyncMock(return_value=fake_result)
    session._handler = handler

    await session._on_command("/status")

    slash_ch.put.assert_called_once_with(sr)
    app.submit_message.assert_not_called()


async def test_on_command_slash_result_renders_via_frontend_markdown() -> None:
    """Slash results should render through the frontend, not raw emit_block text.

    This lets Markdown tables from commands like /mesh-list render properly.
    """
    from unittest.mock import AsyncMock, MagicMock

    from nooa_cli.tui.commands import CommandResult
    from nooa_cli.tui.output import AgentMessage
    from nooa_cli.tui.session import Session

    from nooa.slash_dispatch import SlashCommandResult

    session = Session.__new__(Session)
    session._session_title_requested = False
    session._session_manager = None

    slash_ch = MagicMock()
    agent = MagicMock()
    agent._slash_commands_in = slash_ch
    session.agent = agent
    agent_runner = MagicMock()

    def submit_slash_result(result):
        slash_ch.put(result)
        return True

    agent_runner.submit_slash_result.side_effect = submit_slash_result
    session._local_agent_runner = agent_runner

    app = MagicMock()
    app._agent_task = None
    session._app = app
    session._emit_text = MagicMock()

    frontend = MagicMock()
    frontend.render = AsyncMock()
    session.frontend = frontend

    text = "| handle | status |\n|---|---|\n| `alice` | online |"
    sr = SlashCommandResult(command="mesh-list", args="", value=text, text=text)
    fake_result = CommandResult(success=True, outputs=[])
    fake_result.slash_result = sr

    handler = MagicMock()
    handler.handle = AsyncMock(return_value=fake_result)
    session._handler = handler

    await session._on_command("/mesh-list")

    rendered_agent_messages = [
        call.args[0]
        for call in frontend.render.await_args_list
        if isinstance(call.args[0], AgentMessage)
    ]
    assert len(rendered_agent_messages) == 1
    rendered = rendered_agent_messages[0]
    assert rendered.content == text
    assert rendered.show_rule is False
    app.emit_block.assert_not_called()
    slash_ch.put.assert_called_once_with(sr)


async def test_on_command_slash_result_warns_and_drops_when_no_slash_channel() -> None:
    """Strict routing: a slash result travels ONLY the slash_commands
    channel. If ``self.agent`` has no ``_slash_commands_in`` (a non-TUI
    agent, a partially-initialized object, a test double), there is no
    slash-capable destination — the result must be dropped with a loud
    scrollback warning, NEVER funneled through ``submit_message`` /
    ``user_messages`` where it would masquerade as a typed user message.
    """
    from unittest.mock import AsyncMock, MagicMock

    from nooa_cli.tui.commands import CommandResult
    from nooa_cli.tui.session import Session

    from nooa.slash_dispatch import SlashCommandResult

    session = Session.__new__(Session)
    session._session_title_requested = False
    session._session_manager = None

    agent = MagicMock(spec=[])  # no _slash_commands_in attribute
    session.agent = agent
    agent_runner = MagicMock()
    agent_runner.submit_slash_result.return_value = False
    session._local_agent_runner = agent_runner

    app = MagicMock()
    app._agent_task = None
    session._app = app
    session._emit_text = MagicMock()
    frontend = MagicMock()
    frontend.render = AsyncMock()
    session.frontend = frontend

    sr = SlashCommandResult(command="status", args="", value=None, text="status: ok")
    fake_result = CommandResult(success=True, outputs=[])
    fake_result.slash_result = sr

    handler = MagicMock()
    handler.handle = AsyncMock(return_value=fake_result)
    session._handler = handler

    await session._on_command("/status")

    # NEVER routed through the user-message path.
    app.submit_message.assert_not_called()
    # A loud warning is emitted to scrollback naming the dropped command.
    warned = " ".join(str(c.args[0]) for c in session._emit_text.call_args_list if c.args)
    assert "slash_commands" in warned and "status" in warned


async def test_session_run_real_local_composition_submit_output_and_exit(monkeypatch) -> None:
    """The real composition root carries one submit/output through every boundary."""
    from types import SimpleNamespace

    from nooa_cli.tui.config import Config, DisplayMode
    from nooa_cli.tui.session import Session
    from nooa_cli.tui.tui_application import TUIApplication

    from nooa.runtime.channels import QueueManager

    class EventManagerStub:
        def __init__(self) -> None:
            self._handlers: dict[str, list] = {}

        def on(self, name, callback):
            handlers = self._handlers.setdefault(name, [])
            handlers.append(callback)

            def unsubscribe() -> None:
                if callback in handlers:
                    handlers.remove(callback)

            return unsubscribe

        def items(self):
            return ()

    class AgentStub:
        def __init__(self) -> None:
            self.queue_manager = QueueManager()
            self._user_messages_in = self.queue_manager.queue("user_messages")
            self.event_manager = EventManagerStub()
            self._render_message = None
            self.notifications: list[dict[str, list[object]]] = []

        async def handle(self, notification):
            self.notifications.append(notification)
            assert self._render_message is not None
            self._render_message("agent output")
            return SimpleNamespace(kind="WAIT", explanation="")

    class FrontendStub:
        console = None

        def __init__(self) -> None:
            self.closed = False
            self.outputs = []

        async def render(self, output) -> None:
            self.outputs.append(output)

        def close(self) -> None:
            self.closed = True

    class RegistryStub:
        blocking_llm_health = None

        def commands(self):
            return []

    agent = AgentStub()
    frontend = FrontendStub()
    config = Config()
    session = Session(frontend, agent, config, RegistryStub())
    session._dump_exit_diagnostics = lambda: None
    session._restore_terminal = lambda: None
    session._print_exit_message = lambda: None
    observed: dict[str, object] = {}

    async def exercise_real_app(self: TUIApplication) -> None:
        assert self.display_mode is DisplayMode.FULLSCREEN
        self._loop = asyncio.get_running_loop()
        self.observe_agent()
        assert self._agent_controller.state is not None

        self.submit_message("hello through composition")
        for _ in range(200):
            if agent.notifications and "agent output" in self._fullscreen_transcript.text:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("real local composition did not deliver submit/output")

        observed["app"] = self
        assert agent.notifications[0]["user_messages"] == ["hello through composition"]
        assert "agent output" in self._fullscreen_transcript.text

    monkeypatch.setattr(TUIApplication, "run_async", exercise_real_app)

    await session.run()

    app = observed["app"]
    assert isinstance(app, TUIApplication)
    assert app._agent_controller.state is None
    assert frontend.closed


@pytest.mark.parametrize(
    ("application_fails", "expected_error"),
    [
        (True, "application startup failed"),
        (False, "renderer detach failed"),
    ],
)
async def test_session_run_startup_failure_teardown_order(
    monkeypatch, application_fails: bool, expected_error: str
) -> None:
    """Every teardown phase runs; body errors win, otherwise the first cleanup error does."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import nooa_cli.interactive as interactive_module
    import nooa_cli.tui.agent_event_renderer as renderer_module
    import nooa_cli.tui.config as config_module
    import nooa_cli.tui.input_handler as input_module
    import nooa_cli.tui.local_turn_policy as policy_module
    import nooa_cli.tui.tui_application as app_module
    import pytest
    from nooa_cli.tui.session import Session

    order: list[str] = []

    class FakeRunner:
        def __init__(self, agent, **_kwargs):
            self.session = object()

        def set_dispatch_hooks(self, **_kwargs):
            pass

        def set_user_message_accepted_callback(self, _callback):
            pass

        def run(self, fn):
            return fn()

        async def run_async(self, fn):
            value = fn()
            return await value if hasattr(value, "__await__") else value

        def job_snapshots(self):
            return ()

        def pop_last_user_message(self):
            return None

        cancel_requested = False

        def bind(self):
            pass

        def activate(self, _loop):
            pass

        async def shutdown(self):
            order.append("runner shutdown")
            raise RuntimeError("runner shutdown failed")

    class FakePolicy:
        def __init__(self, *_args, **_kwargs):
            pass

        async def before_handle(self, _agent):
            pass

        async def after_handle(self, _agent, _result):
            pass

        def on_notification(self, _notification):
            pass

        def invalidate_keep_going(self):
            pass

        async def shutdown(self):
            order.append("policy shutdown")
            raise RuntimeError("policy shutdown failed")

    class FakeRenderer:
        def __init__(self, **_kwargs):
            pass

        def attach(self):
            pass

        def detach(self):
            order.append("renderer detach")
            raise RuntimeError("renderer detach failed")

    class FakeApp:
        def __init__(self, **_kwargs):
            self._loop = None

        def runtime_state_changed(self, *_args):
            pass

        def runtime_notification_received(self):
            pass

        def runtime_cancelled(self):
            pass

        def invalidate(self):
            pass

        def close_agent_observation(self):
            order.append("presentation detach")
            raise RuntimeError("presentation detach failed")

        async def run_async(self):
            if application_fails:
                raise RuntimeError("application startup failed")

    monkeypatch.setattr(interactive_module, "LocalAgentRunner", FakeRunner)
    monkeypatch.setattr(policy_module, "LocalTurnPolicy", FakePolicy)
    monkeypatch.setattr(renderer_module, "AgentEventRenderer", FakeRenderer)
    monkeypatch.setattr(app_module, "TUIApplication", FakeApp)
    monkeypatch.setattr(input_module, "SlashCommandCompleter", lambda _registry: object())
    monkeypatch.setattr(config_module, "resolve_display_mode", lambda _config: object())

    session = Session.__new__(Session)

    def _frontend_close() -> None:
        order.append("frontend close")
        raise RuntimeError("frontend close failed")

    session.frontend = SimpleNamespace(console=None, render=AsyncMock(), close=_frontend_close)
    session.agent = SimpleNamespace(event_manager=None)
    session.config = SimpleNamespace(tui=SimpleNamespace())
    session.registry = SimpleNamespace(commands=lambda: [])
    session._handler = SimpleNamespace()

    def _session_manager_close() -> None:
        order.append("session manager close")
        raise RuntimeError("session manager close failed")

    session._session_manager = SimpleNamespace(close=_session_manager_close)
    session._initial_outputs = []
    session._pending_code = {}
    session._background_tasks = set()

    async def _cancel_background_tasks() -> None:
        order.append("background tasks")
        raise RuntimeError("background task cancellation failed")

    async def _bang_shell_close() -> None:
        order.append("bang shell close")
        raise RuntimeError("bang shell close failed")

    session._cancel_background_tasks = _cancel_background_tasks
    session._bang_shell = SimpleNamespace(close=_bang_shell_close)
    session._unsub_activity = None
    session._startup_loop = None
    session._prev_exception_handler = None
    session._session_title_requested = False
    session._dump_exit_diagnostics = lambda: order.append("diagnostics")
    session._restore_terminal = lambda: order.append("terminal restore")
    session._print_exit_message = lambda: order.append("exit message")

    with pytest.raises(RuntimeError, match=expected_error):
        await session.run()

    assert order == [
        "renderer detach",
        "presentation detach",
        "policy shutdown",
        "runner shutdown",
        "background tasks",
        "bang shell close",
        "diagnostics",
        "frontend close",
        "session manager close",
        "terminal restore",
        "exit message",
    ]
