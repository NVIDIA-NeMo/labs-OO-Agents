# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the TUI-local command runner.

Command lifecycle is separate from agent QueueManager jobs: slash and bang
commands get UI feedback, but they are not agent background jobs.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from nooa_cli.tui.command_runner import CommandRunner
from nooa_cli.tui.commands import CommandResult
from nooa_cli.tui.output import BashOutput, CommandStatus
from nooa_cli.tui.session import Session


@pytest.mark.asyncio
async def test_command_runner_serializes_commands_and_renders_lifecycle():
    """Commands run one at a time while lifecycle and queue status update."""
    rendered = []
    dynamic_statuses = []
    dynamic_queues = []

    async def render(output):
        rendered.append(output)

    runner = CommandRunner(
        render,
        set_dynamic_status=dynamic_statuses.append,
        set_dynamic_queue=dynamic_queues.append,
    )
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    order = []

    async def first():
        order.append("first-start")
        first_entered.set()
        await release_first.wait()
        order.append("first-end")

    async def second():
        order.append("second")

    t1 = asyncio.create_task(runner.run(kind="slash", text="/slow", work=first))
    await first_entered.wait()
    t2 = asyncio.create_task(runner.run(kind="bang", text="!echo hi", work=second))
    await asyncio.sleep(0)

    assert order == ["first-start"]
    assert [r.state for r in runner.records] == ["running", "queued"]

    release_first.set()
    await asyncio.gather(t1, t2)

    assert order == ["first-start", "first-end", "second"]
    statuses = [o for o in rendered if isinstance(o, CommandStatus)]
    assert [(s.text, s.state) for s in statuses] == [("/slow", "done"), ("!echo hi", "done")]
    assert dynamic_statuses[0] == "○ 1 queued"
    assert "· /slow" in dynamic_statuses
    assert ["!echo hi"] in dynamic_queues
    assert "✓ /slow done" not in dynamic_statuses
    assert "· !echo hi" in dynamic_statuses
    assert "✓ !echo hi done" not in dynamic_statuses


@pytest.mark.asyncio
async def test_command_runner_cancels_active_slash_and_continues_queue():
    """Esc-style cancellation is acknowledged before the next command starts."""
    rendered = []

    async def render(output):
        rendered.append(output)

    runner = CommandRunner(render)
    first_started = asyncio.Event()
    first_cancelled = asyncio.Event()
    second_ran = asyncio.Event()

    async def first():
        first_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            first_cancelled.set()
            raise

    async def second():
        second_ran.set()

    first_run = asyncio.create_task(runner.run(kind="slash", text="/slow", work=first))
    await first_started.wait()
    second_run = asyncio.create_task(runner.run(kind="slash", text="/next", work=second))
    await asyncio.sleep(0)

    assert runner.cancel_active(kind="bang") is False
    assert runner.cancel_active(kind="slash") is True
    await asyncio.gather(first_run, second_run)

    assert first_cancelled.is_set()
    assert second_ran.is_set()
    statuses = [output for output in rendered if isinstance(output, CommandStatus)]
    assert [(status.text, status.state) for status in statuses] == [
        ("/slow", "cancelled"),
        ("/next", "done"),
    ]
    assert runner.cancel_active(kind="slash") is False


@pytest.mark.asyncio
async def test_command_runner_surfaces_post_done_render_failures_without_hanging():
    """Post-done render failures become failed statuses without blocking the queue."""
    rendered = []

    async def render(output):
        rendered.append(output)

    runner = CommandRunner(render)

    async def work():
        async def post_done():
            raise RuntimeError("render boom")

        return post_done

    await runner.run(kind="slash", text="/boom", work=work)

    statuses = [o for o in rendered if isinstance(o, CommandStatus)]
    assert [(s.text, s.state) for s in statuses] == [("/boom", "done"), ("/boom", "failed")]
    assert "post-completion render failed" in statuses[-1].error


@pytest.mark.asyncio
async def test_command_runner_animates_running_status():
    """Running commands pulse slowly enough for both spinner frames to appear."""
    dynamic_statuses = []

    async def render(_output):
        return None

    runner = CommandRunner(render, set_dynamic_status=dynamic_statuses.append)

    async def slow():
        await asyncio.sleep(0.6)

    await runner.run(kind="slash", text="/slow", work=slow)

    running_statuses = [s for s in dynamic_statuses if s.endswith("/slow")]
    assert any(s.startswith("·") for s in running_statuses)
    assert any(s.startswith("•") for s in running_statuses)
    assert "✓ /slow done" not in dynamic_statuses


@pytest.mark.asyncio
async def test_command_runner_renders_failed_without_converting_to_agent_job():
    """Command failures render in the TUI lifecycle instead of agent job state."""
    rendered = []
    dynamic_statuses = []
    dynamic_queues = []

    async def render(output):
        rendered.append(output)

    runner = CommandRunner(
        render,
        set_dynamic_status=dynamic_statuses.append,
        set_dynamic_queue=dynamic_queues.append,
    )

    async def boom():
        raise RuntimeError("boom")

    await runner.run(kind="slash", text="/boom", work=boom)

    assert runner.records[-1].state == "failed"
    assert runner.records[-1].error == "RuntimeError: boom"
    statuses = [o for o in rendered if isinstance(o, CommandStatus)]
    assert [(s.text, s.state) for s in statuses] == [("/boom", "failed")]
    assert "boom" in statuses[-1].error
    assert dynamic_statuses == ["○ 1 queued", "· /boom", ""]
    assert dynamic_queues == [["/boom"], [], []]


@pytest.mark.asyncio
async def test_session_bang_routes_through_command_runner_and_renders_status():
    """Bang commands use CommandRunner and render status before shell output."""
    session = Session.__new__(Session)
    session.agent = MagicMock()
    session.agent.shell.cwd = "/tmp"
    session.frontend = SimpleNamespace(render=AsyncMock())
    session._bang_shell = MagicMock()
    session._bang_shell.cwd = "/tmp"
    session._bang_shell.run = AsyncMock(
        return_value=SimpleNamespace(stdout="hello", stderr="", returncode=0)
    )
    app = MagicMock()
    app.set_command_status = MagicMock()
    app.set_command_queue = MagicMock()
    session._app = app
    session._command_runner = None

    await session._on_bang("echo hello")

    session._bang_shell.run.assert_awaited_once_with("echo hello")
    rendered = [call.args[0] for call in session.frontend.render.await_args_list]
    statuses = [o for o in rendered if isinstance(o, CommandStatus)]
    assert [(s.text, s.state) for s in statuses] == [("!echo hello", "done")]
    app.set_command_status.assert_any_call("○ 1 queued")
    app.set_command_queue.assert_any_call(["!echo hello"])
    app.set_command_status.assert_any_call("· !echo hello")
    assert not any(
        call.args == ("✓ !echo hello done",) for call in app.set_command_status.call_args_list
    )
    assert isinstance(rendered[0], CommandStatus)
    assert isinstance(rendered[1], BashOutput)
    assert rendered[1].stdout == "hello"


@pytest.mark.asyncio
async def test_session_slash_routes_through_command_runner_and_preserves_result_handling():
    """Slash commands use CommandRunner without changing result routing."""
    session = Session.__new__(Session)
    session._session_title_requested = False
    session._session_manager = None
    session.frontend = SimpleNamespace(render=AsyncMock())
    session._command_runner = None
    session._emit_text = MagicMock()
    app = MagicMock()
    app._agent_task = None
    app.submit_message = MagicMock()
    app.set_command_status = MagicMock()
    app.set_command_queue = MagicMock()
    session._app = app
    session.agent = MagicMock()

    handler = MagicMock()
    handler.handle = AsyncMock(
        return_value=CommandResult(success=True, agent_message="from command")
    )
    session._handler = handler

    await session._on_command("/skill args")

    handler.handle.assert_awaited_once_with("/skill args", render_outputs=False)
    app.submit_message.assert_called_once_with("from command")
    rendered = [call.args[0] for call in session.frontend.render.await_args_list]
    statuses = [o for o in rendered if isinstance(o, CommandStatus)]
    assert [(s.text, s.state) for s in statuses] == [("/skill args", "done")]
    app.set_command_status.assert_any_call("○ 1 queued")
    app.set_command_queue.assert_any_call(["/skill args"])
    app.set_command_status.assert_any_call("· /skill args")
    assert not any(
        call.args == ("✓ /skill args done",) for call in app.set_command_status.call_args_list
    )


@pytest.mark.asyncio
async def test_session_slash_renders_done_before_user_visible_output():
    """User-visible slash output appears after the durable done marker."""
    session = Session.__new__(Session)
    session._session_title_requested = False
    session._session_manager = None
    session.frontend = SimpleNamespace(render=AsyncMock())
    session._command_runner = None
    session._emit_text = MagicMock()
    app = MagicMock()
    app._agent_task = None
    app.set_command_status = MagicMock()
    app.set_command_queue = MagicMock()
    session._app = app
    session.agent = SimpleNamespace()

    class _SlashResult:
        command = "mcp"
        value = "output"
        output_to_agent = False

        def __str__(self):
            return "command output"

    handler = MagicMock()
    handler.handle = AsyncMock(
        return_value=CommandResult(success=True, slash_result=_SlashResult())
    )
    session._handler = handler

    await session._on_command("/mcp list")

    rendered = [call.args[0] for call in session.frontend.render.await_args_list]
    assert isinstance(rendered[0], CommandStatus)
    assert rendered[0].text == "/mcp list"
    assert rendered[0].state == "done"
    assert rendered[1].content == "command output"


@pytest.mark.asyncio
async def test_session_command_result_prefills_input_after_rendering():
    """Approval-style command results populate, but do not submit, the buffer."""
    from nooa_cli.tui.output import TextOutput

    session = Session.__new__(Session)
    session._session_title_requested = False
    session._session_manager = None
    session.frontend = SimpleNamespace(render=AsyncMock())
    session._command_runner = None
    session._emit_text = MagicMock()
    app = MagicMock()
    app._agent_task = None
    app.set_command_status = MagicMock()
    app.set_command_queue = MagicMock()
    app.prefill_input = MagicMock(return_value=True)
    session._app = app
    session.agent = SimpleNamespace()
    session._handler = SimpleNamespace(
        handle=AsyncMock(
            return_value=CommandResult(
                success=True,
                outputs=[TextOutput("Review this", "warning")],
                input_prefill="/mcp approve docs abc123",
            )
        )
    )

    await session._on_command("/mcp approve docs")

    app.prefill_input.assert_called_once_with("/mcp approve docs abc123")


@pytest.mark.asyncio
async def test_session_exit_renders_done_before_goodbye_then_exits():
    """Exit commands render done, then goodbye output, then close the app."""
    from nooa_cli.tui.output import TextOutput

    session = Session.__new__(Session)
    session._session_title_requested = False
    session._session_manager = None
    session.frontend = SimpleNamespace(render=AsyncMock())
    session._command_runner = None
    session._emit_text = MagicMock()
    app = MagicMock()
    app._agent_task = None
    app.set_command_status = MagicMock()
    app.set_command_queue = MagicMock()
    app.exit = MagicMock()
    session._app = app
    session.agent = SimpleNamespace()

    handler = MagicMock()
    handler.handle = AsyncMock(
        return_value=CommandResult(
            success=True,
            outputs=[TextOutput("Goodbye! Stay vibing.", "status")],
            exit=True,
        )
    )
    session._handler = handler

    await session._on_command("/exit")

    rendered = [call.args[0] for call in session.frontend.render.await_args_list]
    assert isinstance(rendered[0], CommandStatus)
    assert rendered[0].text == "/exit"
    assert rendered[0].state == "done"
    assert rendered[1].content == "Goodbye! Stay vibing."
    app.exit.assert_called_once()


def test_terminal_frontend_renders_done_status_as_check_command_done():
    """Done command statuses render as compact check-marked command text."""
    from nooa_cli.tui.config import Config
    from nooa_cli.tui.frontend import TerminalFrontend

    frontend = TerminalFrontend(Config())
    frontend._console.print_success = MagicMock()

    frontend._render_command_status(CommandStatus(id=7, kind="slash", state="done", text="/models"))

    frontend._console.print_success.assert_called_once_with("[command]/models[/command]")


def test_terminal_frontend_escapes_done_status_command_markup():
    """Done command text is escaped before insertion into Rich markup."""
    from nooa_cli.tui.config import Config
    from nooa_cli.tui.frontend import TerminalFrontend

    frontend = TerminalFrontend(Config())
    frontend._console.print_success = MagicMock()

    frontend._render_command_status(
        CommandStatus(id=8, kind="bang", state="done", text="!echo [/command]")
    )

    frontend._console.print_success.assert_called_once_with(r"[command]!echo \[/command][/command]")


@pytest.mark.asyncio
async def test_command_runner_is_idle_only_after_active_and_queued_work_finishes():
    runner = CommandRunner(AsyncMock())
    entered = asyncio.Event()
    release = asyncio.Event()

    async def first():
        entered.set()
        await release.wait()

    async def second():
        return None

    first_task = asyncio.create_task(runner.run(kind="slash", text="/first", work=first))
    await entered.wait()
    second_task = asyncio.create_task(runner.run(kind="bang", text="!second", work=second))
    await asyncio.sleep(0)

    assert not runner.is_idle
    release.set()
    await asyncio.gather(first_task, second_task)
    assert runner.is_idle
