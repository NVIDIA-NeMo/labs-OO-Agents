# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behaviour spec for ``TUIApplication`` (Plan-C, single long-lived app).

Each test pins a behaviour the TUI must preserve. Tests are grouped
by tier of concern:

* Tier 1 — baseline REPL (prompt, enter → agent, buffer clears, exit)
* Tier 2 — input mechanics (Shift+Enter, backspace, history, cursor)
* Tier 3 — commands (/slash dispatch, Tab completion, !bang)
* Tier 4 — type-ahead queue (queued lines, delivery order, cancel)
* Tier 5 — hard cases (Ctrl+C, errors, Rich ANSI, spinner, THE BUG)

Tests read logical state (``app.input_buffer.text``,
``app.status_text()``) via the harness's ``capture_*`` helpers rather
than parsing terminal output — same discipline as the harness canaries.

The ``XFAIL`` mark is still defined below for any future test that
wants to pin an unimplemented behaviour; it's not applied anywhere
right now because every listed behaviour is implemented.
"""

from __future__ import annotations

import asyncio
import io
import threading

import pytest

from .tui_app_harness import (
    FakeAgent,
    MutableRecordingOutput,
    ThreadGate,
    TUIHarness,
    make_local_tui_app,
)

XFAIL = pytest.mark.xfail(strict=True, reason="not yet implemented in Plan-C TUIApplication")

pytestmark = pytest.mark.asyncio


def _last_screen_text(app) -> str:
    screen = app._app.renderer.last_rendered_screen
    if screen is None:
        return ""
    rows: list[str] = []
    for y in range(screen.height):
        row = screen.data_buffer[y]
        end = max(row, default=-1)
        rows.append("".join(row[x].char for x in range(end + 1)))
    return "\n".join(rows)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║ Tier 1 — baseline REPL                                                ║
# ╚══════════════════════════════════════════════════════════════════════╝


async def test_baseline_prompt_visible_at_startup():
    """The app renders the prompt marker and the cursor lives on the input line."""
    async with TUIHarness() as h:
        await h.wait_for(lambda: h.app.prompt_char_visible())  # new API


async def test_runner_activation_failure_still_restores_agent_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = FakeAgent()
    previous_notify = agent.queue_manager._notify_callback
    previous_on_get = agent._user_messages_in._on_get
    app = make_local_tui_app(agent)
    runner = app._test_agent_runner
    assert runner is not None

    def fail_activate(_loop) -> None:
        raise RuntimeError("worker startup failed")

    monkeypatch.setattr(runner, "activate", fail_activate)
    with pytest.raises(RuntimeError, match="worker startup failed"):
        try:
            runner.activate(asyncio.get_running_loop())
        finally:
            app.close_agent_observation()
            await runner.shutdown()

    assert runner.closed
    assert agent.queue_manager._notify_callback is previous_notify
    assert agent._user_messages_in._on_get is previous_on_get


async def test_output_producers_quiesce_before_final_queue_drain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A producer's final output is queued before the consumer is retired."""
    from nooa_cli.tui.host_services import TUIHostServices
    from nooa_cli.tui.tui_application import TUIApplication

    observations: list[tuple[str, bool, bool]] = []
    app: TUIApplication

    async def quiesce() -> None:
        observations.append(
            ("quiesce", app._block_queue is not None, app._consumer_task is not None)
        )
        app.emit_block("final producer output\n")

    app = TUIApplication(
        host_services=TUIHostServices(before_output_drain=quiesce),
        display_mode="native-replay",
    )

    async def exit_immediately(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(app._app, "run_async", exit_immediately)
    monkeypatch.setattr(
        "nooa_cli.tui.stream_forwarder.install_stray_stream_capture",
        lambda *_args, **_kwargs: lambda: None,
    )

    await app.run_async()

    assert observations == [("quiesce", True, True)]
    assert "final producer output" in app.output_buffer.text
    assert app._block_queue is None


async def test_observation_close_failure_does_not_skip_app_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nooa_cli.tui.host_services import TUIHostServices
    from nooa_cli.tui.tui_application import TUIApplication

    phases: list[str] = []

    async def quiesce() -> None:
        phases.append("quiesce")

    app = TUIApplication(
        host_services=TUIHostServices(before_output_drain=quiesce),
        display_mode="native-replay",
    )

    async def exit_immediately(*_args, **_kwargs) -> None:
        return None

    def fail_close() -> None:
        phases.append("observation close")
        raise RuntimeError("observation close failed")

    monkeypatch.setattr(app._app, "run_async", exit_immediately)
    monkeypatch.setattr(app, "close_agent_observation", fail_close)
    monkeypatch.setattr(
        "nooa_cli.tui.stream_forwarder.install_stray_stream_capture",
        lambda *_args, **_kwargs: lambda: phases.append("streams restored"),
    )

    await app.run_async()

    assert phases == ["streams restored", "observation close", "quiesce"]
    assert app._block_queue is None
    assert app._consumer_task is None
    assert app._loop is None


@pytest.mark.parametrize("task_name", ["_clipboard_task", "_link_task"])
async def test_failed_auxiliary_task_does_not_skip_app_teardown(
    monkeypatch: pytest.MonkeyPatch, task_name: str
) -> None:
    from nooa_cli.tui.host_services import TUIHostServices
    from nooa_cli.tui.tui_application import TUIApplication

    phases: list[str] = []

    async def quiesce() -> None:
        phases.append("quiesce")

    async def fail_after_cancellation() -> None:
        try:
            await asyncio.Future()
        except asyncio.CancelledError as exc:
            raise RuntimeError("task teardown failed") from exc

    app = TUIApplication(
        host_services=TUIHostServices(before_output_drain=quiesce),
        display_mode="native-replay",
    )

    async def exit_immediately(*_args, **_kwargs) -> None:
        setattr(app, task_name, asyncio.create_task(fail_after_cancellation()))
        await asyncio.sleep(0)

    monkeypatch.setattr(app._app, "run_async", exit_immediately)
    monkeypatch.setattr(
        "nooa_cli.tui.stream_forwarder.install_stray_stream_capture",
        lambda *_args, **_kwargs: lambda: phases.append("streams restored"),
    )

    await app.run_async()

    assert phases == ["streams restored", "quiesce"]
    assert getattr(app, task_name) is None
    assert app._block_queue is None
    assert app._consumer_task is None
    assert app._loop is None


async def test_pre_run_emit_uses_the_same_normalized_block_as_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nooa_cli.tui.tui_application import TUIApplication

    capture = io.StringIO()
    monkeypatch.setattr("sys.stdout", capture)
    app = TUIApplication(display_mode="native-replay")

    app.emit_block("bootstrap\x1b[2J\r\x07")

    written = capture.getvalue()
    assert "\x1b[2J" not in written
    assert "\x07" not in written
    assert r"bootstrap\x1b[2J\r\x07" in written
    assert app._transcript_blocks[0].source == "bootstrap\x1b[2J\r\x07"


async def test_untagged_transcript_retention_is_bounded_before_resize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nooa_cli.tui.tui_application import TUIApplication

    monkeypatch.setattr("sys.stdout", io.StringIO())
    app = TUIApplication(display_mode="native-replay")

    for index in range(app._untagged_replay_tail + 5):
        app.emit_block(f"block {index}\n")

    assert len(app._transcript_blocks) == app._untagged_replay_tail
    assert app._transcript_blocks[0].source == "block 5\n"


async def test_no_active_identity_preserves_tagged_blocks_but_caps_untagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nooa_cli.tui.tui_application import TUIApplication

    monkeypatch.setattr("sys.stdout", io.StringIO())
    app = TUIApplication()
    app.emit_block("tagged\n", event_id="event-without-manager")
    for index in range(app._untagged_replay_tail + 1):
        app.emit_block(f"untagged {index}\n")

    app._prune_transcript_blocks_for_active_events()

    assert app._transcript_blocks[0].source == "tagged\n"
    assert len(app._transcript_blocks) == app._untagged_replay_tail + 1


async def test_input_composer_has_blank_row_above_and_below_input():
    """The default composer is three rows and expands for multiline input."""
    async with TUIHarness() as h:
        assert h.app is not None
        composer = h.app._input_container
        assert len(composer.children) == 3

        one_line = composer.preferred_height(80, 24)
        # Padding is preferred at ordinary sizes but optional when the terminal
        # is short.  The input row is the only hard minimum.
        assert one_line.min == 1
        assert one_line.preferred == 3

        h.app.input_buffer.text = "first\nsecond"
        two_lines = composer.preferred_height(80, 24)
        assert two_lines.min == 1
        assert two_lines.preferred == 4


async def test_baseline_enter_submits_to_agent():
    async with TUIHarness() as h:
        await h.type_keys("hello world")
        await h.press("enter")
        await h.wait_for(lambda: h.agent.messages_received == ["hello world"])


async def test_opening_system_housekeeping_shares_the_user_turn():
    """An on-consume system request is batched without another agent call."""
    agent = FakeAgent()
    system_messages = agent.queue_manager.queue("system_messages")
    agent._user_messages_in.set_on_get(
        lambda _text: system_messages.put("call self.rename_session")
    )
    notifications: list[dict[str, list]] = []
    original_handle = agent.handle

    async def capture_notification(notification):
        notifications.append(notification)
        return await original_handle(notification)

    agent.handle = capture_notification  # type: ignore[method-assign]

    async with TUIHarness(agent=agent) as h:
        await h.submit_async("hello world")
        await h.wait_for(lambda: len(notifications) == 1)
        assert notifications == [
            {
                "user_messages": ["hello world"],
                "system_messages": ["call self.rename_session"],
            }
        ]


async def test_baseline_agent_message_renders_to_output():
    agent = FakeAgent()

    async with TUIHarness(agent=agent) as h:

        async def step(self: FakeAgent, msg: str):
            h.runner._present(self.render_message("Hi there!"))

        agent.queue(step)
        await h.type_keys("ping")
        await h.press("enter")
        await h.wait_output_contains("Hi there!")


async def test_baseline_input_buffer_cleared_after_submit():
    async with TUIHarness() as h:
        await h.type_keys("something")
        await h.press("enter")
        await h.wait_input_equals("")


async def test_ctrl_u_clears_the_input_buffer():
    async with TUIHarness() as h:
        await h.type_keys("discard this command")
        await h.press("c-u")
        await h.wait_input_equals("")


async def test_prefill_does_not_overwrite_user_typing():
    async with TUIHarness() as h:
        await h.type_keys("already typing")
        await h.wait_input_equals("already typing")
        assert h.app.prefill_input("/mcp approve docs abc123") is False
        assert h.capture_input() == "already typing"


async def test_baseline_ctrl_d_exits():
    # Already pinned by the stub's Ctrl+D binding; keep un-xfailed so a
    # regression flips the test red instead of silently green-to-green.
    async with TUIHarness() as h:
        await h.press("c-d")
        await h.wait_for(lambda: not h.app.is_running)


async def test_baseline_ctrl_c_clears_input_and_requires_confirmation():
    async with TUIHarness() as h:
        await h.type_keys("discard this command")
        await h.wait_input_equals("discard this command")
        await h.press("c-c")
        await h.wait_input_equals("")
        await h.wait_for(lambda: "Press Ctrl+C again to exit" in h.capture_status())
        assert h.app.is_running

        await h.press("c-c")
        await h.wait_for(lambda: h._run_task.done())
        assert not h._run_task.cancelled()
        assert h._run_task.exception() is None


async def test_baseline_typing_disarms_ctrl_c_exit_confirmation():
    async with TUIHarness() as h:
        await h.press("c-c")
        await h.wait_for(lambda: "Press Ctrl+C again to exit" in h.capture_status())

        await h.type_keys("keep working")
        await h.wait_for(lambda: "Press Ctrl+C again to exit" not in h.capture_status())
        assert h.app.is_running


async def test_baseline_command_status_is_dynamic_not_scrollback():
    """Queued/running command state lives in the status area, not transcript."""
    async with TUIHarness() as h:
        h.app.set_command_status("· /mesh-list")
        await h.wait_for(lambda: "/mesh-list" in h.capture_status())
        assert "/mesh-list" not in h.capture_output()
        h.app.set_command_status("")
        await h.wait_for(lambda: "/mesh-list" not in h.capture_status())


async def test_mouse_support_only_enabled_for_subviews():
    """Normal transcript mode must leave native terminal text selection/copy alone."""
    from nooa_cli.tui.subapp import SensitiveTextPromptView

    async with TUIHarness() as h:
        assert bool(h.app._app.mouse_support()) is False
        h.app._active_subview = object()
        assert bool(h.app._app.mouse_support()) is True
        h.app._active_subview = SensitiveTextPromptView("OAuth", "Authorize")
        assert bool(h.app._app.mouse_support()) is False
        h.app._active_subview = None
        assert bool(h.app._app.mouse_support()) is False


async def test_explorer_f2_temporarily_restores_native_terminal_selection():
    from unittest.mock import MagicMock

    from nooa_cli.tui.explorer_base import ExplorerConfig, ExplorerModel, ExplorerView

    view = ExplorerView(
        ExplorerModel([MagicMock(search_text="copy me")]),
        ExplorerConfig(title="Copy Explorer"),
    )
    async with TUIHarness() as h:
        opened = asyncio.create_task(h.app.open_subview(view))
        await h.wait_for(lambda: h.app.active_subview is view)
        assert bool(h.app._app.mouse_support()) is True

        await h.press("f2")
        await h.wait_for(lambda: not bool(h.app._app.mouse_support()))
        assert "F2 mouse/wheel" in view.render(80, 10)

        await h.press("f2")
        await h.wait_for(lambda: bool(h.app._app.mouse_support()))
        await h.press("q")
        await asyncio.wait_for(opened, timeout=1)


async def test_oauth_modal_ctrl_y_copies_full_authorization_url(monkeypatch):
    url = "https://login.example.test/authorize?state=" + "a" * 500
    copied = []

    async with TUIHarness() as h:
        monkeypatch.setattr(
            h.app,
            "_copy_to_clipboard",
            lambda value: copied.append(value) or True,
        )
        prompt = asyncio.create_task(
            h.app.prompt_sensitive("OAuth", "Authorize in your browser.", link_url=url)
        )
        await h.wait_for(lambda: h.app.active_subview is not None)

        await h.press("c-y")
        await h.wait_for(lambda: copied == [url])
        assert "URL copied" in h.app.active_subview.render(80, 10)

        await h.press("escape")
        assert await prompt == ""


async def test_status_text_separates_thinking_and_command_status():
    """Thinking and command statuses render as separated status rows."""
    agent = FakeAgent()
    agent.block.clear()
    async with TUIHarness(agent=agent) as h:
        h.app.submit_message("hold the turn")
        await h.wait_for(h.app.is_thinking)
        h.app.set_command_status("· !find ~/dev/* | grep unified")
        status = h.app.status_text()
        assert "thinking...\n\n· !find" in status
        assert "thinking...\n· !find" not in status
        assert "thinking...   · !find" not in status
        agent.block.set()


async def test_llm_probe_status_is_transient_status_line():
    async with TUIHarness() as h:
        h.app.set_llm_probe_status("probing LLM endpoint...")
        await h.wait_for(lambda: "probing LLM endpoint..." in h.app.status_text())
        assert "probing LLM endpoint..." in h.app.status_text()

        h.app.set_llm_probe_status("")
        assert "probing LLM endpoint..." not in h.app.status_text()


async def test_baseline_command_queue_is_dynamic_not_scrollback():
    """Queued commands live in dynamic UI state, not transcript scrollback."""
    async with TUIHarness() as h:
        h.app.set_command_queue(["/models", "!echo hi"])
        await h.wait_for(lambda: h.app._command_queue_texts == ["/models", "!echo hi"])
        assert "/models" not in h.capture_output()
        assert "!echo hi" not in h.capture_output()
        h.app.set_command_queue([])
        await h.wait_for(lambda: h.app._command_queue_texts == [])


async def test_command_queue_formatted_has_no_trailing_newline():
    """The queue formatter does not append a blank row before the session rule."""
    from prompt_toolkit.formatted_text import fragment_list_to_text

    async with TUIHarness() as h:
        h.app.set_command_queue(["!ls"])
        root = h.app._app.layout.container.get_container()
        queue_container = root.children[1].content
        queue_control = queue_container.content
        assert fragment_list_to_text(queue_control.text()) == "│ 1 command queued\n└─ !ls"


async def test_baseline_command_queue_renders_below_status():
    """The dynamic status row stays directly above the queued-command tree."""
    from prompt_toolkit.formatted_text import fragment_list_to_text

    async with TUIHarness() as h:
        h.app.set_command_status("· !find ~/dev/* | grep unified")
        h.app.set_command_queue(["!ls"])
        root = h.app._app.layout.container.get_container()
        status_control = root.children[0].content
        queue_container = root.children[1].content
        queue_control = queue_container.content
        assert fragment_list_to_text(status_control.text()) == "· !find ~/dev/* | grep unified"
        assert fragment_list_to_text(queue_control.text()).startswith("│ 1 command queued")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║ Tier 2 — input mechanics                                              ║
# ╚══════════════════════════════════════════════════════════════════════╝


async def test_input_shift_enter_inserts_newline():
    async with TUIHarness() as h:
        await h.type_keys("line1")
        await h.press("s-enter")
        await h.type_keys("line2")
        await h.press("enter")
        await h.wait_for(lambda: h.agent.messages_received == ["line1\nline2"])


async def test_input_backspace_deletes_char():
    # Already provable in the stub; keep as a non-xfail regression to
    # guarantee it never breaks as we grow the implementation.
    async with TUIHarness() as h:
        await h.type_keys("abc")
        await h.press("backspace")
        await h.wait_input_equals("ab")


async def test_input_history_up_on_empty_buffer():
    """With no queue, Up on an empty buffer recalls the previous submission."""
    async with TUIHarness() as h:
        await h.type_keys("first")
        await h.press("enter")
        await h.wait_input_equals("")
        await h.press("up")
        await h.wait_input_equals("first")


async def test_ctrl_c_resets_history_navigation_to_most_recent():
    """Clearing a recalled entry makes the next Up start at newest history."""
    async with TUIHarness() as h:
        h.app._history.extend(["first", "second"])
        await h.press("up")
        await h.wait_input_equals("second")

        await h.press("c-c")
        await h.wait_input_equals("")
        await h.press("up")
        await h.wait_input_equals("second")


async def test_input_cursor_home_and_end():
    async with TUIHarness() as h:
        await h.type_keys("abc")
        await h.press("home")
        await h.wait_for(lambda: h.app.input_cursor_position() == 0)
        await h.press("end")
        await h.wait_for(lambda: h.app.input_cursor_position() == 3)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║ Tier 3 — commands                                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝


async def test_commands_slash_dispatches_without_calling_agent():
    """``/help`` routes to the command registry, not ``agent.handle()``."""
    async with TUIHarness() as h:
        await h.type_keys("/help")
        await h.press("enter")
        await h.wait_for(lambda: "/help" in h.app.commands_dispatched())
        assert h.agent.messages_received == []


async def test_commands_slash_fires_on_command_callback():
    """Session wires ``on_command`` to route slash submissions into its
    CommandRegistry; this test pins the hook contract."""

    received: list[str] = []
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from .tui_app_harness import FakeAgent

    agent = FakeAgent()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        app = make_local_tui_app(agent, on_command=received.append)
        import asyncio as _a

        run_task = _a.create_task(app.run_async())
        # wait for ready
        for _ in range(200):
            if app.is_running:
                break
            await _a.sleep(0.01)
        pipe.send_text("/help\r")
        for _ in range(200):
            if received:
                break
            await _a.sleep(0.01)
        assert received == ["/help"]
        app.exit()
        try:
            await _a.wait_for(run_task, 2.0)
        except (TimeoutError, _a.CancelledError):
            run_task.cancel()


async def test_commands_bang_fires_on_bang_callback_with_stripped_body():
    """``on_bang`` receives the body without the leading ``!``."""

    received: list[str] = []
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput

    from .tui_app_harness import FakeAgent

    agent = FakeAgent()
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        app = make_local_tui_app(agent, on_bang=received.append)
        import asyncio as _a

        run_task = _a.create_task(app.run_async())
        for _ in range(200):
            if app.is_running:
                break
            await _a.sleep(0.01)
        pipe.send_text("!echo hi\r")
        for _ in range(200):
            if received:
                break
            await _a.sleep(0.01)
        assert received == ["echo hi"]
        app.exit()
        try:
            await _a.wait_for(run_task, 2.0)
        except (TimeoutError, _a.CancelledError):
            run_task.cancel()


async def test_plain_input_is_blocked_by_actionable_submission_guard():
    """Broken LLM configuration never enters the agent/retry loop."""

    agent = FakeAgent()
    app = make_local_tui_app(
        agent,
        submission_guard=lambda: (
            "Cannot send this message because the configured LLM is unavailable.\n"
            "Use /model <name> to recover."
        ),
    )

    app.submit_message("hi")

    assert agent._user_messages_in.qsize() == 0
    assert "LLM is unavailable" in app.output_buffer.text
    assert "/model <name>" in app.output_buffer.text


async def test_commands_tab_completion_for_slash():
    async with TUIHarness() as h:
        await h.type_keys("/he")
        await h.press("tab")
        # Completion menu should list /help; input buffer either expanded
        # to "/help" or showed the menu — we accept either.
        await h.wait_for(
            lambda: h.capture_input() == "/help" or "/help" in h.app.completion_candidates()
        )


async def test_commands_bang_suspends_app_and_runs_shell():
    """``!echo hi`` uses ``run_in_terminal`` so stdout goes to the real tty."""
    async with TUIHarness() as h:
        await h.type_keys("!echo hi")
        await h.press("enter")
        # Implementation should record that it asked the terminal to
        # suspend + run a shell command. In production that's
        # prompt_toolkit's ``run_in_terminal``; we assert the hook was
        # called with the right command string.
        await h.wait_for(lambda: h.app.last_bang_command() == "echo hi")


# ╔══════════════════════════════════════════════════════════════════════╗
# ║ Tier 4 — type-ahead queue                                             ║
# ╚══════════════════════════════════════════════════════════════════════╝


def _blocking_agent() -> FakeAgent:
    """Agent whose ``handle()`` blocks until we manually set ``block``."""
    agent = FakeAgent()
    agent.block.clear()  # unset → handle() blocks on wait()
    return agent


async def test_queue_displays_pending_message_while_agent_working():
    """Type-ahead remains visibly queued until it enters the transcript."""
    from prompt_toolkit.formatted_text import fragment_list_to_text

    agent = _blocking_agent()
    async with TUIHarness(agent=agent) as h:
        await h.type_keys("trigger")
        await h.press("enter")
        # Agent is now blocked. Retire the accepted trigger's handoff, then the
        # user type-aheads another message that should remain visible.
        await h.wait_for(lambda: h.app.is_thinking())
        h.app.complete_pending_input_handoff("trigger")
        await h.type_keys("queued-msg")
        await h.press("enter")
        await h.wait_for(lambda: h.capture_queued() == ["queued-msg"])

        root = h.app._app.layout.container.get_container()
        queue_container = root.children[1].content
        queue_control = queue_container.content
        assert fragment_list_to_text(queue_control.text()) == "│ queued-msg"


async def test_admitted_input_stays_visible_until_accepted_echo_commits():
    """A fast dequeue cannot create a blank composer→queue→transcript frame."""
    agent = _blocking_agent()
    async with TUIHarness(agent=agent) as h:
        await h.submit_async("trigger")
        await h.wait_for(lambda: h.app.is_thinking())

        # Harness submission bypasses Session's accepted-message UI callback;
        # retire that setup-only handoff before testing the next admission.
        h.app.complete_pending_input_handoff("trigger")

        # Simulate the worker consuming immediately, before Session commits its
        # accepted-message transcript block on the UI owner loop.
        h.app.submit_message("queued-msg")
        assert h.app._pending_input_display() == ["queued-msg"]

        # Dequeue publication may already be empty before the Session callback
        # runs; optimistic queue chrome must still bridge that frame.
        h.runner._on_user_message_get(
            "queued-msg",
            generation=h.runner._binding_generation,
            user_messages=h.runner._user_messages,
            previous=None,
        )
        assert h.runner.state.pending_inputs == ()
        assert h.app._pending_input_display() == ["queued-msg"]

        h.app.complete_pending_input_handoff("queued-msg")
        assert h.app._pending_input_display() == []


async def test_pending_display_preserves_runtime_queue_during_handoff(monkeypatch):
    agent = _blocking_agent()
    async with TUIHarness(agent=agent) as h:
        await h.submit_async("trigger")
        await h.wait_for(lambda: h.app.is_thinking())
        h.app.complete_pending_input_handoff("trigger")

        h.app.submit_message("already-pending")
        h.app.complete_pending_input_handoff("already-pending")
        assert h.runner.state.pending_inputs == ("already-pending",)

        monkeypatch.setattr(h.app._agent_controller, "submit", lambda _text: True)
        h.app.submit_message("newly-admitted")

        assert h.app._pending_input_display() == [
            "already-pending",
            "newly-admitted",
        ]


async def test_submission_exception_retires_only_new_handoff(monkeypatch):
    agent = _blocking_agent()
    async with TUIHarness(agent=agent) as h:
        await h.submit_async("trigger")
        await h.wait_for(lambda: h.app.is_thinking())
        assert [item.text for item in h.app._pending_input_handoff] == ["trigger"]

        def fail_submit(_text: str) -> bool:
            raise RuntimeError("agent transition")

        monkeypatch.setattr(h.app._agent_controller, "submit", fail_submit)
        h.app.submit_message("rejected")

        assert [item.text for item in h.app._pending_input_handoff] == ["trigger"]
        await h.wait_output_contains("Message rejected.")


async def test_coalesced_accepted_echo_retires_all_submission_handoffs():
    agent = _blocking_agent()
    async with TUIHarness(agent=agent) as h:
        await h.submit_async("trigger")
        await h.wait_for(lambda: h.app.is_thinking())

        h.app.complete_pending_input_handoff("trigger")
        h.app.submit_message("one")
        h.app.submit_message("two")
        h.app.submit_message("one\ntwo")

        # FIFO completion must retire the first two submissions, not the later
        # literal multiline duplicate.
        h.app.complete_pending_input_handoff("one\ntwo")
        assert [item.text for item in h.app._pending_input_handoff] == ["one\ntwo"]
        h.app.complete_pending_input_handoff("one\ntwo")
        assert h.app._pending_input_handoff == []


@pytest.mark.parametrize("echo", ["two", "runtime-prefix\ntwo"])
async def test_out_of_order_echo_retires_only_matching_handoff(echo: str) -> None:
    from nooa_cli.tui.tui_application import TUIApplication, _PendingInputHandoff

    app = TUIApplication(display_mode="fullscreen")
    app._pending_input_handoff = [
        _PendingInputHandoff("one"),
        _PendingInputHandoff("two"),
        _PendingInputHandoff("three"),
    ]

    app.complete_pending_input_handoff(echo)

    assert [item.text for item in app._pending_input_handoff] == ["one", "three"]


async def test_coalesced_echo_with_runtime_prefix_retires_tui_handoffs():
    agent = _blocking_agent()
    async with TUIHarness(agent=agent) as h:
        await h.submit_async("trigger")
        await h.wait_for(lambda: h.app.is_thinking())
        h.app.complete_pending_input_handoff("trigger")

        h.runner._user_messages.put("runtime-prefix")
        h.app.submit_message("one")
        h.app.submit_message("two")

        assert h.runner._user_messages.snapshot() == ["runtime-prefix\none\ntwo"]
        assert h.app._pending_input_display() == ["runtime-prefix", "one", "two"]
        h.app.complete_pending_input_handoff("runtime-prefix\none\ntwo")
        assert h.app._pending_input_handoff == []


async def test_withdrawing_coalesced_input_retires_queue_handoff():
    agent = _blocking_agent()
    async with TUIHarness(agent=agent) as h:
        await h.submit_async("trigger")
        await h.wait_for(lambda: h.app.is_thinking())
        h.app.complete_pending_input_handoff("trigger")
        await h.submit_async("one")
        await h.submit_async("two")
        await h.wait_for(lambda: h.app._pending_input_display() == ["one", "two"])

        await h.press("up")
        await h.wait_input_equals("one\ntwo")
        assert h.app._pending_input_display() == []


async def test_clearing_pending_handoffs_removes_old_session_queue_rows() -> None:
    agent = _blocking_agent()
    async with TUIHarness(agent=agent) as h:
        await h.submit_async("trigger")
        await h.wait_for(lambda: h.app.is_thinking())
        h.app.complete_pending_input_handoff("trigger")

        h.app.submit_message("old-session")
        assert h.app._pending_input_handoff

        h.app.clear_pending_input_handoffs()

        assert h.app._pending_input_handoff == []


async def test_queue_multiple_enters_merge_into_one_item():
    """Successive Enters typed while the agent is working compose one
    queued message joined with newlines so the agent isn't asked to
    handle each line of a half-finished thought as its own turn."""
    agent = _blocking_agent()
    async with TUIHarness(agent=agent) as h:
        await h.submit_async("trigger")
        await h.wait_for(lambda: h.app.is_thinking())
        await h.type_keys("one")
        await h.press("enter")
        await h.type_keys("two")
        await h.press("enter")
        # One queue item, two lines.
        await h.wait_for(lambda: h.capture_queued() == ["one\ntwo"])
        # Three lines if the user keeps going.
        await h.type_keys("three")
        await h.press("enter")
        await h.wait_for(lambda: h.capture_queued() == ["one\ntwo\nthree"])


async def test_slash_command_while_agent_working_dispatches_immediately():
    """Forever-loop model: slash commands no longer queue — they fire right away.

    Contrast with the old contract where /exit typed mid-turn waited
    for the agent to finish. Now the agent's handle() runs for the
    whole session, so there's no "next turn" to flush commands into.
    """
    agent = _blocking_agent()
    async with TUIHarness(agent=agent) as h:
        await h.submit_async("trigger")
        await h.wait_for(lambda: h.app.is_thinking())
        await h.type_keys("/exit")
        await h.press("enter")
        await h.wait_for(lambda: h.app.commands_dispatched() == ["/exit"])


async def test_queue_delivered_as_next_turn_when_agent_finishes():
    agent = _blocking_agent()
    async with TUIHarness(agent=agent) as h:
        await h.submit_async("first")
        await h.wait_for(lambda: h.app.is_thinking())
        await h.type_keys("queued")
        await h.press("enter")
        # Let the agent finish → queued message becomes the next handle()
        agent.block.set()
        await h.wait_for(lambda: agent.messages_received == ["first", "queued"])


async def test_interleaved_cmd_msg_msg_commands_fire_immediately():
    """Per-turn dispatcher model: commands dispatch immediately; only
    messages queue (and consecutive ones merge).

    Contrast with the old contract: messages and commands queued
    together, flushed in order after handle() returned. Now
    commands sidestep the agent's input queue and fire as soon as
    typed; consecutive queued messages compose a single multi-line
    item via the dispatcher's submit-merge.
    """
    agent = _blocking_agent()
    commands_seen: list[str] = []
    async with TUIHarness(agent=agent) as h:
        h.app._on_command = commands_seen.append
        await h.submit_async("first")
        await h.wait_for(lambda: h.app.is_thinking())
        # Interleave: message → command → message.
        await h.submit_async("queued-text-1")
        await h.submit_async("/my-cmd")
        await h.submit_async("queued-text-2")
        # Two queued messages around a slash command merge into one
        # multi-line queue item — the slash didn't break the chain
        # because slash commands never touch the user_messages queue.
        await h.wait_for(lambda: h.capture_queued() == ["queued-text-1\nqueued-text-2"])
        # The command fired immediately, without waiting for the agent.
        assert commands_seen == ["/my-cmd"]
        # Let the agent pump the rest.
        agent.block.set()
        await h.wait_for(
            lambda: (
                agent.messages_received
                == [
                    "first",
                    "queued-text-1\nqueued-text-2",
                ]
            )
        )


async def test_queue_esc_soft_cancels_and_delivers_queue():
    agent = _blocking_agent()
    async with TUIHarness(agent=agent) as h:
        await h.submit_async("first")
        await h.wait_for(lambda: h.app.is_thinking())
        await h.type_keys("queued")
        await h.press("enter")
        await h.press("escape")
        await h.wait_for(lambda: agent.messages_received == ["first", "queued"])


async def test_esc_prefers_active_slash_command_over_agent_interrupt():
    agent = _blocking_agent()
    async with TUIHarness(agent=agent) as h:
        await h.submit_async("first")
        await h.wait_for(lambda: h.app.is_thinking())
        cancelled = []
        h.app._on_cancel_command = lambda: cancelled.append(True) or True

        await h.press("escape")

        await h.wait_for(lambda: cancelled == [True])
        assert h.app.is_thinking()
        agent.block.set()


async def test_cancelled_agent_run_async_propagates_to_agent_loop():
    """Slash cancellation reaches async skill work on the separate agent loop."""
    started = ThreadGate()
    cancelled = ThreadGate()

    async def remote_work():
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async with TUIHarness() as h:
        task = asyncio.create_task(h.runner.run_async(remote_work))
        await asyncio.wait_for(started.wait(), timeout=1.0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(cancelled.wait(), timeout=1.0)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║ Tier 5 — hard cases                                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝


async def test_hard_ctrl_c_interrupts_and_clears_buffer():
    """C-c while the agent is working cancels the agent and clears the buffer."""
    agent = _blocking_agent()
    async with TUIHarness(agent=agent) as h:
        await h.submit_async("first")
        await h.wait_for(lambda: h.app.is_thinking())
        await h.type_keys("in-progress")
        await h.wait_input_equals("in-progress")
        await h.press("c-c")
        # Agent gets cancelled and the in-progress input is discarded.
        await h.wait_for(lambda: not h.app.is_thinking())
        assert h.capture_input() == ""


async def test_hard_agent_error_shown_in_output():
    agent = FakeAgent()

    async def step(_self, _msg):
        raise RuntimeError("boom")

    agent.queue(step)
    async with TUIHarness(agent=agent) as h:
        await h.submit_async("go")
        await h.wait_output_contains("boom")


async def test_hard_rich_ansi_preserved_in_output():
    agent = FakeAgent()

    async with TUIHarness(agent=agent) as h:

        async def step(self: FakeAgent, _msg):
            # Rich markdown renders bold ANSI through the explicit presentation sink.
            h.runner._present(self.render_message("**bold**"))

        agent.queue(step)
        await h.submit_async("go")
        await h.wait_for(lambda: "\x1b[" in h.capture_output_ansi())


async def test_hard_spinner_and_session_label_in_status():
    agent = _blocking_agent()
    async with TUIHarness(agent=agent) as h:
        h.app.set_session_label("session-abc")
        await h.submit_async("go")
        await h.wait_for(
            lambda: "thinking" in h.capture_status() and "session-abc" in h.capture_status()
        )


async def test_hard_terminal_resize_does_not_crash():
    output = MutableRecordingOutput()
    async with TUIHarness(output=output) as h:
        await h.resize_from_terminal(40, 20)
        await h.type_keys("still works")
        await h.wait_input_equals("still works")


async def test_hard_keystroke_during_agent_finish_not_lost():
    """THE BUG — Plan-C's reason for being.

    User presses a key in the tiny window while the agent is finishing
    and the app would normally be restarting its prompt. In the one-App
    architecture the input buffer is *always* reading, so the key lands.
    """
    agent = _blocking_agent()
    async with TUIHarness(agent=agent) as h:
        await h.submit_async("first")
        await h.wait_for(lambda: h.app.is_thinking())
        # Simulate the exact race: release the agent and type in the
        # same event-loop tick.
        agent.block.set()
        await h.type_keys("x")
        # The character MUST have landed in the input buffer for the
        # next turn — not dropped, not delivered to a dead typeahead.
        await h.wait_input_equals("x")


async def test_hard_synchronous_commands_dispatch_without_queueing():
    """N synchronous /cmd items typed while agent is working fire immediately.

    Forever-loop contract: commands don't queue behind the agent — the
    agent loop doesn't own them. Each slash command goes straight
    through ``on_command`` as typed; the queue only holds user messages.
    """
    agent = _blocking_agent()
    commands_seen: list[str] = []
    async with TUIHarness(agent=agent) as h:
        h.app._on_command = commands_seen.append
        await h.submit_async("first")  # starts agent; agent blocks
        await h.wait_for(lambda: h.app.is_thinking())
        for i in range(5):
            await h.submit_async(f"/cmd-{i}")
        # Commands all dispatched immediately — none queued.
        assert h.capture_queued() == []
        await h.wait_for(lambda: commands_seen == [f"/cmd-{i}" for i in range(5)])


async def test_hard_sync_on_command_raising_surfaces_to_output():
    """A synchronous ``on_command`` that raises must surface a
    ``[callback error]`` line to scrollback.

    Forever-loop contract: commands no longer queue behind the agent,
    so each failing command reports independently as typed. There's no
    "abort the queue" shortcut any more — there is no command queue.
    """
    agent = _blocking_agent()

    def _raising(_text: str) -> None:
        raise RuntimeError("boom-sync")

    async with TUIHarness(agent=agent) as h:
        h.app._on_command = _raising
        await h.submit_async("first")
        await h.wait_for(lambda: h.app.is_thinking())
        await h.submit_async("/will-fail")
        await h.wait_for(lambda: "[callback error] RuntimeError: boom-sync" in h.capture_output())


async def test_hard_async_on_command_raising_surfaces_to_output() -> None:
    """An async ``on_command`` coroutine that raises must surface the
    error to scrollback via the task's done-callback (not vanish into
    asyncio's default exception handler)."""
    agent = _blocking_agent()

    async def _raising_async(_text: str) -> None:
        raise RuntimeError("boom-async")

    async with TUIHarness(agent=agent) as h:
        h.app._on_command = _raising_async
        await h.submit_async("/go")
        await h.wait_output_contains("[callback error] RuntimeError: boom-async")


async def test_hard_ctrl_c_emits_interrupted_notice_to_scrollback() -> None:
    """Ctrl-C during an agent turn must put a visible ``✗ Interrupted.``
    marker into scrollback so the user knows the cancellation landed —
    not just silently end the turn."""
    agent = _blocking_agent()
    async with TUIHarness(agent=agent) as h:
        await h.submit_async("first")
        await h.wait_for(lambda: h.app.is_thinking())
        await h.press("c-c")
        await h.wait_output_contains("Interrupted")


async def test_hard_sync_blocking_agent_keeps_input_responsive() -> None:
    """A bad synchronous agent step must not starve prompt_toolkit.

    Regression guard for the TUI freezing while the agent does sync work
    on the UI event loop: typing during the blocking turn should still
    update the live input buffer.
    """
    import time

    agent = FakeAgent()

    async def _sync_blocking_handle(notification):
        for items in notification.values():
            for item in items:
                agent.messages_received.append(str(item))
        time.sleep(0.35)
        from nooa_cli.tui.tui_application import DispatcherExit

        raise DispatcherExit()

    agent.handle = _sync_blocking_handle  # type: ignore[method-assign]

    async with TUIHarness(agent=agent) as h:
        await h.submit_async("start")
        await h.wait_for(lambda: h.app.is_thinking())
        await h.type_keys("still responsive")
        await h.wait_input_equals("still responsive", timeout=1.0)


async def test_hard_submit_message_re_entry_pushes_to_queue_not_stomps_task() -> None:
    """Programmatic ``submit_message`` while the forever-loop agent is
    running must push onto the agent's ``user_messages`` queue, not
    replace ``_agent_task``.
    """
    agent = _blocking_agent()
    async with TUIHarness(agent=agent) as h:
        await h.submit_async("first")
        await h.wait_for(lambda: h.app.is_thinking())
        assert h.runner.task is not None
        first_task = h.runner.task

        # Programmatic submission during the blocked turn — pushes onto
        # the hidden InputQueue without replacing _agent_task. The
        # exact tail item depends on whether the dispatcher already
        # consumed "first" (race) — but "second" must be present in
        # whatever the queue currently holds.
        h.app.submit_message("second")
        assert h.runner.task is first_task  # not replaced
        snap = agent._user_messages_in.snapshot()
        assert any("second" in item for item in snap), (
            f"expected 'second' to be queued; snapshot={snap!r}"
        )

        # Pump: release the agent so it drains the queue.
        agent.block.set()
        # "second" must reach the agent — either as its own item (if
        # the dispatcher already consumed "first" before the second
        # submit_message ran) or merged into a "first\nsecond" item
        # (if the merge race went the other way).
        await h.wait_for(lambda: any("second" in m for m in agent.messages_received))


class _DummySubview:
    title = "dummy"

    def __init__(self) -> None:
        self.opened = False
        self.closed = False
        self.keys: list[tuple[str, str]] = []

    def render(self, width: int, height: int) -> str:
        return f"dummy {width}x{height}"

    def handle_key(self, action: str, value: str = "") -> str:
        self.keys.append((action, value))
        if action == "quit":
            return "close"
        return "handled"

    def on_open(self) -> None:
        self.opened = True

    def on_close(self) -> None:
        self.closed = True


async def test_in_app_subview_hosts_keys_without_editing_prompt() -> None:
    view = _DummySubview()
    async with TUIHarness() as h:
        task = asyncio.create_task(h.app.open_subview(view))
        await h.wait_for(lambda: h.app.active_subview is view)

        await h.type_keys("abc")
        await h.wait_for(lambda: ("text", "c") in view.keys)
        assert h.capture_input() == ""
        assert view.opened is True

        await h.press("escape")
        await h.wait_for(lambda: ("escape", "") in view.keys)
        assert h.app.active_subview is view

        await h.press("q")
        await asyncio.wait_for(task, timeout=1)
        assert h.app.active_subview is None
        assert view.closed is True


async def test_prompt_toolkit_resize_polling_disabled_to_avoid_delayed_double_redraw() -> None:
    async with TUIHarness(full_screen=True) as h:
        assert h.app._app.terminal_size_polling_interval is None


async def test_row_only_resize_in_subview_needs_no_transcript_replay() -> None:
    output = MutableRecordingOutput()
    view = _DummySubview()
    async with TUIHarness(full_screen=True, output=output) as h:
        observed = h.app._resize_reflow.observed_size
        assert observed is not None
        task = asyncio.create_task(h.app.open_subview(view))
        await h.wait_for(lambda: h.app.active_subview is view)

        await h.resize_from_terminal(observed[0], max(observed[1] - 1, 1))

        assert h.app._resize_reflow.has_pending_replay is False
        assert h.app._fullscreen_invalidate_count == 0
        await h.press("q")
        await asyncio.wait_for(task, timeout=1)


async def test_production_width_resize_waits_for_subview_to_close() -> None:
    output = MutableRecordingOutput()
    view = _DummySubview()
    async with TUIHarness(full_screen=True, output=output) as h:
        h.app.emit_block("transcript behind subview\n")
        assert h.app._block_queue is not None
        await h.app._block_queue.join()
        task = asyncio.create_task(h.app.open_subview(view))
        await h.wait_for(lambda: h.app.active_subview is view)

        await h.resize_from_terminal(60, 30)
        await h.wait_for(lambda: h.app._resize_replay_timer is None)
        assert h.app._fullscreen_invalidate_count == 0
        assert h.app._resize_reflow.has_pending_replay is True

        await h.press("q")
        await asyncio.wait_for(task, timeout=1)
        await h.wait_for(lambda: h.app._fullscreen_invalidate_count == 1)
        assert h.app._resize_reflow.replayed_width == 60


async def test_fullscreen_mode_rewrites_scrollback_on_resize() -> None:
    """Fullscreen writes transcript once, then resize rewrites the whole scrollback."""
    agent = FakeAgent()

    output = MutableRecordingOutput()
    async with TUIHarness(agent=agent, full_screen=True, output=output) as h:

        async def step(self: FakeAgent, msg: str):
            h.runner._present(self.render_message("A long traceback-ish line in native scrollback"))

        agent.queue(step)
        assert h.app.full_screen is True
        assert h.app._app.full_screen is False
        assert h.app._output_window is None
        await h.submit_async("trigger")
        await h.wait_output_contains("traceback-ish line")
        assert h.app._fullscreen_invalidate_count == 0
        await h.resize_from_terminal(40, 20)
        await h.wait_for(lambda: h.app._fullscreen_invalidate_count == 1)


async def test_fullscreen_streaming_output_rewrites_scrollback_on_resize() -> None:
    output = MutableRecordingOutput()
    async with TUIHarness(full_screen=True, output=output) as h:
        for i in range(25):
            h.app.emit_block(f"chunk {i}\n")
        await h.wait_output_contains("chunk 24")
        assert h.app._fullscreen_invalidate_count == 0
        await h.resize_from_terminal(50, 20)
        await h.wait_for(lambda: h.app._fullscreen_invalidate_count == 1)


async def test_fullscreen_resize_replays_semantic_callbacks_and_clears_scrollback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = MutableRecordingOutput()
    async with TUIHarness(full_screen=True, output=output) as h:
        calls = 0

        def replay() -> str:
            nonlocal calls
            calls += 1
            return f"reflowed width={h.app.output_columns()}\n"

        h.app.emit_block("old width\n", replay=replay)
        await h.wait_output_contains("old width")
        assert h.app._block_queue is not None
        await h.app._block_queue.join()
        capture = io.StringIO()
        monkeypatch.setattr("sys.__stdout__", capture)

        await h.resize_from_terminal(50, 20)
        await h.wait_for(lambda: h.app._fullscreen_invalidate_count == 1)

        rewritten = capture.getvalue()
        assert calls == 1
        assert rewritten.startswith("\x1b[r\x1b[0m\x1b[H\x1b[2J\x1b[3J\x1b[H")
        assert "reflowed width=50" in rewritten
        assert "old width" not in rewritten
        assert h.app._fullscreen_invalidate_count == 1


async def test_semantic_replay_cannot_inject_terminal_commands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = MutableRecordingOutput()
    async with TUIHarness(full_screen=True, output=output) as h:
        h.app.emit_block(
            "safe initial\n",
            replay=lambda: "semantic\x1b[2J\r\x07",
        )
        assert h.app._block_queue is not None
        await h.app._block_queue.join()
        capture = io.StringIO()
        monkeypatch.setattr("sys.__stdout__", capture)

        await h.resize_from_terminal(50, 20)
        await h.wait_for(lambda: h.app._fullscreen_invalidate_count == 1)

        rewritten = capture.getvalue()
        # The one clear sequence belongs to the replay engine.  The callback's
        # erase/CR/BEL are represented as printable diagnostics.
        assert rewritten.count("\x1b[2J") == 1
        assert "semantic\\x1b[2J\\r\\x07" in rewritten
        assert "\x07" not in rewritten


async def test_clear_screen_resets_rewritten_scrollback_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = MutableRecordingOutput()
    async with TUIHarness(full_screen=True, output=output) as h:
        h.app.emit_block("before clear\n")
        await h.wait_output_contains("before clear")
        h.app.clear_transcript()
        h.app.emit_block("after clear\n")
        await h.wait_output_contains("after clear")
        assert h.app._block_queue is not None
        await h.app._block_queue.join()
        capture = io.StringIO()
        monkeypatch.setattr("sys.__stdout__", capture)

        await h.resize_from_terminal(50, 20)
        await h.wait_for(lambda: h.app._fullscreen_invalidate_count == 1)

        replayed = capture.getvalue()
        assert replayed.startswith("\x1b[r\x1b[0m\x1b[H\x1b[2J\x1b[3J\x1b[H")
        assert "after clear" in replayed
        assert "before clear" not in replayed
        assert h.app._fullscreen_invalidate_count == 1


async def test_prompt_toolkit_row_only_resize_redraws_without_rewriting_scrollback() -> None:
    output = MutableRecordingOutput()
    async with TUIHarness(full_screen=True, output=output) as h:
        h.app.emit_block("stable transcript\n")
        await h.wait_output_contains("stable transcript")
        assert h.app._block_queue is not None
        await h.app._block_queue.join()
        observed = h.app._resize_reflow.observed_size
        assert observed is not None
        prior_render_count = h.app._app.render_counter
        output.events.clear()

        await h.resize_from_terminal(observed[0], max(observed[1] - 1, 1))

        assert h.app._app.render_counter > prior_render_count
        assert ("erase_down",) in output.events
        assert h.app._resize_reflow.observed_size == (observed[0], max(observed[1] - 1, 1))
        assert h.app._resize_replay_timer is None
        assert h.app._resize_reflow.has_pending_replay is False
        assert h.app._fullscreen_invalidate_count == 0


async def test_tiny_height_never_uses_window_too_small_and_recovers_once() -> None:
    output = MutableRecordingOutput(columns=80, rows=40)
    async with TUIHarness(full_screen=True, output=output) as h:
        h.app.emit_block("stable transcript\n")
        assert h.app._block_queue is not None
        await h.app._block_queue.join()

        await h.resize_from_terminal(80, 4)

        assert "Window too small" not in _last_screen_text(h.app)
        assert h.app._height_compaction_needs_replay is True
        assert h.app._fullscreen_invalidate_count == 0

        await h.resize_from_terminal(80, 40)
        await h.wait_for(lambda: h.app._fullscreen_invalidate_count == 1)

        assert "Window too small" not in _last_screen_text(h.app)
        assert h.app._height_compaction_needs_replay is False
        assert h.app._resize_reflow.has_pending_replay is False


async def test_oversized_completion_menu_shrinks_instead_of_replacing_layout() -> None:
    from prompt_toolkit.buffer import CompletionState
    from prompt_toolkit.completion import Completion

    output = MutableRecordingOutput(columns=80, rows=6)
    async with TUIHarness(full_screen=True, output=output) as h:
        before = h.app._app.render_counter
        h.app.input_buffer.complete_state = CompletionState(
            h.app.input_buffer.document,
            [Completion(f"/command-{index}") for index in range(20)],
        )
        h.app._app.invalidate()
        await h.wait_for(lambda: h.app._app.render_counter > before)

        assert "Window too small" not in _last_screen_text(h.app)
        assert h.app._fullscreen_invalidate_count == 0
        assert h.app._height_compaction_needs_replay is False

        before = h.app._app.render_counter
        h.app.input_buffer.complete_state = None
        h.app._app.invalidate()
        await h.wait_for(lambda: h.app._app.render_counter > before)
        assert h.app._fullscreen_invalidate_count == 0


async def test_fullscreen_resize_coalesces_transient_sizes_to_final_width() -> None:
    output = MutableRecordingOutput()
    async with TUIHarness(full_screen=True, output=output) as h:
        replay_widths: list[int] = []

        def replay() -> str:
            replay_widths.append(h.app.output_columns())
            return "stable transcript\n"

        h.app.emit_block("stable transcript\n", replay=replay)
        await h.wait_output_contains("stable transcript")

        await h.resize_from_terminal(40, 18)
        await h.resize_from_terminal(50, 20)
        await h.wait_for(lambda: h.app._fullscreen_invalidate_count == 1)

        assert replay_widths == [50]


async def test_production_before_render_path_debounces_to_latest_width() -> None:
    output = MutableRecordingOutput()
    async with TUIHarness(full_screen=True, output=output) as h:
        replay_widths: list[int] = []
        observed = h.app._resize_reflow.observed_size
        assert observed is not None

        def replay() -> str:
            replay_widths.append(h.app.output_columns())
            return "stable transcript\n"

        h.app.emit_block("stable transcript\n", replay=replay)
        assert h.app._block_queue is not None
        await h.app._block_queue.join()

        first = (max(observed[0] - 10, 30), max(observed[1] - 2, 1))
        final = (max(observed[0] - 20, 20), max(observed[1] - 4, 1))
        await h.resize_from_terminal(*first)
        first_timer = h.app._resize_replay_timer
        assert first_timer is not None
        await h.resize_from_terminal(*final)

        assert first_timer.cancelled() is True
        assert h.app._fullscreen_invalidate_count == 0
        await h.wait_for(lambda: h.app._fullscreen_invalidate_count == 1)

        assert replay_widths == [final[0]]


async def test_stale_resize_timer_cannot_enqueue_before_latest_quiet_period() -> None:
    output = MutableRecordingOutput()
    async with TUIHarness(full_screen=True, output=output) as h:
        h.app.emit_block("stable transcript\n")
        assert h.app._block_queue is not None
        await h.app._block_queue.join()

        await h.resize_from_terminal(70, 38)
        stale_generation = h.app._resize_replay_schedule_generation
        await h.resize_from_terminal(60, 36)
        latest_generation = h.app._resize_replay_schedule_generation
        assert latest_generation > stale_generation

        # A cancelled TimerHandle may already be in the loop's ready queue.
        # Its generation check must still make it harmless.
        h.app._start_resize_replay(stale_generation)
        assert h.app._fullscreen_invalidate_count == 0

        await h.wait_for(lambda: h.app._fullscreen_invalidate_count == 1)
        assert h.app._resize_reflow.replayed_width == 60


async def test_slow_output_keeps_at_most_one_resize_barrier_queued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = MutableRecordingOutput()
    first_write_started = asyncio.Event()
    release_first_write = asyncio.Event()
    calls = 0

    async def controlled_run_in_terminal(callback):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_write_started.set()
            await release_first_write.wait()
        return callback()

    monkeypatch.setattr(
        "prompt_toolkit.application.run_in_terminal",
        controlled_run_in_terminal,
    )
    async with TUIHarness(full_screen=True, output=output) as h:
        h.app.emit_block("slow first block\n")
        await asyncio.wait_for(first_write_started.wait(), timeout=1)

        await h.resize_from_terminal(70, 38)
        await h.wait_for(lambda: h.app._queued_resize_replay_generation is not None)
        await h.resize_from_terminal(60, 36)
        await h.wait_for(lambda: h.app._resize_replay_timer is None)

        assert h.app._block_queue is not None
        assert h.app._block_queue.qsize() == 1

        release_first_write.set()
        await h.wait_for(lambda: h.app._fullscreen_invalidate_count == 1)
        await h.app._block_queue.join()
        assert h.app._resize_reflow.replayed_width == 60


async def test_row_change_replaces_a_queued_width_barrier_without_extra_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = MutableRecordingOutput()
    first_write_started = asyncio.Event()
    release_first_write = asyncio.Event()
    calls = 0

    async def controlled_run_in_terminal(callback):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_write_started.set()
            await release_first_write.wait()
        return callback()

    monkeypatch.setattr(
        "prompt_toolkit.application.run_in_terminal",
        controlled_run_in_terminal,
    )
    async with TUIHarness(full_screen=True, output=output) as h:
        h.app.emit_block("slow first block\n")
        await asyncio.wait_for(first_write_started.wait(), timeout=1)

        await h.resize_from_terminal(70, 38)
        await h.wait_for(lambda: h.app._queued_resize_replay_generation is not None)
        first_barrier_generation = h.app._queued_resize_replay_generation
        await h.resize_from_terminal(70, 30)
        assert h.app._resize_reflow.generation > first_barrier_generation

        release_first_write.set()
        await h.wait_for(lambda: h.app._fullscreen_invalidate_count == 1)
        assert h.app._block_queue is not None
        await h.app._block_queue.join()

        assert h.app._resize_reflow.replayed_width == 70
        assert h.app._resize_reflow.observed_size == (70, 30)
        assert h.app._resize_reflow.has_pending_replay is False


async def test_clear_invalidates_a_queued_resize_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = MutableRecordingOutput()
    first_write_started = asyncio.Event()
    release_first_write = asyncio.Event()
    calls = 0

    async def controlled_run_in_terminal(callback):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_write_started.set()
            await release_first_write.wait()
        return callback()

    monkeypatch.setattr(
        "prompt_toolkit.application.run_in_terminal",
        controlled_run_in_terminal,
    )
    capture = io.StringIO()
    monkeypatch.setattr("sys.__stdout__", capture)
    async with TUIHarness(full_screen=True, output=output) as h:
        h.app.emit_block("A cleared transcript\n")
        await asyncio.wait_for(first_write_started.wait(), timeout=1)

        await h.resize_from_terminal(70, 38)
        await h.wait_for(lambda: h.app._queued_resize_replay_generation is not None)
        h.app.clear_transcript()
        h.app.emit_block("B after clear\n")
        release_first_write.set()

        await h.wait_for(lambda: h.app._fullscreen_invalidate_count == 1)
        assert h.app._block_queue is not None
        await h.app._block_queue.join()

        after_last_clear = capture.getvalue().split("\x1b[3J")[-1]
        assert "A cleared transcript" not in after_last_clear
        assert after_last_clear.count("B after clear") == 1
        assert [block.source for block in h.app._transcript_blocks] == ["B after clear\n"]


async def test_clear_invalidates_an_inflight_ordinary_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_started = asyncio.Event()
    release_write = asyncio.Event()

    async def controlled_run_in_terminal(callback):
        write_started.set()
        await release_write.wait()
        return callback()

    monkeypatch.setattr(
        "prompt_toolkit.application.run_in_terminal",
        controlled_run_in_terminal,
    )
    capture = io.StringIO()
    monkeypatch.setattr("sys.__stdout__", capture)
    async with TUIHarness(full_screen=True) as h:
        h.app.emit_block("must stay cleared\n")
        await asyncio.wait_for(write_started.wait(), timeout=1)

        h.app.clear_transcript()
        release_write.set()
        assert h.app._block_queue is not None
        await h.app._block_queue.join()

        assert "must stay cleared" not in capture.getvalue()
        assert "\x1b[3J" in capture.getvalue()
        assert h.app._transcript_blocks == []
        assert h.capture_output() == ""


async def test_clear_physically_purges_a_stale_resize_with_no_later_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = MutableRecordingOutput()
    first_write_started = asyncio.Event()
    release_first_write = asyncio.Event()
    calls = 0

    async def controlled_run_in_terminal(callback):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_write_started.set()
            await release_first_write.wait()
        return callback()

    monkeypatch.setattr(
        "prompt_toolkit.application.run_in_terminal",
        controlled_run_in_terminal,
    )
    capture = io.StringIO()
    monkeypatch.setattr("sys.__stdout__", capture)
    async with TUIHarness(full_screen=True, output=output) as h:
        h.app.emit_block("A before empty clear\n")
        await asyncio.wait_for(first_write_started.wait(), timeout=1)

        await h.resize_from_terminal(70, 38)
        await h.wait_for(lambda: h.app._queued_resize_replay_generation is not None)
        h.app.clear_transcript()
        release_first_write.set()

        await h.wait_for(
            lambda: (
                h.app._resize_replay_timer is None
                and h.app._queued_resize_replay_generation is None
                and not h.app._resize_reflow.has_pending_replay
            )
        )
        assert h.app._block_queue is not None
        await h.app._block_queue.join()

        assert "\x1b[3J" in capture.getvalue()
        assert "A before empty clear" not in capture.getvalue().split("\x1b[3J")[-1]
        assert h.app._transcript_blocks == []


async def test_transient_replay_write_failure_retries_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailFirstReplay(io.StringIO):
        def __init__(self) -> None:
            super().__init__()
            self.failed = False

        def write(self, text: str) -> int:
            if not self.failed and "\x1b[3J" in text:
                self.failed = True
                raise OSError("transient terminal write failure")
            return super().write(text)

    output = MutableRecordingOutput()
    async with TUIHarness(full_screen=True, output=output) as h:
        h.app.emit_block("stable transcript\n")
        assert h.app._block_queue is not None
        await h.app._block_queue.join()
        capture = FailFirstReplay()
        monkeypatch.setattr("sys.__stdout__", capture)

        await h.resize_from_terminal(70, 38)
        await h.wait_for(lambda: h.app._fullscreen_invalidate_count == 1)

        assert capture.failed is True
        assert h.app._resize_reflow.replayed_width == 70
        assert h.app._resize_reflow.has_pending_replay is False


async def test_permanent_replay_failure_stops_after_one_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AlwaysFailReplay(io.StringIO):
        def __init__(self) -> None:
            super().__init__()
            self.attempts = 0

        def write(self, text: str) -> int:
            if "\x1b[3J" in text:
                self.attempts += 1
                raise OSError("permanent terminal write failure")
            return super().write(text)

    output = MutableRecordingOutput()
    async with TUIHarness(full_screen=True, output=output) as h:
        h.app.emit_block("stable transcript\n")
        assert h.app._block_queue is not None
        await h.app._block_queue.join()
        capture = AlwaysFailReplay()
        monkeypatch.setattr("sys.__stdout__", capture)

        await h.resize_from_terminal(70, 38)
        await h.wait_for(
            lambda: (
                capture.attempts == 2
                and h.app._resize_replay_timer is None
                and h.app._queued_resize_replay_generation is None
            )
        )

        for _ in range(5):
            h.app._app.invalidate()
            await asyncio.sleep(0)

        await h.wait_for(
            lambda: (
                h.app._resize_replay_timer is None
                and h.app._queued_resize_replay_generation is None
            )
        )
        await asyncio.sleep(0)

        assert capture.attempts == 2
        assert h.app._resize_reflow.has_pending_replay is True


async def test_ordinary_block_uses_width_at_physical_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = MutableRecordingOutput(columns=20, rows=40)
    write_waiting = asyncio.Event()
    release_write = asyncio.Event()

    async def gated_run_in_terminal(callback):
        write_waiting.set()
        await release_write.wait()
        return callback()

    monkeypatch.setattr(
        "prompt_toolkit.application.run_in_terminal",
        gated_run_in_terminal,
    )
    capture = io.StringIO()
    monkeypatch.setattr("sys.__stdout__", capture)
    async with TUIHarness(full_screen=False, output=output) as h:
        h.app.emit_block("abcdefghijklmnop\n")
        await asyncio.wait_for(write_waiting.wait(), timeout=1)
        output.set_size(6, 40)
        release_write.set()
        assert h.app._block_queue is not None
        await h.app._block_queue.join()

        from nooa_cli.tui.terminal_safety import strip_safe_ansi
        from rich.cells import cell_len

        visible_lines = strip_safe_ansi(capture.getvalue()).splitlines()
        assert "".join(visible_lines) == "abcdefghijklmnop"
        assert all(cell_len(line) <= 5 for line in visible_lines)


async def test_resize_replay_barrier_orders_ui_and_off_thread_output_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = MutableRecordingOutput()
    async with TUIHarness(full_screen=True, output=output) as h:
        h.app.emit_block("before barrier\n")
        assert h.app._block_queue is not None
        await h.app._block_queue.join()
        capture = io.StringIO()
        monkeypatch.setattr("sys.__stdout__", capture)

        observed = h.app._resize_reflow.observed_size
        assert observed is not None
        target = (max(observed[0] - 10, 20), observed[1])
        output.set_size(*target)
        h.app._resize_reflow.observe(target)
        h.app._resize_replay_schedule_generation += 1
        generation = h.app._resize_replay_schedule_generation
        h.app._start_resize_replay(generation)

        h.app.emit_block("after barrier ui\n")
        producer = threading.Thread(
            target=h.app.emit_block,
            args=("after barrier thread\n",),
        )
        producer.start()
        producer.join(timeout=1)
        assert producer.is_alive() is False

        await asyncio.sleep(0)
        await h.app._block_queue.join()

        rendered = capture.getvalue()
        assert rendered.count("before barrier") == 1
        assert rendered.count("after barrier ui") == 1
        assert rendered.count("after barrier thread") == 1
        assert rendered.index("before barrier") < rendered.index("after barrier ui")
        assert rendered.index("after barrier ui") < rendered.index("after barrier thread")


async def test_output_committed_before_resize_barrier_is_in_replayed_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = MutableRecordingOutput()
    async with TUIHarness(full_screen=True, output=output) as h:
        h.app.emit_block("A before resize\n")
        assert h.app._block_queue is not None
        await h.app._block_queue.join()
        capture = io.StringIO()
        monkeypatch.setattr("sys.__stdout__", capture)

        # B occupies the FIFO before the replay marker, so the marker's source
        # snapshot must contain both A and B.
        h.app.emit_block("B before barrier\n")
        observed = h.app._resize_reflow.observed_size
        assert observed is not None
        target = (max(observed[0] - 10, 20), observed[1])
        output.set_size(*target)
        h.app._resize_reflow.observe(target)
        h.app._resize_replay_schedule_generation += 1
        generation = h.app._resize_replay_schedule_generation
        h.app._start_resize_replay(generation)
        await h.app._block_queue.join()

        after_last_clear = capture.getvalue().split("\x1b[3J")[-1]
        assert after_last_clear.count("A before resize") == 1
        assert after_last_clear.count("B before barrier") == 1
        assert after_last_clear.index("A before resize") < after_last_clear.index(
            "B before barrier"
        )


async def test_semantic_replay_emission_is_queued_after_immutable_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = MutableRecordingOutput()
    async with TUIHarness(full_screen=True, output=output) as h:
        callback_count = 0

        def replay() -> str:
            nonlocal callback_count
            callback_count += 1
            if callback_count == 1:
                h.app.emit_block("C emitted during replay\n")
            return "A semantic replay\n"

        h.app.emit_block("A original\n", replay=replay)
        assert h.app._block_queue is not None
        await h.app._block_queue.join()
        capture = io.StringIO()
        monkeypatch.setattr("sys.__stdout__", capture)

        observed = h.app._resize_reflow.observed_size
        assert observed is not None
        await h.resize_from_terminal(max(observed[0] - 10, 20), observed[1])
        await h.wait_for(lambda: h.app._fullscreen_invalidate_count == 1)
        await h.app._block_queue.join()

        after_last_clear = capture.getvalue().split("\x1b[3J")[-1]
        assert after_last_clear.count("A semantic replay") == 1
        assert after_last_clear.count("C emitted during replay") == 1
        assert [block.source for block in h.app._transcript_blocks] == [
            "A original\n",
            "C emitted during replay\n",
        ]


async def test_size_change_during_semantic_render_aborts_stale_replay() -> None:
    output = MutableRecordingOutput()
    async with TUIHarness(full_screen=True, output=output) as h:
        observed = h.app._resize_reflow.observed_size
        assert observed is not None
        narrow = (max(observed[0] - 20, 20), observed[1])
        replay_widths: list[int] = []

        def replay() -> str:
            replay_widths.append(h.app.output_columns())
            if len(replay_widths) == 1:
                # Model a physical tmux pane change while synchronous semantic
                # rendering owns the UI loop. SIGWINCH cannot update Nooa's
                # state until the callback returns, so the final physical-width
                # guard must catch this directly from the Output.
                output.set_size(*observed)
            return "stable transcript\n"

        h.app.emit_block("stable transcript\n", replay=replay)
        assert h.app._block_queue is not None
        await h.app._block_queue.join()

        await h.resize_from_terminal(*narrow)
        await h.wait_for(
            lambda: (
                h.app._resize_replay_timer is None
                and h.app._queued_resize_replay_generation is None
                and not h.app._resize_reflow.has_pending_replay
            )
        )

        assert replay_widths == [narrow[0]]
        assert h.app._fullscreen_invalidate_count == 0
        assert h.app._resize_reflow.replayed_width == observed[0]


async def test_fullscreen_transient_resize_back_to_replayed_width_is_ignored() -> None:
    output = MutableRecordingOutput()
    async with TUIHarness(full_screen=True, output=output) as h:
        h.app.emit_block("stable transcript\n")
        await h.wait_output_contains("stable transcript")
        observed = h.app._resize_reflow.observed_size
        assert observed is not None

        await h.resize_from_terminal(
            max(observed[0] - 20, 20),
            max(observed[1] - 10, 1),
        )
        await h.resize_from_terminal(*observed)
        await h.wait_for(
            lambda: (
                h.app._resize_replay_timer is None
                and h.app._queued_resize_replay_generation is None
                and not h.app._resize_reflow.has_pending_replay
            )
        )
        await asyncio.sleep(0)

        assert h.app._fullscreen_invalidate_count == 0
        assert h.app._resize_replay_timer is None
        assert h.app._resize_reflow.has_pending_replay is False


async def test_pending_resize_is_cancelled_before_terminal_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture = io.StringIO()
    monkeypatch.setattr("sys.__stdout__", capture)
    app = None

    output = MutableRecordingOutput()
    async with TUIHarness(full_screen=True, output=output) as h:
        app = h.app
        h.app.emit_block("stable transcript\n")
        assert h.app._block_queue is not None
        await h.app._block_queue.join()
        observed = h.app._resize_reflow.observed_size
        assert observed is not None
        await h.resize_from_terminal(max(observed[0] - 10, 20), observed[1])

    await asyncio.sleep(0)

    assert app is not None
    assert app._fullscreen_invalidate_count == 0
    assert app._resize_replay_timer is None
    assert "\x1b[3J" not in capture.getvalue()


async def test_inflight_resize_barrier_cannot_clear_after_app_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = MutableRecordingOutput()
    replay_started = asyncio.Event()
    release_replay = asyncio.Event()
    capture = io.StringIO()
    monkeypatch.setattr("sys.__stdout__", capture)

    async def gated_run_in_terminal(callback):
        replay_started.set()
        await release_replay.wait()
        return callback()

    monkeypatch.setattr(
        "prompt_toolkit.application.run_in_terminal",
        gated_run_in_terminal,
    )
    app = None
    async with TUIHarness(full_screen=True, output=output) as h:
        app = h.app
        h.app.emit_block("stable transcript\n")
        await asyncio.wait_for(replay_started.wait(), timeout=1)
        release_replay.set()
        assert h.app._block_queue is not None
        await h.app._block_queue.join()

        replay_started.clear()
        release_replay.clear()
        await h.resize_from_terminal(60, 36)
        await asyncio.wait_for(replay_started.wait(), timeout=1)
        asyncio.get_running_loop().call_later(0.05, release_replay.set)

    assert app is not None
    assert app._fullscreen_invalidate_count == 0
    assert "\x1b[3J" not in capture.getvalue()


async def test_inflight_clear_cannot_purge_terminal_after_app_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clear_started = asyncio.Event()
    release_clear = asyncio.Event()
    capture = io.StringIO()
    monkeypatch.setattr("sys.__stdout__", capture)

    async def gated_run_in_terminal(callback):
        clear_started.set()
        await release_clear.wait()
        return callback()

    monkeypatch.setattr(
        "prompt_toolkit.application.run_in_terminal",
        gated_run_in_terminal,
    )
    async with TUIHarness(full_screen=True) as h:
        h.app.clear_transcript()
        await asyncio.wait_for(clear_started.wait(), timeout=1)
        asyncio.get_running_loop().call_later(0.05, release_clear.set)

    assert "\x1b[3J" not in capture.getvalue()


async def test_non_fullscreen_keeps_native_scrollback_path() -> None:
    async with TUIHarness() as h:
        h.app.emit_block("plain scrollback\n")
        await h.wait_output_contains("plain scrollback")
        assert h.app._fullscreen_invalidate_count == 0


async def test_cancel_status_stays_cancelling_until_agent_cleanup_ack() -> None:
    """Esc keeps a visible cancelling state until the agent turn unwinds."""
    agent = FakeAgent()
    step_started = ThreadGate()
    cleanup_started = ThreadGate()
    release_cleanup = ThreadGate()
    cleanup_done = ThreadGate()

    async def step(_self: FakeAgent, _msg: str) -> None:
        try:
            step_started.set()
            await asyncio.Future()
        except asyncio.CancelledError:
            cleanup_started.set()
            await release_cleanup.wait()
            cleanup_done.set()
            raise

    agent.queue(step)
    async with TUIHarness(agent=agent) as h:
        await h.submit_async("first")
        await h.wait_for(lambda: h.app.is_thinking())
        await asyncio.wait_for(step_started.wait(), timeout=1.0)
        assert h.app.request_agent_cancel(source="escape") is True
        await asyncio.wait_for(cleanup_started.wait(), timeout=1.0)
        assert "cancelling" in h.capture_status()
        assert h.app.is_thinking() is True
        assert cleanup_done.is_set() is False
        release_cleanup.set()
        await asyncio.wait_for(cleanup_done.wait(), timeout=1.0)
        await h.wait_for(lambda: not h.app.is_thinking())
        await h.wait_output_contains("Interrupted")


async def test_cancel_does_not_deliver_queued_message_until_cleanup_ack() -> None:
    """Queued input starts only after cancelled-turn cleanup completes."""
    agent = FakeAgent()
    step_started = ThreadGate()
    cleanup_started = ThreadGate()
    release_cleanup = ThreadGate()
    cleanup_done = ThreadGate()

    async def step(_self: FakeAgent, _msg: str) -> None:
        try:
            step_started.set()
            await asyncio.Future()
        except asyncio.CancelledError:
            cleanup_started.set()
            await release_cleanup.wait()
            cleanup_done.set()
            raise

    agent.queue(step)
    async with TUIHarness(agent=agent) as h:
        await h.submit_async("first")
        await h.wait_for(lambda: h.app.is_thinking())
        await asyncio.wait_for(step_started.wait(), timeout=1.0)
        await h.type_keys("queued")
        await h.press("enter")
        assert h.app.request_agent_cancel(source="escape") is True
        await asyncio.wait_for(cleanup_started.wait(), timeout=1.0)
        assert agent.messages_received == ["first"]
        assert "cancelling" in h.capture_status()
        release_cleanup.set()
        await asyncio.wait_for(cleanup_done.wait(), timeout=1.0)
        await h.wait_for(lambda: agent.messages_received == ["first", "queued"])


async def test_escape_cancels_agent_turn_but_not_spawned_jobs() -> None:
    """Soft Esc cancels the turn only; QueueManager spawned jobs keep running."""
    agent = FakeAgent()
    job_started = ThreadGate()
    handle_holder = {}

    async def background_job():
        job_started.set()
        await asyncio.Future()

    async def step(self: FakeAgent, _msg: str) -> None:
        self.queue_manager.queue("job")
        handle_holder["handle"] = self.queue_manager.spawn(background_job(), channel="job")
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            raise

    agent.queue(step)
    async with TUIHarness(agent=agent) as h:
        await h.submit_async("first")
        await asyncio.wait_for(job_started.wait(), timeout=1.0)
        await h.press("escape")
        await h.wait_for(lambda: not h.app.is_thinking())
        assert handle_holder["handle"].state == "running"
        await agent.queue_manager.shutdown()


async def test_repeated_ctrl_c_exits_while_cancel_is_pending() -> None:
    """First Ctrl-C requests an acknowledged turn cancel; second Ctrl-C exits."""
    agent = FakeAgent()
    step_started = ThreadGate()
    cleanup_started = ThreadGate()

    async def step(_self: FakeAgent, _msg: str) -> None:
        try:
            step_started.set()
            await asyncio.Future()
        except asyncio.CancelledError:
            cleanup_started.set()
            await asyncio.Future()

    agent.queue(step)
    async with TUIHarness(agent=agent) as h:
        await h.submit_async("first")
        await asyncio.wait_for(step_started.wait(), timeout=1.0)
        await h.press("c-c")
        await asyncio.wait_for(cleanup_started.wait(), timeout=1.0)
        assert "cancelling" in h.capture_status()
        assert "Press Ctrl+C again to exit" in h.capture_status()
        await h.press("c-c")
        await h.wait_for(lambda: not h.app.is_running)


async def test_spawned_job_output_restarts_dispatcher_after_cancelled_turn() -> None:
    """Spawned jobs survive Esc, and their later output still wakes the agent."""
    agent = FakeAgent()
    job_started = ThreadGate()
    release_job = ThreadGate()
    cleanup_started = ThreadGate()

    async def background_job():
        job_started.set()
        await release_job.wait()
        return "job-result"

    async def step(self: FakeAgent, _msg: str) -> None:
        self.queue_manager.queue("job")
        self.queue_manager.spawn(background_job(), channel="job")
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cleanup_started.set()
            raise

    agent.queue(step)
    async with TUIHarness(agent=agent) as h:
        await h.submit_async("first")
        await asyncio.wait_for(job_started.wait(), timeout=1.0)
        assert h.app.request_agent_cancel(source="escape") is True
        await asyncio.wait_for(cleanup_started.wait(), timeout=1.0)
        await h.wait_for(lambda: not h.app.is_thinking())
        release_job.set()
        await h.wait_for(lambda: agent.messages_received == ["first", "job-result"])


async def test_session_cancel_agent_turn_is_safe_from_ui_loop() -> None:
    """Session commands can cancel agent-loop dispatcher turns from the UI loop."""
    agent = FakeAgent()
    step_started = ThreadGate()
    cleanup_started = ThreadGate()
    cleanup_done = ThreadGate()

    async def step(_self: FakeAgent, _msg: str) -> None:
        try:
            step_started.set()
            await asyncio.Future()
        except asyncio.CancelledError:
            cleanup_started.set()
            cleanup_done.set()
            raise

    agent.queue(step)
    async with TUIHarness(agent=agent) as h:
        await h.submit_async("first")
        await asyncio.wait_for(step_started.wait(), timeout=1.0)
        assert await h.runner.cancel_for_transition() is True
        await asyncio.wait_for(cleanup_started.wait(), timeout=1.0)
        await asyncio.wait_for(cleanup_done.wait(), timeout=1.0)
        await h.wait_for(lambda: not h.app.is_thinking())


async def test_shutdown_agent_queue_manager_runs_spawn_cleanup_on_agent_loop() -> None:
    """QueueManager.spawn cleanup runs when shutdown happens on the agent loop."""
    agent = FakeAgent()
    job_started = ThreadGate()
    cleanup_started = ThreadGate()
    cleanup_done = ThreadGate()

    async def step(self: FakeAgent, _msg: str) -> None:
        self.queue_manager.queue("job")

        async def background_job():
            try:
                job_started.set()
                await asyncio.Future()
            except asyncio.CancelledError:
                cleanup_started.set()
                await asyncio.sleep(0)
                cleanup_done.set()
                raise

        self.queue_manager.spawn(background_job(), channel="job")

    agent.queue(step)
    async with TUIHarness(agent=agent) as h:
        await h.submit_async("first")
        await asyncio.wait_for(job_started.wait(), timeout=1.0)
        await h.runner.shutdown_queue_manager()
        await asyncio.wait_for(cleanup_started.wait(), timeout=1.0)
        await asyncio.wait_for(cleanup_done.wait(), timeout=1.0)
        assert agent.queue_manager._handles == []
