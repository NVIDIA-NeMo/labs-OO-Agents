# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the observe-only update notice and the in-process /restart command."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from nooa_cli.tui.commands import CommandRegistry, RestartCommand

# ---------------------------------------------------------------------------
# Status-bar notice
# ---------------------------------------------------------------------------


def test_set_status_notice_shows_and_clears_in_status_rows() -> None:
    """The persistent notice renders as its own status row until cleared."""
    from nooa_cli.tui.tui_application import TUIApplication

    app = TUIApplication.__new__(TUIApplication)
    app._persistent_notice = None
    app._session_label = None
    app._transient_status_text = ""
    app._command_status_text = ""
    app._exit_hint_text = ""
    app._llm_probe_status_text = ""
    app._interrupting_agent_turn = False
    app._ctrl_c_exit_armed = False
    app._host_services = SimpleNamespace(auxiliary_status=None)
    app._agent_controller = SimpleNamespace(state=None, failure=None)
    app._app = SimpleNamespace(is_running=False)

    app._pulse_frame = "·"
    app._spinner_frame = "·"
    app._thinking_started_at = None

    app.set_status_notice("Update available — call /restart to reload")
    assert "Update available — call /restart to reload" in app.status_text()

    app.set_status_notice(None)
    assert "Update available" not in app.status_text()


# ---------------------------------------------------------------------------
# Update watch (observe-only)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_watch_shows_notice_once_and_never_restarts() -> None:
    from nooa_cli.tui.session import Session

    session = Session.__new__(Session)
    session._startup_source_revision = "aaa"
    session._update_notice_shown = False
    set_notice = Mock()
    session._app = SimpleNamespace(set_status_notice=set_notice)

    async def _quick_sleep(_delay):
        return None

    original_sleep = asyncio.sleep
    calls = {"n": 0}

    def fake_current():
        calls["n"] += 1
        return "bbb"

    session._current_source_revision = fake_current  # type: ignore[method-assign]

    # Patch asyncio.sleep inside the watch loop to avoid waiting 30s.
    import nooa_cli.tui.session as session_module

    async def instant_sleep(delay, *args, **kwargs):
        return await original_sleep(0)

    session_module.asyncio.sleep = instant_sleep  # type: ignore[assignment]
    try:
        watch = asyncio.create_task(session._watch_source_updates(interval_s=0.01))
        await asyncio.wait_for(watch, timeout=1)
    finally:
        session_module.asyncio.sleep = original_sleep  # type: ignore[assignment]

    assert calls["n"] == 1  # one comparison was enough
    set_notice.assert_called_once_with("Update available — call /restart to reload")
    assert session._update_notice_shown


@pytest.mark.asyncio
async def test_update_watch_without_baseline_is_a_noop() -> None:
    from nooa_cli.tui.session import Session

    session = Session.__new__(Session)
    session._startup_source_revision = None
    # Must return immediately (no baseline => nothing to compare).
    await asyncio.wait_for(session._watch_source_updates(), timeout=1)


@pytest.mark.asyncio
async def test_update_watch_keeps_silent_when_revision_unchanged() -> None:
    from nooa_cli.tui.session import Session

    session = Session.__new__(Session)
    session._startup_source_revision = "aaa"
    session._update_notice_shown = False
    set_notice = Mock()
    session._app = SimpleNamespace(set_status_notice=set_notice)

    session._current_source_revision = lambda: "aaa"  # type: ignore[method-assign]

    original_sleep = asyncio.sleep

    async def instant_sleep(delay, *args, **kwargs):
        return await original_sleep(0)

    import nooa_cli.tui.session as session_module

    session_module.asyncio.sleep = instant_sleep  # type: ignore[assignment]
    try:
        watch = asyncio.create_task(session._watch_source_updates(interval_s=0.01))
        await asyncio.sleep(0.05)
        assert not watch.done()  # still observing, no notice
    finally:
        session_module.asyncio.sleep = original_sleep  # type: ignore[assignment]
        watch.cancel()
        with __import__("contextlib").suppress(asyncio.CancelledError):
            await watch
    set_notice.assert_not_called()
    assert not session._update_notice_shown


# ---------------------------------------------------------------------------
# /restart command
# ---------------------------------------------------------------------------


def _registry(request_restart=None, in_flight=None) -> SimpleNamespace:
    """Minimal registry fake exposing the restart wiring."""
    return SimpleNamespace(request_restart=request_restart, restart_in_flight=in_flight)


def _command(registry) -> RestartCommand:
    return RestartCommand(
        frontend=SimpleNamespace(),
        config=SimpleNamespace(),
        agent=SimpleNamespace(),
        registry=registry,
    )


@pytest.mark.asyncio
async def test_restart_command_latches_the_shared_drain() -> None:
    hook = Mock()
    cmd = _command(_registry(request_restart=hook, in_flight=lambda: False))

    result = await cmd.execute([])

    assert result.success
    hook.assert_called_once()
    assert any("waiting for current work" in getattr(o, "content", "") for o in result.outputs)


@pytest.mark.asyncio
async def test_restart_command_is_idempotent_while_drain_pending() -> None:
    hook = Mock()
    cmd = _command(_registry(request_restart=hook, in_flight=lambda: True))

    result = await cmd.execute([])

    assert result.success
    hook.assert_not_called()
    assert any("already pending" in getattr(o, "content", "") for o in result.outputs)


@pytest.mark.asyncio
async def test_restart_command_reports_unavailable_without_registration() -> None:
    cmd = _command(_registry(request_restart=None, in_flight=lambda: False))

    result = await cmd.execute([])

    assert not result.success
    assert any("unavailable" in getattr(o, "content", "") for o in result.outputs)


def test_restart_command_appears_in_help() -> None:
    assert "/restart" in CommandRegistry.get_help()
