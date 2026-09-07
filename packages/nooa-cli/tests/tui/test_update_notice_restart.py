# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the observe-only update notice and the in-process /restart command."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from nooa_cli.tui.commands import CommandRegistry, RestartCommand


@pytest.fixture(autouse=True)
def _isolate_settings(tmp_path_factory, monkeypatch):
    """Pin layered settings so Config.load() never reads a real user/project
    settings.yaml (e.g. one enabling tui.update_watch) and perturbs assertions."""
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path_factory.mktemp("settings-user")))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path_factory.mktemp("settings-proj")))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)


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
    calls = {"n": 0}

    def fake_current():
        calls["n"] += 1
        return "bbb"

    session._current_source_revision = fake_current  # type: ignore[method-assign]

    # A tiny interval_s makes the loop fast without patching the
    # process-global asyncio.sleep.
    watch = asyncio.create_task(session._watch_source_updates(interval_s=0.01))
    await asyncio.wait_for(watch, timeout=5)

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

    watch = asyncio.create_task(session._watch_source_updates(interval_s=0.01))
    await asyncio.sleep(0.05)
    assert not watch.done()  # still observing, no notice
    watch.cancel()
    with __import__("contextlib").suppress(asyncio.CancelledError):
        await watch
    set_notice.assert_not_called()
    assert not session._update_notice_shown


# ---------------------------------------------------------------------------
# /restart command
# ---------------------------------------------------------------------------


def _registry(request_restart=None, in_flight=None, reason=None) -> SimpleNamespace:
    """Minimal registry fake exposing the restart wiring."""
    return SimpleNamespace(
        request_restart=request_restart,
        restart_in_flight=in_flight,
        restart_unavailable_reason=reason,
    )


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


@pytest.mark.asyncio
async def test_restart_command_reports_configured_reason_when_unavailable() -> None:
    """When main recorded why /restart is unwired, that reason is surfaced."""
    cmd = _command(
        _registry(request_restart=None, in_flight=None, reason="tui.update_watch is off")
    )

    result = await cmd.execute([])

    assert not result.success
    assert any("tui.update_watch is off" in getattr(o, "content", "") for o in result.outputs)


@pytest.mark.asyncio
async def test_restart_command_tolerates_non_callable_in_flight() -> None:
    """A non-callable restart_in_flight (e.g. None) must not raise."""
    hook = Mock()
    cmd = _command(_registry(request_restart=hook, in_flight=None))

    result = await cmd.execute([])

    assert result.success
    hook.assert_called_once()


def test_restart_command_appears_in_help() -> None:
    assert "/restart" in CommandRegistry.get_help()


# ---------------------------------------------------------------------------
# Opt-in configuration (default off)
# ---------------------------------------------------------------------------


def test_update_watch_defaults_off() -> None:
    """The dev-time update watcher is completely opt-in."""
    from nooa_cli.tui.config import TUIConfig

    cfg = TUIConfig()
    assert cfg.update_watch is False
    assert cfg.update_watch_interval_s == 30.0


def test_update_watch_cli_flag_maps_to_tui_setting() -> None:
    """--update-watch maps to tui.update_watch; absent means not provided."""
    from nooa_cli.tui.config import Config

    cfg = Config.load(update_watch=True)
    assert cfg.tui.update_watch is True

    # Absent flag must not overwrite layered settings (False = not provided).
    cfg2 = Config.load()
    assert cfg2.tui.update_watch is False


def test_update_watch_interval_is_bounded() -> None:
    """A zero/negative interval can never turn the watch into a hot loop."""
    from nooa_cli.tui.config import TUIConfig
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TUIConfig(update_watch_interval_s=0)


def test_update_watch_not_started_when_flag_off() -> None:
    """With tui.update_watch off, Session.run never creates a watcher task."""
    from nooa_cli.tui.session import Session

    session = Session.__new__(Session)
    session.config = SimpleNamespace(
        tui=SimpleNamespace(update_watch=False, update_watch_interval_s=30.0)
    )
    spawned = []
    session._fire_and_forget = lambda coro: spawned.append(coro)  # type: ignore[method-assign]

    session._start_update_watch_if_enabled()

    assert spawned == []


def test_update_watch_started_when_flag_on() -> None:
    """With tui.update_watch on, the watcher is started as a tracked task."""
    from nooa_cli.tui.session import Session

    session = Session.__new__(Session)
    session.config = SimpleNamespace(
        tui=SimpleNamespace(update_watch=True, update_watch_interval_s=30.0)
    )
    spawned = []
    session._fire_and_forget = lambda coro: spawned.append(coro)  # type: ignore[method-assign]

    session._start_update_watch_if_enabled()

    assert len(spawned) == 1
    # Close the un-awaited coroutine to avoid warnings.
    spawned[0].close()
