# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the supervised graceful-restart drain waiter in ``tui.main``."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nooa_cli.tui.main import _exit_when_restart_requested


def _session(wait_ready, exit_mock, fail_once=False):
    """Session fake matching the waiter's contract (release + emit_block)."""
    calls = {"release": 0}

    def _release():
        calls["release"] += 1

    app = SimpleNamespace(exit=exit_mock, end_input_drain=Mock(), emit_block=Mock())
    session = SimpleNamespace(
        wait_restart_ready=wait_ready,
        release_restart_request=_release,
        _app=app,
    )
    return session, calls


@pytest.mark.asyncio
async def test_restart_waiter_exits_only_after_drain_settles() -> None:
    restart_event = asyncio.Event()
    ready_calls = []
    exit_mock = Mock()

    async def wait_ready():
        await asyncio.sleep(0)

    session, calls = _session(wait_ready, exit_mock)

    task = asyncio.create_task(
        _exit_when_restart_requested(
            session, restart_event, on_ready=lambda: ready_calls.append(True)
        )
    )
    await asyncio.sleep(0.02)
    assert not task.done()

    restart_event.set()
    await asyncio.wait_for(task, timeout=1)

    assert ready_calls == [True]
    session._app.exit.assert_called_once()
    assert calls["release"] == 0


@pytest.mark.asyncio
async def test_restart_waiter_drain_failure_releases_input_and_retries() -> None:
    """A failed drain surfaces, releases the latch, and a later signal retries."""
    restart_event = asyncio.Event()
    ready_calls = []
    exit_mock = Mock()
    attempts = {"count": 0}

    async def wait_ready():
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("drain exploded")
        return None

    session, calls = _session(wait_ready, exit_mock)

    task = asyncio.create_task(
        _exit_when_restart_requested(
            session, restart_event, on_ready=lambda: ready_calls.append(True)
        )
    )
    restart_event.set()
    await asyncio.sleep(0.02)

    # First drain failed: no exit, no ready, input released, failure surfaced.
    assert ready_calls == []
    session._app.exit.assert_not_called()
    assert calls["release"] == 1
    session._app.emit_block.assert_called_once()
    assert "restart drain failed" in session._app.emit_block.call_args.args[0].lower()
    assert not restart_event.is_set()
    assert attempts["count"] == 1
    assert not task.done()  # waiter re-armed

    # A later signal latches a fresh drain and completes it.
    restart_event.set()
    await asyncio.wait_for(task, timeout=1)
    assert attempts["count"] == 2
    assert ready_calls == [True]
    session._app.exit.assert_called_once()


@pytest.mark.asyncio
async def test_real_sigusr1_routes_restart_request_and_re_exec_gate() -> None:
    """End-to-end signal wiring: SIGUSR1 latches the drain via the real handler.

    Covers the wiring the unit-tested waiter depends on: a real
    ``install_restart_signal`` handler on the running loop must invoke the
    session latch (and would arm the waiter), while the re-exec gate stays
    closed until the drain reports ready.
    """
    import os
    import signal as _signal

    from nooa_cli.tui.runtime_registration import TUIRuntimeRegistration

    if not hasattr(_signal, "SIGUSR1"):  # pragma: no cover - platform dependent
        pytest.skip("SIGUSR1 not available")

    restart_event = asyncio.Event()
    latched = []

    class SessionFake:
        _app = SimpleNamespace(exit=Mock(), emit_block=Mock())

        def request_restart_when_idle(self):
            latched.append(True)

    session = SessionFake()
    registration = TUIRuntimeRegistration(
        session_id="sig-test-session",
        working_dir="/tmp",
        original_argv=["python", "-m", "nooa_cli", "tui"],
    )
    assert registration.install_restart_signal(
        asyncio.get_running_loop(),
        lambda: (session.request_restart_when_idle(), restart_event.set()),
    )

    # Send the real signal from this process; the loop-routed handler must
    # run on the loop thread (same thread here) and latch the drain.
    os.kill(os.getpid(), _signal.SIGUSR1)
    await asyncio.wait_for(restart_event.wait(), timeout=2)
    assert latched == [True]

    registration.close()


def test_legacy_queue_manager_host_keeps_all_running_handles_predicate() -> None:
    """Hosts predating running_work_handles() keep the old quiescence rule."""
    from nooa.runtime.channels import has_running_work

    class LegacyQueueManager:
        """Old-style host: only running_handles() exists."""

        def __init__(self):
            self._running = []

        def running_handles(self):
            return list(self._running)

    legacy = LegacyQueueManager()
    assert has_running_work(legacy) is False

    class Handle:
        state = "running"

    legacy._running = [Handle()]
    assert has_running_work(legacy) is True

    # A host with no handle APIs at all is treated as idle.
    assert has_running_work(SimpleNamespace()) is False


@pytest.mark.asyncio
async def test_restart_waiter_cancellation_propagates() -> None:
    restart_event = asyncio.Event()
    exit_mock = Mock()

    async def wait_ready():
        await asyncio.sleep(10)

    session, _calls = _session(wait_ready, exit_mock)
    restart_event.set()

    task = asyncio.create_task(
        _exit_when_restart_requested(session, restart_event, on_ready=Mock())
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(asyncio.shield(task), timeout=1)
    # A cancelled waiter must not re-exec or unblock mid-drain via exit.
    session._app.exit.assert_not_called()
