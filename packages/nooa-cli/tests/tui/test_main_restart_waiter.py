# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the supervised graceful-restart drain waiter in ``tui.main``."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from nooa_cli.tui.main import _exit_when_restart_requested


def _session(wait_ready, exit_mock):
    app = SimpleNamespace(exit=exit_mock, end_input_drain=Mock())
    return SimpleNamespace(wait_restart_ready=wait_ready, _app=app, _app_exit=exit_mock)


@pytest.mark.asyncio
async def test_restart_waiter_exits_only_after_drain_settles() -> None:
    restart_event = asyncio.Event()
    ready_calls = []
    exit_mock = Mock()

    async def wait_ready():
        await asyncio.sleep(0)

    session = _session(wait_ready, exit_mock)

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


@pytest.mark.asyncio
async def test_restart_waiter_drain_failure_releases_input_and_does_not_exit() -> None:
    restart_event = asyncio.Event()
    ready_calls = []
    exit_mock = Mock()

    async def wait_ready():
        raise RuntimeError("drain exploded")

    session = _session(wait_ready, exit_mock)

    restart_event.set()
    await asyncio.wait_for(
        _exit_when_restart_requested(
            session, restart_event, on_ready=lambda: ready_calls.append(True)
        ),
        timeout=1,
    )

    # The failed drain must not re-exec and must not leave input blocked.
    assert ready_calls == []
    session._app.exit.assert_not_called()
    session._app.end_input_drain.assert_called_once()


@pytest.mark.asyncio
async def test_restart_waiter_cancellation_propagates() -> None:
    restart_event = asyncio.Event()
    exit_mock = Mock()

    async def wait_ready():
        await asyncio.sleep(10)

    session = _session(wait_ready, exit_mock)
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
