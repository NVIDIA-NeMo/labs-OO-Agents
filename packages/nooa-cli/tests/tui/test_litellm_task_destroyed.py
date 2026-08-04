# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests: litellm LoggingWorker task-destroyed warnings suppressed.

The litellm GLOBAL_LOGGING_WORKER singleton creates infinite-loop tasks on
whatever event loop is current. When the agent loop is torn down, those
tasks would emit "Task was destroyed but it is pending!" — which pollutes
the TUI scrollback. These tests verify our suppression layers work.
"""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest


class FakeLoggingWorkerTask:
    """Simulates a litellm LoggingWorker._worker_loop pending task."""

    def __repr__(self):
        return "<Task pending name='Task-99' coro=<LoggingWorker._worker_loop() running at logging_worker.py:121>>"


# ─── Test 1: agent loop exception handler suppresses LoggingWorker ────


@pytest.mark.asyncio
async def test_agent_loop_exception_handler_suppresses_logging_worker():
    """The custom exception handler on the agent loop drops 'Task was destroyed'
    for LoggingWorker tasks and passes other messages through."""
    from nooa_cli.tui.tui_application import TUIApplication

    app = TUIApplication(agent=None)

    # Create the agent loop (starts the thread)
    loop = app._ensure_agent_loop()
    assert loop.is_running()

    # Collect messages that go through the exception handler
    forwarded: list[str] = []

    def _capture_default(ctx):
        forwarded.append(ctx.get("message", ""))

    loop.default_exception_handler = _capture_default

    # Simulate a "Task was destroyed" for LoggingWorker — should be suppressed
    loop.call_soon_threadsafe(
        loop.call_exception_handler,
        {
            "message": "Task was destroyed but it is pending!",
            "task": FakeLoggingWorkerTask(),
        },
    )

    # Simulate a different warning — should NOT be suppressed
    loop.call_soon_threadsafe(
        loop.call_exception_handler,
        {
            "message": "Something else went wrong",
        },
    )

    # Give the loop time to process both
    await asyncio.sleep(0.1)

    # Stop the agent loop
    await app._stop_agent_loop()

    # Only the non-LoggingWorker message should have been forwarded
    assert "Something else went wrong" in forwarded
    assert not any("Task was destroyed" in m for m in forwarded)


# ─── Test 2: _loud_handler on UI loop suppresses LoggingWorker ────────


@pytest.mark.asyncio
async def test_loud_handler_suppresses_logging_worker_task_destroyed():
    """Session._loud_handler drops 'Task was destroyed' for LoggingWorker."""
    from nooa_cli.tui.session import Session

    # We need a minimal Session — mock enough to call _loud_handler
    # _loud_handler is a method that takes (self, loop, context)
    # Let's call it directly with the right context
    emitted: list[str] = []

    class FakeApp:
        def emit_block(self, text):
            emitted.append(text)

    class FakeSession:
        _loud_handler = Session._loud_handler
        _app = FakeApp()
        _in_loud_handler = False
        _loud_handler_reentrant = False

    session = FakeSession()

    loop = asyncio.get_running_loop()

    # Should be suppressed
    session._loud_handler(
        loop,
        {
            "message": "Task was destroyed but it is pending!",
            "task": FakeLoggingWorkerTask(),
        },
    )

    # Should NOT be suppressed
    session._loud_handler(
        loop,
        {
            "message": "Some real error occurred",
        },
    )

    # Only the non-suppressed message should produce output
    assert not any("Task was destroyed" in e for e in emitted)
    assert any("Some real error" in e for e in emitted)


# ─── Test 3: _stop_agent_loop calls litellm stop ─────────────────────


@pytest.mark.asyncio
async def test_stop_agent_loop_stops_litellm_worker():
    """_stop_agent_loop gracefully stops litellm's GLOBAL_LOGGING_WORKER."""
    from nooa_cli.tui.tui_application import TUIApplication

    app = TUIApplication(agent=None)
    app._ensure_agent_loop()

    stop_called = []

    async def fake_stop():
        stop_called.append(True)

    with patch(
        "nooa_cli.tui.tui_application._stop_litellm_worker",
        new=fake_stop,
    ):
        await app._stop_agent_loop()

    assert stop_called, "_stop_litellm_worker was not called during _stop_agent_loop"
