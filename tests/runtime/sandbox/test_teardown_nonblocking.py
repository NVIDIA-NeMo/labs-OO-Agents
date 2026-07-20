# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker teardown must not block the event loop.

Regression: ``_terminate_worker`` did ``proc.join(timeout=1.0)`` (x2, up to ~2s)
synchronously on the event-loop thread during every timeout/crash restart and in
``aclose``, stalling the loop and any concurrent sessions. The async teardown
now runs the blocking join off the loop via ``asyncio.to_thread``.

This uses a fake proc whose ``join`` blocks, so no real worker/fork is needed;
the assertion is purely that a concurrent coroutine keeps making progress while
teardown is in flight.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from nooa import Agent
from nooa.runtime.sandbox.config import SandboxConfig
from nooa.runtime.sandbox.executor import SandboxedExecutor
from nooa.unifiedllm import FakeLLMClient


class _Agent(Agent, llm=FakeLLMClient()):
    pass


class _SlowProc:
    """Stands in for a worker whose reap blocks (ignores the first signal)."""

    JOIN_S = 0.5

    def __init__(self) -> None:
        self._alive = True
        self.joins = 0

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:  # fast signal, non-blocking
        pass

    def kill(self) -> None:
        self._alive = False

    def join(self, timeout: float | None = None) -> None:
        self.joins += 1
        time.sleep(self.JOIN_S)  # blocking reap
        self._alive = False


@pytest.mark.asyncio
async def test_async_teardown_does_not_block_the_event_loop():
    ex = SandboxedExecutor(_Agent(), SandboxConfig(require=False), cell_timeout=10.0)
    proc = _SlowProc()
    ex._proc = proc  # type: ignore[assignment]
    ex._conn = None

    ticks = 0

    async def heartbeat() -> None:
        nonlocal ticks
        for _ in range(40):
            await asyncio.sleep(0.01)
            ticks += 1

    hb = asyncio.create_task(heartbeat())
    await asyncio.sleep(0)  # let the heartbeat task start
    t0 = time.monotonic()
    await ex._aterminate_worker()  # blocks a worker THREAD ~0.5s, not the loop
    teardown_s = time.monotonic() - t0
    # Capture progress made WHILE teardown was in flight (before draining hb):
    # with a blocking (sync) join the loop is frozen and this stays ~0; with the
    # off-loop join the heartbeat keeps ticking throughout the ~0.5s reap.
    ticks_during_teardown = ticks
    await hb

    assert proc.joins >= 1  # the reap really ran
    assert teardown_s >= _SlowProc.JOIN_S * 0.8  # and it actually waited on the join
    assert ticks_during_teardown >= 20, (
        f"event loop was blocked during teardown (only {ticks_during_teardown} ticks)"
    )


def test_sync_terminate_still_available_for_close_sync():
    """The sync path (close_sync) keeps a synchronous _terminate_worker."""
    ex = SandboxedExecutor(_Agent(), SandboxConfig(require=False), cell_timeout=10.0)
    proc = _SlowProc()
    ex._proc = proc  # type: ignore[assignment]
    ex._conn = None
    ex._terminate_worker()  # must not raise; reaps synchronously
    assert proc.joins >= 1 and not proc.is_alive()
    assert ex._proc is None
