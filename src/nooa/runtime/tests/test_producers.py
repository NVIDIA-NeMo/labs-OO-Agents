# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``producers`` module — async helpers for QueueManager.spawn()."""

from __future__ import annotations

import asyncio

import pytest

from nooa.runtime.channels import QueueManager, StreamEnd
from nooa.runtime.producers import after, cron, monitor, run_job, tail


class FakeEventManager:
    def __init__(self):
        self.events: list = []

    def add(self, event):
        self.events.append(event)


# ---------------------------------------------------------------------------
# after
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_after_returns_none():
    result = await after(0.01)
    assert result is None


@pytest.mark.asyncio
async def test_after_with_spawn():
    em = FakeEventManager()
    qm = QueueManager(event_manager=em)
    ch = qm.queue("wakeup")

    h = qm.spawn(after(0.01), channel="wakeup")
    result = await asyncio.wait_for(ch.get(), timeout=1.0)
    assert result is None
    await asyncio.sleep(0.05)
    assert h.state == "done"
    assert any(isinstance(e, StreamEnd) for e in em.events)


# ---------------------------------------------------------------------------
# cron
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cron_yields_incrementing_ticks():
    gen = cron(0.01)
    ticks = []
    async for tick in gen:
        ticks.append(tick)
        if tick >= 3:
            break
    assert ticks == [1, 2, 3]


@pytest.mark.asyncio
async def test_cron_with_spawn_and_cancel():
    qm = QueueManager()
    ch = qm.queue("ticks")

    h = qm.spawn(cron(0.01), channel="ticks", buffer=10)
    t1 = await asyncio.wait_for(ch.get(), timeout=1.0)
    t2 = await asyncio.wait_for(ch.get(), timeout=1.0)
    assert t1 == 1
    assert t2 == 2
    await h.cancel()
    assert h.state == "cancelled"
    assert h.values[:2] == [1, 2]


# ---------------------------------------------------------------------------
# tail
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tail_yields_new_lines(tmp_path):
    f = tmp_path / "test.log"
    f.write_text("existing\n")

    gen = tail(str(f), poll_interval=0.01)

    async def append_later():
        await asyncio.sleep(0.05)
        with open(str(f), "a") as fh:
            fh.write("line1\n")
            fh.write("line2\n")

    task = asyncio.create_task(append_later())
    lines = []
    async for line in gen:
        lines.append(line)
        if len(lines) >= 2:
            break
    await task
    assert lines == ["line1", "line2"]


# ---------------------------------------------------------------------------
# run_job
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_job_wraps_result():
    async def my_job():
        return 42

    result = await run_job(my_job(), "test-123")
    assert result == {"job_id": "test-123", "result": 42}


@pytest.mark.asyncio
async def test_run_job_with_spawn():
    qm = QueueManager()
    ch = qm.queue("jobs")

    async def compute():
        await asyncio.sleep(0.01)
        return "done"

    h = qm.spawn(run_job(compute(), "j1"), channel="jobs")
    result = await asyncio.wait_for(ch.get(), timeout=1.0)
    assert result == {"job_id": "j1", "result": "done"}
    await asyncio.sleep(0.05)
    assert h.state == "done"


# ---------------------------------------------------------------------------
# monitor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_monitor_streams_stdout_lines():
    gen = monitor('echo "hello" && echo "world"')
    lines = []
    async for line in gen:
        lines.append(line)
    assert lines == ["hello", "world"]


@pytest.mark.asyncio
async def test_monitor_with_spawn_and_buffer():
    em = FakeEventManager()
    qm = QueueManager(event_manager=em)
    ch = qm.queue("ci")

    h = qm.spawn(
        monitor('for i in 1 2 3; do echo "step $i"; done'),
        channel="ci",
        buffer=10,
    )

    lines = []
    for _ in range(3):
        lines.append(await asyncio.wait_for(ch.get(), timeout=2.0))
    await asyncio.sleep(0.1)

    assert lines == ["step 1", "step 2", "step 3"]
    assert h.state == "done"
    assert h.values == ["step 1", "step 2", "step 3"]
    assert any(isinstance(e, StreamEnd) for e in em.events)


@pytest.mark.asyncio
async def test_monitor_cancel_kills_subprocess():
    qm = QueueManager()
    ch = qm.queue("long")

    h = qm.spawn(
        monitor("while true; do echo tick; sleep 0.01; done"),
        channel="long",
        buffer=5,
    )
    await asyncio.wait_for(ch.get(), timeout=1.0)
    await h.cancel()
    assert h.state == "cancelled"
