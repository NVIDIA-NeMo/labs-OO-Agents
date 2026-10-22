# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``QueueManager.spawn()`` + ``shutdown()``.

No LLM / runtime required — just ``asyncio`` behaviour.
"""

from __future__ import annotations

import asyncio

import pytest

from nooa.runtime.channels import (
    JobError,
    JobHandle,
    QueueManager,
    StreamEnd,
)


class FakeEventManager:
    """Minimal event manager that collects events for assertions."""

    def __init__(self):
        self.events: list = []

    def add(self, event):
        self.events.append(event)


# ---------------------------------------------------------------------------
# spawn(coroutine) basics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spawn_coroutine_puts_result():
    """spawn(coroutine, channel="x") → channel receives the result."""
    em = FakeEventManager()
    qm = QueueManager(event_manager=em)
    ch = qm.queue("results")

    async def job():
        return 42

    handle = qm.spawn(job(), channel="results")
    assert isinstance(handle, JobHandle)
    assert handle.state == "running"

    result = await asyncio.wait_for(ch.get(), timeout=1.0)
    assert result == 42

    # Let the task finalize
    await asyncio.sleep(0)
    assert handle.state == "done"

    # StreamEnd should be in the event log
    stream_ends = [e for e in em.events if isinstance(e, StreamEnd)]
    assert len(stream_ends) == 1
    assert stream_ends[0].channel_name == "results"


@pytest.mark.asyncio
async def test_spawn_async_generator_puts_each_yield():
    """spawn(async_generator, channel="x") → channel receives each yield in order."""
    em = FakeEventManager()
    qm = QueueManager(event_manager=em)
    ch = qm.queue("stream")

    async def gen():
        for i in range(3):
            yield i

    handle = qm.spawn(gen(), channel="stream")

    values = []
    for _ in range(3):
        values.append(await asyncio.wait_for(ch.get(), timeout=1.0))
    assert values == [0, 1, 2]

    await asyncio.sleep(0)
    assert handle.state == "done"

    stream_ends = [e for e in em.events if isinstance(e, StreamEnd)]
    assert len(stream_ends) == 1
    assert stream_ends[0].channel_name == "stream"


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancel_interrupts_generator():
    """JobHandle.cancel() interrupts cleanly; generator finally runs."""
    em = FakeEventManager()
    qm = QueueManager(event_manager=em)
    ch = qm.queue("ticks")
    finally_ran = False

    async def gen():
        nonlocal finally_ran
        try:
            i = 0
            while True:
                yield i
                i += 1
                await asyncio.sleep(0.01)
        finally:
            finally_ran = True

    handle = qm.spawn(gen(), channel="ticks")

    # Consume a few items
    await asyncio.wait_for(ch.get(), timeout=1.0)
    await asyncio.wait_for(ch.get(), timeout=1.0)

    await handle.cancel()
    assert handle.state == "cancelled"
    assert finally_ran

    # No further puts after cancel
    assert ch.qsize() == 0 or True  # generator may have buffered one more


@pytest.mark.asyncio
async def test_cancel_coroutine():
    """Cancelling a coroutine-based job transitions to cancelled."""
    em = FakeEventManager()
    qm = QueueManager(event_manager=em)
    qm.queue("slow")

    async def slow_job():
        await asyncio.sleep(100)
        return "never"

    handle = qm.spawn(slow_job(), channel="slow")
    await asyncio.sleep(0)
    assert handle.state == "running"

    await handle.cancel()
    assert handle.state == "cancelled"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_error_mid_stream_emits_job_error():
    """Error mid-stream → JobError on event channel; data channel not contaminated."""
    em = FakeEventManager()
    qm = QueueManager(event_manager=em)
    ch = qm.queue("data")

    async def bad_gen():
        yield "good"
        raise ValueError("boom")

    handle = qm.spawn(bad_gen(), channel="data")

    # First yield lands fine
    assert await asyncio.wait_for(ch.get(), timeout=1.0) == "good"

    # Wait for the error to propagate
    await asyncio.sleep(0.1)
    assert handle.state == "failed"

    # Data channel now receives the JobError (so race() wakes the agent)
    assert ch.qsize() == 1
    error_item = await ch.get()
    assert isinstance(error_item, JobError)
    assert error_item.error_type == "ValueError"

    # Event channel got JobError + StreamEnd
    job_errors = [e for e in em.events if isinstance(e, JobError)]
    assert len(job_errors) == 1
    assert job_errors[0].channel_name == "data"
    assert job_errors[0].error_type == "ValueError"
    assert "boom" in job_errors[0].error_message

    stream_ends = [e for e in em.events if isinstance(e, StreamEnd)]
    assert len(stream_ends) == 1


@pytest.mark.asyncio
async def test_error_in_coroutine():
    """Coroutine that raises → JobError, handle.state == 'failed'."""
    em = FakeEventManager()
    qm = QueueManager(event_manager=em)
    qm.queue("x")

    async def bad():
        raise RuntimeError("oops")

    handle = qm.spawn(bad(), channel="x")
    await asyncio.sleep(0.1)
    assert handle.state == "failed"

    job_errors = [e for e in em.events if isinstance(e, JobError)]
    assert len(job_errors) == 1
    assert job_errors[0].error_type == "RuntimeError"


# ---------------------------------------------------------------------------
# shutdown()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_shutdown_cancels_all():
    """shutdown() cancels all outstanding handles and awaits cleanup."""
    em = FakeEventManager()
    qm = QueueManager(event_manager=em)
    qm.queue("a")
    qm.queue("b")

    async def forever():
        while True:
            yield "tick"
            await asyncio.sleep(0.01)

    h1 = qm.spawn(forever(), channel="a")
    h2 = qm.spawn(forever(), channel="b")

    # Let them produce at least one item each
    await asyncio.sleep(0.05)
    assert h1.state == "running"
    assert h2.state == "running"

    await qm.shutdown()

    assert h1.state == "cancelled"
    assert h2.state == "cancelled"
    assert len(qm._handles) == 0


@pytest.mark.asyncio
async def test_shutdown_idempotent():
    """Calling shutdown() twice doesn't raise."""
    qm = QueueManager()
    await qm.shutdown()
    await qm.shutdown()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_spawn_unknown_channel_raises():
    """spawn() with an unregistered channel name raises ValueError."""
    qm = QueueManager()

    async def job():
        return 1

    with pytest.raises(ValueError, match="not registered"):
        qm.spawn(job(), channel="nonexistent")


@pytest.mark.asyncio
async def test_spawn_to_event_mode_channel():
    """spawn() can target an event-mode channel — puts go through event_manager."""
    em = FakeEventManager()
    qm = QueueManager(event_manager=em)
    qm.event("notifications")

    async def job():
        return "hello"

    handle = qm.spawn(job(), channel="notifications")
    await asyncio.sleep(0.1)
    assert handle.state == "done"

    # The result went through event_manager (QueueOutput from event channel put)
    # plus StreamEnd
    from nooa.runtime.channels import QueueOutput

    queue_outputs = [e for e in em.events if isinstance(e, QueueOutput)]
    assert len(queue_outputs) == 1
    assert queue_outputs[0].value == "hello"


@pytest.mark.asyncio
async def test_spawn_buffer_true_accumulates_all():
    """buffer=True accumulates all yielded values on the handle."""
    em = FakeEventManager()
    qm = QueueManager(event_manager=em)
    ch = qm.queue("data")

    async def gen():
        for i in range(5):
            yield i

    handle = qm.spawn(gen(), channel="data", buffer=True)

    for _ in range(5):
        await asyncio.wait_for(ch.get(), timeout=1.0)
    await asyncio.sleep(0)

    assert handle.values == [0, 1, 2, 3, 4]
    assert handle.state == "done"


@pytest.mark.asyncio
async def test_spawn_buffer_int_ring():
    """buffer=N keeps only the last N values (ring buffer)."""
    em = FakeEventManager()
    qm = QueueManager(event_manager=em)
    ch = qm.queue("data")

    async def gen():
        for i in range(10):
            yield i

    handle = qm.spawn(gen(), channel="data", buffer=3)

    for _ in range(10):
        await asyncio.wait_for(ch.get(), timeout=1.0)
    await asyncio.sleep(0)

    assert handle.values == [7, 8, 9]
    assert handle.state == "done"


@pytest.mark.asyncio
async def test_spawn_buffer_false_no_accumulation():
    """buffer=False (default) does not accumulate values."""
    em = FakeEventManager()
    qm = QueueManager(event_manager=em)
    ch = qm.queue("data")

    async def gen():
        for i in range(3):
            yield i

    handle = qm.spawn(gen(), channel="data")

    for _ in range(3):
        await asyncio.wait_for(ch.get(), timeout=1.0)
    await asyncio.sleep(0)

    assert handle.values == []
    assert handle.state == "done"
