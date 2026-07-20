# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-thread Channel and QueueManager tests.

Parameterized to run both same-loop (baseline regression) and cross-loop
(the TUI agent-thread scenario). Every test verifies the identical
observable behavior regardless of whether producer and consumer share a
loop.

The ``mode`` fixture controls threading:
- ``"same_loop"`` — producer and consumer on the test's event loop.
- ``"cross_loop"`` — consumer on the test's loop, producer on a
  background thread with its own loop (mirrors TUI: UI thread produces,
  agent thread consumes).
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest

from nooa.runtime.channels import Channel, QueueManager

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=["same_loop", "cross_loop"])
def mode(request) -> str:
    return request.param


class ProducerHelper:
    """Run a callable on either the current loop or a background thread."""

    def __init__(self, mode: str) -> None:
        self._mode = mode
        self._thread: threading.Thread | None = None
        self._bg_loop: asyncio.AbstractEventLoop | None = None
        self._bg_ready = threading.Event()

    async def start(self) -> None:
        if self._mode == "cross_loop":
            self._bg_loop = asyncio.new_event_loop()
            self._thread = threading.Thread(target=self._run_bg, daemon=True, name="test-producer")
            self._thread.start()
            self._bg_ready.wait(timeout=2.0)

    def _run_bg(self) -> None:
        asyncio.set_event_loop(self._bg_loop)
        self._bg_ready.set()
        self._bg_loop.run_forever()

    def call(self, fn: Callable[[], Any]) -> None:
        """Execute fn() — either inline (same_loop) or on the bg thread."""
        if self._mode == "same_loop":
            fn()
        else:
            assert self._bg_loop is not None
            self._bg_loop.call_soon_threadsafe(fn)

    async def stop(self) -> None:
        if self._bg_loop is not None:
            self._bg_loop.call_soon_threadsafe(self._bg_loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                raise RuntimeError("Producer thread did not stop within 5s")


@pytest.fixture
async def producer(mode):
    p = ProducerHelper(mode)
    await p.start()
    yield p
    await p.stop()


# ---------------------------------------------------------------------------
# Channel.put() / get() — basic delivery
# ---------------------------------------------------------------------------


async def test_put_wakes_pending_get(producer: ProducerHelper, mode: str):
    """A get() blocked before put() receives the item."""
    ch = Channel[str]("test", "queue")

    result = []

    async def consumer():
        item = await ch.get()
        result.append(item)

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.01)  # let consumer register waiter

    producer.call(lambda: ch.put("hello"))
    await asyncio.wait_for(task, timeout=2.0)
    assert result == ["hello"]


async def test_put_before_get_buffers(producer: ProducerHelper, mode: str):
    """Items put() before any get() are buffered and delivered in order."""
    ch = Channel[int]("test", "queue")

    producer.call(lambda: ch.put(1))
    producer.call(lambda: ch.put(2))
    producer.call(lambda: ch.put(3))

    await asyncio.sleep(0.05)  # let cross-thread puts land

    assert await asyncio.wait_for(ch.get(), timeout=1.0) == 1
    assert await asyncio.wait_for(ch.get(), timeout=1.0) == 2
    assert await asyncio.wait_for(ch.get(), timeout=1.0) == 3


async def test_multiple_waiters_fifo(producer: ProducerHelper, mode: str):
    """Multiple concurrent get() calls are served in FIFO order."""
    ch = Channel[str]("test", "queue")

    results = []

    async def consumer(tag):
        item = await ch.get()
        results.append((tag, item))

    t1 = asyncio.create_task(consumer("first"))
    await asyncio.sleep(0.01)
    t2 = asyncio.create_task(consumer("second"))
    await asyncio.sleep(0.01)

    producer.call(lambda: ch.put("a"))
    producer.call(lambda: ch.put("b"))

    await asyncio.wait_for(asyncio.gather(t1, t2), timeout=2.0)
    assert results == [("first", "a"), ("second", "b")]


async def test_put_to_cancelled_waiter_rebuffers(producer: ProducerHelper, mode: str):
    """If the waiter is cancelled, the item is re-buffered for the next consumer."""
    ch = Channel[str]("test", "queue")

    async def consumer():
        return await ch.get()

    task = asyncio.create_task(consumer())
    await asyncio.sleep(0.01)  # let consumer register waiter
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Now put — should buffer since the waiter was cancelled
    producer.call(lambda: ch.put("saved"))
    await asyncio.sleep(0.05)

    # A new consumer should pick it up
    item = await asyncio.wait_for(ch.get(), timeout=1.0)
    assert item == "saved"


async def test_on_get_fires_exactly_once(producer: ProducerHelper, mode: str):
    """on_get callback fires once per delivered item."""
    ch = Channel[str]("test", "queue")
    fired: list[str] = []
    ch.set_on_get(fired.append)

    producer.call(lambda: ch.put("x"))
    await asyncio.sleep(0.05)
    await asyncio.wait_for(ch.get(), timeout=1.0)
    assert fired == ["x"]


async def test_on_get_fires_on_pre_buffered_item(producer: ProducerHelper, mode: str):
    """on_get fires even when the item was already buffered before get()."""
    ch = Channel[str]("test", "queue")
    fired: list[str] = []
    ch.set_on_get(fired.append)

    producer.call(lambda: ch.put("buffered"))
    await asyncio.sleep(0.05)

    item = await asyncio.wait_for(ch.get(), timeout=1.0)
    assert item == "buffered"
    assert fired == ["buffered"]


# ---------------------------------------------------------------------------
# Channel introspection
# ---------------------------------------------------------------------------


async def test_qsize_tracks_buffered_items(producer: ProducerHelper, mode: str):
    ch = Channel[int]("test", "queue")
    assert ch.qsize() == 0

    producer.call(lambda: ch.put(1))
    producer.call(lambda: ch.put(2))
    await asyncio.sleep(0.05)

    assert ch.qsize() == 2
    await ch.get()
    assert ch.qsize() == 1


async def test_snapshot_returns_buffered_items(producer: ProducerHelper, mode: str):
    ch = Channel[str]("test", "queue")
    producer.call(lambda: ch.put("a"))
    producer.call(lambda: ch.put("b"))
    await asyncio.sleep(0.05)

    assert ch.snapshot() == ["a", "b"]
    # Non-consuming
    assert ch.qsize() == 2


async def test_has_waiters_true_when_get_blocked(mode: str):
    """has_waiters is True when a consumer is blocked on get()."""
    ch = Channel[str]("test", "queue")
    assert ch.has_waiters() is False

    task = asyncio.create_task(ch.get())
    await asyncio.sleep(0.01)
    assert ch.has_waiters() is True

    ch.put("release")
    await task
    assert ch.has_waiters() is False


async def test_pop_last_removes_tail(producer: ProducerHelper, mode: str):
    ch = Channel[str]("test", "queue")
    producer.call(lambda: ch.put("first"))
    producer.call(lambda: ch.put("second"))
    await asyncio.sleep(0.05)

    assert ch.pop_last() == "second"
    assert ch.snapshot() == ["first"]


async def test_flush_discards_buffered_items(mode: str):
    """flush() discards all buffered items and returns the count."""
    ch = Channel[str]("test", "queue")
    ch.put("a")
    ch.put("b")
    ch.put("c")
    assert ch.qsize() == 3

    n = ch.flush()
    assert n == 3
    assert ch.qsize() == 0
    assert ch.snapshot() == []


async def test_flush_cancels_blocked_waiters(mode: str):
    """flush() cancels consumers blocked on get()."""
    ch = Channel[str]("test", "queue")

    task = asyncio.create_task(ch.get())
    await asyncio.sleep(0.01)
    assert ch.has_waiters()

    n = ch.flush()
    assert n == 0  # no buffered items, but waiter was cancelled

    with pytest.raises(asyncio.CancelledError):
        await task
    assert not ch.has_waiters()


# ---------------------------------------------------------------------------
# Channel.put() — stress / ordering
# ---------------------------------------------------------------------------


async def test_rapid_puts_maintain_order(producer: ProducerHelper, mode: str):
    """100 rapid puts from producer are delivered in order."""
    ch = Channel[int]("test", "queue")
    N = 100

    for i in range(N):
        producer.call(lambda i=i: ch.put(i))

    await asyncio.sleep(0.2)  # let all cross-thread puts land

    results = []
    while ch.qsize() > 0:
        results.append(await ch.get())
    assert results == list(range(N))


async def test_interleaved_put_get(producer: ProducerHelper, mode: str):
    """Alternating put/get works correctly."""
    ch = Channel[int]("test", "queue")

    for i in range(10):
        producer.call(lambda i=i: ch.put(i))
        await asyncio.sleep(0.02)
        item = await asyncio.wait_for(ch.get(), timeout=1.0)
        assert item == i


# ---------------------------------------------------------------------------
# QueueManager._set_notify / race()
# ---------------------------------------------------------------------------


async def test_race_wakes_on_cross_thread_put(producer: ProducerHelper, mode: str):
    """QueueManager.race() wakes when a channel receives an item from another thread."""
    qm = QueueManager()
    ch = qm.queue("data")

    async def racer():
        return await qm.race()

    task = asyncio.create_task(racer())
    await asyncio.sleep(0.05)  # let race() settle into await

    producer.call(lambda: ch.put("wake"))
    result = await asyncio.wait_for(task, timeout=2.0)
    assert result == [("data", "wake")]


async def test_race_returns_first_of_multiple_channels(producer: ProducerHelper, mode: str):
    """With multiple channels, race returns the first to fire."""
    qm = QueueManager()
    qm.queue("ch1")  # registered but not the target
    ch2 = qm.queue("ch2")

    async def racer():
        return await qm.race()

    task = asyncio.create_task(racer())
    await asyncio.sleep(0.05)

    # Put on ch2 — should win since ch1 is empty
    producer.call(lambda: ch2.put("from-ch2"))
    result = await asyncio.wait_for(task, timeout=2.0)
    assert result == [("ch2", "from-ch2")]


async def test_race_fast_path_buffered_item(producer: ProducerHelper, mode: str):
    """If an item is already buffered when race() is called, it returns immediately."""
    qm = QueueManager()
    ch = qm.queue("data")

    producer.call(lambda: ch.put("pre-buffered"))
    await asyncio.sleep(0.05)

    result = await asyncio.wait_for(qm.race(), timeout=1.0)
    assert result == [("data", "pre-buffered")]


async def test_race_multiple_sequential_calls(producer: ProducerHelper, mode: str):
    """Sequential race() calls each consume one item."""
    qm = QueueManager()
    ch = qm.queue("data")

    for i in range(3):
        producer.call(lambda i=i: ch.put(i))
    await asyncio.sleep(0.05)

    results = []
    for _ in range(3):
        r = await asyncio.wait_for(qm.race(), timeout=1.0)
        results.append(r)
    assert results == [[("data", 0)], [("data", 1)], [("data", 2)]]


async def test_race_cancellation_restores_items(mode: str):
    """Cancelling a race() does not lose items that arrived during the race."""
    qm = QueueManager()
    ch = qm.queue("data")

    task = asyncio.create_task(qm.race())
    await asyncio.sleep(0.05)

    # Cancel before any item arrives
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Now put and race again — item should not be lost
    ch.put("after-cancel")
    result = await asyncio.wait_for(qm.race(), timeout=1.0)
    assert result == [("data", "after-cancel")]


async def test_set_notify_cross_thread_wakes_race(producer: ProducerHelper, mode: str):
    """_set_notify from producer thread correctly wakes race() on consumer loop."""
    qm = QueueManager()
    ch = qm.queue("signals")

    # Start race first to initialize _notify_pair
    task = asyncio.create_task(qm.race())
    await asyncio.sleep(0.05)

    # Put from producer — this goes through _set_notify
    producer.call(lambda: ch.put("signal"))
    result = await asyncio.wait_for(task, timeout=2.0)
    assert result == [("signals", "signal")]


async def test_race_with_event_channel_wake(producer: ProducerHelper, mode: str):
    """Event-mode channel put() wakes race() (returns [])."""

    # QueueManager needs an event_manager for event-mode channels
    class FakeEventManager:
        def add(self, event):
            pass

    qm = QueueManager(event_manager=FakeEventManager())
    qm.queue("q")  # need at least one channel for race() to work
    ev = qm.event("notifications")

    task = asyncio.create_task(qm.race())
    await asyncio.sleep(0.05)

    producer.call(lambda: ev.put("event-data"))
    result = await asyncio.wait_for(task, timeout=2.0)
    # Event-mode wake returns [] (no queue item consumed)
    assert result == []


# ---------------------------------------------------------------------------
# Edge cases: concurrent put + cancel
# ---------------------------------------------------------------------------


async def test_concurrent_puts_to_single_waiter(producer: ProducerHelper, mode: str):
    """Two rapid puts with one waiter: first delivered, second buffered."""
    ch = Channel[str]("test", "queue")

    task = asyncio.create_task(ch.get())
    await asyncio.sleep(0.01)

    producer.call(lambda: ch.put("first"))
    producer.call(lambda: ch.put("second"))

    item1 = await asyncio.wait_for(task, timeout=2.0)
    assert item1 == "first"

    await asyncio.sleep(0.05)
    assert ch.qsize() == 1
    item2 = await asyncio.wait_for(ch.get(), timeout=1.0)
    assert item2 == "second"


async def test_waiter_cancelled_after_cross_thread_put_item_not_lost(mode: str):
    """Simulates the TOCTOU race: waiter cancelled between put and delivery.

    In cross_loop mode, there's a window between call_soon_threadsafe
    scheduling _safe_deliver and the callback executing. If the waiter
    is cancelled in that window, the item must be re-buffered.

    Asserts: the item is NEVER lost — it's either delivered to the
    original waiter (task.result()) or re-buffered for a new consumer.
    """
    if mode == "same_loop":
        pytest.skip("TOCTOU race only applies to cross-loop mode")

    ch = Channel[str]("test", "queue")

    # Start a consumer, then cancel it right after cross-thread put
    task = asyncio.create_task(ch.get())
    await asyncio.sleep(0.01)  # register waiter

    # Simulate: background thread puts, then we immediately cancel
    bg_loop = asyncio.new_event_loop()
    bg_thread = threading.Thread(target=bg_loop.run_forever, daemon=True)
    bg_thread.start()
    await asyncio.sleep(0.01)

    bg_loop.call_soon_threadsafe(lambda: ch.put("contested"))

    # Tiny sleep to let the put schedule _safe_deliver, then cancel waiter
    await asyncio.sleep(0.001)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # Give _safe_deliver time to execute and re-buffer
    await asyncio.sleep(0.05)

    # The item must NOT be lost. Two valid outcomes:
    # 1. Delivered to the original task before cancel propagated
    # 2. Re-buffered by _safe_deliver after detecting the cancel
    delivered_to_original = not task.cancelled() and task.done() and task.result() == "contested"
    rebuffered = ch.qsize() > 0

    assert delivered_to_original or rebuffered, (
        f"Item lost! task.cancelled()={task.cancelled()}, "
        f"task.done()={task.done()}, ch.qsize()={ch.qsize()}"
    )

    if rebuffered:
        item = await asyncio.wait_for(ch.get(), timeout=1.0)
        assert item == "contested"

    bg_loop.call_soon_threadsafe(bg_loop.stop)
    bg_thread.join(timeout=1.0)


async def test_status_reflects_pending_items(producer: ProducerHelper, mode: str):
    """Channel.status() shows pending count and preview."""
    ch = Channel[str]("test", "queue")
    assert ch.status() == ""

    producer.call(lambda: ch.put("hello"))
    producer.call(lambda: ch.put("world"))
    await asyncio.sleep(0.05)

    status = ch.status()
    assert "test: 2 pending" in status
    assert "hello" in status


async def test_queue_manager_status_composite(producer: ProducerHelper, mode: str):
    """QueueManager.status() composes across channels."""
    qm = QueueManager()
    ch1 = qm.queue("alpha")
    ch2 = qm.queue("beta")

    producer.call(lambda: ch1.put("a"))
    producer.call(lambda: ch2.put("b"))
    producer.call(lambda: ch2.put("c"))
    await asyncio.sleep(0.05)

    status = qm.status()
    assert "alpha: 1 pending" in status
    assert "beta: 2 pending" in status


# ---------------------------------------------------------------------------
# Stress: many items across threads
# ---------------------------------------------------------------------------


async def test_stress_1000_items_cross_thread(producer: ProducerHelper, mode: str):
    """1000 items produced and consumed maintain correct count and order."""
    ch = Channel[int]("stress", "queue")
    N = 1000

    for i in range(N):
        producer.call(lambda i=i: ch.put(i))

    # Give cross-thread items time to land
    deadline = time.monotonic() + 5.0
    while ch.qsize() < N and time.monotonic() < deadline:
        await asyncio.sleep(0.01)

    assert ch.qsize() == N

    consumed = []
    for _ in range(N):
        consumed.append(await asyncio.wait_for(ch.get(), timeout=1.0))
    assert consumed == list(range(N))


async def test_stress_concurrent_producers_consumers(producer: ProducerHelper, mode: str):
    """Multiple producers and consumers all get consistent delivery."""
    ch = Channel[int]("stress", "queue")

    produced = set()
    consumed = []
    lock = asyncio.Lock()

    async def consumer():
        while True:
            try:
                item = await asyncio.wait_for(ch.get(), timeout=0.5)
                async with lock:
                    consumed.append(item)
            except (TimeoutError, asyncio.CancelledError):
                break

    # Start 3 consumers
    consumers = [asyncio.create_task(consumer()) for _ in range(3)]

    # Produce 100 items via producer (cross-loop in cross_loop mode)
    for i in range(100):
        producer.call(lambda i=i: ch.put(i))
        produced.add(i)

    # Wait for all to be consumed
    deadline = time.monotonic() + 5.0
    while len(consumed) < 100 and time.monotonic() < deadline:
        await asyncio.sleep(0.01)

    for t in consumers:
        t.cancel()
    await asyncio.gather(*consumers, return_exceptions=True)

    assert set(consumed) == produced
    assert len(consumed) == 100
