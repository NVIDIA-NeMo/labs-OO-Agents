# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for channel notification wakeup logic (issue #154).

Verifies that:
1. Event-mode channel puts wake an idle agent sitting in race().
2. Queue-mode channel puts that buffer (no waiter) set the notify event.
3. The QueueManager._notify event is cleared after race() consumes it.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from nooa.runtime.channels import QueueManager

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_event_manager():
    """Minimal event_manager with .add()."""
    em = MagicMock()
    em.add = MagicMock()
    return em


# ---------------------------------------------------------------------------
# Event-mode put wakes idle race()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_event_mode_put_wakes_race():
    """An event-mode channel put() must wake race() even though no
    queue-mode channel has an item. race() returns [] to signal
    'event-triggered wake — events are already in the prompt'.
    """
    em = _make_event_manager()
    qm = QueueManager(event_manager=em)
    qm.queue("q1")  # need at least one queue channel for race()
    evt_ch = qm.event("notifications")

    waiter = asyncio.create_task(qm.race())
    await asyncio.sleep(0)  # let race enter slow path

    # Fire an event-mode put — should wake race.
    evt_ch.put("something happened")
    result = await asyncio.wait_for(waiter, timeout=0.5)
    assert result == [], "event-mode put should wake race with empty items list"


@pytest.mark.asyncio
async def test_event_mode_put_wakes_race_no_queue_channels():
    """race() with only event-mode channels should still wake on put.
    (Previously raised ValueError for no queue channels.)
    """
    em = _make_event_manager()
    qm = QueueManager(event_manager=em)
    evt_ch = qm.event("events")

    waiter = asyncio.create_task(qm.race())
    await asyncio.sleep(0)

    evt_ch.put("event-data")
    result = await asyncio.wait_for(waiter, timeout=0.5)
    assert result == []


@pytest.mark.asyncio
async def test_queue_put_still_wins_over_event_put():
    """When both a queue-mode and event-mode put arrive, the queue
    item is returned (events are already in the prompt)."""
    em = _make_event_manager()
    qm = QueueManager(event_manager=em)
    q = qm.queue("q1")
    evt_ch = qm.event("notifications")

    waiter = asyncio.create_task(qm.race())
    await asyncio.sleep(0)

    # Both fire in the same tick
    q.put("queue-item")
    evt_ch.put("event-item")
    result = await asyncio.wait_for(waiter, timeout=0.5)
    assert result == [("q1", "queue-item")]


@pytest.mark.asyncio
async def test_notify_event_cleared_after_race():
    """The _notify event must be cleared after race() returns so
    the next race() call blocks until a new put() arrives."""
    em = _make_event_manager()
    qm = QueueManager(event_manager=em)
    q = qm.queue("q1")
    qm.event("notifications")

    # First race: event-triggered wake
    waiter1 = asyncio.create_task(qm.race())
    await asyncio.sleep(0)
    qm._channels["notifications"].put("wake1")
    await asyncio.wait_for(waiter1, timeout=0.5)

    # Second race: should block until new event
    waiter2 = asyncio.create_task(qm.race())
    await asyncio.sleep(0)
    assert not waiter2.done(), "race() should block — notify was consumed"

    # Wake it with a queue item this time
    q.put("item")
    result = await asyncio.wait_for(waiter2, timeout=0.5)
    assert result == [("q1", "item")]


@pytest.mark.asyncio
async def test_race_cancellation_with_notify():
    """Cancelling race() while it waits on _notify must propagate
    CancelledError cleanly without leaving stale state."""
    em = _make_event_manager()
    qm = QueueManager(event_manager=em)
    qm.queue("q1")
    qm.event("notifications")

    waiter = asyncio.create_task(qm.race())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter


@pytest.mark.asyncio
async def test_queue_buffered_put_sets_notify():
    """A queue-mode put() that buffers (no waiter) must set the
    QueueManager _notify event so a subsequent race() picks it up
    immediately via the fast path."""
    em = _make_event_manager()
    qm = QueueManager(event_manager=em)
    q = qm.queue("q1")

    # Put BEFORE anyone is waiting — item buffers
    q.put("buffered")

    # race() should find it immediately via fast path
    result = await asyncio.wait_for(qm.race(), timeout=0.5)
    assert result == [("q1", "buffered")]
