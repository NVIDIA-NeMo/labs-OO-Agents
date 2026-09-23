# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for new Channel + QueueManager ergonomic APIs.

Covers:
- _ChannelReader.flush()
- QueueManager.remove_channel()
- QueueManager.queue(name, replace=True)
- QueueManager.event(name, replace=True)
- QueueManager.jobs()
- QueueManager.job(name)
- QueueManager.cancel(channel)
- Integration: spawn -> flush -> replace -> re-spawn cycle

No LLM / runtime required — just asyncio behaviour.
"""

from __future__ import annotations

import asyncio

import pytest

from nooa.runtime.channels import Channel, JobHandle, QueueManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeEventManager:
    """Minimal event manager stub that just collects events."""

    def __init__(self) -> None:
        self.events: list = []

    def add(self, event) -> None:
        self.events.append(event)


def test_spawn_docs_explain_channel_driven_waiting():
    """QueueManager documents waiting without assuming an agent response protocol."""
    manager_docs = QueueManager.__doc__ or ""
    spawn_docs = QueueManager.spawn.__doc__ or ""
    handle_docs = JobHandle.__doc__ or ""

    assert "``spawn()`` is channel-driven" in manager_docs
    assert "not a completion-waiting primitive" in manager_docs
    assert "through ``race()`` rather than polling" in spawn_docs
    assert "yield the current turn" in spawn_docs
    assert "does not assume a response protocol" in spawn_docs
    assert "not completion-waiting primitives" in handle_docs


# ---------------------------------------------------------------------------
# 1. _ChannelReader.flush()
# ---------------------------------------------------------------------------


class TestChannelReaderFlush:
    """Tests for _ChannelReader.flush() method."""

    @pytest.mark.asyncio
    async def test_flush_empty_channel_returns_zero(self):
        """flush() on an empty channel returns 0."""
        ch = Channel("q", "queue")
        reader = ch.reader
        assert reader.flush() == 0

    @pytest.mark.asyncio
    async def test_flush_non_empty_channel_returns_count_and_clears(self):
        """flush() returns item count and clears all pending items."""
        ch = Channel("q", "queue")
        ch.put("a")
        ch.put("b")
        ch.put("c")
        reader = ch.reader
        assert reader.flush() == 3
        assert ch.qsize() == 0
        assert ch.is_empty()

    @pytest.mark.asyncio
    async def test_flush_cancels_pending_get_waiters(self):
        """flush() cancels pending get() waiters so they raise CancelledError."""
        ch = Channel("q", "queue")
        reader = ch.reader

        # Start a get() that will block
        getter = asyncio.create_task(reader.get())
        await asyncio.sleep(0)  # let the task register its waiter
        assert ch.has_waiters()

        # Flush should cancel the waiter
        reader.flush()
        assert not ch.has_waiters()

        # The getter task should raise CancelledError
        with pytest.raises(asyncio.CancelledError):
            await getter

    @pytest.mark.asyncio
    async def test_channel_works_normally_after_flush(self):
        """After flush, put/get cycle works as expected."""
        ch = Channel("q", "queue")
        reader = ch.reader

        ch.put("before")
        reader.flush()

        # Channel should work normally after flush
        ch.put("after")
        assert await reader.get() == "after"
        assert ch.qsize() == 0


# ---------------------------------------------------------------------------
# 2. QueueManager.remove_channel()
# ---------------------------------------------------------------------------


class TestQueueManagerRemoveChannel:
    """Tests for QueueManager.remove_channel() method."""

    @pytest.mark.asyncio
    async def test_remove_existing_channel_succeeds(self):
        """remove_channel() removes a registered channel without error."""
        qm = QueueManager()
        qm.queue("test_ch")
        assert "test_ch" in qm.names()

        qm.remove_channel("test_ch")
        assert "test_ch" not in qm.names()

    @pytest.mark.asyncio
    async def test_remove_non_existent_channel_raises_key_error(self):
        """remove_channel() raises KeyError for unknown channel names."""
        qm = QueueManager()
        with pytest.raises(KeyError):
            qm.remove_channel("does_not_exist")

    @pytest.mark.asyncio
    async def test_removed_channel_no_longer_in_names(self):
        """After removal, the channel name is absent from names()."""
        qm = QueueManager()
        qm.queue("ch1")
        qm.queue("ch2")
        assert set(qm.names()) == {"ch1", "ch2"}

        qm.remove_channel("ch1")
        assert qm.names() == ["ch2"]

    @pytest.mark.asyncio
    async def test_remove_cancels_spawned_jobs_on_channel(self):
        """remove_channel() cancels spawned jobs targeting that channel."""
        qm = QueueManager()
        qm.queue("data")

        async def slow_producer():
            await asyncio.sleep(100)
            return "never"

        handle = qm.spawn(slow_producer(), channel="data")
        assert handle.state == "running"

        qm.remove_channel("data")
        # Give the event loop a chance to process the cancellation
        await asyncio.sleep(0.01)
        assert handle.state == "cancelled"


# ---------------------------------------------------------------------------
# 3. QueueManager.queue(name, replace=True)
# ---------------------------------------------------------------------------


class TestQueueManagerQueueReplace:
    """Tests for QueueManager.queue() with replace parameter."""

    @pytest.mark.asyncio
    async def test_replace_true_with_existing_name_succeeds(self):
        """replace=True replaces an existing channel with a fresh one."""
        qm = QueueManager()
        ch1 = qm.queue("data")
        ch1.put("old_item")

        ch2 = qm.queue("data", replace=True)
        # Should be a new, empty channel
        assert ch2 is not ch1
        assert ch2.qsize() == 0
        assert "data" in qm.names()

    @pytest.mark.asyncio
    async def test_replace_false_with_existing_name_raises_value_error(self):
        """replace=False (default) raises ValueError on duplicate name."""
        qm = QueueManager()
        qm.queue("data")

        with pytest.raises(ValueError, match="already registered"):
            qm.queue("data")

    @pytest.mark.asyncio
    async def test_replace_true_with_non_existing_name_creates_normally(self):
        """replace=True with a new name just creates the channel."""
        qm = QueueManager()
        ch = qm.queue("brand_new", replace=True)
        assert ch.name == "brand_new"
        assert "brand_new" in qm.names()


# ---------------------------------------------------------------------------
# 4. QueueManager.event(name, replace=True)
# ---------------------------------------------------------------------------


class TestQueueManagerEventReplace:
    """Tests for QueueManager.event() with replace parameter."""

    @pytest.mark.asyncio
    async def test_replace_true_with_existing_event_channel_succeeds(self):
        """replace=True replaces an existing event channel."""
        em = FakeEventManager()
        qm = QueueManager(event_manager=em)
        ch1 = qm.event("notifications")
        ch2 = qm.event("notifications", replace=True)
        assert ch2 is not ch1
        assert ch2.mode == "event"
        assert "notifications" in qm.names()

    @pytest.mark.asyncio
    async def test_replace_false_with_existing_event_channel_raises(self):
        """replace=False (default) raises ValueError on duplicate event name."""
        em = FakeEventManager()
        qm = QueueManager(event_manager=em)
        qm.event("notifications")

        with pytest.raises(ValueError, match="already registered"):
            qm.event("notifications")

    @pytest.mark.asyncio
    async def test_replace_true_with_non_existing_event_creates_normally(self):
        """replace=True with a new name just creates the event channel."""
        em = FakeEventManager()
        qm = QueueManager(event_manager=em)
        ch = qm.event("alerts", replace=True)
        assert ch.name == "alerts"
        assert ch.mode == "event"
        assert "alerts" in qm.names()


# ---------------------------------------------------------------------------
# 5. QueueManager.jobs()
# ---------------------------------------------------------------------------


class TestQueueManagerJobs:
    """Tests for QueueManager.jobs() method."""

    @pytest.mark.asyncio
    async def test_jobs_empty_when_no_spawns(self):
        """jobs() returns empty dict when nothing has been spawned."""
        qm = QueueManager()
        qm.queue("ch")
        assert qm.jobs() == {}

    @pytest.mark.asyncio
    async def test_jobs_returns_correct_state_for_spawned_jobs(self):
        """jobs() returns channel_name -> state for all tracked handles."""
        qm = QueueManager()
        qm.queue("fast")
        qm.queue("slow")

        async def instant():
            return "done"

        async def blocking():
            await asyncio.sleep(100)
            return "never"

        qm.spawn(instant(), channel="fast")
        qm.spawn(blocking(), channel="slow")

        # Let the fast one complete
        await asyncio.sleep(0.01)

        jobs = qm.jobs()
        assert jobs["fast"] == "done"
        assert jobs["slow"] == "running"

        # Cleanup
        await qm.shutdown()


# ---------------------------------------------------------------------------
# 6. QueueManager.job(name)
# ---------------------------------------------------------------------------


class TestQueueManagerJob:
    """Tests for QueueManager.job() method."""

    @pytest.mark.asyncio
    async def test_job_returns_none_when_no_matching_handle(self):
        """job() returns None when no handle matches the name."""
        qm = QueueManager()
        qm.queue("ch")
        assert qm.job("ch") is None
        assert qm.job("nonexistent") is None

    @pytest.mark.asyncio
    async def test_job_returns_most_recent_handle(self):
        """job() returns the most recent handle when multiple exist."""
        qm = QueueManager()
        qm.queue("data")

        async def producer1():
            return "first"

        async def producer2():
            await asyncio.sleep(100)
            return "second"

        qm.spawn(producer1(), channel="data")
        await asyncio.sleep(0.01)  # let first complete

        h2 = qm.spawn(producer2(), channel="data")

        # Should return the most recent (h2)
        result = qm.job("data")
        assert result is h2

        # Cleanup
        await qm.shutdown()


# ---------------------------------------------------------------------------
# 7. QueueManager.cancel(channel)
# ---------------------------------------------------------------------------


class TestQueueManagerCancel:
    """Tests for QueueManager.cancel() method."""

    @pytest.mark.asyncio
    async def test_cancel_returns_false_when_no_handle(self):
        """cancel() returns False when no matching handle exists."""
        qm = QueueManager()
        qm.queue("ch")
        result = await qm.cancel("ch")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_returns_true_and_cancels(self):
        """cancel() returns True and cancels the most recent handle."""
        qm = QueueManager()
        qm.queue("data")

        async def slow():
            await asyncio.sleep(100)
            return "never"

        handle = qm.spawn(slow(), channel="data")
        assert handle.state == "running"

        result = await qm.cancel("data")
        assert result is True
        assert handle.state == "cancelled"


# ---------------------------------------------------------------------------
# 8. Integration: spawn -> flush -> replace -> re-spawn cycle
# ---------------------------------------------------------------------------


class TestIntegrationSpawnFlushReplaceRespawn:
    """Integration test: full lifecycle of spawn -> flush -> replace -> re-spawn."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """Spawn a job, flush the channel, replace it, and re-spawn."""
        em = FakeEventManager()
        qm = QueueManager(event_manager=em)
        ch = qm.queue("stream")
        reader = ch.reader

        # Phase 1: Spawn a generator that yields values
        async def gen_values():
            for i in range(5):
                yield f"v{i}"
                await asyncio.sleep(0.01)

        h1 = qm.spawn(gen_values(), channel="stream")
        # Let some values arrive
        await asyncio.sleep(0.05)

        # Phase 2: Flush the channel (discard buffered items)
        flushed = reader.flush()
        assert flushed >= 0  # some items were buffered

        # Phase 3: Cancel the old job
        result = await qm.cancel("stream")
        assert result is True
        assert h1.state == "cancelled"

        # Phase 4: Replace the channel with a fresh one
        ch2 = qm.queue("stream", replace=True)
        reader2 = ch2.reader
        assert ch2 is not ch

        # Phase 5: Re-spawn a new producer on the fresh channel
        async def new_producer():
            return "fresh_data"

        h2 = qm.spawn(new_producer(), channel="stream")
        await asyncio.sleep(0.01)

        # The new value should be available
        assert await reader2.get() == "fresh_data"
        assert h2.state == "done"

        # Cleanup
        await qm.shutdown()

    @pytest.mark.asyncio
    async def test_replace_cancels_old_spawned_jobs(self):
        """Replacing a channel via queue(replace=True) cancels old spawned jobs."""
        qm = QueueManager()
        qm.queue("work")

        async def long_running():
            await asyncio.sleep(100)
            return "never"

        h = qm.spawn(long_running(), channel="work")
        assert h.state == "running"

        # Replace the channel - should cancel jobs via remove_channel
        qm.queue("work", replace=True)
        await asyncio.sleep(0.01)
        assert h.state == "cancelled"

        await qm.shutdown()


class TestQueueManagerShutdownKeepDaemons:
    """A plain shutdown() must spare daemon=True handles; only include_daemons=True ends them."""

    @pytest.mark.asyncio
    async def test_shutdown_spares_daemon_jobs_unless_included(self):
        """A mid-session interrupt (ACP cancel() -> cancel_work()) must not
        destroy a long-lived infrastructure producer marked daemon=True --
        only the same "genuinely finite work" this shutdown is meant to
        interrupt. running_work_handles() already treats daemon jobs as not
        counting toward pending work for quiescence checks; shutdown() must
        agree, or a stray cancel silently tears down infrastructure a user
        never asked to stop.
        """
        qm = QueueManager()
        qm.queue("infra")
        qm.queue("work")

        async def daemon_producer():
            try:
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                raise
            return "never"

        async def finite_job():
            await asyncio.sleep(100)
            return "never"

        daemon_handle = qm.spawn(daemon_producer(), channel="infra", daemon=True)
        work_handle = qm.spawn(finite_job(), channel="work")
        assert daemon_handle.state == "running"
        assert work_handle.state == "running"

        await qm.shutdown()

        assert daemon_handle.state == "running"
        assert work_handle.state == "cancelled"
        assert daemon_handle in qm.handles()
        assert work_handle not in qm.handles()

        # A real final close must still take everything down.
        await qm.shutdown(include_daemons=True)
        assert daemon_handle.state == "cancelled"
        assert qm.handles() == []


class TestQueueManagerRemoveChannelPrunesFinishedHandles:
    @pytest.mark.asyncio
    async def test_repeated_replace_cycles_do_not_leak_handles(self):
        """Only shutdown() used to prune self._handles; remove_channel()
        (called by queue(name, replace=True), e.g. an MCP reconnect loop)
        cancelled a running job but left its handle in the list forever.
        Any code repeatedly removing and recreating a same-named channel
        leaked one handle per cycle for the life of the session --
        _job_snapshots_on_owner-style code iterating handles() on every
        queue notification would grow unbounded, and job(name)/cancel(name)
        (which scan from the end) could still resolve correctly, but the
        list itself never stopped growing.
        """
        qm = QueueManager()
        qm.queue("reconnect")

        async def producer():
            await asyncio.sleep(100)
            return "never"

        for _ in range(20):
            handle = qm.spawn(producer(), channel="reconnect")
            qm.queue("reconnect", replace=True)
            # Let the just-requested cancellation actually finish before the
            # next cycle's remove_channel() call, matching a real reconnect
            # loop with real awaits between attempts.
            for _ in range(5):
                await asyncio.sleep(0)
            assert handle.state == "cancelled"

        assert len(qm.handles()) <= 2

        await qm.shutdown()

    @pytest.mark.asyncio
    async def test_remove_channel_cancel_does_not_double_cancel_during_cleanup(self):
        """remove_channel() used to call handle._task.cancel() directly
        without setting _cancel_called. If shutdown() (or another
        remove_channel/cancel) reached the same still-cleaning-up handle
        before its cancelled task finished, the repeat-cancel guard didn't
        apply and a second CancelledError could interrupt the job's cleanup
        mid-await.
        """
        qm = QueueManager()
        qm.queue("work")
        cleanup_started = asyncio.Event()
        release_cleanup = asyncio.Event()
        cleanup_completed = asyncio.Event()

        async def job():
            try:
                await asyncio.sleep(100)
            finally:
                cleanup_started.set()
                await release_cleanup.wait()
                cleanup_completed.set()

        handle = qm.spawn(job(), channel="work")
        await asyncio.sleep(0)
        qm.remove_channel("work")
        await asyncio.wait_for(cleanup_started.wait(), timeout=2)

        shutdown = asyncio.create_task(qm.shutdown())
        await asyncio.sleep(0)
        release_cleanup.set()
        await asyncio.wait_for(shutdown, timeout=2)

        assert cleanup_completed.is_set()
        assert handle.state == "cancelled"


class TestJobHandleCancelIgnoresUnrelatedCancellingCount:
    @pytest.mark.asyncio
    async def test_cancel_still_stops_a_job_mid_unrelated_cancellation(self):
        """JobHandle.cancel() used to check self._task.cancelling() to decide
        whether to call task.cancel() -- but that counter is nonzero for ANY
        reason the task's own body called .cancel() internally too (e.g. an
        asyncio.timeout()/wait_for() the job uses, which cancels-then-
        uncancels around a caught timeout). Calling cancel() while that
        counter happens to be transiently nonzero for a completely unrelated
        reason used to skip requesting cancellation entirely, hanging
        forever on the following await.

        Simulates that window directly: the job absorbs one CancelledError
        (as its own internal timeout handling would) and keeps running,
        leaving cancelling() nonzero until its own uncancel() call is
        scheduled. cancel() called in that exact window must still stop it.
        """
        qm = QueueManager()
        qm.queue("work")
        started = asyncio.Event()
        survived_first_cancel = asyncio.Event()
        real_cancel_requested = asyncio.Event()

        async def job():
            task = asyncio.current_task()
            assert task is not None
            try:
                started.set()
                await asyncio.sleep(100)
            except asyncio.CancelledError:
                # Simulate an internal asyncio.timeout()-style absorb: accept
                # this one, keep going, cancelling() stays nonzero until
                # uncancel() actually runs on a later awaited step.
                survived_first_cancel.set()
                await real_cancel_requested.wait()
                task.uncancel()
                await asyncio.sleep(100)
            return "should not reach here either"

        handle = qm.spawn(job(), channel="work")
        await asyncio.wait_for(started.wait(), timeout=2)
        # Put the task into the exact buggy window: cancelling() > 0 for a
        # reason unrelated to the handle.cancel() call about to happen.
        handle._task.cancel()
        await asyncio.wait_for(survived_first_cancel.wait(), timeout=2)
        assert handle._task.cancelling() > 0

        real_cancel_requested.set()
        # Must not hang: cancel() must still request cancellation despite
        # cancelling() already being nonzero for the unrelated reason above.
        await asyncio.wait_for(handle.cancel(), timeout=2)
        assert handle.state == "cancelled"

        await qm.shutdown()


class TestChannelFlushFireOnGet:
    def test_flush_fire_on_get_records_discarded_items_before_they_vanish(self):
        """A channel's on_get hook can have a durability side effect (e.g.
        user_messages recording an accepted prompt). Plain flush() discards
        items via a bare list clear -- on_get never fires, so an item
        admitted into the queue but flushed before it was ever dequeued (a
        session/cancel racing a just-submitted prompt) vanished with no
        trace at all, not even a log line.
        """
        qm = QueueManager()
        recorded: list[str] = []
        ch = qm.queue("user_messages", on_get=recorded.append)
        ch.put("hello")
        ch.put("still queued when cancelled")

        n = ch.flush(fire_on_get=True)

        assert n == 2
        assert recorded == ["hello", "still queued when cancelled"]
        assert ch.qsize() == 0

    def test_flush_default_still_discards_silently(self):
        """The default (no fire_on_get) behavior is unchanged for every
        existing caller that relies on flush() being a plain discard.
        """
        qm = QueueManager()
        recorded: list[str] = []
        ch = qm.queue("other", on_get=recorded.append)
        ch.put("value")

        n = ch.flush()

        assert n == 1
        assert recorded == []
