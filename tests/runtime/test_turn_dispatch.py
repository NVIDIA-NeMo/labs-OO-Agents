# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Non-consuming dispatch and synchronous batch ownership boundaries."""

import asyncio
import contextvars
import threading

import pytest

from nooa.context_blocks import Metadata
from nooa.interactive import Done, InteractiveAgent
from nooa.runtime.channels import QueueManager

TIMEOUT = 3
marker = contextvars.ContextVar("dispatch_marker", default="fresh")


class Agent(InteractiveAgent):
    def __init__(self):
        super().__init__()
        self.notifications = []
        self.contexts = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    broken_none = None

    def sync_raise(self, _notification):
        raise ValueError("sync error")

    def sync_result(self, _notification):
        return Done(explanation="invalid non-awaitable")

    async def handle(self, notification):
        self.notifications.append(notification)
        self.contexts.append(marker.get())
        marker.set("turn-local")
        self.entered.set()
        if notification.get("user_messages") == ["block"]:
            await self.release.wait()
        return Done(explanation="ok")


@pytest.fixture
async def agent():
    value = Agent()
    yield value
    await value.turns.stop()


async def wait_until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), TIMEOUT)


@pytest.mark.parametrize("action", ["cancel", "stop"])
async def test_preparation_cancel_never_consumes(agent, action):
    entered, forever = asyncio.Event(), asyncio.Event()
    observed = []
    agent.event_manager.on("ChannelItemConsumed", observed.append)

    async def prepare():
        entered.set()
        await forever.wait()

    agent.turns.start(prepare=prepare)
    channel = agent.queue_manager.get_channel("user_messages")
    channel.put("queued")
    await asyncio.wait_for(entered.wait(), TIMEOUT)
    await asyncio.wait_for(getattr(agent.turns, action)(), TIMEOUT)
    assert channel.snapshot() == ["queued"]
    assert observed == [] and agent.notifications == []
    assert agent.turns.paused


async def test_idle_pause_wake_does_not_claim_and_resume_reuses_loop(agent):
    agent.turns.start()
    await asyncio.sleep(0)
    task = agent.turns._task
    agent.turns.pause()
    channel = agent.queue_manager.get_channel("user_messages")
    channel.put("one")
    await asyncio.sleep(0.03)
    assert channel.snapshot() == ["one"] and agent.notifications == []
    agent.turns.resume()
    await asyncio.wait_for(agent.entered.wait(), TIMEOUT)
    assert agent.turns._task is task


async def test_pause_after_preparation_rechecks_gate(agent):
    entered, release = asyncio.Event(), asyncio.Event()

    async def prepare():
        entered.set()
        await release.wait()

    agent.turns.start(prepare=prepare)
    channel = agent.queue_manager.get_channel("user_messages")
    channel.put("one")
    await entered.wait()
    agent.turns.pause()
    release.set()
    await wait_until(lambda: not agent.turns.running)
    assert channel.snapshot() == ["one"] and agent.notifications == []


async def test_commit_failure_does_not_spin_or_publish_consumption(agent):
    calls, observed = [], []

    def commit(batch):
        calls.append(batch)
        raise OSError("disk full")

    agent.event_manager.on("ChannelItemConsumed", observed.append)
    agent.turns.start(commit=commit)
    channel = agent.queue_manager.get_channel("user_messages")
    channel.put("one")
    await wait_until(lambda: agent.turns.dispatch_error is not None)
    channel.put("two")
    await asyncio.sleep(0.03)
    assert len(calls) == 1 and observed == []
    assert channel.snapshot() == ["one", "two"]
    assert agent.notifications == [] and agent.turns.started


@pytest.mark.parametrize("method", [None, "missing", "sync_raise", "sync_result"])
async def test_setup_errors_settle_and_loop_continues(agent, method):
    seen = []
    agent.event_manager.on("TurnSettled", seen.append)
    turn_method = {None: "broken_none", "missing": "absent"}.get(method, method)
    agent.turns.start(turn_method=turn_method)
    channel = agent.queue_manager.get_channel("user_messages")
    channel.put("first")
    await wait_until(lambda: len(seen) == 1)
    assert seen[0].kind == "error" and seen[0].committed
    assert not agent.turns.paused
    agent.turns._turn_method = "handle"  # repair configuration, same live loop
    channel.put("second")
    await wait_until(lambda: len(seen) == 2)
    assert seen[1].kind == "done"


async def test_observer_cancelled_error_cannot_skip_later_observer_or_cancel(agent):
    def broken(_):
        raise asyncio.CancelledError()

    seen = []
    agent.event_manager.on("TurnSettled", broken)
    agent.event_manager.on("TurnSettled", seen.append)
    agent.turns.start()
    agent.queue_manager.get_channel("user_messages").put("block")
    await asyncio.wait_for(agent.entered.wait(), TIMEOUT)
    assert await asyncio.wait_for(agent.turns.cancel(), TIMEOUT)
    assert len(seen) == 1 and seen[0].kind == "cancelled"
    assert agent.turns.started


async def test_standalone_aclose_stops_turn_and_jobs(agent):
    cancelled = asyncio.Event()

    async def job():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    agent.queue_manager.spawn(job(), channel="user_messages")
    agent.turns.start()
    agent.queue_manager.get_channel("user_messages").put("block")
    await asyncio.wait_for(agent.entered.wait(), TIMEOUT)
    await asyncio.wait_for(agent.aclose(), TIMEOUT)
    assert not agent.turns.started and not agent.turns.running
    assert cancelled.is_set()


async def test_event_only_wakes_once_and_channel_removal_has_no_lost_wake(agent):
    qm = agent.queue_manager
    qm.remove_channel("user_messages")
    events = qm.event("events")
    events.put("before start")
    agent.turns.start()
    await wait_until(lambda: len(agent.notifications) == 1)
    assert agent.notifications == [{}]
    await asyncio.sleep(0.03)
    assert len(agent.notifications) == 1
    await asyncio.to_thread(events.put, "thread event")
    await wait_until(lambda: len(agent.notifications) == 2)
    qm.remove_channel("events")
    await wait_until(lambda: not agent.turns.started)


async def test_channel_flush_remove_during_prepare_rechecks_readiness(agent):
    entered, release = asyncio.Event(), asyncio.Event()

    async def prepare():
        entered.set()
        await release.wait()

    agent.turns.start(prepare=prepare)
    qm = agent.queue_manager
    ch = qm.queue("extra")
    ch.put("removed")
    await entered.wait()
    ch.flush()
    qm.remove_channel("extra")
    release.set()
    await wait_until(lambda: not agent.turns.running)
    assert agent.notifications == []
    qm.get_channel("user_messages").put("new")
    await asyncio.wait_for(agent.entered.wait(), TIMEOUT)
    assert agent.notifications == [{"user_messages": ["new"]}]


async def test_cross_thread_batch_fifo_and_observer_reentry_is_after_ownership():
    qm = QueueManager()
    a, b = qm.queue("a"), qm.queue("b")
    for value in range(30):
        await asyncio.to_thread(a.put, value)
    b.put("b")
    observed = []

    def observe(item):
        observed.append((item, a.snapshot(), b.snapshot()))
        if item == 0:
            a.put("later")

    a.set_on_get(observe)
    await asyncio.wait_for(qm.wait_ready(), TIMEOUT)
    batch = qm.claim_batch()
    assert batch == {"a": list(range(30)), "b": ["b"]}
    assert observed[0] == (0, [], [])
    assert a.snapshot() == ["later"] and b.snapshot() == []


def test_recorder_reentry_rejects_all_mutation():
    qm = QueueManager()
    ch = qm.queue("a")
    ch.put("first")

    def commit(batch):
        assert batch == {"a": ["first"]}
        with pytest.raises(RuntimeError, match="mutation"):
            ch.put("next")
        with pytest.raises(RuntimeError, match="mutation"):
            ch.flush()
        with pytest.raises(RuntimeError, match="mutation"):
            qm.remove_channel("a")
        with pytest.raises(RuntimeError, match="reentrant"):
            qm.claim_batch()
        batch["a"].clear()  # recorder cannot change the claimed selection

    assert qm.claim_batch(commit) == {"a": ["first"]}
    assert ch.snapshot() == []


def test_cross_thread_put_waits_for_commit_without_entering_selected_batch():
    qm = QueueManager()
    ch = qm.queue("a")
    ch.put("first")
    starting, finished = threading.Event(), threading.Event()

    def producer():
        starting.set()
        ch.put("next")
        finished.set()

    thread = threading.Thread(target=producer)

    def commit(_batch):
        thread.start()
        assert starting.wait(TIMEOUT)
        assert not finished.is_set()

    assert qm.claim_batch(commit) == {"a": ["first"]}
    thread.join(TIMEOUT)
    assert finished.is_set() and ch.snapshot() == ["next"]


async def test_fresh_loop_and_each_turn_context_on_restart(agent):
    token = marker.set("host")
    try:
        agent.turns.start()
    finally:
        marker.reset(token)
    channel = agent.queue_manager.get_channel("user_messages")
    for count in (1, 2):
        channel.put(str(count))
        await wait_until(
            lambda count=count: len(agent.contexts) == count and not agent.turns.running
        )
    assert agent.contexts == ["fresh", "fresh"]
    assert marker.get() == "fresh"
    await agent.turns.stop()
    context = contextvars.Context()
    context.run(marker.set, "loop-hook")
    agent.turns.start(context=context)
    channel.put("restart")
    await wait_until(lambda: len(agent.contexts) == 3)
    assert agent.contexts[-1] == "loop-hook"
    assert len(agent.event_manager._handlers["PythonOutput"]) == 1


@pytest.mark.parametrize("kind", ["Metadata", "*"])
async def test_observer_cancelled_error_isolated_before_recording(kind):
    agent = Agent()
    seen = []

    def broken(_):
        raise asyncio.CancelledError()

    agent.event_manager.on(kind, broken)
    agent.event_manager.on("*", seen.append)
    event = Metadata(description="recorded")
    tag = agent.event_manager.add(event)
    assert seen == [event] and agent.event_manager.get(tag) is event
    await agent.aclose()


async def test_turn_self_stop_is_rejected_without_task_cycle():
    class SelfStopAgent(InteractiveAgent):
        async def handle(self, notification):
            await self.turns.stop()

    agent = SelfStopAgent()
    seen = []
    agent.event_manager.on("TurnSettled", seen.append)
    agent.turns.start()
    agent.queue_manager.get_channel("user_messages").put("self-stop")
    await wait_until(lambda: bool(seen))
    assert seen[0].kind == "error"
    assert "cannot stop its own loop" in seen[0].message
    assert agent.turns.started
    await asyncio.wait_for(agent.aclose(), TIMEOUT)
    assert not agent.turns.started


async def test_aclose_after_restart_stops_new_producer(agent):
    await agent.aclose()
    agent.turns.start()
    cancelled = asyncio.Event()

    async def producer():
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    handle = agent.queue_manager.spawn(producer(), channel="user_messages")
    await asyncio.sleep(0)
    await asyncio.wait_for(agent.aclose(), TIMEOUT)
    assert cancelled.is_set() and handle.state == "cancelled"
    assert not agent.turns.started


async def test_prepare_swallowing_cancel_still_cannot_commit(agent):
    entered = asyncio.Event()
    seen, committed, consumed = [], [], []

    async def prepare():
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return  # cooperative cleanup may intentionally swallow cancellation

    agent.event_manager.on("TurnSettled", seen.append)
    agent.event_manager.on("ChannelItemConsumed", consumed.append)
    agent.turns.start(prepare=prepare, commit=committed.append)
    channel = agent.queue_manager.get_channel("user_messages")
    channel.put("still queued")
    await asyncio.wait_for(entered.wait(), TIMEOUT)
    assert await asyncio.wait_for(agent.turns.cancel(by="review"), TIMEOUT)
    assert channel.snapshot() == ["still queued"]
    assert committed == consumed == agent.notifications == []
    assert len(seen) == 1 and seen[0].kind == "cancelled"
    assert seen[0].cancelled_by == "review"
    assert not seen[0].ran and not seen[0].committed and agent.turns.paused


async def test_pending_event_wake_after_final_removal_ends_instead_of_empty_turn(agent):
    entered, release = asyncio.Event(), asyncio.Event()
    qm = agent.queue_manager
    qm.remove_channel("user_messages")
    channel = qm.event("events")

    async def prepare():
        entered.set()
        await release.wait()

    channel.put("pending wake")
    agent.turns.start(prepare=prepare)
    await asyncio.wait_for(entered.wait(), TIMEOUT)
    qm.remove_channel("events")
    release.set()
    await wait_until(lambda: not agent.turns.started)
    assert agent.notifications == []
    assert qm.claim_batch() is None


def test_recorder_event_put_is_rejected_before_observers_or_wake():
    from nooa.runtime.event_manager import EventManager

    manager = EventManager()
    qm = QueueManager(event_manager=manager)
    queue = qm.queue("queue")
    channel = qm.event("event")
    seen = []
    manager.on("QueueOutput", seen.append)
    queue.put("first")

    def commit(_batch):
        with pytest.raises(RuntimeError, match="mutation"):
            channel.put("not published")

    assert qm.claim_batch(commit) == {"queue": ["first"]}
    assert seen == [] and not qm.ready()
