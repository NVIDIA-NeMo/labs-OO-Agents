# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TurnLoop: races an agent's channels and reports turns as events."""

import asyncio

import pytest

from nooa.interactive import Done, InteractiveAgent, NeedInput
from nooa.runtime.turn_loop import TurnCancelled

TIMEOUT = 5


class ScriptedAgent(InteractiveAgent):
    """Turns are plain Python: each message names what the turn does."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.started = asyncio.Event()

    async def handle(self, notification: dict[str, list]):
        [text, *_] = notification["user_messages"]
        if text == "block":
            self.started.set()
            await asyncio.sleep(60)
        if text == "boom":
            raise ValueError("boom")
        if text == "ask":
            return NeedInput(question="which?")
        if text == "junk":
            return 42
        return Done(explanation=text)


def record(agent, *types):
    seen = []
    for event_type in types:
        agent.event_manager.on(event_type, seen.append)
    return seen


async def settled(seen, count):
    async def poll():
        while sum(e.event_type == "TurnSettled" for e in seen) < count:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), TIMEOUT)


@pytest.fixture
async def agent():
    agent = ScriptedAgent()
    yield agent
    await agent.turns.stop()


async def test_a_put_runs_one_turn_and_reports_it(agent):
    seen = record(agent, "TurnBegan", "TurnSettled")
    agent.turns.start()
    agent.queue_manager.get_channel("user_messages").put("hello")
    await settled(seen, 1)
    began, end = seen
    assert began.notification == {"user_messages": ["hello"]}
    assert (end.kind, end.result) == ("done", Done(explanation="hello"))


async def test_every_way_a_turn_ends_is_a_kind(agent):
    seen = record(agent, "TurnSettled")
    agent.turns.start()
    channel = agent.queue_manager.get_channel("user_messages")
    for text in ("ask", "boom", "junk"):
        channel.put(text)
        await settled(seen, len(seen) + 1)
    ask, boom, junk = seen
    assert ask.kind == "need_input"
    assert (boom.kind, boom.message, type(boom.error)) == ("error", "ValueError: boom", ValueError)
    assert (junk.kind, junk.error) == ("error", None)
    assert "not a turn result" in junk.message


async def test_cancel_settles_before_it_returns_and_tells_the_model(agent):
    seen = record(agent, "TurnSettled")
    agent.turns.start()
    agent.queue_manager.get_channel("user_messages").put("block")
    await asyncio.wait_for(agent.started.wait(), TIMEOUT)
    assert agent.turns.running
    assert await agent.turns.cancel(by="user") is True
    [end] = seen  # subscribers have run by the time cancel() returns
    assert (end.kind, end.cancelled_by) == ("cancelled", "user")
    assert isinstance(agent.event_manager.all_events()[-1], TurnCancelled)
    assert await agent.turns.cancel() is False  # nothing running


async def test_stop_starts_no_new_turn_and_leaves_queued_items(agent):
    seen = record(agent, "TurnBegan")
    agent.turns.start()
    channel = agent.queue_manager.get_channel("user_messages")
    channel.put("block")
    await asyncio.wait_for(agent.started.wait(), TIMEOUT)
    channel.put("next")
    await agent.turns.stop()
    assert len(seen) == 1
    assert channel.snapshot() == ["next"]


async def test_the_loop_ends_when_no_channel_is_left(agent):
    seen = record(agent, "TurnLoopEnded")
    agent.turns.start()
    await asyncio.sleep(0.05)
    for name in list(agent.queue_manager.channels()):
        agent.queue_manager.remove_channel(name)

    async def poll():
        while not seen:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), TIMEOUT)
    assert isinstance(seen[0].error, ValueError)


async def test_a_failing_prepare_blocks_dispatch_without_consuming(agent):
    seen = record(agent, "TurnBegan", "TurnSettled")
    calls = []

    async def prepare():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("could not record the turn")

    agent.turns.start(prepare=prepare)
    channel = agent.queue_manager.get_channel("user_messages")
    channel.put("first")
    await settled(seen, 1)
    [end] = seen
    assert (end.kind, end.ran, type(end.error)) == ("error", False, RuntimeError)
    assert agent.turns.paused and channel.snapshot() == ["first"]
    assert isinstance(agent.turns.dispatch_error, RuntimeError)
    channel.put("second")
    agent.turns.resume()  # explicit retry; first input was not discarded
    await settled(seen, 2)
    assert seen[-1].result == Done(explanation="first")
    assert calls == [True, True]


async def test_a_cancel_during_prepare_settles_without_consuming(agent):
    seen = record(agent, "TurnBegan", "TurnSettled")
    entered, release = asyncio.Event(), asyncio.Event()

    async def prepare():
        entered.set()
        await release.wait()

    agent.turns.start(prepare=prepare)
    agent.queue_manager.get_channel("user_messages").put("hello")
    await asyncio.wait_for(entered.wait(), TIMEOUT)
    cancelling = asyncio.ensure_future(agent.turns.cancel(by="user"))
    await asyncio.sleep(0)
    assert await asyncio.wait_for(cancelling, TIMEOUT) is True
    assert not release.is_set()  # preparation was cancelled, not waited out
    assert agent.queue_manager.get_channel("user_messages").snapshot() == ["hello"]
    assert agent.turns.paused
    [end] = seen  # no TurnBegan: the turn never started
    assert (end.kind, end.ran, end.cancelled_by) == ("cancelled", False, "user")
    assert not any(isinstance(e, TurnCancelled) for e in agent.event_manager.all_events())


async def test_cancel_returns_when_recording_the_cancel_fails(agent, monkeypatch):
    seen = record(agent, "TurnSettled")
    agent.turns.start()
    agent.queue_manager.get_channel("user_messages").put("block")
    await asyncio.wait_for(agent.started.wait(), TIMEOUT)
    add = agent.event_manager.add

    def failing_add(event, **kwargs):
        if isinstance(event, TurnCancelled):
            raise OSError("disk full")
        return add(event, **kwargs)

    monkeypatch.setattr(agent.event_manager, "add", failing_add)
    assert await asyncio.wait_for(agent.turns.cancel(), TIMEOUT) is True
    [end] = seen
    assert (end.kind, type(end.error)) == ("error", OSError)
    assert agent.turns.started  # the loop is still alive


async def test_stop_starting_lets_the_running_turn_finish(agent):
    seen = record(agent, "TurnBegan", "TurnSettled")
    agent.turns.start()
    channel = agent.queue_manager.get_channel("user_messages")
    channel.put("block")
    await asyncio.wait_for(agent.started.wait(), TIMEOUT)
    channel.put("next")
    agent.turns.stop_starting()
    turn = agent.turns._turn
    turn.cancel()  # stands in for the turn finishing on its own
    await settled(seen, 1)
    await asyncio.sleep(0.05)
    assert [e.event_type for e in seen] == ["TurnBegan", "TurnSettled"]
    assert channel.snapshot() == ["next"]


async def test_a_stopped_loop_can_start_again(agent):
    seen = record(agent, "TurnSettled")
    agent.turns.start()
    await agent.turns.stop()
    assert not agent.turns.started
    agent.turns.start()
    agent.queue_manager.get_channel("user_messages").put("again")
    await settled(seen, 1)
    assert seen[0].result == Done(explanation="again")
