# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session: admission, the turn loop, outcomes, events and close."""

import asyncio

import coder_test_agents as agents
import pytest
from coder_test_agents import ask, cell, done, reply, wait_on
from nooa_coder.session.items import (
    Receipt,
    TurnCancelledOutcome,
    TurnEndedUpdate,
)
from nooa_coder.session.session import TurnFailedError
from nooa_coder.session.store import SessionStore

from nooa.interactive import Done, NeedInput

TIMEOUT = 20


def _rows(session, *types):
    return SessionStore._read_rows(session.handle.path, event_types=frozenset(types) or None)


async def test_admission_is_recorded_before_any_turn_runs(make_session):
    session, _ = make_session(start=False)
    receipt = await session.submit("hello")
    assert receipt == Receipt(
        session_id=session.id, channel="user_messages", item_id=receipt.item_id, delivered="queued"
    )
    [(_, admitted)] = _rows(session, "ItemAdmitted")
    assert admitted["item_id"] == receipt.item_id
    assert (admitted["channel"], admitted["item_json"], admitted["source"]) == (
        "user_messages",
        '"hello"',
        "user",
    )
    assert session.agent.queue_manager.get_channel("user_messages").qsize() == 1


async def test_submit_to_an_unknown_channel_is_an_error(make_session):
    session, _ = make_session(start=False)
    with pytest.raises(ValueError, match="nope"):
        await session.submit("x", channel="nope")
    assert _rows(session, "ItemAdmitted") == []


async def test_prompt_returns_done_and_the_reply_is_in_the_transcript(make_session):
    session, llm = make_session(reply("Hi there"))
    outcome = await asyncio.wait_for(session.prompt("hello"), TIMEOUT)
    assert outcome == Done(explanation="answered")
    assert "hello" in str(llm.calls[0].messages)
    assert [(e.role, e.content) for e in session.transcript()] == [
        ("user", "hello"),
        ("agent", "Hi there"),
    ]
    [(_, started)] = _rows(session, "TurnStarted")
    [(_, consumed)] = _rows(session, "ItemConsumed")
    [(_, ended)] = _rows(session, "TurnEnded")
    assert started["item_ids"] == [consumed["item_id"]]
    assert ended["outcome_kind"] == "done"


async def test_need_input_is_returned_and_shown_as_a_question(make_session):
    session, _ = make_session(ask("Which branch?"))
    outcome = await asyncio.wait_for(session.prompt("push it"), TIMEOUT)
    assert outcome == NeedInput(question="Which branch?")
    assert [(e.role, e.content) for e in session.transcript()] == [
        ("user", "push it"),
        ("question", "Which branch?"),
    ]


async def test_waiting_keeps_prompt_open_until_a_later_turn_ends(make_session):
    session, _ = make_session(wait_on("jobs"), done("job finished"), start=False)
    jobs = session.agent.queue_manager.queue("jobs")
    session.start()
    ended = []
    session.subscribe(lambda e: ended.append(e) if isinstance(e, TurnEndedUpdate) else None)

    pending = asyncio.ensure_future(session.prompt("run the job"))
    while not ended:
        await asyncio.sleep(0.01)
    assert ended[0].outcome_kind == "waiting"
    await asyncio.sleep(0.05)
    assert not pending.done()

    jobs.put("job output")
    assert await asyncio.wait_for(pending, TIMEOUT) == Done(explanation="job finished")


async def test_prompt_for_an_item_taken_mid_turn_resolves_with_that_turn(make_session):
    started, block = agents.fresh_events()
    session, _ = make_session(
        cell(
            "STARTED.set()\nawait BLOCK.wait()\n"
            "extra = await self.user_messages.get()\n"
            "return_result(Done(explanation=f'read {extra}'))"
        )
    )
    first = asyncio.ensure_future(session.prompt("first"))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    second = asyncio.ensure_future(session.prompt("second"))
    await asyncio.sleep(0)
    block.set()
    expected = Done(explanation="read second")
    assert await asyncio.wait_for(first, TIMEOUT) == expected
    assert await asyncio.wait_for(second, TIMEOUT) == expected
    [(_, started_row)] = _rows(session, "TurnStarted")
    assert len(started_row["item_ids"]) == 1
    assert len(_rows(session, "ItemConsumed")) == 2


async def test_subscribers_see_admission_then_turn_start_then_turn_end(make_session):
    session, _ = make_session(reply("ok"), start=False)
    seen = []
    unsubscribe = session.subscribe(seen.append)
    session.start()
    await asyncio.wait_for(session.prompt("go"), TIMEOUT)
    kinds = [e.kind for e in seen if e.kind != "agent_event"]
    assert kinds == ["item_admitted", "turn_started", "turn_ended"]
    agent_events = [e for e in seen if e.kind == "agent_event"]
    assert "AgentMessage" in {e.event_type for e in agent_events}
    [message_event] = [e for e in agent_events if e.event_type == "AgentMessage"]
    assert session.agent.event_manager.values()
    assert any(str(e.id) == message_event.event_id for e in session.agent.event_manager.values())

    unsubscribe()
    count = len(seen)
    await session.submit("more")
    assert len(seen) == count


async def test_turns_run_in_a_fresh_context(make_session):
    """Context variables set by whoever starts the session do not leak into turns."""
    token = agents.MARKER.set("outer")
    try:
        session, _ = make_session(
            cell("return_result(Done(explanation=f'marker={MARKER.get()}'))"), start=False
        )
        session.start()
    finally:
        agents.MARKER.reset(token)
    agents.MARKER.set("later")
    outcome = await asyncio.wait_for(session.prompt("x"), TIMEOUT)
    assert outcome == Done(explanation="marker=unset")


async def test_a_failing_turn_raises_from_prompt_and_the_loop_goes_on(make_session):
    session, llm = make_session(done("first"))
    assert await asyncio.wait_for(session.prompt("one"), TIMEOUT) == Done(explanation="first")
    with pytest.raises(TurnFailedError):
        await asyncio.wait_for(session.prompt("two"), TIMEOUT)  # no scripted response left
    [_, (_, ended)] = _rows(session, "TurnEnded")
    assert ended["outcome_kind"] == "error"


async def test_close_is_idempotent_and_closes_the_agent(make_session):
    session, _ = make_session(reply("ok"))
    await asyncio.wait_for(session.prompt("go"), TIMEOUT)
    job = session.agent.queue_manager.spawn(asyncio.Event().wait(), channel="user_messages")
    calls = []

    async def on_agent_close():
        # The agent's background jobs are shut down before the agent closes.
        calls.append(("agent closed", job.state))

    session.agent.event_manager.on_close(on_agent_close)
    closed = []
    session.subscribe(lambda e: closed.append(e.kind) if e.kind == "closed" else None)

    await session.close()
    await session.close()
    assert calls == [("agent closed", "cancelled")]
    assert closed == ["closed"]
    assert session.info.status == "closed"
    assert session.handle._closed
    with pytest.raises(RuntimeError, match="closed"):
        await session.submit("late")


async def test_close_during_a_running_turn_resolves_the_prompt(make_session):
    started, _block = agents.fresh_events()
    session, _ = make_session(cell(agents.BLOCKING_CELL))
    pending = asyncio.ensure_future(session.prompt("go"))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    await asyncio.wait_for(session.close(), TIMEOUT)
    assert await asyncio.wait_for(pending, TIMEOUT) == TurnCancelledOutcome(by="host")
