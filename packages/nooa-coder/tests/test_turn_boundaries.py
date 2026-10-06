# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where the Session meets the agent's TurnLoop: close, turn start and cancel."""

import asyncio
import sqlite3

import coder_test_agents as agents
import pytest
from coder_test_agents import cell, done
from nooa_coder.session.events import TurnStarted
from nooa_coder.session.session import TurnFailedError
from nooa_coder.session.store import SessionStore

from nooa.interactive import Done
from nooa.runtime.turn_loop import TurnCancelled

TIMEOUT = 5


@pytest.mark.parametrize("how", ["submit", "steer"])
async def test_close_starts_no_new_turn_while_children_close(make_session, how):
    """An item a doomed turn took would be recorded as consumed and lost on load."""
    started, block = agents.fresh_events()
    session, _ = make_session(
        cell(agents.BLOCKING_CELL + "return_result(Done(explanation='first'))"), done("second")
    )
    first = asyncio.ensure_future(session.prompt("first"))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    # A queued message, or a steer no model call will see (re-admitted at settle).
    receipt = await (session.submit("QUEUED") if how == "submit" else session.steer("STEERED"))

    children_closing, gate = asyncio.Event(), asyncio.Event()

    async def close_children():
        children_closing.set()
        await gate.wait()

    session._before_close = close_children
    closing = asyncio.ensure_future(session.close())
    await asyncio.wait_for(children_closing.wait(), TIMEOUT)
    block.set()  # the running turn finishes while the children close
    assert await asyncio.wait_for(first, TIMEOUT) == Done(explanation="first")
    await asyncio.sleep(0.1)  # time for the loop to (wrongly) start another turn
    gate.set()
    await asyncio.wait_for(closing, TIMEOUT)

    rows = SessionStore(session.handle.path.parent).load_rows(
        session.id, frozenset({"TurnStarted", "ItemConsumed"})
    )
    assert [kind for kind, _ in rows].count("TurnStarted") == 1
    consumed = {raw["item_id"] for kind, raw in rows if kind == "ItemConsumed"}
    assert receipt.item_id not in consumed  # still unconsumed: a load re-queues it


async def test_a_turn_whose_start_cannot_be_recorded_does_not_run(make_session):
    session, llm = make_session(done("second"))
    record_batch = session.handle.events.record_batch
    failed = []

    def failing_batch(events):
        if any(isinstance(event, TurnStarted) for event in events) and not failed:
            failed.extend(events)
            raise sqlite3.OperationalError("disk full")
        return record_batch(events)

    session.handle.events.record_batch = failing_batch
    with pytest.raises(TurnFailedError) as error:
        await asyncio.wait_for(session.prompt("one"), TIMEOUT)
    assert isinstance(error.value.error, sqlite3.OperationalError)
    assert len(llm.calls) == 0  # the model was never called
    assert session.info.status == "idle"
    assert session._agent.turns.paused
    assert session._agent.queue_manager.get_channel("user_messages").snapshot() == ["one"]
    with pytest.raises(TurnFailedError, match="dispatch is blocked"):
        await session.prompt("two")
    session.resume_dispatch()
    assert await asyncio.wait_for(session.prompt("two"), TIMEOUT) == Done(explanation="second")


async def test_cancel_returns_when_recording_the_cancel_fails(make_session):
    started, _ = agents.fresh_events()
    session, _ = make_session(cell(agents.BLOCKING_CELL), done("after"))
    events = session._agent.event_manager
    add = events.add

    def failing_add(event, **kwargs):
        if isinstance(event, TurnCancelled):
            raise sqlite3.OperationalError("disk full")
        return add(event, **kwargs)

    pending = asyncio.ensure_future(session.prompt("one"))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    events.add = failing_add
    try:
        assert await asyncio.wait_for(session.cancel(), TIMEOUT) is True
    finally:
        del events.add
    with pytest.raises(TurnFailedError, match="recording it failed"):
        await asyncio.wait_for(pending, TIMEOUT)
    # The loop is still alive.
    assert await asyncio.wait_for(session.prompt("two"), TIMEOUT) == Done(explanation="after")
