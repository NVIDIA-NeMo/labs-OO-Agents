# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic consumed/unconsumed steering retention at Session settlement."""

import asyncio

import atom_test_agents as agents
import pytest
from atom_test_agents import cell, done
from nooa_atom.session.items import TurnCancelledOutcome
from nooa_atom.session.store import SessionStore

from nooa.interactive import Done


async def bounded(task):
    return await asyncio.wait_for(task, 10)


@pytest.mark.parametrize("consumed", [False, True])
async def test_true_steer_cancel_retains_only_unconsumed(make_session, consumed):
    started, gate = agents.fresh_events()
    if consumed:
        llm = agents.BlockingLLM([cell(agents.BLOCKING_CELL), done("fence")], block_on=2)
        session, _ = make_session(llm=llm)
    else:
        session, llm = make_session(cell(agents.BLOCKING_CELL), done("pending"), done("fence"))
    ended = asyncio.Event()
    turns = []

    def update(u):
        if u.kind in ("turn_started", "turn_ended"):
            turns.append(u.kind)
        if turns.count("turn_ended") == 2:
            ended.set()

    session.subscribe(update)
    first = asyncio.create_task(session.prompt("FIRST"))
    await bounded(started.wait())
    receipt = await session.steer("STEER")
    if consumed:
        gate.set()
        await bounded(llm.entered.wait())
        assert not session._pending_steers
    assert await bounded(session.cancel())
    assert await bounded(first) == TurnCancelledOutcome(by="user")
    if not consumed:
        await bounded(ended.wait())
        assert turns == ["turn_started", "turn_ended", "turn_started", "turn_ended"]
    assert await bounded(session.prompt("FENCE")) == Done(explanation="fence")
    rows = [
        (k, r.get("channel"))
        for k, r in SessionStore._read_rows(session.handle.path)
        if r.get("item_id") == receipt.item_id
        and k in ("ItemAdmitted", "ItemConsumed", "ItemWithdrawn")
    ]
    expected = [("ItemAdmitted", "steer")]
    if not consumed:
        expected.append(("ItemAdmitted", "user_messages"))
    expected.append(("ItemConsumed", None))
    assert rows == expected
    assert [e.content for e in session.transcript() if e.role == "user"] == [
        "FIRST",
        "STEER",
        "FENCE",
    ]
