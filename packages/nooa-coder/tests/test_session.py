# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session: admission, the turn loop, outcomes, events and close."""

import asyncio
import json
import logging
import sqlite3

import coder_test_agents as agents
import pytest
from coder_test_agents import ask, cell, done, reply, wait_on
from nooa_coder.session import session as session_module
from nooa_coder.session.items import (
    Receipt,
    TurnCancelled,
    TurnCancelledOutcome,
    TurnEndedUpdate,
)
from nooa_coder.session.loader import default_agent_factory
from nooa_coder.session.options import SessionOptions
from nooa_coder.session.session import ItemWithdrawnError, TurnFailedError
from nooa_coder.session.store import SessionStore

from nooa.context_blocks.roles import Role
from nooa.events import Notification, PythonOutput, ResultStatus
from nooa.interactive import Done, NeedInput
from nooa.llm_types import LLMUsage
from nooa.storage.json_snapshot import snapshot_to_json

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


async def test_cancel_during_a_cell_records_the_interrupted_output(make_session, sessions_dir):
    started, _block = agents.fresh_events()
    session, llm = make_session(cell(agents.BLOCKING_CELL), done("second turn"))
    first = asyncio.ensure_future(session.prompt("start"))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    queued = asyncio.ensure_future(session.prompt("queued meanwhile"))
    await asyncio.sleep(0)
    seen = []
    session.subscribe(seen.append)

    assert await asyncio.wait_for(session.cancel(), TIMEOUT) is True
    # cancel() returns only after the cell's cancelled output is recorded.
    events = session.agent.event_manager.values()
    [output] = [
        e
        for e in events
        if isinstance(e, PythonOutput) and e.execution_status is ResultStatus.CANCELLED
    ]
    assert "cell started" in output.stdout
    [cancelled] = [e for e in events if isinstance(e, TurnCancelled)]
    assert (cancelled.by, cancelled.interrupted) == ("user", output.tag)
    assert events.index(cancelled) > events.index(output)
    assert [e.kind for e in seen if e.kind in ("cancelled", "turn_ended")] == [
        "cancelled",
        "turn_ended",
    ]
    assert await asyncio.wait_for(first, TIMEOUT) == TurnCancelledOutcome(by="user")

    # The item queued during the cancelled turn survives and runs next; the
    # model sees the cancel before it.
    assert await asyncio.wait_for(queued, TIMEOUT) == Done(explanation="second turn")
    second_prompt = str(llm.calls[1].messages)
    assert "TurnCancelled" in second_prompt
    assert second_prompt.index("TurnCancelled") < second_prompt.index("queued meanwhile")
    assert ("cancelled", "Stopped by user") in [(e.role, e.content) for e in session.transcript()]

    # After a reload the event is still the model-visible TurnCancelled.
    session_id = session.id
    await session.close()
    with SessionStore(sessions_dir).open(session_id) as handle:
        [reloaded] = [e for e in handle.events.values() if e.event_type == "TurnCancelled"]
    assert isinstance(reloaded, TurnCancelled)
    assert reloaded._role is Role.USER


async def test_cancel_during_a_model_call_has_no_interrupted_cell(make_session):
    llm = agents.BlockingLLM()
    session, _ = make_session(llm=llm)
    pending = asyncio.ensure_future(session.prompt("start"))
    await asyncio.wait_for(llm.entered.wait(), TIMEOUT)
    assert await asyncio.wait_for(session.cancel(by="parent:root"), TIMEOUT) is True
    assert await asyncio.wait_for(pending, TIMEOUT) == TurnCancelledOutcome(by="parent:root")
    [cancelled] = [e for e in session.agent.event_manager.values() if isinstance(e, TurnCancelled)]
    assert (cancelled.by, cancelled.interrupted) == ("parent:root", None)


async def test_idle_cancel_while_waiting_closes_the_prompt_without_an_event(make_session):
    session, _ = make_session(wait_on("jobs"))
    ended = []
    session.subscribe(lambda e: ended.append(e) if e.kind == "turn_ended" else None)
    pending = asyncio.ensure_future(session.prompt("run the job"))
    while not ended:
        await asyncio.sleep(0.01)
    assert await session.cancel() is False
    assert await asyncio.wait_for(pending, TIMEOUT) == TurnCancelledOutcome(by="user")
    assert not [e for e in session.agent.event_manager.values() if isinstance(e, TurnCancelled)]
    assert await session.cancel() is False


async def test_steer_during_a_turn_reaches_the_next_model_call(make_session):
    started, block = agents.fresh_events()
    session, llm = make_session(cell(agents.BLOCKING_CELL), done("steered"))
    pending = asyncio.ensure_future(session.prompt("write the parser"))
    await asyncio.wait_for(started.wait(), TIMEOUT)

    receipt = await session.steer("STEER-focus-on-tests")
    assert (receipt.channel, receipt.delivered) == ("steer", "steered")
    block.set()
    assert await asyncio.wait_for(pending, TIMEOUT) == Done(explanation="steered")

    assert len(llm.calls) == 2
    assert "STEER-focus-on-tests" not in str(llm.calls[0].messages)
    assert "STEER-focus-on-tests" in str(llm.calls[1].messages)
    notes = [e for e in session.agent.event_manager.values() if isinstance(e, Notification)]
    assert [(n.source, n.description) for n in notes] == [("steer:user", "STEER-focus-on-tests")]
    user_lines = [e.content for e in session.transcript() if e.role == "user"]
    assert user_lines == ["write the parser", "STEER-focus-on-tests"]


async def test_steer_after_the_last_model_call_becomes_the_next_message(make_session):
    started, block = agents.fresh_events()
    session, llm = make_session(
        cell(agents.BLOCKING_CELL + "return_result(Done(explanation='first'))"),
        done("second"),
    )
    turns = []
    session.subscribe(lambda e: turns.append(e) if e.kind == "turn_ended" else None)
    pending = asyncio.ensure_future(session.prompt("start"))
    await asyncio.wait_for(started.wait(), TIMEOUT)

    receipt = await session.steer("STEER-late")
    assert receipt.delivered == "steered"
    block.set()
    assert await asyncio.wait_for(pending, TIMEOUT) == Done(explanation="first")
    while len(turns) < 2:
        await asyncio.sleep(0.01)

    assert len(llm.calls) == 2
    assert "STEER-late" not in str(llm.calls[0].messages)
    assert str(llm.calls[1].messages).count("STEER-late") == 1
    assert not [e for e in session.agent.event_manager.values() if isinstance(e, Notification)]
    admitted = [
        raw for _, raw in _rows(session, "ItemAdmitted") if raw["item_id"] == receipt.item_id
    ]
    assert [raw["channel"] for raw in admitted] == ["steer", "user_messages"]
    assert [e.content for e in session.transcript() if e.role == "user"] == ["start", "STEER-late"]


async def test_steer_while_idle_is_a_submit(make_session):
    session, _ = make_session(start=False)
    receipt = await session.steer("hello")
    assert (receipt.channel, receipt.delivered) == ("user_messages", "queued")
    assert session.agent.queue_manager.get_channel("user_messages").qsize() == 1


async def test_a_steer_left_by_a_cancel_is_admitted_after_the_cancel(make_session):
    started, _block = agents.fresh_events()
    session, llm = make_session(cell(agents.BLOCKING_CELL), done("handled the steer"))
    turns = []
    session.subscribe(lambda e: turns.append(e) if e.kind == "turn_ended" else None)
    pending = asyncio.ensure_future(session.prompt("start"))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    receipt = await session.steer("STEER-then-stop")
    assert await asyncio.wait_for(session.cancel(), TIMEOUT) is True
    assert await asyncio.wait_for(pending, TIMEOUT) == TurnCancelledOutcome(by="user")
    while len(turns) < 2:
        await asyncio.sleep(0.01)

    order = [
        (event_type, raw.get("channel"))
        for event_type, raw in _rows(session, "TurnCancelled", "ItemAdmitted")
        if event_type == "TurnCancelled" or raw["item_id"] == receipt.item_id
    ]
    assert order == [
        ("ItemAdmitted", "steer"),
        ("TurnCancelled", None),
        ("ItemAdmitted", "user_messages"),
    ]
    assert "STEER-then-stop" in str(llm.calls[1].messages)


async def test_withdraw_removes_a_queued_item(make_session):
    session, llm = make_session(done("handled"), start=False)
    first = await session.submit("KEEP-ME")
    second = await session.submit("DROP-ME")
    third_prompt = asyncio.ensure_future(session.prompt("DROP-ME"))  # same text, own item
    await asyncio.sleep(0)
    [third_id] = [
        raw["item_id"]
        for _, raw in _rows(session, "ItemAdmitted")
        if raw["item_id"] not in (first.item_id, second.item_id)
    ]
    third = Receipt(
        session_id=session.id, channel="user_messages", item_id=third_id, delivered="queued"
    )

    assert session.withdraw(second) is True
    assert session.withdraw(second) is False
    assert session.withdraw(third) is True
    with pytest.raises(ItemWithdrawnError):
        await asyncio.wait_for(third_prompt, TIMEOUT)
    assert session.agent.queue_manager.get_channel("user_messages").snapshot() == ["KEEP-ME"]
    withdrawn = [raw["item_id"] for _, raw in _rows(session, "ItemWithdrawn")]
    assert withdrawn == [second.item_id, third.item_id]

    session.start()
    ended = []
    session.subscribe(lambda e: ended.append(e) if e.kind == "turn_ended" else None)
    while not ended:
        await asyncio.sleep(0.01)
    assert "KEEP-ME" in str(llm.calls[0].messages)
    assert "DROP-ME" not in str(llm.calls[0].messages)
    assert session.withdraw(first) is False  # consumed


async def test_withdraw_a_buffered_steer(make_session):
    started, block = agents.fresh_events()
    session, llm = make_session(cell(agents.BLOCKING_CELL), done("done"))
    pending = asyncio.ensure_future(session.prompt("start"))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    receipt = await session.steer("STEER-withdrawn")
    assert session.withdraw(receipt) is True
    block.set()
    assert await asyncio.wait_for(pending, TIMEOUT) == Done(explanation="done")
    assert "STEER-withdrawn" not in str(llm.calls[1].messages)
    assert [raw["item_id"] for _, raw in _rows(session, "ItemWithdrawn")] == [receipt.item_id]


def _snapshot_count(path) -> int:
    connection = sqlite3.connect(path)
    try:
        return connection.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]
    finally:
        connection.close()


async def test_checkpoint_is_written_only_when_the_state_changed(
    make_session, sessions_dir, tmp_path
):
    session, _ = make_session(
        done("one"),
        done("two"),
        cell("self.v.answer = 42\nreturn_result(Done(explanation='three'))"),
    )
    await asyncio.wait_for(session.prompt("one"), TIMEOUT)
    after_one = json.dumps(snapshot_to_json(session.agent), sort_keys=True)
    await asyncio.wait_for(session.prompt("two"), TIMEOUT)
    after_two = json.dumps(snapshot_to_json(session.agent), sort_keys=True)
    # Precondition: a no-op turn leaves the serialised state unchanged.
    assert after_one == after_two
    await session.wait_for_checkpoint()
    assert _snapshot_count(session.handle.path) == 1

    await asyncio.wait_for(session.prompt("three"), TIMEOUT)
    await session.wait_for_checkpoint()
    assert _snapshot_count(session.handle.path) == 2

    session_id = session.id
    await session.close()
    with SessionStore(sessions_dir).open(session_id) as handle:
        options = SessionOptions(workspace=tmp_path, agent_spec="coder_test_agents:EchoAgent")
        restored = default_agent_factory(options, handle.storage)
        assert handle.storage.restore_latest_snapshot(restored)
        assert restored.v.answer == 42


async def test_checkpoint_failure_does_not_fail_the_turn(make_session, monkeypatch, caplog):
    def broken(path, blob):
        raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(session_module, "_write_snapshot", broken)
    session, _ = make_session(done("fine"))
    with caplog.at_level(logging.WARNING, logger=session_module.__name__):
        assert await asyncio.wait_for(session.prompt("go"), TIMEOUT) == Done(explanation="fine")
        await session.wait_for_checkpoint()
    assert "checkpoint" in caplog.text.lower()
    assert _snapshot_count(session.handle.path) == 0


async def test_a_user_set_title_wins_over_agent_titles(make_session, sessions_dir):
    session, _ = make_session(start=False)
    seen = []
    session.subscribe(
        lambda e: seen.append((e.title, e.user_set)) if e.kind == "title_changed" else None
    )
    await session.set_title("agent title")
    await session.set_title("My title", user_set=True)
    await session.set_title("later agent title")
    assert (session.info.title, session.info.title_is_user_set) == ("My title", True)
    assert seen == [("agent title", False), ("My title", True)]
    assert SessionStore(sessions_dir).get(session.id).title == "My title"
    await session.set_title("renamed by the user", user_set=True)
    assert session.info.title == "renamed by the user"


async def test_set_mode_records_the_mode_and_tells_listeners(make_session):
    session, _ = make_session(start=False)
    seen = []
    session.subscribe(lambda e: seen.append(e.mode) if e.kind == "mode_changed" else None)
    assert session.info.mode == "auto"
    await session.set_mode("ask")
    assert (session.info.mode, seen) == ("ask", ["ask"])
    with pytest.raises(ValueError):
        await session.set_mode("yolo")


async def test_usage_counts_the_sessions_own_tokens(make_session):
    usage = LLMUsage(input_tokens=100, output_tokens=20, cost_usd=0.5)
    session, _ = make_session(
        cell("x = 1", usage=usage), done("one", usage=usage), done("two", usage=usage)
    )
    ended = []
    session.subscribe(lambda e: ended.append(e) if e.kind == "turn_ended" else None)
    await asyncio.wait_for(session.prompt("one"), TIMEOUT)
    await asyncio.wait_for(session.prompt("two"), TIMEOUT)
    assert (session.info.usage.input_tokens, session.info.usage.output_tokens) == (300, 60)
    assert session.info.usage.cost_usd == pytest.approx(1.5)
    assert [e.usage.input_tokens for e in ended] == [200, 100]
    assert [raw["usage"]["output_tokens"] for _, raw in _rows(session, "TurnEnded")] == [40, 20]
