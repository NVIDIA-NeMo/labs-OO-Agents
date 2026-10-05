# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session durability, lifecycle and prompt outcomes at dispatch boundaries."""

import asyncio
import sqlite3

import coder_test_agents as agents
import pytest
from coder_test_agents import cell, done
from nooa_coder.session.events import ItemConsumed
from nooa_coder.session.items import TurnCancelledOutcome
from nooa_coder.session.session import TurnFailedError

from nooa.interactive import Done

TIMEOUT = 3


async def wait_until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), TIMEOUT)


def rows(session, *types):
    return session.handle._store.load_rows(session.id, frozenset(types))


@pytest.mark.parametrize("action", ["cancel", "close"])
async def test_preparation_cancel_keeps_admission_and_resolves_prompt(make_session, action):
    session, llm = make_session(done("unused"), start=False)
    entered = asyncio.Event()

    async def prepare():
        entered.set()
        await asyncio.Event().wait()

    session._agent.turns.start(prepare=prepare, commit=session._commit_turn)
    pending = asyncio.ensure_future(session.prompt("queued"))
    await asyncio.wait_for(entered.wait(), TIMEOUT)
    await asyncio.wait_for(getattr(session, action)(), TIMEOUT)
    assert isinstance(await asyncio.wait_for(pending, TIMEOUT), TurnCancelledOutcome)
    assert session._agent.queue_manager.get_channel("user_messages").snapshot() == ["queued"]
    assert not llm.calls
    assert rows(session, "ItemConsumed", "TurnStarted") == []


async def test_idle_close_pauses_before_children_and_no_input_consumed(make_session):
    session, llm = make_session(done("unused"))
    entered, release = asyncio.Event(), asyncio.Event()

    async def before_close():
        entered.set()
        await release.wait()

    session._before_close = before_close
    receipt = await session.submit("queued")
    closing = asyncio.ensure_future(session.close())
    await asyncio.wait_for(entered.wait(), TIMEOUT)
    await asyncio.sleep(0.02)
    assert not llm.calls
    assert session._agent.queue_manager.get_channel("user_messages").snapshot() == ["queued"]
    assert rows(session, "ItemConsumed", "TurnStarted") == []
    release.set()
    await asyncio.wait_for(closing, TIMEOUT)
    assert isinstance(await session.outcome(receipt.item_id), TurnCancelledOutcome)


async def test_sqlite_commit_failure_is_atomic_blocks_and_fails_outcome(make_session):
    session, llm = make_session(done("retry"))
    with sqlite3.connect(session.handle.path) as connection:
        connection.execute(
            "CREATE TRIGGER fail_start BEFORE INSERT ON events "
            "WHEN NEW.event_type = 'TurnStarted' "
            "BEGIN SELECT RAISE(ABORT, 'start failed'); END"
        )
    receipt = await session.submit("first")
    pending = asyncio.ensure_future(session.outcome(receipt.item_id))
    with pytest.raises(TurnFailedError, match="start failed"):
        await asyncio.wait_for(pending, TIMEOUT)
    assert session._agent.turns.paused and isinstance(
        session._agent.turns.dispatch_error, sqlite3.IntegrityError
    )
    assert not llm.calls
    assert rows(session, "ItemConsumed", "TurnStarted") == []
    assert session._agent.queue_manager.get_channel("user_messages").snapshot() == ["first"]
    await asyncio.sleep(0.02)
    assert not llm.calls
    with pytest.raises(TurnFailedError, match="dispatch is blocked"):
        await session.prompt("rejected")
    with sqlite3.connect(session.handle.path) as connection:
        connection.execute("DROP TRIGGER fail_start")
    session.resume_dispatch()
    assert await asyncio.wait_for(session.prompt("second"), TIMEOUT) == Done(explanation="retry")
    assert len(rows(session, "TurnStarted")) == 1
    assert len(rows(session, "ItemConsumed")) == 2
    # The first caller's already reported failure is not silently rewritten by retry.
    with pytest.raises(TurnFailedError, match="start failed"):
        await session.outcome(receipt.item_id)


@pytest.mark.parametrize("where", ["settled", "listener"])
async def test_observer_cancelled_error_cannot_skip_prompt_or_close_cleanup(make_session, where):
    session, _ = make_session(done("first"))

    def broken(_):
        raise asyncio.CancelledError()

    if where == "settled":
        handlers = session._agent.event_manager._handlers["TurnSettled"]
        handlers.insert(0, broken)  # explicitly before mandatory Session subscriber
    else:
        session.subscribe(broken)
    assert await asyncio.wait_for(session.prompt("one"), TIMEOUT) == Done(explanation="first")
    await asyncio.wait_for(session.close(), TIMEOUT)
    assert session.handle.closed and session.closing


async def test_midturn_get_record_failure_orphan_does_not_steal_same_object_retry(make_session):
    started, block = agents.fresh_events()
    script = (
        agents.BLOCKING_CELL
        + "extra = await self.user_messages.get()\nreturn_result(Done(explanation='done'))"
    )
    session, _ = make_session(cell(script), done("retry"))
    first = asyncio.ensure_future(session.prompt("first"))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    same_object = {"extra": "same object"}
    receipt = await session.submit(same_object)
    extra = asyncio.ensure_future(session.outcome(receipt.item_id))
    add = session.handle.events.add

    def failing_add(event, **kwargs):
        if isinstance(event, ItemConsumed) and event.item_id == receipt.item_id:
            raise OSError("midturn record failed")
        return add(event, **kwargs)

    session.handle.events.add = failing_add
    block.set()
    for pending in (first, extra):
        with pytest.raises(TurnFailedError, match="consumption recording failed"):
            await asyncio.wait_for(pending, TIMEOUT)
    assert not session._ids["user_messages"]
    channel, orphan, error = session._orphans[receipt.item_id]
    assert channel == "user_messages" and orphan is same_object
    assert isinstance(error, OSError)
    assert session._agent.turns.paused
    assert not any(raw["item_id"] == receipt.item_id for _, raw in rows(session, "ItemConsumed"))
    session.handle.events.add = add
    session.resume_dispatch()
    retry = await session.submit(same_object)  # exact identity, not a serialized copy
    assert await asyncio.wait_for(session.outcome(retry.item_id), TIMEOUT) == Done(
        explanation="retry"
    )
    consumed_ids = {raw["item_id"] for _, raw in rows(session, "ItemConsumed")}
    assert retry.item_id in consumed_ids and receipt.item_id not in consumed_ids
    assert session._agent.queue_manager.get_channel("user_messages").snapshot() == []
    with pytest.raises(TurnFailedError):
        await session.outcome(receipt.item_id)


async def test_recursive_session_close_from_cleanup_child_does_not_deadlock(make_session):
    session, _ = make_session()

    async def cleanup():
        await asyncio.create_task(session.close())

    session._agent.event_manager.on_close(cleanup)
    await asyncio.wait_for(session.close(), TIMEOUT)
    assert session.handle.closed


async def test_preparation_steer_is_queued_not_stranded(make_session):
    session, _ = make_session(done("prepared"), start=False)
    entered, release = asyncio.Event(), asyncio.Event()

    async def prepare():
        entered.set()
        await release.wait()

    session._agent.turns.start(prepare=prepare, commit=session._commit_turn)
    first = asyncio.ensure_future(session.prompt("first"))
    await entered.wait()
    receipt = await session.steer("during prepare")
    assert receipt.delivered == "queued"
    release.set()
    assert await asyncio.wait_for(first, TIMEOUT) == Done(explanation="prepared")
    assert await session.outcome(receipt.item_id) == Done(explanation="prepared")
    assert session._pending_steers == []


async def test_model_swap_attempt_once_retains_old_cleanup_on_cancel(make_session):
    session, _ = make_session(done("unused"), start=False)
    entered = asyncio.Event()

    class Old:
        async def aclose(self):
            entered.set()
            await asyncio.Event().wait()

    old = Old()
    session._owned_llm = old
    new = session._agent.llm
    session._pending_model = ("new", new)
    session.start()
    pending = asyncio.ensure_future(session.prompt("queued"))
    await asyncio.wait_for(entered.wait(), TIMEOUT)
    assert session._agent.llm is new and session.info.model == "new"
    assert session._pending_model is None and old in session._retired_llms
    assert await asyncio.wait_for(session.cancel(), TIMEOUT)
    assert isinstance(await asyncio.wait_for(pending, TIMEOUT), TurnCancelledOutcome)
    assert session._agent.queue_manager.get_channel("user_messages").snapshot() == ["queued"]
    assert old in session._retired_llms
    # Release the test double for final cleanup (production ownership remains retained).
    session._retired_llms.remove(old)


async def test_model_activation_failure_is_not_retried_implicitly(make_session, monkeypatch):
    session, _ = make_session(done("after failure"), start=False)
    old = session._agent.llm
    from nooa_coder.session import session as session_module

    from nooa.unifiedllm import FakeLLMClient

    new = FakeLLMClient([])
    session._pending_model = ("bad", new)
    attempts = []

    def failing(agent):
        attempts.append(agent.llm)
        raise ValueError("bad swap")

    monkeypatch.setattr(session_module, "apply_model_limits", failing)
    session.start()
    with pytest.raises(TurnFailedError, match="bad swap"):
        await asyncio.wait_for(session.prompt("first"), TIMEOUT)
    assert session._agent.llm is old and session._pending_model is None
    assert new in session._retired_llms and attempts == [new]
    session.resume_dispatch()
    assert await asyncio.wait_for(session.prompt("second"), TIMEOUT) == Done(
        explanation="after failure"
    )
    assert attempts == [new]


async def test_model_metadata_failure_retains_client_and_prior_pending(make_session, monkeypatch):
    session, _ = make_session(start=False)

    class Client:
        def __init__(self):
            self.closed = False

        async def aclose(self):
            self.closed = True

    client, prior = Client(), Client()
    session._pending_model = ("prior", prior)
    session._llm_factory = lambda _alias, _workspace: client

    def fail(_alias):
        raise OSError("metadata failed")

    monkeypatch.setattr(session.handle, "set_model", fail)
    with pytest.raises(OSError, match="metadata failed"):
        await session.set_model("new")
    assert session._pending_model == ("prior", prior)
    assert client in session._retired_llms
    await session.close()
    assert client.closed and prior.closed


@pytest.mark.parametrize("failure", ["consumed", "leftover"])
async def test_steer_record_failure_retains_buffer_and_fails_relevant_prompt(make_session, failure):
    started, block = agents.fresh_events()
    session, _ = make_session(cell(agents.BLOCKING_CELL), done("finished"))
    first = asyncio.ensure_future(session.prompt("first"))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    receipts = [await session.steer("steer-a"), await session.steer("steer-b")]
    pending = [asyncio.ensure_future(session.outcome(r.item_id)) for r in receipts]
    add = session.handle.events.add
    from nooa_coder.session.events import ItemAdmitted

    def fail(event, **kwargs):
        if (
            failure == "consumed"
            and isinstance(event, ItemConsumed)
            and event.item_id == receipts[0].item_id
        ):
            raise OSError("steer record failed")
        if (
            failure == "leftover"
            and isinstance(event, ItemAdmitted)
            and event.item_id == receipts[0].item_id
        ):
            raise OSError("steer record failed")
        return add(event, **kwargs)

    session.handle.events.add = fail
    if failure == "consumed":
        block.set()  # next model call attempts to flush both steers
    else:
        cancelling = asyncio.create_task(session.cancel())  # settling requeues leftovers
        await asyncio.wait_for(cancelling, TIMEOUT)
    for outcome in [first, *pending]:
        with pytest.raises(TurnFailedError):
            await asyncio.wait_for(outcome, TIMEOUT)
    assert session._agent.turns.paused
    assert [item_id for item_id, _, _ in session._pending_steers] == [r.item_id for r in receipts]
    assert not any(
        raw["item_id"] in {r.item_id for r in receipts} for _, raw in rows(session, "ItemConsumed")
    )
    session.handle.events.add = add


async def test_failed_steer_notification_retains_unconsumed_recovery(make_session, monkeypatch):
    from nooa.events import Notification

    started, block = agents.fresh_events()
    session, _ = make_session(cell(agents.BLOCKING_CELL), done("finished"))
    first = asyncio.ensure_future(session.prompt("first"))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    receipts = [await session.steer("steer-a"), await session.steer("steer-b")]
    updates = []
    session.subscribe(updates.append)
    add = session._agent.event_manager.add

    def fail(event, **kwargs):
        if isinstance(event, Notification):
            raise OSError("notification unavailable")
        return add(event, **kwargs)

    monkeypatch.setattr(session._agent.event_manager, "add", fail)
    block.set()
    with pytest.raises(TurnFailedError):
        await asyncio.wait_for(first, TIMEOUT)
    assert session._agent.turns.paused
    ids = [r.item_id for r in receipts]
    assert [item_id for item_id, _, _ in session._pending_steers] == ids
    assert not any(raw["item_id"] in ids for _, raw in rows(session, "ItemConsumed"))
    assert not any(e.kind == "item_consumed" and e.item_id in ids for e in updates)
    for receipt in receipts:
        with pytest.raises(TurnFailedError):
            await session.outcome(receipt.item_id)
    monkeypatch.setattr(session._agent.event_manager, "add", add)
    session.resume_dispatch()
    await agents.until(
        lambda: all(
            any(raw["item_id"] == item_id for _, raw in rows(session, "ItemConsumed"))
            for item_id in ids
        ),
        TIMEOUT,
    )
    assert session._pending_steers == []


@pytest.mark.parametrize("failure", ["before_delivery", "after_delivery"])
async def test_failed_steer_notification_replays_on_load(
    registry, root_options, models, sessions_dir, monkeypatch, failure
):
    from nooa_coder.session.registry import SessionRegistry
    from nooa_coder.session.store import SessionStore

    from nooa.events import Notification

    started, block = agents.fresh_events()
    models.scripts[None] = [cell(agents.BLOCKING_CELL), done("finished")]
    root = await registry.create(root_options)
    first = asyncio.ensure_future(root.prompt("first"))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    receipt = await root.steer("recover-this-steer")
    add = root._agent.event_manager.add
    delivered = []
    root._agent.event_manager.on("Notification", delivered.append)

    def fail(event, **kwargs):
        if isinstance(event, Notification):
            if failure == "after_delivery":
                add(event, **kwargs)
            raise OSError("notification delivery ambiguous")
        return add(event, **kwargs)

    monkeypatch.setattr(root._agent.event_manager, "add", fail)
    block.set()
    with pytest.raises(TurnFailedError):
        await asyncio.wait_for(first, TIMEOUT)
    assert not any(raw["item_id"] == receipt.item_id for _, raw in rows(root, "ItemConsumed"))
    assert bool(delivered) == (failure == "after_delivery")
    await registry.close_all()
    later = agents.ScriptedModels({None: [done("replayed")]})
    fresh = SessionRegistry(SessionStore(sessions_dir), agent_factory=later)
    try:
        loaded = await fresh.load(root.id)
        await asyncio.wait_for(loaded.outcome(receipt.item_id), TIMEOUT)
        assert "recover-this-steer" in str(later.llms[None].calls[0].messages)
        assert any(raw["item_id"] == receipt.item_id for _, raw in rows(loaded, "ItemRequeued"))
    finally:
        await fresh.close_all()
