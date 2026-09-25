# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SessionRegistry: the tree of live sessions and their files."""

import asyncio
import contextlib
import sqlite3

import pytest
from coder_test_agents import (
    BLOCKING_CELL,
    ModelFactory,
    ScriptedModels,
    TrackedLLM,
    cell,
    done,
    fresh_events,
    until,
)
from nooa_coder.session.events import TurnEnded
from nooa_coder.session.registry import (
    ChildActiveElsewhereError,
    DepthLimitError,
    SessionRegistry,
)
from nooa_coder.session.session import SessionClosedError, TurnFailedError
from nooa_coder.session.store import SessionStore

from nooa.interactive import Done

TIMEOUT = 20


def _db_files(sessions_dir):
    return sorted(p.name for p in sessions_dir.glob("*.db")) if sessions_dir.exists() else []


async def test_create_publishes_a_started_session(registry, root_options, models):
    models.scripts[None] = [done("hello back")]
    root = await registry.create(root_options)
    assert registry.get(root.id) is root
    assert (root.parent_id, root.depth) == (None, 0)
    assert await asyncio.wait_for(root.prompt("hello"), TIMEOUT) == Done(explanation="hello back")
    assert models.built == [root_options]


async def test_a_failing_build_leaves_no_file_and_no_reservation(
    registry, root_options, sessions_dir
):
    failing = root_options.model_copy(update={"agent_spec": "coder_test_agents:FailingAgent"})
    real = SessionRegistry(registry.store)  # default factory: really imports the spec
    with pytest.raises(RuntimeError, match="construction failed"):
        await real.create(failing)
    assert _db_files(sessions_dir) == []
    assert real.sessions == {} and real._reserved == {}
    assert real.list() == []


async def test_depth_is_capped_by_the_options(registry, root_options, sessions_dir):
    root = await registry.create(root_options.model_copy(update={"max_depth": 1}))
    child = await registry.create(root.options.inherit(name="child"), parent_id=root.id)
    assert child.depth == 1
    before = _db_files(sessions_dir)
    with pytest.raises(DepthLimitError):
        await registry.create(child.options.inherit(name="grandchild"), parent_id=child.id)
    assert _db_files(sessions_dir) == before


async def test_the_parent_is_told_about_a_new_child(registry, root_options):
    root = await registry.create(root_options)
    seen = []
    root.subscribe(lambda e: seen.append(e) if e.kind == "child_created" else None)
    child = await registry.create(
        root.options.inherit(name="Review auth", retain=True), parent_id=root.id
    )
    [created] = seen
    assert (created.child_id, created.name, created.depth, created.retained) == (
        child.id,
        "Review auth",
        1,
        True,
    )


async def test_initial_items_are_in_the_first_notification(registry, root_options, models):
    models.scripts["child"] = [done("got both")]
    root = await registry.create(root_options)
    root.agent.queue_manager.queue("context")
    child_options = root.options.inherit(name="child")

    def add_context_channel(options, storage):
        agent = models(options, storage)
        agent.queue_manager.queue("context")
        return agent

    registry._agent_factory = add_context_channel
    seen = []
    child = await registry.create(
        child_options,
        parent_id=root.id,
        initial_items=[("user_messages", "THE-PROMPT"), ("context", {"k": "THE-CONTEXT"})],
    )
    child.subscribe(lambda e: seen.append(e) if e.kind == "turn_ended" else None)
    await until(lambda: seen)
    first_call = str(models.llms["child"].calls[0].messages)
    assert "THE-PROMPT" in first_call and "THE-CONTEXT" in first_call


async def test_list_and_children_report_live_and_on_disk_status(registry, root_options, models):
    started, block = fresh_events()
    models.scripts["busy"] = [cell(BLOCKING_CELL + "return_result(Done(explanation='x'))")]
    root = await registry.create(root_options)
    kept = await registry.create(root.options.inherit(name="kept", retain=True), parent_id=root.id)
    busy = await registry.create(root.options.inherit(name="busy"), parent_id=root.id)
    gone = await registry.create(root.options.inherit(name="gone"), parent_id=root.id)
    await registry.close(gone.id)
    await busy.submit("work")
    await asyncio.wait_for(started.wait(), TIMEOUT)

    statuses = {info.name: info.status for info in registry.children(root.id)}
    assert statuses == {"kept": "retained", "busy": "running", "gone": "on_disk"}
    [listed] = registry.list()
    assert (listed.id, listed.status) == (root.id, "idle")
    assert {info.id for info in registry.list(roots_only=False)} == {
        root.id,
        kept.id,
        busy.id,
        gone.id,
    }
    assert registry.list(workspace=root_options.workspace / "elsewhere") == []
    block.set()


async def test_close_goes_children_first(registry, root_options):
    root = await registry.create(root_options)
    child = await registry.create(root.options.inherit(name="child"), parent_id=root.id)
    grandchild = await registry.create(child.options.inherit(name="grandchild"), parent_id=child.id)
    other = await registry.create(root_options.model_copy(update={"name": "other"}))
    order = []
    for session in (root, child, grandchild, other):
        session.subscribe(lambda e, s=session: order.append(s.name) if e.kind == "closed" else None)

    await registry.close(root.id)
    assert order == ["grandchild", "child", None]
    assert set(registry.sessions) == {other.id}
    assert registry.get(root.id) is None

    await registry.close_all()
    assert order[-1] == "other" and registry.sessions == {}
    store = SessionStore(root_options.sessions_dir)
    assert {info.status for info in store.list(roots_only=False)} == {"on_disk"}


async def test_load_of_a_live_id_attaches_to_the_same_session(registry, root_options):
    root = await registry.create(root_options)
    assert await registry.load(root.id) is root


async def test_load_restores_state_and_requeues_unhandled_items(
    registry, root_options, sessions_dir, models
):
    started, block = fresh_events()
    models.scripts[None] = [
        cell("self.v.note = 'kept'\nreturn_result(Done(explanation='noted'))"),
        cell(BLOCKING_CELL),
        cell(BLOCKING_CELL),
    ]
    root = await registry.create(root_options)
    assert await asyncio.wait_for(root.prompt("first"), TIMEOUT) == Done(explanation="noted")
    await root.wait_for_checkpoint()

    second = asyncio.ensure_future(root.prompt("second"))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    await root.steer("STEER-SEEN")
    later = await root.submit("LATER")
    withdrawn = await root.submit("WITHDRAWN")
    assert root.withdraw(withdrawn)
    started_again, _ = fresh_events()
    block.set()  # the next model call sees the steer, then the cell blocks again
    await asyncio.wait_for(started_again.wait(), TIMEOUT)
    assert "STEER-SEEN" in str(models.llms[None].calls[2].messages)
    await registry.close_all()
    await second

    resumed_models = ScriptedModels({None: [done("resumed")]})
    resumed_events = []

    def factory(options, storage):
        agent = resumed_models(options, storage)
        agent.event_manager.on("TuiSessionResumed", resumed_events.append)
        return agent

    fresh = SessionRegistry(SessionStore(sessions_dir), agent_factory=factory)
    try:
        loaded = await fresh.load(root.id)
        ended = []
        loaded.subscribe(lambda e: ended.append(e) if e.kind == "turn_ended" else None)
        await until(lambda: ended)
        assert loaded.agent.v.note == "kept"
        [resumed] = resumed_events
        assert (resumed.session_id, resumed.restored) == (root.id, True)
        assert "LATER" in str(resumed_models.llms[None].calls[0].messages)
        # Only the unhandled item is re-queued; the withdrawn one and the steer
        # the model already saw are not.
        turns = fresh.store.load_rows(root.id, frozenset({"TurnStarted"}))
        assert turns[-1][1]["item_ids"] == [later.item_id]
        requeued = [
            raw["item_id"] for _, raw in fresh.store.load_rows(root.id, frozenset({"ItemRequeued"}))
        ]
        assert requeued == [later.item_id]
    finally:
        await fresh.close_all()


async def test_concurrent_loads_share_one_session(registry, root_options, sessions_dir):
    root = await registry.create(root_options)
    await registry.close_all()
    models = ScriptedModels()
    fresh = SessionRegistry(SessionStore(sessions_dir), agent_factory=models)
    try:
        first, second = await asyncio.gather(fresh.load(root.id), fresh.load(root.id))
        assert first is second
        assert len(models.built) == 1
    finally:
        await fresh.close_all()


async def test_loading_a_parent_whose_child_is_live_elsewhere_is_refused(
    registry, root_options, sessions_dir
):
    root = await registry.create(root_options)
    child = await registry.create(
        root.options.inherit(name="child", retain=True), parent_id=root.id
    )
    await registry.close_all()

    elsewhere = SessionRegistry(SessionStore(sessions_dir), agent_factory=ScriptedModels())
    here = SessionRegistry(SessionStore(sessions_dir), agent_factory=ScriptedModels())
    try:
        detached = await elsewhere.load(child.id)  # a detached child is allowed
        assert detached.parent_id == root.id
        with pytest.raises(ChildActiveElsewhereError, match=child.id):
            await here.load(root.id)
        assert here.sessions == {} and here._reserved == {}
        await elsewhere.close_all()
        loaded = await here.load(root.id)
        assert loaded.id == root.id
    finally:
        await elsewhere.close_all()
        await here.close_all()


async def test_a_child_opened_elsewhere_between_check_and_open_is_caught(
    registry, root_options, sessions_dir, monkeypatch
):
    root = await registry.create(root_options)
    child = await registry.create(
        root.options.inherit(name="child", retain=True), parent_id=root.id
    )
    await registry.close_all()

    here = SessionRegistry(SessionStore(sessions_dir), agent_factory=ScriptedModels())
    elsewhere = SessionStore(sessions_dir)
    held = []
    real_open = here.store.open

    def open_after_the_other_process(session_id):
        # The other process opens the child right after this one checked.
        if not held:
            held.append(elsewhere.open(child.id))
        return real_open(session_id)

    monkeypatch.setattr(here.store, "open", open_after_the_other_process)
    try:
        with pytest.raises(ChildActiveElsewhereError, match=child.id):
            await here.load(root.id)
        assert here.sessions == {} and here._reserved == {}
        assert not here.store.is_active(root.id)
    finally:
        for handle in held:
            handle.close()
        await here.close_all()


async def test_a_child_cannot_be_loaded_while_its_parent_is_live_elsewhere(
    registry, root_options, sessions_dir
):
    root = await registry.create(root_options)
    child = await registry.create(
        root.options.inherit(name="child", retain=True), parent_id=root.id
    )
    grandchild = await registry.create(
        child.options.inherit(name="grandchild", retain=True), parent_id=child.id
    )
    await registry.close(child.id)  # the root stays live in `registry`

    other = SessionRegistry(SessionStore(sessions_dir), agent_factory=ScriptedModels())
    try:
        with pytest.raises(ChildActiveElsewhereError, match=root.id):
            await other.load(child.id)
        with pytest.raises(ChildActiveElsewhereError, match=root.id):
            await other.load(grandchild.id)
        assert other.sessions == {}
        # The registry that has the parent live can open it.
        assert (await registry.open_child(root, child.id)).id == child.id
    finally:
        await other.close_all()


async def test_delete_closes_keeps_files_and_leaves_a_tombstone(
    registry, root_options, sessions_dir
):
    root = await registry.create(root_options)
    child = await registry.create(root.options.inherit(name="child"), parent_id=root.id)
    await registry.delete(child.id)
    assert registry.get(child.id) is None
    assert (sessions_dir / f"{child.id}.db").exists()
    [(_, tombstone)] = registry.store.load_rows(root.id, frozenset({"ChildDeleted"}))
    assert (tombstone["child_id"], tombstone["name"]) == (child.id, "child")

    other = await registry.create(root.options.inherit(name="other"), parent_id=root.id)
    await registry.delete(other.id, keep_files=False)
    assert not (sessions_dir / f"{other.id}.db").exists()


async def test_the_llm_factory_builds_owned_clients(root_options, sessions_dir):
    factory = ModelFactory(
        {
            "alias-a": [
                [
                    cell(
                        "c = await self.session.delegate('Other', 'x', model='alias-b')\n"
                        "await c.wait()\n"
                        "return_result(Done(explanation='ok'))"
                    )
                ]
            ],
            "alias-b": [[done("child ok")]],
        }
    )
    registry = SessionRegistry(SessionStore(sessions_dir), llm_factory=factory)
    try:
        root = await registry.create(root_options.model_copy(update={"model": "alias-a"}))
        [root_llm] = factory.made
        assert root.agent.llm is root_llm
        assert factory.calls == [("alias-a", root_options.workspace)]
        assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="ok")
        child_llm = factory.made[1]
        assert (child_llm.alias, factory.calls[1][0]) == ("alias-b", "alias-b")
        await asyncio.wait_for(_until_closed(child_llm), TIMEOUT)  # throwaway child closed
        assert not root_llm.closed
    finally:
        await registry.close_all()
    assert root_llm.closed


async def test_a_given_client_is_not_rebuilt_or_closed(root_options, sessions_dir):
    factory = ModelFactory()
    given = TrackedLLM("given", [])
    registry = SessionRegistry(SessionStore(sessions_dir), llm_factory=factory)
    root = await registry.create(root_options.model_copy(update={"model": "alias-a", "llm": given}))
    assert root.agent.llm is given and factory.calls == []
    await registry.close_all()
    assert not given.closed


async def _until_closed(llm):
    await until(lambda: llm.closed)


async def test_set_model_swaps_the_client_before_the_next_turn(root_options, sessions_dir):
    started, block = fresh_events()
    factory = ModelFactory(
        {
            "alias-a": [[cell(BLOCKING_CELL + "return_result(Done(explanation='on a'))")]],
            "alias-b": [[done("on b")]],
        }
    )
    registry = SessionRegistry(SessionStore(sessions_dir), llm_factory=factory)
    try:
        root = await registry.create(root_options.model_copy(update={"model": "alias-a"}))
        first = asyncio.ensure_future(root.prompt("one"))
        await asyncio.wait_for(started.wait(), TIMEOUT)
        await root.set_model("alias-b")  # built now; swapped in at the next turn
        assert len(factory.made) == 2 and root.agent.llm is factory.made[0]
        with pytest.raises(ValueError, match="bad-alias"):
            await root.set_model("bad-alias")  # fails at the call
        block.set()
        assert await asyncio.wait_for(first, TIMEOUT) == Done(explanation="on a")
        assert await asyncio.wait_for(root.prompt("two"), TIMEOUT) == Done(explanation="on b")
        old, new = factory.made
        assert (old.closed, new.closed) == (True, False)
        assert root.agent.llm is new
        assert (root.info.model, root.options.model) == ("alias-b", "alias-b")
    finally:
        await registry.close_all()
    assert new.closed


async def test_a_same_model_child_shares_the_parents_client(root_options, sessions_dir):
    factory = ModelFactory()
    registry = SessionRegistry(SessionStore(sessions_dir), llm_factory=factory)
    try:
        root = await registry.create(root_options.model_copy(update={"model": "alias-a"}))
        [root_llm] = factory.made
        kid = await registry.create(
            root.options.inherit(name="kid", retain=True), parent_id=root.id
        )
        assert kid.agent.llm is root_llm and len(factory.calls) == 1
        await registry.close(kid.id)
        assert not root_llm.closed  # the child did not own it
        reopened = await registry.open_child(root, kid.id)
        assert reopened.agent.llm is root_llm and len(factory.calls) == 1
        other = await registry.create(
            root.options.inherit(name="other", model="alias-b"), parent_id=root.id
        )
        assert other.agent.llm is not root_llm and len(factory.calls) == 2
    finally:
        await registry.close_all()
    assert root_llm.closed


async def test_set_model_keeps_a_client_a_child_still_uses(root_options, sessions_dir):
    factory = ModelFactory({"alias-b": [[done("on b")]]})
    registry = SessionRegistry(SessionStore(sessions_dir), llm_factory=factory)
    try:
        root = await registry.create(root_options.model_copy(update={"model": "alias-a"}))
        kid = await registry.create(
            root.options.inherit(name="kid", retain=True), parent_id=root.id
        )
        await root.set_model("alias-b")
        assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="on b")
        old, new = factory.made
        assert kid.agent.llm is old and not old.closed
        assert root.agent.llm is new
    finally:
        await registry.close_all()
    assert old.closed and new.closed


@pytest.mark.parametrize("error", [RuntimeError("aclose failed"), asyncio.CancelledError()])
async def test_the_owned_client_closes_when_the_agent_close_fails(
    root_options, sessions_dir, monkeypatch, error
):
    factory = ModelFactory()
    registry = SessionRegistry(SessionStore(sessions_dir), llm_factory=factory)
    root = await registry.create(root_options.model_copy(update={"model": "alias-a"}))

    async def broken_aclose():
        raise error

    # Agent.aclose() awaits the event manager's aclose(), which can raise.
    monkeypatch.setattr(root.agent.event_manager, "aclose", broken_aclose)
    await asyncio.wait_for(registry.close(root.id), TIMEOUT)
    [llm] = factory.made
    assert llm.closed
    assert root.handle._closed and root.info.status == "closed"
    assert registry.get(root.id) is None


async def test_a_failing_handle_close_still_closes_the_session(registry, root_options):
    root = await registry.create(root_options)
    real_close = root.handle.close
    closed = []
    root.subscribe(lambda e: closed.append(e) if e.kind == "closed" else None)

    def broken_close():
        raise sqlite3.OperationalError("database is locked")

    root.handle.close = broken_close
    try:
        await asyncio.wait_for(registry.close(root.id), TIMEOUT)
        await asyncio.wait_for(root.close(), TIMEOUT)  # idempotent: no cached error
        assert len(closed) == 1 and root.info.status == "closed"
        assert registry.get(root.id) is None
    finally:
        real_close()


async def test_mode_and_model_changes_survive_a_reload(root_options, sessions_dir):
    factory = ModelFactory()
    registry = SessionRegistry(SessionStore(sessions_dir), llm_factory=factory)
    try:
        root = await registry.create(root_options.model_copy(update={"model": "alias-a"}))
        await root.set_mode("ask")
        await root.set_model("alias-b")
        kid = await registry.create(root.options.inherit(name="kid"), parent_id=root.id)
    finally:
        await registry.close_all()
    stored = registry.store.get(root.id)
    assert (stored.mode, stored.model) == ("ask", "alias-b")
    assert registry.store.get(kid.id).mode == "ask"  # inherited at creation, and recorded

    fresh_factory = ModelFactory()
    fresh = SessionRegistry(SessionStore(sessions_dir), llm_factory=fresh_factory)
    try:
        loaded = await fresh.load(root.id)
        assert (loaded.options.permission_mode, loaded.info.mode) == ("ask", "ask")
        assert (loaded.options.model, loaded.info.model) == ("alias-b", "alias-b")
        assert fresh_factory.calls[0][0] == "alias-b"
    finally:
        await fresh.close_all()


async def test_set_model_needs_an_llm_factory(registry, root_options):
    root = await registry.create(root_options)
    with pytest.raises(RuntimeError, match="llm_factory"):
        await root.set_model("alias-b")


async def test_prepare_runs_before_publish_and_start(registry, root_options, models, sessions_dir):
    models.scripts["child"] = [done("first turn")]
    root = await registry.create(root_options)
    seen = []

    async def prepare(session):
        assert registry.get(session.id) is None  # not published yet
        assert session._loop_task is None  # not started yet
        session.subscribe(seen.append)

    child = await registry.create(
        root.options.inherit(name="child"),
        parent_id=root.id,
        initial_items=[("user_messages", "go")],
        prepare=prepare,
    )
    await _until_turn_ended(seen)
    assert [e.kind for e in seen if e.kind in ("item_admitted", "turn_started", "turn_ended")] == [
        "item_admitted",
        "turn_started",
        "turn_ended",
    ]
    assert registry.get(child.id) is child

    async def broken(session):
        raise RuntimeError("bridge failed")

    before = _db_files(sessions_dir)
    with pytest.raises(RuntimeError, match="bridge failed"):
        await registry.create(root.options.inherit(name="x"), parent_id=root.id, prepare=broken)
    assert _db_files(sessions_dir) == before
    assert registry._reserved == {}


async def test_prepare_on_load_sees_requeued_turns_but_not_on_attach(
    registry, root_options, sessions_dir
):
    root = await registry.create(root_options)
    root_id = root.id
    await root.cancel()
    # Leave an unhandled item behind: close before the loop can consume it.
    root._loop_task.cancel()
    await root.submit("LEFT-BEHIND")
    await registry.close_all()

    fresh = SessionRegistry(
        SessionStore(sessions_dir), agent_factory=ScriptedModels({None: [done("resumed")]})
    )
    seen = []
    calls = []

    async def prepare(session):
        calls.append(session.id)
        session.subscribe(seen.append)

    try:
        loaded = await fresh.load(root_id, prepare=prepare)
        await _until_turn_ended(seen)
        [admitted] = [e for e in seen if e.kind == "item_admitted"]
        assert admitted.channel == "user_messages"
        assert calls == [root_id]
        assert await fresh.load(root_id, prepare=prepare) is loaded
        assert calls == [root_id]
    finally:
        await fresh.close_all()


async def _until_turn_ended(seen):
    await until(lambda: any(e.kind == "turn_ended" for e in seen))


async def test_load_takes_options_from_the_record(registry, root_options, sessions_dir):
    root = await registry.create(root_options)
    child = await registry.create(
        root.options.inherit(name="worker", model="alias-w", turn_method="handle_batch"),
        parent_id=root.id,
    )
    await registry.close_all()

    fresh = SessionRegistry(SessionStore(sessions_dir), agent_factory=ScriptedModels())
    try:
        loaded = await fresh.load(child.id)
        assert (loaded.options.agent_spec, loaded.options.model, loaded.options.name) == (
            root_options.agent_spec,
            "alias-w",
            "worker",
        )
        assert loaded.options.workspace == root_options.workspace
        assert loaded.options.turn_method == "handle_batch"
        assert (loaded.parent_id, loaded.depth) == (root.id, 1)
        await fresh.close_all()

        # Fields the caller sets win; the rest still come from the record.
        batch = await fresh.load(child.id, host="acp", turn_method="handle")
        assert (batch.options.host, batch.options.turn_method, batch.options.model) == (
            "acp",
            "handle",
            "alias-w",
        )
        await fresh.close_all()
        # Options are not merged in: only explicit keyword overrides count.
        with pytest.raises(TypeError):
            await fresh.load(child.id, root_options)
    finally:
        await fresh.close_all()


async def test_a_throwaway_childs_turn_method_is_recorded(registry, root_options, models):
    models.scripts[None] = [
        cell(
            "c = await self.session.delegate('T', 't')\n"
            "await c.wait()\n"
            "return_result(Done(explanation='ok'))"
        )
    ]
    models.scripts["T"] = [done("t")]
    root = await registry.create(root_options)
    await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    [child] = registry.children(root.id)
    assert child.turn_method == "handle_batch"
    assert registry.store.get(root.id).turn_method == "handle"


async def test_a_turn_cancelled_from_inside_fails_and_the_loop_goes_on(root_options, sessions_dir):
    registry = SessionRegistry(SessionStore(sessions_dir))
    options = root_options.model_copy(update={"agent_spec": "coder_test_agents:SelfCancelAgent"})
    root = await registry.create(options)
    with pytest.raises(TurnFailedError, match="cancelled from inside"):
        await asyncio.wait_for(root.prompt("one"), 5)
    assert not root._loop_task.done()
    assert await asyncio.wait_for(root.prompt("two"), 5) == Done(explanation="finished")
    [first, _] = [raw for _, raw in registry.store.load_rows(root.id, frozenset({"TurnEnded"}))]
    assert first["outcome_kind"] == "error"
    await registry.close_all()
    assert not registry.store.is_active(root.id)


async def test_a_failing_turn_record_does_not_stop_the_loop(registry, root_options, models):
    models.scripts[None] = [done("first"), done("second")]
    root = await registry.create(root_options)
    add = root.handle.events.add
    failures = []

    def failing_add(event, **kwargs):
        if isinstance(event, TurnEnded) and not failures:
            failures.append(event)
            raise sqlite3.OperationalError("database is locked")
        return add(event, **kwargs)

    root.handle.events.add = failing_add
    with pytest.raises(TurnFailedError) as failed:
        await asyncio.wait_for(root.prompt("one"), 5)
    assert isinstance(failed.value.error, sqlite3.OperationalError)
    assert failed.value.__cause__ is failed.value.error
    assert await asyncio.wait_for(root.prompt("two"), 5) == Done(explanation="second")
    await registry.close_all()
    assert not registry.store.is_active(root.id)


async def test_close_all_goes_on_when_one_close_fails(registry, root_options):
    first = await registry.create(root_options)
    second = await registry.create(root_options)

    async def broken():
        raise RuntimeError("children could not close")

    first._before_close = broken
    await registry.close_all()
    assert first._closed and second._closed
    assert not registry.store.is_active(first.id)
    assert not registry.store.is_active(second.id)


async def test_a_failed_prepare_closes_the_half_built_session(root_options, sessions_dir):
    factory = ModelFactory()
    registry = SessionRegistry(SessionStore(sessions_dir), llm_factory=factory)
    closed = []

    async def prepare(session):
        async def on_close():
            closed.append(session.id)

        session.agent.event_manager.on_close(on_close)
        raise RuntimeError("bridge failed")

    with pytest.raises(RuntimeError, match="bridge failed"):
        await registry.create(root_options.model_copy(update={"model": "alias-a"}), prepare=prepare)
    [llm] = factory.made
    assert len(closed) == 1 and llm.closed
    assert _db_files(sessions_dir) == []
    assert not registry.store.is_active(closed[0])


async def test_a_failed_agent_build_closes_the_owned_client(root_options, sessions_dir):
    factory = ModelFactory()
    registry = SessionRegistry(SessionStore(sessions_dir), llm_factory=factory)
    failing = root_options.model_copy(
        update={"model": "alias-a", "agent_spec": "coder_test_agents:FailingAgent"}
    )
    with pytest.raises(RuntimeError, match="construction failed"):
        await registry.create(failing)
    [llm] = factory.made
    assert llm.closed
    assert _db_files(sessions_dir) == []


async def test_the_factorys_default_client_is_owned_too(root_options, sessions_dir):
    factory = ModelFactory({"default-model": [[done("default")]]})
    registry = SessionRegistry(SessionStore(sessions_dir), llm_factory=factory)
    root = await registry.create(root_options)  # no model: the factory's default
    [llm] = factory.made
    assert factory.calls == [(None, root_options.workspace)]
    assert root.agent.llm is llm
    assert root.info.model == "default-model"
    assert await asyncio.wait_for(root.prompt("go"), 5) == Done(explanation="default")
    await registry.close_all()
    assert llm.closed


async def test_a_closing_parent_gets_no_new_children(registry, root_options, sessions_dir):
    root = await registry.create(root_options)
    before = _db_files(sessions_dir)
    root._closing = True  # as while its close() is closing its children
    try:
        with pytest.raises(SessionClosedError):
            await registry.create(root.options.inherit(name="late"), parent_id=root.id)
    finally:
        root._closing = False
    assert _db_files(sessions_dir) == before


async def test_a_steer_buffered_at_a_crash_is_requeued(
    registry, root_options, models, sessions_dir
):
    started, _block = fresh_events()
    models.scripts[None] = [cell(BLOCKING_CELL)]
    root = await registry.create(root_options)
    first = asyncio.ensure_future(root.prompt("start"))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    receipt = await root.steer("STEER-CRASH")
    # Crash: the loop dies without settling the turn, and the file is let go.
    root._loop_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await root._loop_task
    root.handle.close()
    first.cancel()

    later = ScriptedModels({None: [done("resumed")]})
    fresh = SessionRegistry(SessionStore(sessions_dir), agent_factory=later)
    try:
        loaded = await fresh.load(root.id)
        assert await asyncio.wait_for(loaded.outcome(receipt.item_id), TIMEOUT) == Done(
            explanation="resumed"
        )
        assert "STEER-CRASH" in str(later.llms[None].calls[0].messages)
        requeued = [
            raw["item_id"] for _, raw in fresh.store.load_rows(root.id, frozenset({"ItemRequeued"}))
        ]
        assert receipt.item_id in requeued
    finally:
        await fresh.close_all()
