# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SessionRegistry: the tree of live sessions and their files."""

import asyncio

import pytest
from coder_test_agents import BLOCKING_CELL, ScriptedModels, cell, done, fresh_events
from nooa_coder.session.registry import (
    ChildActiveElsewhereError,
    DepthLimitError,
    SessionRegistry,
)
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
    while not seen:
        await asyncio.sleep(0.01)
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
    assert await registry.load(root.id, root_options) is root


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
        loaded = await fresh.load(root.id, root_options)
        ended = []
        loaded.subscribe(lambda e: ended.append(e) if e.kind == "turn_ended" else None)
        while not ended:
            await asyncio.sleep(0.01)
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
        first, second = await asyncio.gather(
            fresh.load(root.id, root_options), fresh.load(root.id, root_options)
        )
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
        detached = await elsewhere.load(child.id, child.options)  # a detached child is allowed
        assert detached.parent_id == root.id
        with pytest.raises(ChildActiveElsewhereError, match=child.id):
            await here.load(root.id, root_options)
        assert here.sessions == {} and here._reserved == {}
        await elsewhere.close_all()
        loaded = await here.load(root.id, root_options)
        assert loaded.id == root.id
    finally:
        await elsewhere.close_all()
        await here.close_all()


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
