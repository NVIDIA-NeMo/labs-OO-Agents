# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Server-owned pending turns after Stop. Strict models and event barriers, not Pool UI proof."""

import asyncio

import pytest
from acp import text_block
from atom_test_agents import BLOCKING_CELL, ScriptedModels, cell, fresh_events, reply
from nooa_atom.session.store import SessionStore

from .test_prompt import POOL, _input_events, _new, _pool_steer


async def bounded(awaitable):
    return await asyncio.wait_for(awaitable, 10)


def ledger(session, item_id):
    return [
        kind
        for kind, row in SessionStore._read_rows(session.handle.path)
        if row.get("item_id") == item_id
        and kind in ("ItemAdmitted", "ItemConsumed", "ItemWithdrawn")
    ]


def acks(client, sid, input_id):
    return [
        meta
        for _, meta in _input_events(client, sid)
        if meta.get("poolside/clientInputId") == input_id
    ]


@pytest.mark.parametrize("cancel_rpc", [False, True])
@pytest.mark.parametrize("double_cancel", [False, True])
async def test_pending_runs_once_in_order_after_cancel(
    make_adapter, workspace, client, cancel_rpc, double_cancel
):
    started, _ = fresh_events()
    models = ScriptedModels({None: [cell(BLOCKING_CELL), reply("PENDING END"), reply("FENCE")]})
    adapter = await make_adapter(models, client_info=POOL)
    sid = await _new(adapter, workspace)
    session = adapter.session(sid)
    lifecycle = []
    pending_done = asyncio.Event()

    def observe(update):
        if update.kind in ("turn_started", "turn_ended"):
            lifecycle.append(update.kind)
        if lifecycle.count("turn_ended") == 2:
            pending_done.set()

    session.subscribe(observe)
    first = asyncio.create_task(adapter.prompt(sid, [text_block("FIRST")]))
    await bounded(started.wait())
    tasks = [
        await _pool_steer(adapter, sid, text, identity)
        for text, identity in [("PENDING A", "id-a"), ("PENDING B", "id-b")]
    ]
    receipts = [receipt for receipt, _ in adapter._followers[sid]]
    for receipt in receipts:
        assert ledger(session, receipt.item_id) == ["ItemAdmitted"]
    if cancel_rpc:
        tasks[0].cancel()
        with pytest.raises(asyncio.CancelledError):
            await bounded(tasks[0])
    if double_cancel:
        await bounded(asyncio.gather(adapter.cancel(sid), adapter.cancel(sid)))
    else:
        await bounded(adapter.cancel(sid))
    assert (await bounded(first)).stop_reason == "cancelled"
    if not cancel_rpc:
        assert await bounded(tasks[0]) == {"inputId": "id-a"}
    assert await bounded(tasks[1]) == {"inputId": "id-b"}
    await bounded(pending_done.wait())
    await adapter._bridges[sid].flush()
    assert lifecycle == ["turn_started", "turn_ended", "turn_started", "turn_ended"]
    # The batch preserves admission order; the original text is not resubmitted.
    messages = str(models.llms[None].calls[1].messages)
    assert messages.index("PENDING A") < messages.index("PENDING B")
    for receipt, identity in zip(receipts, ["id-a", "id-b"], strict=True):
        assert ledger(session, receipt.item_id) == ["ItemAdmitted", "ItemConsumed"]
        assert len(acks(client, sid, identity)) == 1
    assert (await bounded(adapter.prompt(sid, [text_block("FENCE")]))).stop_reason == "end_turn"
    assert len(models.llms[None].calls) == 3
    assert lifecycle == ["turn_started", "turn_ended"] * 3
    assert not adapter._open and not adapter._followers and not adapter._cancels
    assert not adapter._bridges[sid]._client_inputs
    assert not session._futures


async def test_late_cancel_joins_old_scope_not_retained_turn(
    make_adapter, workspace, client, monkeypatch
):
    started, first_gate = fresh_events()
    models = ScriptedModels({None: [cell(BLOCKING_CELL), cell(BLOCKING_CELL), reply("FENCE")]})
    adapter = await make_adapter(models, client_info=POOL)
    sid = await _new(adapter, workspace)
    session = adapter.session(sid)
    first = asyncio.create_task(adapter.prompt(sid, [text_block("FIRST")]))
    await bounded(started.wait())
    follow = await _pool_steer(adapter, sid, "PENDING", "late-id")
    [(receipt, _)] = adapter._followers[sid]
    bridge = adapter._bridges[sid]
    real_flush = bridge.flush
    root_flushing = asyncio.Event()
    release_root = asyncio.Event()

    async def gated_flush():
        if asyncio.current_task() is first:
            root_flushing.set()
            await release_root.wait()
        await real_flush()

    monkeypatch.setattr(bridge, "flush", gated_flush)
    # Cancel first turn, forcing its root response to remain in the old scope.
    started, second_gate = fresh_events()
    await bounded(asyncio.gather(adapter.cancel(sid), adapter.cancel(sid)))
    await bounded(started.wait())
    await bounded(root_flushing.wait())
    assert await bounded(follow) == {"inputId": "late-id"}
    await bounded(adapter.cancel(sid))  # old scope still open: must not cancel next turn
    assert session.info.status == "running"
    assert not second_gate.is_set()
    release_root.set()
    assert (await bounded(first)).stop_reason == "cancelled"
    assert sid not in adapter._cancels
    # With the old scope retired, a deliberate new Stop can stop pending work.
    await bounded(adapter.cancel(sid))
    assert ledger(session, receipt.item_id) == ["ItemAdmitted", "ItemConsumed"]
    assert (await bounded(adapter.prompt(sid, [text_block("FENCE")]))).stop_reason == "end_turn"
    assert len(models.llms[None].calls) == 3
    assert len(acks(client, sid, "late-id")) == 1


async def test_consumed_follower_not_replayed(make_adapter, workspace, client):
    started, first_gate = fresh_events()
    models = ScriptedModels(
        {None: [cell(BLOCKING_CELL), reply("FIRST END"), cell(BLOCKING_CELL), reply("FENCE")]}
    )
    adapter = await make_adapter(models, client_info=POOL)
    sid = await _new(adapter, workspace)
    session = adapter.session(sid)
    first = asyncio.create_task(adapter.prompt(sid, [text_block("FIRST")]))
    await bounded(started.wait())
    follow = await _pool_steer(adapter, sid, "PENDING", "taken-id")
    [(receipt, _)] = adapter._followers[sid]
    started, _ = fresh_events()
    first_gate.set()
    await bounded(started.wait())
    assert await bounded(follow) == {"inputId": "taken-id"}
    assert not first.done()
    await bounded(adapter.cancel(sid))
    assert (await bounded(first)).stop_reason == "cancelled"
    assert (await bounded(adapter.prompt(sid, [text_block("FENCE")]))).stop_reason == "end_turn"
    assert len(models.llms[None].calls) == 4
    assert ledger(session, receipt.item_id) == ["ItemAdmitted", "ItemConsumed"]
    assert len(acks(client, sid, "taken-id")) == 1


@pytest.mark.parametrize("teardown", ["withdraw", "close_session", "disconnect"])
async def test_cancelled_rpc_owner_cleans_up_on_settlement(
    make_adapter, workspace, client, teardown
):
    started, _ = fresh_events()
    models = ScriptedModels({None: [cell(BLOCKING_CELL)]})
    adapter = await make_adapter(models, client_info=POOL)
    sid = await _new(adapter, workspace)
    session = adapter.session(sid)
    bridge = adapter._bridges[sid]
    first = asyncio.create_task(adapter.prompt(sid, [text_block("FIRST")]))
    await bounded(started.wait())
    follow = await _pool_steer(adapter, sid, "PENDING", "orphan-id")
    [(receipt, _)] = adapter._followers[sid]
    owners = [task for task in adapter._background if task.get_name() == "nooa-acp-pool-input"]
    assert len(owners) == 1
    follow.cancel()
    with pytest.raises(asyncio.CancelledError):
        await bounded(follow)
    assert ledger(session, receipt.item_id) == ["ItemAdmitted"]
    if teardown == "withdraw":
        assert session.withdraw(receipt)
        await bounded(owners[0])
        await bounded(adapter.cancel(sid))
    elif teardown == "close_session":
        await bounded(adapter.close_session(sid))
        await bounded(owners[0])
    else:
        await bounded(adapter.close())
    assert (await bounded(first)).stop_reason == "cancelled"
    assert not bridge._client_inputs
    assert not adapter._open and not adapter._followers and not adapter._cancels
    assert not acks(client, sid, "orphan-id")
    assert not session._futures


async def test_new_prompt_does_not_inherit_cancelled_followers(
    make_adapter, workspace, monkeypatch
):
    started, first_gate = fresh_events()
    models = ScriptedModels(
        {
            None: [
                cell(BLOCKING_CELL),
                reply("FIRST END"),
                cell(BLOCKING_CELL),
                reply("NEW PROMPT END"),
            ]
        }
    )
    adapter = await make_adapter(models, client_info=POOL)
    sid = await _new(adapter, workspace)
    first = asyncio.create_task(adapter.prompt(sid, [text_block("FIRST")]))
    await bounded(started.wait())
    follow = await _pool_steer(adapter, sid, "PENDING", "old-scope-id")
    started, _ = fresh_events()
    first_gate.set()
    await bounded(started.wait())
    assert await bounded(follow) == {"inputId": "old-scope-id"}
    bridge = adapter._bridges[sid]
    real_flush = bridge.flush
    root_flushing = asyncio.Event()
    release_root = asyncio.Event()

    async def gated_flush():
        if asyncio.current_task() is first:
            root_flushing.set()
            await release_root.wait()
        await real_flush()

    monkeypatch.setattr(bridge, "flush", gated_flush)
    await bounded(adapter.cancel(sid))
    await bounded(root_flushing.wait())
    # Old root is still answering cancelled. Its consumed cancelled follower
    # must not become a follower of this newly admitted normal prompt.
    assert (
        await bounded(adapter.prompt(sid, [text_block("NEW PROMPT")]))
    ).stop_reason == "end_turn"
    assert not first.done()
    release_root.set()
    assert (await bounded(first)).stop_reason == "cancelled"
    assert len(models.llms[None].calls) == 4
    assert not adapter._open and not adapter._followers and not adapter._cancels
