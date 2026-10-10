# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Child sessions over ACP: announced, mirrored, loadable by id, detached on close."""

import asyncio

from acp import text_block
from acp.schema import SessionInfoUpdate, ToolCallProgress, ToolCallStart, UserMessageChunk
from atom_test_agents import ScriptedModels, cell, done

TIMEOUT = 30


def _models() -> ScriptedModels:
    return ScriptedModels(
        {
            None: [
                cell(
                    "child = await self.session.delegate('helper', 'CHILD-PROMPT', retain=True)\n"
                    "return_result(Done(explanation='spawned'))"
                ),
                done("noted the child's result"),
            ],
            "helper": [cell("print('child work')\nreturn_result(Done(explanation='child done'))")],
        }
    )


async def _parent_with_child(adapter, workspace, client):
    session_id = (await adapter.new_session(str(workspace))).session_id
    await asyncio.wait_for(adapter.prompt(session_id, [text_block("start")]), TIMEOUT)
    [child] = adapter.registry_for(workspace).children(session_id)
    await client.wait_for(
        lambda: any(
            u.status == "completed" and u.tool_call_id.startswith(child.id)
            for u in client.updates(session_id, ToolCallProgress)
        )
    )
    return session_id, child.id


async def test_a_child_is_announced_and_its_tool_cards_are_mirrored(
    make_adapter, workspace, client
):
    adapter = await make_adapter(_models())
    session_id, child_id = await _parent_with_child(adapter, workspace, client)
    [info] = [u for u in client.updates(session_id, SessionInfoUpdate) if u.field_meta]
    assert info.field_meta == {
        "dev.nooa/children": [
            {"sessionId": child_id, "name": "helper", "depth": 1, "retained": True}
        ]
    }
    mirrored = [
        u.tool_call_id
        for u in client.updates(session_id, ToolCallStart)
        if u.tool_call_id.startswith(f"{child_id}:")
    ]
    assert len(mirrored) == 1
    # Nothing is sent under the child's own id until a client opens it.
    assert client.updates(child_id) == []


async def test_loading_a_child_follows_it_and_closing_it_only_detaches(
    make_adapter, workspace, client
):
    adapter = await make_adapter(_models())
    session_id, child_id = await _parent_with_child(adapter, workspace, client)
    child = adapter.session(child_id)
    assert child is not None

    await adapter.load_session(str(workspace), child_id)
    assert adapter.session(child_id) is child
    assert "CHILD-PROMPT\n" in client.texts(UserMessageChunk, child_id)

    await adapter.close_session(child_id)
    assert adapter.session(child_id) is child  # still running under its parent
    await adapter.load_session(str(workspace), child_id)  # and can be followed again


async def test_listing_shows_roots_only(make_adapter, workspace, client):
    adapter = await make_adapter(_models())
    session_id, child_id = await _parent_with_child(adapter, workspace, client)
    listed = [s.session_id for s in (await adapter.list_sessions()).sessions]
    assert listed == [session_id]


async def test_detaching_open_child_form_revokes_only_old_attachment(
    make_adapter, workspace, client
):
    from acp.schema import (
        ClientCapabilities,
        ElicitationCapabilities,
        ElicitationFormCapabilities,
    )

    from nooa.interactive import FormResponse

    models = _models()
    models.scripts["helper"].extend(
        [
            cell(
                "return_result(NeedInputForm(heading='Name?', questions=[TextQuestion(id='answer', label='Name?')]))"
            ),
            cell(
                "self.v.answer = notification['user_messages'][0]\nreturn_result(Done(explanation='answered'))"
            ),
        ]
    )
    adapter = await make_adapter(
        models,
        capabilities=ClientCapabilities(
            elicitation=ElicitationCapabilities(form=ElicitationFormCapabilities())
        ),
    )
    parent_id, child_id = await _parent_with_child(adapter, workspace, client)
    child = adapter.session(child_id)
    await adapter.load_session(str(workspace), child_id)
    entered, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def late(*args, **kwargs):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        return {"action": "accept", "content": {"answer": "obsolete"}}

    client.create_elicitation = late
    pending = asyncio.create_task(adapter.prompt(child_id, [text_block("form please")]))
    await asyncio.wait_for(entered.wait(), TIMEOUT)
    old_bridge = adapter.bridge(child_id)
    try:
        await adapter.close_session(child_id)
        assert adapter.session(child_id) is child and not child.closing
        assert child.parent_id == parent_id
        assert old_bridge not in adapter._attachment_unsubscribes
        await adapter.load_session(str(workspace), child_id)
        assert adapter.bridge(child_id) is not old_bridge
        assert (await asyncio.wait_for(pending, TIMEOUT)).stop_reason == "cancelled"
        assert cancelled.is_set()
        assert child._pending_form is not None
        response = FormResponse(action="accept", content={"answer": "current"})
        receipt = await child.submit(response)
        assert (
            await asyncio.wait_for(child.outcome(receipt.item_id), TIMEOUT)
        ).explanation == "answered"
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
    assert child._agent.v.answer == response
    assert not adapter._asks and not adapter._asking


async def test_detach_waiting_child_prompt_does_not_cancel_child_outcome(
    make_adapter, workspace, client
):
    from atom_test_agents import until

    models = _models()
    models.scripts["helper"].extend(
        [
            cell("return_result(Waiting(explanation='external', on=['user_messages']))"),
            done("continued"),
        ]
    )
    adapter = await make_adapter(models)
    _, child_id = await _parent_with_child(adapter, workspace, client)
    child = adapter.session(child_id)
    await adapter.load_session(str(workspace), child_id)
    pending = asyncio.create_task(adapter.prompt(child_id, [text_block("wait")]))
    await until(lambda: child.info.turn_count == 2 and child.info.status != "running")
    assert not pending.done()
    receipt = adapter._prompts[child_id][0]
    await adapter.close_session(child_id)
    assert (await asyncio.wait_for(pending, TIMEOUT)).stop_reason == "cancelled"
    assert not child.closing
    await child.submit("continue")
    assert (
        await asyncio.wait_for(child.outcome(receipt.item_id), TIMEOUT)
    ).explanation == "continued"
