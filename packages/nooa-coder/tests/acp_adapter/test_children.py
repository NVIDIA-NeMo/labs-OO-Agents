# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Child sessions over ACP: announced, mirrored, loadable by id, detached on close."""

import asyncio

from acp import text_block
from acp.schema import SessionInfoUpdate, ToolCallProgress, ToolCallStart, UserMessageChunk
from coder_test_agents import ScriptedModels, cell, done

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
    [child] = adapter.registry.children(session_id)
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
    child = adapter.registry.get(child_id)
    assert child is not None

    await adapter.load_session(str(workspace), child_id)
    assert adapter.registry.get(child_id) is child
    assert "CHILD-PROMPT\n" in client.texts(UserMessageChunk, child_id)

    await adapter.close_session(child_id)
    assert adapter.registry.get(child_id) is child  # still running under its parent
    await adapter.load_session(str(workspace), child_id)  # and can be followed again
    # Following it again is a new bridge, so the transcript is sent again.
    assert client.texts(UserMessageChunk, child_id).count("CHILD-PROMPT\n") == 2


async def test_listing_shows_roots_only(make_adapter, workspace, client):
    adapter = await make_adapter(_models())
    session_id, child_id = await _parent_with_child(adapter, workspace, client)
    listed = [s.session_id for s in (await adapter.list_sessions()).sessions]
    assert listed == [session_id]
