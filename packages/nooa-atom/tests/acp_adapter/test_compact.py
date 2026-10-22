# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""/compact over ACP: listed, and sent to the agent as a compaction directive."""

import asyncio

from acp import text_block
from acp.schema import AgentMessageChunk, AvailableCommandsUpdate
from atom_test_agents import reply

TIMEOUT = 30


async def _new(adapter, workspace):
    return (await adapter.new_session(str(workspace))).session_id


async def test_compact_is_advertised(atom_adapter, workspace, client):
    adapter = await atom_adapter([])
    session_id = await _new(adapter, workspace)
    await client.wait_for(lambda: client.updates(session_id, AvailableCommandsUpdate))
    [commands] = client.updates(session_id, AvailableCommandsUpdate)[-1:]
    [compact] = [c for c in commands.available_commands if c.name == "compact"]
    assert compact.description == (
        "Summarize older conversation history to free up context while preserving key details."
    )


async def test_compact_sends_the_directive_to_the_agent(atom_adapter, workspace, client):
    adapter = await atom_adapter([reply("Compacted.\n")])
    session_id = await _new(adapter, workspace)
    response = await asyncio.wait_for(
        adapter.prompt(session_id, [text_block("/compact keep the API notes")]), TIMEOUT
    )
    assert response.stop_reason == "end_turn"
    [llm] = adapter.test_models.made
    sent = str(llm.calls[0].messages)
    assert "slash_commands" in sent
    assert "self.events.collapse(start_tag, end_tag, summary_text=...)" in sent
    assert "keep the API notes" in sent
    assert client.texts(AgentMessageChunk, session_id)[-1] == "Compacted.\n\n"
