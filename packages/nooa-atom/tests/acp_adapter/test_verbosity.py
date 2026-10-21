# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ACP-local activity verbosity commands do not spend a model turn."""

import pytest
from acp import RequestError, text_block
from acp.schema import AgentMessageChunk, AvailableCommandsUpdate
from atom_test_agents import ScriptedModels


async def test_verbosity_is_advertised_and_session_local(make_adapter, workspace, client):
    adapter = await make_adapter(ScriptedModels())
    first = (await adapter.new_session(str(workspace))).session_id
    second = (await adapter.new_session(str(workspace))).session_id
    await client.wait_for(lambda: client.updates(first, AvailableCommandsUpdate))
    [commands] = client.updates(first, AvailableCommandsUpdate)
    [command] = [c for c in commands.available_commands if c.name == "verbosity"]
    assert command.input is not None

    async def invoke(text):
        response = await adapter.prompt(first, [text_block(text)])
        assert response.stop_reason == "end_turn"
        return client.texts(AgentMessageChunk, first)[-1]

    assert "ACP verbosity: verbose." in await invoke("/verbosity")
    for level in ("silent", "concise", "verbose"):
        assert await invoke(f"/verbosity {level}") == f"ACP verbosity: {level}.\n\n"
        assert adapter.bridge(first).verbosity == level
        assert adapter.bridge(second).verbosity == "verbose"
    assert "Usage: /verbosity" in await invoke("/verbosity wrong")
    assert "Usage: /verbosity" in await invoke("/verbosity silent extra")
    assert adapter.bridge(first).verbosity == "verbose"
    assert adapter.session(first).info.reply_count == 0
    assert "verbosity" not in {c.name for c in adapter.session(first).commands()}
    for mode in ("queue", "steer"):
        with pytest.raises(RequestError) as exc:
            await adapter._inject({"sessionId": first, "mode": mode, "text": "/verbosity silent"})
        assert "local command" in str(exc.value.data)
    assert adapter.session(first).info.reply_count == 0


async def test_concise_python_is_full_tool_card_not_message(make_adapter, workspace, client):
    from acp.schema import AgentThoughtChunk, ToolCallProgress, ToolCallStart
    from atom_test_agents import cell

    models = ScriptedModels(
        {
            None: [
                cell(
                    "# Check repository status\n"
                    "print('private output')\n"
                    "self.message('User-facing answer')\n"
                    "return_result(Done(explanation='Checked repository\\nExtra detail'))"
                )
            ]
        }
    )
    adapter = await make_adapter(models)
    session_id = (await adapter.new_session(str(workspace))).session_id
    await adapter.prompt(session_id, [text_block("/verbosity concise")])
    client.log.clear()
    response = await adapter.prompt(session_id, [text_block("Check it")])
    assert response.stop_reason == "end_turn"
    assert client.texts(AgentMessageChunk, session_id) == [
        "User-facing answer\n\n",
        "Done: Checked repository\nExtra detail\n\n",
    ]
    assert client.texts(AgentThoughtChunk, session_id) == []
    [start] = client.updates(session_id, ToolCallStart)
    [end] = client.updates(session_id, ToolCallProgress)
    assert start.title == end.title == "python: # Check repository status"
    assert start.kind == "other" and end.status == "completed"
    for card in (start, end):
        assert card.raw_input is not None and "private output" in card.raw_input["code"]
        assert card.content is not None
    assert len(start.content) == 1 and len(end.content) == 2
    assert "private output" in end.content[1].content.text
