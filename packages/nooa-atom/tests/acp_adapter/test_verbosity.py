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
