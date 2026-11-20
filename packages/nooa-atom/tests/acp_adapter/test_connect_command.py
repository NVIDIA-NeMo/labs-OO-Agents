# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``/connect`` is listed in the session's commands, so clients show it in their menu."""

from acp.schema import AvailableCommandsUpdate
from atom_test_agents import ScriptedModels


async def test_connect_is_advertised(make_adapter, workspace, client):
    adapter = await make_adapter(ScriptedModels())
    session_id = (await adapter.new_session(str(workspace))).session_id
    await client.wait_for(lambda: client.updates(session_id, AvailableCommandsUpdate))
    [commands] = client.updates(session_id, AvailableCommandsUpdate)
    [command] = [c for c in commands.available_commands if c.name == "connect"]
    assert "nooa connect" in command.description
    assert command.input is None
