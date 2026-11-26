# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ACP v2 over real standard input and output, in both server modes.

A v2 client gets the v2 wire shapes and prompt lifecycle; a v1 client
still gets v1 (see ``test_router_protocol``).
"""

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_router_protocol import MODES, Server  # noqa: E402

V2_INIT = {
    "protocolVersion": 2,
    "info": {"name": "roy", "title": "Roy", "version": "0.0.0"},
    "capabilities": {"elicitation": {"form": {}, "url": {}}, "auth": {"terminal": {}}},
}


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "workspace"
    path.mkdir()
    return path


@pytest.fixture
async def servers(tmp_path):
    started: list[Server] = []

    async def start(mode: str) -> Server:
        server = await Server.start(tmp_path, *MODES[mode])
        started.append(server)
        return server

    yield start
    for server in started:
        await server.kill()


def updates(server: Server, session_id: str) -> list[dict[str, Any]]:
    return [
        frame.params["update"]
        for frame in server.inbox
        if frame.method == "session/update" and frame.session_id == session_id
    ]


async def wait_idle(server: Server, session_id: str, count: int = 1) -> list[dict[str, Any]]:
    """Read frames until the session reported ``count`` idle states; return its updates."""
    probe = await server.request("session/list", {})
    await server.response(probe)
    for _ in range(200):
        idle = [
            update
            for update in updates(server, session_id)
            if update.get("sessionUpdate") == "state_update" and update["state"] == "idle"
        ]
        if len(idle) >= count:
            return updates(server, session_id)
        probe = await server.request("session/list", {})
        await server.response(probe)
    raise AssertionError(f"no idle state: {updates(server, session_id)}")


@pytest.mark.parametrize("mode", MODES)
async def test_v2_initialize_new_prompt_resume(mode, servers, workspace):
    server = await servers(mode)
    init = (await server.call("initialize", V2_INIT)).message["result"]
    assert init["protocolVersion"] == 2
    assert init["info"]["name"] == "nooa-atom"
    assert "agentCapabilities" not in init and "agentInfo" not in init
    session_caps = init["capabilities"]["session"]
    assert session_caps["mcp"] == {"stdio": {}, "http": {}}
    assert session_caps["delete"] == {}

    frame = await server.call("session/new", {"cwd": str(workspace)})
    result = frame.message["result"]
    session_id = result["sessionId"]
    assert "modes" not in result
    assert all("configId" in option and "id" not in option for option in result["configOptions"])
    assert {command["type"] for command in result["availableCommands"]} == {"command"}

    frame = await server.call(
        "session/prompt",
        {
            "sessionId": session_id,
            "prompt": [{"type": "text", "text": "hello"}],
            "idempotencyKey": "k1",
        },
    )
    message_id = frame.message["result"]["messageId"]
    seen = await wait_idle(server, session_id)
    kinds = [update["sessionUpdate"] for update in seen]
    user = [update for update in seen if update["sessionUpdate"] == "user_message"]
    assert user == [
        {
            "sessionUpdate": "user_message",
            "messageId": message_id,
            "content": [{"type": "text", "text": "hello"}],
        }
    ]
    states = [update for update in seen if update["sessionUpdate"] == "state_update"]
    assert [state["state"] for state in states] == ["running", "idle"]
    assert states[-1]["stopReason"] == "end_turn"
    assert kinds.index("state_update") < kinds.index("agent_message_chunk")
    chunks = [update for update in seen if update["sessionUpdate"] == "agent_message_chunk"]
    assert chunks and all(isinstance(chunk["messageId"], str) for chunk in chunks)
    assert "tool_call" not in kinds and "plan" not in kinds

    # The same idempotency key: the recorded message, no second turn.
    frame = await server.call(
        "session/prompt",
        {
            "sessionId": session_id,
            "prompt": [{"type": "text", "text": "hello"}],
            "idempotencyKey": "k1",
        },
    )
    assert frame.message["result"] == {"messageId": message_id}

    assert (await server.call("session/close", {"sessionId": session_id})).message["result"] == {}
    server.inbox.clear()
    frame = await server.call(
        "session/resume",
        {"sessionId": session_id, "cwd": str(workspace), "replayFrom": {"type": "start"}},
    )
    assert "result" in frame.message, frame.message
    replayed = updates(server, session_id)
    kinds = [update["sessionUpdate"] for update in replayed]
    assert "replay_complete" in kinds
    assert kinds.index("replay_complete") < kinds.index("state_update")
    assert replayed[kinds.index("state_update")]["state"] == "idle"
    users = [
        update
        for update in replayed[: kinds.index("replay_complete")]
        if update["sessionUpdate"] == "user_message_chunk"
    ]
    assert users and users[0]["messageId"] == message_id

    assert (await server.call("session/close", {"sessionId": session_id})).message["result"] == {}
    frame = await server.call("session/delete", {"sessionId": session_id})
    assert frame.message["result"] == {}
    assert await server.finish() == 0


@pytest.mark.parametrize("mode", MODES)
async def test_v2_resume_without_replay_sends_no_history(mode, servers, workspace):
    server = await servers(mode)
    await server.call("initialize", V2_INIT)
    session_id = (await server.call("session/new", {"cwd": str(workspace)})).message["result"][
        "sessionId"
    ]
    await server.call(
        "session/prompt", {"sessionId": session_id, "prompt": [{"type": "text", "text": "hi"}]}
    )
    await wait_idle(server, session_id)
    await server.call("session/close", {"sessionId": session_id})
    server.inbox.clear()
    frame = await server.call("session/resume", {"sessionId": session_id, "cwd": str(workspace)})
    assert "result" in frame.message, frame.message
    kinds = [update["sessionUpdate"] for update in updates(server, session_id)]
    assert "user_message_chunk" not in kinds and "agent_message_chunk" not in kinds
    assert "replay_complete" not in kinds
    assert kinds[-1] == "state_update"
    assert await server.finish() == 0


@pytest.mark.parametrize("mode", MODES)
async def test_v2_rejects_what_v2_removed(mode, servers, workspace):
    server = await servers(mode)
    await server.call("initialize", V2_INIT)
    frame = await server.call(
        "session/new",
        {"cwd": str(workspace), "mcpServers": [{"type": "sse", "name": "x", "url": "http://x"}]},
    )
    assert frame.message["error"]["code"] == -32602
    frame = await server.call(
        "session/new",
        {"cwd": str(workspace), "mcpServers": [{"type": "stdio", "name": "x", "command": "rel"}]},
    )
    assert frame.message["error"]["code"] == -32602
    session_id = (await server.call("session/new", {"cwd": str(workspace)})).message["result"][
        "sessionId"
    ]
    frame = await server.call(
        "session/load", {"sessionId": session_id, "cwd": str(workspace), "mcpServers": []}
    )
    assert frame.message["error"]["code"] == -32601
    frame = await server.call("session/set_mode", {"sessionId": session_id, "modeId": "auto"})
    assert frame.message["error"]["code"] == -32601
    frame = await server.call(
        "session/set_config_option",
        {"sessionId": session_id, "configId": "verbosity", "type": "id", "value": "concise"},
    )
    options = frame.message["result"]["configOptions"]
    [verbosity] = [option for option in options if option["configId"] == "verbosity"]
    assert verbosity["currentValue"] == "concise"
    frame = await server.call(
        "session/prompt", {"sessionId": "nope", "prompt": [{"type": "text", "text": "x"}]}
    )
    assert frame.message["error"]["code"] == -32002
    assert await server.finish() == 0


@pytest.mark.parametrize("mode", MODES)
async def test_v1_client_still_gets_v1(mode, servers, workspace):
    server = await servers(mode)
    init = (await server.initialize()).message["result"]
    assert init["protocolVersion"] == 1 and "agentCapabilities" in init
    session_id = await server.new_session(workspace)
    frame = await server.prompt(session_id)
    assert frame.message["result"] == {"stopReason": "end_turn"}
    kinds = {update["sessionUpdate"] for update in updates(server, session_id)}
    assert "state_update" not in kinds and "user_message" not in kinds
    assert await server.finish() == 0


@pytest.mark.parametrize("mode", MODES)
async def test_v1_probe_carries_the_v2_answer(mode, servers):
    """The installer's routers ask with a probe marker: a v1 answer plus its v2 form."""
    server = await servers(mode)
    probe = {
        "protocolVersion": 1,
        "clientCapabilities": {
            "fs": {"readTextFile": True, "writeTextFile": True},
            "terminal": True,
        },
        "_meta": {"dev.nooa/v2": {"probe": True}},
    }
    init = (await server.call("initialize", probe)).message["result"]
    assert init["protocolVersion"] == 1 and "agentCapabilities" in init
    own = init["_meta"]["dev.nooa/v2"]
    assert own["protocolVersions"] == [1, 2]
    assert own["initialize"]["protocolVersion"] == 2
    assert own["initialize"]["capabilities"]["session"]["delete"] == {}
    assert own["initialize"]["info"]["name"] == "nooa-atom"
    assert await server.finish() == 0
