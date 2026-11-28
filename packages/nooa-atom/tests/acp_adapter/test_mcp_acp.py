# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCP-over-ACP: stateless requests, attribution, MRTR and cancellation."""

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any

import pytest
from acp import RequestError
from nooa_atom.acp import v2
from nooa_atom.acp.mcp_acp import (
    MCP_PROTOCOL_VERSION,
    AcpMCPClient,
    AcpMCPError,
    AcpMCPSession,
    create_acp_server,
)
from nooa_atom.acp.v1_shim import _mcp_server_to_v2, initialize_response
from nooa_atom.session.port import current_port


class _Peer:
    def __init__(self, handler: Any) -> None:
        self.handler = handler
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self.notifications: list[tuple[str, dict[str, Any]]] = []
        self.next_id = 40

    async def request(
        self,
        method: str,
        params: dict[str, Any],
        *,
        on_id: Any = None,
    ) -> Any:
        self.next_id += 1
        if on_id is not None:
            on_id(self.next_id)
        self.requests.append((method, params))
        inner_method = params.get("method") if method == "mcp/message" else method
        result = self.handler(inner_method, params)
        return await result if inspect.isawaitable(result) else result

    async def notify(self, method: str, params: dict[str, Any]) -> None:
        self.notifications.append((method, params))


def _tools_result() -> dict[str, Any]:
    return {
        "tools": [
            {
                "name": "echo",
                "description": "Echo a message",
                "inputSchema": {
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                    "required": ["message"],
                },
            }
        ]
    }


async def test_discovery_is_unattributed_and_calls_use_session_and_tool_ids(monkeypatch):
    calls: list[dict[str, Any]] = []

    async def handler(method: str, params: dict[str, Any]) -> Any:
        calls.append(params)
        if method == "tools/list":
            return {"result": _tools_result()}
        assert method == "tools/call"
        return {
            "result": {
                "resultType": "complete",
                "content": [{"type": "text", "text": "hello"}],
                "isError": False,
            }
        }

    peer = _Peer(handler)
    tool = await create_acp_server(peer, "roy", "server-1")
    discovery = peer.requests[0][1]
    assert "sessionId" not in discovery and "toolCallId" not in discovery
    meta = discovery["params"]["_meta"]
    assert meta["io.modelcontextprotocol/protocolVersion"] == MCP_PROTOCOL_VERSION
    assert meta["io.modelcontextprotocol/clientCapabilities"] == {"elicitation": {}}

    monkeypatch.setattr("nooa_atom.acp.mcp_acp.current_tool_call_id", lambda: "cell-1")
    token = current_port.set(SimpleNamespace(id="session-1"))
    try:
        assert await tool.echo(message="hi") == "hello"
    finally:
        current_port.reset(token)
    invocation = peer.requests[1][1]
    assert invocation["sessionId"] == "session-1"
    assert invocation["toolCallId"] == "cell-1"
    assert invocation["requestId"] != discovery["requestId"]
    assert [params["method"] for _, params in peer.requests] == ["tools/list", "tools/call"]


async def test_cancelling_a_call_cancels_its_outer_acp_request():
    started = asyncio.Event()

    async def handler(method: str, params: dict[str, Any]) -> Any:
        started.set()
        await asyncio.Future()

    peer = _Peer(handler)
    session = AcpMCPSession(AcpMCPClient(peer, "roy", "server-1"))
    task = asyncio.create_task(session.call_tool("echo", {"message": "hi"}))
    await asyncio.wait_for(started.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert peer.notifications == [("$/cancel_request", {"requestId": 41})]


async def test_inner_mcp_errors_keep_their_own_code():
    peer = _Peer(
        lambda method, params: {"error": {"code": -32602, "message": "Unknown tool", "data": "x"}}
    )
    session = AcpMCPSession(AcpMCPClient(peer, "roy", "server-1"))
    with pytest.raises(AcpMCPError) as error:
        await session.call_tool("missing")
    assert error.value.code == -32602 and "Unknown tool" in str(error.value)


async def test_mrtr_elicitation_retries_with_fresh_id_responses_and_opaque_state(monkeypatch):
    attempts: list[dict[str, Any]] = []

    async def handler(method: str, params: dict[str, Any]) -> Any:
        if method == "elicitation/create":
            return {"action": "accept", "content": {"name": "octocat"}}
        attempts.append(params["params"])
        if len(attempts) == 1:
            return {
                "result": {
                    "resultType": "input_required",
                    "inputRequests": {
                        "login": {
                            "method": "elicitation/create",
                            "params": {
                                "mode": "form",
                                "message": "Log in",
                                "requestedSchema": {"type": "object", "properties": {}},
                            },
                        }
                    },
                    "requestState": "opaque:not parsed",
                }
            }
        return {"result": {"content": [{"type": "text", "text": "done"}], "isError": False}}

    peer = _Peer(handler)
    client = AcpMCPClient(peer, "roy", "server-1")
    client.activate()
    monkeypatch.setattr("nooa_atom.acp.mcp_acp.current_tool_call_id", lambda: "cell-1")
    token = current_port.set(SimpleNamespace(id="session-1"))
    try:
        result = await AcpMCPSession(client).call_tool("login", {})
    finally:
        current_port.reset(token)
    assert result.content[0].text == "done"
    retry = attempts[1]
    assert retry["inputResponses"] == {
        "login": {"action": "accept", "content": {"name": "octocat"}}
    }
    assert retry["requestState"] == "opaque:not parsed"
    first_envelope = peer.requests[0][1]
    second_envelope = peer.requests[2][1]
    assert first_envelope["requestId"] != second_envelope["requestId"]
    assert second_envelope["sessionId"] == "session-1"
    assert second_envelope["toolCallId"] == "cell-1"


def test_acp_capability_and_server_validation():
    answer = v2.initialize_result()
    assert answer["capabilities"]["session"]["mcp"]["acp"] == {}
    v1 = initialize_response(1)
    assert v1.agent_capabilities.mcp_capabilities.acp is True
    server = {"type": "acp", "name": "roy", "serverId": "server-1"}
    assert v2.check_mcp_servers([server]) == [server]
    assert _mcp_server_to_v2(server)["type"] == "acp"
    with pytest.raises(RequestError):
        v2.check_mcp_servers([{**server, "serverId": ""}])
    with pytest.raises(RequestError):
        v2.check_mcp_servers([server, {**server, "name": "other"}])
