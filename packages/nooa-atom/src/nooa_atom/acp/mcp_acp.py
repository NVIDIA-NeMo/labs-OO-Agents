# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stateless MCP 2026-07-28 requests carried by ACP ``mcp/message``.

This is the agent side of the draft MCP-over-ACP RFD. There is no MCP
initialize handshake or per-server connection: every operation has fresh
logical and outer request ids and carries its own version and client context.
The generated tools are the same ``MCPTool`` classes used for stdio and HTTP.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
from typing import Any
from uuid import uuid4

from mcp.types import CallToolResult, ListToolsResult

from nooa.mcp.tool import _create_tool_instance, _list_server_tools
from nooa.strategies.codeact import current_tool_call_id
from nooa_atom.acp import v2
from nooa_atom.session.port import current_port

MCP_PROTOCOL_VERSION = "2026-07-28"
_MAX_MRTR_ROUNDS = 8


class AcpMCPError(RuntimeError):
    """An inner MCP error or an invalid MCP-over-ACP carrier."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"MCP error {code}: {message}" + (f" ({data})" if data else ""))
        self.code = code
        self.message = message
        self.data = data


def _plain(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json", by_alias=True, exclude_none=True)
    raise TypeError(f"Expected an object, got {type(value).__name__}")


class AcpMCPClient:
    """A stateless MCP client for one declared ``serverId``."""

    def __init__(self, peer: v2.Peer, name: str, server_id: str) -> None:
        self.peer = peer
        self.name = name
        self.server_id = server_id
        self._active = False

    def activate(self) -> None:
        """Allow session attribution after session setup has completed."""
        self._active = True

    def _attribution(self) -> dict[str, str]:
        if not self._active:
            return {}
        port = current_port.get()
        if port is None:
            return {}  # never guess a root or most-recent session
        session_id = port.id
        attribution = {"sessionId": session_id}
        tool_call_id = current_tool_call_id()
        if tool_call_id:
            attribution["toolCallId"] = tool_call_id
        return attribution

    async def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Run one inner MCP request and return its unwrapped result."""
        inner = dict(params or {})
        supplied_meta = inner.get("_meta")
        meta = dict(supplied_meta) if isinstance(supplied_meta, dict) else {}
        meta.update(
            {
                "io.modelcontextprotocol/protocolVersion": MCP_PROTOCOL_VERSION,
                # Atom can answer form elicitation for MRTR, not sampling or roots.
                "io.modelcontextprotocol/clientCapabilities": {"elicitation": {}},
                "io.modelcontextprotocol/clientInfo": v2.agent_info(),
            }
        )
        inner["_meta"] = meta
        envelope = {
            "serverId": self.server_id,
            "requestId": f"mcp-request:{uuid4()}",
            "method": method,
            "params": inner,
            **self._attribution(),
        }
        outer_id: Any = None

        def remember(request_id: Any) -> None:
            nonlocal outer_id
            outer_id = request_id

        try:
            carrier = await self.peer.request("mcp/message", envelope, on_id=remember)
        except asyncio.CancelledError:
            if outer_id is not None:
                with suppress(Exception):
                    await self.peer.notify("$/cancel_request", {"requestId": outer_id})
            raise
        if not isinstance(carrier, dict):
            raise AcpMCPError(-32603, "MCP carrier is not an object", carrier)
        if "result" in carrier:  # explicit null is a result; result wins over error
            return carrier["result"]
        error = carrier.get("error")
        if isinstance(error, dict) and isinstance(error.get("code"), int):
            raise AcpMCPError(
                error["code"], str(error.get("message") or "MCP request failed"), error.get("data")
            )
        raise AcpMCPError(-32603, "MCP carrier has neither result nor error", carrier)

    @asynccontextmanager
    async def connect_to_server(self):
        """Match the MCP client interface; no initialization handshake occurs."""
        yield AcpMCPSession(self)

    async def input_responses(self, requests: dict[str, Any]) -> dict[str, Any]:
        """Answer an MRTR ``inputRequests`` map with ACP form elicitation."""
        attribution = self._attribution()
        if "sessionId" not in attribution:
            raise AcpMCPError(-32602, "MRTR elicitation needs an active ACP session")
        responses: dict[str, Any] = {}
        for key, request in requests.items():
            if not isinstance(key, str) or not isinstance(request, dict):
                raise AcpMCPError(-32602, "Invalid MRTR inputRequests entry", request)
            method = request.get("method")
            if method != "elicitation/create":
                raise AcpMCPError(
                    -32602, f"Unsupported MRTR request {method!r}; Atom accepts elicitation only"
                )
            params = _plain(request.get("params"))
            if not isinstance(params.get("message"), str) or not isinstance(
                params.get("requestedSchema"), dict
            ):
                raise AcpMCPError(-32602, "Invalid MRTR elicitation request", request)
            # ACP owns the user interaction; the result is echoed as this key's
            # ElicitResult. No answer is automatically accepted or rewritten.
            answer = await self.peer.request("elicitation/create", {**params, **attribution})
            responses[key] = _plain(answer)
        return responses


class AcpMCPSession:
    """The subset of an MCP session used by ``MCPTool`` and discovery."""

    def __init__(self, client: AcpMCPClient) -> None:
        self.client = client

    async def list_tools(self, params: Any = None) -> ListToolsResult:
        result = await self.client.request("tools/list", _plain(params))
        try:
            return ListToolsResult.model_validate(result)
        except Exception as exc:
            raise AcpMCPError(-32603, "Invalid tools/list result", result) from exc

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> CallToolResult:
        original = {"name": name, "arguments": arguments or {}}
        params = dict(original)
        for _ in range(_MAX_MRTR_ROUNDS):
            result = await self.client.request("tools/call", params)
            if not isinstance(result, dict):
                raise AcpMCPError(-32603, "Invalid tools/call result", result)
            if result.get("resultType") != "input_required":
                try:
                    return CallToolResult.model_validate(result)
                except Exception as exc:
                    raise AcpMCPError(-32603, "Invalid tools/call result", result) from exc
            requests = result.get("inputRequests")
            state = result.get("requestState")
            if requests is None and state is None:
                raise AcpMCPError(
                    -32602, "input_required has neither inputRequests nor requestState"
                )
            if requests is not None and not isinstance(requests, dict):
                raise AcpMCPError(-32602, "inputRequests is not an object", requests)
            responses = await self.client.input_responses(requests) if requests else {}
            params = dict(original)
            if requests:
                params["inputResponses"] = responses
            if state is not None:
                if not isinstance(state, str):
                    raise AcpMCPError(-32602, "requestState is not a string", state)
                params["requestState"] = state
        raise AcpMCPError(-32603, f"MRTR exceeded {_MAX_MRTR_ROUNDS} rounds")


async def create_acp_server(peer: v2.Peer, name: str, server_id: str) -> Any:
    """Discover an ACP-backed server and build its ordinary ``MCPTool`` instance."""
    client = AcpMCPClient(peer, name, server_id)
    tools_result = await _list_server_tools(name, client)
    tool = _create_tool_instance(name, client, tools_result)
    # Discovery was deliberately unattributed; calls after setup use current_port.
    client.activate()
    return tool


__all__ = [
    "AcpMCPClient",
    "AcpMCPError",
    "AcpMCPSession",
    "MCP_PROTOCOL_VERSION",
    "create_acp_server",
]
