# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MCPTool._call_tool 401 auto-refresh + flattened errors."""

import httpx
import pytest
from mcp.types import CallToolResult, TextContent

from nooa.mcp.tool import MCPTool


def _http_401() -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "https://example/mcp")
    resp = httpx.Response(401, request=req)
    return httpx.HTTPStatusError("401", request=req, response=resp)


class _FakeSession:
    def __init__(self, fail_times: int):
        self._fail_times = fail_times

    async def call_tool(self, name, args):
        if self._fail_times > 0:
            self._fail_times -= 1
            raise ExceptionGroup("tg", [_http_401()])

        class _R:
            content = [type("C", (), {"text": "ok"})()]

        return _R()


class _FakeClient:
    def __init__(self, fail_times: int):
        self._session = _FakeSession(fail_times)

    def connect_to_server(self):
        session = self._session

        class _CM:
            async def __aenter__(self):
                return session

            async def __aexit__(self, *a):
                return False

        return _CM()


def _make_tool(client, refresh_ctx=None):
    t = object.__new__(MCPTool)
    t.__init__(client, "test-server", refresh_ctx=refresh_ctx)
    return t


@pytest.mark.asyncio
async def test_success_no_refresh():
    tool = _make_tool(_FakeClient(fail_times=0))
    assert await tool._call_tool("x", {}) == "ok"


@pytest.mark.asyncio
async def test_tool_level_error_is_not_returned_as_success():
    class _ErrorSession:
        async def call_tool(self, name, args):
            return CallToolResult(
                isError=True,
                content=[TextContent(type="text", text="permission denied")],
            )

    class _ErrorClient:
        def connect_to_server(self):
            session = _ErrorSession()

            class _CM:
                async def __aenter__(self):
                    return session

                async def __aexit__(self, *a):
                    return False

            return _CM()

    tool = _make_tool(_ErrorClient())

    with pytest.raises(RuntimeError) as exc_info:
        await tool._call_tool("read", {})

    message = str(exc_info.value)
    assert "read" in message
    assert "test-server" in message
    assert "permission denied" in message


@pytest.mark.asyncio
async def test_tool_level_error_without_text_uses_fallback_message():
    class _ErrorSession:
        async def call_tool(self, name, args):
            return CallToolResult(
                isError=True,
                content=[],
            )

    class _ErrorClient:
        def connect_to_server(self):
            session = _ErrorSession()

            class _CM:
                async def __aenter__(self):
                    return session

                async def __aexit__(self, *a):
                    return False

            return _CM()

    tool = _make_tool(_ErrorClient())

    with pytest.raises(RuntimeError) as exc_info:
        await tool._call_tool("read", {})

    message = str(exc_info.value)
    assert "read" in message
    assert "test-server" in message
    assert "returned an error" in message


@pytest.mark.asyncio
async def test_401_without_refresh_ctx_raises_actionable():
    tool = _make_tool(_FakeClient(fail_times=1), refresh_ctx=None)
    with pytest.raises(RuntimeError, match="401 Unauthorized"):
        await tool._call_tool("x", {})


@pytest.mark.asyncio
async def test_401_with_refresh_retries_once(monkeypatch):
    # First call 401s; after a successful refresh the retry succeeds.
    client = _FakeClient(fail_times=1)
    tool = _make_tool(client, refresh_ctx={"server_url": "https://example/mcp"})

    async def fake_refresh():
        return True  # pretend the token refreshed; same fake client now succeeds

    monkeypatch.setattr(tool, "_refresh_access_token", fake_refresh)
    assert await tool._call_tool("x", {}) == "ok"


@pytest.mark.asyncio
async def test_401_refresh_fails_raises_actionable(monkeypatch):
    tool = _make_tool(_FakeClient(fail_times=2), refresh_ctx={"server_url": "https://example/mcp"})

    async def fake_refresh():
        return False

    monkeypatch.setattr(tool, "_refresh_access_token", fake_refresh)
    with pytest.raises(RuntimeError, match="re-authenticate|Re-authenticate"):
        await tool._call_tool("x", {})


class _OneResultClient:
    """A client whose session answers every call with ``result``."""

    def __init__(self, result):
        self._result = result

    def connect_to_server(self):
        result = self._result

        class _Session:
            async def call_tool(self, name, args):
                return result

        class _CM:
            async def __aenter__(self):
                return _Session()

            async def __aexit__(self, *a):
                return False

        return _CM()


@pytest.mark.asyncio
async def test_a_structured_error_field_is_returned_and_counted():
    """Failure in the result (not isError): returned as is, counted for harness metrics."""
    from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

    from nooa.mcp.tool import _create_tool_instance
    from nooa.runtime.harness_metrics import harness_metrics_session

    payload = {"error": "order not found"}
    result = CallToolResult(
        content=[TextContent(type="text", text='{"error": "order not found"}')],
        structuredContent=payload,
    )
    tools = ListToolsResult(
        tools=[Tool(name="lookup", inputSchema={"type": "object"}, outputSchema={"type": "object"})]
    )
    tool = _create_tool_instance("hcb", _OneResultClient(result), tools)
    with harness_metrics_session() as metrics:
        assert await tool.lookup() == payload
    assert metrics.to_span_attributes()["harness.mcp_result_error.calls"] == ["hcb.lookup"]


@pytest.mark.asyncio
async def test_an_error_result_raises_mcp_tool_error_with_the_servers_message():
    from nooa.mcp import MCPToolError

    result = CallToolResult(
        isError=True, content=[TextContent(type="text", text="transfer_rejected: no room")]
    )
    with pytest.raises(MCPToolError) as exc_info:
        await _make_tool(_OneResultClient(result))._call_tool("transfer", {})
    error = exc_info.value
    assert isinstance(error, RuntimeError)  # callers catching RuntimeError still do
    assert (error.server, error.tool, error.message) == (
        "test-server",
        "transfer",
        "transfer_rejected: no room",
    )
    assert str(error) == (
        "MCP tool 'transfer' on 'test-server' returned an error: transfer_rejected: no room"
    )


@pytest.mark.asyncio
async def test_a_snake_case_is_error_result_raises_too():
    from types import SimpleNamespace

    from nooa.mcp import MCPToolError

    result = SimpleNamespace(
        is_error=True, structured_content={"error": {"code": "denied"}}, content=[]
    )
    with pytest.raises(MCPToolError, match='"code": "denied"'):
        await _make_tool(_OneResultClient(result))._call_tool("read", {})


@pytest.mark.asyncio
async def test_structured_content_is_returned_as_a_dict():
    result = CallToolResult(
        content=[TextContent(type="text", text='{"receipt": "PUB-1"}')],
        structuredContent={"receipt": "PUB-1"},
    )
    assert await _make_tool(_OneResultClient(result))._call_tool("publish", {}) == {
        "receipt": "PUB-1"
    }


@pytest.mark.asyncio
async def test_a_text_only_result_is_returned_as_its_text():
    result = CallToolResult(content=[TextContent(type="text", text='{"receipt": "PUB-1"}')])
    assert (
        await _make_tool(_OneResultClient(result))._call_tool("publish", {})
        == '{"receipt": "PUB-1"}'
    )


def test_a_tool_with_an_output_schema_is_annotated_as_returning_a_dict():
    from typing import Any

    from mcp.types import ListToolsResult, Tool

    from nooa.mcp.tool import _create_tool_instance

    tools = ListToolsResult(
        tools=[
            Tool(name="typed", inputSchema={"type": "object"}, outputSchema={"type": "object"}),
            Tool(name="plain", inputSchema={"type": "object"}),
            # FastMCP's wrapper of ``-> str``: the call returns the text, not a dict.
            Tool(
                name="wrapped",
                inputSchema={"type": "object"},
                outputSchema={
                    "type": "object",
                    "properties": {"result": {"type": "string"}},
                    "required": ["result"],
                },
            ),
        ]
    )
    tool = _create_tool_instance("srv", _OneResultClient(None), tools)
    assert type(tool).typed.__annotations__["return"] == dict[str, Any]
    assert type(tool).plain.__annotations__["return"] is Any
    assert type(tool).wrapped.__annotations__["return"] is Any
    assert "MCPToolError is raised" in type(tool).__doc__
    assert "report failure in its result" in type(tool).__doc__


@pytest.mark.asyncio
async def test_a_wrapped_text_result_is_still_returned_as_its_text():
    """FastMCP wraps ``-> str`` / ``-> int`` results as ``{"result": ...}``."""
    for text, wrapped in (("## Notes", "## Notes"), ("5", 5)):
        result = CallToolResult(
            content=[TextContent(type="text", text=text)], structuredContent={"result": wrapped}
        )
        assert await _make_tool(_OneResultClient(result))._call_tool("notes", {}) == text
