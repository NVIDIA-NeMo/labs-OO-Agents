# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for complete MCP tool discovery across paginated responses."""

import asyncio
from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, call, patch

import httpx
import pytest

pytest.importorskip("mcp")

from mcp.types import PaginatedRequestParams  # noqa: E402

import nooa.mcp.tool as tool_module  # noqa: E402
from nooa.mcp.tool import MCPManager, _list_all_tools  # noqa: E402


def _remote_tool(name: str) -> SimpleNamespace:
    """Build the minimum MCP tool shape consumed by the dynamic factory."""
    return SimpleNamespace(
        name=name,
        description=f"Tool {name}",
        inputSchema={"type": "object", "properties": {}},
    )


def _page(
    *tool_names: str,
    next_cursor: str | None = None,
    snake_case: bool = False,
) -> SimpleNamespace:
    """Build an MCP list-tools page using either supported cursor field name."""
    cursor_field = "next_cursor" if snake_case else "nextCursor"
    return SimpleNamespace(
        tools=[_remote_tool(name) for name in tool_names],
        **{cursor_field: next_cursor},
    )


def _client_for(session: AsyncMock):
    """Build a client whose connection yields the supplied mock session."""

    @asynccontextmanager
    async def connect():
        """Yield the supplied session as an MCP connection."""
        yield session

    return SimpleNamespace(connect_to_server=connect)


@pytest.mark.asyncio
async def test_list_all_tools_follows_cursors_and_keeps_empty_pages():
    """Discovery must continue when an intermediate page contains no tools."""
    session = AsyncMock()
    session.list_tools.side_effect = [
        _page("page_one", next_cursor="cursor-2", snake_case=True),
        _page(next_cursor="cursor-3"),
        _page("page_three"),
    ]

    result = await _list_all_tools(session)

    assert [tool.name for tool in result.tools] == ["page_one", "page_three"]
    assert session.list_tools.await_args_list == [
        call(),
        call(params=PaginatedRequestParams(cursor="cursor-2")),
        call(params=PaginatedRequestParams(cursor="cursor-3")),
    ]


@pytest.mark.asyncio
async def test_list_all_tools_preserves_an_empty_string_cursor():
    """An empty cursor is still a present cursor and must be sent unchanged."""
    session = AsyncMock()
    session.list_tools.side_effect = [
        _page("page_one", next_cursor=""),
        _page("page_two"),
    ]

    result = await _list_all_tools(session)

    assert [tool.name for tool in result.tools] == ["page_one", "page_two"]
    assert session.list_tools.await_args_list == [
        call(),
        call(params=PaginatedRequestParams(cursor="")),
    ]


@pytest.mark.asyncio
async def test_list_all_tools_rejects_a_repeated_cursor():
    """A malformed server must not trap discovery in a pagination cycle."""
    session = AsyncMock()
    session.list_tools.side_effect = [
        _page("page_one", next_cursor="same"),
        _page("page_two", next_cursor="same"),
    ]

    with pytest.raises(RuntimeError, match="repeated pagination cursor 'same'"):
        await _list_all_tools(session)

    assert session.list_tools.await_count == 2


@pytest.mark.asyncio
async def test_list_all_tools_limits_non_repeating_infinite_pagination(monkeypatch):
    """Unique cursors must still have a finite safety bound."""
    monkeypatch.setattr(tool_module, "_MAX_MCP_TOOL_DISCOVERY_PAGES", 2)
    session = AsyncMock()
    session.list_tools.side_effect = [
        _page("page_one", next_cursor="cursor-2"),
        _page("page_two", next_cursor="cursor-3"),
    ]

    with pytest.raises(RuntimeError, match="exceeded 2 pages"):
        await _list_all_tools(session)

    assert session.list_tools.await_count == 2


@pytest.mark.asyncio
async def test_async_factory_bounds_total_time_across_pages():
    """Fast individual pages must not bypass the overall discovery deadline."""
    page_count = 0
    connection_closed = False
    session = AsyncMock()

    async def list_tools(*args, **kwargs):
        """Return cursor pages slowly enough to exhaust the shared deadline."""
        nonlocal page_count
        page_count += 1
        await asyncio.sleep(0.02)
        return _page(f"page_{page_count}", next_cursor=f"cursor-{page_count + 1}")

    session.list_tools.side_effect = list_tools

    @asynccontextmanager
    async def connect():
        """Track that timeout cancellation closes the connection context."""
        nonlocal connection_closed
        try:
            yield session
        finally:
            connection_closed = True

    client = SimpleNamespace(connect_to_server=connect)

    with (
        patch("nooa.mcp.tool.create_mcp_client", return_value=client),
        pytest.raises(RuntimeError, match="overall 0.05-second timeout"),
    ):
        await MCPManager.create_stdio_server(
            "slow-pages",
            "paged-server",
            discovery_timeout=timedelta(milliseconds=50),
        )

    assert 1 < page_count < 10
    assert connection_closed


@pytest.mark.asyncio
async def test_async_factory_generates_methods_from_every_page():
    """The async discovery entry point must expose tools beyond page one."""
    session = AsyncMock()
    session.list_tools.side_effect = [
        _page("page_one", next_cursor="cursor-2"),
        _page("page_two", next_cursor="cursor-3"),
        _page("page_three"),
    ]
    client = _client_for(session)

    with patch("nooa.mcp.tool.create_mcp_client", return_value=client):
        tool = await MCPManager.create_stdio_server("paged", "paged-server")

    assert tool._tool_method_names == frozenset(  # type: ignore[attr-defined]
        {"page_one", "page_two", "page_three"}
    )


def test_sync_factory_generates_methods_from_every_page():
    """The synchronous discovery entry point must expose all pages too."""
    session = AsyncMock()
    session.list_tools.side_effect = [
        _page("page_one", next_cursor="cursor-2"),
        _page("page_two"),
    ]
    client = _client_for(session)

    with patch("nooa.mcp.tool.create_mcp_client", return_value=client):
        tool = MCPManager.create_from_server("paged", command="paged-server")

    assert tool._tool_method_names == frozenset(  # type: ignore[attr-defined]
        {"page_one", "page_two"}
    )


@pytest.mark.asyncio
async def test_sync_factory_preserves_timeout_inside_running_loop():
    """A discovery timeout from the worker thread must propagate unchanged."""
    session = AsyncMock()

    async def list_tools(*args, **kwargs):
        """Block until the overall discovery timeout cancels this request."""
        await asyncio.Event().wait()

    session.list_tools.side_effect = list_tools
    client = _client_for(session)

    with (
        patch("nooa.mcp.tool.create_mcp_client", return_value=client),
        pytest.raises(RuntimeError, match="overall 0.02-second timeout"),
    ):
        MCPManager.create_from_server(
            "slow-sync",
            command="paged-server",
            discovery_timeout=timedelta(milliseconds=20),
        )


def test_oauth_retry_generates_methods_from_every_page():
    """A successful OAuth retry must use the same complete discovery path."""
    request = httpx.Request("POST", "https://mcp.example.test/mcp")
    response = httpx.Response(401, request=request)
    unauthorized = httpx.HTTPStatusError("unauthorized", request=request, response=response)

    @asynccontextmanager
    async def fail_to_connect():
        """Simulate a server that requires OAuth authentication."""
        raise unauthorized
        yield  # pragma: no cover

    first_client = SimpleNamespace(connect_to_server=fail_to_connect)

    session = AsyncMock()
    session.list_tools.side_effect = [
        _page("page_one", next_cursor="cursor-2"),
        _page("page_two"),
    ]
    refreshed_client = _client_for(session)
    token = SimpleNamespace(token_type="Bearer", access_token="refreshed")

    with (
        patch(
            "nooa.mcp.tool.create_mcp_client",
            side_effect=[first_client, refreshed_client],
        ),
        patch("nooa.mcp.tool.handle_mcp_oauth", new=AsyncMock(return_value=token)),
    ):
        tool = MCPManager.create_from_server(
            "paged",
            url="https://mcp.example.test/mcp",
            transport="streamable-http",
            oauth_open_browser=False,
        )

    assert tool._tool_method_names == frozenset(  # type: ignore[attr-defined]
        {"page_one", "page_two"}
    )


def test_oauth_flow_uses_the_shared_discovery_deadline():
    """OAuth must not reset or escape the overall discovery timeout."""
    request = httpx.Request("POST", "https://mcp.example.test/mcp")
    response = httpx.Response(401, request=request)
    unauthorized = httpx.HTTPStatusError("unauthorized", request=request, response=response)

    @asynccontextmanager
    async def fail_to_connect():
        """Simulate a server that requires OAuth authentication."""
        raise unauthorized
        yield  # pragma: no cover

    async def wait_for_user(*args, **kwargs):
        """Simulate an OAuth flow waiting indefinitely for user input."""
        await asyncio.Event().wait()

    first_client = SimpleNamespace(connect_to_server=fail_to_connect)

    with (
        patch("nooa.mcp.tool.create_mcp_client", return_value=first_client),
        patch("nooa.mcp.tool.handle_mcp_oauth", new=AsyncMock(side_effect=wait_for_user)) as oauth,
        pytest.raises(RuntimeError, match="overall 0.02-second timeout"),
    ):
        MCPManager.create_from_server(
            "slow-oauth",
            url="https://mcp.example.test/mcp",
            transport="streamable-http",
            oauth_open_browser=False,
            discovery_timeout=timedelta(milliseconds=20),
        )

    oauth.assert_awaited_once()


def test_oauth_timeout_cannot_exceed_discovery_timeout():
    """Reject an OAuth wait that cannot fit within the discovery budget."""
    with (
        patch("nooa.mcp.tool.create_mcp_client") as create_client,
        pytest.raises(ValueError, match="oauth_timeout must not exceed discovery_timeout"),
    ):
        MCPManager.create_from_server(
            "invalid-oauth-timeout",
            command="paged-server",
            oauth_timeout=301,
            discovery_timeout=timedelta(minutes=5),
        )

    create_client.assert_not_called()


@pytest.mark.asyncio
async def test_discovery_timeout_must_be_positive():
    """Reject a disabled deadline instead of permitting unbounded discovery."""
    client = _client_for(AsyncMock())

    with (
        patch("nooa.mcp.tool.create_mcp_client", return_value=client),
        pytest.raises(ValueError, match="discovery_timeout must be greater than zero"),
    ):
        await MCPManager.create_stdio_server(
            "invalid-timeout",
            "paged-server",
            discovery_timeout=timedelta(0),
        )
