# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, Literal, override

import httpx
from mcp import (
    ClientSession,
    StdioServerParameters,
    stdio_client,
)
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client


class MCPBaseClient(ABC):
    """Base client for creating an MCP transport session and connecting to an MCP server.

    Args:
        tool_call_timeout: Timeout for tool calls
    """

    def __init__(
        self,
        tool_call_timeout: timedelta = timedelta(seconds=60),
    ):
        self._tool_call_timeout = tool_call_timeout

    @property
    @abstractmethod
    def transport(self) -> Literal["sse", "stdio", "streamable-http"]:
        """Return the transport type."""

    @property
    @abstractmethod
    def server_config(self) -> dict[str, Any]:
        """Return the server configuration."""

    @property
    def tool_call_timeout(self) -> timedelta:
        """Return the timeout for tool calls."""
        return self._tool_call_timeout

    @abstractmethod
    @asynccontextmanager
    async def connect_to_server(self) -> AsyncGenerator[ClientSession, None]:
        """Establish a session with an MCP server within an async context.

        Raises:
            Various exceptions depending on transport:
            - Connection errors (network failures, DNS errors, timeouts)
            - Authentication errors (401 Unauthorized, OAuth errors)
            - Protocol errors (MCP initialization failures)
            - Process errors (for stdio: command not found, permission denied)
        """
        yield  # type: ignore[misc]


class MCPSSEClient(MCPBaseClient):
    """Client for creating a session and connecting to an MCP server using SSE.

    Args:
        url: The URL of the MCP server
        tool_call_timeout: Timeout for tool calls
    """

    def __init__(
        self,
        url: str,
        tool_call_timeout: timedelta = timedelta(seconds=60),
    ):
        super().__init__(tool_call_timeout=tool_call_timeout)
        self._url = url

    @property
    def transport(self) -> Literal["sse", "stdio", "streamable-http"]:
        """Return the transport type."""
        return "sse"

    @property
    def server_config(self) -> dict[str, Any]:
        """Return the server configuration."""
        return {
            "url": self._url,
            "transport": self.transport,
        }

    @property
    def url(self) -> str:
        """Return the server URL."""
        return self._url

    @asynccontextmanager
    @override
    async def connect_to_server(self):
        """Establish a session with an MCP SSE server within an async context.

        Raises:
            httpx.ConnectError: If connection to server fails (network error, DNS failure)
            httpx.TimeoutException: If connection times out
            httpx.HTTPStatusError: If server returns HTTP error (e.g., 401, 500)
            RuntimeError: If session initialization fails (MCP protocol error)
        """
        async with (
            sse_client(url=self._url) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            yield session


class MCPStdioClient(MCPBaseClient):
    """Client for creating a session and connecting to an MCP server using stdio.

    This is a local transport that spawns the MCP server process and communicates
    with it over stdin/stdout.

    Args:
        command: The command to run
        args: Additional arguments for the command
        env: Environment variables to set for the process
        tool_call_timeout: Timeout for tool calls
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        tool_call_timeout: timedelta = timedelta(seconds=60),
    ):
        super().__init__(tool_call_timeout=tool_call_timeout)
        self._command = command
        self._args = args
        self._env = env

    @property
    def transport(self) -> Literal["sse", "stdio", "streamable-http"]:
        """Return the transport type."""
        return "stdio"

    @property
    def server_config(self) -> dict[str, Any]:
        """Return the server configuration."""
        return {
            "command": self._command,
            "args": self._args or [],
            "env": self._env,
            "transport": self.transport,
        }

    @property
    def command(self) -> str:
        """Return the command to run."""
        return self._command

    @property
    def args(self) -> list[str] | None:
        """Return the command arguments."""
        return self._args

    @property
    def env(self) -> dict[str, str] | None:
        """Return the environment variables."""
        return self._env

    @asynccontextmanager
    @override
    async def connect_to_server(self):
        """Establish a session with an MCP server via stdio within an async context.

        Raises:
            FileNotFoundError: If command is not found in PATH
            PermissionError: If command cannot be executed
            RuntimeError: If process spawn fails or session initialization fails (MCP protocol error)
        """

        server_params = StdioServerParameters(
            command=self._command, args=self._args or [], env=self._env
        )
        async with (
            stdio_client(server_params) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            yield session


class MCPStreamableHTTPClient(MCPBaseClient):
    """Client for creating a session and connecting to an MCP server using streamable-http.

    Args:
        url: The URL of the MCP server
        headers: Optional custom HTTP headers to include in requests (e.g., for authentication)
        tool_call_timeout: Timeout for tool calls
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        tool_call_timeout: timedelta = timedelta(seconds=60),
    ):
        super().__init__(tool_call_timeout=tool_call_timeout)
        self._url = url
        self._headers = headers or {}
        # Callback to retrieve MCP session ID from the transport layer
        self._get_mcp_session_id: Callable[[], str | None] | None = None

    @property
    def transport(self) -> Literal["sse", "stdio", "streamable-http"]:
        """Return the transport type."""
        return "streamable-http"

    @property
    def server_config(self) -> dict[str, Any]:
        """Return the server configuration."""
        return {
            "url": self._url,
            "headers": self._headers,
            "transport": self.transport,
        }

    @property
    def url(self) -> str:
        """Return the server URL."""
        return self._url

    @property
    def headers(self) -> dict[str, str]:
        """Return the custom headers configured for this client."""
        return self._headers

    @property
    def mcp_session_id(self) -> str | None:
        """Return the MCP transport-level session ID if available.

        This is the session ID assigned by the MCP server (from the mcp-session-id header),
        which can be used for correlating backend sessions with MCP server sessions.

        Returns:
            The MCP session ID string, or None if not connected or not available.
        """
        if self._get_mcp_session_id is not None:
            return self._get_mcp_session_id()
        return None

    @asynccontextmanager
    @override
    async def connect_to_server(self):
        """Establish a session with an MCP server via streamable-http within an async context.

        Raises:
            httpx.ConnectError: If connection to server fails (network error, DNS failure)
            httpx.TimeoutException: If connection times out
            httpx.HTTPStatusError: If server returns HTTP error (e.g., 401 Unauthorized, 500)
            RuntimeError: If session initialization fails (MCP protocol error)
        """
        # Create httpx client with custom headers
        # streamable_http_client expects a pre-configured httpx.AsyncClient for headers
        http_client = httpx.AsyncClient(headers=self._headers if self._headers else None)

        try:
            async with (
                http_client,
                streamable_http_client(url=self._url, http_client=http_client) as (
                    read,
                    write,
                    get_session_id,
                ),
            ):
                # Store the session ID callback for later retrieval
                self._get_mcp_session_id = get_session_id
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
        finally:
            # Clear the session ID callback when disconnected
            self._get_mcp_session_id = None


def create_mcp_client(
    transport: Literal["stdio", "sse", "streamable-http"],
    url: str | None = None,
    command: str | None = None,
    args: list[str] | None = None,
    env: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> MCPBaseClient:
    """Create an MCP client based on the transport type and configuration.

    Args:
        transport: Transport type - "stdio", "sse", or "streamable-http"
        url: The URL of the MCP server (required for sse and streamable-http transports)
        command: Command to run the server (required for stdio transport)
        args: Command arguments (optional, for stdio transport)
        env: Environment variables for the server process (optional, for stdio transport)
        headers: Optional custom HTTP headers to include in requests (for HTTP transports)

    Returns:
        An MCPBaseClient instance configured for the specified transport

    Raises:
        ValueError: If required parameters are missing for the specified transport,
            or if both url and command are None, or if transport is unsupported
    """
    if url is None and command is None:
        raise ValueError("Either url or command must be provided")

    match transport:
        case "stdio":
            if command is None:
                raise ValueError("command must be provided for stdio transport")
            return MCPStdioClient(command=command, args=args, env=env)
        case "sse":
            if url is None:
                raise ValueError("url must be provided for sse transport")
            return MCPSSEClient(url=url)
        case "streamable-http":
            if url is None:
                raise ValueError("url must be provided for streamable-http transport")
            return MCPStreamableHTTPClient(url=url, headers=headers)
        case _:
            raise ValueError(
                f"Unsupported transport type: {transport}. Use 'stdio', 'sse', or 'streamable-http'"
            )
