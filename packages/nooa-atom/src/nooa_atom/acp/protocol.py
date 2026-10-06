# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the router and the adapter both need from ACP: the ``initialize`` answer and stdio.

Imports only the ``acp`` library, so the router can use it without loading
the adapter and the agent stack behind it.
"""

from __future__ import annotations

import asyncio
import os
from importlib.metadata import PackageNotFoundError, version

from acp import PROTOCOL_VERSION, InitializeResponse
from acp.core import DEFAULT_STDIO_BUFFER_LIMIT_BYTES
from acp.schema import (
    AgentCapabilities,
    Implementation,
    McpCapabilities,
    SessionCapabilities,
    SessionCloseCapabilities,
    SessionListCapabilities,
)

INJECT_CAPABILITY = {"dev.nooa/inject": {"queue": {}, "steer": {}, "revoke": {}}}
"""``agentCapabilities._meta`` for ``_nooa/session/inject`` and ``revoke_inject``.

They follow the ACP RFD for message injection (agent-client-protocol PR #1261).
"""

POOL_STEER_CAPABILITY = {"poolside/session_steer": True}
"""``agentCapabilities._meta`` that makes Pool send ``_poolside/session_steer``.

With it, Pool sends what the person types during a running turn as that
request instead of keeping it in its own queue.
"""


def initialize_response(protocol_version: int) -> InitializeResponse:
    """The static answer to ``initialize``: what this agent supports.

    ``session/delete`` and ``logout`` are not advertised: the 0.12 library
    does not route them, so a client calling them would get "method not
    found". Deleting is the ``_nooa/session/delete`` extension method.
    """
    try:
        package_version = version("nooa-atom")
    except PackageNotFoundError:
        package_version = "0.0.0"
    return InitializeResponse(
        protocol_version=min(protocol_version, PROTOCOL_VERSION),
        agent_capabilities=AgentCapabilities(
            load_session=True,
            # McpCapabilities defaults to all-false, and a client that honours
            # the handshake then filters its HTTP/SSE servers out of
            # session/new. _create_mcp_tools connects both transports, so say
            # so. `acp` stays off: it is unstable in the spec and not
            # implemented here.
            mcp_capabilities=McpCapabilities(http=True, sse=True),
            session_capabilities=SessionCapabilities(
                list=SessionListCapabilities(),
                close=SessionCloseCapabilities(),
            ),
            field_meta={**INJECT_CAPABILITY, **POOL_STEER_CAPABILITY},
        ),
        auth_methods=[],
        agent_info=Implementation(
            name="nooa-atom",
            title="NVIDIA Labs Object Oriented Agents (NOOA)",
            version=package_version,
        ),
    )


class _WritePipeProtocol(asyncio.BaseProtocol):
    """Flow control for the output pipe (as ``acp.stdio`` does for stdout)."""

    def __init__(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._paused = False
        self._drain_waiter: asyncio.Future[None] | None = None

    def pause_writing(self) -> None:
        self._paused = True
        if self._drain_waiter is None:
            self._drain_waiter = self._loop.create_future()

    def resume_writing(self) -> None:
        self._paused = False
        if self._drain_waiter is not None and not self._drain_waiter.done():
            self._drain_waiter.set_result(None)
        self._drain_waiter = None

    async def _drain_helper(self) -> None:
        if self._paused and self._drain_waiter is not None:
            await self._drain_waiter


async def _stdio_streams(
    input_fd: int, output_fd: int
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """A reader on ``input_fd`` and a writer on ``output_fd`` (the reserved real stdio)."""
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader(limit=DEFAULT_STDIO_BUFFER_LIMIT_BYTES)
    await loop.connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader),
        os.fdopen(input_fd, "rb", buffering=0, closefd=False),
    )
    protocol = _WritePipeProtocol()
    transport, _ = await loop.connect_write_pipe(
        lambda: protocol, os.fdopen(output_fd, "wb", buffering=0, closefd=False)
    )
    return reader, asyncio.StreamWriter(transport, protocol, None, loop)


async def open_stdio(
    input_fd: int | None = None, output_fd: int | None = None
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Streams for ACP frames: on the reserved descriptors when given, else stdin and stdout."""
    if input_fd is not None and output_fd is not None:
        return await _stdio_streams(input_fd, output_fd)
    from acp.stdio import stdio_streams

    return await stdio_streams(limit=DEFAULT_STDIO_BUFFER_LIMIT_BYTES)


__all__ = ["INJECT_CAPABILITY", "POOL_STEER_CAPABILITY", "initialize_response", "open_stdio"]
