# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the router and the adapter both need from ACP: the ``initialize`` answer and stdio.

Imports only the ``acp`` library, so the router can use it without loading
the adapter and the agent stack behind it.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version

from acp import PROTOCOL_VERSION, InitializeResponse
from acp.core import DEFAULT_STDIO_BUFFER_LIMIT_BYTES
from acp.schema import (
    AgentCapabilities,
    ClientCapabilities,
    Implementation,
    McpCapabilities,
    SessionCapabilities,
    SessionCloseCapabilities,
    SessionListCapabilities,
)

from nooa_atom.acp.terminal_auth import connect_auth_method, supports_terminal_auth

INJECT_CAPABILITY = {"dev.nooa/inject": {"queue": {}, "steer": {}, "revoke": {}}}
"""``agentCapabilities._meta`` for ``_nooa/session/inject`` and ``revoke_inject``.

They follow the ACP RFD for message injection (agent-client-protocol PR #1261).
"""


@dataclass(frozen=True)
class PoolDialect:
    """The names one generation of Pool uses for its ACP extensions.

    Method names are as ``ext_method`` sees them: the wire name without its
    leading underscore (``_pool/session_steer`` -> ``pool/session_steer``).
    """

    steer_method: str
    """Pool's steer request, ``{sessionId, inputId, prompt}`` -> ``{inputId}``."""
    form_method: str
    """Pool's form request (``{sessionId, mode, message, requestedSchema}``)."""
    steer_capability: str
    """``agentCapabilities._meta`` key that makes Pool send ``steer_method``."""
    input_event_id: str
    """``session_info_update`` meta key: the id of an input the agent took."""
    client_input_id: str
    """``session_info_update`` meta key: the client's id of that input (a steer's ``inputId``)."""
    form_has_id: bool
    """Whether a form request carries an ``elicitationId``."""


POOLSIDE_DIALECT = PoolDialect(
    steer_method="poolside/session_steer",
    form_method="poolside/elicitation",
    steer_capability="poolside/session_steer",
    input_event_id="poolside/inputEventId",
    client_input_id="poolside/clientInputId",
    form_has_id=False,
)
"""Pool 1.x (measured with Pool 1.0.16 against its own agent, ``pool acp``)."""

POOL_DIALECT = PoolDialect(
    steer_method="pool/session_steer",
    form_method="pool/elicitation",
    steer_capability="pool/session_steer",
    input_event_id="pool/inputEventId",
    client_input_id="pool/clientInputId",
    form_has_id=True,
)
"""Pool 2.x (forge ``pkg/acp/extensions.go``, ``pkg/acp/updates.go`` and
``pkg/agent/elicitation/types.go``), where ``elicitationId`` is required."""

_POOL_MAJOR = re.compile(r"v?(\d+)\.")


def pool_dialects(client_info: Implementation | None) -> tuple[PoolDialect, ...]:
    """The Pool dialects to speak to the client named in ``initialize``.

    Both Pool generations send ``clientInfo.name == "pool"``; the version
    (``1.0.16``, ``v2.0.0-alpha``) tells them apart. ``()`` for a client
    that is not Pool, both dialects for a Pool whose version has no major
    number (a local build reports ``untagged``): Pool reads these meta keys
    by name and ignores the others.
    """
    if client_info is None or client_info.name != "pool":
        return ()
    match = _POOL_MAJOR.match(client_info.version or "")
    if match is None:
        return (POOLSIDE_DIALECT, POOL_DIALECT)
    return (POOL_DIALECT,) if int(match.group(1)) >= 2 else (POOLSIDE_DIALECT,)


POOL_STEER_CAPABILITY = {POOLSIDE_DIALECT.steer_capability: True}
"""``agentCapabilities._meta`` that makes Pool send ``_poolside/session_steer``.

With it, Pool sends what the person types during a running turn as that
request instead of keeping it in its own queue. Clients that are not Pool
get this one, as before Pool 2.
"""


def steer_capability(dialects: tuple[PoolDialect, ...]) -> dict[str, bool]:
    """The steer capabilities to advertise to a client speaking ``dialects``."""
    if not dialects:
        return dict(POOL_STEER_CAPABILITY)
    return {dialect.steer_capability: True for dialect in dialects}


def initialize_response(
    protocol_version: int,
    client_info: Implementation | None = None,
    client_capabilities: ClientCapabilities | None = None,
) -> InitializeResponse:
    """The answer to ``initialize``: what this agent supports.

    A client that runs ``terminal`` auth methods (``auth.terminal``) is
    offered one: ``nooa connect`` in a terminal (``terminal_auth``). Other
    clients get no auth methods.

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
            field_meta={**INJECT_CAPABILITY, **steer_capability(pool_dialects(client_info))},
        ),
        auth_methods=[connect_auth_method()] if supports_terminal_auth(client_capabilities) else [],
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


__all__ = [
    "INJECT_CAPABILITY",
    "POOLSIDE_DIALECT",
    "POOL_DIALECT",
    "POOL_STEER_CAPABILITY",
    "PoolDialect",
    "initialize_response",
    "open_stdio",
    "pool_dialects",
    "steer_capability",
]
