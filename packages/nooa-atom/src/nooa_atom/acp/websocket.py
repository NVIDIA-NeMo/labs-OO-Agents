# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ACP over WebSocket: ``nooa coder --http``.

Follows the WebSocket profile of the ACP remote transport proposal
("Streamable HTTP & WebSocket Transport"): one endpoint, ``/acp``; a
``GET`` with ``Upgrade: websocket`` opens a connection; every JSON-RPC
message is one text frame; the client sends ``initialize`` first. The
proposal lets a server offer WebSocket only, which this does; the
Streamable HTTP profile (``POST`` and server-sent events) is not served.

Each connection gets its own :class:`~nooa_coder.acp.router.Router`, so a
remote client gets exactly what a client on standard input and output
gets: one worker process per root session, started and stopped by that
connection. The router is a line proxy, and a text frame is one line.
Sessions are stored on disk as usual, so a client that reconnects loads
its sessions with ``session/load``; the session lock keeps two
connections from running the same session.

Every upgrade request is checked before it is accepted:

- the path must be ``/acp`` (404 otherwise);
- an ``Origin`` header, which browsers send, must be a loopback origin or
  one given with ``--allowed-origin`` (403 otherwise), so a web page the
  person visits cannot reach the server;
- unless the server runs with ``--no-auth``, the request must carry the
  token as ``Authorization: Bearer <token>`` or, for browsers, which
  cannot set headers on a WebSocket, as ``?token=<token>`` (401
  otherwise).

There is no TLS: the server binds to loopback by default; reach it from
another machine through an SSH tunnel or a TLS-terminating proxy.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import ipaddress
import logging
import uuid
from collections.abc import Callable, Iterable
from http import HTTPStatus
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlsplit

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from nooa_coder.acp.framing import FRAME_LIMIT
from nooa_coder.acp.router import Router, Spawn

logger = logging.getLogger(__name__)

ACP_PATH = "/acp"
CONNECTION_ID_HEADER = "Acp-Connection-Id"


def is_loopback_host(host: str) -> bool:
    """Whether ``host`` (a name or address, IPv6 with or without brackets) is this machine only."""
    name = host.strip("[]").lower()
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


class Gate:
    """Decides whether an upgrade request may open an ACP connection."""

    def __init__(
        self,
        *,
        token: str | None,
        allowed_origins: Iterable[str] = (),
        path: str = ACP_PATH,
    ) -> None:
        self._token = token
        self._allowed_origins = {origin.rstrip("/") for origin in allowed_origins}
        self._path = path

    def check(self, path: str, headers: Headers) -> tuple[HTTPStatus, str] | None:
        """``None`` to accept, else the status and a one-line reason."""
        url = urlsplit(path)
        if url.path != self._path:
            return HTTPStatus.NOT_FOUND, f"ACP is served at {self._path}"
        origin = headers.get("Origin")
        if origin is not None and not self._origin_allowed(origin):
            return HTTPStatus.FORBIDDEN, f"Origin {origin} is not allowed"
        if self._token is not None and not self._token_matches(url.query, headers):
            return HTTPStatus.UNAUTHORIZED, "A valid token is required"
        return None

    def _origin_allowed(self, origin: str) -> bool:
        if origin.rstrip("/") in self._allowed_origins:
            return True
        host = urlsplit(origin).hostname
        return host is not None and is_loopback_host(host)

    def _token_matches(self, query: str, headers: Headers) -> bool:
        assert self._token is not None
        candidates: list[str] = []
        authorization = headers.get("Authorization", "")
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer" and value:
            candidates.append(value.strip())
        candidates.extend(parse_qs(query).get("token", []))
        expected = self._token.encode()
        return any(hmac.compare_digest(candidate.encode(), expected) for candidate in candidates)


class _FrameWriter:
    """The writer side the router needs (``write``, ``drain``, ``close``), sending one text frame per line."""

    def __init__(self, websocket: ServerConnection) -> None:
        self._websocket = websocket
        self._buffer = bytearray()

    def write(self, data: bytes) -> None:
        self._buffer += data

    async def drain(self) -> None:
        while (end := self._buffer.find(b"\n")) >= 0:
            line = bytes(self._buffer[:end])
            del self._buffer[: end + 1]
            if not line.strip():
                continue
            try:
                await self._websocket.send(line.decode("utf-8"))
            except ConnectionClosed as exc:
                # The router treats ConnectionError as "the client stopped reading".
                raise ConnectionError(str(exc)) from exc

    def close(self) -> None:
        pass  # the connection handler closes the socket


def _one_line(text: str) -> bytes:
    """A text frame as one router line.

    A JSON text can only contain a raw line break between tokens (inside a
    string it must be escaped), so replacing line breaks with spaces keeps
    the message intact for a client that sends indented JSON.
    """
    return text.replace("\r", " ").replace("\n", " ").encode("utf-8") + b"\n"


async def _pump_client(websocket: ServerConnection, reader: asyncio.StreamReader) -> None:
    """Feed the client's text frames to the router; end of input when the socket closes."""
    try:
        async for message in websocket:
            if isinstance(message, bytes):
                continue  # binary frames carry nothing in ACP
            reader.feed_data(_one_line(message))
    except ConnectionClosed:
        pass
    finally:
        reader.feed_eof()


async def serve_router(
    websocket: ServerConnection,
    *,
    spawn: Spawn,
    sessions_dir: Path | None,
) -> None:
    """Serve one WebSocket connection through its own router until either side ends it."""
    peer = websocket.remote_address
    logger.info("ACP connection from %s", peer)
    reader = asyncio.StreamReader(limit=FRAME_LIMIT)
    pump = asyncio.create_task(_pump_client(websocket, reader), name="nooa-websocket-client")
    router = Router(spawn=spawn, sessions_dir=sessions_dir)
    try:
        await router.serve(reader, cast(Any, _FrameWriter(websocket)))
    finally:
        pump.cancel()
        await asyncio.gather(pump, return_exceptions=True)
        with contextlib.suppress(Exception):
            await websocket.close()
        logger.info("ACP connection from %s closed", peer)


async def serve_websocket(
    *,
    host: str,
    port: int,
    spawn: Spawn,
    sessions_dir: Path | None,
    gate: Gate,
    stop: asyncio.Event,
    on_listening: Callable[[str], None] | None = None,
) -> None:
    """Accept ACP connections on ``ws://host:port/acp`` until ``stop`` is set.

    On stop, open connections are closed; each router then stops its
    workers, which checkpoint their sessions, before this returns.
    """

    def process_request(connection: ServerConnection, request: Request) -> Response | None:
        refusal = gate.check(request.path, request.headers)
        if refusal is None:
            return None
        status, reason = refusal
        logger.warning("Refused a connection from %s: %s", connection.remote_address, reason)
        return connection.respond(status, reason + "\n")

    def process_response(
        connection: ServerConnection, request: Request, response: Response
    ) -> Response | None:
        if response.status_code == HTTPStatus.SWITCHING_PROTOCOLS:
            response.headers[CONNECTION_ID_HEADER] = uuid.uuid4().hex
        return None

    async def handler(websocket: ServerConnection) -> None:
        await serve_router(websocket, spawn=spawn, sessions_dir=sessions_dir)

    server: Server
    async with serve(
        handler,
        host,
        port,
        process_request=process_request,
        process_response=process_response,
        max_size=FRAME_LIMIT,
    ) as server:
        for sock in server.sockets:
            address = sock.getsockname()
            bound_host = f"[{address[0]}]" if ":" in address[0] else address[0]
            url = f"ws://{bound_host}:{address[1]}{ACP_PATH}"
            logger.info("Listening for ACP connections on %s", url)
            if on_listening is not None:
                on_listening(url)
        await stop.wait()
        logger.info("Stopping: closing every ACP connection")


__all__ = [
    "ACP_PATH",
    "CONNECTION_ID_HEADER",
    "Gate",
    "is_loopback_host",
    "serve_router",
    "serve_websocket",
]
