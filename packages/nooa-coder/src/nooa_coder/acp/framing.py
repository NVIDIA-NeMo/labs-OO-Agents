# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JSON-RPC line framing for the ACP router.

ACP frames are one JSON object per line (the ``acp`` library writes
``json.dumps(message, separators=(",", ":")) + "\\n"``). The router is a
line proxy: it reads each frame, parses only the envelope (``id``,
``method``, ``params.sessionId``) and forwards the original bytes, so a
frame reaches the other side exactly as it was written.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

FRAME_LIMIT = 50 * 1024 * 1024
"""Longest frame accepted, as the library's ``DEFAULT_STDIO_BUFFER_LIMIT_BYTES``.

Readers passed to :func:`read_frame` should be created with this limit;
transcript replays and file reads are far over asyncio's 64 KiB default.
"""


@dataclass(frozen=True, slots=True)
class Frame:
    """One JSON-RPC message: its original line (ending in a newline) and the parsed object."""

    raw: bytes
    message: dict[str, Any]

    @property
    def has_id(self) -> bool:
        return "id" in self.message

    @property
    def id(self) -> Any:
        return self.message.get("id")

    @property
    def method(self) -> str | None:
        method = self.message.get("method")
        return method if isinstance(method, str) else None

    @property
    def params(self) -> dict[str, Any]:
        params = self.message.get("params")
        return params if isinstance(params, dict) else {}

    @property
    def session_id(self) -> str | None:
        session_id = self.params.get("sessionId")
        return session_id if isinstance(session_id, str) else None

    @property
    def is_request(self) -> bool:
        return self.method is not None and self.has_id

    @property
    def is_notification(self) -> bool:
        return self.method is not None and not self.has_id

    @property
    def is_response(self) -> bool:
        return self.method is None and self.has_id


def encode(message: dict[str, Any]) -> bytes:
    """One frame, serialised exactly as the ``acp`` library serialises its own."""
    return (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")


async def read_frame(reader: asyncio.StreamReader) -> Frame | None:
    """The next frame from ``reader``, or ``None`` at end of stream.

    Blank lines are skipped. A line that is not JSON, or is JSON but not an
    object (a batch, a bare value), is dropped with a log line, as the
    library does. A line longer than the reader's limit is read to its end,
    dropped and logged; reading continues with the next line.
    """
    while True:
        try:
            line = await reader.readuntil(b"\n")
        except asyncio.IncompleteReadError as exc:
            line = exc.partial
            if not line.strip():
                return None
            line += b"\n"
        except asyncio.LimitOverrunError as exc:
            await _discard_long_line(reader, exc.consumed)
            continue
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except ValueError:
            logger.warning("Dropped a frame that is not JSON (%d bytes)", len(line))
            continue
        if not isinstance(message, dict):
            kind = "batch" if isinstance(message, list) else type(message).__name__
            logger.warning("Dropped a JSON-RPC %s frame: only single objects are routed", kind)
            continue
        return Frame(line, message)


async def _discard_long_line(reader: asyncio.StreamReader, consumed: int) -> None:
    dropped = 0
    while True:
        chunk = await reader.readexactly(consumed) if consumed else b""
        dropped += len(chunk)
        if chunk.endswith(b"\n"):
            break
        try:
            dropped += len(await reader.readuntil(b"\n"))
            break
        except asyncio.LimitOverrunError as exc:
            consumed = exc.consumed
        except asyncio.IncompleteReadError as exc:
            dropped += len(exc.partial)
            break
    logger.warning("Dropped a frame over the reader's size limit (%d bytes)", dropped)
