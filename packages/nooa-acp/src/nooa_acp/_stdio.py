# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep JSON-RPC on the original stdout while diagnostics go to stderr."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, redirect_stdout
from typing import BinaryIO

from acp.core import DEFAULT_STDIO_BUFFER_LIMIT_BYTES
from acp.stdio import stdio_streams


class _ProtocolOutput(asyncio.WriteTransport):
    """A pipe transport bound to one binary output stream, independent of sys.stdout."""

    def __init__(self, stream: BinaryIO, protocol: asyncio.BaseProtocol) -> None:
        super().__init__()
        self._stream = stream
        self._protocol = protocol
        self._closing = False

    def write(self, data: bytes | bytearray | memoryview) -> None:
        if self._closing:
            raise ConnectionError("ACP stdout is closed")
        self._stream.write(data)
        self._stream.flush()

    def is_closing(self) -> bool:
        return self._closing

    def close(self) -> None:
        if not self._closing:
            self._closing = True
            self._protocol.connection_lost(None)

    def abort(self) -> None:
        self.close()

    def can_write_eof(self) -> bool:
        return False

    def get_extra_info(self, name: str, default=None):
        return default


@asynccontextmanager
async def protocol_stdio() -> AsyncIterator[tuple[asyncio.StreamReader, asyncio.StreamWriter]]:
    stdout = sys.stdout.buffer
    reader, platform_writer = await stdio_streams(limit=DEFAULT_STDIO_BUFFER_LIMIT_BYTES)
    # The SDK's Windows writer resolves sys.stdout.buffer on every write, so
    # redirecting stdout alone would redirect JSON-RPC too. Keep its native
    # input reader and bind protocol output to the original binary stream.
    writer = platform_writer
    if sys.platform == "win32":
        output_reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(output_reader)
        transport = _ProtocolOutput(stdout, protocol)
        protocol.connection_made(transport)
        writer = asyncio.StreamWriter(
            transport, protocol, output_reader, asyncio.get_running_loop()
        )
    try:
        with redirect_stdout(sys.stderr):
            yield reader, writer
    finally:
        writer.close()
        platform_writer.close()
