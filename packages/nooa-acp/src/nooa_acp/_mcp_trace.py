# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in, metadata-only trace of the client's ACP MCP handoff."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from acp.connection import StreamDirection, StreamEvent

logger = logging.getLogger(__name__)


class MCPHandoffTrace:
    """Observe raw session requests without saving MCP commands or credentials."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._enabled = True
        self._write({"event": "trace_started"})

    @classmethod
    def from_env(cls) -> MCPHandoffTrace | None:
        path = os.environ.get("NOOA_ACP_MCP_TRACE")
        return cls(Path(path).expanduser()) if path else None

    def __call__(self, event: StreamEvent) -> None:
        if not self._enabled or event.direction != StreamDirection.INCOMING:
            return
        method = event.message.get("method")
        if method not in ("session/new", "session/load"):
            return
        params = event.message.get("params")
        params = params if isinstance(params, dict) else {}
        servers = params.get("mcpServers")
        if "mcpServers" not in params:
            state = "missing"
        elif isinstance(servers, list):
            state = "list"
        else:
            state = "null" if servers is None else "invalid"
        # Explicitly select fields: URLs, commands, args, env, headers, prompts,
        # and extension metadata must never reach this diagnostic journal.
        summary = []
        for server in servers if isinstance(servers, list) else []:
            server = server if isinstance(server, dict) else {}
            name = server.get("name")
            transport = server.get("type", "stdio")
            summary.append(
                {
                    "name": name if isinstance(name, str) else None,
                    "transport": (
                        transport if transport in ("stdio", "http", "sse", "acp") else "unknown"
                    ),
                }
            )
        self._write({"event": method, "mcpServersField": state, "servers": summary})

    def _write(self, record: dict[str, Any]) -> None:
        # __call__ (an acp.connection observer) is a plain sync callback that
        # runs inline on the receive loop multiplexing every open session in
        # this process; a blocking disk write here would stall all of them
        # for its duration on a slow/contended filesystem. Offload the actual
        # I/O to a worker thread and don't wait on it -- this is opt-in,
        # best-effort diagnostic tracing, not something ACP correctness
        # depends on, so an occasional out-of-order or lost-at-exit line is
        # an acceptable trade for never blocking the loop.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._write_now(record)
            return
        loop.run_in_executor(None, self._write_now, record)

    def _write_now(self, record: dict[str, Any]) -> None:
        try:
            with self._path.open("a", encoding="utf-8") as journal:
                journal.write(json.dumps({"pid": os.getpid(), **record}) + "\n")
        except OSError as exc:
            # A diagnostic file failure must not prevent ordinary ACP use.
            self._enabled = False
            logger.warning("Cannot write ACP MCP handoff trace: %s", exc)
