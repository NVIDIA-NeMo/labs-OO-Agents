# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The worker role: one root session's ACP adapter behind the router.

The router starts one worker process per root session and talks to it
over one end of a Unix socket pair (``--worker-fd N``). The worker is a
plain ACP server: P3's ``AtomACPAgent`` served by ``serve_connection``,
answering ``initialize`` itself (the router replays the client's
``initialize`` as the worker's first request). Its requests to the client
use ids from ``--id-base B`` on, so the router routes replies by id alone.

The worker exits 0 when the router closes its end of the socket, after
closing its sessions. If the router dies without closing it (SIGKILL,
with the worker's loop busy in a long cell), a watchdog thread notices the
parent change and kills the worker's process group.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from typing import Any

from nooa_atom.acp.framing import FRAME_LIMIT
from nooa_atom.acp.server import serve_connection

logger = logging.getLogger(__name__)


async def serve_worker(
    sock: socket.socket,
    *,
    id_base: int,
    agent: Any,
    observers: list[Callable[[Any], None]] | None = None,
) -> None:
    """Serve ACP for ``agent`` on ``sock`` until the router closes it, then close the agent."""
    reader, writer = await asyncio.open_unix_connection(sock=sock, limit=FRAME_LIMIT)
    try:
        await serve_connection(agent, reader, writer, id_base=id_base, observers=observers)
    finally:
        try:
            with suppress(Exception):
                await agent.close()
        finally:
            writer.close()


def run_worker(
    fd: int,
    *,
    id_base: int,
    make_agent: Callable[[], Any],
    observers: list[Callable[[Any], None]] | None = None,
    watchdog: bool = True,
) -> int:
    """The worker process's main: serve on the inherited socket ``fd``; return the exit code."""
    if watchdog:
        start_parent_watchdog()
    sock = socket.socket(fileno=fd)

    async def main() -> None:
        await serve_worker(sock, id_base=id_base, agent=make_agent(), observers=observers)

    asyncio.run(main())
    logger.info("worker %d: the router closed the connection; exiting", id_base >> 32)
    return 0


def start_parent_watchdog(
    *,
    interval: float = 0.5,
    getppid: Callable[[], int] = os.getppid,
    killpg: Callable[[int, int], None] = os.killpg,
) -> threading.Thread:
    """Kill this process group when the parent process changes (the router died).

    A worker normally exits on end of stream, but only if its event loop
    runs; a cell that blocks the loop would outlive a router killed with
    SIGKILL. The check is ``os.getppid()`` every ``interval`` seconds, which
    works on Linux and macOS. The worker leads its own process group
    (the router starts it with ``start_new_session=True``), so group 0 is
    the worker and anything it started.
    """
    parent = getppid()

    def watch() -> None:
        while getppid() == parent:
            time.sleep(interval)
        killpg(0, signal.SIGKILL)

    thread = threading.Thread(target=watch, name="nooa-worker-watchdog", daemon=True)
    thread.start()
    return thread
