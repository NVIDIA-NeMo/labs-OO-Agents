# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The owner of a BashSession's bash process: the only code that ends it."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import subprocess
import sys
import weakref

logger = logging.getLogger(__name__)

_CLOSE_GRACE = 3.0  # Seconds close() waits for bash to exit before SIGKILL

# Reads the inherited pipe fds (argv) until every writer has exited, so
# background jobs never write into a pipe without a reader after close().
# The fds stay non-blocking (shared with asyncio), hence select().
_DRAIN_SCRIPT = (
    "import os, select, sys\n"
    "fds = [int(a) for a in sys.argv[1:]]\n"
    "while fds:\n"
    "    for fd in select.select(fds, [], [])[0]:\n"
    "        try:\n"
    "            if not os.read(fd, 65536):\n"
    "                fds.remove(fd)\n"
    "        except BlockingIOError:\n"
    "            pass\n"
)


class _BashProcess:
    """Owns the bash of one session, across its restarts, and how it ends.

    ``keep_background`` is the one policy: True ends bash only and leaves its
    background jobs running; False ends the whole process group. It is read when
    bash is ended, so it can change while bash runs.

    Holds no reference to the session: the finalizer holds this object, so with one
    the session could never be collected.
    """

    def __init__(self, *, keep_background: bool) -> None:
        self.keep_background = keep_background
        self.process: asyncio.subprocess.Process | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._finalizer: weakref.finalize[..., object] | None = None

    def attach(self, session: object, process: asyncio.subprocess.Process) -> None:
        """Own a bash just started on the running loop; it ends if ``session`` is dropped."""
        if self._finalizer is not None:
            # A bash started without close() first: only the newest one is owned.
            self._finalizer.detach()
        self.process = process
        self._loop = asyncio.get_running_loop()
        self._finalizer = weakref.finalize(session, self.kill_now, process)

    def kill_now(self, process: asyncio.subprocess.Process) -> None:
        """End ``process`` without waiting: for the finalizer, and from another loop."""
        if process.returncode is not None:
            return
        if self.keep_background:
            _spawn_output_drainer(process)
        else:
            with contextlib.suppress(OSError):
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                return
        with contextlib.suppress(Exception):
            process.kill()

    async def close(self) -> None:
        """End bash, gracefully if it was started on the running loop."""
        process, finalizer = self.process, self._finalizer
        if process is None:
            return
        if process.returncode is None:
            if self._loop is not asyncio.get_running_loop():
                # Cross-loop (gl-212): transport is dead, just kill immediately.
                self.kill_now(process)
            elif self.keep_background:
                await self._close_keeping_background(process)
            else:
                await self._close_group(process)
        # Only now: if this close() is cancelled above, a drop still ends bash.
        if finalizer is not None:
            finalizer.detach()
        if self.process is process:  # not restarted meanwhile
            self.process = self._loop = self._finalizer = None

    async def _close_group(self, process: asyncio.subprocess.Process) -> None:
        """SIGTERM the process group, wait, then SIGKILL it on timeout."""
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except OSError:
            with contextlib.suppress(Exception):
                process.kill()
        try:
            await asyncio.wait_for(process.wait(), timeout=_CLOSE_GRACE)
        except TimeoutError:
            with contextlib.suppress(OSError):
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)

    async def _close_keeping_background(self, process: asyncio.subprocess.Process) -> None:
        """End bash only; its background jobs keep running with drained output."""
        _spawn_output_drainer(process)
        if process.stdin is not None:
            # A non-interactive bash exits at EOF without signalling its jobs.
            try:
                process.stdin.close()
                await asyncio.wait_for(process.wait(), timeout=_CLOSE_GRACE)
                return
            except (TimeoutError, OSError):
                pass
        with contextlib.suppress(OSError):
            os.kill(process.pid, signal.SIGKILL)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(process.wait(), timeout=_CLOSE_GRACE)


def _spawn_output_drainer(proc: asyncio.subprocess.Process) -> None:
    """Keep reading bash's stdout/stderr pipes in a detached process.

    Background jobs inherit these pipes; once this process exits they would get
    EPIPE/SIGPIPE on their next write. Best-effort: failures are only logged.
    """
    try:
        transport = proc._transport  # type: ignore[attr-defined]
        fds = [transport.get_pipe_transport(fd).get_extra_info("pipe").fileno() for fd in (1, 2)]
        subprocess.Popen(
            [sys.executable, "-c", _DRAIN_SCRIPT, *map(str, fds)],
            pass_fds=fds,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        logger.warning("Could not start the background-output drainer", exc_info=True)
