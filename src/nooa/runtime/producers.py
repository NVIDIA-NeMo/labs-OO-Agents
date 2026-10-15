# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Async producer helpers for QueueManager.spawn().

Each function returns a coroutine or async generator suitable for
``qm.spawn(producer, channel="name")``.

Usage::

    from nooa.runtime.producers import monitor, after, cron

    qm.spawn(monitor("make test"), channel="ci", buffer=100)
    qm.spawn(after(300), channel="wakeup")
    qm.spawn(cron(60), channel="ticks")
"""

import asyncio
import os
import signal
import sys
from contextlib import aclosing


async def after(delay: float) -> None:
    """One-shot timer: sleep *delay* seconds, then return None."""
    await asyncio.sleep(delay)
    return None


async def cron(interval_seconds: float):
    """Yield a tick counter at fixed intervals.

    Yields an incrementing integer every *interval_seconds*.
    """
    tick = 0
    while True:
        await asyncio.sleep(interval_seconds)
        tick += 1
        yield tick


async def tail(path: str, *, poll_interval: float = 0.5):
    """Tail a file, yielding new lines as they appear.

    Starts from the current end of file. Polls every
    *poll_interval* seconds.
    """
    fh = open(path)
    try:
        fh.seek(0, 2)
        while True:
            line = fh.readline()
            if line:
                yield line.rstrip("\n")
            else:
                await asyncio.sleep(poll_interval)
    finally:
        fh.close()


async def run_job(coro, job_id: str):
    """Wrap a coroutine result with a job_id tag.

    Returns ``{"job_id": job_id, "result": <awaited value>}``.
    """
    result = await coro
    return {"job_id": job_id, "result": result}


async def monitor(cmd: str):
    """Stream stdout/stderr lines from a shell command as they appear.

    Uses PowerShell on Windows and the system shell on Unix.
    Output is decoded as UTF-8 with undecodable bytes replaced, so a
    command that emits non-UTF-8 output does not terminate the stream.
    Each monitor owns an isolated process tree, cleaned up on cancellation
    or early ``aclose()``. Unix uses process groups; Windows uses Job Objects.

    Windows removes CRLF line endings and merges complete lines from both
    output pipes; concurrent writes have no guaranteed cross-stream order.
    """
    if sys.platform == "win32":
        async with aclosing(_monitor_windows(cmd)) as stream:
            async for line in stream:
                yield line
        return

    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    assert proc.stdout is not None
    try:
        async for line in proc.stdout:
            yield line.decode("utf-8", errors="replace").rstrip("\n")
        await proc.wait()
    finally:
        if proc.returncode is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            await proc.wait()


async def _monitor_windows(cmd: str):
    from nooa.tools._shell_session import create_shell_session

    session = create_shell_session()
    pending = {"stdout": "", "stderr": ""}
    try:
        # A monitor can run indefinitely. Closing the stream first releases
        # the session lock and interrupts unfinished work before close().
        async with aclosing(session.run_stream(cmd, timeout=float("inf"))) as stream:
            async for name, chunk in stream:
                if name == "__done__":
                    break  # Exit status is control metadata, not command output.
                lines = (pending[name] + chunk).split("\n")
                pending[name] = lines.pop()
                for line in lines:
                    yield line.removesuffix("\r")
        for remaining in pending.values():
            if remaining:
                yield remaining
    finally:
        # Even a successful command may leave background descendants.
        await session.close()
