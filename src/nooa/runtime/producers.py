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
    *poll_interval* seconds. A line that is still being written is
    held until its newline arrives, so a poll landing mid-write does
    not split one line across two yields.
    """
    fh = open(path)
    try:
        fh.seek(0, 2)
        partial = ""
        while True:
            chunk = fh.readline()
            if chunk:
                partial += chunk
                if partial.endswith("\n"):
                    yield partial.rstrip("\n")
                    partial = ""
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
    """Stream stdout lines from a shell command as they appear.

    Yields each line (stripped) as it's written to stdout.
    stderr is merged into stdout.  Uses ``start_new_session=True``
    for process-group isolation so multiple concurrent monitors
    (and the agent itself) don't contend for ptys or interfere
    with each other.  On cancellation the entire process group
    is killed to prevent orphaned children.
    """
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    assert proc.stdout is not None
    try:
        async for line in proc.stdout:
            yield line.decode().rstrip("\n")
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
