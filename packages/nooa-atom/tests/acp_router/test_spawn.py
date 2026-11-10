# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The router's real spawn: a worker process on a socket pair, in its own process group."""

import asyncio
import os
import sys
from pathlib import Path

from acp import PROTOCOL_VERSION
from nooa_atom.acp.framing import encode, read_frame
from nooa_atom.acp.router import process_spawn

FAKE_AGENT = Path(__file__).parent / "fixtures" / "fake_agent.py"
TIMEOUT = 60


async def test_a_spawned_worker_answers_leads_its_group_and_exits_zero_on_end_of_stream(
    tmp_path,
):
    spawn = process_spawn(
        [
            sys.executable,
            str(FAKE_AGENT),
            "--sessions-dir",
            str(tmp_path / "sessions"),
        ]
    )
    worker = await asyncio.wait_for(spawn(1), TIMEOUT)
    try:
        assert worker.pid is not None
        assert os.getpgid(worker.pid) == worker.pid != os.getpgid(0)
        worker.writer.write(
            encode(
                {
                    "jsonrpc": "2.0",
                    "id": "init",
                    "method": "initialize",
                    "params": {"protocolVersion": PROTOCOL_VERSION},
                }
            )
        )
        await worker.writer.drain()
        frame = await asyncio.wait_for(read_frame(worker.reader), TIMEOUT)
        assert frame is not None and frame.id == "init"
        assert frame.message["result"]["agentInfo"]["name"] == "nooa-atom"
        worker.writer.write_eof()
        assert await asyncio.wait_for(worker.wait(), TIMEOUT) == 0
    finally:
        worker.kill()
        worker.writer.close()


async def test_kill_tolerates_a_group_that_is_gone(tmp_path):
    spawn = process_spawn([sys.executable, "-c", "pass"])
    worker = await asyncio.wait_for(spawn(1), TIMEOUT)
    await asyncio.wait_for(worker.wait(), TIMEOUT)
    worker.kill()
    worker.kill()
    worker.writer.close()


async def test_a_worker_whose_socket_cannot_be_opened_is_killed(monkeypatch):
    import signal

    from nooa_atom.acp import router

    started = []
    real_exec = asyncio.create_subprocess_exec

    async def recording_exec(*args, **kwargs):
        process = await real_exec(*args, **kwargs)
        started.append(process)
        return process

    async def broken_connection(*args, **kwargs):
        raise OSError("cannot open the socket")

    monkeypatch.setattr(router.asyncio, "create_subprocess_exec", recording_exec)
    monkeypatch.setattr(router.asyncio, "open_unix_connection", broken_connection)
    spawn = process_spawn([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        await asyncio.wait_for(spawn(1), TIMEOUT)
    except OSError:
        pass
    else:
        raise AssertionError("the spawn should fail")
    [process] = started
    try:
        assert await asyncio.wait_for(process.wait(), 10) == -signal.SIGKILL
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
