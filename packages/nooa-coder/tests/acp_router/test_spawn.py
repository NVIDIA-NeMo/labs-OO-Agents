# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The router's real spawn: a worker process on a socket pair, in its own process group."""

import asyncio
import os
import sys
from pathlib import Path

from acp import PROTOCOL_VERSION
from nooa_coder.acp.framing import encode, read_frame
from nooa_coder.acp.router import process_spawn

FAKE_AGENT = Path(__file__).parent / "fixtures" / "fake_agent.py"
TIMEOUT = 60


async def test_a_spawned_worker_answers_leads_its_group_and_exits_zero_on_end_of_stream(
    tmp_path,
):
    spawn = process_spawn(
        [
            sys.executable,
            str(FAKE_AGENT),
            "--model",
            "fake",
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
        assert frame.message["result"]["agentInfo"]["name"] == "nooa-coder"
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
