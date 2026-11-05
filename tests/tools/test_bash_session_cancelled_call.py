# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A call cancelled mid-command (e.g. by a python_cell timeout) must not break the session.

Before the fix the cancelled call's output readers kept consuming bash's pipes, so every later
command returned empty stdout/stderr, and the next command first waited behind the abandoned one.
"""

import asyncio
import time

import pytest

from nooa.tools._bash_session import BashSession


async def test_commands_after_a_cancelled_call_return_their_output(tmp_path):
    s = BashSession(cwd=tmp_path)
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(s.run("sleep 30", timeout=60), timeout=0.5)
        t0 = time.monotonic()
        out, err, code = await s.run("echo hello; echo oops >&2", timeout=10)
        assert (out, err, code) == ("hello", "oops", 0)
        assert time.monotonic() - t0 < 8  # did not wait behind the abandoned `sleep 30`
        out, _, code = await s.run("pwd")
        assert (out, code) == (str(tmp_path.resolve()), 0)
    finally:
        await s.close()


async def test_kept_background_session_keeps_the_abandoned_command_running(tmp_path):
    s = BashSession(cwd=tmp_path, keep_background_on_close=True)
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(
                s.run("sh -c 'sleep 5; echo done > done.txt'", timeout=60), timeout=0.3
            )
        out, _, code = await s.run("echo next", timeout=10)
        assert (out, code) == ("next", 0)
        for _ in range(80):
            if (tmp_path / "done.txt").exists():
                break
            await asyncio.sleep(0.1)
        assert (tmp_path / "done.txt").read_text().strip() == "done"
    finally:
        await s.close()


async def test_run_stream_after_a_cancelled_call(tmp_path):
    s = BashSession(cwd=tmp_path)

    async def consume(command):
        return [item async for item in s.run_stream(command, timeout=60)]

    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(consume("sleep 30"), timeout=0.5)
        items = await asyncio.wait_for(consume("echo streamed"), timeout=10)
        assert ("stdout", "streamed\n") in items and items[-1] == ("__done__", "0,0")
    finally:
        await s.close()
