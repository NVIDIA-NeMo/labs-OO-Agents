# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for producers.monitor() process isolation."""

import asyncio
import os
import platform

import pytest

from nooa.runtime.producers import monitor, tail


class TestTail:
    """Verify tail() assembles whole lines."""

    async def test_partial_line_is_not_split(self, tmp_path):
        """A poll landing mid-write must not split one line across two yields."""
        f = tmp_path / "app.log"
        f.write_text("", encoding="utf-8")

        async def write_in_two_parts():
            await asyncio.sleep(0.05)
            with open(f, "a", encoding="utf-8") as fh:
                fh.write("ERROR db timeout")
                fh.flush()
                await asyncio.sleep(0.3)  # tail polls repeatedly during this gap
                fh.write(" after 30s\n")

        async def first_line():
            async for line in tail(str(f), poll_interval=0.01):
                return line
            return None

        writer = asyncio.create_task(write_in_two_parts())
        line = await asyncio.wait_for(first_line(), timeout=5.0)
        await writer
        assert line == "ERROR db timeout after 30s"


class TestMonitorProcessIsolation:
    """Verify monitor() uses pipes with proper process-group isolation."""

    async def test_monitor_streams_stdout(self):
        """Basic: monitor yields stdout lines."""
        lines = []
        async for line in monitor("echo hello && echo world"):
            lines.append(line)
        assert lines == ["hello", "world"]

    async def test_monitor_own_process_group(self):
        """Spawned process must be in its own process group (start_new_session)."""
        agent_pgid = os.getpgid(os.getpid())

        # Spawn a process that prints its PID, then exits
        gen = monitor("echo $$; sleep 0.2")
        pid = None
        async for line in gen:
            line = line.strip()
            if pid is None and line.isdigit():
                pid = int(line)

        assert pid is not None, "Failed to capture subprocess PID"
        # With start_new_session=True, the process gets its own group (pgid == pid).
        # We check BEFORE the process exits by using sleep above.
        # But since the process may have exited, we verify indirectly:
        # re-run and check pgid while process is still alive.
        gen2 = monitor("echo $$; sleep 10")
        pid2 = None
        async for line in gen2:
            line = line.strip()
            if pid2 is None and line.isdigit():
                pid2 = int(line)
                break

        assert pid2 is not None
        try:
            pgid = os.getpgid(pid2)
            assert pgid != agent_pgid, (
                "monitor() subprocess shares agent's process group — "
                "start_new_session=True is missing"
            )
        finally:
            await gen2.aclose()
            await asyncio.sleep(0.1)

    async def test_monitor_concurrent_no_interference(self):
        """Two concurrent monitor() calls must not interfere with each other."""
        lines_a: list[str] = []
        lines_b: list[str] = []

        async def collect(gen, dest):
            async for line in gen:
                dest.append(line)

        await asyncio.gather(
            collect(monitor("echo A1 && sleep 0.1 && echo A2"), lines_a),
            collect(monitor("echo B1 && sleep 0.1 && echo B2"), lines_b),
        )

        assert lines_a == ["A1", "A2"]
        assert lines_b == ["B1", "B2"]

    @pytest.mark.skipif(
        platform.system() != "Linux",
        reason="process-group kill semantics differ on non-Linux; verified in Linux CI",
    )
    async def test_monitor_cancel_kills_process_group(self):
        """Cancelling monitor must kill the entire process group, not just the shell."""
        gen = monitor("bash -c 'echo $BASHPID; sleep 999 & echo ready; wait'")
        shell_pid = None

        async for line in gen:
            line = line.strip()
            if shell_pid is None and line.isdigit():
                shell_pid = int(line)
            if line == "ready":
                break

        assert shell_pid is not None, "Failed to capture shell PID"

        # Cancel/close the generator — should kill the process group
        await gen.aclose()
        # Give the OS a moment to reap
        await asyncio.sleep(0.5)

        # The shell and its children should be dead
        try:
            os.kill(shell_pid, 0)
            pytest.fail(f"Process {shell_pid} still alive after cancel — cleanup is broken")
        except ProcessLookupError:
            pass  # expected: process is dead

    async def test_cancel_one_doesnt_affect_other(self):
        """Cancel one monitor, the other keeps running."""
        survivor_lines: list[str] = []
        victim_started = asyncio.Event()

        async def victim():
            async for line in monitor("echo started; sleep 999"):
                if "started" in line:
                    victim_started.set()

        async def survivor():
            async for line in monitor("sleep 0.3 && echo survived"):
                survivor_lines.append(line)

        victim_task = asyncio.create_task(victim())
        survivor_task = asyncio.create_task(survivor())

        await asyncio.wait_for(victim_started.wait(), timeout=5)
        victim_task.cancel()
        try:
            await victim_task
        except asyncio.CancelledError:
            pass

        await asyncio.wait_for(survivor_task, timeout=5)
        assert "survived" in survivor_lines
