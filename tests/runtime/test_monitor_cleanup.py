# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Monitor cleanup must outlive its process-group leader, not its consumer."""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
import sys
from unittest.mock import AsyncMock, Mock

import pytest

from nooa.runtime.channels import QueueManager
from nooa.runtime.producers import monitor

pytestmark = pytest.mark.skipif(os.name != "posix", reason="monitor() requires POSIX")

_CHILD = """
import os
import sys
import time

with open(sys.argv[1], "a", buffering=1) as heartbeat:
    heartbeat.write("ready\\n")
    print(os.getpid(), flush=True)
    for _ in range(1500):
        heartbeat.write("tick\\n")
        time.sleep(0.01)
"""
_PARENT = """
import subprocess
import sys

child = subprocess.Popen([sys.executable, "-c", sys.argv[1], sys.argv[2]])
if sys.argv[3] == "wait":
    child.wait()
"""


@pytest.mark.parametrize("leader_exited", [False, True])
@pytest.mark.parametrize("stop", ["job-cancel", "consumer-cancel", "aclose"])
async def test_monitor_stops_descendants(tmp_path, monkeypatch, leader_exited, stop):
    """Real children must stop writing after cancellation or explicit close."""
    heartbeat = tmp_path / "heartbeat.txt"
    command = "exec " + shlex.join(
        [sys.executable, "-c", _PARENT, _CHILD, str(heartbeat), "exit" if leader_exited else "wait"]
    )
    processes = []
    create_subprocess = asyncio.create_subprocess_shell

    async def capture_process(*args, **kwargs):
        """Observe the real process solely to synchronize on leader exit."""
        proc = await create_subprocess(*args, **kwargs)
        processes.append(proc)
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_shell", capture_process)
    manager = QueueManager()
    output = manager.queue("output")
    stream = monitor(command)
    consumer = None
    try:
        if stop == "job-cancel":
            handle = manager.spawn(stream, channel="output")
            line = await asyncio.wait_for(output.get(), timeout=5)
        else:
            line = await asyncio.wait_for(anext(stream), timeout=5)
        child_pid = int(line)
        proc = processes[0]
        assert os.getpgid(child_pid) == proc.pid
        if leader_exited:
            # An early proc.wait() can wait for inherited pipe closure too.
            # Observe the exit status without waiting on the still-live child.
            async with asyncio.timeout(5):
                while proc.returncode is None:
                    await asyncio.sleep(0.01)
            assert proc.returncode == 0
        else:
            assert proc.returncode is None

        if stop == "job-cancel":
            await asyncio.wait_for(handle.cancel(), timeout=5)
            assert handle.state == "cancelled"
        elif stop == "consumer-cancel":
            consumer = asyncio.create_task(anext(stream))
            await asyncio.sleep(0)
            consumer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(consumer, timeout=5)
        else:
            await asyncio.wait_for(stream.aclose(), timeout=5)

        assert proc.stdout is not None
        assert await asyncio.wait_for(proc.stdout.read(), timeout=5) == b""
        before = heartbeat.read_bytes()
        # A heartbeat, rather than kill(pid, 0), distinguishes running children
        # from killed-but-unreaped zombies on Linux/container hosts.
        await asyncio.sleep(0.15)
        assert heartbeat.read_bytes() == before, "child kept executing after monitor cleanup"
    finally:
        # Also contain the failing regression on unpatched code or a timeout.
        for proc in processes:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await asyncio.wait_for(proc.wait(), timeout=5)
        if consumer is not None and not consumer.done():
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
        await manager.shutdown()
        await stream.aclose()


@pytest.mark.parametrize("returncode", [None, 0, 7])
@pytest.mark.parametrize("signal_error", [None, ProcessLookupError, PermissionError])
async def test_early_close_signals_group_and_reaps(monkeypatch, returncode, signal_error):
    """Signal an exited leader's group, preserving the live-leader fallback."""

    async def stdout():
        """Suspend the monitor after one public output line."""
        yield b"ready\n"

    proc = Mock(pid=12345, returncode=returncode, stdout=stdout())
    proc.wait = AsyncMock()
    killpg = Mock(side_effect=signal_error)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", AsyncMock(return_value=proc))
    monkeypatch.setattr(os, "killpg", killpg)
    stream = monitor("unused")
    assert await anext(stream) == "ready"
    await stream.aclose()
    killpg.assert_called_once_with(proc.pid, signal.SIGKILL)
    proc.wait.assert_awaited_once()
    if signal_error is not None and returncode is None:
        proc.kill.assert_called_once()
    else:
        proc.kill.assert_not_called()


@pytest.mark.parametrize("returncode", [0, 7])
@pytest.mark.parametrize("chunks", [[], [b"first\n", b"last\n"]])
async def test_normal_exhaustion_does_not_signal_finished_group(monkeypatch, returncode, chunks):
    """Successful exhaustion only waits; it must not signal an old process ID."""

    async def stdout():
        """Provide a complete, successfully decoded output stream."""
        for chunk in chunks:
            yield chunk

    proc = Mock(pid=12345, returncode=returncode, stdout=stdout())
    proc.wait = AsyncMock()
    killpg = Mock()
    monkeypatch.setattr(asyncio, "create_subprocess_shell", AsyncMock(return_value=proc))
    monkeypatch.setattr(os, "killpg", killpg)
    assert [line async for line in monitor("unused")] == [
        chunk.decode().rstrip("\n") for chunk in chunks
    ]
    proc.wait.assert_awaited_once()
    killpg.assert_not_called()
    proc.kill.assert_not_called()


@pytest.mark.parametrize("returncode", [None, 0])
async def test_read_failure_preserves_error_and_cleans_group(monkeypatch, returncode):
    """A reader failure must still clean descendants without replacing the error."""
    original = ValueError("reader failed")

    async def stdout():
        """Fail after the first output, as a subprocess stream may do."""
        yield b"ready\n"
        raise original

    proc = Mock(pid=12345, returncode=returncode, stdout=stdout())
    proc.wait = AsyncMock()
    killpg = Mock()
    monkeypatch.setattr(asyncio, "create_subprocess_shell", AsyncMock(return_value=proc))
    monkeypatch.setattr(os, "killpg", killpg)
    stream = monitor("unused")
    assert await anext(stream) == "ready"
    with pytest.raises(ValueError) as caught:
        await anext(stream)
    assert caught.value is original
    killpg.assert_called_once_with(proc.pid, signal.SIGKILL)
    proc.wait.assert_awaited_once()


async def test_fallback_tolerates_leader_exiting_before_direct_kill(monkeypatch):
    """A process disappearing during fallback must not replace early close."""

    async def stdout():
        """Leave one buffered output before the explicit close."""
        yield b"ready\n"

    proc = Mock(pid=12345, returncode=None, stdout=stdout())
    proc.wait = AsyncMock()
    proc.kill.side_effect = ProcessLookupError
    killpg = Mock(side_effect=ProcessLookupError)
    monkeypatch.setattr(asyncio, "create_subprocess_shell", AsyncMock(return_value=proc))
    monkeypatch.setattr(os, "killpg", killpg)
    stream = monitor("unused")
    assert await anext(stream) == "ready"
    await stream.aclose()
    proc.kill.assert_called_once()
    proc.wait.assert_awaited_once()
