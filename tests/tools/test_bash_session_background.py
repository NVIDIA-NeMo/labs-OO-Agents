# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Background-job lifecycle: timeouts spare earlier jobs; close() may keep them."""

import asyncio
import os
import shlex
import signal
import subprocess
import sys
import time

import pytest

from nooa.tools._bash_session import BashSession


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/status") as f:
            return "zombie" not in f.read()
    except (FileNotFoundError, ProcessLookupError):
        # The process can be reaped between open() and read().
        return False


async def _wait_dead(pid: int, timeout: float = 5.0) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if not _alive(pid):
            return True
        await asyncio.sleep(0.05)
    return False


async def _start_job(session: BashSession, command: str) -> int:
    stdout, _, code = await session.run(f"{command} & echo $!")
    assert code == 0
    return int(stdout.split()[-1])


@pytest.fixture
def reap():
    pids: list[int] = []
    yield pids
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


async def test_tracks_background_job_pids(tmp_path, reap):
    session = BashSession(cwd=tmp_path)
    try:
        pid = await _start_job(session, "sleep 60")
        reap.append(pid)
        assert pid in session._background_pids
        await session.run("true")
        assert pid in session._background_pids
    finally:
        await session.close()


async def test_timeout_spares_earlier_background_jobs(tmp_path, reap):
    session = BashSession(cwd=tmp_path)
    try:
        pid = await _start_job(session, "sleep 60")
        reap.append(pid)

        _, _, code = await session.run("sleep 30", timeout=0.5)

        assert code == 124
        assert _alive(pid)
        stdout, _, code = await session.run("echo still-usable")
        assert (stdout, code) == ("still-usable", 0)
    finally:
        await session.close()


@pytest.mark.parametrize("pipeline", [False, True])
async def test_timeout_kills_nested_command_but_spares_earlier_job(
    tmp_path, reap, pipeline, monkeypatch
):
    """A TERM-resistant grandchild must not survive its parent's early exit."""
    spawn = asyncio.create_subprocess_exec

    async def without_ps(command, *args, **kwargs):
        if command == "ps":
            raise FileNotFoundError("ps is absent in the task image")
        return await spawn(command, *args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", without_ps)
    session = BashSession(cwd=tmp_path, keep_background_on_close=True)
    pid_file = tmp_path / "child.pid"
    child = (
        "import os, signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(60)"
    )
    parent = f"import subprocess, time; subprocess.Popen([{sys.executable!r}, '-c', {child!r}]); time.sleep(60)"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(parent)}"
    if pipeline:
        command += " | cat"
    try:
        earlier = await _start_job(session, "sleep 60")
        reap.append(earlier)
        running = asyncio.create_task(session.run(command, timeout=1.0))
        try:
            for _ in range(100):
                if pid_file.exists() and pid_file.read_text().strip():
                    break
                await asyncio.sleep(0.01)
            nested = int(pid_file.read_text())
            reap.append(nested)
        finally:
            _, _, code = await running
        assert code == 124
        assert await _wait_dead(nested)
        assert _alive(earlier)
        stdout, _, code = await session.run("echo recovered")
        assert (stdout, code) == ("recovered", 0)
    finally:
        await session.close()


async def test_close_kills_background_jobs_by_default(tmp_path, reap):
    session = BashSession(cwd=tmp_path)
    pid = await _start_job(session, "nohup sleep 60 >/dev/null 2>&1")
    reap.append(pid)

    await session.close()

    assert await _wait_dead(pid)


async def test_close_can_keep_background_jobs(tmp_path, reap):
    session = BashSession(cwd=tmp_path, keep_background_on_close=True)
    pid = await _start_job(session, "sleep 60")
    reap.append(pid)
    bash_pid = session._process.pid

    await session.close()

    assert await _wait_dead(bash_pid)
    assert _alive(pid)


async def test_close_keeping_jobs_returns_once_bash_exits(tmp_path, reap):
    """Kept jobs hold bash's stdout and stderr, so close() must not wait for those to close.

    asyncio's Process.wait() returns only once those pipes are closed too, so close() took 3 s
    whenever a kept job was running.
    """
    session = BashSession(cwd=tmp_path, keep_background_on_close=True)
    pid = await _start_job(session, "sleep 60")
    reap.append(pid)
    bash_pid = session._process.pid

    started = time.monotonic()
    await session.close()

    assert time.monotonic() - started < 2
    assert not _alive(bash_pid)
    assert _alive(pid)
    os.kill(pid, signal.SIGKILL)  # here, so its pipes close while this test's loop runs
    assert await _wait_dead(pid)


async def test_close_keeping_jobs_kills_a_busy_bash_without_waiting_for_its_pipes(tmp_path, reap):
    """A bash still running a command does not exit at EOF: close() kills it, then returns.

    Before, close() then waited 3 s more for the pipes, which the command still holds.
    """
    session = BashSession(cwd=tmp_path, keep_background_on_close=True)
    await session.run("true")
    pid_file = tmp_path / "busy.pid"
    # Straight to bash, as a call cancelled mid-command leaves it: busy, no sentinel awaited.
    session._process.stdin.write(f"sh -c 'echo $$ > {pid_file}; exec sleep 30'\n".encode())
    for _ in range(250):
        if pid_file.exists() and pid_file.read_text().strip():
            break
        await asyncio.sleep(0.02)
    busy = int(pid_file.read_text())
    reap.append(busy)
    bash_pid = session._process.pid

    started = time.monotonic()
    await session.close()

    assert time.monotonic() - started < 5  # 3 s for bash to exit at EOF, then SIGKILL
    assert not _alive(bash_pid)
    os.kill(busy, signal.SIGKILL)  # here, so its pipes close while this test's loop runs
    assert await _wait_dead(busy)


def test_kept_job_can_write_after_owner_exits(tmp_path):
    """Output pipes stay drained, so a kept job does not die on EPIPE/SIGPIPE."""
    marker = tmp_path / "ticks"
    pid_file = tmp_path / "job.pid"
    owner = f"""
import asyncio
from nooa.tools._bash_session import BashSession

async def main():
    s = BashSession(cwd={str(tmp_path)!r}, keep_background_on_close=True)
    await s.run(
        "(for i in $(seq 40); do echo tick; echo $i > {marker}; sleep 0.05; done) & "
        "echo $! > {pid_file}"
    )
    await s.close()

asyncio.run(main())
"""
    subprocess.run([sys.executable, "-c", owner], timeout=30, check=True)
    job_pid = int(pid_file.read_text())
    try:
        for _ in range(100):
            if not _alive(job_pid):
                break
            time.sleep(0.05)
        assert marker.read_text().strip() == "40"
    finally:
        try:
            os.kill(job_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
