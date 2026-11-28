# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Background-job lifecycle: timeouts spare earlier jobs; close() may keep them."""

import asyncio
import gc
import os
import shlex
import signal
import subprocess
import sys
import time

import pytest

from nooa.tools._bash_session import BashSession
from nooa.tools.shell_lifecycle import preserve_background_services


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as f:
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


async def _wait_group_gone(pgid: int, timeout: float = 5.0) -> bool:
    """Wait until no process (zombies aside) is left in the process group."""

    def members() -> list[int]:
        pids = []
        for pid in filter(str.isdigit, os.listdir("/proc")):
            try:
                with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as f:
                    # After "pid (comm)": state, ppid, pgrp, ...
                    state, _, pgrp = f.read().rsplit(")", 1)[1].split()[:3]
            except (FileNotFoundError, ProcessLookupError):
                continue
            if int(pgrp) == pgid and state != "Z":
                pids.append(int(pid))
        return pids

    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if not members():
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


@pytest.fixture
def collected():
    """Collect what earlier tests left, before this test has a session to drop.

    An asyncio transport of a closed loop raises in its ``__del__``, and pytest
    keeps that exception: raised while a session runs, its frames would hold it.
    """
    gc.collect()


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


async def test_drop_ends_the_process_group_by_default(tmp_path, reap, collected):
    session = BashSession(cwd=tmp_path)
    pid = await _start_job(session, "sleep 60")
    reap.append(pid)
    bash_pid = session._process.pid
    assert _alive(bash_pid) and _alive(pid)
    # bash leads its own group, which the job shares.
    assert os.getpgid(bash_pid) == os.getpgid(pid) == bash_pid

    del session
    gc.collect()

    assert await _wait_group_gone(bash_pid)


async def test_drop_can_keep_background_jobs(tmp_path, reap, collected):
    session = BashSession(cwd=tmp_path, keep_background_on_close=True)
    pid = await _start_job(session, "sleep 60")
    reap.append(pid)
    bash_pid = session._process.pid
    assert _alive(bash_pid) and _alive(pid)

    del session
    gc.collect()

    assert await _wait_dead(bash_pid)
    assert not await _wait_dead(pid, timeout=0.5)


async def test_drop_after_reset_ends_the_new_process_group(tmp_path, reap, collected):
    """reset() replaces bash: the drop must end the bash that is running now."""
    session = BashSession(cwd=tmp_path)
    await session.run("true")
    first = session._process.pid
    await session.reset()
    pid = await _start_job(session, "sleep 60")
    reap.append(pid)
    second = session._process.pid
    assert second != first
    assert await _wait_group_gone(first)
    assert _alive(second) and _alive(pid)

    del session
    gc.collect()

    assert await _wait_group_gone(second)


async def test_drop_under_preserve_scope_keeps_services(tmp_path, reap):
    """The scope holds a dropped session: its exit ends bash, and the service survives."""
    async with preserve_background_services():
        session = BashSession(cwd=tmp_path)  # the scope, not the session, sets the policy
        pid = await _start_job(session, "sleep 60")
        reap.append(pid)
        bash_pid = session._process.pid

        del session
        gc.collect()

        assert not await _wait_dead(bash_pid, timeout=0.5)
        assert _alive(pid)

    assert await _wait_dead(bash_pid)
    assert not await _wait_dead(pid, timeout=0.5)


async def test_close_detaches_the_finalizer(tmp_path):
    session = BashSession(cwd=tmp_path)
    await session.run("true")
    finalizer = session._process_owner._finalizer
    assert finalizer.alive

    await session.close()

    assert not finalizer.alive
    assert session._process_owner.process is None


async def test_attach_detaches_an_earlier_finalizer():
    from unittest.mock import Mock

    from nooa.tools._bash_process import _BashProcess

    class Session:
        pass

    owner, session = _BashProcess(keep_background=False), Session()
    owner.attach(session, Mock(returncode=None))
    first = owner._finalizer
    owner.attach(session, Mock(returncode=None))

    assert not first.alive
    assert owner._finalizer.alive
    owner._finalizer.detach()


async def test_drop_after_cancelled_close_ends_bash(tmp_path, reap, collected):
    """A cancelled close() has not ended bash yet, so the finalizer must still do so."""
    session = BashSession(cwd=tmp_path)
    # Both ignore SIGTERM, so close() is still waiting when it is cancelled.
    await session.run("trap '' TERM")
    pid = await _start_job(session, "sleep 60")
    reap.append(pid)
    bash_pid = session._process.pid
    closing = asyncio.create_task(session.close())
    await asyncio.sleep(0.2)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert _alive(bash_pid) and _alive(pid)

    del session, closing
    gc.collect()

    assert await _wait_group_gone(bash_pid)


@pytest.mark.parametrize("keep", [False, True])
def test_unclosed_session_ends_bash_at_interpreter_exit(tmp_path, keep):
    """Also with a live reference at exit; a kept job survives and can still write."""
    pids = tmp_path / "pids"
    owner = f"""
import asyncio
from nooa.tools._bash_session import BashSession

session = BashSession(cwd={str(tmp_path)!r}, keep_background_on_close={keep})
asyncio.run(session.run("(while echo tick; do sleep 0.05; done) & echo $$ $! > {pids}"))
"""
    subprocess.run([sys.executable, "-c", owner], timeout=30, check=True)
    bash_pid, job = map(int, pids.read_text().split())
    try:
        time.sleep(0.5)
        assert not _alive(bash_pid)
        assert _alive(job) is keep
    finally:
        try:
            os.kill(job, signal.SIGKILL)
        except ProcessLookupError:
            pass


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
