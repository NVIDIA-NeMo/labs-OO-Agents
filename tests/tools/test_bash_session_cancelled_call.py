# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A call cancelled mid-command (e.g. by a python_cell timeout) must not break the session.

Before the fix the cancelled call's output readers kept consuming bash's pipes, so every later
command returned empty stdout/stderr, and the next command first waited behind the abandoned one.
Now the next call stops the abandoned command as a timeout would, in the same bash.
"""

import asyncio
import os
import shlex
import signal
import sys
import time
from pathlib import Path

import pytest

import nooa.tools._bash_session as bash_session
from nooa.tools._bash_session import BashSession


def _alive(pid: int) -> bool:
    try:
        return "zombie" not in Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return False


async def _wait_dead(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        await asyncio.sleep(0.05)
    return False


async def _abandon(session: BashSession, command: str, ready: Path | None = None) -> None:
    """Cancel a call while ``command`` runs (after it writes ``ready``), as a cell timeout does."""
    call = asyncio.create_task(session.run(command, timeout=60))
    for _ in range(250):
        await asyncio.sleep(0.02)
        if ready is None or (ready.exists() and ready.read_text().strip()):
            break
    await asyncio.sleep(0.2)
    assert not call.done(), "the command finished before the cancel"
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call


@pytest.fixture
def reap():
    pids: list[int] = []
    yield pids
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


async def test_commands_after_a_cancelled_call_return_their_output(tmp_path):
    s = BashSession(cwd=tmp_path)
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(s.run("sleep 30", timeout=60), timeout=0.5)
        t0 = time.monotonic()
        out, err, code = await s.run("echo hello; echo oops >&2", timeout=10)
        assert (out, err, code) == ("hello", "oops", 0)
        assert time.monotonic() - t0 < 2  # did not wait behind `sleep 30` or restart bash
        out, _, code = await s.run("pwd")
        assert (out, code) == (str(tmp_path.resolve()), 0)
    finally:
        await s.close()


@pytest.mark.parametrize("keep_background", [False, True])
async def test_recovery_keeps_the_shell_and_stops_only_the_abandoned_command(
    tmp_path, reap, keep_background
):
    """Variables, functions and earlier background jobs survive; the abandoned command does not.

    With keep_background_on_close too: restarting bash there took 3 s or more and left the
    abandoned command running.
    """
    s = BashSession(cwd=tmp_path, keep_background_on_close=keep_background)
    pid_file = tmp_path / "abandoned.pid"
    try:
        out, _, _ = await s.run("export FOO=bar; f() { echo fn-ok; }; sleep 60 & echo $!")
        job = int(out.split()[-1])
        reap.append(job)
        bash = s._process.pid
        await _abandon(s, f"sh -c 'echo $$ > {pid_file}; exec sleep 30'", ready=pid_file)
        abandoned = int(pid_file.read_text())
        reap.append(abandoned)

        t0 = time.monotonic()
        out, err, code = await s.run('echo "$FOO"; f; echo $$', timeout=10)

        assert (out.splitlines(), err, code) == (["bar", "fn-ok", str(bash)], "", 0)
        assert time.monotonic() - t0 < 2
        assert await _wait_dead(abandoned, 2)
        assert _alive(job)
        # Then end the job: a kept one would make close() wait for its pipes.
        os.kill(job, signal.SIGKILL)
        assert await _wait_dead(job)
    finally:
        await s.close()


async def test_cancel_while_the_script_is_written(tmp_path):
    """A cancel in drain() (e.g. a large script bash has not read yet) is not missed."""
    s = BashSession(cwd=tmp_path)
    try:
        await s.run("true")
        writer = s._process.stdin
        drain = writer.drain

        async def blocked_drain() -> None:
            await asyncio.Event().wait()

        writer.drain = blocked_drain
        try:
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(s.run("sleep 3; echo old", timeout=60), timeout=0.3)
        finally:
            writer.drain = drain
        # Before: the old command's output and exit code (0) came back for this one.
        out, err, code = await s.run("echo new; (exit 7)", timeout=10)
        assert (out, err, code) == ("new", "", 7)
    finally:
        await s.close()


async def test_a_command_that_finished_meanwhile_keeps_its_effects_and_loses_its_output(tmp_path):
    s = BashSession(cwd=tmp_path)
    try:
        await _abandon(s, "sleep 0.5 && echo late && echo late-err >&2 && export DONE=1")
        await asyncio.sleep(1.0)  # it ends on its own: there is nothing to stop
        out, err, code = await s.run('echo "next $DONE"', timeout=10)
        assert (out, err, code) == ("next 1", "", 0)
    finally:
        await s.close()


async def test_a_cancelled_recovery_is_taken_up_by_the_next_call(tmp_path, reap, monkeypatch):
    """The call that stops the abandoned command can be cancelled too, and leaks nothing."""
    monkeypatch.setattr(bash_session, "_SIGTERM_GRACE", 1.0)
    s = BashSession(cwd=tmp_path)
    pid_file = tmp_path / "abandoned.pid"
    ignores_term = (
        "import os, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(60)"
    )
    try:
        await s.run("export KEEP=yes")
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote(ignores_term)}"
        await _abandon(s, command, ready=pid_file)
        abandoned = int(pid_file.read_text())
        reap.append(abandoned)
        with pytest.raises(TimeoutError):  # cancelled within the SIGTERM grace
            await asyncio.wait_for(s.run("echo second", timeout=10), timeout=0.5)

        out, err, code = await s.run('echo "third $KEEP"', timeout=10)

        assert (out, err, code) == ("third yes", "", 0)
        assert await _wait_dead(abandoned, 2)
    finally:
        await s.close()


async def test_snapshot_cwd_env_leaves_the_abandoned_command_to_the_next_call(tmp_path, reap):
    s = BashSession(cwd=tmp_path)
    pid_file = tmp_path / "abandoned.pid"
    try:
        await s.run("export FOO=bar")
        await _abandon(s, f"sh -c 'echo $$ > {pid_file}; exec sleep 30'", ready=pid_file)
        abandoned = int(pid_file.read_text())
        reap.append(abandoned)

        assert await s.snapshot_cwd_env() is None  # at once, without stopping it
        assert _alive(abandoned)

        out, _, code = await s.run("echo ok", timeout=10)
        assert (out, code) == ("ok", 0)
        assert await _wait_dead(abandoned, 2)
        snapshot = await s.snapshot_cwd_env()
        assert snapshot is not None and snapshot[1]["FOO"] == "bar"
    finally:
        await s.close()


async def test_run_stream_after_a_cancelled_call(tmp_path):
    s = BashSession(cwd=tmp_path)

    async def consume(command):
        return [item async for item in s.run_stream(command, timeout=60)]

    try:
        await s.run("export FOO=bar")
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(consume("sleep 30"), timeout=0.5)
        items = await asyncio.wait_for(consume('echo "streamed $FOO"'), timeout=10)
        assert ("stdout", "streamed bar\n") in items and items[-1] == ("__done__", "0,0")
    finally:
        await s.close()
