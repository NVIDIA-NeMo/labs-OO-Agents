# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for producers.monitor() process isolation."""

import asyncio
import base64
import json
import math
import os
import platform
import shlex
import shutil
import sys
from contextlib import aclosing
from pathlib import Path

import pytest

from nooa.runtime.producers import monitor


def _quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _command(*args: str | Path) -> str:
    if sys.platform == "win32":
        return "& " + " ".join(_quote(arg) for arg in args)
    return shlex.join(str(arg) for arg in args)


def _python(code: str) -> str:
    # A base64 payload avoids PowerShell 5.1's legacy native argument quoting.
    blob = base64.b64encode(code.encode("utf-8")).decode("ascii")
    return _command(sys.executable, "-c", f"import base64;exec(base64.b64decode({blob!r}))")


def _sleep(seconds: float) -> str:
    if sys.platform == "win32":
        return f"Start-Sleep -Milliseconds {int(seconds * 1000)}"
    return f"sleep {seconds}"


class TestMonitorProcessIsolation:
    """Verify monitor() uses pipes with proper process-group isolation."""

    async def test_monitor_streams_stdout(self):
        """Basic: monitor yields stdout lines."""
        lines = []
        async for line in monitor("echo hello; echo world"):
            lines.append(line)
        assert lines == ["hello", "world"]

    @pytest.mark.skipif(sys.platform == "win32", reason="Unix process-group contract")
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
            collect(monitor(f"echo A1; {_sleep(0.1)}; echo A2"), lines_a),
            collect(monitor(f"echo B1; {_sleep(0.1)}; echo B2"), lines_b),
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
            async for line in monitor(f"echo started; {_sleep(999)}"):
                if "started" in line:
                    victim_started.set()

        async def survivor():
            async for line in monitor(f"{_sleep(0.3)}; echo survived"):
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


class TestMonitorOutputDecoding:
    """Verify monitor() tolerates command output that is not valid UTF-8."""

    async def test_undecodable_byte_does_not_end_the_stream(self, tmp_path):
        """A non-UTF-8 byte must not kill the producer or drop later lines."""
        emitter = tmp_path / "emit.py"
        emitter.write_text(
            "import sys\n"
            "out = sys.stdout.buffer\n"
            "out.write(b'first\\n')\n"
            "out.write(b'caf\\xe9\\n')\n"  # latin-1 'e-acute': invalid UTF-8
            "out.write(b'last\\n')\n"
            "out.flush()\n",
            encoding="utf-8",
        )

        lines = [line async for line in monitor(_command(sys.executable, emitter))]

        # The undecodable byte becomes U+FFFD rather than being dropped, and
        # "last" proves the stream continued past it.
        assert lines == ["first", "caf\ufffd", "last"]


@pytest.fixture(params=["pwsh", "powershell"])
def windows_shell(request, monkeypatch):
    from nooa.tools._powershell_session import PowerShellSession

    executable = shutil.which(request.param)
    if executable is None:
        pytest.skip(f"{request.param} is not installed")
    monkeypatch.setattr(
        "nooa.tools._shell_session.create_shell_session",
        lambda: PowerShellSession(executable=executable),
    )


class _ProcessHandle:
    """Keep process identity stable even if Windows reuses a terminated PID."""

    def __init__(self, pid: int):
        import _winapi

        self._api = _winapi
        self._handle = _winapi.OpenProcess(0x100000 | 0x0001, False, pid)

    @property
    def alive(self) -> bool:
        return self._api.WaitForSingleObject(self._handle, 0) == 258

    async def wait_dead(self):
        async with asyncio.timeout(5):
            while self.alive:
                await asyncio.sleep(0.02)

    def close(self):
        try:
            if self.alive:
                self._api.TerminateProcess(self._handle, 1)
        finally:
            self._api.CloseHandle(self._handle)


@pytest.mark.skipif(sys.platform != "win32", reason="native Windows process tests")
@pytest.mark.usefixtures("windows_shell")
class TestWindowsMonitor:
    async def test_unicode_lines_arrive_before_command_completion(self, tmp_path):
        release = tmp_path / "release"
        command = _python(
            "import sys,time,pathlib\n"
            "out=sys.stdout.buffer\n"
            "out.write('  first 中文  \\r\\n'.encode());out.flush()\n"
            f"while not pathlib.Path({str(release)!r}).exists(): time.sleep(.01)\n"
            "out.write(b'\\nlast');out.flush()\n"
        )
        async with aclosing(monitor(command)) as stream:
            assert await asyncio.wait_for(anext(stream), 10) == "  first 中文  "
            assert not release.exists()
            release.touch()
            assert [line async for line in stream] == ["", "last"]

    async def test_merges_stderr_without_combining_unfinished_lines(self):
        command = (
            "[Console]::Out.Write('partial ');"
            "[Console]::Error.WriteLine('problem 中文');"
            "[Console]::Out.WriteLine('done');"
            "[Console]::Error.Write('trailing')"
        )
        lines = [line async for line in monitor(command)]
        assert sorted(lines) == ["partial done", "problem 中文", "trailing"]

    async def test_nonzero_exit_ends_stream_without_control_metadata(self):
        lines = [line async for line in monitor(_python("print('finished');raise SystemExit(7)"))]
        assert lines == ["finished"]

    @pytest.mark.parametrize("cancel", [False, True], ids=["aclose", "cancel"])
    async def test_close_or_cancel_reaps_shell_parent_and_grandchild(self, cancel):
        command = "[Console]::WriteLine($PID);" + _python(
            "import subprocess,sys,os,json,time;"
            "child=subprocess.Popen([sys._base_executable,'-c','import time;time.sleep(60)'],"
            "creationflags=subprocess.CREATE_NO_WINDOW);"
            "print(json.dumps([os.getpid(),child.pid]),flush=True);time.sleep(60)"
        )
        handles = []
        pending = None
        stream = monitor(command)
        try:
            handles.append(_ProcessHandle(int(await asyncio.wait_for(anext(stream), 10))))
            pids = json.loads(await asyncio.wait_for(anext(stream), 10))
            handles.extend(_ProcessHandle(pid) for pid in pids)
            assert all(handle.alive for handle in handles)
            if cancel:
                pending = asyncio.create_task(anext(stream))
                await asyncio.sleep(0)
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                await stream.aclose()
            await asyncio.gather(*(handle.wait_dead() for handle in handles))
        finally:
            if pending is not None:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await stream.aclose()
            for handle in handles:
                handle.close()

    async def test_normal_completion_reaps_background_descendants(self, tmp_path):
        script = tmp_path / "background.py"
        identities = tmp_path / "pids.json"
        release = tmp_path / "release"
        script.write_text(
            "import subprocess,sys,os,pathlib,json,time\n"
            "child=subprocess.Popen([sys._base_executable,'-c','import time;time.sleep(60)'],"
            "creationflags=subprocess.CREATE_NO_WINDOW)\n"
            f"identities=pathlib.Path({str(identities)!r})\n"
            "pending=identities.with_suffix('.tmp')\n"
            "pending.write_text(json.dumps([os.getpid(),child.pid]))\n"
            "pending.replace(identities)\n"
            "time.sleep(60)\n",
            encoding="utf-8",
        )
        command = (
            "$start = [Diagnostics.ProcessStartInfo]::new();"
            f"$start.FileName = {_quote(sys.executable)};"
            f"$start.Arguments = {_quote(chr(34) + str(script) + chr(34))};"
            "$start.UseShellExecute = $false; $start.CreateNoWindow = $true;"
            "[void][Diagnostics.Process]::Start($start);"
            f"while (-not [IO.File]::Exists({_quote(identities)})) {{ Start-Sleep -Milliseconds 10 }};"
            f"[Console]::WriteLine([IO.File]::ReadAllText({_quote(identities)}));"
            f"while (-not [IO.File]::Exists({_quote(release)})) {{ Start-Sleep -Milliseconds 10 }}"
        )
        handles = []
        try:
            async with aclosing(monitor(command)) as stream:
                pids = json.loads(await asyncio.wait_for(anext(stream), 10))
                handles.extend(_ProcessHandle(pid) for pid in pids)
                assert all(handle.alive for handle in handles)
                release.touch()
                assert [line async for line in stream] == []
            await asyncio.gather(*(handle.wait_dead() for handle in handles))
        finally:
            for handle in handles:
                handle.close()

    async def test_closing_one_monitor_leaves_another_running(self, tmp_path):
        release = tmp_path / "release"
        survivor_command = _python(
            "import pathlib,time\n"
            "print('ready',flush=True)\n"
            f"while not pathlib.Path({str(release)!r}).exists(): time.sleep(.01)\n"
            "print('survived',flush=True)\n"
        )
        victim_command = "[Console]::WriteLine('ready'); Start-Sleep -Seconds 60"
        async with aclosing(monitor(victim_command)) as victim:
            async with aclosing(monitor(survivor_command)) as survivor:
                assert await asyncio.wait_for(
                    asyncio.gather(anext(victim), anext(survivor)), 10
                ) == ["ready", "ready"]
                await victim.aclose()
                release.touch()
                assert [line async for line in survivor] == ["survived"]


async def test_windows_adapter_has_no_deadline_and_closes_its_session(monkeypatch):
    from nooa.runtime.producers import _monitor_windows

    class Session:
        closed = False

        async def run_stream(self, command, timeout=30):
            assert command == "test command"
            assert timeout is None or math.isinf(timeout), "monitor must not inherit a 30s deadline"
            yield "stdout", "first\r"
            yield "stderr", "problem\r\n"
            yield "stdout", "\n\ntail"
            yield "__done__", "7,0"

        async def close(self):
            self.closed = True

    session = Session()
    monkeypatch.setattr("nooa.tools._shell_session.create_shell_session", lambda: session)
    assert [line async for line in _monitor_windows("test command")] == [
        "problem",
        "first",
        "",
        "tail",
    ]
    assert session.closed
