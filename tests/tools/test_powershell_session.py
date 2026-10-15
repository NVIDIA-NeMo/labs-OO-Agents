# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Windows contracts, using real PowerShell hosts and real process trees."""

from __future__ import annotations

import asyncio
import base64
import gc
import json
import shutil
import sys
from contextlib import aclosing
from pathlib import Path

import pytest

from nooa.tools import PowerShellSession
from nooa.tools.shell_lifecycle import preserve_background_services

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="native Windows process tests")


def _quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _python(code: str) -> str:
    # Avoid Windows PowerShell 5.1's legacy quoting of embedded double quotes.
    blob = base64.b64encode(code.encode("utf-8")).decode("ascii")
    return (
        f"& {_quote(sys.executable)} -c {_quote(f'import base64;exec(base64.b64decode({blob!r}))')}"
    )


@pytest.fixture(params=["pwsh", "powershell"])
def executable(request):
    path = shutil.which(request.param)
    if not path:
        pytest.skip(f"{request.param} is not installed")
    return path


@pytest.fixture
async def session(tmp_path, executable):
    async with PowerShellSession(tmp_path, executable=executable) as shell:
        yield shell


async def test_initial_literal_cwd_is_used_by_native_children(tmp_path, executable):
    directory = tmp_path / "原生 repo's [literal]"
    directory.mkdir()
    async with PowerShellSession(directory, executable=executable) as shell:
        stdout, stderr, code = await shell.run(_python("import os;print(os.getcwd())"))
        assert code == 0 and stderr == ""
        assert stdout == str(directory)
        assert shell.cwd == directory


async def test_relative_temporary_cwd_with_literal_parent_is_restored(tmp_path, executable):
    parent = tmp_path / "原生 repo's [literal]"
    nested = parent / "子目录 space's [literal]"
    nested.mkdir(parents=True)
    async with PowerShellSession(tmp_path, executable=executable) as shell:
        assert (await shell.run(f"Set-Location -LiteralPath {_quote(nested)}"))[2] == 0
        assert shell.cwd == nested
        result = await shell.run(_python("import os;print(os.getcwd())"), cwd="..")
        assert result == (str(parent), "", 0)
        assert shell.cwd == nested
        assert await shell.run(_python("import os;print(os.getcwd())")) == (str(nested), "", 0)


async def test_state_unicode_paths_and_multiline_commands(session, tmp_path):
    directory = tmp_path / "中文 space's [literal]"
    directory.mkdir()
    command = f"""
$value = @'
Unicode 中文 'single' "double" $literal `tick
'@
$env:NOOA_SESSION_PROBE = 'kept'
function Get-Probe {{ $value }}
Set-Location -LiteralPath {_quote(directory)}
"""
    assert (await session.run(command))[2] == 0
    stdout, stderr, code = await session.run("Get-Probe; $env:NOOA_SESSION_PROBE")
    assert code == 0 and stderr == ""
    assert stdout.splitlines() == ["Unicode 中文 'single' \"double\" $literal `tick", "kept"]
    assert session.cwd == directory
    assert (await session.run(_python("import os;print(os.getcwd())")))[0] == str(directory)


@pytest.mark.parametrize("payload", ["", "tail without newline", "a\r\nb\n中文\x00'\"$`\n"])
async def test_native_stdin_is_byte_exact_and_then_returns_to_eof(session, payload):
    command = _python("import sys;print(sys.stdin.buffer.read().hex())")
    assert await session.run(command, stdin=payload) == (payload.encode().hex(), "", 0)
    assert await session.run(command) == ("", "", 0)
    assert await session.run("Write-Output alive") == ("alive", "", 0)


async def test_console_stdin_state_and_temporary_cwd(session, tmp_path):
    target = tmp_path / "工作 dir"
    target.mkdir()
    stdout, stderr, code = await session.run(
        "$saved = [Console]::In.ReadToEnd(); $saved; (Get-Location).ProviderPath",
        stdin="中文 $input",
        cwd=target,
    )
    assert code == 0 and stderr == ""
    assert stdout.splitlines() == ["中文 $input", str(target)]
    assert session.cwd == tmp_path
    assert (await session.run("$saved"))[0] == "中文 $input"
    assert (await session.run("$env:SCOPED = 'yes'", cwd=target))[2] == 0
    assert (await session.run("$env:SCOPED"))[0] == "yes"


async def test_missing_cwd_never_executes_payload(session, tmp_path):
    stdout, stderr, code = await session.run(
        "[IO.File]::WriteAllText('should-not-exist', 'bad')", cwd="missing"
    )
    assert code != 0 and stderr and not stdout
    assert not (tmp_path / "should-not-exist").exists()
    assert session.cwd == tmp_path


@pytest.mark.parametrize(
    "command",
    [
        "'unclosed",
        "if (",
        "throw '中文 failure'",
        "Write-Error '中文 failure'",
        "NooaMissingCommand",
    ],
)
async def test_bad_syntax_and_errors_do_not_consume_control(session, command):
    # Windows PowerShell 5.1's first unknown-command lookup can load modules
    # for several seconds. This tests error recovery, not command discovery speed.
    stdout, stderr, code, timed_out = await session.run_with_timeout_flag(command, timeout=15)
    assert code != 0 and stderr and not timed_out
    assert "\ufffd" not in stderr
    assert await session.run("Write-Output recovered") == ("recovered", "", 0)


async def test_native_exit_codes_are_not_stale(session):
    assert (await session.run(_python("import sys;sys.exit(7)")))[2] == 7
    assert (await session.run("Write-Output good"))[2] == 0
    assert (await session.run(_python("import sys;sys.exit(7)") + "; Write-Output good"))[2] == 0
    assert (await session.run("Write-Output early; return"))[0] == "early"
    assert (await session.run_with_timeout_flag(_python("import sys;sys.exit(124)")))[2:] == (
        124,
        False,
    )
    native_warning = _python("import sys;sys.stderr.write('warning');sys.exit(0)")
    assert await session.run(native_warning) == ("", "warning", 0)
    assert (await session.run(native_warning + "; Write-Error problem"))[2] != 0


@pytest.mark.parametrize("error_action", ["Continue", "Stop"])
async def test_native_stderr_preserves_exit_status_with_error_preferences(
    tmp_path, executable, error_action
):
    init_command = f"$ErrorActionPreference = '{error_action}'; Set-StrictMode -Version Latest"
    async with PowerShellSession(tmp_path, init_command, executable=executable) as shell:
        for exit_code in (0, 7):
            command = _python(
                "import sys;print('OK');print('warning 中文',file=sys.stderr);"
                f"sys.exit({exit_code})"
            )
            assert await shell.run(command) == ("OK", "warning 中文", exit_code)
        assert await shell.run("$ErrorActionPreference") == (error_action, "", 0)
        command = _python("import sys;print('warning',file=sys.stderr)")
        assert await shell.run(command + "; Write-Output after") == ("after", "warning", 0)
        stdout, stderr, code = await shell.run("Write-Error 'problem 中文'; Write-Output after")
        assert "problem 中文" in stderr
        assert stdout == ("" if error_action == "Stop" else "after")
        assert code == (1 if error_action == "Stop" else 0)


async def test_stream_is_live_and_preserves_both_streams(session, tmp_path):
    marker = tmp_path / "completed"
    command = _python(
        "import sys,time,pathlib;print('first 中文',flush=True);time.sleep(0.8);"
        "print('problem 中文',file=sys.stderr,flush=True);"
        f"pathlib.Path({str(marker)!r}).touch();print('last',flush=True);sys.exit(7)"
    )
    async with aclosing(session.run_stream(command)) as stream:
        first = await anext(stream)
        assert first[0] == "stdout" and "first 中文" in first[1]
        assert not marker.exists(), "streaming must yield before the command finishes"
        events = [first, *[event async for event in stream]]
    assert "last" in "".join(text for name, text in events if name == "stdout")
    assert "problem 中文" in "".join(text for name, text in events if name == "stderr")
    assert events[-1] == ("__done__", "7,0")


async def test_large_outputs_are_drained_and_bounded(session):
    stdout, stderr, code = await session.run(
        _python(
            "import sys;print('HEAD'+('中'*100000)+'TAIL');"
            "sys.stderr.write('ERRHEAD'+('文'*100000)+'ERRTAIL')"
        )
    )
    assert code == 0
    assert "HEAD" in stdout and "TAIL" in stdout
    assert "ERRHEAD" in stderr and "ERRTAIL" in stderr
    assert "<truncated-output>" in stdout and "<truncated-output>" in stderr
    assert len(stdout) < 31000 and len(stderr) < 31000
    assert "\ufffd" not in stdout + stderr
    assert (await session.run("Write-Output clean")) == ("clean", "", 0)


async def test_concurrent_calls_are_serialized(session):
    await session.run("$counter = 0")
    results = await asyncio.gather(
        *(session.run("$counter += 1; Start-Sleep -Milliseconds 15; $counter") for _ in range(8))
    )
    assert [int(result[0]) for result in results] == list(range(1, 9))


async def test_slow_stream_consumer_receives_every_byte(session):
    command = _python("import sys;sys.stdout.write('x'*500000);sys.stderr.write('y'*500000)")
    output = {"stdout": [], "stderr": []}
    async with aclosing(session.run_stream(command)) as stream:
        async for name, text in stream:
            if name != "__done__":
                output[name].append(text)
                await asyncio.sleep(0.02)
    assert "".join(output["stdout"]).rstrip("\r\n") == "x" * 500000
    assert "".join(output["stderr"]).rstrip("\r\n") == "y" * 500000
    assert await session.run("Write-Output clean") == ("clean", "", 0)


async def test_timeout_recovers_cwd_and_reruns_init(tmp_path, executable):
    directory = tmp_path / "persisted 中文's [literal]"
    directory.mkdir()
    shell = PowerShellSession(tmp_path, "$env:NOOA_INIT = 'ready'", executable=executable)
    try:
        await shell.run(f"Set-Location -LiteralPath {_quote(directory)}; $volatile = 5")
        assert (await shell.run_with_timeout_flag("Start-Sleep -Seconds 60", 0.15))[2:] == (
            124,
            True,
        )
        assert shell.cwd == directory
        assert (await shell.run("$env:NOOA_INIT; $volatile")) == ("ready", "", 0)
        assert await shell.run(_python("import os;print(os.getcwd())")) == (str(directory), "", 0)
    finally:
        await shell.close()


async def test_explicit_exit_is_reported_and_next_command_restarts(session):
    assert (await session.run("exit 9"))[2] == 9
    assert await session.run("Write-Output restarted") == ("restarted", "", 0)


async def test_explicit_exit_drains_output_for_a_slow_stream_consumer(session):
    command = "[Console]::Write('x' * 1000000); [Console]::Error.Write('y' * 1000000); exit 9"
    output = {"stdout": [], "stderr": []}
    done = None
    async with aclosing(session.run_stream(command)) as stream:
        async for name, text in stream:
            if name == "__done__":
                done = text
            else:
                output[name].append(text)
                await asyncio.sleep(0.02)
    assert "".join(output["stdout"]) == "x" * 1000000
    assert "".join(output["stderr"]) == "y" * 1000000
    assert done == "9,0"
    assert await session.run("Write-Output restarted") == ("restarted", "", 0)


async def test_stream_timeout_does_not_cancel_the_consumer(session):
    async with aclosing(
        session.run_stream("[Console]::WriteLine('ready'); Start-Sleep -Seconds 60", timeout=1.0)
    ) as stream:
        assert "ready" in (await anext(stream))[1]
        # Work between reads belongs to the consumer. The shell's deadline
        # must not inject CancelledError into this unrelated await.
        await asyncio.sleep(1.2)
        assert [event async for event in stream] == [("__done__", "124,1")]
    assert await session.run("Write-Output recovered") == ("recovered", "", 0)


async def _wait_file(path: Path) -> str:
    async with asyncio.timeout(8):
        while True:
            if path.exists() and (text := path.read_text(encoding="utf-8")):
                return text
            await asyncio.sleep(0.02)


class _ProcessHandle:
    """Hold identity stable while testing termination, even if Windows reuses a PID."""

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
        if self.alive:
            self._api.TerminateProcess(self._handle, 1)
        self._api.CloseHandle(self._handle)


@pytest.mark.parametrize("cancel", [False, True])
async def test_timeout_or_cancel_reaps_parent_and_grandchild(session, tmp_path, cancel):
    identities = tmp_path / "pids.json"
    command = _python(
        "import subprocess,sys,os,pathlib,json,time;"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
        f"pathlib.Path({str(identities)!r}).write_text(json.dumps([os.getpid(),child.pid]));"
        "time.sleep(60)"
    )
    task = asyncio.create_task(
        session.run_with_timeout_flag(command, timeout=1.5 if not cancel else 60)
    )
    handles = []
    try:
        handles = [_ProcessHandle(pid) for pid in json.loads(await _wait_file(identities))]
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            assert (await task)[2:] == (124, True)
        for handle in handles:
            await handle.wait_dead()
        assert await session.run("Write-Output recovered") == ("recovered", "", 0)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        for handle in handles:
            handle.close()


def _background_command(path: Path) -> str:
    # Inherit output pipes without creating a console window. The child therefore
    # exercises the detached drainer, not just survival with redirected output.
    return (
        "$start = [Diagnostics.ProcessStartInfo]::new();"
        f"$start.FileName = {_quote(sys.executable)};"
        f"$start.Arguments = {_quote(chr(34) + str(path) + chr(34))};"
        "$start.UseShellExecute = $false; $start.CreateNoWindow = $true;"
        "[Diagnostics.Process]::Start($start).Id"
    )


async def test_completed_background_and_future_descendants_survive_timeout(session, tmp_path):
    script, trigger, pidfile = (tmp_path / name for name in ("service.py", "trigger", "child.pid"))
    script.write_text(
        "import pathlib,subprocess,sys,time\n"
        f"while not pathlib.Path({str(trigger)!r}).exists(): time.sleep(.02)\n"
        "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
        f"pathlib.Path({str(pidfile)!r}).write_text(str(child.pid))\n"
        "time.sleep(60)\n",
        encoding="utf-8",
    )
    result = await session.run(_background_command(script))
    assert result[2] == 0, result
    service = _ProcessHandle(int(result[0]))
    child = None
    try:
        # The grandchild appears during a later foreground command. It must
        # inherit protection, not be mistaken for that command's new work.
        command = _python(
            f"import pathlib,time;pathlib.Path({str(trigger)!r}).touch();time.sleep(60)"
        )
        task = asyncio.create_task(session.run_with_timeout_flag(command, timeout=1.0))
        child = _ProcessHandle(int(await _wait_file(pidfile)))
        assert (await task)[2:] == (124, True)
        assert service.alive and child.alive
        assert await session.run("Write-Output alive") == ("alive", "", 0)
        await session.close()
        await service.wait_dead()
        await child.wait_dead()
    finally:
        service.close()
        if child:
            child.close()


@pytest.mark.parametrize("scope", [False, True])
async def test_explicit_background_preservation_keeps_output_writable(tmp_path, executable, scope):
    script, trigger, completed = (tmp_path / name for name in ("writer.py", "finish", "done"))
    script.write_text(
        "import pathlib,time,sys\n"
        f"while not pathlib.Path({str(trigger)!r}).exists():\n"
        "    print('x'*8192,flush=True)\n    sys.stderr.write('y'*8192);sys.stderr.flush()\n    time.sleep(.01)\n"
        f"pathlib.Path({str(completed)!r}).touch()\n",
        encoding="utf-8",
    )
    if scope:
        async with preserve_background_services():
            shell = PowerShellSession(tmp_path, executable=executable)
            result = await shell.run(_background_command(script))
            handle = _ProcessHandle(int(result[0].splitlines()[0]))
    else:
        shell = PowerShellSession(tmp_path, executable=executable, keep_background_on_close=True)
        result = await shell.run(_background_command(script))
        handle = _ProcessHandle(int(result[0].splitlines()[0]))
        await shell.close()
    try:
        assert handle.alive
        await asyncio.sleep(0.2)  # More than a pipe's capacity is written after close.
        trigger.touch()
        async with asyncio.timeout(5):
            while not completed.exists():
                await asyncio.sleep(0.02)
        await handle.wait_dead()
    finally:
        trigger.touch()
        handle.close()
        await shell.close()


async def test_output_drainer_exits_after_all_writers_finish(tmp_path, monkeypatch):
    import subprocess

    from nooa.tools import _powershell_session as implementation

    release = tmp_path / "release-output"
    finished_output = tmp_path / "output-consumed"
    finish = tmp_path / "finish-writer"
    identity = tmp_path / "drainer.pid"
    # Record the actual interpreter's PID as well as Popen's launcher handle.
    # The venv launcher is a separate Windows process on Python 3.12/3.13.
    monkeypatch.setattr(
        implementation,
        "_DRAIN_SCRIPT",
        "import os; from pathlib import Path; "
        f"Path({str(identity)!r}).write_text(str(os.getpid()))\n" + implementation._DRAIN_SCRIPT,
    )
    drainers = []
    original_popen = subprocess.Popen

    def record_drainer(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        drainers.append(process)
        return process

    monkeypatch.setattr(implementation.subprocess, "Popen", record_drainer)
    producer = await asyncio.create_subprocess_exec(
        sys._base_executable,
        "-c",
        "import pathlib,sys,time\n"
        "print('ready',flush=True)\n"
        f"while not pathlib.Path({str(release)!r}).exists(): time.sleep(.01)\n"
        "sys.stdout.write('x' * 262144);sys.stdout.flush();sys.stdout.close()\n"
        "sys.stderr.write('y' * 262144);sys.stderr.flush()\n"
        f"pathlib.Path({str(finished_output)!r}).write_text('drained')\n"
        f"while not pathlib.Path({str(finish)!r}).exists(): time.sleep(.01)\n",
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    actual_drainer = None
    try:
        assert producer.stdout is not None
        assert (await producer.stdout.readline()).strip() == b"ready"
        implementation._handoff_output(producer)
        assert len(drainers) == 1
        actual_drainer = _ProcessHandle(int(await _wait_file(identity)))
        release.touch()
        assert await _wait_file(finished_output) == "drained"
        # stdout has closed, but stderr is still owned by the live producer.
        # Both pipes have accepted much more than their kernel buffer capacity.
        assert producer.returncode is None
        assert actual_drainer.alive
        finish.touch()
        async with asyncio.timeout(5):
            while producer.returncode is None:
                await asyncio.sleep(0.01)
        producer._transport.close()
        await producer.wait()
        assert producer.returncode == 0
        await actual_drainer.wait_dead()
        assert await asyncio.to_thread(drainers[0].wait, timeout=5) == 0
    finally:
        release.touch()
        finish.touch()
        if producer.returncode is None:
            producer.kill()
        producer._transport.close()
        await producer.wait()
        if actual_drainer is not None:
            actual_drainer.close()
        for drainer in drainers:
            await asyncio.to_thread(drainer.wait, timeout=5)


def test_output_drainer_reports_read_errors_as_failure():
    import subprocess

    from nooa.tools._powershell_session import _DRAIN_SCRIPT

    result = subprocess.run(
        [sys._base_executable, "-c", _DRAIN_SCRIPT, "-1"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=5,
    )
    assert result.returncode == 1
    assert "WinError 6" in result.stderr


async def test_closing_a_stream_early_interrupts_and_recovers(session):
    async with aclosing(
        session.run_stream("[Console]::WriteLine('ready'); Start-Sleep -Seconds 60")
    ) as stream:
        assert "ready" in (await anext(stream))[1]
    assert await session.run("Write-Output recovered") == ("recovered", "", 0)


async def test_closing_a_completed_stream_discards_unread_output(session, tmp_path):
    completed = tmp_path / "stream-completed"
    command = (
        f"[Console]::Write('x' * 300000); [IO.File]::WriteAllText({_quote(completed)}, 'done')"
    )
    async with aclosing(session.run_stream(command)) as stream:
        name, text = await anext(stream)
        assert name == "stdout" and text
        assert await _wait_file(completed) == "done"
        # Allow completion to reach the private control channel while stdout
        # remains unread. Closing must discard this command's remaining output.
        await asyncio.sleep(0.1)
    assert await session.run("Write-Output clean") == ("clean", "", 0)


@pytest.mark.filterwarnings("error::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("error::ResourceWarning")
def test_session_survives_event_loop_recreation(tmp_path, executable):
    shell = PowerShellSession(tmp_path, "$env:INIT_LOOP = 'ready'", executable=executable)
    try:
        assert asyncio.run(shell.run("Write-Output first"))[0] == "first"
        assert asyncio.run(shell.run("$env:INIT_LOOP"))[0] == "ready"
    finally:
        asyncio.run(shell.close())
    # Force delayed transport destructors to run in this test, so a later test
    # cannot hide a leak from the event loops we closed above.
    del shell
    gc.collect()


@pytest.mark.parametrize("close_before_reuse", [False, True])
def test_concurrent_calls_survive_event_loop_recreation(tmp_path, executable, close_before_reuse):
    shell = PowerShellSession(tmp_path, "$counter = 0", executable=executable)

    async def run_concurrent():
        return await asyncio.wait_for(
            asyncio.gather(
                *(
                    shell.run("$counter += 1; Start-Sleep -Milliseconds 15; $counter")
                    for _ in range(3)
                ),
                return_exceptions=True,
            ),
            timeout=15,
        )

    async def first_loop():
        assert await run_concurrent() == [(str(i), "", 0) for i in range(1, 4)]
        if close_before_reuse:
            await shell.close()

    try:
        asyncio.run(first_loop())
        assert asyncio.run(run_concurrent()) == [(str(i), "", 0) for i in range(1, 4)]
    finally:
        asyncio.run(shell.close())


async def test_failed_init_is_reported_and_cleaned_up(tmp_path, executable):
    shell = PowerShellSession(tmp_path, "throw 'init failed'", executable=executable)
    with pytest.raises(RuntimeError, match="init failed"):
        await shell.run("Write-Output must-not-run")
    await shell.close()


@pytest.mark.parametrize("error_action", ["Continue", "Stop"])
async def test_strict_mode_init_handles_an_empty_error_history(tmp_path, executable, error_action):
    init_command = f"$ErrorActionPreference = '{error_action}'; Set-StrictMode -Version Latest"
    async with PowerShellSession(tmp_path, init_command, executable=executable) as shell:
        assert await shell.run("Write-Output ready") == ("ready", "", 0)
        assert await shell.run("$Error.Clear(); Write-Output cleared") == ("cleared", "", 0)
        stdout, stderr, code = await shell.run("$NooaStrictModeUndefinedVariable")
        assert stdout == "" and stderr and code != 0
        assert await shell.run("Write-Output alive") == ("alive", "", 0)
