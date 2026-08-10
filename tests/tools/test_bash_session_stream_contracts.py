# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent-state, process-ownership, and parity contracts for run_stream()."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import signal
import subprocess
from pathlib import Path

import pytest

from nooa.tools._bash_session import BashSession


@pytest.fixture
async def session(tmp_path):
    shell = BashSession(cwd=tmp_path)
    await shell.start()
    try:
        yield shell
    finally:
        await shell.close()


async def _collect(stream) -> list[tuple[str, str]]:
    return [item async for item in stream]


def _stream_result(events: list[tuple[str, str]]) -> tuple[str, str, int, bool]:
    stdout = "".join(value for name, value in events if name == "stdout")
    stderr = "".join(value for name, value in events if name == "stderr")
    code_text, timed_out_text = events[-1][1].split(",")
    return stdout, stderr, int(code_text), bool(int(timed_out_text))


async def _wait_until_exists(path: Path, timeout: float = 3.0) -> None:
    async with asyncio.timeout(timeout):
        while not path.exists():
            await asyncio.sleep(0.01)


def _pid_exists(pid: int) -> bool:
    try:
        result = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True
    state = result.stdout.strip()
    return result.returncode == 0 and bool(state) and not state.startswith("Z")


async def _wait_until_pid_stops(pid: int, timeout: float = 3.0) -> None:
    async with asyncio.timeout(timeout):
        while _pid_exists(pid):
            await asyncio.sleep(0.01)


@pytest.mark.parametrize(
    "command",
    [
        "printf simple",
        "return 7",
        # Commands must observe the persistent interactive shell, not an
        # implementation-only process-substitution source file.
        "printf '%s' \"${BASH_SOURCE[0]-unset}\"",
        'set -- one two; printf \'%s:%s\' "$1" "$2"',
        "parity_fn() { printf function; }; parity_fn",
        "value=$(printf pipeline | tr a-z A-Z); printf '%s' \"$value\"",
        "value=$(cat <<'EOF'\nheredoc\nEOF\n); printf '%s' \"$value\"",
        "exec 9>&1; printf fd-output >&9; exec 9>&-",
        "shopt -s nullglob; printf '%s' \"$(shopt -q nullglob; echo $?)\"",
        "trap 'printf D >&2' DEBUG; printf body; trap - DEBUG",
    ],
    ids=[
        "simple",
        "return",
        "bash-source",
        "positional-args",
        "function",
        "pipeline",
        "heredoc",
        "file-descriptor",
        "shopt",
        "debug-trap",
    ],
)
async def test_run_stream_matches_buffered_run_observables(tmp_path, command):
    buffered = BashSession(cwd=tmp_path)
    streamed = BashSession(cwd=tmp_path)
    try:
        await buffered.start()
        await streamed.start()

        expected_stdout, expected_stderr, expected_code = await buffered.run(command)
        events = await _collect(streamed.run_stream(command))
        actual_stdout, actual_stderr, actual_code, timed_out = _stream_result(events)

        assert timed_out is False
        assert actual_stdout == expected_stdout
        assert actual_stderr.removesuffix("\n") == expected_stderr
        assert actual_code == expected_code
    finally:
        await buffered.close()
        await streamed.close()


@pytest.mark.parametrize(
    "command",
    ("exit 7", "set -e; false; printf forbidden"),
    ids=("exit", "errexit"),
)
async def test_shell_terminating_commands_report_failure_and_recover(tmp_path, command):
    buffered = BashSession(cwd=tmp_path)
    streamed = BashSession(cwd=tmp_path)
    try:
        _, _, buffered_code = await buffered.run(command)
        assert buffered_code != 0
        assert await buffered.run("printf buffered-recovered") == (
            "buffered-recovered",
            "",
            0,
        )

        events = await _collect(streamed.run_stream(command))
        _, _, streamed_code, timed_out = _stream_result(events)
        assert streamed_code != 0
        assert timed_out is False
        assert await streamed.run("printf stream-recovered") == (
            "stream-recovered",
            "",
            0,
        )
    finally:
        await buffered.close()
        await streamed.close()


async def test_successful_stream_keeps_background_job_alive(session, tmp_path):
    """Persistent jobs with redirected output survive their launching command."""
    marker = tmp_path / "background-finished"
    output = tmp_path / "background-output"
    inner = (
        "import pathlib,time; time.sleep(.2); "
        "print('late output', flush=True); "
        f"pathlib.Path({str(marker)!r}).write_text('alive')"
    )

    events = await _collect(
        session.run_stream(
            f"python3 -c {shlex.quote(inner)} > {shlex.quote(str(output))} 2>&1 & printf foreground"
        )
    )

    stdout, stderr, code, timed_out = _stream_result(events)
    assert (stdout, code, timed_out) == ("foreground", 0, False)
    # Interactive Bash reports the launched job on stderr even with monitor
    # mode disabled; the job's own output is redirected as required.
    assert re.fullmatch(r"\[\d+\] \d+\n", stderr)
    await _wait_until_exists(marker, timeout=3)
    assert marker.read_text() == "alive"
    assert output.read_text() == "late output\n"
    stdout, stderr, code = await session.run("printf next")
    assert (stdout, stderr, code) == ("next", "", 0)


async def test_prior_background_job_with_redirected_output_survives_stream(session, tmp_path):
    emitted = tmp_path / "prior-output-emitted"
    output = tmp_path / "prior-output"
    await session.run(
        f"(sleep .1; printf prior-out; printf prior-err >&2; "
        f"touch {shlex.quote(str(emitted))}) </dev/null "
        f"> {shlex.quote(str(output))} 2>&1 &"
    )

    events = await _collect(session.run_stream("printf foreground; sleep .3"))
    await _wait_until_exists(emitted)
    stdout, stderr, _, _ = _stream_result(events)
    assert (stdout, stderr) == ("foreground", "")
    assert output.read_text() == "prior-outprior-err"

    next_stdout, next_stderr, code = await session.run("printf next-out; printf next-err >&2")
    assert (next_stdout, next_stderr, code) == ("next-out", "next-err", 0)


async def test_stream_state_mutations_persist_across_calls(session, tmp_path):
    fd_output = tmp_path / "fd-nine-output"
    command = (
        "set -- first second; "
        "shopt -s nullglob; "
        "set -o noclobber; "
        "trap 'export STREAM_USR2=handled' USR2; "
        f"exec 9> {shlex.quote(str(fd_output))}"
    )

    events = await _collect(session.run_stream(command))
    assert events[-1] == ("__done__", "0,0")

    stdout, stderr, code = await session.run(
        'printf \'%s:%s:%s:%s\' "$#" "$1" "$2" '
        '"$(shopt -q nullglob; echo $?)"; '
        'printf persisted >&9; kill -USR2 $$; printf "::$STREAM_USR2"; exec 9>&-'
    )
    assert (stdout, stderr, code) == ("2:first:second:0::handled", "", 0)
    assert fd_output.read_text() == "persisted"
    options, _, _ = await session.run("set -o | sed -n 's/^noclobber[[:space:]]*//p'")
    assert options == "on"
    trap, _, _ = await session.run("trap -p USR2")
    assert "STREAM_USR2=handled" in trap


async def test_existing_usr1_handler_runs_during_active_stream(session, tmp_path):
    """run_stream must not reserve a user-visible signal from persistent bash."""
    release = tmp_path / "release-usr1"
    handled = tmp_path / "usr1-handled"
    await session.run(
        f"trap 'export USER_USR1=handled; printf handled > {shlex.quote(str(handled))}' USR1"
    )
    stream = session.run_stream(
        f"printf started; while [[ ! -f {shlex.quote(str(release))} ]]; do sleep .01; done; "
        "printf finished"
    )
    assert await asyncio.wait_for(anext(stream), timeout=3) == ("stdout", "started")

    assert session._process is not None
    os.kill(session._process.pid, signal.SIGUSR1)
    handler_ran_while_active = True
    try:
        await _wait_until_exists(handled)
    except TimeoutError:
        handler_ran_while_active = False
    finally:
        release.touch()
    remaining = await _collect(stream)

    assert handler_ran_while_active
    assert "".join(value for name, value in remaining if name == "stdout") == "finished"
    stdout, _, _ = await session.run('printf "$USER_USR1"')
    assert stdout == "handled"


async def test_existing_background_job_can_signal_usr1_during_stream(session, tmp_path):
    release = tmp_path / "release-background-usr1"
    signalled = tmp_path / "background-signalled"
    await session.run("trap 'export BACKGROUND_USR1=handled' USR1")
    await session.run(
        f"(sleep .1; kill -USR1 $$; touch {shlex.quote(str(signalled))}) </dev/null &"
    )
    stream = session.run_stream(
        f"printf started; while [[ ! -f {shlex.quote(str(release))} ]]; do sleep .01; done; "
        "printf finished"
    )
    assert await asyncio.wait_for(anext(stream), timeout=3) == ("stdout", "started")

    await _wait_until_exists(signalled)
    release.touch()
    remaining = await _collect(stream)

    assert "".join(value for name, value in remaining if name == "stdout") == "finished"
    stdout, _, _ = await session.run('printf "$BACKGROUND_USR1"')
    assert stdout == "handled"


async def test_cancellation_honors_existing_sigint_handler(session):
    await session.run("trap 'export USER_INT=handled' INT; export BEFORE_INT_CANCEL=preserved")
    start_count = session._start_count
    stream = session.run_stream("printf started; sleep 30; printf stale")
    assert await asyncio.wait_for(anext(stream), timeout=3) == ("stdout", "started")

    await asyncio.wait_for(stream.aclose(), timeout=3)

    assert session._start_count == start_count
    stdout, stderr, code = await session.run('printf "$BEFORE_INT_CANCEL:$USER_INT"; trap -p INT')
    assert (stderr, code) == ("", 0)
    assert stdout.startswith("preserved:handled")
    assert "USER_INT=handled" in stdout


async def test_cancel_sigkills_descendant_that_ignores_sigterm(session, tmp_path):
    pid_path = tmp_path / "term-resistant.pid"
    inner = (
        "trap '' TERM; "
        f"printf '%s' $$ > {shlex.quote(str(pid_path))}; "
        "printf ready; while :; do sleep 1; done"
    )
    stream = session.run_stream(f"sh -c {shlex.quote(inner)}")
    pid: int | None = None
    try:
        output = ""
        while "ready" not in output:
            name, value = await asyncio.wait_for(anext(stream), timeout=3)
            if name == "stdout":
                output += value
        pid = int(pid_path.read_text())

        await asyncio.wait_for(stream.aclose(), timeout=3)

        await _wait_until_pid_stops(pid)
    finally:
        if pid is None and pid_path.exists():
            pid = int(pid_path.read_text())
        if pid is not None and _pid_exists(pid):
            os.kill(pid, signal.SIGKILL)


async def test_cancel_kills_child_spawned_by_sigterm_handler(session, tmp_path):
    root_pid_path = tmp_path / "spawning-root.pid"
    child_pid_path = tmp_path / "spawned-on-term.pid"
    on_term = f"sleep 30 & printf '%s' $! > {shlex.quote(str(child_pid_path))}; wait"
    inner = (
        f"printf '%s' $$ > {shlex.quote(str(root_pid_path))}; "
        f"trap {shlex.quote(on_term)} TERM; "
        "printf ready; while :; do sleep 1; done"
    )
    stream = session.run_stream(f"sh -c {shlex.quote(inner)}")
    child_pid: int | None = None
    try:
        output = ""
        while "ready" not in output:
            name, value = await asyncio.wait_for(anext(stream), timeout=3)
            if name == "stdout":
                output += value

        await asyncio.wait_for(stream.aclose(), timeout=3)

        await _wait_until_pid_stops(int(root_pid_path.read_text()))
        if child_pid_path.exists():
            child_pid = int(child_pid_path.read_text())
            await _wait_until_pid_stops(child_pid)
    finally:
        if child_pid is not None and _pid_exists(child_pid):
            os.kill(child_pid, signal.SIGKILL)
