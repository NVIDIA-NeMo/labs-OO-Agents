# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core behavior of the DEFAULT ShellTools (run / read / write_file / replace).

This keeps the *default* ShellTools — the one agents actually use — covered for
its primary file/run surface. Search-anchor behavior is covered separately in
test_shell_tools_modern.py.
"""

import asyncio
import shlex

import pytest

from nooa.tools.shell_tools import Match, ShellTools


@pytest.fixture
def sh(tmp_path):
    return ShellTools(cwd=str(tmp_path))


@pytest.mark.asyncio
async def test_run_persists_state(sh, tmp_path):
    r = await sh.run("echo hello")
    assert r.success
    assert "hello" in r.stdout
    # cd persists across calls in the same session.
    (tmp_path / "sub").mkdir()
    await sh.run("cd sub")
    r2 = await sh.run("pwd")
    assert r2.stdout.strip().endswith("sub")


@pytest.mark.asyncio
async def test_run_reports_failure(sh):
    r = await sh.run("false")
    assert not r.success
    assert r.returncode != 0


@pytest.mark.asyncio
async def test_write_file_then_read(sh, tmp_path):
    await sh.write_file("f.txt", "line1\nline2\nline3\n")
    assert (tmp_path / "f.txt").read_text() == "line1\nline2\nline3\n"
    # read with a numbered gutter (default) -> Match; inspect via .numbered/.text.
    view = await sh.read("f.txt")
    assert "line2" in view.numbered
    # read a line window -> Match for just that line.
    window = await sh.read("f.txt", (2, 2))
    assert "line2" in window.text
    assert "line1" not in window.text


@pytest.mark.asyncio
async def test_replace_path_unique(sh, tmp_path):
    await sh.write_file("f.py", "x = 1\ny = 2\nz = 3\n")
    await sh.replace("f.py", "y = 2", "y = 22")
    assert (tmp_path / "f.py").read_text() == "x = 1\ny = 22\nz = 3\n"


@pytest.mark.asyncio
async def test_replace_path_ambiguous_errors(sh, tmp_path):
    await sh.write_file("f.py", "a = 1\na = 1\n")
    with pytest.raises(ValueError, match="matched 2 times"):
        # Two matches -> must error rather than guess.
        await sh.replace("f.py", "a = 1", "a = 2")


@pytest.mark.asyncio
async def test_write_file_is_overwrite(sh, tmp_path):
    await sh.write_file("f.txt", "old")
    await sh.write_file("f.txt", "new")
    assert (tmp_path / "f.txt").read_text() == "new"


@pytest.mark.asyncio
async def test_file_operations_reject_paths_outside_cwd(sh, tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("secret")

    with pytest.raises(ValueError, match="escapes ShellTools cwd"):
        await sh.read(f"../{outside.name}")
    with pytest.raises(ValueError, match="escapes ShellTools cwd"):
        await sh.replace(f"../{outside.name}", "secret", "changed")
    with pytest.raises(ValueError, match="escapes ShellTools cwd"):
        await sh.replace(Match(f"../{outside.name}", 1, 1, "secret"), "changed")
    with pytest.raises(ValueError, match="escapes ShellTools cwd"):
        await sh.write_file(str(outside), "overwritten")

    assert outside.read_text() == "secret"


@pytest.mark.asyncio
async def test_close_terminates_underlying_bash_session(sh):
    """Verify close() terminates BashSession and the shell lazily restarts."""
    r = await sh.run("echo started")
    assert r.success
    assert sh._session._process is not None

    await sh.close()

    assert sh._session._process is None
    assert not sh._session._started

    # The shell remains reusable after close(); a fresh session starts lazily.
    r2 = await sh.run("echo restarted")
    assert r2.success
    assert "restarted" in r2.stdout
    await sh.close()


@pytest.mark.asyncio
async def test_abandoned_stream_interrupts_command_and_resynchronizes_shell(sh):
    await sh.run("export STREAM_CANCEL_STATE=preserved")
    start_count = sh.session._start_count
    stream = sh.run_stream("printf started; sleep 30; printf stale")

    first = await asyncio.wait_for(anext(stream), timeout=1.0)
    assert first.kind == "stdout"
    assert first.text == "started"

    await asyncio.wait_for(stream.aclose(), timeout=3.0)

    assert sh.session._start_count == start_count
    recovered = await asyncio.wait_for(
        sh.run('printf "recovered:$STREAM_CANCEL_STATE"'), timeout=2.0
    )
    assert recovered.stdout == "recovered:preserved"
    assert "stale" not in recovered.stdout


@pytest.mark.asyncio
async def test_late_stream_close_preserves_shelltools_state(sh, tmp_path):
    subdir = tmp_path / "stream-cwd"
    subdir.mkdir()
    await sh.run("export PRESERVE=yes")
    start_count = sh.session._start_count
    stream = sh.run_stream(f"cd {shlex.quote(str(subdir))}; printf finished")

    first = await asyncio.wait_for(anext(stream), timeout=1)
    assert first.kind == "stdout"
    assert first.text == "finished"
    await asyncio.sleep(0.1)
    await asyncio.wait_for(stream.aclose(), timeout=1)

    assert sh.session._start_count == start_count
    assert sh.cwd == subdir
    recovered = await sh.run('printf "$PRESERVE:$PWD"')
    assert recovered.stdout == f"yes:{subdir}"


@pytest.mark.asyncio
async def test_stream_delivers_large_stdout_and_stderr_losslessly(sh):
    command = (
        'python3 -c "import os; '
        "os.write(1, b'x' * 100000); os.write(2, b'y' * 70000); raise SystemExit(7)\""
    )

    events = [event async for event in sh.run_stream(command)]

    assert "".join(event.text for event in events if event.kind == "stdout") == "x" * 100_000
    assert "".join(event.text for event in events if event.kind == "stderr") == "y" * 70_000
    assert events[-1].kind == "done"
    assert events[-1].returncode == 7
    assert events[-1].timed_out is False
    assert sum(event.kind == "done" for event in events) == 1


@pytest.mark.asyncio
async def test_stream_delivers_output_before_command_finishes(sh, tmp_path):
    release = tmp_path / "release-shelltools-stream"
    command = (
        "printf first; "
        f"while [ ! -f {shlex.quote(str(release))} ]; do sleep 0.01; done; "
        "printf second"
    )
    stream = sh.run_stream(command)

    first = await asyncio.wait_for(anext(stream), timeout=1.0)
    assert first.kind == "stdout"
    assert first.text == "first"

    release.write_text("")
    remaining = [event async for event in stream]
    assert "".join(event.text for event in remaining if event.kind == "stdout") == "second"
    assert remaining[-1].kind == "done"
    assert remaining[-1].returncode == 0
    assert remaining[-1].timed_out is False


@pytest.mark.asyncio
async def test_stream_reports_timeout_and_shell_remains_reusable(sh):
    events = [
        event async for event in sh.run_stream("printf before-timeout; sleep 10", timeout=0.1)
    ]

    assert "".join(event.text for event in events if event.kind == "stdout") == "before-timeout"
    assert events[-1].kind == "done"
    assert events[-1].returncode == 124
    assert events[-1].timed_out is True

    recovered = await asyncio.wait_for(sh.run("printf recovered"), timeout=2.0)
    assert recovered.stdout == "recovered"


@pytest.mark.asyncio
async def test_stream_does_not_confuse_exit_124_with_timeout(sh):
    events = [event async for event in sh.run_stream("bash -c 'exit 124'", timeout=2.0)]

    assert events[-1].kind == "done"
    assert events[-1].returncode == 124
    assert events[-1].timed_out is False
