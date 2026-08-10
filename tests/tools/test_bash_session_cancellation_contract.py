# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adversarial cancellation and timeout contracts for the primary shell tool."""

from __future__ import annotations

import asyncio
import shlex
from pathlib import Path

import pytest

from nooa.tools._bash_session import BashSession
from nooa.tools.shell_tools import ShellTools


@pytest.fixture
async def session(tmp_path):
    shell = BashSession(cwd=tmp_path)
    await shell.start()
    try:
        yield shell
    finally:
        await shell.close()


async def _wait_until_exists(path: Path, timeout: float = 3.0) -> None:
    async with asyncio.timeout(timeout):
        while not path.exists():
            await asyncio.sleep(0.01)


def _stdout(events: list[tuple[str, str]]) -> str:
    return "".join(value for name, value in events if name == "stdout")


async def test_closing_stream_prevents_later_side_effects(session, tmp_path):
    """Cancellation aborts the shell program, not merely its blocking child."""
    after = tmp_path / "after-cancel"
    start_count = session._start_count
    stream = session.run_stream(
        "printf started; sleep 30; "
        f"touch {shlex.quote(str(after))}; "
        "export AFTER_CANCEL=ran; printf stale"
    )

    assert await asyncio.wait_for(anext(stream), timeout=3) == ("stdout", "started")
    await asyncio.wait_for(stream.aclose(), timeout=3)

    assert not after.exists()
    assert session._start_count == start_count
    stdout, stderr, code = await session.run('printf "$AFTER_CANCEL"')
    assert (stdout, stderr, code) == ("", "", 0)


async def test_stream_timeout_prevents_later_output_and_side_effects(session, tmp_path):
    after = tmp_path / "after-timeout"

    events = [
        event
        async for event in session.run_stream(
            "printf before; sleep 30; "
            f"touch {shlex.quote(str(after))}; "
            "export AFTER_TIMEOUT=ran; printf stale",
            timeout=0.1,
        )
    ]

    assert _stdout(events) == "before"
    assert events[-1] == ("__done__", "124,1")
    assert not after.exists()
    stdout, stderr, code = await session.run('printf "$AFTER_TIMEOUT"')
    assert (stdout, stderr, code) == ("", "", 0)


async def test_buffered_timeout_prevents_later_output_and_side_effects(session, tmp_path):
    after = tmp_path / "buffered-after-timeout"

    stdout, _stderr, code, timed_out = await session.run_with_timeout_flag(
        "printf before; sleep 30; "
        f"touch {shlex.quote(str(after))}; "
        "export BUFFERED_AFTER_TIMEOUT=ran; printf stale",
        timeout=0.1,
    )

    assert stdout == "before"
    assert (code, timed_out) == (124, True)
    assert not after.exists()
    stdout, stderr, code = await session.run('printf "$BUFFERED_AFTER_TIMEOUT"')
    assert (stdout, stderr, code) == ("", "", 0)


@pytest.mark.parametrize("streamed", (False, True), ids=("buffered", "streamed"))
async def test_timeout_preserves_background_job_started_by_earlier_command(
    session, tmp_path, streamed
):
    prior_finished = tmp_path / f"prior-timeout-{streamed}"
    active_finished = tmp_path / f"active-timeout-{streamed}"
    await session.run(
        f"(sleep .3; printf alive > {shlex.quote(str(prior_finished))}) "
        "</dev/null >/dev/null 2>&1 &"
    )

    command = f"sleep 30; touch {shlex.quote(str(active_finished))}"
    if streamed:
        events = [event async for event in session.run_stream(command, timeout=0.05)]
        assert events[-1] == ("__done__", "124,1")
    else:
        _, _, code, timed_out = await session.run_with_timeout_flag(command, timeout=0.05)
        assert (code, timed_out) == (124, True)

    await _wait_until_exists(prior_finished)
    assert prior_finished.read_text() == "alive"
    assert not active_finished.exists()


async def test_cancelling_buffered_run_aborts_command_and_releases_shell(session, tmp_path):
    started = tmp_path / "buffered-started"
    after = tmp_path / "buffered-after"
    await session.run("export BEFORE_BUFFER_CANCEL=preserved")
    start_count = session._start_count
    task = asyncio.create_task(
        session.run(
            f"touch {shlex.quote(str(started))}; sleep 30; "
            f"touch {shlex.quote(str(after))}; printf stale"
        )
    )
    await _wait_until_exists(started)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not after.exists()
    assert session._start_count == start_count
    stdout, stderr, code = await asyncio.wait_for(
        session.run('printf "recovered:$BEFORE_BUFFER_CANCEL"'),
        timeout=2,
    )
    assert (stdout, stderr, code) == ("recovered:preserved", "", 0)


async def test_cancelling_buffered_run_preserves_prior_background_job(session, tmp_path):
    prior_finished = tmp_path / "prior-buffer-cancel"
    active_finished = tmp_path / "active-buffer-cancel"
    active_started = tmp_path / "active-buffer-started"
    await session.run(
        f"(sleep .3; printf alive > {shlex.quote(str(prior_finished))}) "
        "</dev/null >/dev/null 2>&1 &"
    )
    task = asyncio.create_task(
        session.run(
            f"touch {shlex.quote(str(active_started))}; sleep .15; "
            f"touch {shlex.quote(str(active_finished))}"
        )
    )
    await _wait_until_exists(active_started)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    await _wait_until_exists(prior_finished)
    assert prior_finished.read_text() == "alive"
    assert not active_finished.exists()


async def test_stream_timeout_updates_bash_session_cwd(session, tmp_path):
    subdir = tmp_path / "timed-out-cwd"
    subdir.mkdir()

    events = [
        event
        async for event in session.run_stream(
            f"cd {shlex.quote(str(subdir))}; printf ready; sleep 30",
            timeout=0.1,
        )
    ]

    assert _stdout(events) == "ready"
    assert events[-1] == ("__done__", "124,1")
    assert session.cwd == subdir
    stdout, _, _ = await session.run("pwd")
    assert stdout == str(subdir)


async def test_buffered_timeout_updates_bash_session_cwd(session, tmp_path):
    subdir = tmp_path / "buffered-timed-out-cwd"
    subdir.mkdir()

    stdout, _stderr, code, timed_out = await session.run_with_timeout_flag(
        f"cd {shlex.quote(str(subdir))}; printf ready; sleep 30",
        timeout=0.1,
    )

    assert stdout == "ready"
    assert (code, timed_out) == (124, True)
    assert session.cwd == subdir
    stdout, _, _ = await session.run("pwd")
    assert stdout == str(subdir)


async def test_stream_timeout_updates_shelltools_cwd(tmp_path):
    subdir = tmp_path / "shelltools-timed-out-cwd"
    subdir.mkdir()
    shell = ShellTools(cwd=str(tmp_path))
    try:
        events = [
            event
            async for event in shell.run_stream(
                f"cd {shlex.quote(str(subdir))}; printf ready; sleep 30",
                timeout=0.1,
            )
        ]

        assert "".join(event.text for event in events if event.kind == "stdout") == "ready"
        assert events[-1].kind == "done"
        assert events[-1].timed_out is True
        assert shell.cwd == subdir
        result = await shell.run("pwd")
        assert result.stdout == str(subdir)
    finally:
        await shell.close()


async def test_shelltools_close_stream_prevents_later_side_effects(tmp_path):
    after = tmp_path / "shelltools-after-cancel"
    shell = ShellTools(cwd=str(tmp_path))
    try:
        stream = shell.run_stream(
            "printf started; sleep 30; "
            f"touch {shlex.quote(str(after))}; "
            "export SHELLTOOLS_AFTER_CANCEL=ran; printf stale"
        )
        first = await asyncio.wait_for(anext(stream), timeout=3)
        assert (first.kind, first.text) == ("stdout", "started")

        await asyncio.wait_for(stream.aclose(), timeout=3)

        assert not after.exists()
        result = await shell.run('printf "$SHELLTOOLS_AFTER_CANCEL"')
        assert result.stdout == ""
    finally:
        await shell.close()


async def test_cancelling_shelltools_run_releases_shell_cleanly(tmp_path):
    started = tmp_path / "shelltools-run-started"
    after = tmp_path / "shelltools-run-after"
    shell = ShellTools(cwd=str(tmp_path))
    try:
        task = asyncio.create_task(
            shell.run(
                f"touch {shlex.quote(str(started))}; sleep 30; "
                f"touch {shlex.quote(str(after))}; printf stale"
            )
        )
        await _wait_until_exists(started)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert not after.exists()
        result = await asyncio.wait_for(shell.run("printf recovered"), timeout=2)
        assert result.stdout == "recovered"
    finally:
        await shell.close()
