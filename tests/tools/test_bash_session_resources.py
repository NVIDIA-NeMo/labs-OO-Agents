# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Artifact, descriptor, process-inspection, and FIFO lifecycle contracts."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import stat
import tempfile
import time
from pathlib import Path

import pytest

import nooa.tools._bash_session as bash_session_module
from nooa.agentdoc import FileBackedTruncatingStringIO
from nooa.tools._bash_session import BashSession, _CommandOutputPipes

_ARTIFACT_PATH = re.compile(r"full untruncated output .* is in: (.+)\n")


def _artifact_path(output: str) -> Path:
    match = _ARTIFACT_PATH.search(output)
    assert match is not None
    return Path(match.group(1))


def _fifo_dirs(root: Path) -> list[Path]:
    return sorted(root.glob("nooa-bash-output-*"))


def _shell_artifacts(root: Path) -> list[Path]:
    return sorted(root.glob("nooa-shell-output-*/nooa_shell_*.txt"))


def _fd_count() -> int | None:
    for fd_root in (Path("/proc/self/fd"), Path("/dev/fd")):
        if fd_root.is_dir():
            return len(list(fd_root.iterdir()))
    return None


async def _wait_for_fd_count_at_most(expected: int, timeout: float = 3.0) -> None:
    async with asyncio.timeout(timeout):
        while True:
            current = _fd_count()
            if current is None or current <= expected:
                return
            await asyncio.sleep(0.01)


async def test_truncated_artifact_is_private_complete_and_survives_close(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    session = BashSession(cwd=tmp_path)
    await session.start()
    content = "HEAD" + "x" * 50_000 + "TAIL"

    stdout, stderr, code = await session.run(
        f'python3 -c "import sys; sys.stdout.write({content!r})"'
    )
    path = _artifact_path(stdout)

    assert (stderr, code) == ("", 0)
    assert path.read_text() == content
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    await session.close()
    assert path.read_text() == content


async def test_multiple_published_artifacts_survive_reset_and_close(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    session = BashSession(cwd=tmp_path)
    await session.start()
    paths = []
    for marker in ("A", "B", "C"):
        stdout, _, _ = await session.run(f"python3 -c \"print({marker!r} * 50000, end='')\"")
        paths.append(_artifact_path(stdout))

    await session.reset()
    assert all(path.exists() for path in paths)

    await session.close()
    assert all(path.exists() for path in paths)


async def test_large_init_output_does_not_leave_unreachable_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    session = BashSession(
        cwd=tmp_path,
        init_command="python3 -c \"print('I' * 50000)\"",
    )
    try:
        await session.start()
        assert _shell_artifacts(tmp_path) == []
    finally:
        await session.close()


async def test_disk_capture_creation_failure_falls_back_without_leaking(tmp_path, monkeypatch):
    real_mkstemp = tempfile.mkstemp

    def selective_failure(*args, **kwargs):
        if str(kwargs.get("prefix", "")).startswith("nooa_shell_"):
            raise OSError("simulated full or unavailable temp storage")
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr(tempfile, "mkstemp", selective_failure)
    session = BashSession(cwd=tmp_path)
    try:
        await session.start()
        stdout, stderr, code = await session.run("python3 -c \"print('x' * 50000)\"")
        assert (stderr, code) == ("", 0)
        assert "not recoverable" in stdout
        assert "full untruncated output" not in stdout
        assert _shell_artifacts(tmp_path) == []
    finally:
        await session.close()


async def test_partial_backing_file_is_removed_after_write_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    def fail_backing_write(self, value):
        assert self._file is not None
        self._file.write(value[:10])
        self._file.flush()
        raise OSError("simulated partial artifact write")

    monkeypatch.setattr(FileBackedTruncatingStringIO, "_write_file", fail_backing_write)
    session = BashSession(cwd=tmp_path)
    try:
        await session.start()
        stdout, _, _ = await session.run("python3 -c \"print('x' * 50000)\"")
        assert "not recoverable" in stdout
        assert _shell_artifacts(tmp_path) == []
    finally:
        await session.close()


async def test_cancelling_buffered_run_removes_unpublished_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    started = tmp_path / "large-run-started"
    session = BashSession(cwd=tmp_path)
    try:
        await session.start()
        task = asyncio.create_task(
            session.run(f"python3 -c \"print('x' * 50000)\"; touch {started}; sleep 30")
        )
        async with asyncio.timeout(1):
            while not started.exists():
                await asyncio.sleep(0.01)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert _shell_artifacts(tmp_path) == []
    finally:
        await session.close()


async def test_timed_out_buffered_run_artifact_survives_close(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    session = BashSession(cwd=tmp_path)
    await session.start()

    stdout, _, code, timed_out = await session.run_with_timeout_flag(
        "python3 -c \"print('x' * 50000)\"; sleep 30",
        timeout=0.1,
    )
    path = _artifact_path(stdout)

    assert (code, timed_out) == (124, True)
    assert path.exists()
    await session.close()
    assert path.exists()


def test_shell_artifact_capture_has_finite_per_file_and_session_disk_limits():
    per_file = getattr(bash_session_module, "MAX_OUTPUT_ARTIFACT_BYTES", None)
    per_session = getattr(bash_session_module, "MAX_OUTPUT_ARTIFACT_TOTAL_BYTES", None)
    assert isinstance(per_file, int) and per_file > 0
    assert isinstance(per_session, int) and per_session >= per_file


async def test_artifact_limit_bounds_disk_and_does_not_claim_complete_capture(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    artifact_limit = 60_000
    monkeypatch.setattr(
        bash_session_module,
        "MAX_OUTPUT_ARTIFACT_BYTES",
        artifact_limit,
        raising=False,
    )
    session = BashSession(cwd=tmp_path)
    try:
        await session.start()
        stdout, stderr, code = await session.run(
            "python3 -c \"import sys; sys.stdout.write('€' * 40000)\""
        )

        artifacts = _shell_artifacts(tmp_path)
        assert (stderr, code) == ("", 0)
        assert len(artifacts) == 1
        assert artifacts[0].stat().st_size <= artifact_limit
        artifacts[0].read_text()  # The byte boundary must not split UTF-8.
        assert str(artifacts[0]) in stdout
        assert "full untruncated output" not in stdout
        assert "artifact is incomplete" in stdout.lower()
    finally:
        await session.close()


async def test_session_total_artifact_quota_bounds_aggregate_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(
        bash_session_module,
        "MAX_OUTPUT_ARTIFACT_BYTES",
        50_000,
        raising=False,
    )
    total_limit = 90_000
    monkeypatch.setattr(
        bash_session_module,
        "MAX_OUTPUT_ARTIFACT_TOTAL_BYTES",
        total_limit,
        raising=False,
    )
    session = BashSession(cwd=tmp_path)
    try:
        await session.start()
        published: list[Path] = []
        last_stdout = ""
        for marker in ("A", "B", "C", "D"):
            stdout, _, code = await session.run(f"python3 -c \"print({marker!r} * 40000, end='')\"")
            last_stdout = stdout
            assert code == 0
            assert "<truncated-output>" in stdout
            match = _ARTIFACT_PATH.search(stdout)
            if match is not None:
                published.append(Path(match.group(1)))
            assert all(path.exists() for path in published)

        artifacts = _shell_artifacts(tmp_path)
        assert artifacts
        assert sum(path.stat().st_size for path in artifacts) <= total_limit
        assert all(path.exists() for path in published)
        assert "full untruncated output" not in last_stdout
        assert "artifact was not retained" in last_stdout.lower()
    finally:
        await session.close()


async def test_start_reaps_expired_dead_owner_but_preserves_live_and_recent_owners(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    ttl = getattr(bash_session_module, "OUTPUT_ARTIFACT_TTL_SECONDS", None)
    assert isinstance(ttl, int) and ttl > 0

    def owner_dir(name: str, *, pid: int, start_token: str, orphaned_at: float) -> Path:
        directory = tmp_path / f"nooa-shell-output-{name}"
        directory.mkdir(mode=0o700)
        (directory / "owner.json").write_text(
            json.dumps(
                {
                    "pid": pid,
                    "start_token": start_token,
                    "orphaned_at": orphaned_at,
                }
            )
        )
        (directory / "nooa_shell_stdout_test.txt").write_text(name)
        return directory

    own_token = bash_session_module._process_start_token(os.getpid())
    expired_dead = owner_dir(
        "expired-dead",
        pid=999_999_991,
        start_token="dead",
        orphaned_at=time.time() - ttl - 1,
    )
    recent_dead = owner_dir(
        "recent-dead",
        pid=999_999_992,
        start_token="dead",
        orphaned_at=time.time(),
    )
    old_live = owner_dir(
        "old-live",
        pid=os.getpid(),
        start_token=own_token,
        orphaned_at=time.time() - ttl - 1,
    )
    non_private = owner_dir(
        "non-private",
        pid=999_999_993,
        start_token="dead",
        orphaned_at=time.time() - ttl - 1,
    )
    non_private.chmod(0o755)

    session = BashSession(cwd=tmp_path)
    try:
        await session.start()
        assert not expired_dead.exists()
        assert recent_dead.exists()
        assert old_live.exists()
        assert non_private.exists()
    finally:
        await session.close()


async def test_output_pipe_directory_and_fifos_are_private(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    pipes = await _CommandOutputPipes.open()
    try:
        assert stat.S_IMODE(pipes.directory.stat().st_mode) == 0o700
        assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in pipes.paths.values())
    finally:
        await pipes.close()
    assert _fifo_dirs(tmp_path) == []


async def test_repeated_stream_lifecycles_do_not_leak_fds_or_fifo_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    session = BashSession(cwd=tmp_path)
    await session.start()
    baseline_fds = _fd_count()
    try:
        for index in range(10):
            events = [item async for item in session.run_stream(f"printf success-{index}")]
            assert events[-1] == ("__done__", "0,0")

        for _ in range(5):
            stream = session.run_stream("printf started; sleep 30")
            assert await asyncio.wait_for(anext(stream), timeout=3) == ("stdout", "started")
            await asyncio.wait_for(stream.aclose(), timeout=3)

        for _ in range(5):
            events = [
                item async for item in session.run_stream("printf timeout; sleep 30", timeout=0.01)
            ]
            assert events[-1] == ("__done__", "124,1")

        await asyncio.sleep(0)
        assert _fifo_dirs(tmp_path) == []
        if baseline_fds is not None:
            await _wait_for_fd_count_at_most(baseline_fds)
    finally:
        await session.close()


async def test_repeated_output_boundary_failures_do_not_leak_resources(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    session = BashSession(cwd=tmp_path)
    await session.start()
    baseline_fds = _fd_count()
    try:
        for _ in range(5):
            with pytest.raises(RuntimeError, match="stdout closed before"):
                _ = [item async for item in session.run_stream("exec 1>&-")]
            assert await session.run("printf recovered") == ("recovered", "", 0)

        await asyncio.sleep(0)
        assert _fifo_dirs(tmp_path) == []
        if baseline_fds is not None:
            await _wait_for_fd_count_at_most(baseline_fds)
    finally:
        await session.close()


async def test_output_pipe_partial_open_failure_cleans_everything(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    loop = asyncio.get_running_loop()
    real_connect = loop.connect_read_pipe
    calls = 0

    async def fail_second_connect(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated connect_read_pipe failure")
        return await real_connect(*args, **kwargs)

    monkeypatch.setattr(loop, "connect_read_pipe", fail_second_connect)

    with pytest.raises(OSError, match="simulated"):
        await _CommandOutputPipes.open()
    await asyncio.sleep(0)
    assert _fifo_dirs(tmp_path) == []


async def test_second_process_snapshot_failure_resets_without_fifo_leak(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    session = BashSession(cwd=tmp_path)
    await session.start()
    real_parent_map = session._process_parent_map
    calls = 0

    async def fail_second_snapshot():
        nonlocal calls
        calls += 1
        if calls == 2:
            return None
        return await real_parent_map()

    monkeypatch.setattr(session, "_process_parent_map", fail_second_snapshot)
    start_count = session._start_count
    try:
        stream = session.run_stream("printf started; sleep 30")
        assert await asyncio.wait_for(anext(stream), timeout=3) == ("stdout", "started")
        await asyncio.wait_for(stream.aclose(), timeout=3)

        assert session._start_count == start_count + 1
        assert _fifo_dirs(tmp_path) == []
        assert await session.run("printf recovered") == ("recovered", "", 0)
    finally:
        await session.close()


async def test_reset_failure_during_broken_pipe_does_not_leak_fifo_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    session = BashSession(cwd=tmp_path)
    await session.start()
    assert session._process is not None

    class BrokenStdin:
        def write(self, _data):
            raise BrokenPipeError("simulated broken stdin")

        async def drain(self):
            return None

    session._process.stdin = BrokenStdin()  # type: ignore[assignment]

    async def failed_reset():
        raise RuntimeError("simulated reset failure")

    monkeypatch.setattr(session, "reset", failed_reset)
    try:
        with pytest.raises(RuntimeError, match="simulated reset failure"):
            await anext(session.run_stream("printf never"))
        assert _fifo_dirs(tmp_path) == []
    finally:
        for directory in _fifo_dirs(tmp_path):
            shutil.rmtree(directory)
        await session.close()
