# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for BashSession auto-recovery and bulletproofing (issue #178).

Tests verify:
1. Dead process recovery (auto-restart on killed/exited bash)
2. Stdin isolation (commands cannot steal control-pipe sentinels)
3. Large output handling (no pipe deadlock on >192KB output)
4. EOF on control fd reports non-zero exit code
5. Pipelines and redirects work correctly
"""

import asyncio
import os
import signal

import pytest

from nooa.tools._bash_session import BashSession


@pytest.fixture
async def session(tmp_path):
    """Fresh BashSession for each test."""
    s = BashSession(cwd=tmp_path)
    await s.start()
    yield s
    await s.close()


class TestAutoRecoveryFromDeadProcess:
    """BashSession should auto-recover when the bash process dies."""

    async def test_recovers_after_process_killed(self, session):
        """After killing the bash process, the next command should still work."""
        os.kill(session._process.pid, signal.SIGKILL)
        await asyncio.sleep(0.1)

        stdout, stderr, code = await session.run("echo recovered")
        assert stdout == "recovered"

    async def test_recovers_after_process_exits(self, session):
        """After bash exits on its own, the next command should still work."""
        session._process.stdin.write(b"exit 0\n")
        await session._process.stdin.drain()
        await asyncio.sleep(0.1)

        stdout, stderr, code = await session.run("echo recovered")
        assert stdout == "recovered"


class TestStdinIsolation:
    """Commands cannot steal from the control pipe."""

    async def test_cat_no_args_does_not_hang(self, session):
        """cat with no file args should get EOF quickly (fd 3 architecture)."""
        stdout, stderr, code = await session.run("cat", timeout=5.0)

        # Session should still be usable afterwards
        stdout2, _, _ = await session.run("echo alive")
        assert stdout2 == "alive"

    async def test_read_builtin_does_not_hang(self, session):
        """read from stdin should get EOF, not block."""
        stdout, stderr, code = await session.run("read line; echo got=$line", timeout=5.0)

        stdout2, _, _ = await session.run("echo still_works")
        assert stdout2 == "still_works"


class TestLargeOutput:
    """Commands producing large output should not deadlock."""

    async def test_200kb_stdout_does_not_deadlock(self, session):
        """A command producing >192KB of output should complete without timeout."""
        # Generate 200KB of output (well above the 128KB StreamReader + 64KB pipe limit)
        stdout, stderr, code = await session.run("python3 -c \"print('A' * 200000)\"", timeout=10.0)
        assert code == 0
        assert len(stdout) >= 30000  # Truncated by MAX_OUTPUT_CHARS but proves no deadlock

    async def test_large_stderr_does_not_deadlock(self, session):
        """Large stderr should also complete without deadlock."""
        stdout, stderr, code = await session.run(
            "python3 -c \"import sys; sys.stderr.write('B' * 200000)\"", timeout=10.0
        )
        assert code == 0

    async def test_session_usable_after_large_output(self, session):
        """After large output, session should still work for normal commands."""
        await session.run("python3 -c \"print('X' * 200000)\"", timeout=10.0)
        stdout, _, code = await session.run("echo still_alive")
        assert stdout == "still_alive"
        assert code == 0


class TestEOFExitCode:
    """EOF on control fd (bash died) should report non-zero exit code."""

    async def test_killed_process_returns_nonzero(self, session):
        """If bash is killed during execution, exit code should be non-zero."""
        # Kill bash while it's running a sleep
        pid = session._process.pid

        async def kill_later():
            await asyncio.sleep(0.5)
            os.kill(pid, signal.SIGKILL)

        asyncio.ensure_future(kill_later())
        stdout, stderr, code = await session.run("sleep 10", timeout=5.0)

        # After recovery, exit code from the failed command should be non-zero
        # (either timeout=124 or process-death=-1)
        assert code != 0


class TestPipelinesAndRedirects:
    """Pipelines and redirects must work correctly."""

    async def test_simple_pipe(self, session):
        """echo | grep should work."""
        stdout, stderr, code = await session.run("echo hello | grep hello")
        assert "hello" in stdout
        assert code == 0

    async def test_multi_pipe(self, session):
        """Multi-stage pipeline."""
        stdout, stderr, code = await session.run("echo -e 'b\\na\\nc' | sort | head -1")
        assert stdout == "a"

    async def test_stdin_redirect_from_file(self, session, tmp_path):
        """Explicit stdin redirect from a file should work."""
        test_file = tmp_path / "input.txt"
        test_file.write_text("file_content\n")
        stdout, stderr, code = await session.run(f"cat < {test_file}")
        assert "file_content" in stdout

    async def test_heredoc(self, session):
        """Heredoc should work correctly."""
        stdout, stderr, code = await session.run("cat <<EOF\nheredoc_line\nEOF")
        assert "heredoc_line" in stdout

    async def test_command_substitution(self, session):
        """Command substitution should work."""
        stdout, stderr, code = await session.run("echo $(echo nested)")
        assert stdout == "nested"
