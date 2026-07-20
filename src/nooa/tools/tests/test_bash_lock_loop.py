# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for BashSession asyncio.Lock loop-mismatch fix (gl-212)."""

import asyncio
import os
import signal

import pytest

from nooa.tools._bash_session import BashSession


@pytest.fixture
def bash_session():
    """Create a BashSession for testing."""
    session = BashSession()
    yield session
    # Best-effort cleanup — kill the subprocess if still alive
    if session._process is not None and session._process.returncode is None:
        try:
            os.killpg(os.getpgid(session._process.pid), signal.SIGKILL)
        except Exception:
            try:
                session._process.kill()
            except Exception:
                pass


class TestLockLoopMismatch:
    """Verify that BashSession survives event loop changes without Lock errors."""

    @pytest.mark.asyncio
    async def test_basic_run(self, bash_session):
        """Sanity check: run works on a single loop."""
        stdout, stderr, code = await bash_session.run("echo hello")
        assert stdout.strip() == "hello"
        assert code == 0

    def test_run_across_loops(self, bash_session):
        """Lock is recreated when the event loop changes between calls."""

        async def run_on_this_loop(session):
            stdout, _, code = await session.run("echo loop_ok")
            return stdout.strip(), code

        # First call on loop A
        result_a = asyncio.run(run_on_this_loop(bash_session))
        assert result_a == ("loop_ok", 0)

        # Second call on loop B (different loop) — would raise
        # "is bound to a different event loop" without the fix
        result_b = asyncio.run(run_on_this_loop(bash_session))
        assert result_b == ("loop_ok", 0)

    def test_run_with_timeout_flag_across_loops(self, bash_session):
        """run_with_timeout_flag also handles loop changes."""

        async def run_on_this_loop(session):
            stdout, _, code, timed_out = await session.run_with_timeout_flag("echo ok")
            return stdout.strip(), code, timed_out

        result_a = asyncio.run(run_on_this_loop(bash_session))
        assert result_a == ("ok", 0, False)

        result_b = asyncio.run(run_on_this_loop(bash_session))
        assert result_b == ("ok", 0, False)

    def test_run_stream_across_loops(self, bash_session):
        """run_stream also handles loop changes."""

        async def stream_on_this_loop(session):
            chunks = []
            async for name, data in session.run_stream("echo streamed"):
                chunks.append((name, data))
            return chunks

        chunks_a = asyncio.run(stream_on_this_loop(bash_session))
        assert any(name == "__done__" for name, _ in chunks_a)
        stdout_chunks = [data for name, data in chunks_a if name == "stdout"]
        assert any("streamed" in c for c in stdout_chunks)

        chunks_b = asyncio.run(stream_on_this_loop(bash_session))
        assert any(name == "__done__" for name, _ in chunks_b)

    @pytest.mark.asyncio
    async def test_ensure_lock_noop_when_same_loop(self, bash_session):
        """_ensure_lock_on_current_loop is a no-op when loop hasn't changed."""
        await bash_session.run("echo start")
        original_lock = bash_session._lock

        # Same loop — lock should NOT be recreated
        bash_session._ensure_lock_on_current_loop()
        assert bash_session._lock is original_lock

    @pytest.mark.asyncio
    async def test_ensure_lock_recreates_on_loop_change(self, bash_session):
        """_ensure_lock_on_current_loop recreates lock when loop differs."""
        await bash_session.run("echo start")
        original_lock = bash_session._lock

        # Simulate a loop change by setting _started_on_loop to a different object
        bash_session._started_on_loop = object()
        bash_session._ensure_lock_on_current_loop()
        assert bash_session._lock is not original_lock

    def test_run_after_close_on_new_loop(self, bash_session):
        """Lock is fresh after close(), so a new loop works without error."""

        async def start_and_close(session):
            await session.run("echo first")
            await session.close()

        async def run_after_close(session):
            stdout, _, code = await session.run("echo second")
            return stdout.strip(), code

        # Start and close on loop A
        asyncio.run(start_and_close(bash_session))

        # Use on loop B — would fail without lock reset in close()
        result = asyncio.run(run_after_close(bash_session))
        assert result == ("second", 0)
