# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for new BashSession APIs added in the concurrency fix."""

import asyncio

import pytest

from nooa.tools._bash_session import BashSession


class TestRunWithTimeoutFlag:
    """Tests for BashSession.run_with_timeout_flag()."""

    @pytest.fixture
    async def session(self, tmp_path):
        session = BashSession(cwd=tmp_path)
        await session.start()
        yield session
        await session.close()

    async def test_returns_four_tuple_on_success(self, session):
        stdout, stderr, code, timed_out = await session.run_with_timeout_flag("echo hello")
        assert stdout == "hello"
        assert code == 0
        assert timed_out is False

    async def test_returns_timed_out_true_on_timeout(self, session):
        stdout, stderr, code, timed_out = await session.run_with_timeout_flag(
            "sleep 10", timeout=0.5
        )
        assert code == 124
        assert timed_out is True

    async def test_exit_code_124_not_confused_with_timeout(self, session):
        """A command that exits 124 naturally should NOT report timed_out=True."""
        stdout, stderr, code, timed_out = await session.run_with_timeout_flag(
            "bash -c 'exit 124'", timeout=5
        )
        assert code == 124
        assert timed_out is False


class TestAsyncContextManager:
    """Tests for BashSession async with protocol."""

    async def test_async_with_starts_and_closes(self, tmp_path):
        async with BashSession(cwd=tmp_path) as session:
            stdout, stderr, code = await session.run("echo works")
            assert stdout == "works"
            assert code == 0
        # After exit, session should be closed
        assert session._started is False
        assert session._process is None

    async def test_async_with_closes_on_exception(self, tmp_path):
        with pytest.raises(ValueError):
            async with BashSession(cwd=tmp_path) as session:
                raise ValueError("test")
        assert session._started is False


class TestLockSerialization:
    """Tests that concurrent run() calls are serialized."""

    async def test_concurrent_runs_dont_corrupt(self, tmp_path):
        """Two concurrent run() calls should both succeed without corruption."""
        session = BashSession(cwd=tmp_path)
        await session.start()
        try:
            results = await asyncio.gather(
                session.run("echo first"),
                session.run("echo second"),
            )
            outputs = {r[0] for r in results}
            assert "first" in outputs
            assert "second" in outputs
            # Both should have exit code 0
            assert all(r[2] == 0 for r in results)
        finally:
            await session.close()
