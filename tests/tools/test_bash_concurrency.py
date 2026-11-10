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


class TestConcurrentStart:
    """Concurrent first calls on a fresh session share a single bash start."""

    async def test_gathered_runs_on_unstarted_session_all_succeed(self, tmp_path):
        session = BashSession(cwd=tmp_path)
        try:
            results = await asyncio.gather(*(session.run(f"sleep 0.1; echo {n}") for n in range(4)))
            assert [stdout for stdout, _err, _code in results] == ["0", "1", "2", "3"]
            assert session._start_count == 1
        finally:
            await session.close()

    async def test_gathered_starts_spawn_one_process(self, tmp_path):
        session = BashSession(cwd=tmp_path)
        try:
            await asyncio.gather(*(session.start() for _ in range(3)))
            assert session._start_count == 1
            assert (await session.run("echo ok"))[0] == "ok"
        finally:
            await session.close()

    async def test_shell_tools_gathered_runs_are_serialized(self, tmp_path):
        from nooa.tools.shell_tools import ShellTools

        shell = ShellTools(cwd=str(tmp_path))
        try:
            results = await asyncio.gather(*(shell.run(f"echo {n}") for n in range(5)))
            assert [result.stdout.strip() for result in results] == ["0", "1", "2", "3", "4"]
        finally:
            await shell.close()

    async def test_init_command_with_concurrent_first_calls(self, tmp_path):
        session = BashSession(cwd=tmp_path, init_command="export NOOA_INIT=yes")
        try:
            results = await asyncio.gather(*(session.run("echo $NOOA_INIT") for _ in range(3)))
            assert [stdout for stdout, _err, _code in results] == ["yes"] * 3
            assert session._start_count == 1
        finally:
            await session.close()

    async def test_reset_during_init_fails_start_instead_of_skipping_init(
        self, tmp_path, monkeypatch
    ):
        """A reset inside the init command (e.g. failed timeout recovery) must not
        leave a ready session whose bash never ran the init command."""
        session = BashSession(cwd=tmp_path, init_command="export NOOA_INIT=yes")
        real_send = session._send_and_wait
        calls = 0

        async def send_and_wait(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:  # the first init command: bash is lost and reset
                await session.reset()
            return await real_send(*args, **kwargs)

        monkeypatch.setattr(session, "_send_and_wait", send_and_wait)
        try:
            with pytest.raises(RuntimeError, match="init command"):
                await session.start()
            assert session._ready is False
            stdout, _err, _code = await session.run("echo $NOOA_INIT")
            assert stdout == "yes"
        finally:
            await session.close()


class TestResetDuringStartupDrain:
    async def test_failed_startup_drain_fails_start_instead_of_deadlocking(
        self, tmp_path, monkeypatch
    ):
        """reset() from the startup drain (failed timeout recovery) runs while
        start() holds _start_lock; it must fail the start, not wait for the lock."""
        session = BashSession(cwd=tmp_path)
        real_read = session._read_control_until
        calls = 0

        async def read_control_until(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:  # the startup drain: recovery failed, so it resets
                await session.reset()
            return await real_read(*args, **kwargs)

        monkeypatch.setattr(session, "_read_control_until", read_control_until)
        try:
            with pytest.raises(RuntimeError, match="stopped while starting"):
                await asyncio.wait_for(session.start(), timeout=10)
            assert session._ready is False
            stdout, _err, _code = await session.run("echo ok")
            assert stdout == "ok"
        finally:
            await session.close()


class TestCancelledStart:
    async def test_cancelled_start_closes_its_bash_before_a_retry(self, tmp_path):
        """Cancelling start() during a slow init must not leave that bash running:
        the retry would replace self._process and close() would miss it."""
        session = BashSession(cwd=tmp_path, init_command="sleep 5; export INIT_DONE=1")
        task = asyncio.create_task(session.start())
        await asyncio.sleep(0.5)  # start() is now inside the init command
        first = session._process
        assert first is not None
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert session._process is None and session._ready is False
        await asyncio.wait_for(first.wait(), timeout=10)  # the first bash is gone
        try:
            session._init_command = "export INIT_DONE=1"
            stdout, _err, _code = await session.run("echo $INIT_DONE")
            assert stdout == "1"
        finally:
            await session.close()


class TestStartDuringInit:
    """#447 review finding 2: run()/reset() during a slow init command wait for it."""

    @staticmethod
    def _track_spawns(session, monkeypatch):
        """Record how many bash starts are in flight at once."""
        state = {"active": 0, "max": 0, "spawns": 0}
        original = session._start_unlocked

        async def tracked():
            state["active"] += 1
            state["spawns"] += 1
            state["max"] = max(state["max"], state["active"])
            try:
                await original()
            finally:
                state["active"] -= 1

        monkeypatch.setattr(session, "_start_unlocked", tracked)
        return state

    async def test_runs_during_init_wait_and_see_the_init_env(self, tmp_path, monkeypatch):
        session = BashSession(cwd=tmp_path, init_command="sleep 0.5; export INIT_DONE=1")
        spawns = self._track_spawns(session, monkeypatch)

        async def later(command):
            await asyncio.sleep(0.2)  # start() is now inside the init command
            return await session.run(command)

        try:
            _, first, second = await asyncio.gather(
                session.start(), later("echo a $INIT_DONE"), later("echo b $INIT_DONE")
            )
        finally:
            await session.close()
        assert first == ("a 1", "", 0)
        assert second == ("b 1", "", 0)
        assert spawns["spawns"] == 1

    async def test_reset_during_init_waits_and_spawns_one_bash_at_a_time(
        self, tmp_path, monkeypatch
    ):
        session = BashSession(cwd=tmp_path, init_command="sleep 0.5; export INIT_DONE=1")
        spawns = self._track_spawns(session, monkeypatch)

        async def later_reset():
            await asyncio.sleep(0.2)
            await session.reset()

        try:
            await asyncio.gather(session.start(), later_reset())
            assert await session.run("echo $INIT_DONE") == ("1", "", 0)
        finally:
            await session.close()
        assert spawns["spawns"] == 2  # the first start, then the reset's
        assert spawns["max"] == 1  # never two bash starts at once
