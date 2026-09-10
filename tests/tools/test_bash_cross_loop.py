# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for cross-loop Lock bug in BashSession (gl-212).

Reproduces the failure where BashSession's internal asyncio.Lock gets bound
to one event loop via contention, then fails on a different loop. This mirrors
the TUI architecture where the agent loop can be recreated after crashes.

The bug triggers specifically when:
1. Concurrent run() calls cause Lock contention (_get_loop() binds the Lock)
2. The event loop is closed/recreated
3. Subsequent contended Lock use on the new loop raises RuntimeError
"""

import asyncio
import threading

from nooa.tools._bash_session import BashSession
from nooa.tools.shell_tools import ShellTools


class TestCrossLoopLockContention:
    """Reproduce: Lock contends on loop A, then contends on loop B -> RuntimeError."""

    async def test_lock_survives_loop_restart_under_contention(self, tmp_path):
        """BashSession Lock contends on loop A, loop closed, contends on loop B.

        This is the exact TUI failure mode: agent loop processes concurrent
        shell commands, then crashes/restarts, then processes concurrent
        commands again on a new loop.
        """
        session = BashSession(cwd=tmp_path)

        # Phase 1: contend the Lock on loop A (simulates first agent session)
        phase1_done = threading.Event()
        phase1_error = [None]

        def agent_loop_A():
            loopA = asyncio.new_event_loop()
            asyncio.set_event_loop(loopA)
            try:

                async def concurrent_commands():
                    await session.start()
                    results = await asyncio.gather(
                        session.run("echo first"),
                        session.run("echo second"),
                    )
                    return results

                results = loopA.run_until_complete(concurrent_commands())
                outputs = {r[0] for r in results}
                assert "first" in outputs
                assert "second" in outputs
            except Exception as e:
                phase1_error[0] = e
            finally:
                loopA.close()
                phase1_done.set()

        t1 = threading.Thread(target=agent_loop_A)
        t1.start()
        timed_out = not phase1_done.wait(timeout=15)
        t1.join(timeout=2)

        assert not timed_out, "Phase 1 thread timed out (> 15 s)"
        assert not t1.is_alive(), "Phase 1 thread still running after join"
        assert phase1_error[0] is None, f"Phase 1 failed: {phase1_error[0]}"

        # Phase 2: use on loop B (simulates agent loop restart after crash)
        phase2_done = threading.Event()
        phase2_error = [None]

        def agent_loop_B():
            loopB = asyncio.new_event_loop()
            asyncio.set_event_loop(loopB)
            try:

                async def concurrent_commands():
                    results = await asyncio.gather(
                        session.run("echo third"),
                        session.run("echo fourth"),
                    )
                    return results

                results = loopB.run_until_complete(concurrent_commands())
                outputs = {r[0] for r in results}
                assert "third" in outputs
                assert "fourth" in outputs
            except Exception as e:
                phase2_error[0] = e
            finally:
                loopB.close()
                phase2_done.set()

        t2 = threading.Thread(target=agent_loop_B)
        t2.start()
        timed_out = not phase2_done.wait(timeout=15)
        t2.join(timeout=2)

        assert not timed_out, "Phase 2 thread timed out (> 15 s)"
        assert not t2.is_alive(), "Phase 2 thread still running after join"
        # This is the actual assertion — without the fix, RuntimeError is raised:
        # "asyncio.locks.Lock ... is bound to a different event loop"
        assert phase2_error[0] is None, f"Cross-loop contention raised: {phase2_error[0]}"

    async def test_single_loop_contention_still_works(self, tmp_path):
        """Verify concurrent run() on a single loop doesn't regress."""
        session = BashSession(cwd=tmp_path)
        await session.start()
        try:
            for i in range(3):
                results = await asyncio.gather(
                    session.run(f"echo round{i}_a"),
                    session.run(f"echo round{i}_b"),
                )
                assert results[0][2] == 0
                assert results[1][2] == 0
        finally:
            await session.close()

    async def test_loop_change_reset_keeps_concurrent_runs_serialized(self, tmp_path, monkeypatch):
        """A loop-change reset must not replace a lock held by run()."""
        session = BashSession(cwd=tmp_path)
        session._started = True
        session._started_on_loop = object()

        restart_started = asyncio.Event()
        allow_restart = asyncio.Event()
        command_started = asyncio.Event()
        allow_command = asyncio.Event()

        async def start():
            session._started = True
            session._started_on_loop = asyncio.get_running_loop()
            restart_started.set()
            await allow_restart.wait()

        async def send_and_wait(_script, _sentinel, timeout):
            command_started.set()
            await allow_command.wait()
            return ["0", str(tmp_path)], "", "", False

        monkeypatch.setattr(session, "start", start)
        monkeypatch.setattr(session, "_send_and_wait", send_and_wait)

        first = asyncio.create_task(session.run("echo first"))
        await asyncio.wait_for(restart_started.wait(), timeout=1)
        second = asyncio.create_task(session.run("echo second"))

        entered_during_reset = False
        try:
            await asyncio.sleep(0)
            entered_during_reset = command_started.is_set()
        finally:
            allow_restart.set()
            allow_command.set()
            results = await asyncio.wait_for(
                asyncio.gather(first, second, return_exceptions=True), timeout=1
            )

        assert not entered_during_reset
        assert results == [("", "", 0), ("", "", 0)]

    async def test_shell_tools_survives_loop_restart(self, tmp_path):
        """ShellTools concurrent usage survives loop restart."""
        shell = ShellTools(cwd=tmp_path)

        # Phase 1: concurrent usage on loop A
        phase1_done = threading.Event()
        phase1_error = [None]

        def loop_A():
            loopA = asyncio.new_event_loop()
            asyncio.set_event_loop(loopA)
            try:

                async def concurrent():
                    r1, r2 = await asyncio.gather(
                        shell.run("echo a1"),
                        shell.run("echo a2"),
                    )
                    assert r1.returncode == 0
                    assert r2.returncode == 0

                loopA.run_until_complete(concurrent())
            except Exception as e:
                phase1_error[0] = e
            finally:
                loopA.close()
                phase1_done.set()

        t1 = threading.Thread(target=loop_A)
        t1.start()
        timed_out = not phase1_done.wait(timeout=15)
        t1.join(timeout=2)

        assert not timed_out, "Phase 1 thread timed out (> 15 s)"
        assert not t1.is_alive(), "Phase 1 thread still running after join"
        assert phase1_error[0] is None, f"Phase 1: {phase1_error[0]}"

        # Phase 2: concurrent usage on loop B
        phase2_done = threading.Event()
        phase2_error = [None]

        def loop_B():
            loopB = asyncio.new_event_loop()
            asyncio.set_event_loop(loopB)
            try:

                async def concurrent():
                    r1, r2 = await asyncio.gather(
                        shell.run("echo b1"),
                        shell.run("echo b2"),
                    )
                    assert r1.returncode == 0
                    assert r2.returncode == 0

                loopB.run_until_complete(concurrent())
            except Exception as e:
                phase2_error[0] = e
            finally:
                loopB.close()
                phase2_done.set()

        t2 = threading.Thread(target=loop_B)
        t2.start()
        timed_out = not phase2_done.wait(timeout=15)
        t2.join(timeout=2)

        assert not timed_out, "Phase 2 thread timed out (> 15 s)"
        assert not t2.is_alive(), "Phase 2 thread still running after join"
        assert phase2_error[0] is None, f"Phase 2: {phase2_error[0]}"
