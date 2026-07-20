# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test BashSession resilience to event loop recreation (gl#212).

Reproduces the scenario where the TUI agent loop dies and is recreated,
leaving BashSession with stale subprocess pipes bound to the dead loop.
"""

import asyncio
import tempfile
import threading

import pytest

from nooa.tools._bash_session import BashSession

pytestmark = pytest.mark.asyncio


async def test_bash_session_survives_loop_change():
    """BashSession should auto-recover when the event loop changes.

    Simulates the TUI scenario:
    1. Start BashSession on loop A (agent loop)
    2. Kill loop A (dispatcher crash / Esc interrupt)
    3. Use BashSession on loop B (new agent loop)
    4. Should work (auto-reset) instead of raising RuntimeError
    """
    cwd = tempfile.gettempdir()

    # Phase 1: start and use the session on a background loop (simulates agent loop A)
    session = BashSession(cwd=cwd)
    loop_a_result = None
    loop_a_error = None
    loop_a_ready = threading.Event()
    loop_a_stop = threading.Event()

    def run_loop_a():
        nonlocal loop_a_result, loop_a_error
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def use_session():
            nonlocal loop_a_result
            stdout, stderr, code = await session.run("echo loop_a_ok")
            loop_a_result = stdout

        try:
            loop.run_until_complete(use_session())
        except Exception as e:
            loop_a_error = str(e)
        finally:
            loop_a_ready.set()
            # Wait for signal to stop (simulates dispatcher exit)
            loop_a_stop.wait(timeout=5)
            loop.close()

    thread_a = threading.Thread(target=run_loop_a, daemon=True)
    thread_a.start()
    loop_a_ready.wait(timeout=5)

    assert loop_a_error is None, f"Loop A failed: {loop_a_error}"
    assert loop_a_result == "loop_a_ok"
    assert session._started is True

    # Phase 2: kill loop A (simulates dispatcher crash → loop stops → thread exits)
    loop_a_stop.set()
    thread_a.join(timeout=5)

    # Phase 3: use session on loop B (simulates new agent loop after recreation)
    loop_b_result = None
    loop_b_error = None
    loop_b_ready = threading.Event()

    def run_loop_b():
        nonlocal loop_b_result, loop_b_error
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def use_session_again():
            nonlocal loop_b_result
            stdout, stderr, code = await session.run("echo loop_b_ok")
            loop_b_result = stdout

        try:
            loop.run_until_complete(use_session_again())
        except Exception as e:
            loop_b_error = str(e)
        finally:
            loop_b_ready.set()
            # Cleanup
            try:
                loop.run_until_complete(session.close())
            except Exception:
                pass
            loop.close()

    thread_b = threading.Thread(target=run_loop_b, daemon=True)
    thread_b.start()
    loop_b_ready.wait(timeout=10)

    # This is the assertion that currently FAILS (RuntimeError: Future attached to different loop)
    assert loop_b_error is None, f"Loop B failed (cross-loop bug): {loop_b_error}"
    assert loop_b_result == "loop_b_ok"


async def test_bash_session_works_same_loop():
    """Baseline: BashSession works fine when used on the same loop throughout."""
    session = BashSession(cwd=tempfile.gettempdir())
    try:
        stdout, stderr, code = await session.run("echo hello")
        assert stdout == "hello"
        assert code == 0

        stdout2, stderr2, code2 = await session.run("echo world")
        assert stdout2 == "world"
        assert code2 == 0
    finally:
        await session.close()
