# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression test: hooks must be registered in every run_until_complete Task.

The eval pipeline's persistent subprocess worker runs one asyncio Task per
evaluation task via ``loop.run_until_complete(run_task(...))``.  Each call
creates a new Task whose context is a copy of the *main thread's* context —
which never has hooks set.

Before the fix, ``enable_tracing()`` only called ``set_hooks()`` in the
first-time-setup path.  Every task after the first started with
``get_hooks() == None``, so no AGENT/GENERATION spans were emitted.

After the fix, the "already enabled" path calls ``_re_register_hooks()`` which
re-sets the hooks ContextVar in the current task context.
"""

import asyncio
import tempfile


class TestHooksRegisteredPerTask:
    """Verify hooks are available in every run_until_complete Task, not just the first."""

    def test_task1_has_hooks(self):
        """Sanity: the very first task always has hooks (first-time-setup path)."""
        from nooa.runtime.hooks import get_hooks
        from nooa.tracing import enable_tracing, exporters

        result = {}

        async def task_coro(tmpdir):
            enable_tracing(exporters=[exporters.jsonl(tmpdir)])
            result["hooks"] = get_hooks()

        loop = asyncio.new_event_loop()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                loop.run_until_complete(task_coro(tmpdir))
        finally:
            loop.close()

        assert result["hooks"] is not None, "Task 1 should have hooks (first-time setup)"

    def test_task2_has_hooks(self):
        """Task 2+ in a persistent worker must also have hooks (regression: they didn't).

        Before the fix: enable_tracing() for task 2 went to the "already enabled"
        fast-path and returned without calling set_hooks().  The asyncio Task
        inherited the main thread's context (hooks=None), so get_hooks() returned
        None and no AGENT/GENERATION spans were emitted.
        """
        from nooa.runtime.hooks import get_hooks
        from nooa.tracing import enable_tracing, exporters

        task_hooks = {}

        async def task_coro(tmpdir, task_num):
            # Mirrors subprocess_worker.py: enable_tracing() is called at the
            # start of every run_task(), even though the provider is already set up.
            enable_tracing(exporters=[exporters.jsonl(tmpdir)])
            task_hooks[task_num] = get_hooks()

        loop = asyncio.new_event_loop()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                loop.run_until_complete(task_coro(tmpdir, 1))
                loop.run_until_complete(task_coro(tmpdir, 2))
                loop.run_until_complete(task_coro(tmpdir, 3))
        finally:
            loop.close()

        assert task_hooks[1] is not None, "Task 1 must have hooks"
        assert task_hooks[2] is not None, "Task 2 must have hooks (was None before fix)"
        assert task_hooks[3] is not None, "Task 3 must have hooks (was None before fix)"

    def test_main_thread_context_unchanged(self):
        """set_hooks() inside a Task must not leak to the main thread's context.

        Each Task's ContextVar mutation is local to that Task.  This test ensures
        the fix doesn't accidentally make hooks visible outside tasks.
        """
        from nooa.runtime.hooks import get_hooks
        from nooa.tracing import enable_tracing, exporters

        # Main thread: hooks should be None before any Task runs
        assert get_hooks() is None

        async def task_coro(tmpdir):
            enable_tracing(exporters=[exporters.jsonl(tmpdir)])

        loop = asyncio.new_event_loop()
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                loop.run_until_complete(task_coro(tmpdir))
        finally:
            loop.close()

        # Main thread: still None — ContextVar mutations in Tasks don't propagate back
        assert get_hooks() is None, (
            "Hooks set inside a Task must not leak into the main thread's context"
        )
