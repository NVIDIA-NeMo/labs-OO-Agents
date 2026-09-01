# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for session isolation between concurrent async contexts.

Verifies that set_session() / get_session() are isolated between concurrent
async contexts, preventing cross-context leakage during parallel execution.
"""

import asyncio

import pytest

from nooa.tracing._session import get_session, set_session


class TestSessionIsolation:
    """Test that session ID is isolated between async contexts."""

    @pytest.mark.asyncio
    async def test_concurrent_session_isolation(self):
        """Each async context should see its own session ID.

        Simulates parallel eval samples each calling set_session() with
        different IDs. Without proper ContextVar isolation, they would
        overwrite each other's session.
        """
        results = {}
        all_started = asyncio.Event()
        start_count = 0

        async def set_and_check(context_name: str, session_id: str):
            nonlocal start_count

            set_session(session_id)
            start_count += 1
            if start_count == 3:
                all_started.set()

            await all_started.wait()
            await asyncio.sleep(0.01)

            results[context_name] = get_session()

        await asyncio.gather(
            set_and_check("ctx_a", "session-a"),
            set_and_check("ctx_b", "session-b"),
            set_and_check("ctx_c", "session-c"),
        )

        # Each context should see its own session
        assert results["ctx_a"] == "session-a"
        assert results["ctx_b"] == "session-b"
        assert results["ctx_c"] == "session-c"

    @pytest.mark.asyncio
    async def test_same_context_sees_its_own_session(self):
        """Within the same async context, session should be consistent."""
        set_session("my-session")

        assert get_session() == "my-session"

        await asyncio.sleep(0.01)

        assert get_session() == "my-session"

    @pytest.mark.asyncio
    async def test_session_none_by_default(self):
        """Session should be None before any set_session() call."""
        assert get_session() is None

    @pytest.mark.asyncio
    async def test_set_session_none_clears(self):
        """set_session(None) should clear the session."""
        set_session("some-session")
        assert get_session() == "some-session"

        set_session(None)
        assert get_session() is None


class TestSessionScope:
    def test_restores_nested_scopes_after_exception(self):
        from nooa.tracing._session import session_scope

        set_session("original")
        try:
            with pytest.raises(RuntimeError, match="boom"):
                with session_scope("outer"):
                    assert get_session() == "outer"
                    with session_scope("inner"):
                        assert get_session() == "inner"
                        raise RuntimeError("boom")

            assert get_session() == "original"
        finally:
            set_session(None)

    def test_none_temporarily_clears_session(self):
        from nooa.tracing._session import session_scope

        set_session("original")
        with session_scope(None):
            assert get_session() is None
        assert get_session() == "original"

    @pytest.mark.asyncio
    async def test_isolated_between_concurrent_tasks(self):
        from nooa.tracing._session import session_scope

        ready = asyncio.Event()
        entered = 0

        async def observe(session_id: str):
            nonlocal entered
            with session_scope(session_id):
                entered += 1
                if entered == 2:
                    ready.set()
                await ready.wait()
                before = get_session()
                await asyncio.sleep(0)
                return before, get_session()

        first, second = await asyncio.gather(observe("first"), observe("second"))
        assert first == ("first", "first")
        assert second == ("second", "second")
