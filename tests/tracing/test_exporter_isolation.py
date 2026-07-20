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
