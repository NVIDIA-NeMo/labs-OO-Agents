# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for stack isolation between concurrent async contexts.

These tests verify that _agent_call_stack and _generation_id_stack are isolated
between concurrent async contexts, preventing context leakage during parallel execution.

The tests use the copy-on-write helper functions (_push_*, _pop_*) which are the
correct API for managing stacks in async contexts. Direct mutation of the stack
lists would cause the parallel isolation to break.
"""

import asyncio

import pytest

from nooa.runtime.actor import (
    _pop_generation_id,
    _push_generation_id,
)
from nooa.runtime.context_vars import (
    _pop_agent_call_id,
    _push_agent_call_id,
)


class TestStackIsolation:
    """Test that runtime stacks are isolated between async contexts."""

    @pytest.mark.asyncio
    async def test_generation_id_stack_isolation(self) -> None:
        """Generation ID stacks should be isolated between concurrent async contexts.

        This test simulates parallel agent method calls that each push to the
        generation_id_stack. Without proper isolation, they would see each
        other's generation IDs.
        """
        from nooa.runtime.actor import ActorRuntime

        # Create a mock agent
        class MockAgent:
            def __init__(self) -> None:
                self.event_manager = None

        agent = MockAgent()
        runtime = ActorRuntime(agent)

        results: dict[str, dict[str, object]] = {}
        context_a_pushed = asyncio.Event()
        context_b_checked = asyncio.Event()

        async def context_a() -> None:
            # Push a generation ID using copy-on-write helper
            _push_generation_id("gen_id_from_context_a")

            # Signal that we pushed
            context_a_pushed.set()

            # Wait for context B to check
            await context_b_checked.wait()

            # Record what we see
            results["ctx_a"] = {
                "stack": list(runtime._generation_id_stack),
                "top": runtime._generation_id_stack[-1] if runtime._generation_id_stack else None,
            }

            # Clean up using copy-on-write helper
            _pop_generation_id()

        async def context_b() -> None:
            # Wait for context A to push
            await context_a_pushed.wait()

            # Now check what we see - we should NOT see context A's generation ID
            # if isolation is working
            results["ctx_b"] = {
                "stack": list(runtime._generation_id_stack),
                "top": runtime._generation_id_stack[-1] if runtime._generation_id_stack else None,
                "sees_context_a": "gen_id_from_context_a" in runtime._generation_id_stack,
            }

            # Signal we're done checking
            context_b_checked.set()

        await asyncio.gather(context_a(), context_b())

        # With proper isolation, context B should NOT see context A's generation ID
        assert not results["ctx_b"]["sees_context_a"], (
            f"Context B saw context A's generation ID! "
            f"Stack in context B: {results['ctx_b']['stack']}. "
            "This indicates context leakage."
        )

    @pytest.mark.asyncio
    async def test_agent_call_stack_isolation(self) -> None:
        """Agent call stacks should be isolated between concurrent async contexts.

        This test simulates parallel agent method calls that each push to the
        agent_call_stack. Without proper isolation, they would see each
        other's call IDs.
        """
        from nooa.runtime.actor import ActorRuntime

        # Create a mock agent
        class MockAgent:
            def __init__(self) -> None:
                self.event_manager = None

        agent = MockAgent()
        runtime = ActorRuntime(agent)

        results: dict[str, dict[str, object]] = {}
        context_a_pushed = asyncio.Event()
        context_b_checked = asyncio.Event()

        async def context_a() -> None:
            # Push a call ID using copy-on-write helper
            _push_agent_call_id("call_id_from_context_a")

            # Signal that we pushed
            context_a_pushed.set()

            # Wait for context B to check
            await context_b_checked.wait()

            # Record what we see
            results["ctx_a"] = {
                "stack": list(runtime._agent_call_stack),
                "agent_call_id": runtime._agent_call_id,
            }

            # Clean up using copy-on-write helper
            _pop_agent_call_id()

        async def context_b() -> None:
            # Wait for context A to push
            await context_a_pushed.wait()

            # Now check what we see - we should NOT see context A's call ID
            # if isolation is working
            results["ctx_b"] = {
                "stack": list(runtime._agent_call_stack),
                "agent_call_id": runtime._agent_call_id,
                "sees_context_a": "call_id_from_context_a" in runtime._agent_call_stack,
            }

            # Signal we're done checking
            context_b_checked.set()

        await asyncio.gather(context_a(), context_b())

        # With proper isolation, context B should NOT see context A's call ID
        assert not results["ctx_b"]["sees_context_a"], (
            f"Context B saw context A's call ID! "
            f"Stack in context B: {results['ctx_b']['stack']}. "
            "This indicates context leakage."
        )

    @pytest.mark.asyncio
    async def test_same_context_sees_its_own_stack(self) -> None:
        """Within the same async context, stack operations should be consistent."""
        from nooa.runtime.actor import ActorRuntime

        # Create a mock agent
        class MockAgent:
            def __init__(self) -> None:
                self.event_manager = None

        agent = MockAgent()
        runtime = ActorRuntime(agent)

        # Push some IDs using copy-on-write helpers
        _push_generation_id("gen_1")
        _push_agent_call_id("call_1")

        # Should see our own IDs
        assert runtime._generation_id_stack[-1] == "gen_1"
        assert runtime._agent_call_id == "call_1"

        # Even after await, should still see our IDs
        await asyncio.sleep(0.01)

        assert runtime._generation_id_stack[-1] == "gen_1"
        assert runtime._agent_call_id == "call_1"

        # Clean up using copy-on-write helpers
        _pop_generation_id()
        _pop_agent_call_id()

    @pytest.mark.asyncio
    async def test_multiple_concurrent_contexts_isolated(self) -> None:
        """Multiple concurrent contexts should each have their own stack tracking."""
        from nooa.runtime.actor import ActorRuntime

        # Create a mock agent
        class MockAgent:
            def __init__(self) -> None:
                self.event_manager = None

        agent = MockAgent()
        runtime = ActorRuntime(agent)

        results: dict[str, dict[str, object]] = {}
        all_started = asyncio.Event()
        start_count = 0

        async def push_and_check(context_name: str, gen_id: str, call_id: str) -> None:
            nonlocal start_count

            # Push our IDs using copy-on-write helpers
            _push_generation_id(gen_id)
            _push_agent_call_id(call_id)

            start_count += 1
            if start_count == 3:
                all_started.set()

            # Wait for all contexts to push
            await all_started.wait()

            # Small delay to allow race conditions to manifest
            await asyncio.sleep(0.01)

            # Check what we see
            gen_top = runtime._generation_id_stack[-1] if runtime._generation_id_stack else None
            call_top = runtime._agent_call_id

            results[context_name] = {
                "expected_gen": gen_id,
                "actual_gen": gen_top,
                "expected_call": call_id,
                "actual_call": call_top,
                "gen_is_own": gen_top == gen_id,
                "call_is_own": call_top == call_id,
            }

            # Clean up using copy-on-write helpers
            _pop_generation_id()
            _pop_agent_call_id()

        # Run 3 contexts concurrently
        await asyncio.gather(
            push_and_check("ctx_a", "gen_a", "call_a"),
            push_and_check("ctx_b", "gen_b", "call_b"),
            push_and_check("ctx_c", "gen_c", "call_c"),
        )

        # With proper isolation, each context should only see its own IDs
        for ctx_name, result in results.items():
            if result["actual_gen"] is not None:
                assert result["gen_is_own"], (
                    f"{ctx_name} expected generation ID '{result['expected_gen']}' "
                    f"but saw '{result['actual_gen']}'! This indicates context leakage."
                )
            if result["actual_call"] is not None:
                assert result["call_is_own"], (
                    f"{ctx_name} expected call ID '{result['expected_call']}' "
                    f"but saw '{result['actual_call']}'! This indicates context leakage."
                )
