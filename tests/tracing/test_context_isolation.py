# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for async context isolation in OpenInferenceHooks.

These tests verify that span tracking is isolated between concurrent async contexts,
preventing context leakage during parallel execution (e.g., parallel eval samples).
"""

import asyncio

import pytest
from opentelemetry.sdk.trace import TracerProvider

from nooa.tracing._hooks_impl import OpenInferenceHooks


class MockAgent:
    """Mock agent for testing hooks."""

    pass


@pytest.fixture
def hooks():
    """Create a fresh OpenInferenceHooks instance for testing."""
    provider = TracerProvider()
    tracer = provider.get_tracer("test")
    return OpenInferenceHooks(tracer)


class TestContextIsolation:
    """Test that active spans are isolated between async contexts."""

    @pytest.mark.asyncio
    async def test_concurrent_contexts_do_not_share_spans(self, hooks):
        """Spans created in one async context should not be visible in another.

        This test simulates the bug where parallel eval samples could pick up
        spans from other concurrent contexts due to shared _active_spans dict.
        """
        agent = MockAgent()
        spans_seen_by_context_b = []
        context_a_started = asyncio.Event()
        context_b_checked = asyncio.Event()

        async def context_a():
            """Create a span and hold it open while context B checks."""
            # Start a generation span
            ctx = hooks.before_generation(
                agent=agent,
                method_name="test_method",
                strategy="test",
                generation_id="context_a_gen_id",
                parent_generation_id=None,
            )

            # Signal that span is created
            context_a_started.set()

            # Wait for context B to check
            await context_b_checked.wait()

            # Clean up
            hooks.after_generation(
                agent=agent,
                method_name="test_method",
                result=None,
                exception=None,
                context=ctx,
                generation_id="context_a_gen_id",
            )

        async def context_b():
            """Check what spans are visible from this context."""
            # Wait for context A to create its span
            await context_a_started.wait()

            # Now try to create a span - the "fallback to most recent" logic
            # should NOT see context_a's span if isolation is working
            ctx = hooks.before_code_execution(
                agent=agent,
                code="print('test')",
                execution_id="context_b_exec_id",
                generation_id=None,  # No generation_id, so it uses fallback logic
            )

            # Check if the span has a parent (it shouldn't if isolated)
            span = ctx.get("span")
            if span and span.parent:
                # Record what parent was found - this indicates leakage!
                spans_seen_by_context_b.append(span.parent.span_id)

            # Signal we're done checking
            context_b_checked.set()

            # Clean up
            hooks.after_code_execution(
                agent=agent,
                code="print('test')",
                result=None,
                exception=None,
                context=ctx,
                execution_id="context_b_exec_id",
            )

        # Run both contexts concurrently
        await asyncio.gather(context_a(), context_b())

        # With proper isolation, context B should NOT see context A's span
        # If the test fails here, it means context leakage is occurring
        assert len(spans_seen_by_context_b) == 0, (
            f"Context B saw spans from context A! "
            f"This indicates context leakage. Span IDs seen: {spans_seen_by_context_b}"
        )

    @pytest.mark.asyncio
    async def test_same_context_sees_its_own_spans(self, hooks):
        """Spans should be visible within the same async context."""
        agent = MockAgent()

        # Create a generation span
        gen_ctx = hooks.before_generation(
            agent=agent,
            method_name="test_method",
            strategy="test",
            generation_id="my_gen_id",
            parent_generation_id=None,
        )

        # Create an execution span in the same context - it should find the generation
        exec_ctx = hooks.before_code_execution(
            agent=agent,
            code="print('test')",
            execution_id="my_exec_id",
            generation_id="my_gen_id",  # Explicitly pass generation_id
        )

        # The execution span should have found the generation span as parent
        exec_span = exec_ctx.get("span")
        gen_span = gen_ctx.get("span")

        assert exec_span is not None
        assert gen_span is not None
        # The execution span should be parented to the generation span
        assert exec_span.parent is not None
        assert exec_span.parent.span_id == gen_span.get_span_context().span_id

        # Clean up
        hooks.after_code_execution(
            agent=agent,
            code="print('test')",
            result=None,
            exception=None,
            context=exec_ctx,
            execution_id="my_exec_id",
        )
        hooks.after_generation(
            agent=agent,
            method_name="test_method",
            result=None,
            exception=None,
            context=gen_ctx,
            generation_id="my_gen_id",
        )

    @pytest.mark.asyncio
    async def test_multiple_concurrent_contexts_isolated(self, hooks):
        """Multiple concurrent contexts should each have their own span tracking."""
        agent = MockAgent()
        results = {}
        all_started = asyncio.Event()
        start_count = 0

        async def create_spans(context_name: str, gen_id: str):
            nonlocal start_count

            # Create a generation span
            gen_ctx = hooks.before_generation(
                agent=agent,
                method_name="test",
                strategy="test",
                generation_id=gen_id,
                parent_generation_id=None,
            )

            start_count += 1
            if start_count == 3:
                all_started.set()

            # Wait for all contexts to start
            await all_started.wait()

            # Try to create a code execution with fallback parent lookup
            exec_ctx = hooks.before_code_execution(
                agent=agent,
                code="test",
                execution_id=f"{context_name}_exec",
                generation_id=None,  # Use fallback logic
            )

            exec_span = exec_ctx.get("span")
            gen_span = gen_ctx.get("span")

            # Record if we found a parent and if it's OUR generation span
            parent_is_correct = False
            if exec_span and exec_span.parent:
                parent_is_correct = exec_span.parent.span_id == gen_span.get_span_context().span_id

            results[context_name] = {
                "has_parent": exec_span.parent is not None if exec_span else False,
                "parent_is_own": parent_is_correct,
            }

            # Clean up
            hooks.after_code_execution(
                agent=agent,
                code="test",
                result=None,
                exception=None,
                context=exec_ctx,
                execution_id=f"{context_name}_exec",
            )
            hooks.after_generation(
                agent=agent,
                method_name="test",
                result=None,
                exception=None,
                context=gen_ctx,
                generation_id=gen_id,
            )

        # Run 3 contexts concurrently
        await asyncio.gather(
            create_spans("ctx_a", "gen_a"),
            create_spans("ctx_b", "gen_b"),
            create_spans("ctx_c", "gen_c"),
        )

        # With proper isolation:
        # - Each context either has no parent (isolated, no fallback found)
        # - Or if it has a parent, it should be its OWN generation span
        for ctx_name, result in results.items():
            if result["has_parent"]:
                assert result["parent_is_own"], (
                    f"{ctx_name} found a parent span from a different context! This indicates context leakage."
                )
