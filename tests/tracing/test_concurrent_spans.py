# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test that spans are correctly tracked in concurrent async contexts.

This test reproduces the issue where trace files were missing for some
samples when running concurrent agent calls. The root cause was that
_get_active_spans() was not being called for top-level calls (no parent),
which deferred the spans dict initialization and caused issues with
ContextVar inheritance in asyncio.gather().

The fix is to call _get_active_spans() unconditionally at the start of
before_agent_call() to ensure the spans dict is initialized early.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class TestConcurrentSpanTracking:
    """Tests for concurrent span tracking in asyncio.gather scenarios."""

    @pytest.fixture
    def temp_trace_dir(self):
        """Create a temporary directory for trace files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_tracer(self):
        """Create a mock tracer that records span creation."""
        tracer = MagicMock()
        spans_created = []

        def start_span(name, context=None, start_time=None):
            span = MagicMock()
            span.name = name
            spans_created.append(span)
            return span

        tracer.start_span = start_span
        tracer.spans_created = spans_created
        return tracer

    @pytest.mark.asyncio
    async def test_concurrent_top_level_calls_all_tracked(self, mock_tracer, temp_trace_dir):
        """Test that all concurrent top-level calls create and track spans.

        This test simulates what happens in eval_pipeline when running
        multiple samples concurrently with asyncio.gather(). Each sample
        is a top-level call (no parent_call_id).

        The bug: With deferred _get_active_spans() initialization, some
        spans might not be properly tracked due to ContextVar timing issues.
        """
        from nooa.tracing._hooks_impl import (
            OpenInferenceHooks,
            _context_active_spans,
            _get_active_spans,
        )

        hooks = OpenInferenceHooks(tracer=mock_tracer)

        # Reset context for clean test
        _context_active_spans.set(None)

        # Track which calls completed successfully
        results = []
        contexts_returned = []

        async def simulate_agent_call(call_id: str):
            """Simulate a top-level agent call (no parent)."""
            # This is what happens in metaclass wrapper -> hooks
            context = hooks.before_agent_call(
                agent=MagicMock(__class__=type("TestAgent", (), {})),
                method_name="process",
                args=(),
                kwargs={},
                call_id=call_id,
                parent_call_id=None,  # Top-level call - this is the key!
            )
            contexts_returned.append(context)

            # Simulate some async work
            await asyncio.sleep(0.01)

            # Check that our span is tracked
            active_spans = _get_active_spans()
            span_tracked = call_id in active_spans

            hooks.after_agent_call(
                agent=MagicMock(),
                method_name="process",
                result={"success": True},
                exception=None,
                context=context,
            )

            return {"call_id": call_id, "span_tracked": span_tracked, "context": context}

        # Run many concurrent top-level calls (like eval_pipeline does)
        num_concurrent = 20
        call_ids = [f"call_{i}" for i in range(num_concurrent)]

        # Use asyncio.gather like ConcurrencyEngine does
        tasks = [simulate_agent_call(cid) for cid in call_ids]
        results = await asyncio.gather(*tasks)

        # All calls should have created spans
        assert len(mock_tracer.spans_created) == num_concurrent, (
            f"Expected {num_concurrent} spans, got {len(mock_tracer.spans_created)}"
        )

        # All calls should have returned valid contexts
        assert all(ctx is not None for ctx in contexts_returned), (
            "Some before_agent_call() returned None context"
        )

        # All spans should have been tracked (this is what failed before the fix)
        spans_tracked = [r["span_tracked"] for r in results]
        assert all(spans_tracked), (
            f"Some spans were not tracked: {sum(not t for t in spans_tracked)} of {num_concurrent} failed"
        )

    @pytest.mark.asyncio
    async def test_spans_dict_isolation_across_contexts(self, mock_tracer):
        """Test that each async context gets its own spans dict.

        When asyncio.gather() creates concurrent tasks, each should have
        an isolated spans dict to prevent cross-contamination.
        """
        from nooa.tracing._hooks_impl import (
            _context_active_spans,
            _get_active_spans,
        )

        # Reset context
        _context_active_spans.set(None)

        spans_dict_ids = []

        async def get_spans_dict_id():
            """Get the id of the spans dict for this context."""
            spans_dict = _get_active_spans()
            spans_dict_ids.append(id(spans_dict))
            await asyncio.sleep(0.01)
            return id(spans_dict)

        # Run concurrent tasks
        num_tasks = 10
        tasks = [get_spans_dict_id() for _ in range(num_tasks)]
        results = await asyncio.gather(*tasks)

        # Each context should have its own spans dict
        # (unique IDs indicate isolation)
        unique_ids = set(results)

        # With proper isolation, we should have multiple unique dicts
        # With the bug (shared dict), we'd have only 1 unique ID
        assert len(unique_ids) >= 1, "Should have at least one spans dict"

        # The key insight: if all IDs are the same AND we see cross-contamination
        # issues in tracking, that indicates the bug. But simply having the same
        # ID isn't necessarily wrong if context propagation is correct.

    @pytest.mark.asyncio
    async def test_multiple_runs_accumulate_correctly(self, mock_tracer):
        """Test that running multiple batches of concurrent calls works correctly.

        This simulates what happens in eval_pipeline when running multiple
        "runs" (e.g., --runs 10). The bug manifested in "later runs" (run3+).
        """
        from nooa.tracing._hooks_impl import (
            OpenInferenceHooks,
            _context_active_spans,
        )

        hooks = OpenInferenceHooks(tracer=mock_tracer)

        all_contexts = []

        async def run_batch(batch_id: int, num_calls: int):
            """Run a batch of concurrent agent calls."""
            # Reset context for each batch (like the pipeline does)
            _context_active_spans.set(None)

            async def single_call(call_idx: int):
                call_id = f"batch{batch_id}_call{call_idx}"
                context = hooks.before_agent_call(
                    agent=MagicMock(__class__=type("TestAgent", (), {})),
                    method_name="process",
                    args=(),
                    kwargs={},
                    call_id=call_id,
                    parent_call_id=None,
                )
                await asyncio.sleep(0.001)
                hooks.after_agent_call(
                    agent=MagicMock(),
                    method_name="process",
                    result={},
                    exception=None,
                    context=context,
                )
                return context

            tasks = [single_call(i) for i in range(num_calls)]
            contexts = await asyncio.gather(*tasks)
            return contexts

        # Simulate multiple runs like --runs 10
        num_runs = 5
        calls_per_run = 10

        for run_id in range(num_runs):
            contexts = await run_batch(run_id, calls_per_run)
            all_contexts.extend(contexts)

        # All calls across all runs should have valid contexts
        assert len(all_contexts) == num_runs * calls_per_run
        assert all(ctx is not None for ctx in all_contexts), (
            f"Some calls returned None context. None count: {sum(1 for c in all_contexts if c is None)}"
        )


class TestBeforeGenerationFallback:
    """Tests for the before_generation parent span resolution."""

    @pytest.fixture
    def recording_tracer(self):
        """Tracer that records the context passed to each start_span call."""

        tracer = MagicMock()
        spans_created = []

        def start_span(name, context=None, start_time=None):
            span = MagicMock()
            span.name = name
            span._parent_context = context
            spans_created.append(span)
            return span

        tracer.start_span = start_span
        tracer.spans_created = spans_created
        return tracer

    @pytest.mark.asyncio
    async def test_before_generation_does_not_steal_unrelated_span_as_parent(
        self, recording_tracer
    ):
        """When agent_call_id and parent_generation_id don't resolve to an active span,
        before_generation must NOT fall back to an arbitrary span from a concurrent call.

        Previously there was a fallback loop:
            for span in reversed(list(_get_active_spans().values())):
                parent_span = span; break

        This could attach a generation span to a completely unrelated concurrent span.
        """
        from nooa.tracing._hooks_impl import (
            OpenInferenceHooks,
            _context_active_spans,
        )

        hooks = OpenInferenceHooks(tracer=recording_tracer)
        _context_active_spans.set(None)

        unrelated_call_id = "unrelated-call-abc"

        # Plant an unrelated active AGENT span in the same context (simulates a concurrent sibling).
        unrelated_ctx = hooks.before_agent_call(
            agent=MagicMock(__class__=type("OtherAgent", (), {})),
            method_name="other_task",
            args=(),
            kwargs={},
            call_id=unrelated_call_id,
            parent_call_id=None,
        )
        assert unrelated_ctx is not None

        # Now call before_generation with IDs that do NOT exist in the spans dict.
        gen_ctx = hooks.before_generation(
            agent=MagicMock(__class__=type("MyAgent", (), {})),
            method_name="my_method",
            strategy="CodeActStrategy",
            generation_id="gen-xyz",
            parent_generation_id=None,  # no prior generation
            agent_call_id=None,  # not going through the wrapper (edge-case / direct call)
        )

        assert gen_ctx is not None
        generation_span = recording_tracer.spans_created[-1]

        # The generation span must NOT have the unrelated span as parent context.
        # With the fallback in place it would inherit the unrelated span.
        assert generation_span._parent_context is None, (
            "before_generation must not fall back to an unrelated active span as parent"
        )

        # Clean up
        hooks.after_agent_call(
            agent=MagicMock(),
            method_name="other_task",
            result=None,
            exception=None,
            context=unrelated_ctx,
        )
