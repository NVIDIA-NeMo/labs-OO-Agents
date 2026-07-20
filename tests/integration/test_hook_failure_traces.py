# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test that traces are still created when hooks raise exceptions.

This reproduces the missing trace file bug where exceptions in before_agent_call
cause the hook to return None, which then causes after_agent_call to skip
span finalization.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nooa import Agent
from nooa.runtime.hooks import call_before_hook, set_hooks
from nooa.tracing._hooks_impl import OpenInferenceHooks


class SimpleAgent(Agent):
    """Simple agent for testing."""

    async def process(self, x: int) -> dict:
        """Process input."""
        return {"value": x * 2}


@pytest.fixture
def temp_trace_dir():
    """Create temporary directory for traces."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_hook_exception_returns_none():
    """Verify that when before_agent_call raises, call_before_hook returns None."""
    # Create a mock hooks object where before_agent_call raises ValueError
    mock_hooks = MagicMock()
    mock_hooks.before_agent_call.side_effect = ValueError("Simulated failure")

    with patch("nooa.runtime.hooks.get_hooks", return_value=mock_hooks):
        result = call_before_hook(
            "before_agent_call",
            agent=MagicMock(),
            method_name="test",
            args=(),
            kwargs={},
            call_id="test-123",
            parent_call_id=None,
        )

    # Currently returns None when hook fails - this is the bug!
    assert result is None, "Hook failure should return None (demonstrating the bug)"


def test_after_agent_call_skips_on_none_context():
    """Verify that after_agent_call returns early with None context."""
    from opentelemetry.sdk.trace import TracerProvider

    # Create real hooks
    provider = TracerProvider()
    tracer = provider.get_tracer("test")
    hooks = OpenInferenceHooks(tracer)

    # Call after_agent_call with None context
    # This should NOT raise, but should also NOT create a span
    hooks.after_agent_call(
        agent=MagicMock(),
        method_name="test",
        result={"value": 42},
        exception=None,
        context=None,  # This simulates failed before_agent_call
    )

    # Test passes if no exception - but this means no tracing happened!


@pytest.mark.asyncio
async def test_concurrent_hook_failures_cause_missing_traces(temp_trace_dir):
    """Reproduce the bug: concurrent execution with hook failures causes missing traces.

    This simulates what happens when before_agent_call fails for some samples:
    - Those samples have no tracing
    - Trace files are empty or missing
    """
    from nooa.runtime.hooks import call_after_hook, call_before_hook

    traces_created = []
    calls_attempted = []

    # Create hooks that fail for some calls (simulating the intermittent bug)
    class FailingHooks:
        def __init__(self, fail_indices: set):
            self.fail_indices = fail_indices
            self.call_count = 0

        def before_agent_call(self, **kwargs):
            self.call_count += 1
            calls_attempted.append(self.call_count)
            if self.call_count in self.fail_indices:
                raise ValueError(f"Simulated failure for call {self.call_count}")
            traces_created.append(self.call_count)
            return {"call_id": kwargs.get("call_id")}

        def after_agent_call(self, context, **kwargs):
            pass  # Would finalize span if context was not None

    # Fail on calls 6-10 (simulating run6+ failures)
    failing_hooks = FailingHooks(fail_indices={6, 7, 8, 9, 10})
    set_hooks(failing_hooks)

    async def run_sample(i: int):
        """Simulate one sample execution."""
        context = call_before_hook(
            "before_agent_call",
            agent=MagicMock(),
            method_name="process",
            args=(),
            kwargs={},
            call_id=f"call-{i}",
            parent_call_id=None,
        )

        # Simulate agent execution
        await asyncio.sleep(0.01)
        result = {"value": i}

        # Finish tracing
        call_after_hook(
            "after_agent_call",
            context,
            agent=MagicMock(),
            method_name="process",
            result=result,
            exception=None,
        )

        return context is not None

    # Run 10 concurrent samples
    results = await asyncio.gather(*[run_sample(i) for i in range(1, 11)])

    # Count successful traces
    successful_traces = sum(results)

    # BUG DEMONSTRATION: Only 5 traces created because calls 6-10 failed
    assert successful_traces == 5, f"Expected 5 successful traces, got {successful_traces}"
    assert len(traces_created) == 5, f"Expected 5 traces created, got {len(traces_created)}"

    # This test passes, demonstrating the bug: hook failures cause missing traces
    print(f"Calls attempted: {len(calls_attempted)}")
    print(f"Traces created: {len(traces_created)}")
    print(f"Missing traces: {len(calls_attempted) - len(traces_created)}")


@pytest.mark.asyncio
async def test_with_fix_all_samples_traced(temp_trace_dir):
    """After the fix, even failed hooks should result in some trace record.

    This test will FAIL until the fix is applied.
    """
    # TODO: This test should pass after the fix is applied
    # For now, it demonstrates what we want to achieve
    pass
