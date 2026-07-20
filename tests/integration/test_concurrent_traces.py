# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for concurrent trace file creation bug.

Reproduces the issue where some concurrent agent calls don't create trace files.
Uses a fake LLM to avoid real API calls while testing the full execution path.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from nooa import Agent
from nooa.runtime.hooks import get_hooks, set_hooks
from nooa.tracing import (
    get_session,
    set_session,
)
from nooa.tracing._hooks_impl import (
    OpenInferenceHooks,
    _get_active_spans,
)
from nooa.unifiedllm import LLMResponse, ToolCall


class FakeLLM:
    """Fake LLM that returns a proper execute_python tool call.

    Text-only responses (content=..., tool_calls=None) are converted by
    CodeActStrategy into synthetic comment calls — they never execute
    code. To actually terminate the CodeAct loop we must return a real
    LLMResponse with finish_reason="tool_calls" and an execute_python tool
    call whose body calls return_result().
    """

    def __init__(self):
        self.call_count = 0

    async def acall(self, messages, tools=None, **kwargs):
        """Return an execute_python tool call that calls return_result()."""
        self.call_count += 1

        # Simulate small delay like real LLM
        await asyncio.sleep(0.01)

        # Return a proper tool-call response so CodeAct executes the code
        # and terminates via return_result() on the first iteration.
        return LLMResponse(
            raw_response=None,
            content="",
            tool_calls=[
                ToolCall(
                    id=f"fake_{self.call_count}",
                    name="execute_python",
                    arguments=json.dumps(
                        {"code": 'return_result({"value": 42, "task_id": "test"})'}
                    ),
                )
            ],
            finish_reason="tool_calls",
            assistant_message={},
        )


class SampleAgent(Agent):
    """Test agent with a simple method."""

    async def process(self, task_id: int) -> dict:
        """Process a task and return a result dict.

        Returns a dict with 'value' and 'task_id' keys.
        """
        ...


@pytest.fixture
def fake_llm():
    """Create a fake LLM instance."""
    return FakeLLM()


@pytest.fixture
def temp_trace_dir():
    """Create a temporary directory for trace files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def setup_tracing(temp_trace_dir):
    """Set up tracing infrastructure for tests that need it.

    This fixture explicitly sets up tracing components rather than relying
    on enable_tracing()'s idempotency behavior. This ensures each test gets
    properly configured hooks regardless of global state from prior tests.

    Returns the hooks instance for tests that need to propagate hooks to child tasks.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    from nooa.tracing import NemoOOAgentsInstrumentor
    from nooa.tracing._otlp_file_exporter import (
        OtlpJsonFileExporter,
    )
    from nooa.tracing._session_processor import SessionSpanProcessor

    # Create a fresh exporter for this test's temp directory
    exporter = OtlpJsonFileExporter(temp_trace_dir)

    # Get or create tracer provider
    existing_provider = trace.get_tracer_provider()
    if hasattr(existing_provider, "add_span_processor"):
        tracer_provider = existing_provider
    else:
        tracer_provider = TracerProvider()
        trace.set_tracer_provider(tracer_provider)

    # Add session processor first, then exporter
    tracer_provider.add_span_processor(SessionSpanProcessor())
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Instrument nooa - this sets hooks
    NemoOOAgentsInstrumentor().instrument(tracer_provider=tracer_provider)

    # Return hooks for tests that need to propagate to child tasks
    hooks = get_hooks()

    yield hooks

    # Cleanup: remove hooks after test
    set_hooks(None)


@pytest.mark.asyncio
async def test_contextvar_isolation_in_gather():
    """Verify ContextVars are isolated across concurrent tasks."""
    results = []

    async def task(i: int):
        set_session(f"session-{i}")

        # Simulate async work
        await asyncio.sleep(0.01)

        # Verify isolation - our value shouldn't be overwritten
        actual = get_session()
        results.append((i, actual))

    # Run 20 concurrent tasks
    await asyncio.gather(*[task(i) for i in range(20)])

    # Each task should have its own session
    for i, actual in results:
        expected = f"session-{i}"
        assert actual == expected, f"Task {i}: expected {expected}, got {actual}"


@pytest.mark.asyncio
async def test_spans_dict_created_per_context():
    """Verify each async context gets its own spans dict."""
    spans_dict_ids = []

    async def task(i: int):
        # This should create a new spans dict for this context
        spans_dict = _get_active_spans()
        spans_dict_ids.append((i, id(spans_dict)))

        # Simulate work
        await asyncio.sleep(0.01)

        # Verify we still have the same dict
        same_dict = _get_active_spans()
        assert id(same_dict) == id(spans_dict), f"Task {i}: spans dict changed!"

    await asyncio.gather(*[task(i) for i in range(20)])

    # Each task should have a unique spans dict ID
    unique_ids = {dict_id for _, dict_id in spans_dict_ids}
    assert len(unique_ids) == 20, f"Expected 20 unique spans dicts, got {len(unique_ids)}"


@pytest.mark.asyncio
async def test_hooks_called_for_all_concurrent_agents(fake_llm, temp_trace_dir, setup_tracing):
    """Verify hooks.before_agent_call is invoked for every concurrent agent call."""
    _ = setup_tracing

    hook_calls = []
    original_before = OpenInferenceHooks.before_agent_call

    def tracking_before(self, agent, method_name, call_id, parent_call_id, args, kwargs, **extra):
        hook_calls.append(
            {
                "task": asyncio.current_task().get_name(),
                "agent": type(agent).__name__,
                "method": method_name,
                "call_id": call_id[:8],
            }
        )
        return original_before(
            self, agent, method_name, call_id, parent_call_id, args, kwargs, **extra
        )

    async def run_sample(i: int):
        """Simulate one eval sample."""
        set_session(f"sample_{i:03d}")

        agent = SampleAgent(llm=fake_llm)
        try:
            result = await agent.process(task_id=i)
            return (i, "success", result)
        except Exception as e:
            return (i, "error", str(e))

    # Patch to track hook calls
    with patch.object(OpenInferenceHooks, "before_agent_call", tracking_before):
        # Run 20 concurrent samples
        tasks = [run_sample(i) for i in range(20)]
        results = await asyncio.gather(*tasks)

    # Count successes and errors
    successes = sum(1 for _, status, _ in results if status == "success")
    errors = [(i, msg) for i, status, msg in results if status == "error"]

    print(f"Results: {successes} successes, {len(errors)} errors")
    for i, msg in errors[:5]:
        print(f"  Task {i}: {msg[:100]}")

    # CRITICAL ASSERTION: Every task should have triggered before_agent_call
    assert len(hook_calls) >= 20, (
        f"Only {len(hook_calls)}/20 agent calls triggered before_agent_call hook. "
        f"This indicates the concurrency bug is present."
    )


@pytest.mark.asyncio
async def test_trace_files_created_for_all_samples(fake_llm, temp_trace_dir, setup_tracing):
    """Verify trace files are created for all concurrent samples."""
    parent_hooks = setup_tracing

    async def run_sample(i: int) -> tuple[int, Path, bool]:
        """Run one sample and return whether trace file was created."""
        set_hooks(parent_hooks)

        session_id = f"sample_{i:03d}"
        set_session(session_id)
        trace_file = temp_trace_dir / f"{session_id}.jsonl"

        agent = SampleAgent(llm=fake_llm)
        try:
            await agent.process(task_id=i)
        except Exception:
            pass  # Ignore agent errors, we're testing trace creation

        # Check if trace file exists and has content
        exists = trace_file.exists()
        has_content = exists and trace_file.stat().st_size > 0
        return (i, trace_file, has_content)

    # Run 20 concurrent samples
    results = await asyncio.gather(*[run_sample(i) for i in range(20)])

    # Check results
    missing = [(i, path) for i, path, has_content in results if not has_content]

    if missing:
        print(f"Missing trace files for samples: {[i for i, _ in missing]}")

    # CRITICAL ASSERTION: All samples should have trace files
    assert len(missing) == 0, (
        f"{len(missing)}/20 samples missing trace files: {[i for i, _ in missing]}. "
        f"This reproduces the missing trace files bug."
    )


@pytest.mark.asyncio
async def test_multiple_runs_sequential(fake_llm, temp_trace_dir, setup_tracing):
    """Test multiple sequential runs (like eval --runs 10).

    The original bug appeared in run3+ suggesting state accumulation.
    """
    parent_hooks = setup_tracing

    all_results = []

    for run_id in range(10):
        # Capture run_id in closure properly
        async def run_sample(i: int, current_run_id: int = run_id) -> tuple[int, int, bool]:
            set_hooks(parent_hooks)

            session_id = f"run{current_run_id}_sample_{i:03d}"
            set_session(session_id)
            trace_file = temp_trace_dir / f"{session_id}.jsonl"

            agent = SampleAgent(llm=fake_llm)
            try:
                await agent.process(task_id=i)
            except Exception:
                pass

            has_content = trace_file.exists() and trace_file.stat().st_size > 0
            return (current_run_id, i, has_content)

        # Run 5 concurrent samples per run
        run_results = await asyncio.gather(*[run_sample(i) for i in range(5)])
        all_results.extend(run_results)

    # Group by run to see if later runs fail more often
    by_run = {}
    for run_id, _sample_id, has_content in all_results:
        by_run.setdefault(run_id, []).append(has_content)

    print("Results by run:")
    for run_id in sorted(by_run.keys()):
        success_count = sum(by_run[run_id])
        print(f"  Run {run_id}: {success_count}/5 trace files created")

    # Check for pattern where later runs fail
    missing_by_run = {run_id: 5 - sum(results) for run_id, results in by_run.items()}

    total_missing = sum(missing_by_run.values())
    assert total_missing == 0, (
        f"Missing trace files by run: {missing_by_run}. "
        f"Pattern suggests state accumulation bug if later runs fail more."
    )
