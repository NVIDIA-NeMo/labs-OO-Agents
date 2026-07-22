# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test to reproduce missing trace files / empty results bug.

BUG DESCRIPTION:
From production eval_pipeline runs, we observe intermittent failures where:
1. After N runs, ALL models fail simultaneously with 'NoneType' object is not callable
2. The failure manifests as 0.1s execution time with error
3. No trace file is created
4. The simultaneous failure across all models suggests global state corruption

OBSERVATIONS from eval_debug_router_2_20260126_181726.log:
- Run 1-3: All 6 models pass
- Run 4+: ALL 6 models fail simultaneously with 'NoneType' object is not callable
- 876 total failures matching 876 missing traces

INTERMITTENT NATURE:
- Bug does NOT reproduce consistently
- Other runs with same code show 0 failures
- Unit tests cannot reproduce the issue

FAILED REPRODUCTION ATTEMPTS:
1. FakeLLM with SimpleAgent - passes 20/20 runs
2. FakeLLM with RouterTestWrapper - passes 5/5 runs
3. Parallel execution with tracing enabled - passes 20/20 runs
4. linecache pollution test - stable at 2 entries

REMAINING HYPOTHESES:
1. Race condition in closure variable capture (make_factory)
2. Timing-dependent ContextVar corruption
3. Something in real LLM client response handling
4. Resource exhaustion in file handle caching
"""

import asyncio
import json
import linecache
from dataclasses import dataclass
from typing import TypedDict

import pytest

from nooa import Agent


class SimpleResult(TypedDict):
    computed: bool
    value: int


class SimpleAgent(Agent):
    """Simple agent for testing repeated runs without nested LLM calls."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.run_count = 0

    async def compute(self, x: int, y: int) -> SimpleResult:
        """Compute the sum of x and y.

        Just return the sum.
        """
        ...


@dataclass
class FakeToolCall:
    id: str
    name: str
    arguments: str


@dataclass
class FakeLLMResponse:
    finish_reason: str
    tool_calls: list[FakeToolCall]
    content: str | None = None


class FakeLLM:
    """Fake LLM that returns deterministic code."""

    def __init__(self, should_fail_after: int = -1):
        self.call_count = 0
        self.model_name = "fake-llm"
        self.should_fail_after = should_fail_after

    async def acall(self, messages, **kwargs):
        """Return code that computes and returns the result."""
        self.call_count += 1

        # Simulate the bug: after N calls, return empty/wrong result
        if self.should_fail_after > 0 and self.call_count > self.should_fail_after:
            code = """return_result(computed=False, value=0)"""
        else:
            code = """return_result(computed=True, value=x + y)"""

        return FakeLLMResponse(
            finish_reason="tool_calls",
            tool_calls=[
                FakeToolCall(
                    id=f"call_{self.call_count}",
                    name="execute_python",
                    arguments=json.dumps({"code": code}),
                )
            ],
            content=None,
        )


@pytest.mark.asyncio
async def test_repeated_runs_same_task():
    """Run the same task many times and check for degradation."""

    # Track results
    results = []
    failed_count = 0

    # Run 40 times (matching the eval that showed the bug)
    for run_id in range(1, 41):
        # Create fresh agent each time (like eval pipeline does)
        llm = FakeLLM()
        agent = SimpleAgent(llm=llm)

        # Run computation
        result = await agent.compute(x=5, y=3)
        results.append(result)

        # Check if result is correct
        if result.get("computed") and result.get("value") == 8:
            print(f"Run {run_id}: OK - {result}")
        else:
            failed_count += 1
            print(f"Run {run_id}: FAILED - {result} (failed_count={failed_count})")

    # All runs should succeed
    assert failed_count == 0, f"{failed_count}/40 runs failed"


@pytest.mark.asyncio
async def test_parallel_runs_same_task():
    """Run many tasks in parallel (like eval pipeline with parallel=20)."""

    async def run_single(run_id: int) -> tuple[int, SimpleResult]:
        llm = FakeLLM()
        agent = SimpleAgent(llm=llm)
        result = await agent.compute(x=5, y=3)
        return run_id, result

    # Run 40 tasks with concurrency of 20
    semaphore = asyncio.Semaphore(20)

    async def run_with_limit(run_id: int):
        async with semaphore:
            return await run_single(run_id)

    tasks = [run_with_limit(i) for i in range(1, 41)]
    results = await asyncio.gather(*tasks)

    # Check for failed results
    failed_runs = [
        run_id
        for run_id, result in results
        if not result.get("computed") or result.get("value") != 8
    ]

    assert len(failed_runs) == 0, f"Runs {failed_runs} failed"


@pytest.mark.asyncio
async def test_shared_llm_pollution():
    """Test if a shared LLM causes issues (it shouldn't in production)."""

    # In eval pipeline, each sample gets a fresh LLM client
    # But let's test what happens with a shared one
    shared_llm = FakeLLM()

    results = []
    for run_id in range(1, 21):
        # Reuse the same LLM (like a potential bug scenario)
        agent = SimpleAgent(llm=shared_llm)
        result = await agent.compute(x=run_id, y=1)
        results.append((run_id, result))

    # All should succeed even with shared LLM
    failed = [
        (run_id, r)
        for run_id, r in results
        if not r.get("computed") or r.get("value") != run_id + 1
    ]

    assert len(failed) == 0, f"Failed runs: {failed}"


@pytest.mark.asyncio
async def test_simulated_degradation():
    """Simulate the bug by having LLM fail after N calls.

    This test is expected to FAIL when should_fail_after is reached,
    demonstrating the failure mode.
    """
    # Simulate LLM that degrades after 5 calls
    llm = FakeLLM(should_fail_after=5)

    results = []
    for run_id in range(1, 11):
        agent = SimpleAgent(llm=llm)  # Same LLM, fresh agent
        result = await agent.compute(x=5, y=3)
        results.append((run_id, result))
        print(f"Run {run_id}: {result}")

    # First 5 should succeed, rest should fail
    first_five = results[:5]
    last_five = results[5:]

    assert all(r.get("computed") and r.get("value") == 8 for _, r in first_five), (
        "First 5 should succeed"
    )
    assert all(not r.get("computed") or r.get("value") == 0 for _, r in last_five), (
        "Last 5 should show degradation"
    )


@pytest.mark.asyncio
@pytest.mark.skip(
    reason="Requires real LLM - run manually with: pytest -k test_real_llm --run-slow"
)
async def test_real_llm_repeated_runs():
    """Test with real LLM to check for degradation.

    Run with: pytest tests/capability/test_router_repeated_runs.py::test_real_llm_repeated_runs -v -s

    This test uses actual LLM calls which can be expensive.
    """
    import os

    from nooa.unifiedllm import CompletionClient
    from tests.capability.agents.router import RouterTestWrapper

    # Use a fast, cheap model
    llm = CompletionClient(
        model="azure/gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY", ""),
    )

    results = []
    for run_id in range(1, 21):
        # Create fresh agent each time
        agent = RouterTestWrapper(llm=llm)

        try:
            result = await agent.process(user_message="validate this data", values=[1, 2, 3, 4, 5])
            print(f"Run {run_id}: OK - {result}")
            results.append((run_id, "ok", result))
        except Exception as e:
            print(f"Run {run_id}: ERROR - {e}")
            results.append((run_id, "error", str(e)))

    errors = [r for r in results if r[1] == "error"]
    assert len(errors) == 0, f"Errors in runs: {errors}"


@pytest.mark.asyncio
async def test_linecache_pollution():
    """Test if linecache accumulates entries across runs.

    linecache.cache is a global dict that stores Cell code for tracebacks.
    If it grows without bounds, it could cause memory issues or other problems.
    """
    # Track linecache size before/after
    initial_cell_count = len([k for k in linecache.cache if k.startswith("Cell ")])

    results = []
    for run_id in range(1, 21):
        llm = FakeLLM()
        agent = SimpleAgent(llm=llm)
        result = await agent.compute(x=5, y=3)
        results.append(result)

        cell_count = len([k for k in linecache.cache if k.startswith("Cell ")])
        print(f"Run {run_id}: result={result}, Cell entries in linecache: {cell_count}")

    final_cell_count = len([k for k in linecache.cache if k.startswith("Cell ")])
    growth = final_cell_count - initial_cell_count

    print(f"\nLinecache growth: {initial_cell_count} -> {final_cell_count} (+{growth} entries)")

    # Check if all runs succeeded
    failed = [i for i, r in enumerate(results, 1) if not r.get("computed") or r.get("value") != 8]
    assert len(failed) == 0, f"Runs {failed} failed"

    # Warn if linecache is growing significantly
    if growth > 20:
        print(f"WARNING: linecache grew by {growth} entries - potential memory leak")


@pytest.mark.asyncio
async def test_router_with_fake_llm():
    """Test the actual RouterTestWrapper with a FakeLLM.

    This is closer to the production scenario where subagents are created.
    """
    from tests.capability.agents.router import RouterTestWrapper

    class RouterFakeLLM:
        """Fake LLM for router tests that generates appropriate subagent code."""

        def __init__(self):
            self.call_count = 0
            self.model_name = "router-fake-llm"

        async def acall(self, messages, **kwargs):
            """Return code based on context."""
            self.call_count += 1

            # Parse the last message to understand what we're generating
            last_msg = messages[-1] if messages else {}
            content = str(last_msg.get("content", ""))

            # Check if this is for a subagent (Analyzer, Validator, Transformer)
            if "AnalyzerSubAgent" in content or "analyze" in content.lower():
                if "sum" in content.lower() or "mean" in content.lower():
                    # Subagent computing statistics
                    code = """
total = sum(values)
count = len(values)
mean = total / count if count > 0 else 0
return_result(sum=total, mean=mean, count=count)
"""
                else:
                    code = """return_result(sum=15.0, mean=3.0, count=5)"""
            elif "ValidatorSubAgent" in content or "validate" in content.lower():
                code = """
all_pos = all(v > 0 for v in values)
no_dups = len(values) == len(set(values))
is_sorted = values == sorted(values)
return_result(all_positive=all_pos, no_duplicates=no_dups, is_sorted=is_sorted)
"""
            elif "TransformerSubAgent" in content or "transform" in content.lower():
                code = """
if format == "CSV":
    result = ",".join(str(v) for v in values)
elif format == "JSON":
    result = str(values)
else:
    result = f"Total: {sum(values)}"
return_result(result)
"""
            else:
                # Router code - delegate to validator
                code = """
validator = self.ValidatorSubAgent(llm=self._llm)
result = await validator.validate(values)
return_result(agents_called=["Validator"], results={"Validator": result})
"""

            return FakeLLMResponse(
                finish_reason="tool_calls",
                tool_calls=[
                    FakeToolCall(
                        id=f"call_{self.call_count}",
                        name="execute_python",
                        arguments=json.dumps({"code": code}),
                    )
                ],
                content=None,
            )

    results = []
    for run_id in range(1, 21):
        llm = RouterFakeLLM()
        agent = RouterTestWrapper(llm=llm)

        try:
            result = await agent.process(user_message="validate this data", values=[1, 2, 3, 4, 5])
            print(f"Run {run_id}: {result}")
            results.append((run_id, "ok", result))
        except Exception as e:
            print(f"Run {run_id}: ERROR - {e}")
            results.append((run_id, "error", str(e)))

    # Check for empty results (the bug symptom)
    empty_results = [
        (r, status, res)
        for r, status, res in results
        if status == "ok" and res.get("agents_called") == []
    ]

    assert len(empty_results) == 0, (
        f"Runs returned empty results: {[r for r, _, _ in empty_results]}"
    )


if __name__ == "__main__":
    asyncio.run(test_repeated_runs_same_task())
