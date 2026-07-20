# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for actor runtime."""

import asyncio

import pytest

from nooa.agent import Agent
from nooa.unifiedllm import FakeLLMClient

# Module-level test LLM (can be overridden at instantiation)
_TEST_LLM = FakeLLMClient()


@pytest.mark.asyncio
async def test_actor_runtime_basic_plan_call():
    """Test basic @strategy method call through runtime."""

    class TestAgent(Agent, llm=_TEST_LLM):
        def __init__(self):
            super().__init__()
            self.result = 0

        async def increment(self, value: int) -> int:
            self.result = value
            return value + 1

    agent_instance = TestAgent()
    result = await agent_instance.increment(41)

    assert result == 42
    assert agent_instance.result == 41


@pytest.mark.asyncio
async def test_actor_runtime_concurrent_execution():
    """Test that implemented methods execute concurrently (with cooperative multitasking)."""

    class TestAgent(Agent, llm=_TEST_LLM):
        def __init__(self):
            super().__init__()
            self.calls = []

        async def slow_task(self, task_id: int) -> int:
            self.calls.append(f"start-{task_id}")
            await asyncio.sleep(0.01)
            self.calls.append(f"end-{task_id}")
            return task_id

    agent_instance = TestAgent()

    # Start multiple tasks concurrently
    results = await asyncio.gather(
        agent_instance.slow_task(1),
        agent_instance.slow_task(2),
        agent_instance.slow_task(3),
    )

    # Tasks should complete
    assert results == [1, 2, 3]

    # Execution should be concurrent - all tasks start before any complete
    # (Cooperative multitasking at await points)
    assert agent_instance.calls[0] == "start-1"
    assert agent_instance.calls[1] == "start-2"
    assert agent_instance.calls[2] == "start-3"
    # Then all complete (order may vary slightly but all started first)
    assert "end-1" in agent_instance.calls[3:]
    assert "end-2" in agent_instance.calls[3:]
    assert "end-3" in agent_instance.calls[3:]


@pytest.mark.asyncio
async def test_lifetime_cache_once():
    """Test ONCE lifetime (no caching)."""

    class TestAgent(Agent, llm=_TEST_LLM):
        def __init__(self):
            super().__init__()
            self.call_count = 0

        async def generate(self) -> int:
            self.call_count += 1
            return self.call_count

    agent_instance = TestAgent()

    # Call multiple times
    result1 = await agent_instance.generate()
    result2 = await agent_instance.generate()
    result3 = await agent_instance.generate()

    # Should regenerate every time (ONCE lifetime)
    assert result1 == 1
    assert result2 == 2
    assert result3 == 3


@pytest.mark.asyncio
async def test_concurrent_stdout_isolation():
    """Test that stdout is isolated between concurrent code executions.

    This is a regression test for the async-unsafe redirect_stdout issue.
    When multiple async tasks run concurrently, each should capture its own
    stdout without contamination from other tasks.
    """

    class TestAgent(Agent, llm=_TEST_LLM):
        def __init__(self):
            super().__init__()

        async def task_with_print(self, task_id: int) -> str:
            """Each task prints and returns its own ID."""
            print(f"output-from-task-{task_id}")
            await asyncio.sleep(0.01)  # Allow other tasks to interleave
            return f"result-{task_id}"

    agent_instance = TestAgent()

    # Execute code directly through the runtime to test stdout capture
    runtime = agent_instance.runtime

    async def run_code_execution(task_id: int) -> str:
        """Run code that prints task_id and return captured stdout."""
        code = f"""
print("output-from-task-{task_id}")
import asyncio
await asyncio.sleep(0.01)
result = "result-{task_id}"
"""
        result = await runtime.execute_code(code, wrap_in_function=True)
        return result.stdout.strip()

    # Run multiple executions concurrently
    stdouts = await asyncio.gather(
        run_code_execution(1),
        run_code_execution(2),
        run_code_execution(3),
    )

    # Each execution should only see its own print output
    assert stdouts[0] == "output-from-task-1", f"Task 1 stdout contaminated: {stdouts[0]}"
    assert stdouts[1] == "output-from-task-2", f"Task 2 stdout contaminated: {stdouts[1]}"
    assert stdouts[2] == "output-from-task-3", f"Task 3 stdout contaminated: {stdouts[2]}"


@pytest.mark.asyncio
async def test_nested_code_execution_stdout_isolation():
    """Test stdout isolation with nested code executions (like subagents).

    When a parent code execution calls a child (e.g., router calling subagent),
    stdout from the child should not leak into the parent's buffer.
    """

    class TestAgent(Agent, llm=_TEST_LLM):
        def __init__(self):
            super().__init__()

    agent_instance = TestAgent()
    runtime = agent_instance.runtime

    # Parent code that spawns a nested execution
    parent_code = """
print("parent-start")
# Simulate a nested execution (like calling a subagent)
import asyncio
await asyncio.sleep(0.01)
print("parent-end")
result = "parent-result"
"""

    child_code = """
print("child-output")
result = "child-result"
"""

    async def run_parent_and_child():
        """Run parent while a concurrent child executes."""
        parent_result, child_result = await asyncio.gather(
            runtime.execute_code(parent_code, wrap_in_function=True),
            runtime.execute_code(child_code, wrap_in_function=True),
        )
        return parent_result, child_result

    parent_result, child_result = await run_parent_and_child()

    # Parent should only see parent output
    parent_stdout = parent_result.stdout
    assert "parent-start" in parent_stdout, f"Missing parent-start: {parent_stdout}"
    assert "parent-end" in parent_stdout, f"Missing parent-end: {parent_stdout}"
    assert "child-output" not in parent_stdout, f"Child leaked into parent: {parent_stdout}"

    # Child should only see child output
    child_stdout = child_result.stdout
    assert "child-output" in child_stdout, f"Missing child-output: {child_stdout}"
    assert "parent-start" not in child_stdout, f"Parent leaked into child: {child_stdout}"
    assert "parent-end" not in child_stdout, f"Parent leaked into child: {child_stdout}"
