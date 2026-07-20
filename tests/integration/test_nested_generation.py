# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for nested generation - calling @strategy methods from within generated code."""

import asyncio

import pytest

from nooa import strategy
from nooa.agent import Agent
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse


def _resp(content: str) -> LLMResponse:
    """Create a test LLM response with the given content."""
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": content},
    )


# Module-level test LLM (can be overridden at instantiation)
_TEST_LLM = FakeLLMClient()


@pytest.mark.asyncio
async def test_nested_generation_simple():
    """Test that a @strategy method can call another @strategy method that needs generation."""

    class TodoAgent(Agent, llm=_TEST_LLM):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.results = []

        @strategy(
            PurePythonStrategy(),
        )
        async def main_task(self) -> str:
            """Main task that delegates to helper."""
            ...

        @strategy(
            PurePythonStrategy(),
        )
        async def helper_task(self) -> str:
            """Helper task."""
            ...

    fake_llm = FakeLLMClient(
        scripted_responses=[
            # main_task generation - REPL-style
            _resp("result = await self.helper_task()\nself.results.append('main')\nreturn result"),
            # helper_task generation - REPL-style
            _resp("self.results.append('helper')\nreturn 'done'"),
        ]
    )

    agent_instance = TodoAgent(llm=fake_llm)

    # Call main task - it should call helper task during execution
    result = await agent_instance.main_task()

    # Verify result
    assert result == "done"

    # Verify both methods generated code
    assert fake_llm.call_count == 2

    # Verify execution order (helper runs during main's execution)
    assert agent_instance.results == ["helper", "main"]

    print("✓ Nested generation works - main called helper, both generated code")


@pytest.mark.asyncio
async def test_nested_generation_chain():
    """Test a chain of nested generations (3 levels deep)."""

    class ChainAgent(Agent, llm=_TEST_LLM):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.call_order = []

        @strategy(
            PurePythonStrategy(),
        )
        async def level_1(self) -> str:
            """Level 1 - calls level 2."""
            ...

        @strategy(
            PurePythonStrategy(),
        )
        async def level_2(self) -> str:
            """Level 2 - calls level 3."""
            ...

        @strategy(
            PurePythonStrategy(),
        )
        async def level_3(self) -> str:
            """Level 3 - does actual work."""
            ...

    fake_llm = FakeLLMClient(
        scripted_responses=[
            # level_1 generation - REPL-style
            _resp(
                "self.call_order.append(1)\nresult = await self.level_2()\nreturn f'L1-{result}'"
            ),
            # level_2 generation - REPL-style
            _resp(
                "self.call_order.append(2)\nresult = await self.level_3()\nreturn f'L2-{result}'"
            ),
            # level_3 generation - REPL-style
            _resp("self.call_order.append(3)\nreturn 'L3'"),
        ]
    )

    agent_instance = ChainAgent(llm=fake_llm)

    result = await agent_instance.level_1()

    # Verify result flows back up the chain
    assert result == "L1-L2-L3"

    # Verify all three generated
    assert fake_llm.call_count == 3

    # Verify execution order (deepest first)
    assert agent_instance.call_order == [1, 2, 3]

    print("✓ 3-level nested generation chain works")


# Deleted: test_nested_generation_with_persistent_lifetime
# PERSISTENT lifetime caching was removed - all generation methods now regenerate every call


@pytest.mark.asyncio
async def test_nested_generation_parallel_calls():
    """Test calling multiple @strategy methods in parallel from generated code."""

    class ParallelAgent(Agent, llm=_TEST_LLM):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.task_results = []

        @strategy(
            PurePythonStrategy(),
        )
        async def coordinator(self) -> list[str]:
            """Coordinator that calls multiple tasks in parallel."""
            ...

        @strategy(
            PurePythonStrategy(),
        )
        async def task_a(self) -> str:
            """Task A."""
            ...

        @strategy(
            PurePythonStrategy(),
        )
        async def task_b(self) -> str:
            """Task B."""
            ...

    fake_llm = FakeLLMClient(
        scripted_responses=[
            # coordinator generation - REPL-style
            _resp("results = await asyncio.gather(self.task_a(), self.task_b())\nreturn results"),
            # task_a generation - REPL-style
            _resp("await asyncio.sleep(0.01)\nself.task_results.append('A')\nreturn 'result_a'"),
            # task_b generation - REPL-style
            _resp("await asyncio.sleep(0.01)\nself.task_results.append('B')\nreturn 'result_b'"),
        ]
    )

    agent_instance = ParallelAgent(llm=fake_llm)

    results = await agent_instance.coordinator()

    # Verify results
    assert results == ["result_a", "result_b"]

    # Verify all three generated
    assert fake_llm.call_count == 3

    # Verify both tasks executed
    assert len(agent_instance.task_results) == 2
    assert "A" in agent_instance.task_results
    assert "B" in agent_instance.task_results

    print("✓ Parallel nested generation works with asyncio.gather")


@pytest.mark.asyncio
async def test_no_deadlock_on_nested_generation():
    """Explicit test that nested generation doesn't deadlock (regression test)."""

    class DeadlockTestAgent(Agent, llm=_TEST_LLM):
        @strategy(
            PurePythonStrategy(),
        )
        async def outer(self) -> str:
            """Outer method using ITERATIVE strategy."""
            ...

        @strategy(
            PurePythonStrategy(),
        )
        async def inner(self) -> str:
            """Inner method using DIRECT strategy."""
            ...

    fake_llm = FakeLLMClient(
        scripted_responses=[
            # outer generation - REPL-style
            _resp("result = await self.inner()\nreturn f'outer-{result}'"),
            # inner generation - REPL-style
            _resp("return 'inner'"),
        ]
    )

    agent_instance = DeadlockTestAgent(llm=fake_llm)

    # This should complete without deadlock (with timeout as safety)
    result = await asyncio.wait_for(agent_instance.outer(), timeout=5.0)

    assert result == "outer-inner"
    assert fake_llm.call_count == 2

    print("✓ No deadlock on nested generation (different strategies)")


if __name__ == "__main__":
    asyncio.run(test_nested_generation_simple())
    asyncio.run(test_nested_generation_chain())
    # test_nested_generation_with_persistent_lifetime - removed (PERSISTENT caching no longer exists)
    asyncio.run(test_nested_generation_parallel_calls())
    asyncio.run(test_no_deadlock_on_nested_generation())
    print("\n✅ All nested generation tests passed!")
