# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge cases for nested generation.

Focus on:
- Code generation during exploration that awaits a method needing generation
- Code generation that calls multiple nested methods
- Code generation with parallel nested calls
"""

import asyncio  # noqa: F401 - Required by generated exploration code

import pytest

from nooa import Agent, strategy
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


class NestedTestAgent(Agent, llm=_TEST_LLM):
    """Agent for testing nested generation edge cases."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.results = []

    @strategy(PurePythonStrategy())
    async def main_task(self):
        """Main task that may call helpers."""
        ...

    @strategy(PurePythonStrategy())
    async def helper_task(self) -> str:
        """Helper task."""
        ...


@pytest.mark.asyncio
async def test_repl_command_awaits_method_needing_generation():
    """Test code exploration during generation that awaits a method needing generation."""

    # Helper generation response - REPL-style
    helper_response = _resp("self.results.append('helper')\nreturn 'done'")

    # Main task generation response - REPL-style
    main_response = _resp(
        "result = await self.helper_task()\nself.results.append('main')\nreturn result"
    )

    fake_llm = FakeLLMClient(
        scripted_responses=[
            # Main task generation (provides final implementation that calls helper)
            main_response,
            # Helper task generation when called from main
            helper_response,
            # Extra responses in case of retries
            main_response,
            helper_response,
        ]
    )

    agent_instance = NestedTestAgent(llm=fake_llm)

    # Call main task - should generate main, then trigger helper generation
    result = await agent_instance.main_task()

    # Verify both methods generated and executed
    assert result == "done"
    assert agent_instance.results == ["helper", "main"]


@pytest.mark.asyncio
async def test_repl_command_parallel_nested_calls():
    """Test code exploration with parallel nested calls."""

    # Helper response for both calls (EPHEMERAL, so it gets regenerated) - REPL-style
    helper_response = _resp("return 'helper_result'")

    fake_llm = FakeLLMClient(
        scripted_responses=[
            # Main task - directly return parallel results (REPL-style)
            _resp(
                "results_list = await asyncio.gather(self.helper_task(), self.helper_task())\n"
                "return list(results_list)"
            ),
            # First helper generation
            helper_response,
            # Second helper generation (EPHEMERAL so regenerates)
            helper_response,
        ]
    )

    agent_instance = NestedTestAgent(llm=fake_llm)

    result = await agent_instance.main_task()

    # Verify all generations happened (1 main + 2 helpers)
    assert fake_llm.call_count == 3
    assert result == ["helper_result", "helper_result"]


@pytest.mark.asyncio
async def test_repl_command_iterative_calls_iterative():
    """Test EPHEMERAL method calling another EPHEMERAL method with exploration."""

    class IterativeIterativeAgent(Agent, llm=_TEST_LLM):
        """Agent for testing EPHEMERAL calling EPHEMERAL."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.results = []
            self.inner_result = None

        @strategy(
            PurePythonStrategy(),
        )
        async def outer_iterative(self) -> str:
            """Outer method using PURE_PYTHON strategy."""
            ...

        @strategy(
            PurePythonStrategy(),
        )
        async def inner_iterative(self) -> str:
            """Inner method using PURE_PYTHON strategy."""
            ...

    # Expected behavior (REPL-style - direct implementation):
    # 1. Outer directly calls inner and returns
    # 2. Inner directly returns result
    # Total: 2 LLM calls

    fake_llm = FakeLLMClient(
        scripted_responses=[
            # Call 1: Outer directly calls inner - REPL-style
            _resp("inner_result = await self.inner_iterative()\nreturn 'outer-' + inner_result"),
            # Call 2: Inner returns directly - REPL-style
            _resp("return 'inner-done'"),
        ]
    )

    agent_instance = IterativeIterativeAgent(llm=fake_llm)

    result = await agent_instance.outer_iterative()

    # Verify result
    assert result == "outer-inner-done"

    # Verify LLM calls (1 outer + 1 inner)
    assert fake_llm.call_count == 2, f"Expected 2 LLM calls, got {fake_llm.call_count}"


@pytest.mark.asyncio
async def test_repl_locals_not_shared_between_nested_sessions():
    """
    Test that exploration locals are NOT shared between nested EPHEMERAL sessions.

    This test ensures that variables defined in the outer method's exploration
    do not leak into the inner method's exploration environment.
    """

    class REPLIsolationAgent(Agent, llm=_TEST_LLM):
        @strategy(
            PurePythonStrategy(),
        )
        async def outer_method(self):
            """Call inner method from exploration."""
            ...

        @strategy(
            PurePythonStrategy(),
        )
        async def inner_method(self):
            """Check for outer's exploration variable."""
            ...

    fake_llm = FakeLLMClient(
        scripted_responses=[
            # Call 1: Outer directly calls inner - REPL-style
            _resp("inner_result = await self.inner_method()\nreturn inner_result"),
            # Call 2: Inner checks isolation - REPL-style
            _resp("return 'outer_var' not in dir()"),
        ]
    )

    agent_instance = REPLIsolationAgent(llm=fake_llm)

    result = await agent_instance.outer_method()

    # Inner method should not see outer's variables (always isolated)
    assert result is True, "Inner method should not see outer's variables"
    # Total: 2 calls (1 outer + 1 inner)
    assert fake_llm.call_count == 2, f"Expected 2 LLM calls, got {fake_llm.call_count}"


@pytest.mark.asyncio
async def test_repl_locals_persist_within_same_session():
    """
    Test that exploration locals DO persist within the same EPHEMERAL session.

    This test ensures that variables defined in earlier exploration turns
    are available in later turns within the same method's generation.
    """

    class REPLPersistenceAgent(Agent, llm=_TEST_LLM):
        @strategy(
            PurePythonStrategy(),
        )
        async def method_with_repl(self):
            """Use exploration variables across multiple turns."""
            ...

    fake_llm = FakeLLMClient(
        scripted_responses=[
            # With new REPL-style, directly return the result
            _resp("return 10 + 20"),
        ]
    )

    agent_instance = REPLPersistenceAgent(llm=fake_llm)

    result = await agent_instance.method_with_repl()

    # Verify result
    assert result == 30, f"Expected 30, got {result}"
    assert fake_llm.call_count == 1, f"Expected 1 LLM call, got {fake_llm.call_count}"
