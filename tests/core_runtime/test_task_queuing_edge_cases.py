# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge cases for task queuing and generation lock.

Focus on:
- Serialized generation with concurrent method calls
- Methods with bodies calling methods needing generation
- Signal queued during generation session
- Nested generation with different strategies
"""

import asyncio

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


class EdgeCaseAgent(Agent, llm=_TEST_LLM):
    """Agent for testing edge cases."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logged_events = []  # Renamed to avoid shadowing self.events (Events view)
        self.counter = 0

    @strategy(PurePythonStrategy())
    async def method_a(self) -> str:
        """Method A."""
        ...

    @strategy(PurePythonStrategy())
    async def method_b(self) -> str:
        """Method B."""
        ...

    async def method_with_body(self) -> str:
        """Method with body that calls another method."""
        # This method has a body, so it executes directly
        result = await self.method_a()
        return f"body-{result}"

    async def log_signal(self, msg: str) -> None:
        """Signal for testing."""
        self.logged_events.append(f"signal-{msg}")


@pytest.mark.asyncio
async def test_multiple_plan_methods_concurrent_generation():
    """Test multiple @strategy methods called concurrently - generation serialized."""

    fake_llm = FakeLLMClient(scripted_responses=[_resp("return 'a'"), _resp("return 'b'")])

    agent_instance = EdgeCaseAgent(llm=fake_llm)

    # Call both methods concurrently
    results = await asyncio.gather(agent_instance.method_a(), agent_instance.method_b())

    # Both should complete
    assert results == ["a", "b"]
    # Both should have generated (serialized generation - one at a time)
    assert fake_llm.call_count == 2


@pytest.mark.asyncio
async def test_plan_with_body_calls_plan_needing_generation():
    """Test @strategy method with body calling @strategy method needing generation."""

    fake_llm = FakeLLMClient(scripted_responses=[_resp("return 'a'")])

    agent_instance = EdgeCaseAgent(llm=fake_llm)

    # Call method with body - it should call method_a which needs generation
    result = await agent_instance.method_with_body()

    # Should complete successfully
    assert result == "body-a"
    # Method A should have generated
    assert fake_llm.call_count == 1


@pytest.mark.asyncio
async def test_nested_generation_multi_turn():
    """Test multi-turn generation where outer method explores then defines code."""

    class StrategyTestAgent(Agent, llm=_TEST_LLM):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.results = []

        @strategy(
            PurePythonStrategy(),
        )
        async def outer_method(self) -> str:
            """Outer method that calls inner_method."""
            ...

        @strategy(
            PurePythonStrategy(),
        )
        async def inner_method(self) -> str:
            """Inner method."""
            ...

    fake_llm = FakeLLMClient(
        scripted_responses=[
            # Turn 1: Outer explores (no method defined yet)
            _resp("# Exploring what inner_method returns\nprint('planning outer')"),
            # Turn 2: Outer calls inner - REPL-style
            _resp("inner_result = await self.inner_method()\nreturn f'outer-{inner_result}'"),
            # Turn 3: Inner method generation (nested call from outer execution) - REPL-style
            _resp("return 'inner'"),
        ]
    )

    agent_instance = StrategyTestAgent(llm=fake_llm)

    result = await agent_instance.outer_method()

    # Both should generate and execute correctly
    assert result == "outer-inner"
    # LLM calls: 2 for outer (explore + define) + 1 for inner = 3
    assert fake_llm.call_count == 3
