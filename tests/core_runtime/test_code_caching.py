# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for code generation behavior.

Focus on:
- Generation methods always regenerate code (ephemeral behavior)
- Each call is independent
"""

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


@pytest.mark.asyncio
async def test_implemented_plan_method():
    """Test implemented @strategy method (with body) is called directly."""

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

    # Should execute directly every time
    assert result1 == 1
    assert result2 == 2
    assert result3 == 3


@pytest.mark.asyncio
async def test_ephemeral_generates_every_call():
    """Test generation methods: LLM called every time."""

    class CounterAgent(Agent, llm=_TEST_LLM):
        """Agent with generation method - generates code every call."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.count = 0
            self.generation_count = 0  # Track how many times LLM was called

        @strategy(PurePythonStrategy())
        async def increment(self, amount: int) -> int:
            """
            Increment counter by amount.
            Should generate new code every call.
            """
            ...

    # Create LLM with 3 responses (one for each call) - REPL-style
    fake_llm = FakeLLMClient(
        scripted_responses=[
            _resp("self.count += amount\nself.generation_count += 1\nreturn self.count"),
            _resp("self.count += amount\nself.generation_count += 1\nreturn self.count"),
            _resp("self.count += amount\nself.generation_count += 1\nreturn self.count"),
        ]
    )

    counter = CounterAgent(llm=fake_llm)

    # Call 1
    result1 = await counter.increment(5)
    assert result1 == 5, f"Expected 5, got {result1}"
    assert counter.count == 5
    assert counter.generation_count == 1, "LLM should be called on first call"

    # Call 2 - should generate again
    result2 = await counter.increment(10)
    assert result2 == 15, f"Expected 15, got {result2}"
    assert counter.count == 15
    assert counter.generation_count == 2, "LLM should be called on second call"

    # Call 3 - should generate again
    result3 = await counter.increment(3)
    assert result3 == 18, f"Expected 18, got {result3}"
    assert counter.count == 18
    assert counter.generation_count == 3, "LLM should be called on third call"

    # Verify LLM was called 3 times
    assert fake_llm.call_count == 3, f"Expected 3 LLM calls, got {fake_llm.call_count}"
