# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for code execution.

Focus on:
- Code execution via sandbox
- Method body execution (non-generated methods)
- Generated code execution (cached and fresh)
- Error handling (ExecutionError vs GenerationError)
"""

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
async def test_method_body_execution():
    """Test that methods with bodies execute directly without generation."""

    class TestAgent(Agent, llm=_TEST_LLM):
        def __init__(self):
            super().__init__()
            self.counter = 0

        async def increment(self, value: int) -> int:
            """Method with body - executes directly."""
            self.counter += value
            return self.counter

    agent_instance = TestAgent()
    # Call method - should execute body directly
    result = await agent_instance.increment(5)

    assert result == 5
    assert agent_instance.counter == 5

    # Call again - should execute body again
    result2 = await agent_instance.increment(3)

    assert result2 == 8
    assert agent_instance.counter == 8


@pytest.mark.asyncio
async def test_generated_code_execution():
    """Test that generated code executes correctly."""

    class TestAgent(Agent, llm=_TEST_LLM):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.count = 0

        @strategy(
            PurePythonStrategy(),
        )
        async def increment(self, amount: int) -> int:
            """Increment counter by amount."""
            ...

    fake_llm = FakeLLMClient(scripted_responses=[_resp("self.count += amount\nreturn self.count")])

    agent_instance = TestAgent(llm=fake_llm)

    # Call method - should generate and execute
    result = await agent_instance.increment(10)

    assert result == 10
    assert agent_instance.count == 10


@pytest.mark.asyncio
async def test_ephemeral_code_execution():
    """Test that code regenerates on every call (ephemeral behavior)."""

    class TestAgent(Agent, llm=_TEST_LLM):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.count = 0

        @strategy(
            PurePythonStrategy(),
        )
        async def increment(self, amount: int) -> int:
            """Increment counter by amount."""
            ...

    # Provide multiple responses since each call regenerates - REPL-style
    fake_llm = FakeLLMClient(
        scripted_responses=[
            _resp("self.count += amount\nreturn self.count"),
            _resp("self.count += amount\nreturn self.count"),
        ]
    )

    agent_instance = TestAgent(llm=fake_llm)

    # First call - generates code
    result1 = await agent_instance.increment(5)
    assert result1 == 5
    assert agent_instance.count == 5
    assert fake_llm.call_count == 1

    # Second call - regenerates code (ephemeral behavior)
    result2 = await agent_instance.increment(10)
    assert result2 == 15
    assert agent_instance.count == 15
    assert fake_llm.call_count == 2  # Regenerated on each call
