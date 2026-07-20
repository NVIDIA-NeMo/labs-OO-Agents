# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for implemented @strategy methods (methods with full body, not ... ellipsis).

These test that:
- Implemented @strategy methods execute as normal Python
- @strategy methods can call other @strategy methods
- Context vars are set so utilities can access agent if needed
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
async def test_implemented_plan_executes():
    """
    Test that implemented @strategy methods execute as normal Python.

    Implemented methods (with full body, not ...) run directly.
    Unlike generated code, they don't have utilities auto-injected.
    Users should use explicit imports or access via self.
    """

    class TestAgent(Agent, llm=_TEST_LLM):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stored_data = None

        async def process_data(self, data: str) -> str:
            """
            Process some data.

            THIS IS A FULL IMPLEMENTATION (not ... body).
            """
            # Store on self
            self.stored_data = data
            return f"Processed: {data}"

    agent_instance = TestAgent()

    result = await agent_instance.process_data("test_data")

    # Verify it worked
    assert result == "Processed: test_data"
    assert agent_instance.stored_data == "test_data"


@pytest.mark.asyncio
async def test_nested_implemented_plan_methods():
    """
    Test that implemented @strategy methods can call other @strategy methods.

    Pattern:
    - outer_method() is an implemented ellipsis method
    - Calls inner_method() which needs generation
    """

    class TestAgent(Agent, llm=_TEST_LLM):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        async def outer_method(self, x: int) -> int:
            """Outer orchestration (implemented)."""
            result = await self.inner_method(x)
            return result * 2

        @strategy(
            PurePythonStrategy(),
        )
        async def inner_method(self, x: int) -> int:
            """Inner computation (needs generation)."""
            ...

    fake_llm = FakeLLMClient(scripted_responses=[_resp("return x + 10")])

    agent_instance = TestAgent(llm=fake_llm)

    result = await agent_instance.outer_method(5)

    # inner_method(5) returns 15, outer doubles it
    assert result == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
