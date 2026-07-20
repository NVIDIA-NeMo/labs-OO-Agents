# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge cases for generation lock.

Focus on:
- Multiple @strategy methods called concurrently (different methods)
- @strategy method with body calling @strategy method needing generation
- Signal during generation session
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


class LockTestAgent(Agent, llm=_TEST_LLM):
    """Agent for testing generation lock edge cases."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logged_events = []  # Renamed to avoid shadowing self.events (Events view)

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
async def test_multiple_plan_methods_concurrent():
    """Test multiple @strategy methods called concurrently (different methods)."""

    fake_llm = FakeLLMClient(scripted_responses=[_resp("return 'a'"), _resp("return 'b'")])

    agent_instance = LockTestAgent(llm=fake_llm)

    # Call both methods concurrently
    results = await asyncio.gather(agent_instance.method_a(), agent_instance.method_b())

    # Both should complete
    assert results == ["a", "b"]
    # Both should have generated (serialized generation)
    assert fake_llm.call_count == 2


@pytest.mark.asyncio
async def test_plan_with_body_calls_plan_needing_generation():
    """Test @strategy method with body calling @strategy method needing generation."""

    fake_llm = FakeLLMClient(scripted_responses=[_resp("return 'a'")])

    agent_instance = LockTestAgent(llm=fake_llm)

    # Call method with body - it should call method_a which needs generation
    result = await agent_instance.method_with_body()

    # Should complete successfully
    assert result == "body-a"
    # Method A should have generated
    assert fake_llm.call_count == 1
