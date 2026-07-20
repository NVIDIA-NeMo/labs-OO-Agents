# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Test all permutations of calls between @strategy methods and regular methods.

Design Constraints:
- Ellipsis methods can call regular methods (sync or async) ✅
- Ellipsis methods can call other ellipsis methods ✅
- Regular async methods can call ellipsis methods ✅
- Regular sync methods should be read-only (best practice)
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


class MethodCallTestAgent(Agent, llm=_TEST_LLM):
    """Agent with ellipsis methods and regular methods for testing cross-calls."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.value = 0
        self.messages = []

    # Ellipsis methods (async, can generate code)
    @strategy(PurePythonStrategy())
    async def plan_method(self) -> str:
        """Ellipsis method that can call regular methods."""
        ...

    async def plan_calls_sync_method(self) -> str:
        """Ellipsis method (implemented) that calls sync method."""
        result = self.sync_method()
        return f"Plan got: {result}"

    async def plan_calls_plan(self) -> str:
        """Ellipsis method (implemented) that calls another ellipsis method."""
        result = await self.plan_calls_sync_method()
        return f"Outer: {result}"

    # Regular async methods (no decorator)

    async def async_method(self, msg: str) -> None:
        """Regular async method that stores messages."""
        self.messages.append(msg)

    async def async_calls_sync(self) -> None:
        """Regular async method that calls sync method."""
        result = self.sync_method()
        self.messages.append(f"Async got: {result}")

    async def async_calls_plan(self) -> None:
        """Regular async method that calls ellipsis method."""
        result = await self.plan_calls_sync_method()
        self.messages.append(f"Async->Plan: {result}")

    # Regular sync methods (read-only)

    def sync_method(self) -> str:
        """Sync method that returns current value."""
        return f"value={self.value}"

    def sync_returns_data(self) -> dict:
        """Sync method that returns dict."""
        return {"value": self.value, "messages": len(self.messages)}


# =============================================================================
# Test ellipsis methods calling other methods
# =============================================================================


@pytest.mark.asyncio
async def test_plan_generated_calls_sync():
    """Test LLM-generated ellipsis method calling sync method."""
    # LLM generates code that calls sync method
    fake_llm = FakeLLMClient(
        scripted_responses=[_resp('result = self.sync_method()\nreturn f"Generated: {result}"')]
    )

    agent = MethodCallTestAgent(llm=fake_llm)
    result = await agent.plan_method()

    assert "Generated: value=0" in result


@pytest.mark.asyncio
async def test_plan_implemented_calls_sync():
    """Test implemented ellipsis method calling sync method."""
    agent = MethodCallTestAgent()
    result = await agent.plan_calls_sync_method()

    assert result == "Plan got: value=0"


@pytest.mark.asyncio
async def test_plan_implemented_calls_plan():
    """Test implemented ellipsis method calling another ellipsis method."""
    agent = MethodCallTestAgent()
    result = await agent.plan_calls_plan()

    assert result == "Outer: Plan got: value=0"


# =============================================================================
# Test regular async methods calling other methods
# =============================================================================


@pytest.mark.asyncio
async def test_async_calls_sync():
    """Test regular async method calling sync method."""
    agent = MethodCallTestAgent()
    await agent.async_calls_sync()

    assert "Async got: value=0" in agent.messages


@pytest.mark.asyncio
async def test_async_calls_plan():
    """Test regular async method calling ellipsis method."""
    agent = MethodCallTestAgent()
    await agent.async_calls_plan()

    assert any("Async->Plan: Plan got: value=0" in msg for msg in agent.messages)


# =============================================================================
# Test sync methods (should only read, not call other methods)
# =============================================================================


@pytest.mark.asyncio
async def test_sync_returns_simple_data():
    """Test sync method returns simple data."""
    agent = MethodCallTestAgent()
    agent.value = 42

    result = agent.sync_method()
    assert result == "value=42"


@pytest.mark.asyncio
async def test_sync_returns_complex_data():
    """Test sync method returns complex data."""
    agent = MethodCallTestAgent()
    agent.value = 42
    agent.messages = ["a", "b", "c"]

    result = agent.sync_returns_data()
    assert result == {"value": 42, "messages": 3}


# =============================================================================
# Test edge cases
# =============================================================================


@pytest.mark.asyncio
async def test_nested_plan_calls_plan_calls_sync():
    """Test nested calls: outer ellipsis method -> inner ellipsis method -> sync method."""

    # Create agent with two ellipsis methods

    class NestedAgent(Agent, llm=_TEST_LLM):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.data = "test"

        def get_data(self) -> str:
            return self.data

        async def inner_plan(self) -> str:
            # Calls sync method
            return f"Inner: {self.get_data()}"

        async def outer_plan(self) -> str:
            # Calls inner ellipsis method which calls sync method
            inner_result = await self.inner_plan()
            return f"Outer: {inner_result}"

    test_agent = NestedAgent()
    result = await test_agent.outer_plan()

    assert result == "Outer: Inner: test"


@pytest.mark.asyncio
async def test_sync_called_from_generated_code_in_plan():
    """Test that LLM-generated code in ellipsis methods can call sync methods."""
    # LLM-generated code calling a sync method
    fake_llm = FakeLLMClient(
        scripted_responses=[
            _resp(
                "# Call sync method from generated code\n"
                "data = self.sync_returns_data()\n"
                "return f\"Got {data['messages']} messages, value={data['value']}\"\n"
            )
        ]
    )

    agent = MethodCallTestAgent(llm=fake_llm)
    agent.value = 99
    agent.messages = ["a", "b"]

    result = await agent.plan_method()

    assert "Got 2 messages, value=99" in result


if __name__ == "__main__":
    import asyncio

    print("Testing ellipsis method -> sync (generated)...")
    asyncio.run(test_plan_generated_calls_sync())
    print("✅ PASS\n")

    print("Testing ellipsis method -> sync (implemented)...")
    asyncio.run(test_plan_implemented_calls_sync())
    print("✅ PASS\n")

    print("Testing ellipsis method -> ellipsis method...")
    asyncio.run(test_plan_implemented_calls_plan())
    print("✅ PASS\n")

    print("Testing async -> sync...")
    asyncio.run(test_async_calls_sync())
    print("✅ PASS\n")

    print("Testing async -> ellipsis method...")
    asyncio.run(test_async_calls_plan())
    print("✅ PASS\n")

    print("Testing sync returns simple...")
    asyncio.run(test_sync_returns_simple_data())
    print("✅ PASS\n")

    print("Testing sync returns complex...")
    asyncio.run(test_sync_returns_complex_data())
    print("✅ PASS\n")

    print("Testing nested ellipsis method->ellipsis method->sync...")
    asyncio.run(test_nested_plan_calls_plan_calls_sync())
    print("✅ PASS\n")

    print("Testing sync from generated code...")
    asyncio.run(test_sync_called_from_generated_code_in_plan())
    print("✅ PASS\n")

    print("🎉 All permutation tests pass!")
