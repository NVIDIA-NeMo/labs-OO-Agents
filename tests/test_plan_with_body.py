# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test ellipsis methods with actual implementations (not ...)."""

import asyncio

import pytest

from nooa import Agent
from nooa.unifiedllm import FakeLLMClient

# Module-level test LLM (can be overridden at instantiation)
_TEST_LLM = FakeLLMClient()


class PlanWithBodyAgent(Agent, llm=_TEST_LLM):
    """Test agent with ellipsis methods that have actual code."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.value = 0
        self.items = []

    async def increment_by(self, amount: int) -> int:
        """Increment value by amount (implemented ellipsis method)."""
        self.value += amount
        return self.value

    async def add_item(self, item: str) -> list:
        """Add item to list (implemented ellipsis method)."""
        self.items.append(item)
        return self.items.copy()

    async def get_double(self, x: int) -> int:
        """Double a number (implemented ellipsis method)."""
        return x * 2


@pytest.mark.asyncio
async def test_plan_with_implementation():
    """Test that ellipsis methods with actual code execute correctly."""

    agent = PlanWithBodyAgent()
    print("Testing ellipsis methods with implementation...")

    # Test 1: Simple ellipsis method with implementation
    result = await agent.increment_by(5)
    print(f"increment_by(5) = {result}, value = {agent.value}")
    assert result == 5
    assert agent.value == 5

    # Test 2: Another call
    result = await agent.increment_by(3)
    print(f"increment_by(3) = {result}, value = {agent.value}")
    assert result == 8
    assert agent.value == 8

    # Test 3: Ellipsis method with strategy specified
    result = await agent.add_item("apple")
    print(f"add_item('apple') = {result}")
    assert result == ["apple"]
    assert agent.items == ["apple"]

    result = await agent.add_item("banana")
    print(f"add_item('banana') = {result}")
    assert result == ["apple", "banana"]
    assert agent.items == ["apple", "banana"]

    # Test 4: Another ellipsis method with implementation
    result = await agent.get_double(5)
    print(f"get_double(5) = {result}")
    assert result == 10

    result = await agent.get_double(7)
    print(f"get_double(7) = {result}")
    assert result == 14

    print("✅ All ellipsis method with implementation tests passed!")


if __name__ == "__main__":
    asyncio.run(test_plan_with_implementation())
    print("✅ Ellipsis method with body test passed!")
