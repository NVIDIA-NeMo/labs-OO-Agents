# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for task queuing and signal semantics.

Focus on:
- Serialized generation (only one LLM generation session at a time via `_generation_lock`)
- Concurrent execution (methods execute concurrently, generation is serialized)
- Methods with bodies execute directly (no generation, no serialization)
- Methods with ellipsis (`...`) trigger generation (serialized)
- Signal queuing and fire-and-forget semantics
- Signal execution at await points
"""

import asyncio

import pytest

from nooa.agent import Agent
from nooa.unifiedllm import FakeLLMClient

# Module-level test LLM (can be overridden at instantiation)
_TEST_LLM = FakeLLMClient()


@pytest.mark.asyncio
async def test_basic_plan_call():
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
async def test_concurrent_execution():
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
