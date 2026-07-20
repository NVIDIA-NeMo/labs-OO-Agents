# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Test @strategy decorator wrapping behavior.

Tests that @strategy methods:
- Are wrapped at decoration time (not runtime)
- Preserve decorator metadata
- Route through runtime correctly
"""

import pytest

from nooa import Agent, strategy
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient

# Module-level test LLM (can be overridden at instantiation)
_TEST_LLM = FakeLLMClient()


class StubTestAgent(Agent, llm=_TEST_LLM):
    """Test agent with @strategy methods."""

    def __init__(self):
        super().__init__()
        self.status = "idle"
        self.messages = []
        self.result = None

    @strategy(PurePythonStrategy())
    async def do_work(self, value: int) -> int:
        """Do some work."""
        ...  # Will be generated

    async def handle_message(self, msg: str) -> None:
        """Handle a message."""
        self.messages.append(msg)

    def get_status(self) -> str:
        """Get current status."""
        return self.status


@pytest.mark.asyncio
async def test_regular_method_works():
    """Test that non-decorated methods work normally."""
    agent_inst = StubTestAgent()

    agent_inst.status = "working"
    status = agent_inst.get_status()

    assert status == "working"


@pytest.mark.asyncio
async def test_plan_method_has_decorator_metadata():
    """Test that ellipsis methods have auto-wrapper metadata."""
    agent_inst = StubTestAgent()

    # Check that method has decorator metadata (set by metaclass auto-wrapper)
    assert hasattr(agent_inst.do_work, "_agent_decorator")
    assert agent_inst.do_work._agent_decorator == "auto"
    assert agent_inst.do_work.__name__ == "do_work"


@pytest.mark.asyncio
async def test_agent_has_runtime_after_init():
    """Test that agent has runtime reference after initialization."""
    agent_inst = StubTestAgent()

    assert hasattr(agent_inst, "runtime")
    assert agent_inst.runtime is not None
    assert agent_inst.runtime.agent == agent_inst
