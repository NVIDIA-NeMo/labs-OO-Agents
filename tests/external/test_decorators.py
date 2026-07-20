# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for agent decorators."""

from nooa import strategy
from nooa.agent import Agent
from nooa.ellipsis_detection import has_ellipsis_body
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient

# Module-level test LLM (can be overridden at instantiation)
_TEST_LLM = FakeLLMClient()


def test_has_ellipsis_body_true():
    """Test detecting ellipsis body."""

    def example():
        """Docstring."""
        ...

    assert has_ellipsis_body(example) is True


def test_has_ellipsis_body_false():
    """Test detecting non-ellipsis body."""

    def example():
        """Docstring."""
        return 42

    assert has_ellipsis_body(example) is False


def test_agent_decorator():
    """Test Agent class configuration via __init_subclass__."""

    class TestAgent(Agent, llm=_TEST_LLM):
        pass

    # Should have LLM client stored at class level
    assert hasattr(TestAgent, "_agent_llm")
    assert TestAgent._agent_llm is _TEST_LLM


def test_plan_decorator():
    """Test @strategy decorator for ellipsis methods."""

    class TestAgent(Agent):
        @strategy(PurePythonStrategy())
        async def process(self, data: str) -> dict:
            """Process data."""
            ...

    # Should have metadata (set by metaclass auto-wrapper)
    assert hasattr(TestAgent.process, "_agent_decorator")
    assert TestAgent.process._agent_decorator == "auto"  # Metaclass auto-wraps
    assert isinstance(TestAgent.process._plan_strategy, PurePythonStrategy)
    assert TestAgent.process._needs_generation is True


def test_plan_decorator_string_enums():
    """Test @strategy decorator with strategy instances."""

    class TestAgent(Agent):
        @strategy(PurePythonStrategy())
        async def process(self, data: str) -> dict:
            """Process data."""
            ...

    # Should store strategy instance
    assert isinstance(TestAgent.process._plan_strategy, PurePythonStrategy)


def test_plan_decorator_defaults():
    """Test metaclass auto-wrapping with default values (no explicit decorator)."""

    class TestAgent(Agent):
        async def process(self, data: str) -> dict:
            """Process data."""
            ...

    # Should have metadata from metaclass auto-wrapper
    assert hasattr(TestAgent.process, "_agent_decorator")
    assert TestAgent.process._agent_decorator == "auto"  # Auto-wrapped by metaclass
    # Strategy is resolved at runtime, so _plan_strategy is None
    assert TestAgent.process._plan_strategy is None
    assert TestAgent.process._needs_generation is True


def test_plan_decorator_partial_defaults():
    """Test mix of auto-wrapped and @strategy decorated methods."""

    class TestAgent(Agent):
        async def process1(self, data: str) -> dict:
            """Process data."""
            ...

        @strategy(PurePythonStrategy())
        async def process2(self, data: str) -> dict:
            """Process data."""
            ...

    # process1 has no explicit strategy (will use default PurePythonStrategy at runtime)
    # The _plan_strategy attribute is None to avoid circular imports
    assert TestAgent.process1._plan_strategy is None
    # process2 has explicit strategy set
    assert isinstance(TestAgent.process2._plan_strategy, PurePythonStrategy)


def test_agent_instantiation():
    """Test that agents can be instantiated."""

    class SimpleAgent(Agent, llm=_TEST_LLM):
        def __init__(self):
            super().__init__()
            self.data = []

    agent_instance = SimpleAgent()
    assert agent_instance.agent_id is not None
    assert isinstance(agent_instance.data, list)
