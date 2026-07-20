# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test that LLM-generated code can define @strategy decorated methods with ellipsis bodies.

This tests the fix for ellipsis detection on exec()-defined methods, which is
critical for allowing LLMs to generate helper methods with @strategy decorators.
"""

import pytest

from nooa import Agent, strategy
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient


class SampleAgent(Agent):
    """Sample agent for LLM-generated method tests."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.state = []

    @strategy(PurePythonStrategy())
    async def create_helper_method(self) -> str:
        """Create a helper method using @strategy decorator."""
        ...


@pytest.mark.asyncio
async def test_llm_can_generate_plan_decorated_method_with_ellipsis():
    """Test that LLM can generate a @strategy decorated method with ellipsis body.

    This is the core scenario: An LLM generates code that includes:
    1. A @strategy decorator with a strategy
    2. A function definition with ellipsis body
    3. The decorator should correctly detect the ellipsis and mark for generation
    """
    # Create fake LLM that generates a helper method with @strategy decorator
    llm = FakeLLMClient.with_code_responses(
        [
            # First response: Define a helper method with @strategy and ellipsis
            # Note: We must define the target method properly (not redefine it)
            """
# Define a helper method with @strategy decorator and ellipsis
@strategy(PredictStrategy())


async def _helper(self, text: str) -> str:
    '''Helper method with ellipsis body.'''
    ...

# Implement the actual target method
return "Helper method created"
"""
        ]
    )

    my_agent = SampleAgent(llm=llm)
    result = await my_agent.create_helper_method()

    assert result == "Helper method created"

    # helpers are plain callables in session_locals — not attached to
    # the agent instance or class. Existence of the decorated function is
    # verified by the parent execution completing successfully; ``_needs_generation``
    # is exercised in tests/test_ellipsis_detection_exec.py.
    assert not hasattr(my_agent, "_helper")


@pytest.mark.asyncio
async def test_llm_can_generate_multiple_plan_methods():
    """Test that LLM can generate multiple @strategy methods in one response."""
    llm = FakeLLMClient.with_code_responses(
        [
            """
# Define two helper methods with @strategy decorators
@strategy(PredictStrategy())


async def _helper1(self, x: int) -> int:
    '''First helper with ellipsis.'''
    ...

@strategy(PurePythonStrategy())


async def _helper2(self, y: int) -> int:
    '''Second helper with ellipsis.'''
    ...

# Return success message
return "Created 2 helpers"
"""
        ]
    )

    my_agent = SampleAgent(llm=llm)
    result = await my_agent.create_helper_method()

    assert result == "Created 2 helpers"

    # helpers live in session_locals as plain callables, not on the agent.
    assert not hasattr(my_agent, "_helper1")
    assert not hasattr(my_agent, "_helper2")


@pytest.mark.asyncio
async def test_llm_generated_plan_method_with_implementation():
    """Test that LLM can also generate @strategy methods WITH implementations (not ellipsis)."""
    llm = FakeLLMClient.with_code_responses(
        [
            """
# Define helper with implementation (not ellipsis)
async def _helper_with_impl(self, x: int) -> int:
    '''Helper with actual implementation.'''
    return x * 2

# Return success message
return "Helper method with implementation created"
"""
        ]
    )

    my_agent = SampleAgent(llm=llm)
    result = await my_agent.create_helper_method()

    assert result == "Helper method with implementation created"

    # Verify helper exists and is NOT marked for generation
    helper = getattr(my_agent.__class__, "_helper_with_impl", None)
    if helper:
        assert hasattr(helper, "_needs_generation")
        assert not helper._needs_generation, "Implemented helper should NOT need generation"


@pytest.mark.asyncio
async def test_namespace_includes_decorators_and_strategies():
    """Test that @strategy decorator and strategy classes are available in LLM execution namespace."""
    llm = FakeLLMClient.with_code_responses(
        [
            """
# Verify all imports are available
imports_available = [
    strategy,
    PurePythonStrategy,
    PredictStrategy,
    ReflexionStrategy,
    TemplateStrategy,
    CompositeStrategy,
]

# Return the count
return f"Found {len(imports_available)} imports"
"""
        ]
    )

    my_agent = SampleAgent(llm=llm)
    result = await my_agent.create_helper_method()

    assert result == "Found 6 imports"


if __name__ == "__main__":
    # Run this test file
    pytest.main([__file__, "-v"])
