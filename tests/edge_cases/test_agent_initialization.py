# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for detecting missing super().__init__() in Agent subclasses.

This test suite verifies that the system provides clear error messages when
users forget to call super().__init__() in their custom Agent __init__ methods.
"""

import pytest

from nooa import Agent
from nooa.unifiedllm import FakeLLMClient


@pytest.mark.asyncio
async def test_missing_super_init_detection():
    """Test that missing super().__init__() is detected with clear error."""

    test_llm = FakeLLMClient()

    class BrokenAgent(Agent, llm=test_llm):
        """Agent with __init__ that forgets super().__init__()."""

        def __init__(self):
            # Missing super().__init__() - bug!
            self.data = []

        async def process(self):
            """Process something."""
            ...

    agent = BrokenAgent()

    # Should raise RuntimeError with clear message about missing super().__init__()
    with pytest.raises(RuntimeError) as exc_info:
        await agent.process()

    error_msg = str(exc_info.value)
    assert "not properly initialized" in error_msg
    assert "super().__init__()" in error_msg
    assert "BrokenAgent" in error_msg
    # Should NOT mention RuntimeServices (that was the confusing old error)
    assert "RuntimeServices" not in error_msg


@pytest.mark.asyncio
async def test_missing_super_init_with_params():
    """Test detection when __init__ has parameters but no super().__init__()."""

    test_llm = FakeLLMClient()

    class BrokenAgentWithParams(Agent, llm=test_llm):
        """Agent with parameterized __init__ that forgets super().__init__()."""

        def __init__(self, name: str, count: int = 0):
            # Missing super().__init__() - bug!
            self.name = name
            self.count = count

        async def get_info(self) -> str:
            """Get agent info."""
            ...

    agent = BrokenAgentWithParams("test", 42)

    with pytest.raises(RuntimeError) as exc_info:
        await agent.get_info()

    error_msg = str(exc_info.value)
    assert "not properly initialized" in error_msg
    assert "super().__init__()" in error_msg


@pytest.mark.asyncio
async def test_correct_super_init_works():
    """Test that correct super().__init__() usage works as expected."""

    test_llm = FakeLLMClient()

    class CorrectAgent(Agent, llm=test_llm):
        """Agent with proper __init__ that calls super().__init__()."""

        def __init__(self, initial_value: int = 0):
            super().__init__()  # Correct!
            self.value = initial_value

        async def get_value(self) -> int:
            """Get the current value."""
            return self.value

    agent = CorrectAgent(42)

    # This should work without errors
    result = await agent.get_value()
    assert result == 42


@pytest.mark.asyncio
async def test_no_custom_init_works():
    """Test that agents without custom __init__ work fine."""

    test_llm = FakeLLMClient()

    class SimpleAgent(Agent, llm=test_llm):
        """Agent with no custom __init__ - uses parent's."""

        async def greet(self) -> str:
            """Return a greeting."""
            return "Hello!"

    agent = SimpleAgent()

    # This should work without errors
    result = await agent.greet()
    assert result == "Hello!"


@pytest.mark.asyncio
async def test_super_init_with_llm_override():
    """Test that super().__init__() with llm parameter override works."""

    test_llm1 = FakeLLMClient()
    test_llm2 = FakeLLMClient()

    class AgentWithLLMOverride(Agent, llm=test_llm1):
        """Agent that overrides LLM at instance level."""

        def __init__(self, custom_llm):
            super().__init__(llm=custom_llm)  # Override class-level LLM
            self.initialized = True

        async def check_init(self) -> bool:
            """Check if initialized."""
            return self.initialized

    agent = AgentWithLLMOverride(test_llm2)

    # Verify it's using the instance llm, not class llm
    assert agent._llm is test_llm2

    # This should work without errors
    result = await agent.check_init()
    assert result is True


@pytest.mark.asyncio
async def test_error_message_includes_pattern():
    """Test that error message includes the expected __init__ pattern."""

    test_llm = FakeLLMClient()

    class BrokenAgent(Agent, llm=test_llm):
        def __init__(self):
            self.x = 1  # Missing super().__init__()

        async def method(self): ...

    agent = BrokenAgent()

    with pytest.raises(RuntimeError) as exc_info:
        await agent.method()

    error_msg = str(exc_info.value)
    # Should show the expected pattern with proper indentation
    assert "def __init__(self, ...):" in error_msg
    assert "super().__init__()" in error_msg
    assert "# Your initialization code here" in error_msg
