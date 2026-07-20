# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test runtime.list_methods() and runtime.print_methods()."""

import pytest

from nooa import Agent, strategy
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient

# Module-level test LLM (can be overridden at instantiation)
_TEST_LLM = FakeLLMClient()


class SimpleAgent(Agent, llm=_TEST_LLM):
    """Test agent with various method types."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.value = 0

    @strategy(PurePythonStrategy())
    async def generator_method(self):
        """Generator method."""
        ...

    async def my_signal(self, msg: str):
        """A signal method."""
        self.value += 1

    async def _implemented_method(self) -> str:
        """An implemented method (private helper)."""
        return "implemented"

    def _sync_method(self):
        """A synchronous method (private helper)."""
        return 42


@pytest.mark.asyncio
async def test_list_methods_before_generation():
    """Test list_methods before any code generation."""
    agent = SimpleAgent()

    methods = agent.runtime.list_methods()

    # Should have all public methods (private methods excluded)
    assert "generator_method" in methods
    assert "my_signal" in methods

    # Check generator methods (auto-wrapped by metaclass)
    assert methods["generator_method"]["type"] == "generator"
    assert methods["generator_method"]["decorator"] == "@auto"  # Metaclass auto-wraps
    assert isinstance(methods["generator_method"]["strategy"], PurePythonStrategy)
    assert methods["generator_method"]["is_async"] is True
    assert methods["generator_method"]["has_code"] is False

    # Check my_signal (implemented async method, no decorator needed)
    assert methods["my_signal"]["type"] == "implemented"  # Has implementation
    assert methods["my_signal"]["is_async"] is True


@pytest.mark.asyncio
async def test_print_methods():
    """Test print_methods formatting."""
    agent = SimpleAgent()

    # Should print without errors
    agent.runtime.print_methods()


if __name__ == "__main__":
    import asyncio

    print("Testing list_methods...")
    asyncio.run(test_list_methods_before_generation())
    print("✅ list_methods before generation works\n")

    asyncio.run(test_print_methods())
    print("✅ print_methods works\n")

    print("🎉 All list_methods tests pass!")
