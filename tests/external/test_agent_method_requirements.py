# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for agent method requirements:
1. All methods on an agent must be ellipsis methods or implemented methods
2. Ellipsis methods without generation can be called externally as entry points
"""

import pytest

from nooa import strategy
from nooa.agent import Agent
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient

# Module-level test LLM (can be overridden at instantiation)
_TEST_LLM = FakeLLMClient()

# ==============================================================================
# Requirement 1: All methods must be decorated (or private/special)
# ==============================================================================


def test_private_methods_allowed_without_decorator():
    """Test that private methods (_method) don't need decorators."""

    # This should not raise an error

    class GoodAgent(Agent, llm=_TEST_LLM):
        async def public_method(self) -> None:
            """Public method."""
            ...

        # Private methods are allowed without decorators
        def _private_helper(self):
            """Private helper."""
            return 42

        async def _async_private_helper(self):
            """Async private helper."""
            return "data"


def test_special_methods_allowed_without_decorator():
    """Test that special methods (__init__, __str__) don't need decorators."""

    # This should not raise an error

    class GoodAgent(Agent, llm=_TEST_LLM):
        def __init__(self):
            """Custom init."""
            super().__init__()
            self.data = []

        def __str__(self):
            """String representation."""
            return "GoodAgent"

        async def process(self) -> None:
            """Process data."""
            ...


def test_callable_tools_allowed_without_decorator():
    """Test that callable tools (standalone functions) don't need decorators."""

    # Standalone function (callable tool)

    def helper_function(x: int) -> int:
        """Helper function."""
        return x * 2

    # This should not raise an error

    class GoodAgent(Agent, llm=_TEST_LLM):
        # Assign callable tool as class attribute
        helper = helper_function

        async def process(self, x: int) -> int:
            """Process using helper."""
            return self.helper(x)


def test_all_decorator_types_accepted():
    """Test that agent accepts all three decorator types."""

    # This should not raise an error

    class CompleteAgent(Agent, llm=_TEST_LLM):
        async def generate_something(self) -> str:
            """Generate something."""
            ...

        @strategy(PurePythonStrategy())
        async def generate_quick(self) -> str:
            """Generate quickly."""
            ...

        async def on_event(self, data: str) -> None:
            """Handle event."""
            self.last_event = data

        def get_status(self) -> dict:
            """Get status."""
            return {"status": "ok"}


# ==============================================================================
# Requirement 2: @strategy methods without generation are external entry points
# ==============================================================================


@pytest.mark.asyncio
async def test_plan_with_implementation_is_entry_point():
    """Test that @strategy method with implementation can be called externally."""

    class CalculatorAgent(Agent, llm=_TEST_LLM):
        async def add(self, a: int, b: int) -> int:
            """Add two numbers (implemented method, not generated)."""
            return a + b

    # Instantiate and call
    calc = CalculatorAgent()

    result = await calc.add(5, 3)
    assert result == 8


@pytest.mark.asyncio
async def test_plan_without_generation_executes_directly():
    """Test that @strategy method with implementation executes directly (no LLM)."""

    class DataAgent(Agent, llm=_TEST_LLM):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.call_count = 0

        async def increment(self) -> int:
            """Increment counter (no LLM needed)."""
            self.call_count += 1
            return self.call_count

    data_agent = DataAgent()

    # Call multiple times - should execute directly
    assert await data_agent.increment() == 1
    assert await data_agent.increment() == 2
    assert await data_agent.increment() == 3


@pytest.mark.asyncio
async def test_plan_implementation_can_call_other_plans():
    """Test that @strategy method with implementation can call other @strategy methods."""

    class OrchestratorAgent(Agent, llm=_TEST_LLM):
        async def validate_data(self, data: list) -> bool:
            """Validate data (implemented)."""
            return len(data) > 0

        async def process_data(self, data: list) -> dict:
            """Process data by delegating to validator."""
            is_valid = await self.validate_data(data)
            if is_valid:
                return {"status": "processed", "count": len(data)}
            return {"status": "invalid"}

    orch_agent = OrchestratorAgent()

    # Call implemented method that calls another implemented method
    result = await orch_agent.process_data([1, 2, 3])
    assert result == {"status": "processed", "count": 3}

    result = await orch_agent.process_data([])
    assert result == {"status": "invalid"}


@pytest.mark.asyncio
async def test_mixed_generator_and_implemented_methods():
    """Test agent with both generator and implemented @strategy methods."""

    class MixedAgent(Agent, llm=_TEST_LLM):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.results = []

        async def store_result(self, value: str) -> None:
            """Store a result (implemented)."""
            self.results.append(value)

        async def process(self, data: str) -> str:
            """Process data (generated)."""
            ...

    # Both methods should be accessible as entry points
    mixed_agent = MixedAgent()

    # Can call implemented method
    await mixed_agent.store_result("test1")
    assert mixed_agent.results == ["test1"]

    # Generator method exists and is callable (even if it needs LLM)
    assert hasattr(mixed_agent, "process")
    assert callable(mixed_agent.process)


@pytest.mark.asyncio
async def test_strategy_on_implemented_method_is_allowed():
    """Test that @strategy on implemented method is allowed (marks as entry point)."""

    # This should NOT raise - @strategy on implemented methods is valid
    class GoodAgent(Agent, llm=_TEST_LLM):
        @strategy(PurePythonStrategy())  # Valid - marks as entry point
        async def implemented_method(self) -> int:
            """This has implementation and specifies strategy."""
            return 42

    agent = GoodAgent()
    result = await agent.implemented_method()
    assert result == 42


@pytest.mark.asyncio
async def test_plan_implementation_with_multiple_calls():
    """Test that implemented @strategy method can be called multiple times."""

    class CounterAgent(Agent, llm=_TEST_LLM):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.count = 0

        async def increment(self, amount: int = 1) -> int:
            """Increment counter."""
            self.count += amount
            return self.count

    counter_agent = CounterAgent()

    # Multiple calls
    assert await counter_agent.increment() == 1
    assert await counter_agent.increment(5) == 6
    assert await counter_agent.increment(10) == 16


@pytest.mark.asyncio
async def test_plan_implementation_can_use_asyncio():
    """Test that implemented @strategy method can use asyncio operations."""
    import asyncio

    class AsyncAgent(Agent, llm=_TEST_LLM):
        async def wait_and_return(self, value: str, delay: float = 0.001) -> str:
            """Wait for delay and return value."""
            await asyncio.sleep(delay)
            return f"processed: {value}"

    async_agent = AsyncAgent()

    result = await async_agent.wait_and_return("test")
    assert result == "processed: test"


@pytest.mark.asyncio
async def test_plan_implementation_entry_point_metadata():
    """Test that ellipsis and implemented methods have correct metadata from metaclass.

    Since Agent has _enable_tracing = True by default, ALL methods are wrapped.
    """

    class MetadataAgent(Agent, llm=_TEST_LLM):
        async def implemented(self) -> str:
            """Implemented method."""
            return "result"

        async def generator(self) -> str:
            """Generator method."""
            ...

    # Implemented methods ARE wrapped (for tracing)
    assert hasattr(MetadataAgent.implemented, "_agent_decorator")
    assert MetadataAgent.implemented._agent_decorator == "auto"
    assert MetadataAgent.implemented._needs_generation is False  # Tracing-only

    # Generator methods ARE wrapped (for generation + tracing)
    assert hasattr(MetadataAgent.generator, "_agent_decorator")
    assert MetadataAgent.generator._agent_decorator == "auto"
    assert MetadataAgent.generator._needs_generation is True


@pytest.mark.asyncio
async def test_plan_implementation_as_coordinator_entry():
    """Test using implemented ellipsis method as coordinator entry point."""

    class WorkerAgent(Agent, llm=_TEST_LLM):
        async def work(self, task_id: int) -> dict:
            """Do work."""
            ...

    class CoordinatorAgent(Agent, llm=_TEST_LLM):
        # Entry point - implemented method that delegates to generators
        async def coordinate(self, task_ids: list[int]) -> list[dict]:
            """Coordinate multiple workers (entry point)."""
            results = []
            for task_id in task_ids:
                # Would normally create workers and call them
                # For this test, just simulate
                results.append({"task_id": task_id, "status": "queued"})
            return results

    coordinator = CoordinatorAgent()

    # Use implemented method as entry point
    results = await coordinator.coordinate([1, 2, 3])
    assert len(results) == 3
    assert results[0] == {"task_id": 1, "status": "queued"}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
