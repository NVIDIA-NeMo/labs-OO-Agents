# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for AgentMeta metaclass and auto-wrapping functionality."""

import pytest

from nooa.agent import Agent
from nooa.config.strategy_config import ReflexionConfig
from nooa.decorators import strategy
from nooa.metaclass import no_trace
from nooa.strategies import ReflexionStrategy
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient

_TEST_LLM = FakeLLMClient()


def test_auto_wrap_public_ellipsis():
    """Public ellipsis methods are auto-wrapped."""

    class TestAgent(Agent, llm=_TEST_LLM):
        async def task(self): ...

    assert hasattr(TestAgent.task, "_agent_decorator")
    assert TestAgent.task._agent_decorator == "auto"
    assert TestAgent.task._needs_generation is True


def test_auto_wrap_public_implemented():
    """Public implemented methods ARE auto-wrapped for tracing when _enable_tracing=True."""

    class TestAgent(Agent, llm=_TEST_LLM):
        async def task(self):
            return "implemented"

    # Agent has _enable_tracing = True by default, so regular methods are wrapped for tracing
    assert hasattr(TestAgent.task, "_agent_decorator")
    assert TestAgent.task._agent_decorator == "auto"
    # But they don't need generation (not ellipsis)
    assert TestAgent.task._needs_generation is False


def test_private_ellipsis_generated_and_traced():
    """Private ellipsis methods are generated AND traced by default."""

    class TestAgent(Agent, llm=_TEST_LLM):
        async def _helper(self): ...

    assert hasattr(TestAgent._helper, "_agent_decorator")
    assert TestAgent._helper._needs_generation is True
    # Private methods are now traced by default (use @no_trace to opt-out)


def test_dunder_ellipsis_generated_and_traced():
    """Dunder ellipsis methods are generated AND traced by default."""

    class TestAgent(Agent, llm=_TEST_LLM):
        async def __custom__(self): ...

    assert hasattr(TestAgent.__custom__, "_agent_decorator")
    assert TestAgent.__custom__._needs_generation is True
    # Dunder methods are now traced by default (use @no_trace to opt-out)


def test_no_trace_decorator():
    """@no_trace prevents tracing but not generation."""

    class TestAgent(Agent, llm=_TEST_LLM):
        @no_trace
        async def task(self): ...

    assert hasattr(TestAgent.task, "_agent_decorator")
    assert TestAgent.task._needs_generation is True
    # Method is wrapped but won't have tracing hooks


def test_strategy_decorator_override():
    """@strategy overrides default strategy."""
    reflexion = ReflexionStrategy(config=ReflexionConfig(max_reflections=3))

    class TestAgent(Agent, llm=_TEST_LLM):
        @strategy(reflexion)
        async def task(self): ...

    assert TestAgent.task._plan_strategy is reflexion


def test_default_strategy():
    """Methods default to PurePythonStrategy (resolved at runtime)."""

    class TestAgent(Agent, llm=_TEST_LLM):
        async def task(self): ...

    # Strategy is None at class creation time (lazy resolution)
    # It will be resolved to PurePythonStrategy at runtime
    assert TestAgent.task._plan_strategy is None


def test_llm_class_level():
    """Agent subclass can specify llm at class level."""

    class TestAgent(Agent, llm=_TEST_LLM):
        async def task(self): ...

    agent = TestAgent()
    assert agent._llm is _TEST_LLM


def test_llm_instance_override():
    """Instance llm parameter overrides class llm."""

    class TestAgent(Agent, llm=_TEST_LLM):
        async def task(self): ...

    custom_llm = FakeLLMClient()
    agent = TestAgent(llm=custom_llm)
    assert agent._llm is custom_llm


def test_llm_not_required_at_class_level():
    """Agent can be defined without class-level llm (provide at instantiation)."""

    class TestAgent(Agent):
        async def task(self): ...

    # Should be able to create class without error
    agent = TestAgent(llm=_TEST_LLM)
    assert agent._llm is _TEST_LLM


def test_method_inheritance():
    """Overridden methods wrap independently."""

    class BaseAgent(Agent, llm=_TEST_LLM):
        async def task(self): ...

    class ChildAgent(BaseAgent):
        async def task(self): ...

    # Both should be wrapped independently
    assert hasattr(BaseAgent.task, "_agent_decorator")
    assert hasattr(ChildAgent.task, "_agent_decorator")


def test_sync_methods_wrapped_for_tracing_when_enable_tracing():
    """Sync methods on Agent subclasses are wrapped for tracing (issue 181)."""

    class TestAgent(Agent, llm=_TEST_LLM):
        def sync_method(self):
            return "sync"

    # Agent has _enable_tracing = True, so sync methods are wrapped for tracing
    assert hasattr(TestAgent.sync_method, "_agent_decorator")
    assert TestAgent.sync_method._agent_decorator == "auto"
    assert TestAgent.sync_method._needs_generation is False
    # Wrapper preserves the original for introspection
    assert TestAgent.sync_method._original.__name__ == "sync_method"


def test_no_wrap_inherited_methods():
    """Inherited methods are not re-wrapped."""

    class BaseAgent(Agent, llm=_TEST_LLM):
        async def task(self): ...

    class ChildAgent(BaseAgent):
        pass  # Doesn't override task

    # Only base should have wrapper
    assert hasattr(BaseAgent.task, "_agent_decorator")
    # Child inherits the same wrapper (not re-wrapped)
    assert BaseAgent.task is ChildAgent.task


def test_strategy_decorator_stacking_error():
    """Multiple @strategy decorators raise error."""
    with pytest.raises(ValueError, match="Cannot stack multiple @strategy"):

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy())
            @strategy(ReflexionStrategy())
            async def task(self): ...


def test_context_manager_available():
    """Agent instances have a dict-like context manager."""

    class TestAgent(Agent, llm=_TEST_LLM):
        async def task(self): ...

    agent = TestAgent()
    # Context manager supports dict-like API
    agent.context_manager["test_key"] = "test_value"
    assert "test_key" in agent.context_manager
    assert agent.context_manager["test_key"] == "test_value"

    # DynamicContext values via set_dynamic()
    agent.context_manager.set_dynamic("dynamic_key", "'dynamic'")
    assert "dynamic_key" in agent.context_manager


def test_multiple_inheritance_support():
    """Agent works with multiple inheritance."""

    class Mixin:
        def mixin_method(self):
            return "mixin"

    class TestAgent(Agent, Mixin, llm=_TEST_LLM):
        async def task(self): ...

    agent = TestAgent()
    assert agent.mixin_method() == "mixin"
    assert hasattr(TestAgent.task, "_agent_decorator")


def test_strategy_class_autowrap():
    """GenerationStrategy methods also auto-wrap."""
    from nooa.strategies.base import GenerationStrategy

    class CustomStrategy(GenerationStrategy):
        async def generate_code(self, runtime): ...

    # Should be auto-wrapped
    assert hasattr(CustomStrategy.generate_code, "_agent_decorator")
    assert CustomStrategy.generate_code._needs_generation is True


def test_strategy_methods_not_traced():
    """Strategy methods are never traced (only Agent methods)."""
    from nooa.strategies.base import GenerationStrategy

    class CustomStrategy(GenerationStrategy):
        async def public_method(self): ...

    # Wrapped for generation, but should skip tracing (not Agent)
    assert hasattr(CustomStrategy.public_method, "_agent_decorator")
    # This is a strategy, so public methods shouldn't be traced


def test_private_methods_traced_by_default():
    """Private methods are traced by default (need @no_trace to opt-out)."""

    class TestAgent(Agent, llm=_TEST_LLM):
        async def _private(self): ...

    # Private methods are wrapped and traced by default
    assert hasattr(TestAgent._private, "_agent_decorator")
    # To skip tracing, use @no_trace decorator explicitly


def test_private_method_opt_out_with_no_trace():
    """Private methods can opt out of tracing with @no_trace."""

    class TestAgent(Agent, llm=_TEST_LLM):
        @no_trace
        async def _helper(self): ...

    assert hasattr(TestAgent._helper, "_agent_decorator")
    assert TestAgent._helper._needs_generation is True
    # Method has @no_trace, so it should have _no_trace attribute
    assert getattr(TestAgent._helper._original, "_no_trace", False) is True


def test_dunder_method_opt_out_with_no_trace():
    """Dunder methods can opt out of tracing with @no_trace."""

    class TestAgent(Agent, llm=_TEST_LLM):
        @no_trace
        async def __custom__(self): ...

    assert hasattr(TestAgent.__custom__, "_agent_decorator")
    assert TestAgent.__custom__._needs_generation is True
    # Dunder method with @no_trace should have _no_trace attribute
    assert getattr(TestAgent.__custom__._original, "_no_trace", False) is True


def test_original_func_preserved():
    """Wrapped methods preserve reference to original function."""

    class TestAgent(Agent, llm=_TEST_LLM):
        async def task(self): ...

    # Should preserve original for introspection
    assert hasattr(TestAgent.task, "_original")


def test_no_reserved_parameter_names():
    """No parameter names are reserved (reasoning() builtin was removed)."""

    class TestAgent(Agent, llm=_TEST_LLM):
        async def task(self, reasoning: str): ...

    assert hasattr(TestAgent.task, "_original")


def test_compatible_with_existing_decorators():
    """Metaclass works alongside @strategy decorator."""

    # Ellipsis methods with @strategy decorator should work

    class AgentWithStrategy(Agent, llm=_TEST_LLM):
        @strategy(PurePythonStrategy())
        async def task(self): ...

    # Without decorator should also work
    class AgentWithoutDecorator(Agent, llm=_TEST_LLM):
        async def task(self): ...

    # Both should be functional
    assert hasattr(AgentWithStrategy, "_agent_llm")
    assert hasattr(AgentWithoutDecorator, "_agent_llm")


def test_strategy_decorator_preserved_not_replaced():
    """Metaclass does NOT replace @strategy decorator wrapper.

    This is a regression test for the missing AGENT spans bug.
    The metaclass was incorrectly replacing @strategy wrappers because
    inspect.getsource() follows __wrapped__ to the original ellipsis body.
    """

    class TestAgent(Agent, llm=_TEST_LLM):
        @strategy(PurePythonStrategy(max_iterations=5))
        async def task(self): ...

    # The wrapper should be from @strategy, not metaclass
    # @strategy sets _agent_decorator to "auto" on its wrapper
    assert hasattr(TestAgent.task, "_agent_decorator")
    assert TestAgent.task._agent_decorator == "auto"

    # Strategy should be preserved (PurePythonStrategy with max_iterations=5)
    assert hasattr(TestAgent.task, "_plan_strategy")
    assert isinstance(TestAgent.task._plan_strategy, PurePythonStrategy)
    assert TestAgent.task._plan_strategy.max_iterations == 5


@pytest.mark.asyncio
async def test_metaclass_wrapper_calls_hooks():
    """Metaclass wrapper calls instrumentation hooks for AGENT spans.

    This is a regression test for the missing AGENT spans bug.
    Auto-wrapped methods must call before_agent_call/after_agent_call hooks.
    """
    from unittest.mock import MagicMock

    from nooa.runtime.hooks import InstrumentationHooks, set_hooks

    # Create mock hooks
    mock_hooks = MagicMock(spec=InstrumentationHooks)
    mock_hooks.before_agent_call.return_value = {"test": "context"}

    # Define agent without explicit @strategy decorator
    class TestAgent(Agent, llm=_TEST_LLM):
        async def task(self, x: int) -> int:
            """Compute: {x}"""
            ...

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()

        # Call the method - it will fail due to FakeLLMClient, but hooks should be called
        try:
            await agent.task(42)
        except Exception:
            pass  # Expected - FakeLLMClient doesn't implement real LLM calls

        # Verify before_agent_call was called with correct arguments
        mock_hooks.before_agent_call.assert_called_once()
        call_kwargs = mock_hooks.before_agent_call.call_args.kwargs
        assert call_kwargs["agent"] is agent
        assert call_kwargs["method_name"] == "task"
        assert call_kwargs["args"] == (42,)
        assert "call_id" in call_kwargs
        assert "parent_call_id" in call_kwargs

        # Verify after_agent_call was called
        mock_hooks.after_agent_call.assert_called_once()
        after_kwargs = mock_hooks.after_agent_call.call_args.kwargs
        assert after_kwargs["agent"] is agent
        assert after_kwargs["method_name"] == "task"
        assert after_kwargs["context"] == {"test": "context"}

    finally:
        set_hooks(None)


@pytest.mark.asyncio
async def test_private_methods_call_hooks():
    """Private methods call instrumentation hooks for tracing.

    This test verifies that private methods (starting with _) are traced
    by default and call before_agent_call/after_agent_call hooks.
    """
    from unittest.mock import MagicMock

    from nooa.runtime.hooks import InstrumentationHooks, set_hooks

    # Create mock hooks
    mock_hooks = MagicMock(spec=InstrumentationHooks)
    mock_hooks.before_agent_call.return_value = {"test": "context"}

    # Define agent with private method
    class TestAgent(Agent, llm=_TEST_LLM):
        async def _private_helper(self, x: int) -> int:
            """Private helper: {x}"""
            ...

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()

        # Call the private method - it will fail due to FakeLLMClient, but hooks should be called
        try:
            await agent._private_helper(42)
        except Exception:
            pass  # Expected - FakeLLMClient doesn't implement real LLM calls

        # Verify before_agent_call was called with correct arguments
        mock_hooks.before_agent_call.assert_called_once()
        call_kwargs = mock_hooks.before_agent_call.call_args.kwargs
        assert call_kwargs["agent"] is agent
        assert call_kwargs["method_name"] == "_private_helper"
        assert call_kwargs["args"] == (42,)
        assert "call_id" in call_kwargs
        assert "parent_call_id" in call_kwargs

        # Verify after_agent_call was called
        mock_hooks.after_agent_call.assert_called_once()
        after_kwargs = mock_hooks.after_agent_call.call_args.kwargs
        assert after_kwargs["agent"] is agent
        assert after_kwargs["method_name"] == "_private_helper"
        assert after_kwargs["context"] == {"test": "context"}

    finally:
        set_hooks(None)


@pytest.mark.asyncio
async def test_strategy_decorator_calls_hooks():
    """@strategy decorator calls instrumentation hooks for AGENT spans."""
    from unittest.mock import MagicMock

    from nooa.runtime.hooks import InstrumentationHooks, set_hooks

    # Create mock hooks
    mock_hooks = MagicMock(spec=InstrumentationHooks)
    mock_hooks.before_agent_call.return_value = {"test": "context"}

    # Define agent with explicit @strategy decorator
    class TestAgent(Agent, llm=_TEST_LLM):
        @strategy(PurePythonStrategy(max_iterations=3))
        async def task(self, x: int) -> int:
            """Compute: {x}"""
            ...

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()

        # Call the method - it will fail due to FakeLLMClient, but hooks should be called
        try:
            await agent.task(42)
        except Exception:
            pass  # Expected - FakeLLMClient doesn't implement real LLM calls

        # Verify before_agent_call was called with correct arguments
        mock_hooks.before_agent_call.assert_called_once()
        call_kwargs = mock_hooks.before_agent_call.call_args.kwargs
        assert call_kwargs["agent"] is agent
        assert call_kwargs["method_name"] == "task"
        assert call_kwargs["args"] == (42,)
        assert "call_id" in call_kwargs
        assert "parent_call_id" in call_kwargs

        # Verify after_agent_call was called
        mock_hooks.after_agent_call.assert_called_once()
        after_kwargs = mock_hooks.after_agent_call.call_args.kwargs
        assert after_kwargs["agent"] is agent
        assert after_kwargs["method_name"] == "task"
        assert after_kwargs["context"] == {"test": "context"}

    finally:
        set_hooks(None)


def test_auto_wrapped_equivalent_to_strategy_decorated():
    """Auto-wrapped methods have same attributes as @strategy decorated methods.

    This ensures the metaclass wrapper is functionally equivalent to
    the @strategy decorator for instrumentation and execution purposes.
    """

    # Agent with explicit @strategy decorator
    class DecoratedAgent(Agent, llm=_TEST_LLM):
        @strategy(PurePythonStrategy(max_iterations=5))
        async def task(self): ...

    # Agent relying on metaclass auto-wrapping (no decorator)
    class AutoWrappedAgent(Agent, llm=_TEST_LLM):
        async def task(self): ...

    decorated_method = DecoratedAgent.task
    auto_method = AutoWrappedAgent.task

    # Both should have _agent_decorator attribute
    assert hasattr(decorated_method, "_agent_decorator")
    assert hasattr(auto_method, "_agent_decorator")
    assert decorated_method._agent_decorator == "auto"
    assert auto_method._agent_decorator == "auto"

    # Both should have _needs_generation = True (ellipsis body)
    assert hasattr(decorated_method, "_needs_generation")
    assert hasattr(auto_method, "_needs_generation")
    assert decorated_method._needs_generation is True
    assert auto_method._needs_generation is True

    # Both should have _plan_strategy attribute
    # (decorated has explicit strategy, auto has None for lazy resolution)
    assert hasattr(decorated_method, "_plan_strategy")
    assert hasattr(auto_method, "_plan_strategy")
    assert isinstance(decorated_method._plan_strategy, PurePythonStrategy)
    assert auto_method._plan_strategy is None  # Resolved at runtime


@pytest.mark.asyncio
async def test_auto_wrapped_and_decorated_both_call_hooks_same_way():
    """Both auto-wrapped and @strategy decorated methods call hooks identically.

    This is the key equivalence test - both wrapping mechanisms must
    produce the same hook call pattern for instrumentation to work.
    """
    from unittest.mock import MagicMock

    from nooa.runtime.hooks import InstrumentationHooks, set_hooks

    # Agent with explicit @strategy decorator
    class DecoratedAgent(Agent, llm=_TEST_LLM):
        @strategy(PurePythonStrategy())
        async def task(self, x: int): ...

    # Agent relying on metaclass auto-wrapping
    class AutoWrappedAgent(Agent, llm=_TEST_LLM):
        async def task(self, x: int): ...

    # Test decorated agent
    mock_hooks = MagicMock(spec=InstrumentationHooks)
    mock_hooks.before_agent_call.return_value = {"ctx": "decorated"}

    try:
        set_hooks(mock_hooks)
        decorated_agent = DecoratedAgent()
        try:
            await decorated_agent.task(100)
        except Exception:
            pass

        decorated_before_kwargs = mock_hooks.before_agent_call.call_args.kwargs
        decorated_after_kwargs = mock_hooks.after_agent_call.call_args.kwargs
    finally:
        set_hooks(None)

    # Test auto-wrapped agent
    mock_hooks.reset_mock()
    mock_hooks.before_agent_call.return_value = {"ctx": "auto"}

    try:
        set_hooks(mock_hooks)
        auto_agent = AutoWrappedAgent()
        try:
            await auto_agent.task(100)
        except Exception:
            pass

        auto_before_kwargs = mock_hooks.before_agent_call.call_args.kwargs
        auto_after_kwargs = mock_hooks.after_agent_call.call_args.kwargs
    finally:
        set_hooks(None)

    # Both should call hooks with same REQUIRED keys
    # (metaclass may pass extra strategy.* keys, which is fine)
    required_before_keys = {"agent", "method_name", "args", "kwargs", "call_id", "parent_call_id"}
    assert required_before_keys <= set(decorated_before_kwargs.keys())
    assert required_before_keys <= set(auto_before_kwargs.keys())

    # Both should pass same args
    assert decorated_before_kwargs["args"] == (100,)
    assert auto_before_kwargs["args"] == (100,)

    # Both should have same method name
    assert decorated_before_kwargs["method_name"] == "task"
    assert auto_before_kwargs["method_name"] == "task"

    # Check after_agent_call has same REQUIRED keys
    required_after_keys = {"agent", "method_name", "result", "exception", "context"}
    assert required_after_keys <= set(decorated_after_kwargs.keys())
    assert required_after_keys <= set(auto_after_kwargs.keys())


# ============================================================================
# TDD Tests for Trace-All-Methods Feature
# ============================================================================


@pytest.mark.asyncio
async def test_regular_python_methods_are_traced_when_enable_tracing():
    """Regular Python methods (non-ellipsis) should be traced when _enable_tracing = True."""
    from unittest.mock import MagicMock

    from nooa.runtime.hooks import InstrumentationHooks, set_hooks

    # Create mock hooks
    mock_hooks = MagicMock(spec=InstrumentationHooks)
    mock_hooks.before_agent_call.return_value = {"test": "context"}

    # Define agent with _enable_tracing = True
    class TestAgent(Agent, llm=_TEST_LLM):
        _enable_tracing = True

        async def regular_method(self, x: int) -> int:
            """This is a regular Python method, not ellipsis."""
            return x * 2

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()

        # Call the regular Python method
        result = await agent.regular_method(21)

        # Verify result is correct
        assert result == 42

        # Verify before_agent_call was called for the regular method
        mock_hooks.before_agent_call.assert_called_once()
        call_kwargs = mock_hooks.before_agent_call.call_args.kwargs
        assert call_kwargs["agent"] is agent
        assert call_kwargs["method_name"] == "regular_method"
        assert call_kwargs["args"] == (21,)

        # Verify after_agent_call was called
        mock_hooks.after_agent_call.assert_called_once()
        after_kwargs = mock_hooks.after_agent_call.call_args.kwargs
        assert after_kwargs["agent"] is agent
        assert after_kwargs["method_name"] == "regular_method"
        assert after_kwargs["result"] == 42

    finally:
        set_hooks(None)


@pytest.mark.asyncio
async def test_private_methods_are_traced_when_enable_tracing():
    """Private methods should be traced when _enable_tracing = True."""
    from unittest.mock import MagicMock

    from nooa.runtime.hooks import InstrumentationHooks, set_hooks

    # Create mock hooks
    mock_hooks = MagicMock(spec=InstrumentationHooks)
    mock_hooks.before_agent_call.return_value = {"test": "context"}

    # Define agent with _enable_tracing = True
    class TestAgent(Agent, llm=_TEST_LLM):
        _enable_tracing = True

        async def _private_helper(self, x: int) -> int:
            """Private method."""
            return x + 1

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()

        # Call the private method
        result = await agent._private_helper(41)

        # Verify result is correct
        assert result == 42

        # Verify before_agent_call was called for the private method
        mock_hooks.before_agent_call.assert_called_once()
        call_kwargs = mock_hooks.before_agent_call.call_args.kwargs
        assert call_kwargs["agent"] is agent
        assert call_kwargs["method_name"] == "_private_helper"
        assert call_kwargs["args"] == (41,)

        # Verify after_agent_call was called
        mock_hooks.after_agent_call.assert_called_once()

    finally:
        set_hooks(None)


@pytest.mark.asyncio
async def test_no_trace_decorator_works_on_regular_methods():
    """@no_trace should prevent tracing on regular Python methods."""
    from unittest.mock import MagicMock

    from nooa.runtime.hooks import InstrumentationHooks, set_hooks

    # Create mock hooks
    mock_hooks = MagicMock(spec=InstrumentationHooks)

    # Define agent with _enable_tracing = True and @no_trace
    class TestAgent(Agent, llm=_TEST_LLM):
        _enable_tracing = True

        @no_trace
        async def untraced_method(self, x: int) -> int:
            """This should not be traced."""
            return x * 2

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()

        # Call the method
        result = await agent.untraced_method(21)

        # Verify result is correct
        assert result == 42

        # Verify hooks were NOT called
        mock_hooks.before_agent_call.assert_not_called()
        mock_hooks.after_agent_call.assert_not_called()

    finally:
        set_hooks(None)


@pytest.mark.asyncio
async def test_parent_child_relationship_regular_calls_ellipsis():
    """Regular method calling ellipsis method should create parent-child span relationship."""
    from unittest.mock import MagicMock

    from nooa.runtime.hooks import InstrumentationHooks, set_hooks

    # Track all hook calls
    hook_calls = []

    def mock_before_agent_call(**kwargs):
        hook_calls.append(("before", kwargs))
        return {"call_id": kwargs["call_id"]}

    def mock_after_agent_call(**kwargs):
        hook_calls.append(("after", kwargs))

    # Create mock hooks
    mock_hooks = MagicMock(spec=InstrumentationHooks)
    mock_hooks.before_agent_call.side_effect = mock_before_agent_call
    mock_hooks.after_agent_call.side_effect = mock_after_agent_call

    # Define agent with both regular and ellipsis methods
    class TestAgent(Agent, llm=_TEST_LLM):
        _enable_tracing = True

        async def orchestrator(self, x: int) -> str:
            """Regular method that calls ellipsis method."""
            result = await self.compute(x)
            return f"Result: {result}"

        async def compute(self, x: int) -> int:
            """Compute: {x} * 2"""
            ...

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()

        # Call the orchestrator - will fail on LLM call but we can check hook pattern
        try:
            await agent.orchestrator(42)
        except Exception:
            pass  # Expected - FakeLLMClient doesn't implement real LLM calls

        # Should have at least 2 before_agent_call calls (orchestrator and compute)
        before_calls = [call for call in hook_calls if call[0] == "before"]
        assert len(before_calls) >= 2

        # First call should be orchestrator
        first_call = before_calls[0][1]
        assert first_call["method_name"] == "orchestrator"
        orchestrator_call_id = first_call["call_id"]
        assert first_call["parent_call_id"] is None  # Top-level call

        # Second call should be compute with orchestrator as parent
        second_call = before_calls[1][1]
        assert second_call["method_name"] == "compute"
        assert second_call["parent_call_id"] == orchestrator_call_id  # Child of orchestrator

    finally:
        set_hooks(None)


@pytest.mark.asyncio
async def test_regular_methods_capture_source_code():
    """Regular Python methods should capture their source code in traces."""
    from unittest.mock import MagicMock

    from nooa.runtime.hooks import InstrumentationHooks, set_hooks

    # Create mock hooks
    mock_hooks = MagicMock(spec=InstrumentationHooks)
    mock_hooks.before_agent_call.return_value = {"test": "context"}

    # Define agent with _enable_tracing = True
    class TestAgent(Agent, llm=_TEST_LLM):
        _enable_tracing = True

        async def orchestrator(self, x: int) -> int:
            """Regular method with actual code."""
            result = x * 2
            return result + 1

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()

        # Call the method
        result = await agent.orchestrator(20)

        # Verify result is correct
        assert result == 41

        # Verify before_agent_call was called
        mock_hooks.before_agent_call.assert_called_once()
        call_kwargs = mock_hooks.before_agent_call.call_args.kwargs

        # Verify source code was captured
        assert "source_code" in call_kwargs
        source_code = call_kwargs["source_code"]

        # Should contain the method body
        assert "result = x * 2" in source_code
        assert "return result + 1" in source_code

        # Verify input (args) was captured
        assert call_kwargs["args"] == (20,)
        assert call_kwargs["kwargs"] == {}

        # Verify output (result) was captured in after_agent_call
        after_kwargs = mock_hooks.after_agent_call.call_args.kwargs
        assert after_kwargs["result"] == 41

    finally:
        set_hooks(None)


# ============================================================================
# Tests for @strategy(context=...) decorator
# ============================================================================


def test_strategy_decorator_context_stored():
    """@strategy(context=...) stores context on the wrapper."""
    from nooa.context_blocks import ScopedContext

    class TestAgent(Agent, llm=_TEST_LLM):
        @strategy(PurePythonStrategy(), context=ScopedContext(context={"focus": "testing"}))
        async def task(self): ...

    assert TestAgent.task._strategy_context == {"focus": "testing"}


def test_strategy_decorator_context_with_dynamic():
    """@strategy(context=...) supports DynamicContext values."""
    from nooa.context_blocks import DynamicContext, ScopedContext

    class TestAgent(Agent, llm=_TEST_LLM):
        @strategy(context=ScopedContext(context={"status": DynamicContext("self.get_status()")}))
        async def task(self): ...

    ctx = TestAgent.task._strategy_context
    assert isinstance(ctx["status"], DynamicContext)
    assert ctx["status"].expr == "self.get_status()"


def test_strategy_decorator_context_none_default():
    """@strategy without context= has None context."""

    class TestAgent(Agent, llm=_TEST_LLM):
        @strategy(PurePythonStrategy())
        async def task(self): ...

    assert TestAgent.task._strategy_context is None


@pytest.mark.asyncio
async def test_no_trace_outer_strategy_inner_ordering_also_suppresses_hooks():
    """@no_trace applied AFTER @strategy (outer decorator) must also suppress hooks.

    Both orderings must work:
      @strategy @no_trace  (no_trace inner — verified by other tests)
      @no_trace @strategy  (no_trace outer — this test)

    Previously the outer-order silently failed: @strategy baked needs_tracing=True
    into its closure before @no_trace ran, so hooks still fired.
    """
    from unittest.mock import MagicMock

    from nooa.runtime.hooks import InstrumentationHooks, set_hooks
    from nooa.unifiedllm import FakeLLMClient, LLMResponse

    def _resp(content):
        return LLMResponse(
            raw_response=None,
            content=content,
            tool_calls=[],
            finish_reason="stop",
            assistant_message={"role": "assistant", "content": content},
        )

    mock_hooks = MagicMock(spec=InstrumentationHooks)
    fake_llm = FakeLLMClient(scripted_responses=[_resp("return 'hi'")])

    class TestAgent(Agent, llm=_TEST_LLM):
        @no_trace  # outer — applied second
        @strategy(PurePythonStrategy())  # inner — applied first
        async def untraced_gen(self) -> str:
            """Return something without an AGENT span."""
            ...

    try:
        set_hooks(mock_hooks)
        agent = TestAgent(llm=fake_llm)
        result = await agent.untraced_gen()
        assert result == "hi"
        mock_hooks.before_agent_call.assert_not_called()
        mock_hooks.after_agent_call.assert_not_called()
    finally:
        set_hooks(None)


@pytest.mark.asyncio
async def test_no_trace_decorator_suppresses_hooks_on_generation_method():
    """@no_trace on a generation method (ellipsis body) must suppress before/after_agent_call hooks."""
    from unittest.mock import MagicMock

    from nooa.runtime.hooks import InstrumentationHooks, set_hooks
    from nooa.unifiedllm import FakeLLMClient, LLMResponse

    def _resp(content):
        return LLMResponse(
            raw_response=None,
            content=content,
            tool_calls=[],
            finish_reason="stop",
            assistant_message={"role": "assistant", "content": content},
        )

    mock_hooks = MagicMock(spec=InstrumentationHooks)
    fake_llm = FakeLLMClient(scripted_responses=[_resp("return 'hello'")])

    class TestAgent(Agent, llm=_TEST_LLM):
        @strategy(PurePythonStrategy())
        @no_trace
        async def untraced_gen(self) -> str:
            """Return a greeting."""
            ...

    try:
        set_hooks(mock_hooks)
        agent = TestAgent(llm=fake_llm)
        result = await agent.untraced_gen()
        assert result == "hello"
        mock_hooks.before_agent_call.assert_not_called()
        mock_hooks.after_agent_call.assert_not_called()
    finally:
        set_hooks(None)


@pytest.mark.asyncio
async def test_no_trace_on_generation_method_in_enable_tracing_class_suppresses_hooks():
    """@no_trace on a generation method in an _enable_tracing class suppresses hooks."""
    from unittest.mock import MagicMock

    from nooa.runtime.hooks import InstrumentationHooks, set_hooks
    from nooa.unifiedllm import FakeLLMClient, LLMResponse

    def _resp(content):
        return LLMResponse(
            raw_response=None,
            content=content,
            tool_calls=[],
            finish_reason="stop",
            assistant_message={"role": "assistant", "content": content},
        )

    mock_hooks = MagicMock(spec=InstrumentationHooks)
    fake_llm = FakeLLMClient(
        scripted_responses=[
            _resp("return 'hello'"),  # untraced_gen
            _resp("return 'world'"),  # traced_gen
        ]
    )

    class TestAgent(Agent, llm=_TEST_LLM):
        _enable_tracing = True

        @strategy(PurePythonStrategy())
        @no_trace
        async def untraced_gen(self) -> str:
            """Return a greeting."""
            ...

        @strategy(PurePythonStrategy())
        async def traced_gen(self) -> str:
            """Return something else."""
            ...

    try:
        set_hooks(mock_hooks)
        agent = TestAgent(llm=fake_llm)
        result = await agent.untraced_gen()
        assert result == "hello"
        mock_hooks.before_agent_call.assert_not_called()
        mock_hooks.after_agent_call.assert_not_called()

        mock_hooks.reset_mock()
        result = await agent.traced_gen()
        assert result == "world"
        mock_hooks.before_agent_call.assert_called_once()
        mock_hooks.after_agent_call.assert_called_once()
    finally:
        set_hooks(None)


@pytest.mark.asyncio
async def test_no_trace_plain_ellipsis_without_strategy_decorator_suppresses_hooks():
    """@no_trace on a plain ellipsis method (no @strategy, metaclass path) suppresses hooks.

    The test verifies that hooks are never called even if the generation itself
    ultimately fails (no valid LLM response configured). The absence of hook calls
    is the contract being tested, not the generation result.
    """
    from unittest.mock import MagicMock

    from nooa.errors import GenerationError
    from nooa.runtime.hooks import InstrumentationHooks, set_hooks

    mock_hooks = MagicMock(spec=InstrumentationHooks)

    class TestAgent(Agent, llm=_TEST_LLM):
        @no_trace
        async def untraced(self) -> str:
            """Return something without an AGENT span."""
            ...

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()
        with pytest.raises(GenerationError):
            await agent.untraced()
        # Hooks must not be called regardless of whether generation succeeds
        mock_hooks.before_agent_call.assert_not_called()
        mock_hooks.after_agent_call.assert_not_called()
    finally:
        set_hooks(None)


# ============================================================================
# Sync method tracing (issue 181)
# ============================================================================


def test_sync_method_traced_when_enable_tracing():
    """Sync methods on Agent subclasses are wrapped and call hooks (issue 181)."""
    from unittest.mock import MagicMock

    from nooa.runtime.hooks import InstrumentationHooks, set_hooks

    mock_hooks = MagicMock(spec=InstrumentationHooks)
    mock_hooks.before_agent_call.return_value = {"test": "context"}

    class TestAgent(Agent, llm=_TEST_LLM):
        def double(self, x: int) -> int:
            return x * 2

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()
        result = agent.double(21)
        assert result == 42

        mock_hooks.before_agent_call.assert_called_once()
        before_kwargs = mock_hooks.before_agent_call.call_args.kwargs
        assert before_kwargs["agent"] is agent
        assert before_kwargs["method_name"] == "double"
        assert before_kwargs["args"] == (21,)
        assert "call_id" in before_kwargs
        assert "parent_call_id" in before_kwargs

        mock_hooks.after_agent_call.assert_called_once()
        after_kwargs = mock_hooks.after_agent_call.call_args.kwargs
        assert after_kwargs["result"] == 42
        assert after_kwargs["context"] == {"test": "context"}
    finally:
        set_hooks(None)


def test_sync_method_returns_value_directly_not_coroutine():
    """Sync wrapping must NOT change the calling convention to async."""
    import inspect as _inspect

    class TestAgent(Agent, llm=_TEST_LLM):
        def get_value(self) -> int:
            return 7

    agent = TestAgent()
    result = agent.get_value()
    assert result == 7
    assert not _inspect.iscoroutine(result)
    # The wrapper itself must be a sync function, not async
    assert not _inspect.iscoroutinefunction(TestAgent.get_value)


def test_sync_private_method_traced_by_default():
    """Private sync methods (e.g. _search) are wrapped and traced (issue 181)."""
    from unittest.mock import MagicMock

    from nooa.runtime.hooks import InstrumentationHooks, set_hooks

    mock_hooks = MagicMock(spec=InstrumentationHooks)
    mock_hooks.before_agent_call.return_value = {}

    class TestAgent(Agent, llm=_TEST_LLM):
        def _scrape(self, url: str) -> str:
            return f"scraped:{url}"

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()
        out = agent._scrape("http://x")
        assert out == "scraped:http://x"
        mock_hooks.before_agent_call.assert_called_once()
        assert mock_hooks.before_agent_call.call_args.kwargs["method_name"] == "_scrape"
    finally:
        set_hooks(None)


def test_sync_method_source_code_captured():
    """source_code attribute is set on before_agent_call for sync methods."""
    from unittest.mock import MagicMock

    from nooa.runtime.hooks import InstrumentationHooks, set_hooks

    mock_hooks = MagicMock(spec=InstrumentationHooks)
    mock_hooks.before_agent_call.return_value = {}

    class TestAgent(Agent, llm=_TEST_LLM):
        def compute(self, x: int) -> int:
            doubled = x * 2
            return doubled + 1

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()
        assert agent.compute(20) == 41
        before_kwargs = mock_hooks.before_agent_call.call_args.kwargs
        assert "source_code" in before_kwargs
        assert "doubled = x * 2" in before_kwargs["source_code"]
        assert "return doubled + 1" in before_kwargs["source_code"]
    finally:
        set_hooks(None)


def test_sync_method_no_trace_decorator_suppresses_hooks():
    """@no_trace on a sync method skips wrapping entirely (no hooks fire)."""
    from unittest.mock import MagicMock

    from nooa.runtime.hooks import InstrumentationHooks, set_hooks

    mock_hooks = MagicMock(spec=InstrumentationHooks)

    class TestAgent(Agent, llm=_TEST_LLM):
        @no_trace
        def helper(self, x: int) -> int:
            return x + 1

    # Sync wrapping is tracing-only; @no_trace skips it entirely.
    assert not hasattr(TestAgent.helper, "_agent_decorator")
    assert getattr(TestAgent.helper, "_no_trace", False) is True

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()
        assert agent.helper(41) == 42
        mock_hooks.before_agent_call.assert_not_called()
        mock_hooks.after_agent_call.assert_not_called()
    finally:
        set_hooks(None)


def test_sync_method_not_traced_when_class_does_not_enable_tracing():
    """Sync methods on a non-tracing class (no _enable_tracing) are not wrapped."""

    class NonAgent(metaclass=type(Agent)):
        def helper(self, x: int) -> int:
            return x

    assert not hasattr(NonAgent.helper, "_agent_decorator")


def test_sync_dunder_methods_not_wrapped():
    """Custom sync dunder methods are NOT wrapped (avoids recursion / __init__ issues)."""

    class TestAgent(Agent, llm=_TEST_LLM):
        def __custom__(self) -> str:
            return "custom"

    assert not hasattr(TestAgent.__custom__, "_agent_decorator")


def test_sync_classmethod_and_staticmethod_not_wrapped():
    """@classmethod and @staticmethod are descriptors, not functions — not wrapped."""

    class TestAgent(Agent, llm=_TEST_LLM):
        @classmethod
        def factory(cls) -> str:
            return "factory"

        @staticmethod
        def util() -> str:
            return "util"

    assert not hasattr(TestAgent.factory, "_agent_decorator")
    assert not hasattr(TestAgent.util, "_agent_decorator")


def test_sync_property_not_wrapped():
    """@property descriptors are not wrapped."""

    class TestAgent(Agent, llm=_TEST_LLM):
        @property
        def label(self) -> str:
            return "x"

    # Access via the class returns the property object itself
    descriptor = TestAgent.__dict__["label"]
    assert not hasattr(descriptor, "_agent_decorator")
    # Calling the property still works on instances
    agent = TestAgent()
    assert agent.label == "x"


def test_sync_method_inheritance():
    """A subclass overriding a parent's sync method gets its own wrapper."""

    class BaseAgent(Agent, llm=_TEST_LLM):
        def helper(self) -> str:
            return "base"

    class ChildAgent(BaseAgent):
        def helper(self) -> str:
            return "child"

    assert hasattr(BaseAgent.helper, "_agent_decorator")
    assert hasattr(ChildAgent.helper, "_agent_decorator")
    assert BaseAgent.helper is not ChildAgent.helper


def test_sync_method_already_decorated_skipped():
    """A sync method that already has _agent_decorator is not re-wrapped."""

    def make_pre_wrapped():
        def already_wrapped(self):
            return "preexisting"

        already_wrapped._agent_decorator = "manual"  # type: ignore[attr-defined]
        return already_wrapped

    pre_wrapped = make_pre_wrapped()

    class TestAgent(Agent, llm=_TEST_LLM):
        helper = pre_wrapped

    # Same object — metaclass left it alone
    assert TestAgent.helper is pre_wrapped
    assert TestAgent.helper._agent_decorator == "manual"


@pytest.mark.asyncio
async def test_sync_child_of_async_parent_has_correct_parent_call_id():
    """Sync helper called inside async method: sync's parent_call_id == async's call_id."""
    from unittest.mock import MagicMock

    from nooa.runtime.hooks import InstrumentationHooks, set_hooks

    hook_calls: list[tuple[str, dict]] = []

    def before(**kwargs):
        hook_calls.append(("before", kwargs))
        return {"call_id": kwargs["call_id"]}

    def after(**kwargs):
        hook_calls.append(("after", kwargs))

    mock_hooks = MagicMock(spec=InstrumentationHooks)
    mock_hooks.before_agent_call.side_effect = before
    mock_hooks.after_agent_call.side_effect = after

    class TestAgent(Agent, llm=_TEST_LLM):
        async def orchestrator(self) -> str:
            return self.sync_helper("ok")

        def sync_helper(self, value: str) -> str:
            return f"helper:{value}"

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()
        result = await agent.orchestrator()
        assert result == "helper:ok"
    finally:
        set_hooks(None)

    befores = [kw for tag, kw in hook_calls if tag == "before"]
    method_names = [b["method_name"] for b in befores]
    assert "orchestrator" in method_names
    assert "sync_helper" in method_names
    orchestrator_before = next(b for b in befores if b["method_name"] == "orchestrator")
    sync_before = next(b for b in befores if b["method_name"] == "sync_helper")
    assert sync_before["parent_call_id"] == orchestrator_before["call_id"]


def test_sync_method_exception_pops_stack_and_fires_after_hook():
    """If a sync method raises: exception propagates, after_agent_call sees it, stack pops."""
    from unittest.mock import MagicMock

    from nooa.runtime.hooks import InstrumentationHooks, set_hooks

    mock_hooks = MagicMock(spec=InstrumentationHooks)
    mock_hooks.before_agent_call.return_value = {}

    class TestAgent(Agent, llm=_TEST_LLM):
        def boom(self) -> None:
            raise RuntimeError("nope")

        def follow_up(self) -> int:
            return 1

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()

        with pytest.raises(RuntimeError, match="nope"):
            agent.boom()

        boom_after = mock_hooks.after_agent_call.call_args.kwargs
        assert boom_after["method_name"] == "boom"
        assert isinstance(boom_after["exception"], RuntimeError)

        # Stack must have popped — follow_up has no traced ancestor.
        mock_hooks.reset_mock()
        mock_hooks.before_agent_call.return_value = {}
        assert agent.follow_up() == 1
        assert mock_hooks.before_agent_call.call_args.kwargs["parent_call_id"] is None
    finally:
        set_hooks(None)


def test_sync_method_calling_sync_method_chains_parent_call_id():
    """Sync `outer` calling sync `inner`: inner's parent_call_id == outer's call_id."""
    from unittest.mock import MagicMock

    from nooa.runtime.hooks import InstrumentationHooks, set_hooks

    hook_calls: list[dict] = []

    def before(**kwargs):
        hook_calls.append(kwargs)
        return {}

    mock_hooks = MagicMock(spec=InstrumentationHooks)
    mock_hooks.before_agent_call.side_effect = before

    class TestAgent(Agent, llm=_TEST_LLM):
        def outer(self) -> int:
            return self.inner() + 1

        def inner(self) -> int:
            return 1

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()
        assert agent.outer() == 2
    finally:
        set_hooks(None)

    outer_call = next(c for c in hook_calls if c["method_name"] == "outer")
    inner_call = next(c for c in hook_calls if c["method_name"] == "inner")
    assert inner_call["parent_call_id"] == outer_call["call_id"]


def test_agent_init_succeeds_with_sync_tracing_active():
    """Agent.__init__ calls sync helpers BEFORE self.runtime exists; must not crash."""
    from unittest.mock import MagicMock

    from nooa.runtime.hooks import InstrumentationHooks, set_hooks

    mock_hooks = MagicMock(spec=InstrumentationHooks)
    mock_hooks.before_agent_call.return_value = {}

    class TestAgent(Agent, llm=_TEST_LLM):
        def custom_helper(self, x: int) -> int:
            return x

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()  # Must not raise
        # Helpers like _resolve_llm fired during init have no runtime yet —
        # the fast-path skips hook firing for them.
        for call in mock_hooks.before_agent_call.call_args_list:
            assert call.kwargs.get("method_name") not in {
                "_resolve_llm",
                "_resolve_truncation",
                "_resolve_event_query",
                "_apply_context_dict",
                "_system_prompt",
            }
        # Sanity: post-init sync calls DO fire hooks
        mock_hooks.reset_mock()
        assert agent.custom_helper(7) == 7
        mock_hooks.before_agent_call.assert_called_once()
    finally:
        set_hooks(None)


# ============================================================================
# Generator methods (issue #38)
#
# Two distinct defects, fixed separately:
#
# 1. Generation is unreachable for generators. `has_ellipsis_body` is True for
#    `async def f(): yield 1; ...`, but `iscoroutinefunction` is False, so the
#    method never reaches `_should_generate` and comes out `_needs_generation
#    False`. The author writes a generation stub and silently gets a plain
#    generator. That combination is always a mistake -> hard TypeError.
#
# 2. Tracing spanned the wrong window. Generators were routed to the sync
#    wrapper, which pushed the call id around generator *creation* and popped
#    it before the body ran. Calls the body made at consumption time were
#    parented to whoever drained the generator. The generator wrappers below
#    push the call id around each *resumption* instead, so body calls parent to
#    the generator and consumer calls (made between yields) do not.
# ============================================================================


def _record_hooks():
    """Return (mock_hooks, hook_calls) recording every before_agent_call kwargs."""
    from unittest.mock import MagicMock

    from nooa.runtime.hooks import InstrumentationHooks

    hook_calls: list[dict] = []

    def before(**kwargs):
        hook_calls.append(kwargs)
        return {}

    mock_hooks = MagicMock(spec=InstrumentationHooks)
    mock_hooks.before_agent_call.side_effect = before
    return mock_hooks, hook_calls


def _one(hook_calls: list[dict], method_name: str) -> dict:
    # Not a bare `next()`: a StopIteration escaping inside an `async def` test
    # surfaces as "coroutine raised StopIteration" and hides the real problem.
    matches = [c for c in hook_calls if c["method_name"] == method_name]
    assert matches, f"no before_agent_call recorded for {method_name!r}"
    return matches[0]


def _all(hook_calls: list[dict], method_name: str) -> list[dict]:
    return [c for c in hook_calls if c["method_name"] == method_name]


# --- Defect 1: ellipsis body on a generator ---------------------------------


def test_async_generator_with_ellipsis_body_raises_at_class_creation():
    """`async def` with both yield and an ellipsis body can never be generated."""
    with pytest.raises(TypeError, match="cannot be LLM-generated"):

        class BadAgent(Agent, llm=_TEST_LLM):
            async def review(self, items: list[str]):
                yield items
                ...


def test_sync_generator_with_ellipsis_body_raises_at_class_creation():
    """`def` with both yield and an ellipsis body can never be generated."""
    with pytest.raises(TypeError, match="cannot be LLM-generated"):

        class BadAgent(Agent, llm=_TEST_LLM):
            def review(self, items: list[str]):
                yield items
                ...


def test_generator_ellipsis_error_names_the_method():
    """The TypeError must name the offending method so authors can find it."""
    with pytest.raises(TypeError, match=r"BadAgent\.review"):

        class BadAgent(Agent, llm=_TEST_LLM):
            async def review(self, items: list[str]):
                yield items
                ...


def test_generator_without_ellipsis_is_accepted():
    """A generator with a real body is fine — only the ellipsis combination is rejected."""

    class GoodAgent(Agent, llm=_TEST_LLM):
        async def review(self, items: list[str]):
            for item in items:
                yield item

    assert hasattr(GoodAgent.review, "_agent_decorator")
    assert GoodAgent.review._needs_generation is False


@pytest.mark.asyncio
async def test_generator_allows_ordinary_ellipsis_expressions():
    """Annotations and indexing use Ellipsis as data, not as generation syntax."""

    class EllipsisIndex:
        def __getitem__(self, key):
            assert key is ...
            return 7

    class GoodAgent(Agent, llm=_TEST_LLM):
        async def values(self):
            def nested_placeholder(): ...

            selected = EllipsisIndex()[...]
            annotated: tuple[int, ...] = (selected,)
            assert nested_placeholder() is None
            yield annotated[0]

    assert [item async for item in GoodAgent().values()] == [7]


def test_yield_ellipsis_generator_gets_single_result_error():
    """``yield ...`` is diagnosed as an attempted generated stream."""
    with pytest.raises(TypeError, match="produce one final result"):

        class BadAgent(Agent, llm=_TEST_LLM):
            async def review(self):
                yield ...


def test_nested_yield_ellipsis_generator_gets_single_result_error():
    """An ellipsis yielded through another yield is still a generation marker."""
    with pytest.raises(TypeError, match="produce one final result"):

        class BadAgent(Agent, llm=_TEST_LLM):
            async def review(self):
                yield (yield ...)


def test_nested_yield_from_ellipsis_generator_gets_single_result_error():
    """Yield-from traversal must find an ellipsis inside its yielded expression."""
    with pytest.raises(TypeError, match="produce one final result"):

        class BadAgent(Agent, llm=_TEST_LLM):
            def review(self):
                yield from (yield ...)


def test_strategy_generated_generator_explains_single_result_contract():
    """@strategy must not misdiagnose an async generator as non-async."""
    with pytest.raises(TypeError, match="must produce one final result"):

        class BadAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy())
            async def review(self):
                yield ...


def test_strategy_deterministic_generator_says_to_remove_decorator():
    """A deterministic generator needs no generation strategy."""
    with pytest.raises(TypeError, match=r"Remove @strategy"):

        class BadAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy())
            async def review(self):
                yield "implemented"


# --- Defect 2: span scoping across yields -----------------------------------


@pytest.mark.asyncio
async def test_async_generator_body_calls_parent_to_the_generator():
    """Issue #38: calls made *inside* an async generator body must parent to it."""
    from nooa.runtime.hooks import set_hooks

    mock_hooks, hook_calls = _record_hooks()

    class TestAgent(Agent, llm=_TEST_LLM):
        async def produce(self, n: int):
            for i in range(n):
                await self.in_body(i)
                yield i

        async def in_body(self, i: int) -> int:
            return i

        async def between_yields(self, v: int) -> int:
            return v

        async def drain(self) -> list[int]:
            out = []
            async for v in self.produce(2):
                await self.between_yields(v)
                out.append(v)
            return out

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()
        assert await agent.drain() == [0, 1]
    finally:
        set_hooks(None)

    drain = _one(hook_calls, "drain")
    produce = _one(hook_calls, "produce")

    assert produce["parent_call_id"] == drain["call_id"]

    # The bug: these were parented to `drain`, which never issued them.
    in_body = _all(hook_calls, "in_body")
    assert len(in_body) == 2
    assert all(c["parent_call_id"] == produce["call_id"] for c in in_body)

    # The mirror-image error a naive fix introduces: work the *consumer* does
    # between yields must NOT be attributed to the generator.
    between = _all(hook_calls, "between_yields")
    assert len(between) == 2
    assert all(c["parent_call_id"] == drain["call_id"] for c in between)


@pytest.mark.asyncio
async def test_sync_generator_body_calls_parent_to_the_generator():
    """Same guarantee for `def` with yield."""
    from nooa.runtime.hooks import set_hooks

    mock_hooks, hook_calls = _record_hooks()

    class TestAgent(Agent, llm=_TEST_LLM):
        def produce(self, n: int):
            for i in range(n):
                self.in_body(i)
                yield i

        def in_body(self, i: int) -> int:
            return i

        def between_yields(self, v: int) -> int:
            return v

        def drain(self) -> list[int]:
            out = []
            for v in self.produce(2):
                self.between_yields(v)
                out.append(v)
            return out

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()
        assert agent.drain() == [0, 1]
    finally:
        set_hooks(None)

    drain = _one(hook_calls, "drain")
    produce = _one(hook_calls, "produce")

    assert produce["parent_call_id"] == drain["call_id"]

    in_body = _all(hook_calls, "in_body")
    assert len(in_body) == 2
    assert all(c["parent_call_id"] == produce["call_id"] for c in in_body)

    between = _all(hook_calls, "between_yields")
    assert len(between) == 2
    assert all(c["parent_call_id"] == drain["call_id"] for c in between)


@pytest.mark.asyncio
async def test_async_generator_keeps_its_calling_convention():
    """Wrapping must not turn an async generator into a coroutine."""
    import inspect as _inspect

    from nooa.runtime.hooks import set_hooks

    mock_hooks, _ = _record_hooks()

    class TestAgent(Agent, llm=_TEST_LLM):
        async def produce(self, n: int):
            for i in range(n):
                yield i

    assert _inspect.isasyncgenfunction(TestAgent.produce)

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()
        assert [v async for v in agent.produce(3)] == [0, 1, 2]
    finally:
        set_hooks(None)


def test_sync_generator_keeps_its_calling_convention():
    """Wrapping must not eagerly drain a sync generator."""
    import inspect as _inspect

    from nooa.runtime.hooks import set_hooks

    mock_hooks, _ = _record_hooks()

    class TestAgent(Agent, llm=_TEST_LLM):
        def produce(self, n: int):
            yield from range(n)

    assert _inspect.isgeneratorfunction(TestAgent.produce)

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()
        gen = agent.produce(3)
        assert _inspect.isgenerator(gen)
        assert list(gen) == [0, 1, 2]
    finally:
        set_hooks(None)


@pytest.mark.asyncio
async def test_async_generator_span_closes_at_exhaustion_not_creation():
    """after_agent_call must fire once, when the generator is exhausted."""
    from nooa.runtime.hooks import set_hooks

    mock_hooks, _ = _record_hooks()

    class TestAgent(Agent, llm=_TEST_LLM):
        async def produce(self, n: int):
            for i in range(n):
                yield i

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()
        gen = agent.produce(2)
        assert await gen.__anext__() == 0
        # Body has started but is not finished — the span must still be open.
        assert mock_hooks.after_agent_call.call_count == 0
        assert await gen.__anext__() == 1
        with pytest.raises(StopAsyncIteration):
            await gen.__anext__()
        assert mock_hooks.after_agent_call.call_count == 1
    finally:
        set_hooks(None)


@pytest.mark.asyncio
async def test_async_generator_span_closes_on_explicit_aclose():
    """Abandoning a generator part-way closes its span exactly once, on aclose().

    Closing an async generator is itself async, so the span closes at `aclose()`
    rather than at the `break` that abandoned it. In ordinary code that is not a
    leak: once the generator becomes unreachable, asyncio's async-generator
    finalizer calls `aclose()` for you within a couple of event-loop ticks (see
    test_async_generator_span_closes_when_unreachable). Only a live reference
    that is never closed keeps the span open — which leaks the generator itself,
    not just the span. Early close is not an error, so `exception` stays None.
    """
    from nooa.runtime.hooks import set_hooks

    mock_hooks, _ = _record_hooks()

    class TestAgent(Agent, llm=_TEST_LLM):
        async def produce(self, n: int):
            for i in range(n):
                yield i

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()
        gen = agent.produce(10)
        assert await gen.__anext__() == 0
        assert mock_hooks.after_agent_call.call_count == 0
        await gen.aclose()
        assert mock_hooks.after_agent_call.call_count == 1
        assert mock_hooks.after_agent_call.call_args.kwargs["exception"] is None
        # aclose() is idempotent — the span must not close twice.
        await gen.aclose()
        assert mock_hooks.after_agent_call.call_count == 1
    finally:
        set_hooks(None)


@pytest.mark.asyncio
async def test_async_generator_span_closes_when_break_uses_aclosing():
    """`aclosing()` + break is the deterministic way to close an abandoned generator."""
    import contextlib

    from nooa.runtime.hooks import set_hooks

    mock_hooks, _ = _record_hooks()

    class TestAgent(Agent, llm=_TEST_LLM):
        async def produce(self, n: int):
            for i in range(n):
                yield i

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()
        async with contextlib.aclosing(agent.produce(10)) as gen:
            async for _v in gen:
                # Still inside the generator's lifetime: pre-fix, the span had
                # already closed at creation and this assertion is what fails.
                assert mock_hooks.after_agent_call.call_count == 0
                break
        assert mock_hooks.after_agent_call.call_count == 1
    finally:
        set_hooks(None)


def test_sync_generator_span_closes_on_early_break():
    """Sync generators close deterministically at `break` — close() is synchronous."""
    from nooa.runtime.hooks import set_hooks

    mock_hooks, _ = _record_hooks()

    class TestAgent(Agent, llm=_TEST_LLM):
        def produce(self, n: int):
            yield from range(n)

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()
        gen = agent.produce(10)
        assert next(gen) == 0
        assert mock_hooks.after_agent_call.call_count == 0
        gen.close()
        assert mock_hooks.after_agent_call.call_count == 1
        assert mock_hooks.after_agent_call.call_args.kwargs["exception"] is None
    finally:
        set_hooks(None)


@pytest.mark.asyncio
async def test_async_generator_exception_reaches_after_hook():
    """A raising generator body must report the exception through after_agent_call."""
    from nooa.runtime.hooks import set_hooks

    mock_hooks, _ = _record_hooks()

    class TestAgent(Agent, llm=_TEST_LLM):
        async def produce(self):
            yield 0
            raise RuntimeError("nope")

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()
        with pytest.raises(RuntimeError, match="nope"):
            async for _v in agent.produce():
                pass
        after = mock_hooks.after_agent_call.call_args.kwargs
        assert isinstance(after["exception"], RuntimeError)
    finally:
        set_hooks(None)


@pytest.mark.asyncio
async def test_no_trace_generator_is_not_traced():
    """@no_trace must suppress the generator's span and keep parent linkage intact."""
    from nooa.runtime.hooks import set_hooks

    mock_hooks, hook_calls = _record_hooks()

    class TestAgent(Agent, llm=_TEST_LLM):
        @no_trace
        async def produce(self, n: int):
            for i in range(n):
                await self.in_body(i)
                yield i

        async def in_body(self, i: int) -> int:
            return i

        async def drain(self) -> list[int]:
            return [v async for v in self.produce(2)]

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()
        assert await agent.drain() == [0, 1]
    finally:
        set_hooks(None)

    assert _all(hook_calls, "produce") == []
    # Children skip the untraced generator and attach to the nearest traced ancestor.
    drain = _one(hook_calls, "drain")
    in_body = _all(hook_calls, "in_body")
    assert len(in_body) == 2
    assert all(c["parent_call_id"] == drain["call_id"] for c in in_body)


def test_generator_on_untraced_class_is_left_alone():
    """AgentMeta is generic; a class without _enable_tracing gets no wrapper at all."""
    from nooa.metaclass import AgentMeta

    class Plain(metaclass=AgentMeta):
        def chunks(self, xs):
            yield from xs

        async def achunks(self, xs):
            for x in xs:
                yield x

    assert not hasattr(Plain.chunks, "_agent_decorator")
    assert not hasattr(Plain.achunks, "_agent_decorator")


@pytest.mark.asyncio
async def test_async_generator_forwards_asend_values():
    """The wrapper must stay transparent to `asend()` — sent values reach the body."""
    from nooa.runtime.hooks import set_hooks

    mock_hooks, _ = _record_hooks()

    class TestAgent(Agent, llm=_TEST_LLM):
        async def echo(self):
            received = []
            while True:
                sent = yield received
                if sent is None:
                    break
                received.append(sent)

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()
        gen = agent.echo()
        assert await gen.asend(None) == []
        assert await gen.asend("a") == ["a"]
        assert await gen.asend("b") == ["a", "b"]
        await gen.aclose()
    finally:
        set_hooks(None)


@pytest.mark.asyncio
async def test_async_generator_forwards_athrow_to_the_body():
    """The wrapper must stay transparent to `athrow()` — the body can catch it."""
    from nooa.runtime.hooks import set_hooks

    mock_hooks, _ = _record_hooks()

    class TestAgent(Agent, llm=_TEST_LLM):
        async def resilient(self):
            while True:
                try:
                    yield "ok"
                except ValueError:
                    yield "caught"

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()
        gen = agent.resilient()
        assert await gen.__anext__() == "ok"
        assert await gen.athrow(ValueError("boom")) == "caught"
        await gen.aclose()
    finally:
        set_hooks(None)


def test_sync_generator_forwards_send_values():
    """Same transparency guarantee for sync `send()`."""
    from nooa.runtime.hooks import set_hooks

    mock_hooks, _ = _record_hooks()

    class TestAgent(Agent, llm=_TEST_LLM):
        def echo(self):
            received = []
            while True:
                sent = yield received
                if sent is None:
                    break
                received.append(sent)

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()
        gen = agent.echo()
        assert gen.send(None) == []
        assert gen.send("a") == ["a"]
        gen.close()
    finally:
        set_hooks(None)


def test_sync_generator_forwards_throw_to_the_body():
    """Same transparency guarantee for sync `throw()`."""
    from nooa.runtime.hooks import set_hooks

    mock_hooks, _ = _record_hooks()

    class TestAgent(Agent, llm=_TEST_LLM):
        def resilient(self):
            while True:
                try:
                    yield "ok"
                except ValueError:
                    yield "caught"

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()
        gen = agent.resilient()
        assert next(gen) == "ok"
        assert gen.throw(ValueError("boom")) == "caught"
        gen.close()
    finally:
        set_hooks(None)


@pytest.mark.asyncio
async def test_generator_body_finally_runs_before_span_closes():
    """The wrapped generator's own cleanup must complete before the span closes."""
    from nooa.runtime.hooks import set_hooks

    order: list[str] = []

    mock_hooks, _ = _record_hooks()
    mock_hooks.after_agent_call.side_effect = lambda *a, **k: order.append("span_closed")

    class TestAgent(Agent, llm=_TEST_LLM):
        async def produce(self):
            try:
                yield 0
                yield 1
            finally:
                order.append("body_cleanup")

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()
        gen = agent.produce()
        assert await gen.__anext__() == 0
        await gen.aclose()
    finally:
        set_hooks(None)

    assert order == ["body_cleanup", "span_closed"]


@pytest.mark.asyncio
async def test_async_generator_span_closes_when_unreachable():
    """An abandoned generator's span closes on its own once nothing references it.

    `break` does not close an async generator, but dropping the last reference
    hands it to asyncio's async-generator finalizer, which calls `aclose()` a
    couple of loop ticks later. Documents that abandoning a generator in
    ordinary code does not strand an open span.
    """
    import asyncio

    from nooa.runtime.hooks import set_hooks

    mock_hooks, _ = _record_hooks()

    class TestAgent(Agent, llm=_TEST_LLM):
        async def produce(self, n: int):
            for i in range(n):
                yield i

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()

        async def consumer():
            # The generator is a temporary of this frame; when the frame exits
            # it becomes unreachable and the finalizer takes over.
            async for _v in agent.produce(1000):
                assert mock_hooks.after_agent_call.call_count == 0
                break

        await consumer()

        for _ in range(10):
            if mock_hooks.after_agent_call.call_count:
                break
            await asyncio.sleep(0)

        assert mock_hooks.after_agent_call.call_count == 1
        assert mock_hooks.after_agent_call.call_args.kwargs["exception"] is None
    finally:
        set_hooks(None)


# ============================================================================
# Regressions found by review of the generator wrappers.
#
# Every defect below was introduced by the generator wrappers themselves and
# would have shipped green: the original tests asserted exclusively through
# `mock_hooks` and never once through the agent-call event stream, which is
# where three of the four regressions showed up.
# ============================================================================


def _record_agent_call_events(agent) -> list[tuple]:
    """Capture (phase, method, success, exception_type) from the event stream."""
    seen: list[tuple] = []
    agent.event_manager.on("BeforeAgentCall", lambda e: seen.append(("before", e.method_name)))
    agent.event_manager.on(
        "AfterAgentCall",
        lambda e: seen.append(("after", e.method_name, e.success, e.exception_type)),
    )
    return seen


def test_sync_generator_return_value_survives_the_wrapper():
    """`return <value>` from a sync generator must reach `yield from` / StopIteration.value."""

    class TestAgent(Agent, llm=_TEST_LLM):
        def produce(self):
            yield 1
            yield 2
            return "DONE"

        def consume(self):
            value = yield from self.produce()
            return value

    gen = TestAgent().consume()
    items = []
    try:
        while True:
            items.append(next(gen))
    except StopIteration as stop:
        assert items == [1, 2]
        assert stop.value == "DONE"


def test_generator_span_closes_even_when_body_cleanup_raises():
    """A close() that raises must not strand the span — the after hook still fires."""
    from nooa.runtime.hooks import set_hooks

    mock_hooks, _ = _record_hooks()

    class TestAgent(Agent, llm=_TEST_LLM):
        def produce(self):
            try:
                yield 1
            finally:
                raise ValueError("cleanup failed")

    try:
        set_hooks(mock_hooks)
        gen = TestAgent().produce()
        assert next(gen) == 1
        assert mock_hooks.after_agent_call.call_count == 0
        with pytest.raises(ValueError, match="cleanup failed"):
            gen.close()
        assert mock_hooks.after_agent_call.call_count == 1
    finally:
        set_hooks(None)


@pytest.mark.asyncio
async def test_async_generator_span_closes_even_when_body_cleanup_raises():
    """Async counterpart: a failing aclose() must not leak the span."""
    from nooa.runtime.hooks import set_hooks

    mock_hooks, _ = _record_hooks()

    class TestAgent(Agent, llm=_TEST_LLM):
        async def produce(self):
            try:
                yield 1
            finally:
                raise ValueError("cleanup failed")

    try:
        set_hooks(mock_hooks)
        gen = TestAgent().produce()
        assert await gen.__anext__() == 1
        assert mock_hooks.after_agent_call.call_count == 0
        with pytest.raises(ValueError, match="cleanup failed"):
            await gen.aclose()
        assert mock_hooks.after_agent_call.call_count == 1
    finally:
        set_hooks(None)


def test_generator_signature_error_emits_no_unpaired_event():
    """A bad call signature must not leave a BeforeAgentCall without its AfterAgentCall."""

    class TestAgent(Agent, llm=_TEST_LLM):
        def produce(self, n):
            yield n

    agent = TestAgent()
    events = _record_agent_call_events(agent)

    # Deliberately timing-agnostic: the wrapper currently defers argument
    # binding to the first resumption, but the contract under test is only that
    # a failed call leaves no unpaired BeforeAgentCall behind.
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        list(agent.produce())

    assert [e for e in events if e[1] == "produce"] == []


def test_generator_early_close_is_reported_as_success():
    """Abandoning a generator is ordinary control flow, not a failed agent call."""

    class TestAgent(Agent, llm=_TEST_LLM):
        def produce(self, n):
            yield from range(n)

    agent = TestAgent()
    events = _record_agent_call_events(agent)

    gen = agent.produce(10)
    assert next(gen) == 0
    gen.close()

    assert ("before", "produce") in events
    assert ("after", "produce", True, None) in events


def test_generator_exhaustion_and_failure_report_distinct_event_outcomes():
    """Exhaustion is a success; a raising body is a failure. Both reach the event stream."""

    class TestAgent(Agent, llm=_TEST_LLM):
        def good(self, n):
            yield from range(n)

        def bad(self):
            yield 0
            raise RuntimeError("nope")

    agent = TestAgent()
    events = _record_agent_call_events(agent)

    assert list(agent.good(2)) == [0, 1]
    with pytest.raises(RuntimeError, match="nope"):
        list(agent.bad())

    assert ("after", "good", True, None) in events
    assert ("after", "bad", False, "RuntimeError") in events


@pytest.mark.asyncio
async def test_cancelled_generator_is_reported_as_failure():
    """Cancellation is a real failure — the method did not complete."""
    import asyncio

    class TestAgent(Agent, llm=_TEST_LLM):
        async def slow(self):
            yield 0
            await asyncio.sleep(10)
            yield 1

    agent = TestAgent()
    events = _record_agent_call_events(agent)

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.05):
            async for _v in agent.slow():
                pass

    after = [e for e in events if e[0] == "after" and e[1] == "slow"]
    assert after and after[0][2] is False
    assert after[0][3] == "CancelledError"


# --- guard holes: a generator hidden behind a decorator or descriptor -------


def _identity_decorator(fn):
    import functools

    @functools.wraps(fn)
    def inner(*args, **kwargs):
        return fn(*args, **kwargs)

    return inner


def test_contextmanager_methods_keep_their_calling_convention():
    """Dispatch must read the attribute's own shape, not the shape it wraps.

    `@contextmanager` wraps a generator function but returns a context manager.
    Dispatching on the unwrapped shape installs the generator wrapper, and
    `with agent.session()` then fails with "'generator' object does not support
    the context manager protocol".
    """
    import contextlib

    class TestAgent(Agent, llm=_TEST_LLM):
        @contextlib.contextmanager
        def session(self):
            yield "RESOURCE"

    with TestAgent().session() as resource:
        assert resource == "RESOURCE"


@pytest.mark.asyncio
async def test_asynccontextmanager_methods_keep_their_calling_convention():
    """Async counterpart — `@asynccontextmanager` wraps an async generator function."""
    import contextlib

    class TestAgent(Agent, llm=_TEST_LLM):
        @contextlib.asynccontextmanager
        async def session(self):
            yield "ARESOURCE"

    async with TestAgent().session() as resource:
        assert resource == "ARESOURCE"


def test_decorator_that_consumes_a_generator_returns_its_value():
    """A decorator draining a generator must not be re-dispatched as a generator."""
    import functools

    def collect(fn):
        @functools.wraps(fn)
        def inner(*args, **kwargs):
            return list(fn(*args, **kwargs))

        return inner

    class TestAgent(Agent, llm=_TEST_LLM):
        @collect
        def produce(self, n: int):
            yield from range(n)

    assert TestAgent().produce(2) == [0, 1]


def test_decorated_generator_with_ellipsis_body_is_rejected():
    """A decorator must not smuggle an ellipsis generator past the guard."""
    with pytest.raises(TypeError, match="cannot be LLM-generated"):

        class BadAgent(Agent, llm=_TEST_LLM):
            @_identity_decorator
            def produce(self, xs):
                yield xs
                ...


def test_staticmethod_generator_with_ellipsis_body_is_rejected():
    """staticmethod hides the function from `isfunction`; the guard must still fire."""
    with pytest.raises(TypeError, match="cannot be LLM-generated"):

        class BadAgent(Agent, llm=_TEST_LLM):
            @staticmethod
            def produce(xs):
                yield xs
                ...


def test_classmethod_generator_with_ellipsis_body_is_rejected():
    """Same for classmethod."""
    with pytest.raises(TypeError, match="cannot be LLM-generated"):

        class BadAgent(Agent, llm=_TEST_LLM):
            @classmethod
            def produce(cls, xs):
                yield xs
                ...


def test_sync_dunder_generator_with_ellipsis_body_is_rejected():
    """The ellipsis guard runs before the dunder skip, so `__iter__` cannot slip past."""
    with pytest.raises(TypeError, match="cannot be LLM-generated"):

        class BadAgent(Agent, llm=_TEST_LLM):
            def __iter__(self):
                yield 1
                ...


def test_ellipsis_generator_rejected_even_when_tracing_is_disabled():
    """The contradiction is about generation, not tracing — it must not depend on _enable_tracing."""
    from nooa.metaclass import AgentMeta

    with pytest.raises(TypeError, match="cannot be LLM-generated"):

        class Plain(metaclass=AgentMeta):
            def produce(self, xs):
                yield xs
                ...


def test_generator_ellipsis_error_uses_correct_article():
    """'is an generator' is a user-visible typo; both shapes must read correctly."""
    with pytest.raises(TypeError, match=r"is a generator \(def with yield\)"):

        class SyncBad(Agent, llm=_TEST_LLM):
            def produce(self, xs):
                yield xs
                ...

    with pytest.raises(TypeError, match=r"is an async generator \(async def with yield\)"):

        class AsyncBad(Agent, llm=_TEST_LLM):
            async def produce(self, xs):
                yield xs
                ...


# --- linkage properties the fix depends on ---------------------------------


@pytest.mark.asyncio
async def test_nested_generators_link_two_levels_deep():
    """outer(gen) -> inner(gen) -> leaf must chain, not collapse onto the drainer."""
    from nooa.runtime.hooks import set_hooks

    mock_hooks, hook_calls = _record_hooks()

    class TestAgent(Agent, llm=_TEST_LLM):
        async def inner(self, n: int):
            for i in range(n):
                await self.leaf(i)
                yield i

        async def outer(self, n: int):
            async for v in self.inner(n):
                yield v

        async def leaf(self, i: int) -> int:
            return i

        async def drain(self) -> list[int]:
            return [v async for v in self.outer(2)]

    try:
        set_hooks(mock_hooks)
        assert await TestAgent().drain() == [0, 1]
    finally:
        set_hooks(None)

    drain = _one(hook_calls, "drain")
    outer = _one(hook_calls, "outer")
    inner = _one(hook_calls, "inner")
    assert outer["parent_call_id"] == drain["call_id"]
    assert inner["parent_call_id"] == outer["call_id"]
    leaf = _all(hook_calls, "leaf")
    assert len(leaf) == 2
    assert all(c["parent_call_id"] == inner["call_id"] for c in leaf)


@pytest.mark.asyncio
async def test_concurrent_generators_do_not_share_a_call_stack():
    """The call id stack is a ContextVar; gathered drains must stay isolated."""
    import asyncio

    from nooa.runtime.hooks import set_hooks

    mock_hooks, hook_calls = _record_hooks()

    class TestAgent(Agent, llm=_TEST_LLM):
        async def produce(self, tag: str):
            for i in range(3):
                await self.in_body(f"{tag}{i}")
                yield i
                await asyncio.sleep(0)

        async def in_body(self, tag: str) -> str:
            return tag

        async def drain(self, tag: str) -> list[int]:
            return [v async for v in self.produce(tag)]

    try:
        set_hooks(mock_hooks)
        agent = TestAgent()
        results = await asyncio.gather(agent.drain("a"), agent.drain("b"))
        assert results == [[0, 1, 2], [0, 1, 2]]
    finally:
        set_hooks(None)

    produces = _all(hook_calls, "produce")
    assert len(produces) == 2

    # Set equality would pass under ANY scrambling of body calls across the two
    # generators. Check each body call against the generator that made it, via
    # the tag threaded through the call args.
    id_by_tag = {c["args"][0]: c["call_id"] for c in produces}
    assert set(id_by_tag) == {"a", "b"}

    body = _all(hook_calls, "in_body")
    assert len(body) == 6
    for call in body:
        assert call["parent_call_id"] == id_by_tag[call["args"][0][0]]
