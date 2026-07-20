# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for hook-based instrumentation."""

import pytest

from nooa.runtime.hooks import (
    InstrumentationHooks,
    call_after_hook,
    call_before_hook,
    get_hooks,
    set_hooks,
)


class TestHooksAPI:
    """Test set_hooks and get_hooks functions."""

    def test_get_hooks_returns_none_by_default(self):
        """get_hooks returns None when no hooks are installed."""
        # Reset hooks first
        set_hooks(None)
        assert get_hooks() is None

    def test_set_and_get_hooks(self):
        """set_hooks installs hooks that get_hooks retrieves."""

        class TestHooks:
            def before_agent_call(self, agent, method_name, args, kwargs, call_id, parent_call_id):
                pass

            def after_agent_call(self, agent, method_name, result, exception, context):
                pass

            def before_generation(
                self, agent, method_name, strategy, generation_id, parent_generation_id
            ):
                pass

            def after_generation(
                self, agent, method_name, result, exception, context, generation_id
            ):
                pass

            def before_code_execution(self, agent, code, execution_id):
                pass

            def after_code_execution(self, agent, code, result, exception, context, execution_id):
                pass

        hooks = TestHooks()
        set_hooks(hooks)

        try:
            assert get_hooks() is hooks
        finally:
            set_hooks(None)

    def test_set_hooks_none_removes_hooks(self):
        """set_hooks(None) removes installed hooks."""

        class TestHooks:
            def before_agent_call(self, agent, method_name, args, kwargs, call_id, parent_call_id):
                pass

            def after_agent_call(self, agent, method_name, result, exception, context):
                pass

            def before_generation(
                self, agent, method_name, strategy, generation_id, parent_generation_id
            ):
                pass

            def after_generation(
                self, agent, method_name, result, exception, context, generation_id
            ):
                pass

            def before_code_execution(self, agent, code, execution_id):
                pass

            def after_code_execution(self, agent, code, result, exception, context, execution_id):
                pass

        hooks = TestHooks()
        set_hooks(hooks)
        assert get_hooks() is hooks

        set_hooks(None)
        assert get_hooks() is None


class TestInstrumentationHooksProtocol:
    """Test the InstrumentationHooks protocol."""

    def test_protocol_is_runtime_checkable(self):
        """InstrumentationHooks is a runtime-checkable Protocol."""

        class FullHooks:
            def before_agent_call(self, agent, method_name, args, kwargs, call_id, parent_call_id):
                return {}

            def after_agent_call(self, agent, method_name, result, exception, context):
                pass

            def before_generation(
                self, agent, method_name, strategy, generation_id, parent_generation_id
            ):
                return {}

            def after_generation(
                self, agent, method_name, result, exception, context, generation_id
            ):
                pass

            def before_code_execution(self, agent, code, execution_id, generation_id=None):
                return {}

            def after_code_execution(self, agent, code, result, exception, context, execution_id):
                pass

            def before_method_invocation(self, agent, method_name, args, kwargs, invocation_id):
                return {}

            def after_method_invocation(
                self, agent, method_name, result, exception, context, invocation_id
            ):
                pass

            def before_tool_execution(
                self, agent, tool_name, arguments, execution_id, generation_id=None
            ):
                return {}

            def after_tool_execution(
                self, agent, tool_name, arguments, result, exception, context, execution_id
            ):
                pass

            def on_messages_built(self, agent, method_name, messages, generation_id, **kwargs):
                pass

        hooks = FullHooks()
        assert isinstance(hooks, InstrumentationHooks)

    def test_partial_implementation_not_protocol_instance(self):
        """Class missing methods is not an InstrumentationHooks instance."""

        class PartialHooks:
            def before_agent_call(self, agent, method_name, args, kwargs, call_id, parent_call_id):
                pass

            # Missing other methods

        hooks = PartialHooks()
        assert not isinstance(hooks, InstrumentationHooks)


class MockHooks:
    """Mock hooks implementation for testing hook call sites."""

    def __init__(self):
        self.calls = []

    def before_agent_call(self, agent, method_name, args, kwargs, call_id, parent_call_id):
        self.calls.append(("before_agent_call", method_name, args, kwargs, call_id, parent_call_id))
        return {"call_id": call_id}

    def after_agent_call(self, agent, method_name, result, exception, context):
        self.calls.append(("after_agent_call", method_name, result, exception, context))

    def before_generation(
        self, agent, method_name, strategy, generation_id, parent_generation_id, **kwargs
    ):
        self.calls.append(
            ("before_generation", method_name, strategy, generation_id, parent_generation_id)
        )
        return {"generation_id": generation_id}

    def after_generation(self, agent, method_name, result, exception, context, generation_id):
        self.calls.append(
            ("after_generation", method_name, result, exception, context, generation_id)
        )

    def before_code_execution(self, agent, code, execution_id):
        self.calls.append(("before_code_execution", code[:50], execution_id))
        return {"execution_id": execution_id}

    def after_code_execution(self, agent, code, result, exception, context, execution_id):
        self.calls.append(
            ("after_code_execution", code[:50], result, exception, context, execution_id)
        )

    def on_messages_built(self, agent, method_name, messages, generation_id, **kwargs):
        self.calls.append(("on_messages_built", method_name, len(messages), generation_id))


class FailingHooks:
    """Hooks implementation that always raises exceptions."""

    def before_agent_call(self, agent, method_name, args, kwargs, call_id, parent_call_id):
        raise RuntimeError("Hook failure: before_agent_call")

    def after_agent_call(self, agent, method_name, result, exception, context):
        raise RuntimeError("Hook failure: after_agent_call")

    def before_generation(
        self, agent, method_name, strategy, generation_id, parent_generation_id, **kwargs
    ):
        raise RuntimeError("Hook failure: before_generation")

    def after_generation(self, agent, method_name, result, exception, context, generation_id):
        raise RuntimeError("Hook failure: after_generation")

    def before_code_execution(self, agent, code, execution_id):
        raise RuntimeError("Hook failure: before_code_execution")

    def after_code_execution(self, agent, code, result, exception, context, execution_id):
        raise RuntimeError("Hook failure: after_code_execution")


class TestTraceableSkipsHooks:
    """Test that non-traceable strategies skip generation hooks in actor."""

    @pytest.fixture(autouse=True)
    def _cleanup_hooks(self):
        """Reset hooks after each test."""
        yield
        set_hooks(None)

    @pytest.mark.asyncio
    async def test_execute_nested_skips_hooks_for_non_traceable_strategy(self):
        """execute_nested must NOT call generation hooks when strategy.traceable is False."""
        from unittest.mock import Mock

        from nooa.runtime.actor import ActorRuntime
        from nooa.strategies.current_call import CurrentCall
        from nooa.strategies.template import TemplateStrategy
        from tests.strategies.test_template_strategy import MockRuntime

        mock_hooks = MockHooks()
        set_hooks(mock_hooks)

        # Create a real ActorRuntime with a mock agent
        agent = Mock()
        actor = ActorRuntime(agent)

        # Patch expand_variables on the actor so TemplateStrategy can render
        mock_runtime = MockRuntime(agent=agent)
        actor.expand_variables = mock_runtime.expand_variables

        strategy = TemplateStrategy()
        call = CurrentCall(
            id="test",
            method_name="test",
            decorator="plan",
            docstring="Hello {name}",
            kwargs={"name": "world"},
        )

        # Call the real execute_nested — this exercises the actual guard logic
        result = await actor.execute_nested(strategy, call)
        assert result == "Hello world"

        # Verify: NO generation hooks were called
        generation_calls = [c for c in mock_hooks.calls if "generation" in c[0]]
        assert generation_calls == [], f"Expected no generation hook calls, got: {generation_calls}"

    @pytest.mark.asyncio
    async def test_execute_nested_calls_hooks_for_traceable_strategy(self):
        """Control test: traceable=True strategy DOES call generation hooks via execute_nested."""
        from unittest.mock import Mock

        from nooa.runtime.actor import ActorRuntime
        from nooa.strategies.base import GenerationStrategy
        from nooa.strategies.current_call import CurrentCall

        class TraceableStrategy(GenerationStrategy):
            async def execute(self, runtime, call):
                return "result"

        mock_hooks = MockHooks()
        set_hooks(mock_hooks)

        agent = Mock()
        actor = ActorRuntime(agent)

        strategy = TraceableStrategy()
        call = CurrentCall(
            id="test",
            method_name="test",
            decorator="plan",
            docstring="test",
            kwargs={},
        )

        result = await actor.execute_nested(strategy, call)
        assert result == "result"

        generation_calls = [c for c in mock_hooks.calls if "generation" in c[0]]
        assert len(generation_calls) == 2
        assert generation_calls[0][0] == "before_generation"
        assert generation_calls[1][0] == "after_generation"


class TestHooksContextPassthrough:
    """Test that context is correctly passed from before to after hooks."""

    def test_context_passed_to_after_agent_call(self):
        """Context from before_agent_call is passed to after_agent_call."""
        mock = MockHooks()
        set_hooks(mock)

        try:
            # Simulate what stub.py does
            hooks = get_hooks()
            call_id = "test-call-123"

            context = hooks.before_agent_call(
                agent=None,
                method_name="test_method",
                args=(1, 2),
                kwargs={"key": "value"},
                call_id=call_id,
                parent_call_id=None,
            )

            # Context should contain the call_id
            assert context == {"call_id": call_id}

            hooks.after_agent_call(
                agent=None,
                method_name="test_method",
                result="success",
                exception=None,
                context=context,
            )

            # Verify the context was passed through
            after_call = [c for c in mock.calls if c[0] == "after_agent_call"][0]
            assert after_call[4] == {"call_id": call_id}  # context parameter

        finally:
            set_hooks(None)

    def test_context_passed_to_after_generation(self):
        """Context from before_generation is passed to after_generation."""
        mock = MockHooks()
        set_hooks(mock)

        try:
            hooks = get_hooks()
            generation_id = "gen-456"

            context = hooks.before_generation(
                agent=None,
                method_name="test_method",
                strategy="PurePythonStrategy",
                generation_id=generation_id,
                parent_generation_id=None,
            )

            assert context == {"generation_id": generation_id}

            hooks.after_generation(
                agent=None,
                method_name="test_method",
                result="generated",
                exception=None,
                context=context,
                generation_id=generation_id,
            )

            after_call = [c for c in mock.calls if c[0] == "after_generation"][0]
            assert after_call[4] == {"generation_id": generation_id}

        finally:
            set_hooks(None)

    def test_context_passed_to_after_code_execution(self):
        """Context from before_code_execution is passed to after_code_execution."""
        mock = MockHooks()
        set_hooks(mock)

        try:
            hooks = get_hooks()
            execution_id = "exec-789"
            code = "x = 1 + 1"

            context = hooks.before_code_execution(
                agent=None,
                code=code,
                execution_id=execution_id,
            )

            assert context == {"execution_id": execution_id}

            hooks.after_code_execution(
                agent=None,
                code=code,
                result=2,
                exception=None,
                context=context,
                execution_id=execution_id,
            )

            after_call = [c for c in mock.calls if c[0] == "after_code_execution"][0]
            assert after_call[4] == {"execution_id": execution_id}

        finally:
            set_hooks(None)


class TestHookDispatchMetrics:
    """Hook dispatch records tracing overhead into harness metrics."""

    @pytest.fixture(autouse=True)
    def _cleanup_hooks(self):
        yield
        set_hooks(None)

    def test_before_and_after_hooks_record_tracing_overhead(self):
        from nooa.runtime.harness_metrics import harness_metrics_session

        mock = MockHooks()
        set_hooks(mock)

        with harness_metrics_session() as hm:
            context = call_before_hook(
                "before_generation",
                agent=None,
                method_name="method",
                strategy="strategy",
                generation_id="gen-1",
                parent_generation_id=None,
            )
            call_after_hook(
                "after_generation",
                context,
                agent=None,
                method_name="method",
                result="ok",
                exception=None,
                generation_id="gen-1",
            )

            assert hm.time_tracing_overhead.count == 2
            assert hm.time_tracing_overhead.total_s > 0
            attrs = hm.to_span_attributes()
            assert attrs["harness.time.tracing_overhead.count"] == 2

    def test_failing_hook_still_records_tracing_overhead(self):
        from nooa.runtime.harness_metrics import harness_metrics_session

        set_hooks(FailingHooks())

        with harness_metrics_session() as hm:
            assert (
                call_before_hook(
                    "before_generation",
                    agent=None,
                    method_name="method",
                    strategy="strategy",
                    generation_id="gen-1",
                    parent_generation_id=None,
                )
                is None
            )

            assert hm.time_tracing_overhead.count == 1
            assert hm.time_tracing_overhead.total_s > 0
