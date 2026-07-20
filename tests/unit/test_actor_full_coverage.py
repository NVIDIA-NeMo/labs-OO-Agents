# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests targeting uncovered lines in nooa/runtime/actor.py.

Each test class has a docstring explaining the exact line(s) it covers.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.context_blocks.models import ContextWindowStats
from nooa.context_blocks.renderer import RenderResult
from nooa.strategies.codeact import CodeActStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

# Minimal stats object for mocking render_context return values
_EMPTY_STATS = ContextWindowStats(
    context_blocks_count=0,
    events_count=0,
    context_blocks_chars=0,
    events_chars=0,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEST_LLM = FakeLLMClient()


def _resp(content: str, tool_calls: list | None = None) -> LLMResponse:
    """Create a test LLM response."""
    finish_reason = "tool_calls" if tool_calls else "stop"
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        assistant_message={"role": "assistant", "content": content},
    )


def _return_result(result: Any = None, call_id: str = "call_return") -> ToolCall:
    return ToolCall(
        id=call_id,
        name="return_result",
        arguments=json.dumps({"result": result}),
    )


def _tool_call(code: str, call_id: str = "call_1") -> ToolCall:
    return ToolCall(
        id=call_id,
        name="execute_python",
        arguments=json.dumps({"code": code}),
    )


# ===========================================================================
# Line 333: current_call property
# ===========================================================================


class TestCurrentCallProperty:
    """Line 333: runtime.current_call returns _current_call_var.get().

    Verify the property is accessible and returns None outside a generation
    session, then returns a real CurrentCall during generation.
    """

    def test_current_call_returns_none_outside_generation(self):
        """current_call returns None when no generation is running."""

        class _Agent(Agent, llm=_TEST_LLM):
            async def task(self) -> str:
                """Do something."""
                ...

        agent = _Agent()
        assert agent.runtime.current_call is None

    @pytest.mark.asyncio
    async def test_current_call_returns_call_during_generation(self):
        """current_call returns the CurrentCall object during generation."""

        class _Agent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def task(self) -> int:
                """Do it."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                # Capture current_call onto the agent instance
                _resp(
                    "",
                    tool_calls=[
                        _tool_call("self._captured_call = self.runtime.current_call"),
                    ],
                ),
                _resp("", tool_calls=[_return_result(result=42)]),
            ]
        )
        agent = _Agent(llm=fake_llm)
        result = await agent.task()
        assert result == 42
        # The current_call was captured during execution
        assert hasattr(agent, "_captured_call")
        assert agent._captured_call is not None
        assert agent._captured_call.method_name == "task"


# ===========================================================================
# Line 456: non-string, non-BaseModel LLM response content
# ===========================================================================


class TestNonStringLLMContent:
    """Line 454-456: elif not isinstance(content, str) -> content = str(content).

    When the LLM response content is neither a BaseModel nor a string (e.g.,
    an integer or list), it should be converted to string.
    """

    @pytest.mark.asyncio
    async def test_non_string_content_converted(self):
        """LLM returning numeric content gets str()-ified."""

        class _Agent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def task(self) -> int:
                """Compute."""
                ...

        # Build a response where content is an integer (not str, not BaseModel)
        numeric_resp = LLMResponse(
            raw_response=None,
            content=12345,  # numeric, not str
            tool_calls=[_return_result(result=99)],
            finish_reason="tool_calls",
            assistant_message={"role": "assistant", "content": "12345"},
        )
        fake_llm = FakeLLMClient(scripted_responses=[numeric_resp])
        agent = _Agent(llm=fake_llm)
        result = await agent.task()
        assert result == 99


# ===========================================================================
# Line 851: no-timeout path in wrap_in_function mode
# ===========================================================================


class TestExecuteCodeNoTimeoutWrapped:
    """Line 851: result_value = await coro (no timeout).

    When execute_code is called with timeout=None and wrap_in_function=True,
    the async path should await the coroutine directly without asyncio.wait_for.
    """

    @pytest.mark.asyncio
    async def test_no_timeout_wrap_in_function(self):
        """Execute code with timeout=None and wrap_in_function=True."""

        class _Agent(Agent, llm=_TEST_LLM):
            async def task(self) -> str:
                """Work."""
                ...

        agent = _Agent()
        result = await agent.runtime.execute_code(
            "x = 1 + 1",
            wrap_in_function=True,
            timeout=None,
        )
        assert result.error is None


# ===========================================================================
# Lines 881-883: AttributeError on _generated_source for builtins (wrap mode)
# ===========================================================================


class TestGeneratedSourceAttributeErrorWrapped:
    """Lines 881-883: try/except (AttributeError, TypeError) when setting
    _generated_source on a builtin in the wrap_in_function code path.

    method_sources uses ast.walk (recursive) to find defs with self param,
    but func_defs only checks tree.body (top-level). A nested def with self
    param appears in method_sources but not in func_defs. If we inject a
    frozen callable of the same name via builtins, it stays in exec_globals
    and the _generated_source assignment fails.
    """

    @pytest.mark.asyncio
    async def test_frozen_callable_in_exec_globals_triggers_except(self):
        """A __slots__ callable injected as builtin triggers the except branch."""

        class FrozenCallable:
            """Callable that refuses attribute assignment."""

            __slots__ = ()

            def __call__(self, *a, **kw):
                return None

        class _Agent(Agent, llm=_TEST_LLM):
            async def task(self) -> str:
                """Work."""
                ...

        agent = _Agent()

        # Code that defines a method with self param NESTED inside another function.
        # ast.walk finds it -> method_sources["inner_method"] gets populated.
        # But tree.body only has outer_fn -> func_defs won't include inner_method.
        # The frozen callable injected via builtins stays in exec_globals.
        code = (
            "def outer_fn():\n"
            "    def inner_method(self):\n"
            "        return 42\n"
            "    return inner_method\n"
        )

        frozen = FrozenCallable()
        result = await agent.runtime.execute_code(
            code,
            builtins={"inner_method": frozen},
            wrap_in_function=True,
            timeout=None,
        )
        # The code executes fine; _generated_source assignment silently fails
        assert result.error is None


# ===========================================================================
# Lines 928-934: TimeoutError and no-timeout in non-wrap async path
# ===========================================================================


class TestExecuteCodeNonWrapAsyncPaths:
    """Lines 928-934: In the non-wrap_in_function path, when code has 'await',
    it is wrapped in __wrapper__. Test both the timeout and no-timeout branches.
    """

    @pytest.mark.asyncio
    async def test_no_timeout_async_non_wrap(self):
        """Line 934: await coro without timeout in non-wrap async path."""

        class _Agent(Agent, llm=_TEST_LLM):
            async def task(self) -> str:
                """Work."""
                ...

        agent = _Agent()
        # Code with await triggers the async wrapper path (not wrap_in_function)
        code = "result = await asyncio.sleep(0)"
        result = await agent.runtime.execute_code(
            code,
            wrap_in_function=False,
            timeout=None,
        )
        assert result.error is None

    @pytest.mark.asyncio
    async def test_timeout_error_async_non_wrap(self):
        """Lines 928-932: TimeoutError in non-wrap async path."""

        class _Agent(Agent, llm=_TEST_LLM):
            async def task(self) -> str:
                """Work."""
                ...

        agent = _Agent()
        # Code with await that will timeout
        code = "await asyncio.sleep(999)"
        result = await agent.runtime.execute_code(
            code,
            wrap_in_function=False,
            timeout=0.01,  # very short timeout
        )
        assert result.error is not None
        assert "timed out" in str(result.error)


# ===========================================================================
# Lines 949-951: _generated_source on builtins in non-wrap path
# ===========================================================================


class TestGeneratedSourceAttributeErrorNonWrap:
    """Lines 949-951: try/except (AttributeError, TypeError) when setting
    _generated_source in the non-wrap_in_function code path.

    Same pattern as 881-883 but for the code path after the else branch.
    Uses the same nested-def + frozen-callable technique.
    """

    @pytest.mark.asyncio
    async def test_frozen_callable_non_wrap_triggers_except(self):
        """A __slots__ callable injected as builtin triggers the except branch."""

        class FrozenCallable:
            __slots__ = ()

            def __call__(self, *a, **kw):
                return None

        class _Agent(Agent, llm=_TEST_LLM):
            async def task(self) -> str:
                """Work."""
                ...

        agent = _Agent()

        # Same technique: nested def with self param + frozen callable via builtins.
        code = (
            "def outer_fn():\n"
            "    def inner_method(self):\n"
            "        return 42\n"
            "    return inner_method\n"
        )

        frozen = FrozenCallable()
        result = await agent.runtime.execute_code(
            code,
            builtins={"inner_method": frozen},
            wrap_in_function=False,
            timeout=None,
        )
        assert result.error is None


# ===========================================================================
# Lines 1080, 1082, 1084, 1086: Strategy config extraction for tracing
# ===========================================================================


class TestExecuteNestedStrategyConfig:
    """Lines 1079-1086: Extract max_iterations, max_retries, max_reflections,
    has_prefill from strategy for tracing kwargs passed to before_generation hook.
    """

    @pytest.mark.asyncio
    async def test_strategy_config_extraction(self):
        """execute_nested extracts strategy attributes for hook kwargs."""
        from nooa.runtime.actor import (
            _current_call_var,
            _current_llm_var,
            _current_method_var,
            _push_generation_id,
        )
        from nooa.strategies.base import GenerationStrategy
        from nooa.strategies.current_call import CurrentCall

        class ConfigStrategy(GenerationStrategy):
            """A strategy with all config attributes for testing."""

            max_iterations = 10
            max_retries = 3
            max_reflections = 5
            prefill = "some prefill"

            async def execute(self, runtime, call):
                return "done"

        class _Agent(Agent, llm=_TEST_LLM):
            async def task(self) -> str:
                """Work."""
                ...

        agent = _Agent()
        runtime = agent.runtime

        strat = ConfigStrategy()
        call = CurrentCall(
            id="test_call",
            method_name="task",
            decorator="plan",
        )

        # Set context vars needed by execute_nested
        async def _method(): ...

        token_call = _current_call_var.set(call)
        token_method = _current_method_var.set(_method)
        token_llm = _current_llm_var.set(_TEST_LLM)
        _push_generation_id("parent-gen-id")

        try:
            with (
                patch("nooa.runtime.actor.call_before_hook") as mock_before,
                patch("nooa.runtime.actor.call_after_hook"),
            ):
                mock_before.return_value = MagicMock()
                result = await runtime.execute_nested(strat, call)

                assert result == "done"
                # Verify before_generation was called with strategy kwargs
                mock_before.assert_called_once()
                call_kwargs = mock_before.call_args
                # The kwargs should include our strategy config
                assert call_kwargs.kwargs.get("max_iterations") == 10
                assert call_kwargs.kwargs.get("max_retries") == 3
                assert call_kwargs.kwargs.get("max_reflections") == 5
                assert call_kwargs.kwargs.get("has_prefill") is True
        finally:
            _current_call_var.reset(token_call)
            _current_method_var.reset(token_method)
            _current_llm_var.reset(token_llm)


# ===========================================================================
# Lines 1110-1112: Exception capture in strategy execution for hooks
# ===========================================================================


class TestExecuteNestedStrategyException:
    """Lines 1110-1112: When strategy.execute() raises, exception_caught is set
    and re-raised. The after_generation hook sees the exception.
    """

    @pytest.mark.asyncio
    async def test_exception_propagated_and_captured(self):
        """Strategy exception is captured and re-raised."""
        from nooa.runtime.actor import (
            _current_call_var,
            _current_llm_var,
            _current_method_var,
            _push_generation_id,
        )
        from nooa.strategies.base import GenerationStrategy
        from nooa.strategies.current_call import CurrentCall

        class FailingStrategy(GenerationStrategy):
            async def execute(self, runtime, call):
                raise ValueError("strategy boom")

        class _Agent(Agent, llm=_TEST_LLM):
            async def task(self) -> str:
                """Work."""
                ...

        agent = _Agent()
        runtime = agent.runtime

        call = CurrentCall(
            id="test_call",
            method_name="task",
            decorator="plan",
        )

        async def _method(): ...

        token_call = _current_call_var.set(call)
        token_method = _current_method_var.set(_method)
        token_llm = _current_llm_var.set(_TEST_LLM)
        _push_generation_id("parent-gen-id")

        try:
            with (
                patch("nooa.runtime.actor.call_before_hook") as mock_before,
                patch("nooa.runtime.actor.call_after_hook") as mock_after,
            ):
                mock_before.return_value = MagicMock()

                with pytest.raises(ValueError, match="strategy boom"):
                    await runtime.execute_nested(FailingStrategy(), call)

                # after_generation hook should have been called with the exception
                mock_after.assert_called_once()
                after_kwargs = mock_after.call_args
                assert after_kwargs.kwargs.get("exception") is not None
                assert "strategy boom" in str(after_kwargs.kwargs["exception"])
        finally:
            _current_call_var.reset(token_call)
            _current_method_var.reset(token_method)
            _current_llm_var.reset(token_llm)


# ===========================================================================
# Line 1424: isinstance(attr, property) in list_methods — callable property
# ===========================================================================


class TestListMethodsCallablePropertyDescriptor:
    """Line 1424: isinstance(attr, property) check in list_methods.

    Standard property objects are NOT callable, so they get filtered by the
    callable() check on line 1417. This line catches exotic descriptors that
    are BOTH callable and isinstance(property). In practice, this means
    subclassing property to make it callable.
    """

    def test_callable_property_skipped(self):
        """A callable property-like descriptor is skipped by list_methods."""

        class CallableProperty(property):
            """A property subclass that is also callable (exotic)."""

            def __call__(self, *args, **kwargs):
                return "called"

        class _Agent(Agent, llm=_TEST_LLM):
            @CallableProperty
            def exotic_prop(self):
                return "value"

            async def real_method(self) -> str:
                """A real method."""
                ...

        agent = _Agent()
        methods = agent.runtime.list_methods()
        assert "exotic_prop" not in methods
        assert "real_method" in methods


# ===========================================================================
# Line 1447: inspect.signature failure
# ===========================================================================


class TestListMethodsSignatureFailure:
    """Line 1447: inspect.signature(attr) raises (ValueError, TypeError),
    falling back to f"{name}(...)".
    """

    def test_bad_signature_fallback(self):
        """Method with broken signature still appears in list_methods."""

        class _Agent(Agent, llm=_TEST_LLM):
            async def normal_method(self) -> str:
                """Normal method."""
                ...

        agent = _Agent()

        # Create a callable that raises ValueError on inspect.signature()
        class _NoSig:
            """Callable with no inspectable signature."""

            def __call__(self):
                pass

            # Override __signature__ to force ValueError
            @property
            def __signature__(self):
                raise ValueError("no signature")

        type(agent).bad_sig_method = _NoSig()

        try:
            methods_info = agent.runtime.list_methods()
            # The callable should be present with fallback signature
            assert "bad_sig_method" in methods_info
            assert methods_info["bad_sig_method"]["signature"] == "bad_sig_method(...)"
        finally:
            delattr(type(agent), "bad_sig_method")


# ===========================================================================
# Lines 2069-2077: DynamicContext eval failure
# ===========================================================================


class TestDynamicContextEvalFailure:
    """Lines 2069-2077: When a DynamicContext expression raises an exception,
    the error is returned inline as "ExceptionType: message".
    """

    @pytest.mark.asyncio
    async def test_bad_expression_returns_error_string(self):
        """DynamicContext with broken expression returns error inline."""

        class _Agent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def task(self) -> str:
                """Work."""
                ...

        agent = _Agent()
        runtime = agent.runtime

        # Use set_dynamic to add a DynamicContext with a broken expression
        agent.context.set_dynamic("bad_block", "1 / 0")

        # Now call _prepare_context which resolves DynamicContext blocks
        method = type(agent).task
        blocks = await runtime._prepare_context(method)

        # Find the block for "bad_block" and check it has the error
        bad_block_found = False
        for block in blocks:
            if hasattr(block, "content") and "ZeroDivisionError" in str(block.content):
                bad_block_found = True
                break
            if hasattr(block, "key") and block.key == "bad_block":
                assert "ZeroDivisionError" in str(block.content)
                bad_block_found = True
                break

        assert bad_block_found, (
            f"Expected ZeroDivisionError in blocks, got: {[str(b) for b in blocks]}"
        )


# ===========================================================================
# Line 2079: DynamicContext result is None
# ===========================================================================


class TestDynamicContextResultNone:
    """Line 2079: When a DynamicContext expression evaluates to None,
    return "None" string.
    """

    @pytest.mark.asyncio
    async def test_none_result_returns_none_string(self):
        """DynamicContext evaluating to None returns 'None' string."""

        class _Agent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def task(self) -> str:
                """Work."""
                ...

        agent = _Agent()
        runtime = agent.runtime

        agent.context.set_dynamic("none_block", "None")

        method = type(agent).task
        blocks = await runtime._prepare_context(method)

        # Find the none_block and check content is "None"
        found = False
        for block in blocks:
            if hasattr(block, "key") and block.key == "none_block":
                assert block.content == "None"
                found = True
                break

        assert found, (
            f"Expected none_block in blocks, got keys: {[getattr(b, 'key', '?') for b in blocks]}"
        )


# ===========================================================================
# Lines 2082-2084: DynamicContext result is non-string (truncating_pformat path)
# ===========================================================================


class TestDynamicContextNonStringResult:
    """Lines 2082-2084: When a DynamicContext expression returns a non-string
    value (e.g., a list or dict), it goes through truncating_pformat().
    """

    @pytest.mark.asyncio
    async def test_non_string_result_formatted(self):
        """DynamicContext returning a list goes through truncating_pformat."""

        class _Agent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def task(self) -> str:
                """Work."""
                ...

        agent = _Agent()
        agent.my_list = [1, 2, 3]
        runtime = agent.runtime

        agent.context.set_dynamic("list_block", "self.my_list")

        method = type(agent).task
        blocks = await runtime._prepare_context(method)

        # Find the list_block
        found = False
        for block in blocks:
            if hasattr(block, "key") and block.key == "list_block":
                # truncating_pformat should convert [1, 2, 3] to a string representation
                assert "1" in block.content
                assert "2" in block.content
                assert "3" in block.content
                found = True
                break

        assert found, (
            f"Expected list_block in blocks, got keys: {[getattr(b, 'key', '?') for b in blocks]}"
        )


# ===========================================================================
# Lines 2133-2136: ImportError and Exception for openinference tracing
# ===========================================================================


class TestBuildMessagesOpeninferenceErrors:
    """Lines 2133-2136: _build_messages catches ImportError and general Exception
    when trying to import openinference_instrumentation_nooa.
    """

    @pytest.mark.asyncio
    async def test_import_error_silenced(self):
        """ImportError from openinference import is silently caught."""
        from nooa.runtime.actor import _current_llm_var

        class _Agent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def task(self) -> str:
                """Work."""
                ...

        agent = _Agent()
        runtime = agent.runtime

        method = type(agent).task

        # Set the LLM context var (needed by _build_messages)
        token = _current_llm_var.set(_TEST_LLM)
        try:
            # Patch the import to raise ImportError
            with patch(
                "nooa.runtime.actor.render_context",
                return_value=RenderResult(output=[], stats=_EMPTY_STATS, messages=[]),
            ):
                with patch.dict(
                    "sys.modules",
                    {
                        "nooa.tracing": None,
                        "nooa.tracing._context_sideband": None,
                    },
                ):
                    # This should not raise — the ImportError is caught
                    await runtime._build_messages(method)
        finally:
            _current_llm_var.reset(token)

    @pytest.mark.asyncio
    async def test_general_exception_silenced(self):
        """General Exception from set_context_blocks is caught and logged."""
        from nooa.runtime.actor import _current_llm_var

        class _Agent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def task(self) -> str:
                """Work."""
                ...

        agent = _Agent()
        runtime = agent.runtime

        method = type(agent).task

        token = _current_llm_var.set(_TEST_LLM)
        try:
            # Create a mock module that raises on set_context_blocks
            mock_sideband = MagicMock()
            mock_sideband.set_context_blocks = MagicMock(side_effect=RuntimeError("tracing boom"))

            with patch.dict(
                "sys.modules",
                {
                    "nooa.tracing._context_sideband": mock_sideband,
                    "nooa.tracing": MagicMock(),
                },
            ):
                with patch(
                    "nooa.runtime.actor.render_context",
                    return_value=RenderResult(output=[], stats=_EMPTY_STATS, messages=[]),
                ):
                    # Should not raise — the Exception is caught
                    await runtime._build_messages(method)
        finally:
            _current_llm_var.reset(token)
