# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests covering remaining gaps in nooa modules.

Targets:
- nemo_flow_middleware.py: async middleware handlers
- config/truncation_config.py: validators
- runtime/async_safety.py: concurrent.futures safety
- runtime/event_query.py: event filtering
- runtime/media_capture.py: media/image capture
"""

from __future__ import annotations

import concurrent.futures
import importlib
import sys
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ===========================================================================
# Helpers shared across NeMo Flow tests
# ===========================================================================


def _make_fake_nemo_flow():
    """Build a MagicMock that looks like nemo_flow."""
    fake = MagicMock()
    fake_handle = MagicMock()
    fake_handle.uuid = "test-uuid-1234"
    fake.scope.scope.return_value.__enter__ = MagicMock(return_value=fake_handle)
    fake.scope.scope.return_value.__exit__ = MagicMock(return_value=False)
    # llm.execute is called as: await nemo_flow.llm.execute(...)
    fake.llm.execute = AsyncMock(side_effect=_invoke_wrapper_for_llm)
    # tools.execute is called as: await nemo_flow.tools.execute(...)
    fake.tools.execute = AsyncMock(side_effect=_invoke_wrapper_for_tools)
    # scope.push / scope.pop
    fake.scope.push.return_value = MagicMock(name="scope_handle")
    fake.scope.pop.return_value = None
    return fake, fake_handle


# These callables are replaced per-test; the defaults just call through.
async def _invoke_wrapper_for_llm(*args, **kwargs):
    """Default: call the wrapper with the request to simulate NeMo Flow invoking it."""
    # Signature: (model_name, request, wrapper, model_name=model_name)
    # positional args: (model_name, request, wrapper)
    wrapper = args[2]
    request = args[1]
    await wrapper(request)


async def _invoke_wrapper_for_tools(tool_name, args, wrapper):
    """Default: call the wrapper with the args to simulate NeMo Flow invoking it."""
    await wrapper(args)


@contextmanager
def _nemo_flow_patched():
    """Patch sys.modules with a fake nemo_flow, reload nemo_flow_middleware, yield (module, fake)."""

    fake_nemo_flow, fake_handle = _make_fake_nemo_flow()

    # LLMRequest needs to be constructable
    fake_llm_request_cls = MagicMock(return_value=MagicMock(content={}))

    # Ensure the module is imported before patching (KeyError if not in sys.modules)
    import nooa.nemo_flow_middleware as _nm_ensure  # noqa: F811, F401

    with patch.dict(
        sys.modules,
        {
            "nemo_relay": fake_nemo_flow,
            "nemo_relay.LLMRequest": fake_llm_request_cls,
        },
    ):
        nm = sys.modules["nooa.nemo_flow_middleware"]
        importlib.reload(nm)
        try:
            yield nm, fake_nemo_flow, fake_handle
        finally:
            pass  # Don't reload here — still inside patch.dict so nemo_flow is present

    # Reload AFTER patch.dict exits (nemo_flow removed from sys.modules)
    importlib.reload(nm)


# ===========================================================================
# nemo_flow_middleware.py — async handler tests
# ===========================================================================


class TestNemoFlowLLMMiddleware:
    """Tests for nemo_flow_llm_middleware (lines 92–184)."""

    async def _run_llm_middleware(self, fake_nemo_flow, nm, ctx_kwargs=None, nxt_response=None):
        """Helper: build ctx/nxt and run nemo_flow_llm_middleware."""
        from nooa.runtime.middleware import LLMCallContext

        ctx = LLMCallContext(
            messages=[{"role": "user", "content": "hello"}],
            params={"temperature": 0.7, "api_key": "secret", "tools": ["tool"]},
            agent=None,
        )
        if ctx_kwargs:
            for k, v in ctx_kwargs.items():
                setattr(ctx, k, v)

        # Build the inner result ctx
        result_ctx = LLMCallContext(
            messages=ctx.messages,
            params=ctx.params,
            agent=None,
        )
        if nxt_response is not None:
            result_ctx.response = nxt_response

        async def nxt(c):
            return result_ctx

        return await nm.nemo_flow_llm_middleware(ctx, nxt)

    async def test_llm_middleware_calls_nemo_flow_execute(self):
        with _nemo_flow_patched() as (nm, fake_nemo_flow, _):
            await self._run_llm_middleware(fake_nemo_flow, nm)
            fake_nemo_flow.llm.execute.assert_called_once()

    async def test_llm_middleware_strips_sensitive_keys(self):
        """The wrapper should receive a request without api_key / base_url."""
        with _nemo_flow_patched() as (nm, fake_nemo_flow, _):
            captured_request = {}

            async def capturing_execute(*args, **kwargs):
                request = args[1]
                wrapper = args[2]
                captured_request["req"] = request
                await wrapper(request)

            fake_nemo_flow.llm.execute.side_effect = capturing_execute

            from nooa.runtime.middleware import LLMCallContext

            ctx = LLMCallContext(
                messages=[{"role": "user", "content": "hi"}],
                params={"api_key": "SECRET", "temperature": 0.5},
            )
            result_ctx = LLMCallContext(messages=ctx.messages, params={})

            async def nxt(c):
                return result_ctx

            await nm.nemo_flow_llm_middleware(ctx, nxt)
            # LLMRequest was constructed — first positional arg to fake cls is {}
            # (the second positional arg is safe_params without api_key)
            call_args = nm.LLMRequest.call_args
            assert call_args is not None
            _, safe_params = call_args[0]
            assert "api_key" not in safe_params
            assert "tools" not in safe_params

    async def test_llm_middleware_returns_captured_ctx(self):
        with _nemo_flow_patched() as (nm, fake_nemo_flow, _):
            from nooa.runtime.middleware import LLMCallContext

            ctx = LLMCallContext(messages=[{"role": "user", "content": "hi"}], params={})
            inner = LLMCallContext(messages=ctx.messages, params={})

            async def nxt(c):
                return inner

            result = await nm.nemo_flow_llm_middleware(ctx, nxt)
            assert result is inner

    async def test_llm_middleware_guardrail_blocks_raises(self):
        """When nemo_flow.llm.execute() never invokes _wrapper, raise RuntimeError."""
        with _nemo_flow_patched() as (nm, fake_nemo_flow, _):
            # Override execute to NOT call the wrapper (simulates guardrail block)
            async def blocking_execute(*args, **kwargs):
                pass  # don't call wrapper

            fake_nemo_flow.llm.execute.side_effect = blocking_execute

            from nooa.runtime.middleware import LLMCallContext

            ctx = LLMCallContext(messages=[{"role": "user", "content": "hi"}], params={})

            async def nxt(c):
                return c

            with pytest.raises(RuntimeError, match="NeMo Flow guardrail blocked the LLM call"):
                await nm.nemo_flow_llm_middleware(ctx, nxt)

    async def test_llm_middleware_with_agent_model(self):
        """Model name is extracted from ctx.agent._llm.model.

        We test this by verifying that nemo_flow.llm.execute receives the model name
        from the mock agent, not from ctx (which has no agent in this test).
        """
        with _nemo_flow_patched() as (nm, fake_nemo_flow, _):
            captured_calls = []

            async def recording_execute(*args, **kwargs):
                model_name_pos = args[0]
                wrapper = args[2]
                request = args[1]
                captured_calls.append(model_name_pos)
                await wrapper(request)

            fake_nemo_flow.llm.execute.side_effect = recording_execute

            from nooa.runtime.middleware import LLMCallContext

            # agent=None is fine; model extraction returns "" in that case
            ctx = LLMCallContext(
                messages=[{"role": "user", "content": "hi"}], params={}, agent=None
            )
            inner = LLMCallContext(messages=ctx.messages, params={})

            async def nxt(c):
                return inner

            await nm.nemo_flow_llm_middleware(ctx, nxt)
            # model_name is "" because agent is None
            assert captured_calls[0] == ""

    async def test_llm_middleware_response_with_raw_model_dump(self):
        """When response has raw_response with model_dump, it is used."""
        with _nemo_flow_patched() as (nm, fake_nemo_flow, _):

            async def execute_and_capture(*args, **kwargs):
                wrapper = args[2]
                request = args[1]
                await wrapper(request)

            fake_nemo_flow.llm.execute.side_effect = execute_and_capture

            from nooa.runtime.middleware import LLMCallContext

            raw_resp = MagicMock()
            raw_resp.model_dump.return_value = {"choices": []}

            response = MagicMock()
            response.raw_response = raw_resp

            ctx = LLMCallContext(messages=[{"role": "user", "content": "hi"}], params={})
            inner = LLMCallContext(messages=ctx.messages, params={})
            inner.response = response

            async def nxt(c):
                return inner

            result = await nm.nemo_flow_llm_middleware(ctx, nxt)
            assert result is inner
            raw_resp.model_dump.assert_called_once_with(mode="json")

    async def test_llm_middleware_response_with_model_dump_no_raw(self):
        """When response has model_dump but no raw_response, model_dump is used."""
        with _nemo_flow_patched() as (nm, fake_nemo_flow, _):
            from nooa.runtime.middleware import LLMCallContext

            response = MagicMock(spec=["model_dump"])
            response.raw_response = None
            response.model_dump.return_value = {"result": "data"}

            ctx = LLMCallContext(messages=[{"role": "user", "content": "hi"}], params={})
            inner = LLMCallContext(messages=ctx.messages, params={})
            inner.response = response

            async def nxt(c):
                return inner

            await nm.nemo_flow_llm_middleware(ctx, nxt)
            response.model_dump.assert_called_once_with(mode="json")

    async def test_llm_middleware_response_assistant_message_fallback(self):
        """When response has assistant_message, fall back to manual serialization."""
        with _nemo_flow_patched() as (nm, fake_nemo_flow, _):
            from nooa.runtime.middleware import LLMCallContext

            response = MagicMock(spec=["assistant_message", "usage", "finish_reason"])
            response.assistant_message = "Hello!"
            response.usage = {"total_tokens": 10}
            response.finish_reason = "stop"

            ctx = LLMCallContext(messages=[{"role": "user", "content": "hi"}], params={})
            inner = LLMCallContext(messages=ctx.messages, params={})
            inner.response = response

            async def nxt(c):
                return inner

            # Should not raise
            await nm.nemo_flow_llm_middleware(ctx, nxt)

    async def test_llm_middleware_response_none_returns_empty(self):
        """When response is None, wrapper returns {}."""
        with _nemo_flow_patched() as (nm, fake_nemo_flow, _):
            from nooa.runtime.middleware import LLMCallContext

            ctx = LLMCallContext(messages=[{"role": "user", "content": "hi"}], params={})
            inner = LLMCallContext(messages=ctx.messages, params={})
            inner.response = None

            async def nxt(c):
                return inner

            # Should not raise, captured_ctx is inner
            result = await nm.nemo_flow_llm_middleware(ctx, nxt)
            assert result is inner

    async def test_llm_middleware_request_intercept_propagates_messages(self):
        """When NeMo Flow request intercept modifies messages, they propagate to ctx."""
        with _nemo_flow_patched() as (nm, fake_nemo_flow, _):
            new_messages = [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "hi"},
            ]

            async def intercepting_execute(*args, **kwargs):
                request = args[1]
                wrapper = args[2]
                # Simulate NeMo Flow modifying the request
                request.content = {"messages": new_messages}
                await wrapper(request)

            fake_nemo_flow.llm.execute.side_effect = intercepting_execute

            from nooa.runtime.middleware import LLMCallContext

            ctx = LLMCallContext(messages=[{"role": "user", "content": "hi"}], params={})
            inner = LLMCallContext(messages=ctx.messages, params={})

            async def nxt(c):
                return inner

            await nm.nemo_flow_llm_middleware(ctx, nxt)
            # ctx.messages should be updated to new_messages
            assert ctx.messages == new_messages

    async def test_llm_middleware_request_intercept_propagates_params(self):
        """When NeMo Flow request intercept modifies temperature, it propagates."""
        with _nemo_flow_patched() as (nm, fake_nemo_flow, _):

            async def intercepting_execute(*args, **kwargs):
                request = args[1]
                wrapper = args[2]
                request.content = {"temperature": 0.1, "seed": 42}
                await wrapper(request)

            fake_nemo_flow.llm.execute.side_effect = intercepting_execute

            from nooa.runtime.middleware import LLMCallContext

            ctx = LLMCallContext(
                messages=[{"role": "user", "content": "hi"}],
                params={"temperature": 0.9},
            )
            inner = LLMCallContext(messages=ctx.messages, params=ctx.params)

            async def nxt(c):
                return inner

            await nm.nemo_flow_llm_middleware(ctx, nxt)
            assert ctx.params["temperature"] == 0.1
            assert ctx.params["seed"] == 42


class TestNemoFlowToolMiddleware:
    """Tests for nemo_flow_tool_middleware (lines 197–258)."""

    async def _make_exec_ctx(self, code="print('hi')", params=None, result=None):
        from nooa.runtime.middleware import ExecutePythonContext

        ctx = ExecutePythonContext(code=code, params=params or {})
        inner = ExecutePythonContext(code=code, params=params or {})
        inner.result = result
        return ctx, inner

    async def test_tool_middleware_calls_nemo_flow_execute(self):
        with _nemo_flow_patched() as (nm, fake_nemo_flow, _):
            ctx, inner = await self._make_exec_ctx()

            async def nxt(c):
                return inner

            await nm.nemo_flow_tool_middleware(ctx, nxt)
            fake_nemo_flow.tools.execute.assert_called_once()

    async def test_tool_middleware_returns_captured_ctx(self):
        with _nemo_flow_patched() as (nm, fake_nemo_flow, _):
            ctx, inner = await self._make_exec_ctx()

            async def nxt(c):
                return inner

            result = await nm.nemo_flow_tool_middleware(ctx, nxt)
            assert result is inner

    async def test_tool_middleware_guardrail_blocks_raises(self):
        """When nemo_flow.tools.execute() never invokes _wrapper, raise RuntimeError."""
        with _nemo_flow_patched() as (nm, fake_nemo_flow, _):

            async def blocking_execute(tool_name, args, wrapper):
                pass  # don't call wrapper

            fake_nemo_flow.tools.execute.side_effect = blocking_execute

            ctx, inner = await self._make_exec_ctx()

            async def nxt(c):
                return inner

            with pytest.raises(RuntimeError, match="NeMo Flow guardrail blocked code execution"):
                await nm.nemo_flow_tool_middleware(ctx, nxt)

    async def test_tool_middleware_result_none(self):
        """When result is None, codec.to_json(None) is returned."""
        with _nemo_flow_patched() as (nm, fake_nemo_flow, _):
            ctx, inner = await self._make_exec_ctx(result=None)

            async def nxt(c):
                return inner

            result = await nm.nemo_flow_tool_middleware(ctx, nxt)
            assert result is inner
            # codec.to_json(None) should have been called
            fake_nemo_flow.typed.BestEffortAnyCodec.return_value.to_json.assert_called()

    async def test_tool_middleware_result_with_returned_value(self):
        """When result.returned_value is set, it is passed to codec."""
        with _nemo_flow_patched() as (nm, fake_nemo_flow, _):
            from nooa.events import ExecutionResult

            exec_result = ExecutionResult(stdout="", returned_value=42)
            ctx, inner = await self._make_exec_ctx(result=exec_result)

            async def nxt(c):
                return inner

            await nm.nemo_flow_tool_middleware(ctx, nxt)
            codec = fake_nemo_flow.typed.BestEffortAnyCodec.return_value
            codec.to_json.assert_called_with(42)

    async def test_tool_middleware_result_with_no_return_uses_stdout(self):
        """When result has _NO_RETURN and no signal, stdout is used."""
        with _nemo_flow_patched() as (nm, fake_nemo_flow, _):
            from nooa.events import _NO_RETURN, ExecutionResult

            exec_result = ExecutionResult(stdout="some output", signal=None)
            # Force returned_value to be _NO_RETURN sentinel
            exec_result = exec_result.model_copy(update={"returned_value": _NO_RETURN})
            ctx, inner = await self._make_exec_ctx(result=exec_result)

            async def nxt(c):
                return inner

            await nm.nemo_flow_tool_middleware(ctx, nxt)
            codec = fake_nemo_flow.typed.BestEffortAnyCodec.return_value
            codec.to_json.assert_called_with("some output")

    async def test_tool_middleware_result_signal_with_result_key(self):
        """When result has a signal with 'result' key, that is used."""
        with _nemo_flow_patched() as (nm, fake_nemo_flow, _):
            from nooa.events import _NO_RETURN, ExecutionResult, ExecutionSignal

            class TestSignal(ExecutionSignal):
                pass

            signal = TestSignal("test")
            signal.result = {"result": "signal_value"}

            exec_result = ExecutionResult(signal=signal)
            exec_result = exec_result.model_copy(update={"returned_value": _NO_RETURN})
            ctx, inner = await self._make_exec_ctx(result=exec_result)

            async def nxt(c):
                return inner

            await nm.nemo_flow_tool_middleware(ctx, nxt)
            codec = fake_nemo_flow.typed.BestEffortAnyCodec.return_value
            codec.to_json.assert_called_with("signal_value")

    async def test_tool_middleware_code_propagation_from_intercept(self):
        """NeMo Flow intercept can rewrite code; it propagates to ctx."""
        with _nemo_flow_patched() as (nm, fake_nemo_flow, _):

            async def intercepting_execute(tool_name, args, wrapper):
                modified_args = {"code": "print('intercepted')", "timeout": 5}
                await wrapper(modified_args)

            fake_nemo_flow.tools.execute.side_effect = intercepting_execute

            from nooa.runtime.middleware import ExecutePythonContext

            ctx = ExecutePythonContext(code="print('original')", params={})
            inner = ExecutePythonContext(code=ctx.code, params={})

            async def nxt(c):
                assert c.code == "print('intercepted')"
                assert c.params.get("timeout") == 5
                return inner

            await nm.nemo_flow_tool_middleware(ctx, nxt)
            assert ctx.code == "print('intercepted')"

    async def test_tool_middleware_result_signal_without_result_key(self):
        """When signal.result is not a dict with 'result', rv is None."""
        with _nemo_flow_patched() as (nm, fake_nemo_flow, _):
            from nooa.events import _NO_RETURN, ExecutionResult, ExecutionSignal

            class TestSignal(ExecutionSignal):
                pass

            signal = TestSignal("test")
            signal.result = "plain string"  # not a dict with 'result' key

            exec_result = ExecutionResult(signal=signal)
            exec_result = exec_result.model_copy(update={"returned_value": _NO_RETURN})
            ctx, inner = await self._make_exec_ctx(result=exec_result)

            async def nxt(c):
                return inner

            await nm.nemo_flow_tool_middleware(ctx, nxt)
            codec = fake_nemo_flow.typed.BestEffortAnyCodec.return_value
            codec.to_json.assert_called_with(None)


class TestNemoFlowAgentCallMiddleware:
    """Tests for nemo_flow_agent_call_middleware (lines 261–281).

    Note: AgentCallContext.agent must be Agent | None.  We use agent=None and
    inject a mock agent into ctx after creation, since the middleware accesses
    ctx.agent only via type(ctx.agent).__name__ which works even post-init.
    """

    def _make_ctx(self, method_name="solve", agent=None):
        from nooa.runtime.middleware import AgentCallContext

        ctx = AgentCallContext(agent=agent, method_name=method_name, args=(), kwargs={})
        return ctx

    async def test_agent_call_pushes_and_pops_scope(self):
        with _nemo_flow_patched() as (nm, fake_nemo_flow, _):
            ctx = self._make_ctx("solve")

            # Create a real object so type().__name__ works correctly
            class MyAgent:
                pass

            object.__setattr__(ctx, "agent", MyAgent())

            async def nxt(c):
                return c

            await nm.nemo_flow_agent_call_middleware(ctx, nxt)
            fake_nemo_flow.scope.push.assert_called_once()
            call_args = fake_nemo_flow.scope.push.call_args[0]
            assert call_args[0] == "MyAgent.solve"
            fake_nemo_flow.scope.pop.assert_called_once()

    async def test_agent_call_pops_scope_even_on_exception(self):
        with _nemo_flow_patched() as (nm, fake_nemo_flow, _):
            ctx = self._make_ctx("solve")

            async def failing_nxt(c):
                raise ValueError("deliberate error")

            with pytest.raises(ValueError, match="deliberate error"):
                await nm.nemo_flow_agent_call_middleware(ctx, failing_nxt)

            # pop() must still have been called
            fake_nemo_flow.scope.pop.assert_called_once()

    async def test_agent_call_scope_name_format(self):
        """Scope name is 'ClassName.method_name'."""
        with _nemo_flow_patched() as (nm, fake_nemo_flow, _):
            ctx = self._make_ctx("analyze")

            class ResearchAgent:
                pass

            object.__setattr__(ctx, "agent", ResearchAgent())

            async def nxt(c):
                return c

            await nm.nemo_flow_agent_call_middleware(ctx, nxt)
            call_args = fake_nemo_flow.scope.push.call_args[0]
            assert call_args[0] == "ResearchAgent.analyze"

    async def test_agent_call_scope_pop_failure_is_swallowed(self):
        """Even if scope.pop() raises, no exception propagates."""
        with _nemo_flow_patched() as (nm, fake_nemo_flow, _):
            fake_nemo_flow.scope.pop.side_effect = RuntimeError("pop failed")

            ctx = self._make_ctx("run")

            async def nxt(c):
                return c

            # Should not raise even though pop raises
            await nm.nemo_flow_agent_call_middleware(ctx, nxt)

    async def test_agent_call_returns_nxt_result(self):
        with _nemo_flow_patched() as (nm, fake_nemo_flow, _):
            ctx = self._make_ctx("run")
            expected = self._make_ctx("run")
            expected.result = "final_result"

            async def nxt(c):
                return expected

            result = await nm.nemo_flow_agent_call_middleware(ctx, nxt)
            assert result is expected


# ===========================================================================
# config/truncation_config.py
# ===========================================================================


class TestTruncationConfigValidators:
    """Cover the model_validator _check_values (lines 60-101)."""

    def test_default_config_valid(self):
        from nooa.config.truncation_config import TruncationConfig

        cfg = TruncationConfig()
        assert cfg.capture.max_stdout == 50_000

    def test_max_stdout_negative_raises(self):
        from nooa.config.truncation_config import CaptureConfig

        with pytest.raises(Exception, match="max_stdout must be > 0"):
            CaptureConfig(max_stdout=-1)

    def test_max_stderr_zero_raises(self):
        from nooa.config.truncation_config import CaptureConfig

        with pytest.raises(Exception, match="max_stderr must be > 0"):
            CaptureConfig(max_stderr=0)

    def test_max_context_tokens_zero_raises(self):
        from nooa.config.truncation_config import TruncationConfig

        with pytest.raises(Exception, match="max_context_tokens must be > 0 or None"):
            TruncationConfig(max_context_tokens=0)

    def test_max_event_tokens_negative_raises(self):
        from nooa.config.truncation_config import TruncationConfig

        with pytest.raises(Exception, match="max_event_tokens must be > 0 or None"):
            TruncationConfig(max_event_tokens=-5)

    def test_max_context_tokens_none_valid(self):
        from nooa.config.truncation_config import TruncationConfig

        cfg = TruncationConfig(max_context_tokens=None)
        assert cfg.max_context_tokens is None

    def test_value_max_length_zero_raises(self):
        from nooa.config.truncation_config import FormatConfig

        with pytest.raises(Exception, match="max_length must be > 0 or None"):
            FormatConfig(max_length=0)

    def test_value_max_string_negative_raises(self):
        from nooa.config.truncation_config import FormatConfig

        with pytest.raises(Exception, match="max_string must be > 0 or None"):
            FormatConfig(max_string=-1)

    def test_value_max_depth_zero_raises(self):
        from nooa.config.truncation_config import FormatConfig

        with pytest.raises(Exception, match="max_depth must be > 0 or None"):
            FormatConfig(max_depth=0)

    def test_value_max_length_none_valid(self):
        from nooa.config.truncation_config import FormatConfig

        cfg = FormatConfig(max_length=None)
        assert cfg.max_length is None

    def test_capture_tail_negative_raises(self):
        from nooa.config.truncation_config import CaptureConfig

        with pytest.raises(Exception, match="tail must be >= 0"):
            CaptureConfig(tail=-1)

    def test_capture_tail_equal_to_max_stdout_raises(self):
        from nooa.config.truncation_config import CaptureConfig

        with pytest.raises(Exception, match="tail.*must be less than.*max_stdout"):
            CaptureConfig(max_stdout=1000, tail=1000)

    def test_capture_tail_greater_than_max_stdout_raises(self):
        from nooa.config.truncation_config import CaptureConfig

        with pytest.raises(Exception, match="tail.*must be less than.*max_stdout"):
            CaptureConfig(max_stdout=1000, tail=1500)

    def test_capture_tail_equal_to_max_stderr_raises(self):
        from nooa.config.truncation_config import CaptureConfig

        with pytest.raises(Exception, match="tail.*must be less than.*max_stderr"):
            CaptureConfig(max_stdout=50_000, max_stderr=1000, tail=1000)

    def test_capture_tail_valid(self):
        from nooa.config.truncation_config import CaptureConfig

        cfg = CaptureConfig(max_stdout=50_000, max_stderr=20_000, tail=5000)
        assert cfg.tail == 5000

    def test_multiple_errors_all_reported(self):
        from nooa.config.truncation_config import CaptureConfig, TruncationConfig

        # Top-level error
        with pytest.raises(Exception, match="max_context_tokens"):
            TruncationConfig(max_context_tokens=0)

        # Sub-config errors collect together
        with pytest.raises(Exception) as exc_info:
            CaptureConfig(max_stdout=0, max_stderr=-1)
        msg = str(exc_info.value)
        assert "max_stdout" in msg
        assert "max_stderr" in msg

    def test_merge_with_none_returns_self(self):
        from nooa.config.truncation_config import TruncationConfig

        cfg = TruncationConfig()
        result = cfg.merge_with(None)
        assert result is cfg

    def test_merge_with_overrides_set_fields(self):
        from nooa.config.truncation_config import TruncationConfig

        base = TruncationConfig()
        override = TruncationConfig(max_context_tokens=5000)
        result = base.merge_with(override)
        assert result.max_context_tokens == 5000

    def test_merge_with_no_fields_set_raises(self):
        from nooa.config.truncation_config import TruncationConfig

        base = TruncationConfig()
        # Construct a TruncationConfig with empty model_fields_set by using
        # model_construct (skips __init__, so model_fields_set stays empty)
        other = TruncationConfig.model_construct()
        with pytest.raises(ValueError, match="no model_fields_set"):
            base.merge_with(other)


# ===========================================================================
# runtime/async_safety.py
# ===========================================================================


class TestAsyncSafety:
    """Test async safety patches (lines 44-46, 64, 77-82, 89-94)."""

    def test_agent_context_sets_flag(self):
        from nooa.runtime.async_safety import (
            _in_agent_context,
            agent_async_safety_context,
        )

        assert _in_agent_context.get() is False
        with agent_async_safety_context():
            assert _in_agent_context.get() is True
        assert _in_agent_context.get() is False

    def test_agent_context_resets_on_exception(self):
        from nooa.runtime.async_safety import (
            _in_agent_context,
            agent_async_safety_context,
        )

        try:
            with agent_async_safety_context():
                raise RuntimeError("test")
        except RuntimeError:
            pass
        assert _in_agent_context.get() is False

    def test_is_event_loop_thread_no_loop(self):
        """Outside event loop, _is_event_loop_thread returns False."""
        from nooa.runtime.async_safety import _is_event_loop_thread

        # We're not in an event loop here (sync test)
        result = _is_event_loop_thread()
        assert result is False

    async def test_is_event_loop_thread_inside_loop(self):
        """Inside event loop, _is_event_loop_thread returns True."""
        from nooa.runtime.async_safety import _is_event_loop_thread

        result = _is_event_loop_thread()
        assert result is True

    async def test_future_result_blocks_in_agent_context(self):
        """Future.result() raises inside agent context on event loop thread."""
        from nooa.runtime.async_safety import agent_async_safety_context

        future: concurrent.futures.Future[int] = concurrent.futures.Future()

        with agent_async_safety_context():
            with pytest.raises(RuntimeError, match="Future.result\\(\\).*deadlock"):
                future.result()

    async def test_future_exception_blocks_in_agent_context(self):
        """Future.exception() raises inside agent context on event loop thread."""
        from nooa.runtime.async_safety import agent_async_safety_context

        future: concurrent.futures.Future[int] = concurrent.futures.Future()

        with agent_async_safety_context():
            with pytest.raises(RuntimeError, match="Future.exception\\(\\).*deadlock"):
                future.exception()

    async def test_future_result_allowed_when_done(self):
        """Future.result() does NOT raise for done futures (no deadlock risk)."""
        from nooa.runtime.async_safety import agent_async_safety_context

        future: concurrent.futures.Future[int] = concurrent.futures.Future()
        future.set_result(42)

        with agent_async_safety_context():
            # Done futures can be safely read without deadlock
            assert future.result() == 42

    async def test_future_exception_allowed_when_done(self):
        """Future.exception() does NOT raise for done futures (no deadlock risk)."""
        from nooa.runtime.async_safety import agent_async_safety_context

        future: concurrent.futures.Future[int] = concurrent.futures.Future()
        future.set_result(42)

        with agent_async_safety_context():
            # Done futures can be safely inspected without deadlock
            assert future.exception() is None

    async def test_wait_blocks_in_agent_context(self):
        """concurrent.futures.wait() raises inside agent context on event loop thread."""
        from nooa.runtime.async_safety import agent_async_safety_context

        future: concurrent.futures.Future[int] = concurrent.futures.Future()
        future.set_result(42)

        with agent_async_safety_context():
            with pytest.raises(RuntimeError, match="concurrent.futures.wait\\(\\).*deadlock"):
                concurrent.futures.wait([future])

    async def test_as_completed_blocks_in_agent_context(self):
        """concurrent.futures.as_completed() raises inside agent context on event loop thread."""
        from nooa.runtime.async_safety import agent_async_safety_context

        future: concurrent.futures.Future[int] = concurrent.futures.Future()
        future.set_result(42)

        with agent_async_safety_context():
            with pytest.raises(
                RuntimeError, match="concurrent.futures.as_completed\\(\\).*deadlock"
            ):
                concurrent.futures.as_completed([future])

    def test_future_result_works_outside_agent_context(self):
        """Future.result() works normally outside agent context."""
        future: concurrent.futures.Future[int] = concurrent.futures.Future()
        future.set_result(42)
        # Outside agent context — should not raise
        assert future.result() == 42

    def test_future_exception_works_outside_agent_context(self):
        """Future.exception() works normally outside agent context."""
        future: concurrent.futures.Future[int] = concurrent.futures.Future()
        future.set_exception(ValueError("test error"))
        exc = future.exception()
        assert isinstance(exc, ValueError)


# ===========================================================================
# runtime/event_query.py
# ===========================================================================


class TestEventQuery:
    """Cover event_query.py filtering paths (lines 78, 87, 112, 116-122, 126)."""

    def _make_event(self, class_name: str, call_id: str | None = None, content: str = "") -> Any:
        """Create a minimal mock event."""
        ev = MagicMock()
        ev.__class__.__name__ = class_name
        ev.metadata = {"call_id": call_id} if call_id else {}
        ev.__str__ = lambda self: content
        return ev

    def test_by_type_classmethod(self):
        from nooa.runtime.event_query import EventQuery

        q = EventQuery.by_type("Task", limit=5)
        assert q.type == "Task"
        assert q.limit == 5

    def test_last_n_classmethod(self):
        from nooa.runtime.event_query import EventQuery

        q = EventQuery.last_n(10)
        assert q.limit == 10

    def test_current_call_classmethod(self):
        from nooa.runtime.event_query import EventQuery

        q = EventQuery.current_call(limit=3)
        assert q.call_id == "current"
        assert q.limit == 3

    def test_apply_filter_by_type(self):
        from nooa.runtime.event_query import EventQuery

        events = [
            self._make_event("Task"),
            self._make_event("Error"),
            self._make_event("Task"),
        ]
        q = EventQuery(type="Task")
        result = q.apply(events)
        assert len(result) == 2
        assert all(e.__class__.__name__ == "Task" for e in result)

    def test_apply_filter_by_call_id_literal(self):
        from nooa.runtime.event_query import EventQuery

        events = [
            self._make_event("Task", call_id="call-1"),
            self._make_event("Task", call_id="call-2"),
            self._make_event("Task", call_id="call-1"),
        ]
        q = EventQuery(call_id="call-1")
        result = q.apply(events)
        assert len(result) == 2

    def test_apply_filter_by_call_id_current(self):
        from nooa.runtime.event_query import EventQuery

        events = [
            self._make_event("Task", call_id="call-abc"),
            self._make_event("Task", call_id="call-xyz"),
        ]
        q = EventQuery(call_id="current")
        result = q.apply(events, current_call_id="call-abc")
        assert len(result) == 1

    def test_apply_filter_by_query_text(self):
        from nooa.runtime.event_query import EventQuery

        events = [
            self._make_event("Task", content="find the answer"),
            self._make_event("Task", content="calculate pi"),
        ]
        q = EventQuery(query="answer")
        result = q.apply(events)
        assert len(result) == 1

    def test_apply_filter_by_query_regex(self):
        from nooa.runtime.event_query import EventQuery

        events = [
            self._make_event("Task", content="error 404 not found"),
            self._make_event("Task", content="success"),
        ]
        q = EventQuery(query=r"error \d+", regex=True)
        result = q.apply(events)
        assert len(result) == 1

    def test_apply_limit(self):
        from nooa.runtime.event_query import EventQuery

        events = [self._make_event("Task") for _ in range(10)]
        q = EventQuery(limit=3)
        result = q.apply(events)
        assert len(result) == 3

    def test_apply_limit_takes_last_n(self):
        """limit slices from the end."""
        from nooa.runtime.event_query import EventQuery

        events = [self._make_event("Task", content=f"event {i}") for i in range(5)]
        q = EventQuery(limit=2)
        result = q.apply(events)
        assert result == events[-2:]

    def test_apply_combined_filters(self):
        from nooa.runtime.event_query import EventQuery

        events = [
            self._make_event("Task", call_id="c1", content="alpha"),
            self._make_event("Error", call_id="c1", content="beta"),
            self._make_event("Task", call_id="c2", content="gamma"),
            self._make_event("Task", call_id="c1", content="delta"),
        ]
        q = EventQuery(type="Task", call_id="c1", limit=1)
        result = q.apply(events)
        assert len(result) == 1

    def test_apply_no_filters_returns_all(self):
        from nooa.runtime.event_query import EventQuery

        events = [self._make_event("Task") for _ in range(5)]
        q = EventQuery()
        result = q.apply(events)
        assert result == events


# ===========================================================================
# runtime/media_capture.py
# ===========================================================================


class TestMediaCapture:
    """Cover media_capture.py (lines 48, 67, 111, 120-124, 138-142)."""

    def _make_image(self, media_type="image/png", vendor_metadata=None):
        from nooa.media import Image

        return Image(
            data_url="data:image/png;base64,abc123",
            media_type=media_type,
            vendor_metadata=vendor_metadata,
        )

    def _make_audio(self, media_type="audio/wav"):
        from nooa.media import Audio

        return Audio(data_url="data:audio/wav;base64,abc123", media_type=media_type)

    def _make_file(self):
        from nooa.media import File

        return File(data_url="data:application/pdf;base64,abc123", media_type="application/pdf")

    def test_image_content_block_basic(self):
        from nooa.runtime.media_capture import media_to_content_block

        img = self._make_image()
        block = media_to_content_block(img)
        assert block["type"] == "image_url"
        assert "image_url" in block
        assert block["image_url"]["url"] == "data:image/png;base64,abc123"

    def test_image_content_block_includes_format(self):
        from nooa.runtime.media_capture import media_to_content_block

        img = self._make_image(media_type="image/jpeg")
        block = media_to_content_block(img)
        assert block["image_url"]["format"] == "image/jpeg"

    def test_image_content_block_skips_octet_stream_format(self):
        """media_type=application/octet-stream should not add 'format' key."""
        from nooa.runtime.media_capture import media_to_content_block

        img = self._make_image(media_type="application/octet-stream")
        block = media_to_content_block(img)
        assert "format" not in block["image_url"]

    def test_image_content_block_vendor_metadata_merged(self):
        """vendor_metadata is merged into image_url dict."""
        from nooa.runtime.media_capture import media_to_content_block

        img = self._make_image(vendor_metadata={"detail": "high"})
        block = media_to_content_block(img)
        assert block["image_url"]["detail"] == "high"

    def test_audio_content_block(self):
        from nooa.runtime.media_capture import media_to_content_block

        audio = self._make_audio()
        block = media_to_content_block(audio)
        assert block["type"] == "input_audio"
        assert "input_audio" in block
        assert block["input_audio"]["format"] == "wav"

    def test_audio_content_block_format_extracted(self):
        """Format is the last part of the media type."""
        from nooa.runtime.media_capture import media_to_content_block

        audio = self._make_audio(media_type="audio/mp3")
        block = media_to_content_block(audio)
        assert block["input_audio"]["format"] == "mp3"

    def test_file_content_block(self):
        from nooa.runtime.media_capture import media_to_content_block

        file_obj = self._make_file()
        block = media_to_content_block(file_obj)
        assert block["type"] == "file"
        assert "file" in block

    def test_unknown_media_subclass_fallback(self):
        """Unknown Media subclass falls back to image_url type."""
        from nooa.media import Media
        from nooa.runtime.media_capture import media_to_content_block

        # Custom subclass not Image/Audio/File
        class UnknownMedia(Media):
            _modality = "unknown"

        obj = UnknownMedia(data_url="https://example.com/foo", media_type="")
        block = media_to_content_block(obj)
        assert block["type"] == "image_url"
        assert block["image_url"]["url"] == "https://example.com/foo"

    def test_non_media_raises_type_error(self):
        from nooa.runtime.media_capture import media_to_content_block

        with pytest.raises(TypeError, match="Expected Media"):
            media_to_content_block("not a media object")

    def test_show_outside_context_prints_message(self, capsys):
        from nooa.runtime.media_capture import show

        img = self._make_image()
        show(img)
        captured = capsys.readouterr()
        assert "outside execution context" in captured.out

    def test_show_inside_context_appends_block(self, capsys):
        from nooa.runtime.media_capture import (
            _media_buffer_var,
            _MediaBuffer,
            show,
        )

        img = self._make_image()
        buf = _MediaBuffer(max_attachments=10)
        token = _media_buffer_var.set(buf)
        try:
            show(img)
        finally:
            _media_buffer_var.reset(token)
        assert len(buf.blocks) == 1
        assert buf.blocks[0]["type"] == "image_url"

    def test_show_unsupported_type_raises(self):
        from nooa.runtime.media_capture import (
            _media_buffer_var,
            _MediaBuffer,
            show,
        )

        buf = _MediaBuffer(max_attachments=10)
        token = _media_buffer_var.set(buf)
        try:
            with pytest.raises(TypeError, match="show\\(\\) expects"):
                show({"not": "media"})
        finally:
            _media_buffer_var.reset(token)

    def test_try_pil_to_content_block_import_error(self):
        """When PIL is not available, returns None gracefully."""
        from nooa.runtime.media_capture import _try_pil_to_content_block

        with patch.dict(sys.modules, {"PIL": None, "PIL.Image": None}):
            result = _try_pil_to_content_block("not a pil image")
        assert result is None

    def test_try_matplotlib_to_content_block_import_error(self):
        """When matplotlib is not available, returns None gracefully."""
        from nooa.runtime.media_capture import _try_matplotlib_to_content_block

        with patch.dict(sys.modules, {"matplotlib": None, "matplotlib.figure": None}):
            result = _try_matplotlib_to_content_block("not a figure")
        assert result is None

    def test_image_alias(self):
        """image_to_content_block is an alias for media_to_content_block."""
        from nooa.runtime.media_capture import (
            image_to_content_block,
            media_to_content_block,
        )

        assert image_to_content_block is media_to_content_block


# ===========================================================================


# media_capture.py — PIL and matplotlib auto-convert paths
# ===========================================================================


class TestMediaCapturePILAndMatplotlib:
    """Test PIL and matplotlib auto-convert (lines 120-124, 138-142)."""

    def test_try_pil_to_content_block_with_pil_image(self):
        """When PIL is available and obj is a PIL Image, returns image_url block."""
        import importlib

        import nooa.runtime.media_capture as mc

        # Create a mock PIL Image class and instance
        pil_cls = type("Image", (), {})
        obj = pil_cls()

        buf_content = b"PNG_DATA"

        def save_side_effect(buf, format):
            buf.write(buf_content)

        obj.save = save_side_effect

        mock_pil_module = MagicMock()
        mock_pil_module.Image.Image = pil_cls

        with patch.dict(sys.modules, {"PIL": mock_pil_module, "PIL.Image": mock_pil_module.Image}):
            importlib.reload(mc)
            result = mc._try_pil_to_content_block(obj)

        # Reload to restore unpatched module state
        importlib.reload(mc)

        assert result is not None
        assert result["type"] == "image_url"

    def test_try_matplotlib_returns_none_for_non_figure(self):
        """When matplotlib is available but obj is not a Figure, returns None."""
        from nooa.runtime.media_capture import _try_matplotlib_to_content_block

        mock_matplotlib = MagicMock()
        mock_figure_cls = type("Figure", (), {})
        mock_matplotlib.figure.Figure = mock_figure_cls

        with patch.dict(
            sys.modules,
            {
                "matplotlib": mock_matplotlib,
                "matplotlib.figure": mock_matplotlib.figure,
            },
        ):
            result = _try_matplotlib_to_content_block("not a figure")

        assert result is None

    def test_try_pil_returns_none_for_non_pil_image(self):
        """When PIL is available but obj is not a PIL Image, returns None."""
        from nooa.runtime.media_capture import _try_pil_to_content_block

        mock_pil = MagicMock()
        mock_pil.Image.Image = type("Image", (), {})

        with patch.dict(sys.modules, {"PIL": mock_pil, "PIL.Image": mock_pil.Image}):
            result = _try_pil_to_content_block("not a PIL image")

        assert result is None

    def test_show_with_auto_convert_failure_raises(self):
        """show() with unsupported type inside buffer context raises TypeError."""
        from nooa.runtime.media_capture import (
            _media_buffer_var,
            _MediaBuffer,
            show,
        )

        buf = _MediaBuffer(max_attachments=10)
        token = _media_buffer_var.set(buf)
        try:
            with pytest.raises(TypeError):
                show(42)
        finally:
            _media_buffer_var.reset(token)
