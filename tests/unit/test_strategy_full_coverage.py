# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests targeting specific uncovered lines in codeact.py, predict.py, and pure_python.py.

Each test class or method documents which source lines it targets.
"""

import json
import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.config.strategy_config import PredictConfig
from nooa.errors import GenerationError
from nooa.events import ExecutionResult
from nooa.strategies.codeact import (
    CodeActStrategy,
    _ReturnResultSignal,
)
from nooa.strategies.predict import PredictStrategy
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_TEST_LLM = FakeLLMClient()


def _resp(content: str, tool_calls: list | None = None) -> LLMResponse:
    finish_reason = "tool_calls" if tool_calls else "stop"
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        assistant_message={"role": "assistant", "content": content},
    )


def _tool_call(code: str, call_id: str = "call_1") -> ToolCall:
    return ToolCall(id=call_id, name="execute_python", arguments=json.dumps({"code": code}))


def _return_result(call_id: str = "call_return", result: Any = None) -> ToolCall:
    return ToolCall(
        id=call_id,
        name="return_result",
        arguments=json.dumps({"result": result}),
    )


def _unknown_tool(name: str = "my_tool", call_id: str = "call_u1", **kwargs) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=json.dumps(kwargs))


def _llm_resp(content: str, reasoning: str | None = None) -> LLMResponse:
    """Create an LLMResponse with string content (for PREDICT tests)."""
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": content},
        reasoning=reasoning,
    )


# ===========================================================================
# CODEACT: Line 345 — is_from_blocked_module check in _format_namespace_doc
# ===========================================================================


class TestCodeActBlockedModuleFiltering:
    """Cover line 345: is_from_blocked_module(obj, blocked) → continue."""

    @pytest.mark.asyncio
    async def test_execution_context_filters_blocked_modules(self):
        """When the module context includes a blocked module, it should be
        filtered out of the execution context output (line 345)."""
        import subprocess  # This is in DEFAULT_BLOCKED_MODULES

        strat = CodeActStrategy(config=CodeActConfig())

        # Create a fake agent module that has a blocked import
        fake_module = types.ModuleType("test_agent_module")
        fake_module.__name__ = "test_agent_module"
        fake_module.__dict__["subprocess"] = subprocess  # blocked
        fake_module.__dict__["json"] = json  # not blocked

        # Mock RuntimeServices with an agent whose module has blocked imports
        mock_agent = MagicMock()
        mock_agent.__class__.__name__ = "TestAgent"

        # Patch _extract_module_context to return our controlled context
        context = {"subprocess": subprocess, "json": json}
        with patch.object(strat, "_extract_module_context", return_value=context):
            mock_runtime = MagicMock()
            mock_runtime.agent = mock_agent
            # Call execution_context which filters via is_from_blocked_module
            result = await strat.execution_context(mock_runtime)

        # 'json' should be present, 'subprocess' should be filtered
        assert "json" in result
        assert "subprocess" not in result


# ===========================================================================
# CODEACT: Lines 436-437 — has_context check adds pin/unpin instructions
# ===========================================================================


class TestCodeActHasContextSkillInstructions:
    """Skills table was removed from execution_context — pin/unpin no longer rendered."""

    @pytest.mark.asyncio
    async def test_context_visible_no_longer_adds_pin_unpin(self):
        """After removing Skills table, pin/unpin instructions are not in execution_context."""
        from nooa.agentdoc import spec
        from nooa.skill import Skill

        class AgentWithContext(Agent, llm=_TEST_LLM):
            my_skill: Any = None

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                spec(self, "context", hidden=False)

        spec(AgentWithContext, "context", hidden=False)

        agent = AgentWithContext(llm=_TEST_LLM)
        agent.my_skill = Skill(content="A test skill")

        strat = CodeActStrategy(config=CodeActConfig())
        mock_runtime = MagicMock()
        mock_runtime.agent = agent

        with patch.object(strat, "_extract_module_context", return_value={}):
            result = await strat.execution_context(mock_runtime)

        assert "Pin to context" not in result
        assert "## Skills" not in result


# ===========================================================================
# CODEACT: Line 587 — generation_id is None → raise RuntimeError
# ===========================================================================


class TestCodeActGenerationIdNone:
    """Cover line 587: generation_id is None → RuntimeError."""

    @pytest.mark.asyncio
    async def test_generation_id_none_raises(self):
        """When runtime.get_generation_id() returns None during execute,
        a RuntimeError should be raised (line 586-590)."""
        from nooa.runtime.actor import ActorRuntime

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def do_task(self) -> int:
                """Do a task."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[_resp("", tool_calls=[_return_result(result=42)])]
        )
        agent = TestAgent(llm=fake_llm)

        # Patch get_generation_id on the actual runtime implementation
        with patch.object(ActorRuntime, "get_generation_id", return_value=None):
            with pytest.raises(RuntimeError, match="get_generation_id.*returned None"):
                await agent.do_task()


# ===========================================================================
# CODEACT: Lines 626-627 — BlockSyntaxError handler + continue
# ===========================================================================


class TestCodeActBlockSyntaxError:
    """Cover lines 625-627: BlockSyntaxError catch → handler → continue."""

    @pytest.mark.asyncio
    async def test_block_syntax_error_recovery(self):
        """When generate() raises BlockSyntaxError, the loop should
        handle it and continue to the next iteration (lines 625-627)."""
        from nooa.context_blocks.exceptions import BlockSyntaxError

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=3)))
            async def do_task(self) -> int:
                """Do a task."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                # Second call after error recovery
                _resp("", tool_calls=[_return_result(result=99)]),
            ]
        )
        agent = TestAgent(llm=fake_llm)

        call_count = [0]

        # Patch within the agent execution
        from nooa.runtime.actor import ActorRuntime

        orig_gen = ActorRuntime.generate

        async def patched_generate(self_rt, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise BlockSyntaxError(
                    key="test_block",
                    expr="invalid python [[[",
                    original_error=SyntaxError("invalid syntax"),
                )
            return await orig_gen(self_rt, *args, **kwargs)

        call_count[0] = 0
        with patch.object(ActorRuntime, "generate", patched_generate):
            result = await agent.do_task()

        assert result == 99


# ===========================================================================
# CODEACT: Lines 919, 921 — tool call result handling after translation
# ===========================================================================


class TestCodeActTranslatedToolCallResults:
    """Cover lines 918-921: result handling after translated tool call execution."""

    @pytest.mark.asyncio
    async def test_translated_tool_call_basic(self):
        """When an unknown tool call is translated, lines 897-917 execute
        the translation path through _handle_execute_python."""

        class TestAgent(Agent, llm=_TEST_LLM):
            def get_value(self) -> int:
                """Return a value."""
                return 42

            @strategy(CodeActStrategy(config=CodeActConfig(translate_tool_calls=True)))
            async def do_task(self) -> int:
                """Do a task."""
                ...

        # LLM tries to call get_value directly (unknown tool name),
        # gets translated to execute_python.
        fake_llm = FakeLLMClient(
            scripted_responses=[
                # Unknown tool "get_value" — will be translated to execute_python
                _resp("", tool_calls=[_unknown_tool("get_value", call_id="tc_1")]),
                # After translated call returns without TASK_COMPLETE, return result
                _resp("", tool_calls=[_return_result(result=42)]),
            ]
        )
        agent = TestAgent(llm=fake_llm)
        result = await agent.do_task()
        assert result == 42

    @pytest.mark.asyncio
    async def test_translated_tool_call_task_complete(self):
        """When _handle_execute_python returns TASK_COMPLETE after translation,
        line 921 returns the completed result."""

        class TestAgent(Agent, llm=_TEST_LLM):
            def get_value(self) -> int:
                """Return a value."""
                return 42

            @strategy(CodeActStrategy(config=CodeActConfig(translate_tool_calls=True)))
            async def do_task(self) -> int:
                """Do a task."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                # Unknown tool "get_value" — will be translated
                _resp("", tool_calls=[_unknown_tool("get_value", call_id="tc_1")]),
            ]
        )
        agent = TestAgent(llm=fake_llm)

        # Mock _handle_execute_python to return TASK_COMPLETE

        async def mock_handle(
            self_strat,
            runtime,
            tool_call,
            args,
            builtins,
            session,
            method_name,
            return_type,
            tool_call_event_id,
        ):
            return ("TASK_COMPLETE", 42)

        with patch.object(CodeActStrategy, "_handle_execute_python", mock_handle):
            result = await agent.do_task()
        assert result == 42

    @pytest.mark.asyncio
    async def test_translated_tool_call_error_returns_empty(self):
        """When translated tool call returns None (error),
        _ToolCallsResult() is returned (line 918-919)."""

        class TestAgent(Agent, llm=_TEST_LLM):
            def broken_method(self) -> int:
                """Method that raises an error."""
                raise ValueError("broken")

            @strategy(
                CodeActStrategy(config=CodeActConfig(translate_tool_calls=True, max_iterations=3))
            )
            async def do_task(self) -> int:
                """Do a task."""
                ...

        # LLM calls the broken method (gets translated to execute_python),
        # execution returns error → line 918-919 → then LLM provides correct result
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "",
                    tool_calls=[_unknown_tool("broken_method", call_id="tc_1")],
                ),
                _resp("", tool_calls=[_return_result(result=99)]),
            ]
        )
        agent = TestAgent(llm=fake_llm)
        result = await agent.do_task()
        assert result == 99


# ===========================================================================
# CODEACT: Lines 1122-1123, 1128 — error text formatting in return_result path
# ===========================================================================


class TestCodeActInlineReturnResultWithError:
    """Cover lines 1121-1128: error formatting when inline return_result has execution error."""

    @pytest.mark.asyncio
    async def test_inline_return_result_with_error_and_stderr(self):
        """When _execute_code returns a result with both signal (_ReturnResultSignal)
        AND error set, lines 1121-1128 format the error text (defensive path)."""
        from nooa.runtime.actor import ActorRuntime

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def do_task(self) -> int:
                """Do a task."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call("return_result(result=42)")]),
            ]
        )
        agent = TestAgent(llm=fake_llm)

        # Create a mock ExecutionResult that has BOTH signal and error
        mock_result = ExecutionResult(
            stdout="some output",
            stderr="some warning",
            error=ValueError("something went wrong"),
            signal=_ReturnResultSignal(result={"result": 42}),
            defined_methods={},
            returned_value=42,
            explicit_return=False,
        )

        async def mock_execute_code(self_rt, *args, **kwargs):
            return mock_result

        with patch.object(ActorRuntime, "execute_code", mock_execute_code):
            result = await agent.do_task()
        assert result == 42

    @pytest.mark.asyncio
    async def test_inline_return_result_with_error_no_stderr(self):
        """When error_text exists but stderr is empty, the 'else' branch
        of the ternary on line 1128-1131 is hit."""
        from nooa.runtime.actor import ActorRuntime

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def do_task(self) -> int:
                """Do a task."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call("return_result(result=42)")]),
            ]
        )
        agent = TestAgent(llm=fake_llm)

        # No stderr, but has error
        mock_result = ExecutionResult(
            stdout="",
            stderr="",
            error=ValueError("error occurred"),
            signal=_ReturnResultSignal(result={"result": 42}),
            defined_methods={},
        )

        async def mock_execute_code(self_rt, *args, **kwargs):
            return mock_result

        with patch.object(ActorRuntime, "execute_code", mock_execute_code):
            result = await agent.do_task()
        assert result == 42


# ===========================================================================
# CODEACT: Lines 1194-1195 — Exception in auto-completion validation
# ===========================================================================


class TestCodeActAutoCompletionException:
    """Cover lines 1194-1195: exception during auto-completion validation."""

    @pytest.mark.asyncio
    async def test_auto_completion_validation_exception(self):
        """When _try_validate_return_value raises an unexpected exception,
        the loop should continue (lines 1194-1195)."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def do_task(self) -> int:
                """Do a task."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                # Code with explicit return — triggers auto-completion validation
                _resp("", tool_calls=[_tool_call("return 'not_an_int'", call_id="call_1")]),
                # After validation fails, LLM returns correct result
                _resp("", tool_calls=[_return_result(result=42)]),
            ]
        )
        agent = TestAgent(llm=fake_llm)

        # Patch _try_validate_return_value to raise an unexpected exception
        with patch.object(
            CodeActStrategy,
            "_try_validate_return_value",
            side_effect=RuntimeError("unexpected validation error"),
        ):
            result = await agent.do_task()
        assert result == 42


# ===========================================================================
# CODEACT: Line 1208 — PydanticValidationError isinstance check
# ===========================================================================


class TestCodeActPydanticValidationErrorFormat:
    """Cover line 1205-1210: PydanticValidationError isinstance check for error formatting."""

    @pytest.mark.asyncio
    async def test_pydantic_validation_error_in_normal_path(self):
        """When code execution produces a PydanticValidationError with a
        returned_value, the error is formatted via format_validation_error
        (lines 1204-1210)."""
        from nooa.runtime.actor import ActorRuntime

        class Result(BaseModel):
            score: int
            label: str

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def do_task(self) -> Result:
                """Do a task returning Result."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                # Code execution that produces PydanticValidationError
                _resp(
                    "",
                    tool_calls=[_tool_call("x = 1", call_id="call_1")],
                ),
                # Follow up with correct result
                _resp(
                    "",
                    tool_calls=[_return_result(result={"score": 85, "label": "good"})],
                ),
            ]
        )
        agent = TestAgent(llm=fake_llm)

        # Create a PydanticValidationError
        try:
            Result(score="not_int", label=123)
        except PydanticValidationError as e:
            pydantic_error = e

        # Mock execute_code to return result with PydanticValidationError and returned_value
        mock_result = ExecutionResult(
            stdout="",
            stderr="",
            error=pydantic_error,
            defined_methods={},
            returned_value={"score": "not_int", "label": 123},
            explicit_return=False,
        )

        call_count = [0]
        orig_execute_code = ActorRuntime.execute_code

        async def mock_execute_code(self_rt, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_result
            return await orig_execute_code(self_rt, *args, **kwargs)

        with patch.object(ActorRuntime, "execute_code", mock_execute_code):
            result = await agent.do_task()
        assert isinstance(result, Result)
        assert result.score == 85

    @pytest.mark.asyncio
    async def test_inline_return_result_validation_failure(self):
        """When inline return_result validation fails, lines 1109-1118 and
        1133-1138 and 1159 are hit (validation_error path)."""
        from nooa.runtime.actor import ActorRuntime

        class Result(BaseModel):
            score: int

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def do_task(self) -> Result:
                """Do a task returning Result."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                # First: inline return_result with invalid data
                _resp("", tool_calls=[_tool_call("return_result(result={'score': 'bad'})")]),
                # Second: correct result via return_result tool
                _resp("", tool_calls=[_return_result(result={"score": 42})]),
            ]
        )
        agent = TestAgent(llm=fake_llm)

        # Create a mock result with signal but invalid data (will fail validation)
        mock_result = ExecutionResult(
            stdout="",
            stderr="",
            error=None,
            signal=_ReturnResultSignal(result={"result": {"score": "not_int"}}),
            defined_methods={},
        )

        call_count = [0]
        orig_execute_code = ActorRuntime.execute_code

        async def mock_execute_code(self_rt, *args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_result
            return await orig_execute_code(self_rt, *args, **kwargs)

        with patch.object(ActorRuntime, "execute_code", mock_execute_code):
            result = await agent.do_task()
        assert isinstance(result, Result)
        assert result.score == 42


# ===========================================================================
# CODEACT: Line 1494 — stub return for execute_python (never called)
# CODEACT: Line 1601 — stub return for return_result (never called)
# ===========================================================================


class TestCodeActStubReturns:
    """Cover lines 1494, 1601: stub callables inside tool definitions.

    These stubs exist as callable bodies for the Tool definitions but are
    never actually called (tool calls are handled in the execute loop).
    We call them directly to cover the lines.
    """

    def test_execute_python_stub(self):
        """execute_python stub returns empty string (line 1494)."""
        strat = CodeActStrategy(config=CodeActConfig())
        tool = strat._build_execute_python_tool()
        # The callable is the inner function
        result = tool.callable("print('hello')")
        assert result == ""

    def test_return_result_stub(self):
        """return_result stub returns the passed result (line 1601)."""
        strat = CodeActStrategy(config=CodeActConfig())
        tool = strat._build_return_result_tool(int, "test_method")
        result = tool.callable(result=42)
        assert result == 42

    def test_return_result_stub_none(self):
        """return_result stub with None return type (line 1601)."""
        strat = CodeActStrategy(config=CodeActConfig())
        tool = strat._build_return_result_tool(None, "test_method")
        result = tool.callable()
        assert result is None


# ===========================================================================
# CODEACT: Lines 1851-1852 — ImportError for PredictStrategy in _execute_code
# ===========================================================================


class TestCodeActPredictStrategyImportError:
    """Cover lines 1851-1852: ImportError when importing PredictStrategy."""

    @pytest.mark.asyncio
    async def test_predict_strategy_import_error(self):
        """When PredictStrategy cannot be imported in _execute_code,
        the execution should still succeed (lines 1851-1852)."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def do_task(self) -> int:
                """Do a task."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call("x = 2 + 2\nprint(x)")]),
                _resp("", tool_calls=[_return_result(result=4)]),
            ]
        )
        agent = TestAgent(llm=fake_llm)

        # Patch the import to raise ImportError
        original_import = (
            __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
        )

        def mock_import(name, *args, **kwargs):
            if name == "nooa.strategies.predict":
                raise ImportError("test: predict not available")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = await agent.do_task()
        assert result == 4


# ===========================================================================
# CODEACT: Lines 1994-1995 — Exception in _import_dynamic_classes
# ===========================================================================


class TestCodeActImportDynamicClassesException:
    """Cover lines 1994-1995: Exception in _import_dynamic_classes → pass."""

    def test_import_dynamic_classes_exception_is_swallowed(self):
        """When _import_dynamic_classes encounters an exception iterating
        attributes, it should be silently swallowed (lines 1994-1995)."""
        strat = CodeActStrategy(config=CodeActConfig())
        fake_module = types.ModuleType("test_module")
        context: dict[str, Any] = {}

        # Create an agent whose attribute access raises an exception
        class BrokenAgent:
            @property
            def bad_attr(self):
                raise RuntimeError("broken attribute")

        # Patch _iter_agent_attrs to raise
        with patch(
            "nooa.strategies.codeact._iter_agent_attrs",
            side_effect=RuntimeError("iteration error"),
        ):
            # Should not raise — exception is silently caught
            strat._import_dynamic_classes(BrokenAgent(), fake_module, context)

        # Context should remain unchanged
        assert context == {}


# ===========================================================================
# PREDICT: Lines 216-233 — Malformed LLM response fallback
# ===========================================================================


class TestPredictMalformedResponseFallback:
    """Cover lines 216-233: fallback extraction when raw_response_content is None."""

    @pytest.mark.asyncio
    async def test_llm_response_none_fallback(self):
        """When raw_response_content is None and llm_response is also None,
        the fallback produces '(llm_response was None)' (lines 217-219)."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(PredictStrategy(config=PredictConfig(max_retries=2)))
            async def compute(self) -> int:
                """Compute something."""
                ...

        # Force the path where both raw_response_content and llm_response are None
        # by making _extract_raw_from_llm_response return None and _call_llm_raw
        # raise immediately (so llm_response is never assigned)
        call_count = [0]

        async def mock_call_llm_raw(self_strat, runtime, response_model):
            call_count[0] += 1
            if call_count[0] == 1:
                # Return a response with weird content type that won't parse
                # but also patch _extract_raw to return None
                resp = LLMResponse(
                    raw_response=None,
                    content="not json at all!!!",
                    tool_calls=[],
                    finish_reason="stop",
                    assistant_message={"role": "assistant", "content": "not json"},
                )
                return resp, "evt_1"
            # Second call: correct response
            resp = _llm_resp(json.dumps({"value": 42}))
            return resp, "evt_2"

        fake_llm = FakeLLMClient(scripted_responses=[])
        agent = TestAgent(llm=fake_llm)

        extract_count = [0]

        def mock_extract(self_strat, llm_response):
            extract_count[0] += 1
            if extract_count[0] <= 2:
                return None  # Force None path in exception handler
            return str(llm_response.content) if llm_response.content else None

        with (
            patch.object(PredictStrategy, "_call_llm_raw", mock_call_llm_raw),
            patch.object(PredictStrategy, "_extract_raw_from_llm_response", mock_extract),
        ):
            result = await agent.compute()
        assert result == 42

    @pytest.mark.asyncio
    async def test_llm_response_with_unknown_fields_fallback(self):
        """When raw_response_content is None and llm_response has no standard
        fields, extraction produces 'extraction failed, type=...' (lines 232-235)."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(PredictStrategy(config=PredictConfig(max_retries=2)))
            async def compute(self) -> int:
                """Compute something."""
                ...

        call_count = [0]

        # Custom class with no standard fields — no content, reasoning, raw, text, message
        class WeirdResponse:
            def __init__(self):
                self.foo = "bar"

        async def mock_call_llm_raw(self_strat, runtime, response_model):
            call_count[0] += 1
            if call_count[0] == 1:
                return WeirdResponse(), "evt_1"
            resp = _llm_resp(json.dumps({"value": 42}))
            return resp, "evt_2"

        fake_llm = FakeLLMClient(scripted_responses=[])
        agent = TestAgent(llm=fake_llm)

        extract_count = [0]

        def mock_extract(self_strat, llm_response):
            extract_count[0] += 1
            if extract_count[0] <= 2:
                return None
            return str(getattr(llm_response, "content", "")) or None

        # Also need to patch _parse_llm_response for the WeirdResponse so it raises
        original_parse = PredictStrategy._parse_llm_response
        parse_count = [0]

        def mock_parse(self_strat, llm_response, method_name):
            parse_count[0] += 1
            if parse_count[0] == 1:
                raise ValueError("Cannot parse weird response")
            return original_parse(self_strat, llm_response, method_name)

        with (
            patch.object(PredictStrategy, "_call_llm_raw", mock_call_llm_raw),
            patch.object(PredictStrategy, "_extract_raw_from_llm_response", mock_extract),
            patch.object(PredictStrategy, "_parse_llm_response", mock_parse),
        ):
            result = await agent.compute()
        assert result == 42

    @pytest.mark.asyncio
    async def test_llm_response_with_known_fields_fallback(self):
        """When raw_response_content is None and llm_response has 'content' field,
        extraction produces 'extraction failed, fields: content=...' (lines 222-231)."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(PredictStrategy(config=PredictConfig(max_retries=2)))
            async def compute(self) -> int:
                """Compute something."""
                ...

        call_count = [0]

        # Custom class WITH 'content' field — triggers lines 222-231
        class ResponseWithContent:
            def __init__(self):
                self.content = "some_content_value"
                self.reasoning = None

        async def mock_call_llm_raw(self_strat, runtime, response_model):
            call_count[0] += 1
            if call_count[0] == 1:
                return ResponseWithContent(), "evt_1"
            resp = _llm_resp(json.dumps({"value": 42}))
            return resp, "evt_2"

        fake_llm = FakeLLMClient(scripted_responses=[])
        agent = TestAgent(llm=fake_llm)

        extract_count = [0]

        def mock_extract(self_strat, llm_response):
            extract_count[0] += 1
            if extract_count[0] <= 2:
                return None  # Force None path
            return str(getattr(llm_response, "content", "")) or None

        # Also need to patch _parse_llm_response for the custom response
        original_parse = PredictStrategy._parse_llm_response
        parse_count = [0]

        def mock_parse(self_strat, llm_response, method_name):
            parse_count[0] += 1
            if parse_count[0] == 1:
                raise ValueError("Cannot parse custom response")
            return original_parse(self_strat, llm_response, method_name)

        with (
            patch.object(PredictStrategy, "_call_llm_raw", mock_call_llm_raw),
            patch.object(PredictStrategy, "_extract_raw_from_llm_response", mock_extract),
            patch.object(PredictStrategy, "_parse_llm_response", mock_parse),
        ):
            result = await agent.compute()
        assert result == 42


# ===========================================================================
# PREDICT: Line 282 — "Should never reach here" after retry loop
# ===========================================================================


class TestPredictLlmResponseNonePath:
    """Cover line 219: raw_response_content fallback when llm_response is None."""

    @pytest.mark.asyncio
    async def test_llm_response_is_none_fallback(self):
        """When _call_llm_raw raises a caught exception before llm_response is assigned,
        llm_response remains None and line 219 is hit."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(PredictStrategy(config=PredictConfig(max_retries=2)))
            async def compute(self) -> int:
                """Compute something."""
                ...

        call_count = [0]

        async def mock_call_llm_raw(self_strat, runtime, response_model):
            call_count[0] += 1
            if call_count[0] == 1:
                # Raise ValueError before llm_response is assigned
                raise ValueError("simulated LLM failure")
            resp = _llm_resp(json.dumps({"value": 42}))
            return resp, "evt_2"

        fake_llm = FakeLLMClient(scripted_responses=[])
        agent = TestAgent(llm=fake_llm)

        with patch.object(PredictStrategy, "_call_llm_raw", mock_call_llm_raw):
            result = await agent.compute()
        assert result == 42


class TestPredictShouldNeverReachHere:
    """Cover line 282: GenerationError raised after retry loop exhausts."""

    @pytest.mark.asyncio
    async def test_exhausted_retry_loop(self):
        """When max_retries is set to 0, the for-loop body never executes
        and the code falls through to line 282."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(PredictStrategy(config=PredictConfig(max_retries=0)))
            async def compute(self) -> int:
                """Compute something."""
                ...

        fake_llm = FakeLLMClient(scripted_responses=[])
        agent = TestAgent(llm=fake_llm)
        with pytest.raises(GenerationError, match="failed after 0 attempts"):
            await agent.compute()


# ===========================================================================
# PREDICT: Line 567 — "Shouldn't reach here with output_model"
# ===========================================================================


class TestPredictUnexpectedResponseType:
    """Cover line 566-570: GenerationError for unexpected response type."""

    def test_parse_llm_response_unexpected_type(self):
        """When _parse_llm_response receives content that is not BaseModel,
        dict, or str, it raises GenerationError (line 567)."""
        strat = PredictStrategy(config=PredictConfig())

        # Create a response with an unexpected content type (e.g., a list)
        mock_response = MagicMock()
        mock_response.content = 42  # int, not str/dict/BaseModel
        mock_response.reasoning = None

        with pytest.raises(GenerationError, match="Unexpected response type"):
            strat._parse_llm_response(mock_response, "test_method")

    def test_parse_llm_response_list_content(self):
        """When content is a list (not dict, str, BaseModel), it should raise."""
        strat = PredictStrategy(config=PredictConfig())

        mock_response = MagicMock()
        mock_response.content = [1, 2, 3]
        mock_response.reasoning = None

        with pytest.raises(GenerationError, match="Unexpected response type"):
            strat._parse_llm_response(mock_response, "test_method")


# ===========================================================================
# PURE_PYTHON: Lines 61-63 — httpx ImportError fallback
# ===========================================================================


class TestPurePythonHttpxImportError:
    """Cover lines 61-63: httpx ImportError fallback."""

    def test_httpx_timeout_exceptions_defined(self):
        """_HTTPX_TIMEOUT_EXCEPTIONS should be defined regardless of httpx availability."""
        from nooa.strategies.pure_python import _HTTPX_TIMEOUT_EXCEPTIONS

        assert isinstance(_HTTPX_TIMEOUT_EXCEPTIONS, tuple)
        assert len(_HTTPX_TIMEOUT_EXCEPTIONS) > 0

    def test_httpx_import_error_fallback(self):
        """When httpx is not available, fallback to (TimeoutError,) (lines 61-63)."""
        import importlib

        import nooa.strategies.pure_python as pp_mod

        # Temporarily make httpx unavailable
        original_httpx = sys.modules.get("httpx")
        sys.modules["httpx"] = None  # type: ignore[assignment]  # Causes ImportError on import

        try:
            importlib.reload(pp_mod)
            assert pp_mod._HTTPX_TIMEOUT_EXCEPTIONS == (TimeoutError,)
        finally:
            # Restore httpx
            if original_httpx is not None:
                sys.modules["httpx"] = original_httpx
            else:
                sys.modules.pop("httpx", None)
            importlib.reload(pp_mod)  # Restore normal state


# ===========================================================================
# PURE_PYTHON: Line 245 — generation_id is None check
# ===========================================================================


class TestPurePythonGenerationIdNone:
    """Cover line 244-248: generation_id is None → RuntimeError."""

    @pytest.mark.asyncio
    async def test_generation_id_none_raises(self):
        """When runtime.get_generation_id() returns None, a RuntimeError
        should be raised (line 244-248)."""
        from nooa.runtime.actor import ActorRuntime

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy())
            async def do_task(self) -> int:
                """Do a task."""
                ...

        fake_llm = FakeLLMClient(scripted_responses=[_resp("return 42")])
        agent = TestAgent(llm=fake_llm)

        with patch.object(ActorRuntime, "get_generation_id", return_value=None):
            with pytest.raises(RuntimeError, match="get_generation_id.*returned None"):
                await agent.do_task()


# ===========================================================================
# PURE_PYTHON: Lines 694-695 — ImportError for PredictStrategy/ReflexionStrategy
# ===========================================================================


class TestPurePythonStrategyImportError:
    """Cover lines 694-695: ImportError when importing PredictStrategy/ReflexionStrategy."""

    @pytest.mark.asyncio
    async def test_predict_reflexion_import_error(self):
        """When PredictStrategy/ReflexionStrategy cannot be imported in
        _execute_code, the execution should still work (lines 694-695)."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy())
            async def do_task(self) -> int:
                """Do a task."""
                ...

        fake_llm = FakeLLMClient(scripted_responses=[_resp("return 42")])
        agent = TestAgent(llm=fake_llm)

        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "nooa.strategies.predict":
                raise ImportError("test: predict not available")
            if name == "nooa.strategies.reflexion":
                raise ImportError("test: reflexion not available")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = await agent.do_task()
        assert result == 42


# ===========================================================================
# PURE_PYTHON: Lines 723-729 — helper method binding errors
# ===========================================================================


class TestPurePythonHelperMethodErrors:
    """Cover lines 711-729: helper method binding errors. removed the "rejected on target-name collision" path — the test
    for that behavior was deleted.
    """

    @pytest.mark.asyncio
    async def test_helper_method_binding_error(self):
        """When helper method binding produces errors, an error event is
        added and execution returns empty result (lines 722-729)."""
        from nooa.strategies.generated_code import (
            HelperApplyResult,
            HelperFunctionManager,
        )

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy(max_iterations=3))
            async def do_task(self) -> int:
                """Do a task."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("def helper(self):\n    return 42\nreturn helper(self)"),
                _resp("return 99"),
            ]
        )
        agent = TestAgent(llm=fake_llm)

        real_apply_method = HelperFunctionManager.apply
        call_count = [0]

        def counting_apply(self_hm, code, ag, session_locals, *, namespace):
            call_count[0] += 1
            if call_count[0] == 1:
                return HelperApplyResult(
                    installed=[],
                    errors=["Error defining method `helper`: SyntaxError: invalid syntax"],
                )
            return real_apply_method(
                self_hm,
                code,
                ag,
                session_locals,
                namespace=namespace,
            )

        with patch.object(HelperFunctionManager, "apply", counting_apply):
            result = await agent.do_task()
        assert result == 99
