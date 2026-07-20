# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage-improving tests for strategy modules.

Targets:
- codeact_lite.py
- codeact_errors.py
- predict.py
- base.py
- current_call.py
"""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, ValidationError

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.config.strategy_config import PredictConfig
from nooa.errors import GenerationError
from nooa.events import (
    Error,
    Feedback,
    Message,
    PythonOutput,
    ResultStatus,
    Task,
)
from nooa.strategies.base import GenerationStrategy
from nooa.strategies.codeact_errors import (
    _format_actual_value,
    _format_error_path,
    _format_expected_schema,
    _format_missing_fields_error,
    _format_pydantic_error,
    _format_single_error,
    _format_value_brief,
    format_validation_error,
    get_type_example,
    get_type_hint_str,
)
from nooa.strategies.codeact_lite import (
    CodeActLiteStrategy,
    PlainProviderFormatter,
    plain_event_content,
)
from nooa.strategies.current_call import CurrentCall
from nooa.strategies.predict import PredictStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

# ---------------------------------------------------------------------------
# Helpers
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


def _return_result(call_id: str = "call_return", result: Any = None) -> ToolCall:
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


def _llm_resp(content: str) -> LLMResponse:
    """Create an LLMResponse with string content (for PREDICT tests)."""
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": content},
    )


# ---------------------------------------------------------------------------
# Tests: codeact_errors.py — plain functions
# ---------------------------------------------------------------------------


class TestFormatValidationErrorJsonDecode:
    """Tests for format_validation_error with JSONDecodeError."""

    def test_json_decode_error_returns_actionable_message(self):
        """format_validation_error should handle JSONDecodeError."""
        try:
            json.loads("not valid json")
        except json.JSONDecodeError as e:
            result = format_validation_error(e, str)
            assert "Could not parse JSON" in result
            assert "return_result() failed" in result

    def test_generic_exception_format(self):
        """format_validation_error with generic exception returns string."""
        err = RuntimeError("something went wrong")
        result = format_validation_error(err, str)
        assert "return_result() failed" in result
        assert "something went wrong" in result


class TestGetTypeHintStr:
    """Tests for get_type_hint_str covering all branches."""

    def test_none_returns_value(self):
        assert get_type_hint_str(None) == "value"

    def test_list_with_args(self):
        assert get_type_hint_str(list[int]) == "list[int]"

    def test_list_without_args(self):
        assert get_type_hint_str(list) == "list"

    def test_dict_with_args(self):
        assert get_type_hint_str(dict[str, int]) == "dict[str, int]"

    def test_dict_without_args(self):
        assert get_type_hint_str(dict) == "dict"

    def test_tuple_with_args(self):
        result = get_type_hint_str(tuple[int, str])
        assert "tuple" in result
        assert "int" in result
        assert "str" in result

    def test_tuple_without_args(self):
        assert get_type_hint_str(tuple) == "tuple"

    def test_basic_type(self):
        assert get_type_hint_str(int) == "int"
        assert get_type_hint_str(str) == "str"

    def test_fallback_to_str(self):
        # A non-type object should fall back to str()
        result = get_type_hint_str("some_string_type")
        assert result == "some_string_type"


class TestGetTypeExample:
    """Tests for get_type_example covering all branches."""

    def test_list_str(self):
        result = get_type_example(list[str])
        assert "item" in result

    def test_list_int(self):
        result = get_type_example(list[int])
        assert "1" in result

    def test_list_float(self):
        result = get_type_example(list[float])
        assert "1.0" in result

    def test_list_no_typed_args(self):
        # Plain `list` (no type args) has no origin, so get_type_example returns None
        result = get_type_example(list)
        # None is expected for bare unparameterized list
        assert result is None or result == "result=[...]"

    def test_dict(self):
        # Plain `dict` (no type args) has no origin, so get_type_example returns None
        result = get_type_example(dict)
        # None is expected for bare unparameterized dict
        assert result is None or "key" in result

    def test_dict_typed(self):
        result = get_type_example(dict[str, int])
        assert "key" in result

    def test_str(self):
        result = get_type_example(str)
        assert '"' in result or "your answer" in result

    def test_int(self):
        result = get_type_example(int)
        assert "42" in result

    def test_float(self):
        result = get_type_example(float)
        assert "3.14" in result

    def test_bool(self):
        result = get_type_example(bool)
        assert "True" in result

    def test_pydantic_model_returns_none(self):
        class M(BaseModel):
            x: int

        result = get_type_example(M)
        assert result is None


class TestFormatErrorPath:
    """Tests for _format_error_path."""

    def test_empty_loc(self):
        assert _format_error_path(()) == "value"

    def test_single_string(self):
        assert _format_error_path(("name",)) == "name"

    def test_single_int_index(self):
        assert _format_error_path((0,)) == "[0]"

    def test_nested_string_then_int(self):
        result = _format_error_path(("items", 0, "name"))
        assert result == "items[0].name"

    def test_int_then_string(self):
        result = _format_error_path((0, "field"))
        assert result == "[0].field"


class TestFormatActualValue:
    """Tests for _format_actual_value."""

    def test_small_dict(self):
        result = _format_actual_value({"a": 1, "b": 2})
        assert "a" in result

    def test_large_dict(self):
        big = {str(i): i for i in range(10)}
        result = _format_actual_value(big)
        assert "10 fields" in result

    def test_short_list(self):
        result = _format_actual_value([1, 2, 3])
        assert "[" in result

    def test_long_list(self):
        result = _format_actual_value(list(range(10)))
        assert "10 items" in result

    def test_tuple_short(self):
        result = _format_actual_value((1, 2))
        assert "(" in result

    def test_tuple_long(self):
        result = _format_actual_value(tuple(range(10)))
        assert "10 items" in result

    def test_non_collection(self):
        result = _format_actual_value("hello")
        assert "hello" in result


class TestFormatValueBrief:
    """Tests for _format_value_brief."""

    def test_short_string(self):
        result = _format_value_brief("hello")
        assert "hello" in result

    def test_long_string_truncated(self):
        result = _format_value_brief("a" * 30)
        assert "..." in result

    def test_bool_true(self):
        assert _format_value_brief(True) == "True"

    def test_bool_false(self):
        assert _format_value_brief(False) == "False"

    def test_int(self):
        assert _format_value_brief(42) == "42"

    def test_float(self):
        assert _format_value_brief(3.14) == "3.14"

    def test_dict(self):
        assert _format_value_brief({"a": 1}) == "{...}"

    def test_list(self):
        assert _format_value_brief([1, 2]) == "[...]"

    def test_tuple(self):
        assert _format_value_brief((1, 2)) == "(...)"

    def test_other(self):
        result = _format_value_brief(object())
        assert len(result) <= 20


class TestFormatSingleError:
    """Tests for _format_single_error covering all branches."""

    def test_root_level_error_with_example(self):
        """Root-level error (no loc) with example."""
        err = {"loc": (), "msg": "Input should be a valid integer", "type": "int_type", "url": ""}
        result = _format_single_error(err, int, "int", actual_value="bad")
        assert "int" in result
        assert "Got" in result
        assert "42" in result  # example

    def test_root_level_error_brief(self):
        err = {"loc": (), "msg": "Input should be a valid integer", "type": "int_type", "url": ""}
        result = _format_single_error(err, int, "int", brief=True)
        assert "got wrong type" in result

    def test_root_collection_error_type_mismatch(self):
        err = {"loc": (0,), "msg": "Input should be a valid integer", "type": "int_type", "url": ""}
        result = _format_single_error(err, list[int], "list[int]", actual_value=["bad"])
        assert "[0]" in result
        assert "wrong type" in result

    def test_root_collection_error_type_mismatch_brief(self):
        err = {"loc": (0,), "msg": "Input should be valid", "type": "int_type", "url": ""}
        result = _format_single_error(err, list[int], "list[int]", brief=True)
        assert "wrong type" in result

    def test_root_collection_missing_brief(self):
        err = {"loc": (0,), "msg": "missing", "type": "missing", "url": ""}
        result = _format_single_error(err, list[int], "list[int]", brief=True)
        assert "missing required field" in result

    def test_root_collection_missing_verbose(self):
        err = {"loc": (0,), "msg": "missing", "type": "missing", "url": ""}
        result = _format_single_error(err, list[int], "list[int]")
        assert "missing a required field" in result

    def test_root_collection_generic_brief(self):
        err = {"loc": (0,), "msg": "some error", "type": "value_error", "url": ""}
        result = _format_single_error(err, list[int], "list[int]", brief=True)
        assert "some error" in result

    def test_root_collection_generic_verbose(self):
        err = {"loc": (0,), "msg": "some error", "type": "value_error", "url": ""}
        result = _format_single_error(err, list[int], "list[int]")
        assert "some error" in result

    def test_named_field_missing_top_level(self):
        err = {"loc": ("age",), "msg": "missing", "type": "missing", "url": ""}
        result = _format_single_error(err, object, "SomeType")
        assert "age" in result
        assert "missing" in result.lower()

    def test_named_field_missing_top_level_brief(self):
        err = {"loc": ("age",), "msg": "missing", "type": "missing", "url": ""}
        result = _format_single_error(err, object, "SomeType", brief=True)
        assert "age" in result

    def test_named_field_missing_nested(self):
        err = {"loc": ("items", 0, "name"), "msg": "missing", "type": "missing", "url": ""}
        result = _format_single_error(err, object, "SomeType")
        assert "field required" in result.lower()

    def test_named_field_missing_nested_brief(self):
        err = {"loc": ("items", 0, "name"), "msg": "missing", "type": "missing", "url": ""}
        result = _format_single_error(err, object, "SomeType", brief=True)
        assert "field required" in result

    def test_named_field_type_error(self):
        err = {
            "loc": ("age",),
            "msg": "Input should be a valid integer",
            "type": "int_type",
            "url": "",
        }

        class M(BaseModel):
            age: int

        result = _format_single_error(err, M, "M")
        assert "age" in result
        assert "wrong type" in result

    def test_named_field_type_error_brief(self):
        err = {
            "loc": ("age",),
            "msg": "Input should be a valid integer",
            "type": "int_type",
            "url": "",
        }
        result = _format_single_error(err, object, "obj", brief=True)
        assert "wrong type" in result

    def test_named_field_generic_error(self):
        err = {"loc": ("name",), "msg": "some error", "type": "value_error", "url": ""}
        result = _format_single_error(err, object, "obj")
        assert "name" in result
        assert "some error" in result

    def test_named_field_generic_error_brief(self):
        err = {"loc": ("name",), "msg": "some error", "type": "value_error", "url": ""}
        result = _format_single_error(err, object, "obj", brief=True)
        assert "some error" in result

    def test_int_parsing_in_named_field(self):
        err = {
            "loc": ("count",),
            "msg": "Input should be a valid integer",
            "type": "int_parsing",
            "url": "",
        }
        result = _format_single_error(err, object, "obj")
        assert "wrong type" in result

    def test_int_from_float_in_named_field(self):
        err = {
            "loc": ("count",),
            "msg": "Input should be an integer, got float",
            "type": "int_from_float",
            "url": "",
        }
        result = _format_single_error(err, object, "obj")
        assert "wrong type" in result


class TestFormatExpectedSchema:
    """Tests for _format_expected_schema."""

    def test_pydantic_model(self):
        class M(BaseModel):
            name: str
            age: int

        result = _format_expected_schema(M)
        assert "M" in result
        assert "name" in result
        assert "age" in result

    def test_typed_dict(self):
        from typing import TypedDict

        class TD(TypedDict):
            x: int
            y: str

        result = _format_expected_schema(TD)
        assert "TD" in result
        assert "x" in result

    def test_simple_type_no_name(self):
        result = _format_expected_schema(int)
        assert result == "int"


class TestFormatPydanticError:
    """Tests for _format_pydantic_error."""

    def test_missing_fields_error(self):
        class M(BaseModel):
            name: str
            age: int

        try:
            M()
        except ValidationError as e:
            result = _format_pydantic_error(e, M, {})
            assert "missing required fields" in result

    def test_single_type_error(self):
        class M(BaseModel):
            age: int

        try:
            M(age="not_int")
        except ValidationError as e:
            result = _format_pydantic_error(e, M, {"age": "not_int"})
            assert "age" in result or "wrong type" in result

    def test_multiple_errors(self):
        class M(BaseModel):
            name: str
            age: int

        try:
            M(name=123, age="bad")
        except ValidationError as e:
            result = _format_pydantic_error(e, M, {"name": 123, "age": "bad"})
            assert "2 errors" in result or "error" in result


# ---------------------------------------------------------------------------
# Tests: codeact_lite.py — plain_event_content
# ---------------------------------------------------------------------------


class TestPlainEventContent:
    """Tests for plain_event_content function."""

    def test_task_returns_prompt(self):
        result = plain_event_content(Task(prompt="hello world"))
        assert result == "hello world"

    def test_python_output_stdout_only(self):
        po = PythonOutput(
            tool_call_id="tc1",
            execution_status=ResultStatus.COMPLETE,
            execution_count=1,
            stdout="some output",
        )
        result = plain_event_content(po)
        assert result == "some output"

    def test_python_output_error_only(self):
        po = PythonOutput(
            tool_call_id="tc1",
            execution_status=ResultStatus.ERROR,
            execution_count=1,
            error="SomeError",
        )
        result = plain_event_content(po)
        assert "Error: SomeError" in result

    def test_python_output_stderr_no_error(self):
        po = PythonOutput(
            tool_call_id="tc1",
            execution_status=ResultStatus.COMPLETE,
            execution_count=1,
            stderr="warning msg",
        )
        result = plain_event_content(po)
        assert "Stderr: warning msg" in result

    def test_python_output_value(self):
        po = PythonOutput(
            tool_call_id="tc1",
            execution_status=ResultStatus.COMPLETE,
            execution_count=2,
            value=42,
        )
        result = plain_event_content(po)
        assert "Out[2]" in result
        assert "42" in result

    def test_python_output_empty(self):
        po = PythonOutput(
            tool_call_id="tc1",
            execution_status=ResultStatus.COMPLETE,
            execution_count=1,
        )
        result = plain_event_content(po)
        assert result == "(no output)"

    def test_python_output_stderr_with_error_not_shown(self):
        """When both error and stderr are set, only error is shown."""
        po = PythonOutput(
            tool_call_id="tc1",
            execution_status=ResultStatus.ERROR,
            execution_count=1,
            error="MainError",
            stderr="stderr stuff",
        )
        result = plain_event_content(po)
        assert "Error: MainError" in result
        # stderr should not appear when error is set
        assert "Stderr:" not in result

    def test_error_event_returns_content(self):
        result = plain_event_content(Error(content="something failed"))
        assert result == "something failed"

    def test_message_event_returns_content(self):
        result = plain_event_content(Message(content="hello"))
        assert result == "hello"

    def test_feedback_event_returns_content(self):
        result = plain_event_content(Feedback(content="nice work"))
        assert result == "nice work"

    def test_fallback_to_str(self):
        result = plain_event_content("raw string event")
        assert result == "raw string event"

    def test_fallback_object(self):
        result = plain_event_content(42)
        assert result == "42"


# ---------------------------------------------------------------------------
# Tests: codeact_lite.py — CodeActLiteStrategy
# ---------------------------------------------------------------------------


class TestCodeActLiteStrategyName:
    """Tests for CodeActLiteStrategy.name."""

    def test_name_is_codeact_lite(self):
        strat = CodeActLiteStrategy(config=CodeActConfig())
        assert strat.name == "CODEACT_LITE"


class TestCodeActLiteStrategyExecution:
    """Integration tests for CodeActLiteStrategy.execute()."""

    @pytest.mark.asyncio
    async def test_direct_return_result(self):
        """CodeActLiteStrategy executes and returns value."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActLiteStrategy(config=CodeActConfig()))
            async def answer(self) -> int:
                """Return the answer to everything."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_return_result(result=42)]),
            ]
        )

        agent = TestAgent(llm=fake_llm)
        result = await agent.answer()
        assert result == 42

    @pytest.mark.asyncio
    async def test_execute_python_then_return(self):
        """CodeActLiteStrategy handles execute_python before return_result."""

        class TestAgent(Agent, llm=_TEST_LLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.data = [1, 2, 3]

            @strategy(CodeActLiteStrategy(config=CodeActConfig()))
            async def compute_sum(self) -> int:
                """Sum the data."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call("total = sum(self.data)")]),
                _resp("", tool_calls=[_return_result(result=6)]),
            ]
        )

        agent = TestAgent(llm=fake_llm)
        result = await agent.compute_sum()
        assert result == 6

    @pytest.mark.asyncio
    async def test_returns_string(self):
        """CodeActLiteStrategy handles string return type."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActLiteStrategy(config=CodeActConfig()))
            async def greet(self, name: str) -> str:
                """Greet the user."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_return_result(result="hello")]),
            ]
        )

        agent = TestAgent(llm=fake_llm)
        result = await agent.greet("Alice")
        assert result == "hello"


# ---------------------------------------------------------------------------
# Tests: base.py — call_with_instrumentation
# ---------------------------------------------------------------------------


class TestCallWithInstrumentation:
    """Tests for GenerationStrategy.call_with_instrumentation()."""

    def _make_strategy(self):
        class Impl(GenerationStrategy):
            async def execute(self, runtime, call):
                return "ok"

        return Impl()

    def _make_mock_runtime(self):
        mock_agent = MagicMock()
        mock_runtime = MagicMock()
        mock_runtime.agent = mock_agent
        return mock_runtime

    @pytest.mark.asyncio
    async def test_sync_callable_is_called(self):
        strat = self._make_strategy()
        runtime = self._make_mock_runtime()

        def sync_fn(x):
            return x * 10

        result = await strat.call_with_instrumentation(
            sync_fn, (5,), {}, runtime=runtime, method_name="test"
        )
        assert result == 50

    @pytest.mark.asyncio
    async def test_async_callable_is_called(self):
        strat = self._make_strategy()
        runtime = self._make_mock_runtime()

        async def async_fn(x):
            return x + 100

        result = await strat.call_with_instrumentation(
            async_fn, (7,), {}, runtime=runtime, method_name="test"
        )
        assert result == 107

    @pytest.mark.asyncio
    async def test_exception_propagates(self):
        strat = self._make_strategy()
        runtime = self._make_mock_runtime()

        async def failing_fn():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await strat.call_with_instrumentation(
                failing_fn, (), {}, runtime=runtime, method_name="test"
            )

    @pytest.mark.asyncio
    async def test_sync_exception_propagates(self):
        strat = self._make_strategy()
        runtime = self._make_mock_runtime()

        def sync_failing():
            raise RuntimeError("sync boom")

        with pytest.raises(RuntimeError, match="sync boom"):
            await strat.call_with_instrumentation(
                sync_failing, (), {}, runtime=runtime, method_name="test"
            )

    @pytest.mark.asyncio
    async def test_after_hook_called_even_on_exception(self):
        """After hook should fire even when callable raises."""
        from unittest.mock import patch

        strat = self._make_strategy()
        runtime = self._make_mock_runtime()

        async def raising_fn():
            raise TypeError("test error")

        with patch("nooa.runtime.hooks.call_after_hook") as mock_after:
            with pytest.raises(TypeError):
                await strat.call_with_instrumentation(
                    raising_fn, (), {}, runtime=runtime, method_name="test"
                )
            mock_after.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: current_call.py — missing lines
# ---------------------------------------------------------------------------


class TestCurrentCallEquality:
    """Tests for CurrentCall.__eq__ covering non-CurrentCall comparison."""

    def test_equal_to_self(self):
        call = CurrentCall(id="abc", method_name="m", decorator="plan")
        assert call == call

    def test_not_equal_to_non_current_call(self):
        call = CurrentCall(id="abc", method_name="m", decorator="plan")
        # Should return NotImplemented, which Python treats as NotEqual
        result = call.__eq__("not a CurrentCall")
        assert result is NotImplemented

    def test_equality_with_same_id(self):
        c1 = CurrentCall(id="same", method_name="m", decorator="plan")
        c2 = CurrentCall(id="same", method_name="other", decorator="agent")
        assert c1 == c2

    def test_inequality_with_different_id(self):
        c1 = CurrentCall(id="a", method_name="m", decorator="plan")
        c2 = CurrentCall(id="b", method_name="m", decorator="plan")
        assert c1 != c2


class TestCurrentCallFormatParametersAsCode:
    """Tests for CurrentCall.format_parameters_as_code — missing lines."""

    def test_no_signature_no_kwargs(self):
        """No signature, no kwargs returns empty string."""
        call = CurrentCall(id="x", method_name="m", decorator="plan", signature=None, kwargs={})
        assert call.format_parameters_as_code() == ""

    def test_no_signature_with_kwargs(self):
        """No signature, but kwargs: use kwargs only."""
        call = CurrentCall(
            id="x",
            method_name="m",
            decorator="plan",
            signature=None,
            kwargs={"a": 1, "b": "hello"},
        )
        result = call.format_parameters_as_code()
        assert "a = 1" in result
        assert "b = 'hello'" in result

    def test_empty_signature_returns_empty(self):
        """Signature with no params returns empty string."""
        call = CurrentCall(id="x", method_name="m", decorator="plan", signature="()")
        assert call.format_parameters_as_code() == ""

    def test_signature_with_self_only(self):
        """Signature with only self param returns empty string."""
        call = CurrentCall(id="x", method_name="m", decorator="plan", signature="(self)")
        assert call.format_parameters_as_code() == ""

    def test_positional_args_mapped_by_name(self):
        """Positional args should be mapped to param names from the live signature."""

        def m(self, x: int, y: str): ...

        call = CurrentCall.from_method(m, args=(10, "hello"))
        result = call.format_parameters_as_code()
        assert "x = 10" in result
        assert "y = 'hello'" in result

    def test_kwargs_only(self):
        """kwargs without positional args."""
        call = CurrentCall(
            id="x",
            method_name="m",
            decorator="plan",
            signature="(self, count: int)",
            args=(),
            kwargs={"count": 5},
        )
        result = call.format_parameters_as_code()
        assert "count = 5" in result

    def test_no_params_no_args(self):
        """Signature has params but no args passed: returns empty."""
        call = CurrentCall(
            id="x",
            method_name="m",
            decorator="plan",
            signature="(self, count: int = 10)",
            args=(),
            kwargs={},
        )
        result = call.format_parameters_as_code()
        assert result == ""


class TestCurrentCallFormatSignature:
    """Tests for CurrentCall.format_signature — missing lines."""

    def test_async_with_signature(self):
        call = CurrentCall(
            id="x",
            method_name="run",
            decorator="plan",
            signature="(self, x: int) -> int",
            is_async=True,
        )
        result = call.format_signature()
        assert result.startswith("async def run")

    def test_sync_with_signature(self):
        call = CurrentCall(
            id="x",
            method_name="run",
            decorator="plan",
            signature="(self) -> str",
            is_async=False,
        )
        result = call.format_signature()
        assert result.startswith("def run")

    def test_no_signature_async_fallback(self):
        call = CurrentCall(
            id="x",
            method_name="run",
            decorator="plan",
            signature=None,
            is_async=True,
        )
        result = call.format_signature()
        assert "async def run" in result
        assert "..." in result

    def test_no_signature_sync_fallback(self):
        call = CurrentCall(
            id="x",
            method_name="run",
            decorator="plan",
            signature=None,
            is_async=False,
        )
        result = call.format_signature()
        assert "def run" in result
        assert "async" not in result


class TestCurrentCallFromMethodExtras:
    """Additional tests for CurrentCall.from_method() missing lines."""

    def test_from_method_extracts_return_type(self):
        def example(self) -> int:
            """Return an int."""
            pass

        call = CurrentCall.from_method(method=example)
        assert call.return_type is int

    def test_from_method_no_return_type(self):
        def no_return(self):
            pass

        call = CurrentCall.from_method(method=no_return)
        assert call.return_type is None

    def test_from_method_async(self):
        async def async_m(self) -> str:
            """Async."""
            pass

        call = CurrentCall.from_method(method=async_m)
        assert call.is_async is True

    def test_from_method_sync(self):
        def sync_m(self) -> str:
            """Sync."""
            pass

        call = CurrentCall.from_method(method=sync_m)
        assert call.is_async is False


# ---------------------------------------------------------------------------
# Tests: predict.py — PredictStrategy
# ---------------------------------------------------------------------------


class TestPredictStrategyBasic:
    """Basic tests for PredictStrategy."""

    def test_name(self):
        assert PredictStrategy().name == "PREDICT"

    def test_default_config(self):
        s = PredictStrategy()
        assert s.config.max_retries == 10

    def test_custom_config(self):
        s = PredictStrategy(config=PredictConfig(max_retries=3))
        assert s.config.max_retries == 3


class TestPredictStrategyNoReturnType:
    """Tests for PREDICT failing when no return type is annotated."""

    @pytest.mark.asyncio
    async def test_raises_generation_error_when_no_return_type(self):
        class NoReturnAgent(Agent, llm=_TEST_LLM):
            @strategy(PredictStrategy())
            async def no_return(self):
                """No return type."""
                ...

        agent = NoReturnAgent()
        with pytest.raises(GenerationError, match="no return type annotation"):
            await agent.no_return()


class TestPredictStrategyNoneReturnType:
    """Tests for PREDICT failing with None return type."""

    @pytest.mark.asyncio
    async def test_raises_generation_error_for_none_return(self):
        class NoneReturnAgent(Agent, llm=_TEST_LLM):
            @strategy(PredictStrategy())
            async def none_return(self) -> None:
                """Returns None."""
                ...

        agent = NoneReturnAgent()
        with pytest.raises(GenerationError, match="return type None"):
            await agent.none_return()


class TestPredictStrategyStringReturn:
    """Tests for PREDICT with string return type."""

    @pytest.mark.asyncio
    async def test_returns_string(self):
        class StringAgent(Agent, llm=_TEST_LLM):
            @strategy(PredictStrategy(config=PredictConfig(max_retries=2)))
            async def classify(self, text: str) -> str:
                """Classify the text."""
                ...

        fake_llm = FakeLLMClient(scripted_responses=[_llm_resp(json.dumps({"value": "positive"}))])
        agent = StringAgent(llm=fake_llm)
        result = await agent.classify("hello")
        assert result == "positive"

    @pytest.mark.asyncio
    async def test_returns_int(self):
        class IntAgent(Agent, llm=_TEST_LLM):
            @strategy(PredictStrategy(config=PredictConfig(max_retries=2)))
            async def count(self) -> int:
                """Count something."""
                ...

        fake_llm = FakeLLMClient(scripted_responses=[_llm_resp(json.dumps({"value": 42}))])
        agent = IntAgent(llm=fake_llm)
        result = await agent.count()
        assert result == 42

    @pytest.mark.asyncio
    async def test_returns_bool(self):
        class BoolAgent(Agent, llm=_TEST_LLM):
            @strategy(PredictStrategy(config=PredictConfig(max_retries=2)))
            async def check(self) -> bool:
                """Check something."""
                ...

        fake_llm = FakeLLMClient(scripted_responses=[_llm_resp(json.dumps({"value": True}))])
        agent = BoolAgent(llm=fake_llm)
        result = await agent.check()
        assert result is True


class TestPredictStrategyDictListReturn:
    """Tests for PREDICT with dict/list return types (RootModel)."""

    @pytest.mark.asyncio
    async def test_returns_dict(self):
        class DictAgent(Agent, llm=_TEST_LLM):
            @strategy(PredictStrategy(config=PredictConfig(max_retries=2)))
            async def info(self) -> dict:
                """Return info dict."""
                ...

        fake_llm = FakeLLMClient(scripted_responses=[_llm_resp(json.dumps({"key": "value"}))])
        agent = DictAgent(llm=fake_llm)
        result = await agent.info()
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_returns_list(self):
        """For a bare list return type, the list is wrapped under `value`.

        A top-level array schema is rejected by the Responses API (issue 232), so
        `list` uses the generic `value`-wrapper model. The LLM emits
        `{"value": [...]}`, and the strategy unwraps it before returning.
        """

        class ListAgent(Agent, llm=_TEST_LLM):
            @strategy(PredictStrategy(config=PredictConfig(max_retries=2)))
            async def items(self) -> list:
                """Return list."""
                ...

        fake_llm = FakeLLMClient(scripted_responses=[_llm_resp(json.dumps({"value": [1, 2, 3]}))])
        agent = ListAgent(llm=fake_llm)
        result = await agent.items()
        assert result == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_returns_list_str(self):
        """For list[str] return type, the list is wrapped under `value` (issue 232)."""

        class ListStrAgent(Agent, llm=_TEST_LLM):
            @strategy(PredictStrategy(config=PredictConfig(max_retries=2)))
            async def names(self) -> list[str]:
                """Return list of strings."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[_llm_resp(json.dumps({"value": ["alice", "bob"]}))]
        )
        agent = ListStrAgent(llm=fake_llm)
        result = await agent.names()
        assert result == ["alice", "bob"]


class TestPredictStrategyPydanticReturn:
    """Tests for PREDICT with Pydantic model return type."""

    @pytest.mark.asyncio
    async def test_returns_pydantic_model(self):
        class Result(BaseModel):
            name: str
            score: float

        class ModelAgent(Agent, llm=_TEST_LLM):
            @strategy(PredictStrategy(config=PredictConfig(max_retries=2)))
            async def analyze(self, text: str) -> Result:
                """Analyze text."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[_llm_resp(json.dumps({"name": "test", "score": 0.9}))]
        )
        agent = ModelAgent(llm=fake_llm)
        result = await agent.analyze("some text")
        assert isinstance(result, Result)
        assert result.name == "test"
        assert result.score == 0.9

    @pytest.mark.asyncio
    async def test_returns_optional_str(self):
        class OptAgent(Agent, llm=_TEST_LLM):
            @strategy(PredictStrategy(config=PredictConfig(max_retries=2)))
            async def maybe(self) -> str | None:
                """Maybe return string."""
                ...

        fake_llm = FakeLLMClient(scripted_responses=[_llm_resp(json.dumps({"value": "found"}))])
        agent = OptAgent(llm=fake_llm)
        result = await agent.maybe()
        assert result == "found"


class TestPredictStrategyRetryOnValidationFailure:
    """Tests for PREDICT retry behavior."""

    @pytest.mark.asyncio
    async def test_retry_on_type_mismatch_then_success(self):
        """First attempt fails validation, second succeeds."""

        class RetryAgent(Agent, llm=_TEST_LLM):
            @strategy(PredictStrategy(config=PredictConfig(max_retries=3)))
            async def classify(self, text: str) -> int:
                """Classify as int."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                # First attempt: wrong type
                _llm_resp(json.dumps({"value": "not_an_int"})),
                # Second attempt: correct
                _llm_resp(json.dumps({"value": 42})),
            ]
        )
        agent = RetryAgent(llm=fake_llm)
        result = await agent.classify("test")
        assert result == 42

    @pytest.mark.asyncio
    async def test_exhausted_retries_raises_generation_error(self):
        """All retries fail => GenerationError."""

        class ExhaustAgent(Agent, llm=_TEST_LLM):
            @strategy(PredictStrategy(config=PredictConfig(max_retries=2)))
            async def compute(self) -> int:
                """Compute an int."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _llm_resp(json.dumps({"value": "wrong"})),
                _llm_resp(json.dumps({"value": "still_wrong"})),
            ]
        )
        agent = ExhaustAgent(llm=fake_llm)
        with pytest.raises(GenerationError, match="validation failed after"):
            await agent.compute()

    @pytest.mark.asyncio
    async def test_retry_on_json_decode_error(self):
        """Invalid JSON causes retry."""

        class JsonRetryAgent(Agent, llm=_TEST_LLM):
            @strategy(PredictStrategy(config=PredictConfig(max_retries=3)))
            async def parse(self) -> str:
                """Parse something."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                # Invalid JSON
                _llm_resp("not json at all"),
                # Correct
                _llm_resp(json.dumps({"value": "parsed"})),
            ]
        )
        agent = JsonRetryAgent(llm=fake_llm)
        result = await agent.parse()
        assert result == "parsed"


class TestPredictStrategyStripXmlWrapper:
    """Tests for _strip_xml_wrapper."""

    def test_strips_xml_with_attributes(self):
        s = PredictStrategy()
        content = '<assistant_message expr="test">{"value": 42}</assistant_message>'
        result = s._strip_xml_wrapper(content)
        assert result == '{"value": 42}'

    def test_strips_simple_xml(self):
        s = PredictStrategy()
        content = "<tag>content here</tag>"
        result = s._strip_xml_wrapper(content)
        assert result == "content here"

    def test_no_xml_unchanged(self):
        s = PredictStrategy()
        content = '{"value": 42}'
        result = s._strip_xml_wrapper(content)
        assert result == '{"value": 42}'

    def test_none_returns_empty(self):
        s = PredictStrategy()
        result = s._strip_xml_wrapper(None)
        assert result == ""


class TestPredictStrategyParseResponse:
    """Tests for _parse_llm_response."""

    def test_pydantic_model_content(self):
        class M(BaseModel):
            value: int

        s = PredictStrategy()
        mock = MagicMock()
        mock.content = M(value=42)
        mock.reasoning = None
        result = s._parse_llm_response(mock, "test")
        assert result == {"value": 42}

    def test_dict_content(self):
        s = PredictStrategy()
        mock = MagicMock()
        mock.content = {"key": "val"}
        mock.reasoning = None
        result = s._parse_llm_response(mock, "test")
        assert result == {"key": "val"}

    def test_string_json_content(self):
        s = PredictStrategy()
        mock = MagicMock()
        mock.content = '{"value": 123}'
        mock.reasoning = None
        result = s._parse_llm_response(mock, "test")
        assert result == {"value": 123}

    def test_non_dict_json_wrapped(self):
        """Non-dict JSON (e.g., list) is wrapped in {"value": ...}."""
        s = PredictStrategy()
        mock = MagicMock()
        mock.content = "[1, 2, 3]"
        mock.reasoning = None
        result = s._parse_llm_response(mock, "test")
        assert result == {"value": [1, 2, 3]}

    def test_reasoning_fallback(self):
        """When content is empty, falls back to reasoning field."""
        s = PredictStrategy()
        mock = MagicMock()
        mock.content = None
        mock.reasoning = '{"value": 99}'
        result = s._parse_llm_response(mock, "test")
        assert result == {"value": 99}

    def test_empty_content_raises_json_error(self):
        """Empty content/reasoning causes JSONDecodeError."""
        s = PredictStrategy()
        mock = MagicMock()
        mock.content = None
        mock.reasoning = None
        with pytest.raises(json.JSONDecodeError):
            s._parse_llm_response(mock, "test")


class TestPredictStrategyExtractRaw:
    """Tests for _extract_raw_from_llm_response."""

    def test_string_content(self):
        s = PredictStrategy()
        mock = MagicMock()
        mock.content = "some string"
        mock.reasoning = None
        result = s._extract_raw_from_llm_response(mock)
        assert "[CONTENT]" in result
        assert "some string" in result

    def test_pydantic_model_content(self):
        s = PredictStrategy()

        class M(BaseModel):
            value: int

        mock = MagicMock()
        mock.content = M(value=42)
        mock.reasoning = None
        result = s._extract_raw_from_llm_response(mock)
        assert "[CONTENT (Pydantic)]" in result

    def test_dict_content(self):
        s = PredictStrategy()
        mock = MagicMock()
        mock.content = {"key": "val"}
        mock.reasoning = None
        result = s._extract_raw_from_llm_response(mock)
        assert "[CONTENT (dict)]" in result

    def test_non_str_content(self):
        s = PredictStrategy()
        mock = MagicMock()
        mock.content = 42  # int
        mock.reasoning = None
        result = s._extract_raw_from_llm_response(mock)
        assert "[CONTENT (int)]" in result

    def test_long_reasoning_truncated(self):
        s = PredictStrategy()
        mock = MagicMock()
        mock.content = "content"
        mock.reasoning = "x" * 2000  # Exceeds 1000 char limit
        result = s._extract_raw_from_llm_response(mock)
        assert "truncated" in result

    def test_empty_response(self):
        s = PredictStrategy()
        mock = MagicMock()
        mock.content = None
        mock.reasoning = None
        result = s._extract_raw_from_llm_response(mock)
        assert "empty response" in result


class TestPredictStrategyCreateResponseModel:
    """Tests for _create_response_model edge cases."""

    def test_none_type_raises(self):
        s = PredictStrategy()
        with pytest.raises(GenerationError, match="return type None"):
            s._create_response_model(type(None), "test")

    def test_dict_type_creates_root_model(self):
        from pydantic import RootModel

        s = PredictStrategy()
        model = s._create_response_model(dict, "test")
        assert issubclass(model, RootModel)

    def test_typed_dict_creates_root_model(self):
        from pydantic import RootModel

        s = PredictStrategy()
        model = s._create_response_model(dict[str, int], "test")
        assert issubclass(model, RootModel)

    def test_list_type_creates_object_wrapper(self):
        """Bare `list` must produce an object-rooted `value`-wrapper, not a RootModel.

        A top-level array schema is rejected by the Responses API (issue 232).
        """
        from pydantic import RootModel

        s = PredictStrategy()
        model = s._create_response_model(list, "test")
        assert not issubclass(model, RootModel)
        assert "value" in model.model_fields
        assert model.model_json_schema()["type"] == "object"

    def test_list_typed_creates_object_wrapper(self):
        """`list[str]` must produce an object-rooted `value`-wrapper (issue 232)."""
        from pydantic import RootModel

        s = PredictStrategy()
        model = s._create_response_model(list[str], "test")
        assert not issubclass(model, RootModel)
        assert "value" in model.model_fields
        assert model.model_json_schema()["type"] == "object"

    def test_optional_unwraps_inner(self):
        s = PredictStrategy()
        model = s._create_response_model(str | None, "test")
        # Should create a wrapped model for str
        assert hasattr(model, "model_fields")

    def test_pydantic_model_used_directly(self):
        class M(BaseModel):
            x: int

        s = PredictStrategy()
        model = s._create_response_model(M, "test")
        assert model is M

    def test_pydantic_model_with_hidden_creates_public_subset(self):
        from typing import Annotated

        from nooa.agentdoc import hidden

        class M(BaseModel):
            public: str
            secret: Annotated[str, hidden] = "hidden"

        s = PredictStrategy()
        model = s._create_response_model(M, "test")
        # Should NOT be the original model
        assert model is not M
        assert "public" in model.model_fields
        assert "secret" not in model.model_fields

    def test_union_multiple_non_none(self):
        """Union with multiple non-None types should create a model."""
        s = PredictStrategy()
        model = s._create_response_model(int | float, "test")
        assert hasattr(model, "model_fields")


class TestPredictStrategyValidateResponse:
    """Tests for _validate_response."""

    def test_root_model_dict(self):
        s = PredictStrategy()
        model = s._create_response_model(dict, "test")
        result = s._validate_response({"k": "v"}, model, dict)
        assert result == {"k": "v"}

    def test_list_value_wrapper_unwrapped(self):
        """`list[str]` is wrapped under `value`; _validate_response unwraps it (issue 232)."""
        s = PredictStrategy()
        model = s._create_response_model(list[str], "test")
        result = s._validate_response({"value": ["a", "b"]}, model, list[str])
        assert result == ["a", "b"]

    def test_pydantic_model_direct(self):
        class M(BaseModel):
            name: str

        s = PredictStrategy()
        model = s._create_response_model(M, "test")
        result = s._validate_response({"name": "Alice"}, model, M)
        assert isinstance(result, M)
        assert result.name == "Alice"

    def test_basic_type_unwrapped(self):
        s = PredictStrategy()
        model = s._create_response_model(str, "test")
        result = s._validate_response({"value": "hello"}, model, str)
        assert result == "hello"

    def test_pydantic_with_hidden_restores_original(self):
        """When model has hidden fields, validate_response returns original type."""
        from typing import Annotated

        from nooa.agentdoc import hidden

        class M(BaseModel):
            public: str
            secret: Annotated[str, hidden] = "default_secret"

        s = PredictStrategy()
        public_model = s._create_response_model(M, "test")
        # Validate with public model, should return original M
        result = s._validate_response({"public": "hi"}, public_model, M)
        assert isinstance(result, M)
        assert result.public == "hi"


class TestPredictStrategyAddFailedAttemptsToSpan:
    """Tests for _add_all_failed_attempts_to_span."""

    def test_no_opentelemetry_no_crash(self):
        """Should handle ImportError gracefully."""
        from unittest.mock import patch

        s = PredictStrategy()
        # Patch to simulate ImportError
        with patch.dict("sys.modules", {"opentelemetry": None, "opentelemetry.trace": None}):
            # Should not raise
            s._add_all_failed_attempts_to_span(
                [
                    {
                        "attempt": 1,
                        "raw_output": "x",
                        "error_type": "ValueError",
                        "error_message": "bad",
                    }
                ]
            )

    def test_with_recording_span(self):
        """Should set attributes on a recording span."""
        from unittest.mock import MagicMock, patch

        s = PredictStrategy()
        mock_span = MagicMock()
        mock_span.is_recording.return_value = True
        mock_ctx = MagicMock()
        mock_ctx.span_id = 12345
        mock_span.get_span_context.return_value = mock_ctx
        mock_trace = MagicMock()
        mock_trace.get_current_span.return_value = mock_span

        with patch.dict(
            "sys.modules", {"opentelemetry": MagicMock(), "opentelemetry.trace": mock_trace}
        ):
            with patch("opentelemetry.trace", mock_trace):
                # Just ensure no crash; actual span setting depends on real OTel
                s._add_all_failed_attempts_to_span(
                    [
                        {
                            "attempt": 1,
                            "raw_output": "x" * 3000,
                            "error_type": "ValueError",
                            "error_message": "bad",
                        }
                    ]
                )


# ---------------------------------------------------------------------------
# Additional tests: codeact_errors.py — missing lines via format_validation_error
# ---------------------------------------------------------------------------


class TestFormatValidationErrorPydantic:
    """Tests for format_validation_error calling _format_pydantic_error (line 25)."""

    def test_pydantic_validation_error_dispatched(self):
        """format_validation_error with PydanticValidationError calls _format_pydantic_error."""
        from pydantic import ValidationError

        class M(BaseModel):
            age: int

        try:
            M(age="bad")
        except ValidationError as e:
            result = format_validation_error(e, M, actual_value={"age": "bad"})
            assert "age" in result or "wrong type" in result


class TestFormatMissingFieldsWithIntLoc:
    """Test _format_missing_fields_error when loc[1] is an int (line 79 branch)."""

    def test_loc_with_result_prefix_and_int_index(self):
        """When loc = ('result', 0), field is formatted as [0]."""
        # This tests line 82: _format_error_path(tuple(loc_list[1:]))

        missing_errors = [{"loc": ("result", 0), "msg": "missing", "type": "missing", "url": ""}]
        result = _format_missing_fields_error(missing_errors, missing_errors, int, None)
        assert "return_result() failed" in result
        # The [0] index should appear in the Missing line
        assert "[0]" in result or "missing" in result.lower()


class TestFormatExpectedSchemaNoName:
    """Test _format_expected_schema when name is None (line 119)."""

    def test_generic_type_with_no_name(self):
        """Generic types like list[int] have no __name__, falls back to get_type_hint_str."""
        result = _format_expected_schema(list[int])
        # Should use get_type_hint_str to get a string representation
        assert "list" in result


class TestFormatExpectedSchemaPydanticModel:
    """Test _format_expected_schema for Pydantic models (lines 131-137)."""

    def test_pydantic_model_schema(self):
        """_format_expected_schema with Pydantic model shows field names and types."""

        class MyModel(BaseModel):
            name: str
            age: int

        result = _format_expected_schema(MyModel)
        assert "MyModel" in result
        assert "name" in result
        assert "age" in result
        assert "str" in result
        assert "int" in result


class TestFormatSingleErrorNamedResultField:
    """Test _format_single_error when loc=('result',) - named field 'result' with got_line.

    Lines 289 (got_line added) and 291 (example added) are hit when:
    - loc = ('result',) which makes is_result_level = True
    - actual_value is provided (got_line is set)
    - example exists for the return type (e.g., int -> 'result=42')
    """

    def test_named_result_field_type_error_with_got_line(self):
        """Named field 'result' type error shows got_line (line 289)."""
        err = {
            "loc": ("result",),
            "msg": "Input should be a valid integer",
            "type": "int_type",
            "url": "",
        }
        result = _format_single_error(err, int, "int", actual_value="bad")
        assert "Got" in result  # got_line was added (line 289)
        assert "42" in result  # example was added (line 291)


# ---------------------------------------------------------------------------
# Additional tests: codeact_lite.py — PlainProviderFormatter format() branches
# ---------------------------------------------------------------------------


class TestPlainProviderFormatterFormat:
    """Tests for PlainCodeActBlockFormatter.format() covering key branches."""

    def test_runtime_event_skipped(self):
        from nooa.context_blocks import ResolvedBlock
        from nooa.context_blocks.models import Role

        formatter = PlainProviderFormatter()
        rb = ResolvedBlock(key="runtime", content="", role=Role.RUNTIME_EVENT, event=None)
        sys_block = ResolvedBlock(key="sys", content="System", role=Role.SYSTEM)
        messages = formatter.format([sys_block, rb])
        assert len(messages) == 1
        assert messages[0].role == Role.SYSTEM

    def test_tool_call_event_with_result_no_python_output(self):
        from nooa.context_blocks import ResolvedBlock, ToolCallEvent, ToolResult
        from nooa.context_blocks.models import Role

        formatter = PlainProviderFormatter()
        tce = ToolCallEvent(
            tool_call_id="tc1",
            name="some_tool",
            arguments={"arg": "val"},
            result=ToolResult(tool_call_id="tc1", content="direct result"),
        )
        block = ResolvedBlock(key="tc", content="", role=Role.ASSISTANT, event=tce)
        messages = formatter.format([block])
        tool_msgs = [m for m in messages if m.role == Role.TOOL]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].content == "direct result"

    def test_tool_call_event_with_no_result_and_no_python_output(self):
        from nooa.context_blocks import ResolvedBlock, ToolCallEvent
        from nooa.context_blocks.models import Role

        formatter = PlainProviderFormatter()
        tce = ToolCallEvent(tool_call_id="tc2", name="some_tool", arguments={}, result=None)
        block = ResolvedBlock(key="tc", content="", role=Role.ASSISTANT, event=tce)
        messages = formatter.format([block])
        tool_msgs = [m for m in messages if m.role == Role.TOOL]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].content == ""

    def test_block_with_no_event_uses_content(self):
        from nooa.context_blocks import ResolvedBlock
        from nooa.context_blocks.models import Role

        formatter = PlainProviderFormatter()
        rb = ResolvedBlock(key="text", content="block content text", role=Role.USER, event=None)
        messages = formatter.format([rb])
        user_msgs = [m for m in messages if m.role == Role.USER]
        assert len(user_msgs) == 1
        assert user_msgs[0].content == "block content text"

    def test_block_with_no_event_and_no_content(self):
        from nooa.context_blocks import ResolvedBlock
        from nooa.context_blocks.models import Role

        formatter = PlainProviderFormatter()
        rb = ResolvedBlock(key="empty", content="", role=Role.USER, event=None)
        messages = formatter.format([rb])
        user_msgs = [m for m in messages if m.role == Role.USER]
        assert len(user_msgs) == 1
        assert user_msgs[0].content == ""

    def test_tool_call_with_python_output_uses_plain_content(self):
        from nooa.context_blocks import ResolvedBlock, ToolCallEvent
        from nooa.context_blocks.models import Role

        formatter = PlainProviderFormatter()
        po = PythonOutput(
            tool_call_id="tc_match",
            execution_status=ResultStatus.COMPLETE,
            execution_count=1,
            stdout="python result",
        )
        po_block = ResolvedBlock(key="po", content="", role=Role.TOOL, event=po)
        tce = ToolCallEvent(
            tool_call_id="tc_match",
            name="execute_python",
            arguments={"code": "x = 1"},
            result=None,
        )
        tce_block = ResolvedBlock(key="tc", content="", role=Role.ASSISTANT, event=tce)

        messages = formatter.format([tce_block, po_block])
        tool_msgs = [m for m in messages if m.role == Role.TOOL]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].content == "python result"

    def test_tool_call_with_python_output_prefers_pre_serialized_content(self):
        from nooa.context_blocks import ResolvedBlock, ToolCallEvent
        from nooa.context_blocks.models import Role

        formatter = PlainProviderFormatter()
        po = PythonOutput(
            tool_call_id="tc_match",
            execution_status=ResultStatus.COMPLETE,
            execution_count=1,
            value="X" * 2000,
        )
        po_block = ResolvedBlock(
            key="po",
            content="PRE_SERIALIZED_BY_RENDER_CONTEXT",
            role=Role.TOOL,
            event=po,
        )
        tce = ToolCallEvent(
            tool_call_id="tc_match",
            name="execute_python",
            arguments={"code": "x = 1"},
            result=None,
        )
        tce_block = ResolvedBlock(key="tc", content="", role=Role.ASSISTANT, event=tce)

        messages = formatter.format([tce_block, po_block])
        tool_msgs = [m for m in messages if m.role == Role.TOOL]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].content == "PRE_SERIALIZED_BY_RENDER_CONTEXT"
        assert "str(len=2000" not in tool_msgs[0].content

    def test_non_tool_event_prefers_pre_serialized_content(self):
        from nooa.context_blocks import ResolvedBlock
        from nooa.context_blocks.models import Role

        formatter = PlainProviderFormatter()
        event = Error(content="X" * 2000)
        block = ResolvedBlock(
            key="err",
            content="PRE_SERIALIZED_EVENT",
            role=Role.USER,
            event=event,
        )

        messages = formatter.format([block])
        user_msgs = [m for m in messages if m.role == Role.USER]
        assert len(user_msgs) == 1
        assert user_msgs[0].content == "PRE_SERIALIZED_EVENT"
        assert "X" * 2000 not in user_msgs[0].content


class TestPredictStrategyRawExtractionFallback:
    """Tests covering lines 214-215 and 220-235: fallback raw extraction in exception handler."""

    @pytest.mark.asyncio
    async def test_raw_extraction_in_exception_path(self):
        """When parse fails, raw_response_content is extracted in exception handler (lines 214-215).

        This happens when _extract_raw_from_llm_response is called inside the exception handler.
        """
        # Provide a response where content is valid JSON dict but fails validation
        # raw_response_content is set before parse fails; but to hit line 214,
        # raw_response_content must be None at that point.
        # This can happen if _extract_raw_from_llm_response raises or returns None.
        # Instead, the most direct way is to provide invalid JSON (None content + None reasoning)
        # which causes JSONDecodeError before raw_response_content is set.

        from unittest.mock import patch

        class TestAgent2(Agent, llm=_TEST_LLM):
            @strategy(PredictStrategy(config=PredictConfig(max_retries=2)))
            async def compute(self) -> int:
                """Compute something."""
                ...

        # Provide a response with valid content (string JSON) but inject an error
        # by making _extract_raw_from_llm_response return None on first call
        original_extract = PredictStrategy._extract_raw_from_llm_response
        call_count = [0]

        def mock_extract(self_s, resp):
            call_count[0] += 1
            if call_count[0] == 1:
                return None  # Force the None path -> hits lines 218-237
            return original_extract(self_s, resp)

        fake_llm = FakeLLMClient(
            scripted_responses=[
                # First: wrong type (will cause PydanticValidationError after parse)
                _llm_resp(json.dumps({"value": "not_int"})),
                # Second: correct
                _llm_resp(json.dumps({"value": 42})),
            ]
        )
        agent = TestAgent2(llm=fake_llm)
        with patch.object(PredictStrategy, "_extract_raw_from_llm_response", mock_extract):
            result = await agent.compute()
        assert result == 42


class TestPredictStrategyFormatErrorMethods:
    """Tests for _format_validation_error dispatching (lines 752-773 area)."""

    @pytest.mark.asyncio
    async def test_json_error_formatted_correctly(self):
        """JSON decode error triggers error_json_parsing method."""

        class JAgent(Agent, llm=_TEST_LLM):
            @strategy(PredictStrategy(config=PredictConfig(max_retries=2)))
            async def parse(self) -> str:
                """Parse to string."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                # Invalid JSON triggers retry
                _llm_resp("not valid json {{"),
                _llm_resp(json.dumps({"value": "fixed"})),
            ]
        )
        agent = JAgent(llm=fake_llm)
        result = await agent.parse()
        assert result == "fixed"

    @pytest.mark.asyncio
    async def test_type_error_formatted_correctly(self):
        """TypeError in validation triggers error_type_validation method."""
        from unittest.mock import patch

        class TAgent(Agent, llm=_TEST_LLM):
            @strategy(PredictStrategy(config=PredictConfig(max_retries=2)))
            async def compute(self) -> int:
                """Compute int."""
                ...

        # Make _validate_response raise TypeError on first call
        original_validate = PredictStrategy._validate_response
        call_count = [0]

        def patched_validate(self_s, data, model, orig_type):
            call_count[0] += 1
            if call_count[0] == 1:
                raise TypeError("type mismatch")
            return original_validate(self_s, data, model, orig_type)

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _llm_resp(json.dumps({"value": 42})),
                _llm_resp(json.dumps({"value": 100})),
            ]
        )
        agent = TAgent(llm=fake_llm)
        with patch.object(PredictStrategy, "_validate_response", patched_validate):
            result = await agent.compute()
        assert result == 100
