# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests that `from __future__ import annotations` (PEP 563) does not break
CodeActStrategy, PredictStrategy, ReturnValueValidator, or CurrentCall.

With PEP 563 all annotations in this module are stored as strings at runtime,
exactly reproducing the bug described in issue #140.
"""

from __future__ import annotations

import inspect
import json
from typing import Annotated

import pytest
from pydantic import BaseModel, BeforeValidator

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.strategies.codeact import CodeActStrategy
from nooa.strategies.current_call import CurrentCall
from nooa.strategies.generated_code import ReturnValueValidator
from nooa.strategies.predict import PredictStrategy
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

# ---------------------------------------------------------------------------
# Types defined in this module — stringified by PEP 563
# ---------------------------------------------------------------------------


def _coerce(v: object) -> str:
    return str(v)


CSVResult = Annotated[str, BeforeValidator(_coerce)]


class MyResult(BaseModel):
    value: str


# ---------------------------------------------------------------------------
# _build_return_result_tool — unit tests (no LLM needed)
# ---------------------------------------------------------------------------


class TestBuildReturnResultToolPEP563:
    """_build_return_result_tool must accept resolved Annotated types."""

    def _resolved_return_type(self, fn):
        """Use typing.get_type_hints (as the fix does) to get the return type."""
        from typing import get_type_hints

        hints = get_type_hints(fn, include_extras=True)
        return hints.get("return", inspect.Parameter.empty)

    def test_annotated_csv_result_resolves_to_type(self):
        """get_type_hints must resolve 'CSVResult' string to the actual Annotated type."""
        from typing import get_type_hints

        def solve(question: str) -> CSVResult: ...

        hints = get_type_hints(solve, include_extras=True)
        rt = hints.get("return", inspect.Parameter.empty)
        # Must NOT be a string
        assert not isinstance(rt, str), f"Expected resolved type, got string: {rt!r}"
        # Must be the Annotated type
        from typing import get_origin

        assert get_origin(rt) is Annotated

    def test_build_return_result_tool_with_annotated_pep563(self):
        """_build_return_result_tool works with types resolved under PEP 563."""
        strategy = CodeActStrategy(config=CodeActConfig())

        def solve(question: str) -> CSVResult: ...

        rt = self._resolved_return_type(solve)
        # Must not raise PydanticUserError
        tool = strategy._build_return_result_tool(rt, "solve")
        assert tool.name == "return_result"

    def test_build_return_result_tool_pydantic_model_pep563(self):
        """_build_return_result_tool works with a Pydantic model under PEP 563."""
        strategy = CodeActStrategy(config=CodeActConfig())

        def solve(question: str) -> MyResult: ...

        rt = self._resolved_return_type(solve)
        assert not isinstance(rt, str)
        tool = strategy._build_return_result_tool(rt, "solve")
        assert tool.name == "return_result"

    def test_build_return_result_tool_plain_str_pep563(self):
        """Plain 'str' annotation resolves correctly under PEP 563."""
        strategy = CodeActStrategy(config=CodeActConfig())

        def solve(question: str) -> str: ...

        rt = self._resolved_return_type(solve)
        assert rt is str
        tool = strategy._build_return_result_tool(rt, "solve")
        assert tool.name == "return_result"


# ---------------------------------------------------------------------------
# CurrentCall.from_method — return_type field must be a resolved type
# ---------------------------------------------------------------------------


class TestCurrentCallPEP563:
    """CurrentCall.from_method must store resolved types, not strings."""

    def test_return_type_is_not_string_for_annotated(self):
        """call.return_type must not be a plain string under PEP 563."""

        def solve(question: str) -> CSVResult: ...

        call = CurrentCall.from_method(solve, args=("test",))
        assert call.return_type is not None
        assert not isinstance(call.return_type, str), (
            f"Expected resolved type, got string: {call.return_type!r}"
        )

    def test_return_type_is_not_string_for_pydantic_model(self):
        def solve(question: str) -> MyResult: ...

        call = CurrentCall.from_method(solve, args=("test",))
        assert call.return_type is MyResult

    def test_return_type_none_annotation(self):
        def do_nothing() -> None: ...

        call = CurrentCall.from_method(do_nothing)
        # Under PEP 563, `-> None` becomes string "None"; get_type_hints
        # resolves it to NoneType (type(None)).
        # Either None (the object) or type(None) are acceptable — both mean void.
        assert call.return_type in (None, type(None))


# ---------------------------------------------------------------------------
# ReturnValueValidator — must not error with PEP 563 string annotations
# ---------------------------------------------------------------------------


class TestReturnValueValidatorPEP563:
    """ReturnValueValidator.validate must work under PEP 563."""

    def _make_runtime(self, method_name: str, method_fn):
        """Create a minimal fake runtime that maps method_name to a callable."""

        class FakeAgent:
            pass

        agent = FakeAgent()
        setattr(agent, method_name, method_fn)

        class FakeRuntime:
            @property
            def truncation_config(self):
                from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG

                return DEFAULT_TRUNCATION_CONFIG

        rt = FakeRuntime()
        rt.agent = agent
        return rt

    def test_validate_str_return_pep563(self):
        """validate() must not raise on a plain str return type under PEP 563."""

        def solve(question: str) -> str: ...

        rt = self._make_runtime("solve", solve)
        validator = ReturnValueValidator()
        # Should not raise; "hello" is a valid str
        result = validator.validate("hello", rt, "solve")
        assert result == "hello"

    def test_validate_pydantic_model_pep563(self):
        """validate() must not raise on a Pydantic model return type under PEP 563."""

        def solve(question: str) -> MyResult: ...

        rt = self._make_runtime("solve", solve)
        validator = ReturnValueValidator()
        obj = MyResult(value="test")
        result = validator.validate(obj, rt, "solve")
        assert result is obj


# ---------------------------------------------------------------------------
# End-to-end: CodeActStrategy.execute() with a PEP 563 agent
# ---------------------------------------------------------------------------


def _return_result_call(result, call_id: str = "call_ret") -> ToolCall:
    return ToolCall(
        id=call_id,
        name="return_result",
        arguments=json.dumps({"result": result}),
    )


def _resp(content: str = "", tool_calls: list | None = None) -> LLMResponse:
    finish_reason = "tool_calls" if tool_calls else "stop"
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        assistant_message={"role": "assistant", "content": content},
    )


_DUMMY_LLM = FakeLLMClient()


class TestCodeActExecutePEP563:
    """CodeActStrategy.execute() must not raise PydanticUserError under PEP 563.

    The Agent class is defined in this module, so its method annotations are
    stringified by `from __future__ import annotations`.
    """

    @pytest.mark.asyncio
    async def test_execute_with_annotated_return_type(self):
        """execute() resolves Annotated return type from a PEP 563 module."""

        class SolveAgent(Agent, llm=_DUMMY_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def solve(self, question: str) -> MyResult:
                """Answer {question}."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(tool_calls=[_return_result_call({"value": "42"})]),
            ]
        )
        agent = SolveAgent(llm=fake_llm)
        result = await agent.solve("What is 6*7?")
        assert isinstance(result, MyResult)
        assert result.value == "42"

    @pytest.mark.asyncio
    async def test_execute_with_plain_str_return_type(self):
        """execute() with plain str return type under PEP 563."""

        class StrAgent(Agent, llm=_DUMMY_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def greet(self, name: str) -> str:
                """Greet {name}."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(tool_calls=[_return_result_call("Hello, world!")]),
            ]
        )
        agent = StrAgent(llm=fake_llm)
        result = await agent.greet("world")
        assert result == "Hello, world!"

    @pytest.mark.asyncio
    async def test_execute_with_csv_result_annotated_type(self):
        """execute() with Annotated[str, BeforeValidator(...)] — the exact repro from #140."""

        class CSVAgent(Agent, llm=_DUMMY_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def solve(self, question: str) -> CSVResult:
                """Answer {question} as CSV."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(tool_calls=[_return_result_call("col1,col2\na,b")]),
            ]
        )
        agent = CSVAgent(llm=fake_llm)
        result = await agent.solve("give me data")
        assert result == "col1,col2\na,b"


# ---------------------------------------------------------------------------
# End-to-end: PredictStrategy.execute() with a PEP 563 agent
# ---------------------------------------------------------------------------


class TestPredictExecutePEP563:
    """PredictStrategy.execute() must resolve types under PEP 563."""

    @pytest.mark.asyncio
    async def test_predict_with_pydantic_model(self):
        """PredictStrategy resolves Pydantic model return type under PEP 563."""

        class PredictAgent(Agent, llm=_DUMMY_LLM):
            @strategy(PredictStrategy())
            async def classify(self, text: str) -> MyResult:
                """Classify {text}."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(content='{"value": "positive"}'),
            ]
        )
        agent = PredictAgent(llm=fake_llm)
        result = await agent.classify("great product")
        assert isinstance(result, MyResult)
        assert result.value == "positive"

    @pytest.mark.asyncio
    async def test_predict_with_plain_str(self):
        """PredictStrategy resolves plain str return type under PEP 563."""

        class PredictStrAgent(Agent, llm=_DUMMY_LLM):
            @strategy(PredictStrategy())
            async def summarize(self, text: str) -> str:
                """Summarize {text}."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(content='{"value": "short summary"}'),
            ]
        )
        agent = PredictStrAgent(llm=fake_llm)
        result = await agent.summarize("long text here")
        assert result == "short summary"


# ---------------------------------------------------------------------------
# End-to-end: PurePythonStrategy._is_task_complete() with PEP 563
# ---------------------------------------------------------------------------


class TestPurePythonIsTaskCompletePEP563:
    """PurePythonStrategy._is_task_complete() must detect void returns under PEP 563.

    Under PEP 563, `-> None` becomes the string "None". Without get_type_hints,
    this string doesn't match `type(None)` or `None`, causing the strategy to
    think the method needs a return value when it doesn't.
    """

    @pytest.mark.asyncio
    async def test_void_method_completes_without_return(self):
        """A -> None method under PEP 563 should complete even without a return value."""

        class VoidAgent(Agent, llm=_DUMMY_LLM):
            @strategy(PurePythonStrategy(max_iterations=2))
            async def do_work(self) -> None:
                """Do some side-effect work."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(content="print('done')"),
            ]
        )
        agent = VoidAgent(llm=fake_llm)
        # Should complete without raising GenerationError (void return accepted)
        result = await agent.do_work()
        assert result is None

    @pytest.mark.asyncio
    async def test_str_method_returns_value(self):
        """A -> str method under PEP 563 should return the value from code."""

        class StrPurePythonAgent(Agent, llm=_DUMMY_LLM):
            @strategy(PurePythonStrategy(max_iterations=3))
            async def compute(self, x: int) -> str:
                """Return str(x * 2)."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(content="return str(x * 2)"),
            ]
        )
        agent = StrPurePythonAgent(llm=fake_llm)
        result = await agent.compute(21)
        assert result == "42"


# ---------------------------------------------------------------------------
# Fallback path: get_type_hints failure → falls back to inspect.signature
# ---------------------------------------------------------------------------


class TestGetTypeHintsFallback:
    """When get_type_hints raises, the code should fall back to inspect.signature."""

    def test_current_call_fallback_on_unresolvable_forward_ref(self):
        """CurrentCall.from_method falls back gracefully for unresolvable refs.

        We simulate this by creating a function whose __globals__ doesn't
        contain the referenced type, causing get_type_hints to raise NameError.
        """
        # Create a function with a forward reference that can't be resolved
        code = "def solve(question: str) -> 'CompletelyUnknownType': ..."
        ns: dict = {}
        exec(code, ns)  # noqa: S102
        fn = ns["solve"]

        # get_type_hints will raise NameError because 'CompletelyUnknownType'
        # is not in fn.__globals__. The fix should fall back to
        # sig.return_annotation which returns the string 'CompletelyUnknownType'.
        call = CurrentCall.from_method(fn, args=("test",))
        # Should not crash; return_type will be the unresolved string
        # (inspect.signature preserves the quotes from the forward ref syntax)
        assert "CompletelyUnknownType" in str(call.return_type)

    def test_validator_fallback_on_unresolvable_forward_ref(self):
        """ReturnValueValidator falls back to inspect.signature for unresolvable refs."""
        code = "def solve(question: str) -> 'NoSuchType': ..."
        ns: dict = {}
        exec(code, ns)  # noqa: S102
        fn = ns["solve"]

        class FakeAgent:
            pass

        agent = FakeAgent()
        agent.solve = fn

        class FakeRuntime:
            @property
            def truncation_config(self):
                from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG

                return DEFAULT_TRUNCATION_CONFIG

        rt = FakeRuntime()
        rt.agent = agent

        validator = ReturnValueValidator()
        # Should not crash — falls back to string annotation, which won't
        # match any special case, so it returns the value as-is
        result = validator.validate("hello", rt, "solve")
        assert result == "hello"
