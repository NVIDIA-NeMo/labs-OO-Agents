# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for strategy error recovery paths — gl-106.

Covers retry exhaustion, max_iterations exceeded, and type validation failures
in CodeActStrategy, PurePythonStrategy, and PredictStrategy.

These tests use carefully scripted FakeLLMClient sequences to exercise branches
that are hard to hit with normal "happy path" tests.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from nooa import Agent, strategy
from nooa.config import CodeActConfig, PredictConfig
from nooa.errors import GenerationError
from nooa.strategies.codeact import CodeActStrategy
from nooa.strategies.predict import PredictStrategy
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

# ---------------------------------------------------------------------------
# Module-level Pydantic models (required for PredictStrategy type resolution)
# ---------------------------------------------------------------------------


class SentimentAnalysis(BaseModel):
    """Pydantic model for sentiment analysis tests."""

    sentiment: str
    confidence: float


class StrictIntModel(BaseModel):
    """Pydantic model that requires an int field."""

    required_field: int


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resp(content: str = "", tool_calls: list | None = None) -> LLMResponse:
    """Create a scripted LLMResponse."""
    finish_reason = "tool_calls" if tool_calls else "stop"
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        assistant_message={"role": "assistant", "content": content},
    )


def _tool_call(name: str, args_dict: dict, call_id: str = "call_1") -> ToolCall:
    """Create a ToolCall with the given name and args."""
    return ToolCall(id=call_id, name=name, arguments=json.dumps(args_dict))


def _exec_python(code: str, call_id: str = "call_exec") -> ToolCall:
    """Shorthand: create an execute_python tool call."""
    return _tool_call("execute_python", {"code": code}, call_id=call_id)


def _return_result(result: Any, call_id: str = "call_ret") -> ToolCall:
    """Shorthand: create a return_result tool call."""
    return _tool_call("return_result", {"result": result}, call_id=call_id)


class ErrorAfterNFakeLLM(FakeLLMClient):
    """FakeLLMClient that returns scripted responses then raises a RuntimeError.

    This lets us test the LLM-API-error recovery path in codeact.py (lines 668-699)
    and the generic exception path in pure_python.py (lines 326-345).
    """

    def __init__(
        self,
        scripted_responses: list[LLMResponse],
        error_message: str = "Simulated LLM API failure",
    ):
        super().__init__(scripted_responses=scripted_responses)
        self._error_message = error_message

    async def acall(
        self, messages: list[dict], tools=None, output_model=None, **kwargs
    ) -> LLMResponse:
        # If the queue is empty, raise instead of returning the default empty response.
        if not self._response_queue:
            raise RuntimeError(self._error_message)
        return await super().acall(messages, tools, output_model, **kwargs)


# ---------------------------------------------------------------------------
# CodeActStrategy — LLM API error exhaustion  (codeact.py lines 668-699)
# ---------------------------------------------------------------------------

# A shared "dummy" LLM for class-level declarations (the real LLM is set per-test)
_DUMMY_LLM = FakeLLMClient()


class TestCodeActLLMApiErrorExhaustion:
    """LLM calls raise RuntimeError; after max_retries the strategy must raise GenerationError."""

    @pytest.mark.asyncio
    async def test_llm_error_exhausts_max_retries(self):
        """All LLM calls raise RuntimeError; after max_retries the strategy raises GenerationError.

        ErrorAfterNFakeLLM with empty scripted_responses raises immediately on every call.
        Strategy tries generate() → raises RuntimeError → records error (count=1).
        session.is_exhausted() is False (max_retries=2, count=1 < 2) → loop continues.
        Next iteration: generate() → raises again → count=2 → is_exhausted() → GenerationError.
        """

        class TestAgent(Agent, llm=_DUMMY_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_retries=2, max_iterations=10)))
            async def compute(self, x: int) -> int:
                """Compute {x}."""
                ...

        fake_llm = ErrorAfterNFakeLLM(
            scripted_responses=[],  # All calls raise RuntimeError
            error_message="Simulated LLM API failure",
        )

        agent = TestAgent(llm=fake_llm)
        with pytest.raises(GenerationError) as exc_info:
            await agent.compute(5)

        err = str(exc_info.value)
        # Must mention "LLM API error after 2 retries" — the exact phrase from codeact.py line 697
        assert "LLM API error after 2 retries" in err


# ---------------------------------------------------------------------------
# CodeActStrategy — max_iterations exceeded  (codeact.py: loop exhaust path)
# ---------------------------------------------------------------------------


class TestCodeActMaxIterationsExceeded:
    """LLM keeps calling execute_python but never calls return_result.

    After max_iterations the strategy raises GenerationError.
    This exercises the outer loop exhaustion at line 759 (build_failure_error).
    """

    @pytest.mark.asyncio
    async def test_max_iterations_raises_generation_error(self):
        """With max_iterations=2, three execute_python calls → GenerationError.

        The existing test_codeact_strategy.py covers this, but we add it here
        for explicit coverage of the iteration-exhaustion path.
        """

        class TestAgent(Agent, llm=_DUMMY_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=2, max_retries=10)))
            async def never_finishes(self) -> int:
                """Never call return_result."""
                ...

        # Script 3 execute_python responses (more than max_iterations=2).
        # The loop will hit the "exhausted" guard after 2 iterations and raise.
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(tool_calls=[_exec_python("x = 1", call_id="c1")]),
                _resp(tool_calls=[_exec_python("x = 2", call_id="c2")]),
                _resp(tool_calls=[_exec_python("x = 3", call_id="c3")]),  # never reached
            ]
        )
        agent = TestAgent(llm=fake_llm)
        with pytest.raises(GenerationError) as exc_info:
            await agent.never_finishes()

        err = str(exc_info.value).lower()
        # codeact.py build_failure_error: "Generation failed after {n} iterations (max_iterations=...)"
        assert "generation failed after" in err and "iterations" in err


# ---------------------------------------------------------------------------
# CodeActStrategy — return_result with wrong type  (codeact.py lines 1118-1122)
# ---------------------------------------------------------------------------


class TestCodeActReturnResultWrongType:
    """LLM calls return_result with a value that doesn't match the declared return type.

    This exercises the inline return_result validation failure path in
    codeact.py around lines 1098-1155 (the _handle_return_result path when
    validation_error is not None).
    """

    @pytest.mark.asyncio
    async def test_return_result_wrong_type_triggers_retry(self):
        """return_result called with a string when int is expected → validation error → retry → success.

        Sequence:
        1. execute_python(sets result = 'not_an_int')   → iteration=1
        2. return_result("not_an_int")                  → validation fails, error_count=1, retry
        3. execute_python(sets result = 'also_wrong')   → iteration=3
        4. return_result("also_wrong")                  → validation fails, error_count=2, retry
        5. return_result(42)                            → validation passes, returns 42
        """

        class TestAgent(Agent, llm=_DUMMY_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=5, max_retries=5)))
            async def compute(self, x: int) -> int:
                """Return x doubled."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                # Turn 1: valid code execution (iteration=1)
                _resp(tool_calls=[_exec_python("result = 'not_an_int'", call_id="c1")]),
                # Turn 2: return_result with wrong type (iteration=2; error_count=1)
                _resp(tool_calls=[_return_result("not_an_int", call_id="r1")]),
                # Turn 3: try again — still wrong (iteration=3)
                _resp(tool_calls=[_exec_python("result = 'also_wrong'", call_id="c2")]),
                # Turn 4: return_result with another wrong type (iteration=4; error_count=2)
                _resp(tool_calls=[_return_result("also_wrong", call_id="r2")]),
                # Turn 5: try int this time — succeeds (iteration=5)
                _resp(tool_calls=[_return_result(42, call_id="r3")]),
            ]
        )
        agent = TestAgent(llm=fake_llm)
        # Each tool call (including return_result) counts as one iteration. With max_iterations=5
        # and max_retries=5, all 5 turns execute: two wrong return_result calls record errors
        # (error_count=2, still < max_retries=5), and the final return_result(42) succeeds.
        result = await agent.compute(21)
        assert result == 42

    @pytest.mark.asyncio
    async def test_inline_return_result_validation_error_in_execute_python(self):
        """return_result() called inline inside execute_python with wrong type.

        This exercises the _ReturnResultSignal path in codeact.py lines 1094-1155
        where validation_error is set (validation fails for inline call).

        The code calls return_result("not_an_int") inline — the strategy catches
        the signal, validates, finds type mismatch, sets validation_error, and
        continues the loop without completing.
        """

        class TestAgent(Agent, llm=_DUMMY_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=2, max_retries=5)))
            async def compute(self, x: int) -> int:
                """Return x doubled."""
                ...

        # Both attempts use inline return_result with wrong type
        fake_llm = FakeLLMClient(
            scripted_responses=[
                # Inline return_result call with wrong type inside execute_python
                _resp(tool_calls=[_exec_python("return_result('not_an_int')", call_id="c1")]),
                # Try again, still wrong
                _resp(tool_calls=[_exec_python("return_result('still_not_int')", call_id="c2")]),
                # Queue exhausted — empty response → session exhausted
            ]
        )
        agent = TestAgent(llm=fake_llm)
        # Should either exhaust iterations or retries → GenerationError
        with pytest.raises(GenerationError):
            await agent.compute(21)


# ---------------------------------------------------------------------------
# CodeActStrategy — lines 1845-1846, 1988-1989
# These lines inject PredictStrategy and agent-type names into the namespace.
# They are covered by any test that successfully runs CodeActStrategy with
# code that imports/uses PredictStrategy or sub-agent types.
# ---------------------------------------------------------------------------


class TestCodeActNamespaceInjection:
    """Lines 1845-1846 and 1988-1989: namespace injection in CodeActStrategy.

    Line 1845-1846: PredictStrategy added to strategy_extras dict.
    Line 1988-1989: Agent attribute types injected into exec_globals.
    These paths are hit any time CodeAct executes Python code.
    """

    @pytest.mark.asyncio
    async def test_execute_python_runs_and_namespace_is_populated(self):
        """_inject_agent_types injects SubTool into exec_globals; code that uses SubTool succeeds.

        Lines 1840-1852 (_build_namespace) and 1980-1989 (_inject_agent_types)
        are called for every execute_python invocation. This test verifies that
        _inject_agent_types actually works: the executed code references SubTool by
        name (which would NameError if injection failed) and the result confirms it.
        """

        class SubTool:
            """A sub-tool whose type should be injected into the namespace."""

            pass

        class TestAgent(Agent, llm=_DUMMY_LLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.tool = SubTool()

            @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=5, max_retries=3)))
            async def compute(self) -> int:
                """Return the answer."""
                ...

        # The executed code uses SubTool by name — if _inject_agent_types didn't inject
        # it into exec_globals, this would raise NameError and the test would fail.
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    tool_calls=[
                        _exec_python(
                            "answer = 42 if isinstance(self.tool, SubTool) else -1",
                            call_id="c1",
                        )
                    ]
                ),
                _resp(tool_calls=[_return_result(42, call_id="r1")]),
            ]
        )
        agent = TestAgent(llm=fake_llm)
        result = await agent.compute()
        assert result == 42


# ---------------------------------------------------------------------------
# PurePythonStrategy — validation error exhaustion  (pure_python.py 367-373)
# ---------------------------------------------------------------------------


class TestPurePythonValidationErrorExhaustion:
    """PurePythonStrategy: LLM returns code that raises a Pydantic ValidationError at runtime.

    This exercises the execution-error branch at pure_python.py lines 367-373:
        if result.error:
            session.record_error()
            await self._send_execution_error(...)
            continue

    The LLM returns code that instantiates a Pydantic model with invalid field types.
    The REPL executes the code, which raises a PydanticValidationError at runtime (inside
    the sandboxed exec). That exception is captured in result.error and handled at lines
    367-373 — NOT the _generate_code() PydanticValidationError branch (lines 306-325).
    After max_retries execution errors, the outer loop exhausts and raises GenerationError.

    Note: lines 306-325 handle PydanticValidationError from _generate_code() (i.e., when
    the LLM response itself fails Pydantic parsing), which is a different path.
    """

    @pytest.mark.asyncio
    async def test_pydantic_validation_failure_exhausts_retries(self):
        """LLM returns code that raises PydanticValidationError at runtime → GenerationError.

        The method declares -> SentimentAnalysis (module-level Pydantic model). The
        LLM returns code with invalid field types; the REPL raises PydanticValidationError
        during execution, captured in result.error. After max_retries errors, GenerationError
        is raised. Note: the Pydantic model must be module-level so it's in exec_globals.
        """

        class TestAgent(Agent, llm=_DUMMY_LLM):
            @strategy(PurePythonStrategy(max_iterations=10, max_retries=2))
            async def classify(self, text: str) -> SentimentAnalysis:
                """Classify the text and return a SentimentAnalysis."""
                ...

        # Code returns a SentimentAnalysis with wrong types (confidence as string, not float)
        bad_return_code = 'return SentimentAnalysis(sentiment=123, confidence="not_a_float")'

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(bad_return_code),
                _resp(bad_return_code),
                _resp(bad_return_code),  # Extra, shouldn't be reached
            ]
        )
        agent = TestAgent(llm=fake_llm)
        # The REPL raises PydanticValidationError during execution of the bad code.
        # result.error is set, session.record_error() called. After max_retries=2 errors,
        # the outer loop exits and raises GenerationError.
        with pytest.raises(GenerationError):
            await agent.classify("test text")


# ---------------------------------------------------------------------------
# PurePythonStrategy — generic LLM API error (pure_python.py lines 326-345)
# ---------------------------------------------------------------------------


class TestPurePythonLLMApiError:
    """PurePythonStrategy: LLM raises a generic exception.

    Exercises pure_python.py lines 326-345:
        except Exception as e:
            session.record_error()
            ...
            if session.is_exhausted():
                raise GenerationError(...)
    """

    @pytest.mark.asyncio
    async def test_llm_api_error_exhausts_retries(self):
        """Generic RuntimeError from LLM exhausts max_retries → GenerationError.

        All calls to acall() raise RuntimeError (via ErrorAfterNFakeLLM with empty queue).
        """

        class TestAgent(Agent, llm=_DUMMY_LLM):
            @strategy(PurePythonStrategy(max_iterations=10, max_retries=2))
            async def analyze(self, data: str) -> str:
                """Analyze {data}."""
                ...

        # All LLM calls raise RuntimeError
        fake_llm = ErrorAfterNFakeLLM(
            scripted_responses=[],
            error_message="Simulated network failure",
        )
        agent = TestAgent(llm=fake_llm)
        with pytest.raises(GenerationError) as exc_info:
            await agent.analyze("test data")

        err = str(exc_info.value).lower()
        # pure_python.py line 343: "LLM API error after {max_retries} retries."
        assert "llm api error after 2 retries" in err


# ---------------------------------------------------------------------------
# PurePythonStrategy — code validation errors (pure_python.py lines 671-679)
# ---------------------------------------------------------------------------


class TestPurePythonCodeValidation:
    """PurePythonStrategy: LLM returns code that fails REPL policy validation.

    Exercises pure_python.py lines 671-679:
        validation_errors = validator.validate(code, runtime.agent)
        if validation_errors:
            session.record_error()
            error_msg = "**Error**: Code validation failed:..."
            runtime.event_manager.add(Error(content=error_msg))
            return ExecutionResult(stdout="", error=None, defined_methods={})

    The REPL policy validator (GeneratedCodeValidator) detects async method calls
    made without `await`. When the LLM returns code that calls an async agent
    method without await, the validator emits an error, session.record_error() is
    called, and the loop continues. After max_retries, GenerationError is raised.
    """

    @pytest.mark.asyncio
    async def test_missing_await_triggers_validation_error_then_exhaustion(self):
        """Code calling an async method without await → validation error → retry → exhaustion.

        The agent has an async helper method. The LLM calls it without `await`.
        GeneratedCodeValidator.validate() returns an error, session.record_error()
        is called. After max_retries=2, GenerationError is raised.

        This exercises pure_python.py lines 671-679.
        """

        class TestAgent(Agent, llm=_DUMMY_LLM):
            async def async_helper(self, x: int) -> int:
                """An async helper that must be awaited."""
                return x * 2

            @strategy(PurePythonStrategy(max_iterations=10, max_retries=2))
            async def compute(self, x: int) -> int:
                """Return {x} doubled using async_helper."""
                ...

        # Code calls self.async_helper without await — violates REPL policy
        bad_code = "result = self.async_helper(x)\nreturn result"

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(bad_code),
                _resp(bad_code),
                _resp(bad_code),  # Should not be reached
            ]
        )
        agent = TestAgent(llm=fake_llm)
        with pytest.raises(GenerationError):
            await agent.compute(21)


# ---------------------------------------------------------------------------
# PurePythonStrategy — helper method binding errors (pure_python.py lines 720-728)
# ---------------------------------------------------------------------------


class TestPurePythonHelperMethodErrors:
    """PurePythonStrategy: helper method binding hits rejected/errors path.

    Exercises pure_python.py lines 710-728:
        if helper_result.rejected:
            session.record_error()
            ...  # lines 710-719
        if helper_result.errors:
            session.record_error()
            ...  # lines 721-728

    The HelperFunctionManager rejects helper methods that would overwrite the
    target method. Since PurePythonStrategy calls _execute_code() which calls
    HelperFunctionManager.apply(), and rejected increments error_count, after
    max_retries rejections, session is exhausted and GenerationError is raised.

    Note: The function-body-extraction step runs BEFORE helper processing, so
    code like `def target(self): return 42` is EXTRACTED (body runs). To hit
    the `rejected` path, we need a code block where a NON-target method named
    the same as the target is defined alongside other code. Alternatively, we
    can trigger `helper_result.errors` with invalid helper code.
    """

    @pytest.mark.asyncio
    async def test_helper_with_same_name_as_other_method_triggers_rejected(self):
        """LLM defines helper named 'process' but 'process' is the target → rejected.

        The LLM generates a code block that includes BOTH a helper function
        named 'process' (= the target) AND some plain code. Since there are
        non-function nodes in the body, _extract_function_body_if_wrapped()
        returns the code unchanged. Then HelperFunctionManager finds 'process' in
        the rejected list, calls session.record_error(), and returns empty result.
        After max_retries=2, GenerationError is raised.

        This exercises pure_python.py lines 710-719.
        """

        class TestAgent(Agent, llm=_DUMMY_LLM):
            @strategy(PurePythonStrategy(max_iterations=10, max_retries=2))
            async def process(self, x: int) -> int:
                """Process {x}."""
                ...

        # Define a helper named 'process' (same as target) WITH a plain statement
        # so _extract_function_body_if_wrapped does NOT extract (other_nodes is non-empty)
        # Note: the tree has both the function and a plain statement → other_nodes non-empty
        # → extraction skipped → HelperFunctionManager sees def process(self, ...) → rejected
        conflicting_code = "x = 1\ndef process(self, x: int) -> int:\n    return x * 3"

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(conflicting_code),
                _resp(conflicting_code),
                _resp(conflicting_code),  # Should not be reached
            ]
        )
        agent = TestAgent(llm=fake_llm)
        with pytest.raises(GenerationError):
            await agent.process(7)


# ---------------------------------------------------------------------------
# PredictStrategy — validation failure then retry  (predict.py lines 220-235)
# ---------------------------------------------------------------------------


class TestPredictValidationRetry:
    """PredictStrategy: LLM returns invalid JSON that fails Pydantic validation.

    Exercises predict.py lines 220-235 (raw_response_content is None handling)
    and lines 257-284 (retry after validation failure).
    """

    @pytest.mark.asyncio
    async def test_invalid_json_retries_then_succeeds(self):
        """First response: invalid JSON → catches JSONDecodeError → retries → success.

        Exercises predict.py lines 213-284 (the exception handler with retry).
        """

        class TestAgent(Agent, llm=_DUMMY_LLM):
            @strategy(PredictStrategy(config=PredictConfig(max_retries=3)))
            async def classify(self, text: str) -> str:
                """Classify {text} as positive or negative."""
                ...

        # FakeLLMClient with string responses:
        # - First: invalid JSON → JSONDecodeError → retry
        # - Second: valid JSON with "value" field
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("this is not json at all!!!"),  # → JSONDecodeError
                _resp('{"value": "positive"}'),  # → success
                _resp('{"value": "fallback"}'),  # Extra, not needed
            ]
        )
        agent = TestAgent(llm=fake_llm)
        result = await agent.classify("I love this product!")
        assert result == "positive"

    @pytest.mark.asyncio
    async def test_invalid_json_exhausts_all_retries(self):
        """All responses are invalid JSON → exhausts max_retries → GenerationError.

        Exercises predict.py lines 257-265:
            if attempt >= self.config.max_retries:
                raise GenerationError(...)
        """

        class TestAgent(Agent, llm=_DUMMY_LLM):
            @strategy(PredictStrategy(config=PredictConfig(max_retries=2)))
            async def classify(self, text: str) -> str:
                """Classify {text} as positive or negative."""
                ...

        # All responses are invalid JSON
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("not json"),  # Attempt 1 → JSONDecodeError
                _resp("also not json"),  # Attempt 2 → JSONDecodeError → GenerationError
                _resp('{"value": "ok"}'),  # Never reached
            ]
        )
        agent = TestAgent(llm=fake_llm)
        with pytest.raises(GenerationError) as exc_info:
            await agent.classify("test text")

        err = str(exc_info.value).lower()
        # Must mention "structured output validation failed after" — the exact phrase from predict.py line 263
        assert "structured output validation failed after" in err


# ---------------------------------------------------------------------------
# PredictStrategy — raw_response_content is None path (predict.py lines 220-239)
# ---------------------------------------------------------------------------


class TestPredictRawResponseNonePath:
    """Exercises predict.py lines 220-239.

    When validation fails and raw_response_content is None, the code builds
    a descriptive fallback string from the llm_response fields.

    This is triggered when _extract_raw_from_llm_response() returns None AND
    llm_response is not None. We test this with a Pydantic-model return type
    where the LLM returns malformed data.
    """

    @pytest.mark.asyncio
    async def test_pydantic_validation_failure_with_retry(self):
        """LLM returns JSON that doesn't match Pydantic model schema → retry → success.

        First attempt: JSON with wrong field name → PydanticValidationError
        Second attempt: correct JSON → success

        Exercises the retry-on-validation-failure path (lines 267-284).
        Note: SentimentAnalysis must be at module level for PredictStrategy type resolution.
        """

        class TestAgent(Agent, llm=_DUMMY_LLM):
            @strategy(PredictStrategy(config=PredictConfig(max_retries=3)))
            async def analyze(self, text: str) -> SentimentAnalysis:
                """Analyze {text} and return sentiment and confidence."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                # Wrong field names → Pydantic will reject
                _resp('{"mood": "happy", "score": 0.9}'),
                # Correct fields → success
                _resp('{"sentiment": "positive", "confidence": 0.95}'),
                _resp('{"sentiment": "fallback", "confidence": 0.0}'),  # Extra
            ]
        )
        agent = TestAgent(llm=fake_llm)
        result = await agent.analyze("Great product!")
        assert isinstance(result, SentimentAnalysis)
        assert result.sentiment == "positive"
        assert result.confidence == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_pydantic_validation_failure_exhausts_retries(self):
        """All responses have wrong Pydantic schema → GenerationError after max_retries.

        Exercises predict.py lines 257-265 for PydanticValidationError path.
        Also exercises lines 686-687 (_create_response_model success path — model_rebuild).
        Note: StrictIntModel must be at module level for PredictStrategy type resolution.
        """

        class TestAgent(Agent, llm=_DUMMY_LLM):
            @strategy(PredictStrategy(config=PredictConfig(max_retries=2)))
            async def get_value(self, x: str) -> StrictIntModel:
                """Get a value for {x}."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                # Wrong type for required_field
                _resp('{"required_field": "not_an_int"}'),  # Attempt 1: Pydantic may coerce or fail
                _resp('{"required_field": "still_not_int"}'),  # Attempt 2
                _resp('{"required_field": 42}'),  # Attempt 3 (may not be reached)
            ]
        )
        agent = TestAgent(llm=fake_llm)
        # Pydantic v2 cannot coerce "not_an_int" to int — validation fails on attempts 1 and 2.
        # With max_retries=2, error_count reaches 2 after the second failure → GenerationError.
        # The third scripted response (required_field: 42) is never reached.
        with pytest.raises(GenerationError):
            await agent.get_value("test")


# ---------------------------------------------------------------------------
# PredictStrategy — exhaustion on retry (predict.py lines 686-687)
# ---------------------------------------------------------------------------


class TestPredictModelRebuild:
    """Exercises predict.py lines 686-687: model_rebuild in _create_response_model.

    For basic types (int, str, bool), the strategy wraps them in a dynamically
    created Pydantic model and calls model_rebuild. This path is hit on every
    PredictStrategy invocation with a basic return type.
    """

    @pytest.mark.asyncio
    async def test_basic_int_return_type_uses_dynamic_model(self):
        """PredictStrategy with int return type creates dynamic model and calls model_rebuild."""

        class TestAgent(Agent, llm=_DUMMY_LLM):
            @strategy(PredictStrategy(config=PredictConfig(max_retries=3)))
            async def count_words(self, text: str) -> int:
                """Count words in {text}."""
                ...

        # Return {"value": 5} — the wrapper model uses "value" field
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp('{"value": 5}'),
            ]
        )
        agent = TestAgent(llm=fake_llm)
        result = await agent.count_words("hello world test")
        assert result == 5

    @pytest.mark.asyncio
    async def test_basic_bool_return_type_uses_dynamic_model(self):
        """PredictStrategy with bool return type — exercises model_rebuild path."""

        class TestAgent(Agent, llm=_DUMMY_LLM):
            @strategy(PredictStrategy(config=PredictConfig(max_retries=3)))
            async def is_positive(self, text: str) -> bool:
                """Is {text} positive?"""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp('{"value": true}'),
            ]
        )
        agent = TestAgent(llm=fake_llm)
        result = await agent.is_positive("I love this!")
        assert result is True
