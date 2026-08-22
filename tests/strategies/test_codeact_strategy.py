# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CodeActStrategy.

Tests the CodeAct strategy which combines tool-based Python execution
with the return_result tool for structured final responses.

The two-tool approach:
- execute_python(code): Run Python code for computation
- return_result(...): Return the final structured answer
"""

import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.events import PythonOutput, ResultStatus
from nooa.strategies.codeact import CodeActStrategy
from nooa.strategies.codeact_errors import (
    format_validation_error,
    get_type_example,
    get_type_hint_str,
)
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


def _resp(content: str, tool_calls: list | None = None) -> LLMResponse:
    """Create a test LLM response with the given content."""
    finish_reason = "tool_calls" if tool_calls else "stop"
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        assistant_message={"role": "assistant", "content": content},
    )


def _tool_call(code: str, call_id: str = "call_1") -> ToolCall:
    """Create an execute_python tool call."""
    return ToolCall(
        id=call_id,
        name="execute_python",
        arguments=json.dumps({"code": code}),
    )


def _return_result(call_id: str = "call_return", result: Any = None) -> ToolCall:
    """Create a return_result tool call with given result.

    Universal format: _return_result(result=<value>)
    Works for all types: int, dict, list, Pydantic models, TypedDict, etc.
    """
    return ToolCall(
        id=call_id,
        name="return_result",
        arguments=json.dumps({"result": result}),
    )


# Module-level test LLM (can be overridden at instantiation)
_TEST_LLM = FakeLLMClient()


class TestCodeActStrategyProperties:
    """Tests for CodeActStrategy properties."""

    def test_name_is_codeact(self):
        """CodeActStrategy.name should be 'CODEACT'."""
        strat = CodeActStrategy(config=CodeActConfig())
        assert strat.name == "CODEACT"

    def test_block_overrides_provides_strategy_prompt(self):
        """CodeActStrategy.get_block_overrides() should provide strategy_prompt block."""
        strat = CodeActStrategy(config=CodeActConfig())
        blocks = strat.get_block_overrides()
        assert "strategy_prompt" in blocks
        assert blocks["strategy_prompt"] is not None


class TestCodeActStrategyConfig:
    """Tests for CodeActStrategy configuration."""

    def test_default_max_iterations_is_unlimited(self):
        """Default ``max_iterations`` is ``None`` — unlimited iterations.

        Long-running agent loops (especially in interactive contexts
        like the TUI) shouldn't hit a finite default they didn't
        opt into. Callers that *want* a cap pass it explicitly.
        """
        strat = CodeActStrategy(config=CodeActConfig())
        assert strat.config.max_iterations is None

    def test_default_max_retries(self):
        """Default max_retries should be 3."""
        strat = CodeActStrategy(config=CodeActConfig())
        assert strat.config.max_retries == 3

    def test_custom_max_iterations(self):
        """Should accept custom max_iterations via config."""
        strat = CodeActStrategy(config=CodeActConfig(max_iterations=5))
        assert strat.config.max_iterations == 5

    def test_custom_max_retries(self):
        """Should accept custom max_retries via config."""
        strat = CodeActStrategy(config=CodeActConfig(max_retries=2))
        assert strat.config.max_retries == 2

    def test_accepts_config_object(self):
        """CodeActStrategy accepts CodeActConfig object."""
        strat = CodeActStrategy(config=CodeActConfig(max_iterations=5))
        assert strat.config.max_iterations == 5

    def test_default_config(self):
        """``CodeActStrategy()`` defaults: unlimited iterations, no cell timeout."""
        strat = CodeActStrategy(config=CodeActConfig())
        assert strat.config.max_iterations is None
        assert strat.config.cell_timeout is None

    def test_rejects_old_flat_kwargs(self):
        """CodeActStrategy rejects old-style flat kwargs."""
        with pytest.raises(TypeError):
            CodeActStrategy(max_iterations=5)  # old flat-kwarg API — must fail

    def test_sampling_kwargs_exclude_none(self):
        """_build_sampling_kwargs() excludes None values."""
        strat = CodeActStrategy(config=CodeActConfig())
        sampling = strat._build_sampling_kwargs()
        assert "max_tokens" not in sampling
        assert "temperature" not in sampling

    def test_sampling_kwargs_include_set_values(self):
        """_build_sampling_kwargs() includes non-None values."""
        strat = CodeActStrategy(config=CodeActConfig(temperature=0.7, max_tokens=1000))
        sampling = strat._build_sampling_kwargs()
        assert sampling["temperature"] == 0.7
        assert sampling["max_tokens"] == 1000
        assert "top_p" not in sampling  # None, excluded


class TestCodeActStrategyInheritance:
    """Tests for CodeActStrategy inheritance."""

    def test_is_generation_strategy(self):
        """CodeActStrategy should inherit from GenerationStrategy."""
        from nooa.strategies.base import GenerationStrategy

        strat = CodeActStrategy(config=CodeActConfig())
        assert isinstance(strat, GenerationStrategy)

    def test_has_execute_method(self):
        """CodeActStrategy should implement execute()."""
        strat = CodeActStrategy(config=CodeActConfig())
        assert hasattr(strat, "execute")
        assert callable(strat.execute)


class TestCodeActStrategySimpleExecution:
    """Tests for CodeActStrategy execute() with simple scenarios."""

    @pytest.mark.asyncio
    async def test_direct_return_result(self):
        """LLM calling return_result directly should work."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def answer(self) -> int:
                """Return the answer to everything."""
                ...

        # LLM calls return_result directly (knows the answer)
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_return_result(result=42)]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.answer()

        assert result == 42

    @pytest.mark.asyncio
    async def test_single_tool_call_then_result(self):
        """LLM calling execute_python once then return_result."""

        class TestAgent(Agent, llm=_TEST_LLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.data = [1, 2, 3, 4, 5]

            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute_sum(self) -> int:
                """Compute the sum of self.content."""
                ...

        # First response: tool call to compute
        # Second response: return_result with result
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "", tool_calls=[_tool_call("total = sum(self.data)\nprint(f'Sum is {total}')")]
                ),
                _resp("", tool_calls=[_return_result(result=15)]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.compute_sum()

        assert result == 15

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_then_result(self):
        """LLM calling execute_python multiple times before return_result."""

        class TestAgent(Agent, llm=_TEST_LLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.x = 10
                self.y = 20

            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute(self) -> dict:
                """Compute sum and product of x and y."""
                ...

        # Multiple tool calls, then return_result
        # Note: For dict return types, use result= parameter
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call("s = self.x + self.y\nprint(f'Sum: {s}')")]),
                _resp("", tool_calls=[_tool_call("p = self.x * self.y\nprint(f'Product: {p}')")]),
                _resp("", tool_calls=[_return_result(result={"sum": 30, "product": 200})]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.compute()

        assert result == {"sum": 30, "product": 200}


class TestCodeActStrategyPydanticOutput:
    """Tests for CodeActStrategy with Pydantic model return types."""

    @pytest.mark.asyncio
    async def test_pydantic_model_return(self):
        """LLM returning Pydantic model via return_result should work."""
        from pydantic import BaseModel

        class AnalysisResult(BaseModel):
            score: int
            label: str

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def analyze(self, text: str) -> AnalysisResult:
                """Analyze the given text."""
                ...

        # LLM calls return_result with Pydantic model fields
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_return_result(result={"score": 85, "label": "positive"})]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.analyze("Great product!")

        assert isinstance(result, AnalysisResult)
        assert result.score == 85
        assert result.label == "positive"

    @pytest.mark.asyncio
    async def test_pydantic_constructor_string_return_remains_compatible(self):
        """A literal constructor string is still corrected without another LLM turn."""

        class AnalysisResult(BaseModel):
            score: int
            label: str

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=1)))
            async def analyze(self) -> AnalysisResult:
                """Return the analysis result."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "",
                    tool_calls=[
                        _return_result(result='AnalysisResult(score=85, label="positive")')
                    ],
                ),
            ]
        )

        result = await TestAgent(llm=fake_llm).analyze()

        assert result == AnalysisResult(score=85, label="positive")
        assert fake_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_pydantic_validation_error_triggers_retry(self):
        """Test that Pydantic validation errors trigger regeneration."""
        from pydantic import BaseModel

        class UserProfile(BaseModel):
            username: str
            email: str
            age: int

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_retries=3)))
            async def get_user(self) -> UserProfile:
                """Return user profile with username, email, and age."""
                ...

        # First attempt: missing required 'age' field
        # Second attempt: valid with all fields
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "",
                    tool_calls=[
                        _return_result(result={"username": "alice", "email": "alice@example.com"})
                    ],
                ),  # Missing age
                _resp(
                    "",
                    tool_calls=[
                        _return_result(
                            result={"username": "bob", "email": "bob@example.com", "age": 25}
                        )
                    ],
                ),  # Complete
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.get_user()

        assert isinstance(result, UserProfile)
        assert result.username == "bob"
        assert result.age == 25
        assert fake_llm.call_count == 2  # Retry after validation error


def _return_result_raw_args(call_id: str = "call_return", **kwargs) -> ToolCall:
    """Create a return_result tool call with raw (pre-serialized) arguments.

    Unlike _return_result which json.dumps the kwargs, this allows passing
    a raw arguments string to simulate LLM edge cases like Python literals.
    """
    return ToolCall(
        id=call_id,
        name="return_result",
        arguments=kwargs.get("_raw_args", "{}"),
    )


class TestCodeActStrategyDictAndListReturnTypes:
    """Tests for dict and list return types using RootModel.

    For dict and list return types, the LLM uses return_result(result=...)
    with the value directly.
    """

    @pytest.mark.asyncio
    async def test_dict_return_type(self):
        """Dict return types should work via return_result(result=...)."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def process(self) -> dict:
                """Return a dict with message and action_taken fields."""
                ...

        # LLM uses return_result with result= for dict types
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "",
                    tool_calls=[
                        _return_result(result={"message": "Hello!", "action_taken": "greeted"})
                    ],
                ),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.process()

        assert result == {"message": "Hello!", "action_taken": "greeted"}
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_typed_dict_return(self):
        """dict[str, Any] should also work via return_result."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def analyze(self, data: str) -> dict[str, int]:
                """Analyze data and return counts."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_return_result(result={"words": 5, "chars": 25})]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.analyze("hello world")

        assert result == {"words": 5, "chars": 25}

    @pytest.mark.asyncio
    async def test_list_return_type(self):
        """List return types should work via return_result(result=...)."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def get_items(self) -> list:
                """Get a list of items."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_return_result(result=["apple", "banana", "cherry"])]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.get_items()

        assert result == ["apple", "banana", "cherry"]
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_typed_list_return(self):
        """list[str] should also work via return_result."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def get_names(self) -> list[str]:
                """Get list of names."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_return_result(result=["Alice", "Bob", "Charlie"])]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.get_names()

        assert result == ["Alice", "Bob", "Charlie"]

    @pytest.mark.asyncio
    async def test_list_as_python_string_single_quotes(self):
        """List returned as Python string with single quotes should be parsed.

        Some LLMs return lists as Python literal strings like "['a', 'b']"
        instead of actual JSON arrays. The strategy should handle this.
        """

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def get_labels(self) -> list[str]:
                """Get sentiment labels."""
                ...

        # Simulate LLM returning Python list syntax as a string (single quotes)
        # The arguments JSON has the list value as a string, not a proper array
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "",
                    tool_calls=[
                        _return_result_raw_args(
                            _raw_args="{\"result\": \"['positive', 'negative', 'neutral']\"}"
                        )
                    ],
                ),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.get_labels()

        assert result == ["positive", "negative", "neutral"]
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_dict_as_python_string_single_quotes(self):
        """Dict returned as Python string with single quotes should be parsed.

        Some LLMs return dicts as Python literal strings like "{'key': 'value'}"
        instead of actual JSON objects. The strategy should handle this.
        """

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def get_counts(self) -> dict[str, int]:
                """Get word counts."""
                ...

        # Simulate LLM returning Python dict syntax as a string (single quotes)
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "",
                    tool_calls=[
                        _return_result_raw_args(
                            _raw_args="{\"result\": \"{'words': 10, 'chars': 50}\"}"
                        )
                    ],
                ),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.get_counts()

        assert result == {"words": 10, "chars": 50}
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_dict_after_tool_calls(self):
        """Dict return should work after tool calls (the original bug scenario)."""

        class TestAgent(Agent, llm=_TEST_LLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.order_items = []

            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def process_message(self, user_message: str) -> dict:
                """Process user message and return response with action."""
                ...

        # Tool call to add items, then return_result
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call("self.order_items.append('pizza')")]),
                _resp(
                    "",
                    tool_calls=[
                        _return_result(
                            result={
                                "message": "Added pizza to your order.",
                                "action_taken": "added pizza",
                            }
                        )
                    ],
                ),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.process_message("I want a pizza")

        assert result == {"message": "Added pizza to your order.", "action_taken": "added pizza"}
        assert agent_instance.order_items == ["pizza"]

    @pytest.mark.asyncio
    async def test_string_return_with_json_like_content_not_parsed(self):
        """String return type with JSON-like content should NOT be parsed.

        Regression test: When return type is str, LLM might return valid JSON strings
        like "[10, 20, 30]" that should remain as strings, not be parsed into lists.
        """

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def transform(self, values: list[int]) -> str:
                """Transform values and return as JSON string."""
                ...

        # LLM correctly returns a JSON string as the result
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_return_result(result="[10, 20, 30]")]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.transform([1, 2, 3])

        # Result should be the string "[10, 20, 30]", NOT parsed into a list
        assert result == "[10, 20, 30]"
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_string_return_with_dict_like_content_not_parsed(self):
        """String return type with dict-like JSON content should NOT be parsed.

        Another regression test for the same bug: dict-like JSON strings.
        """

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def get_json_response(self) -> str:
                """Return a JSON string representation."""
                ...

        # LLM correctly returns a JSON object as a string
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_return_result(result='{"name": "Alice", "age": 30}')]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.get_json_response()

        # Result should be the string, NOT parsed into a dict
        assert result == '{"name": "Alice", "age": 30}'
        assert isinstance(result, str)


class TestCodeActStrategyToolResults:
    """Tests for tool result formatting."""

    @pytest.mark.asyncio
    async def test_tool_result_includes_stdout(self):
        """Tool result should include stdout from code execution."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def debug_task(self) -> str:
                """A simple task that uses print."""
                ...

        # First: tool call with print
        # Second: return_result
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call("print('Debug output')")]),
                _resp("", tool_calls=[_return_result(result="done")]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.debug_task()

        # Result should be correct
        assert result == "done"

        # Check that execution output was added to history
        history_events = agent_instance.event_manager.values()
        exec_output_events = [e for e in history_events if e.event_type == "PythonOutput"]

        # Should have one execute_python event (for execute_python)
        assert len(exec_output_events) == 1
        exec_event = exec_output_events[0]

        # Should contain the printed output in stdout
        assert "Debug output" in exec_event.stdout

    @pytest.mark.asyncio
    async def test_explicit_return_auto_completes_when_type_matches(self):
        """Explicit `return x` auto-completes the task if type matches."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute_task(self) -> int:
                """A task that returns an integer."""
                ...

        # Just one call needed - explicit return with matching type auto-completes
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call("return 42")]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.compute_task()

        # Result comes from auto-completion of explicit return statement
        assert result == 42

    @pytest.mark.asyncio
    async def test_bare_expression_does_not_auto_complete(self):
        """Bare expression (IPython-style) does NOT auto-complete, requires return_result."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute_task(self) -> int:
                """A task that returns an integer."""
                ...

        # First: bare expression 42 (shown as Out:, but doesn't auto-complete)
        # Second: explicit return_result needed to complete the task
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call("42")]),  # Bare expression, not return
                _resp("", tool_calls=[_return_result(result=42)]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.compute_task()

        # Result comes from explicit return_result, not auto-completion
        assert result == 42

        # Check that execution output shows "Out[n]:" (Jupyter-style)
        history_events = agent_instance.event_manager.values()
        exec_output_events = [e for e in history_events if e.event_type == "PythonOutput"]

        assert len(exec_output_events) == 1  # One execute_python with output
        first_exec_event = exec_output_events[0]

        # Check the value is captured (rendered as Out[n] by formatter)
        assert first_exec_event.value == 42
        assert first_exec_event.execution_count == 1

    @pytest.mark.asyncio
    async def test_explicit_return_type_mismatch_shows_out_continues(self):
        """Explicit return with type mismatch shows 'Out:' and continues loop."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute_task(self) -> dict:
                """A task that returns a dict."""
                ...

        # First: explicit return of int (type mismatch with dict, shown as Out:)
        # Second: return_result with correct dict type
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call("return 42")]),  # Type mismatch
                _resp("", tool_calls=[_return_result(result={"key": "value"})]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.compute_task()

        assert result == {"key": "value"}

        # Check that execution output shows "Out[n]:" (Jupyter-style)
        history_events = agent_instance.event_manager.values()
        exec_output_events = [e for e in history_events if e.event_type == "PythonOutput"]

        # First execute_python is from execute_python (showing "Out[n]: 42")
        assert len(exec_output_events) >= 1
        first_exec_event = exec_output_events[0]

        # Check the value is captured (rendered as Out[n] by formatter)
        assert first_exec_event.value == 42


class TestCodeActStrategyOutAccessor:
    """Tests for Out[n] accessor with sparse execution counts (Jupyter-style)."""

    @pytest.mark.asyncio
    async def test_out_accessor_sparse_execution_counts(self):
        """Test Out[n] with blocks where some return values and some don't.

        Jupyter behavior:
        - Block 1: returns 42 → Out[1] = 42
        - Block 2: assignment only (x = 5) → Out[2] raises KeyError
        - Block 3: returns Out[1] + 10 → Out[3] = 52

        This matches Jupyter where Out[n] only exists for cells that produce output.
        """

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute_with_gaps(self) -> int:
                """A task that computes with gaps in Out indices."""
                ...

        # Simulate: block 1 returns, block 2 no return, block 3 uses Out[1]
        fake_llm = FakeLLMClient(
            scripted_responses=[
                # Block 1: bare expression returns 42 → Out[1] = 42
                _resp("", tool_calls=[_tool_call("42")]),
                # Block 2: assignment only, no return → Out[2] doesn't exist
                _resp("", tool_calls=[_tool_call("x = 5")]),
                # Block 3: use Out[1] to compute result
                _resp("", tool_calls=[_tool_call("Out[1] + 10")]),
                # Final: return the computed value
                _resp("", tool_calls=[_return_result(result=52)]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.compute_with_gaps()

        # Task should complete with correct result
        assert result == 52

        # Verify execution output events show correct Out[] indices
        history_events = agent_instance.event_manager.values()
        exec_output_events = [e for e in history_events if e.event_type == "PythonOutput"]

        # Should have 3 execute_python events (one per execute_python call)
        assert len(exec_output_events) == 3

        # First execution output (block 1): Out[1]: 42
        assert exec_output_events[0].execution_count == 1
        assert exec_output_events[0].value == 42

        # Second execution output (block 2): no Out[] (assignment only, value is None)
        assert exec_output_events[1].execution_count == 2
        assert exec_output_events[1].value is None

        # Third execution output (block 3): Out[3]: 52
        assert exec_output_events[2].execution_count == 3
        assert exec_output_events[2].value == 52

    @pytest.mark.asyncio
    async def test_out_accessor_negative_indexing(self):
        """Test Out[-1] returns last output regardless of gaps.

        Out[-1] should return the most recent output value,
        even if there are gaps in the execution count sequence.
        """

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute_with_negative_index(self) -> int:
                """A task that uses negative Out indexing."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                # Block 1: returns 100 → Out[1] = 100
                _resp("", tool_calls=[_tool_call("100")]),
                # Block 2: assignment only → no Out[2]
                _resp("", tool_calls=[_tool_call("y = 200")]),
                # Block 3: Out[-1] should be 100 (last recorded)
                _resp("", tool_calls=[_tool_call("Out[-1] * 2")]),
                # Final return
                _resp("", tool_calls=[_return_result(result=200)]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.compute_with_negative_index()

        assert result == 200

        # Verify Out[-1] worked correctly in block 3
        history_events = agent_instance.event_manager.values()
        exec_output_events = [e for e in history_events if e.event_type == "PythonOutput"]

        # Should have 3 execution outputs (one per execute_python call)
        assert len(exec_output_events) == 3

        # Block 1: Out[1]: 100
        assert exec_output_events[0].value == 100

        # Block 2: assignment only, no value
        assert exec_output_events[1].value is None

        # Block 3 result: Out[3]: 200 (Out[-1] * 2 = 100 * 2)
        assert exec_output_events[2].execution_count == 3
        assert exec_output_events[2].value == 200

    @pytest.mark.asyncio
    async def test_out_accessor_key_error_for_missing_index(self):
        """Test that accessing Out[n] for non-output block raises KeyError."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def trigger_key_error(self) -> str:
                """A task that triggers KeyError on Out[2]."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                # Block 1: returns value → Out[1] exists
                _resp("", tool_calls=[_tool_call("'first'")]),
                # Block 2: no return → Out[2] doesn't exist
                _resp("", tool_calls=[_tool_call("z = 10")]),
                # Block 3: try to access Out[2] → should error
                _resp("", tool_calls=[_tool_call("Out[2]")]),
                # Final return after error
                _resp("", tool_calls=[_return_result(result="done")]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.trigger_key_error()

        assert result == "done"

        # Verify block 3 shows the KeyError in execute_python
        history_events = agent_instance.event_manager.values()
        exec_output_events = [e for e in history_events if e.event_type == "PythonOutput"]
        assert isinstance(exec_output_events[0], PythonOutput)
        # Find the execute_python with the error (status: error)
        error_outputs = [e for e in exec_output_events if e.execution_status == ResultStatus.ERROR]
        assert len(error_outputs) >= 1

        # Block 3 should contain KeyError message in error field
        error_event = error_outputs[0]
        error_text = error_event.error or error_event.stderr
        assert "KeyError" in error_text or "No output for execution 2" in error_text


class TestCodeActStrategyErrorHandling:
    """Tests for error handling in CodeActStrategy."""

    @pytest.mark.asyncio
    async def test_max_iterations_exceeded(self):
        """Should raise GenerationError when max_iterations exceeded."""
        from nooa.errors import GenerationError

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=2)))
            async def never_finishes(self) -> str:
                """This task never returns a result."""
                ...

        # LLM keeps calling tools, never returns structured output
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call("print('iter 1')")]),
                _resp("", tool_calls=[_tool_call("print('iter 2')")]),
                _resp("", tool_calls=[_tool_call("print('iter 3')")]),  # Should not be reached
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)

        with pytest.raises(GenerationError) as exc_info:
            await agent_instance.never_finishes()

        assert "iterations" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_missing_return_type_raises_error(self):
        """Should raise GenerationError if method has no return type."""
        from nooa.errors import GenerationError

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def no_return_type(self):
                """No return type annotation."""
                ...

        agent_instance = TestAgent(llm=FakeLLMClient())

        with pytest.raises(GenerationError) as exc_info:
            await agent_instance.no_return_type()

        assert "return type" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_code_execution_error_in_tool_result(self):
        """Code execution errors should be reported in tool result."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def error_task(self) -> str:
                """A task that might have errors."""
                ...

        # First: tool call with syntax error
        # Second: return_result (after error)
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call("this is not valid python code!!!")]),
                _resp("", tool_calls=[_return_result(result="recovered")]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.error_task()

        assert result == "recovered"

        # Check that execution event includes error message
        # Note: PythonOutput contains the error directly, not as tool_result
        history_events = agent_instance.event_manager.values()
        exec_events = [e for e in history_events if e.event_type == "PythonOutput"]

        assert len(exec_events) >= 1
        exec_event = exec_events[0]

        # Should have an error attribute
        assert exec_event.error is not None
        assert "error" in str(exec_event.error).lower() or "syntax" in str(exec_event.error).lower()


class TestCodeActStrategyEventSequence:
    """Tests for correct event sequence in history."""

    @pytest.mark.asyncio
    async def test_tool_call_creates_correct_event_sequence(self):
        """After tool calls, history should have proper sequence.

        The architecture nests ToolResult inside ToolCallEvent.result,
        so there are no separate tool_result events.

        Sequence: task -> tool_call (with nested result) -> execute_python -> tool_call (with nested result)
        """

        class TestAgent(Agent, llm=_TEST_LLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.value = 42

            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def get_value(self) -> int:
                """Get the value."""
                ...

        # First response: execute_python tool call
        # Second response: return_result tool call
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call("print(self.value)", call_id="call_abc123")]),
                _resp("", tool_calls=[_return_result(call_id="call_return", result=42)]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.get_value()

        assert result == 42

        events = agent_instance.event_manager.values()
        event_types = [e.event_type for e in events]

        # Architecture: ToolResult is nested in ToolCallEvent.result (no separate tool_result events)
        # Sequence: Task -> ToolCallEvent -> PythonOutput -> ToolCallEvent
        assert event_types == [
            "Task",
            "ToolCallEvent",
            "PythonOutput",
            "ToolCallEvent",
        ], f"Expected ['Task', 'ToolCallEvent', 'PythonOutput', 'ToolCallEvent'], got {event_types}"

        # Verify first ToolCallEvent has correct data (execute_python)
        tool_call_event = events[1]
        assert tool_call_event.event_type == "ToolCallEvent"
        assert tool_call_event.tool_call_id == "call_abc123"
        assert tool_call_event.name == "execute_python"
        assert "code" in tool_call_event.arguments

        # Verify ToolResult is nested in ToolCallEvent.result
        assert tool_call_event.result is not None
        assert tool_call_event.result.tool_call_id == "call_abc123"

        # Verify execute_python event contains the deferred output
        exec_output_event = events[2]
        assert exec_output_event.event_type == "PythonOutput"
        assert exec_output_event.tool_call_id == "call_abc123"

        # Verify second ToolCallEvent is return_result
        return_call_event = events[3]
        assert return_call_event.event_type == "ToolCallEvent"
        assert return_call_event.name == "return_result"
        assert return_call_event.result is not None

    @pytest.mark.asyncio
    async def test_no_empty_assistant_event_before_tool_call(self):
        """Empty LLMOutputs should be removed before ToolCallEvents.

        When the LLM returns a tool call with no content, the CodeAct strategy
        removes the empty LLMOutput to keep history clean.
        """

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def simple_task(self) -> str:
                """A simple task."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call("print('hello')")]),
                _resp("", tool_calls=[_return_result(result="done")]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.simple_task()

        assert result == "done"

        # Check that we don't have empty LLMOutputs before ToolCallEvents
        events = agent_instance.event_manager.values()
        event_types = [e.event_type for e in events]

        # Verify no empty LLMOutput events remain (they should be removed when tool calls are made)
        for i in range(len(event_types) - 1):
            if event_types[i] == "LLMOutput" and event_types[i + 1] == "ToolCallEvent":
                # Check if it's an empty assistant event
                if not events[i].content:
                    pytest.fail(
                        f"Found empty LLMOutput before ToolCallEvent at index {i}. "
                        f"Event sequence: {event_types}"
                    )

    @pytest.mark.asyncio
    async def test_text_only_stop_response_routes_through_return_result(self):
        """Text-only LLM response (finish_reason=stop) routes through return_result validation.

        When the LLM returns finish_reason="stop" with text content, the strategy
        constructs a synthetic return_result(content) tool call and routes it through
        validation. If the return type matches (e.g. str), the session terminates
        successfully with the content as the return value.
        """

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def think_and_answer(self) -> str:
                """A task that requires thinking."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                # LLM says "stop" with text — for str return type, this becomes the result
                _resp("The answer is 42."),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.think_and_answer()

        # With str return type, the text content passes validation and becomes the result
        assert result == "The answer is 42."

        events = agent_instance.event_manager.values()
        event_types = [e.event_type for e in events]

        # The synthetic return_result ToolCallEvent should be present
        tool_call_events = [e for e in events if e.event_type == "ToolCallEvent"]
        assert len(tool_call_events) == 1
        synthetic = tool_call_events[0]
        assert synthetic.name == "return_result"
        assert synthetic.result is not None
        assert synthetic.result.result_status == ResultStatus.COMPLETE

        # No error event should be added
        assert "Error" not in event_types

    @pytest.mark.asyncio
    async def test_text_only_basemodel_response_routes_through_return_result(self):
        """BaseModel text-only response is serialized to JSON and routed through return_result.

        LLMResponse.content can be a BaseModel (e.g. from structured output). The same
        stop→return_result path applies; content is serialized via model_dump_json()
        before being passed as the result argument.
        """
        from pydantic import BaseModel as PydanticBaseModel

        class ThoughtModel(PydanticBaseModel):
            thought: str

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def think_and_answer(self) -> str:
                """A task that requires thinking."""
                ...

        model_response = LLMResponse(
            raw_response=None,
            content=ThoughtModel(thought="I need to reason carefully here."),
            tool_calls=[],
            finish_reason="stop",
            assistant_message={"role": "assistant", "content": ""},
        )
        fake_llm = FakeLLMClient(
            scripted_responses=[
                model_response,
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.think_and_answer()

        # BaseModel serialized as JSON string passes str return type validation
        assert "I need to reason carefully here." in result

        events = agent_instance.event_manager.values()
        tool_call_events = [e for e in events if e.event_type == "ToolCallEvent"]
        assert len(tool_call_events) == 1
        assert tool_call_events[0].name == "return_result"
        assert tool_call_events[0].result.result_status == ResultStatus.COMPLETE

    @pytest.mark.asyncio
    async def test_text_only_stop_with_typed_return_gives_validation_error(self):
        """Text-only stop with non-str return type fails validation → LLM self-corrects.

        This is the fix for issue #185: when the LLM emits text without a tool call
        (infinite loop scenario), routing through return_result() gives the LLM an
        actionable validation error instead of a no-op synthetic comment call.
        The LLM then self-corrects by calling return_result() with proper typed data.
        """

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute_stats(self) -> dict:
                """Compute statistics and return a dict."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                # First response: text-only "I'm done!" (the bug trigger)
                _resp("I have successfully completed the computation!"),
                # Second response: LLM self-corrects after seeing validation error
                _resp("", tool_calls=[_return_result(result={"mean": 42, "count": 10})]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.compute_stats()

        # The LLM self-corrected and returned the proper typed result
        assert result == {"mean": 42, "count": 10}

        events = agent_instance.event_manager.values()
        # Should see: Task → synthetic return_result (with error) → real return_result (success)
        tool_call_events = [e for e in events if e.event_type == "ToolCallEvent"]
        assert len(tool_call_events) == 2

        # First tool call is the synthetic return_result that failed validation
        first = tool_call_events[0]
        assert first.name == "return_result"
        assert first.result is not None
        assert first.result.result_status == ResultStatus.ERROR
        assert "Invalid result" in first.result.content

        # Second tool call is the real return_result that succeeded
        second = tool_call_events[1]
        assert second.name == "return_result"
        assert second.result.result_status == ResultStatus.COMPLETE

    @pytest.mark.asyncio
    async def test_stop_no_content_with_none_return_type_terminates(self):
        """finish_reason=stop with no content and -> None return type terminates cleanly.

        Common in optimizer scenarios where the agent's work is done via side effects
        and it just needs to signal completion.
        """

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def do_side_effects(self) -> None:
                """Perform work via side effects."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                # LLM says stop with no content — it's done
                _resp(""),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.do_side_effects()

        assert result is None

        events = agent_instance.event_manager.values()
        tool_call_events = [e for e in events if e.event_type == "ToolCallEvent"]
        assert len(tool_call_events) == 1
        assert tool_call_events[0].name == "return_result"
        assert tool_call_events[0].result.result_status == ResultStatus.COMPLETE

    @pytest.mark.asyncio
    async def test_text_only_whitespace_response_treated_as_empty(self):
        """Whitespace-only text response (no tool calls) is treated as empty, not synthetic.

        "   " is truthy but str.strip() is falsy, so it should fall through to the
        empty-response error handler rather than creating a synthetic comment.
        """
        from nooa.errors import GenerationError

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_retries=1)))
            async def whitespace_task(self) -> str:
                """A task where LLM returns only whitespace."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("   "),  # whitespace-only, no tool calls
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)

        with pytest.raises(GenerationError):
            await agent_instance.whitespace_task()

        all_events = agent_instance.event_manager.values()
        event_types = [e.event_type for e in all_events]
        # Should produce an error event, NOT a synthetic tool_call
        assert "Error" in event_types, f"Expected error event, got: {event_types}"
        synthetic_calls = [
            e for e in all_events if e.event_type == "ToolCallEvent" and e.metadata.get("synthetic")
        ]
        assert len(synthetic_calls) == 0, (
            f"Whitespace-only response should not create synthetic events, got: {synthetic_calls}"
        )

    @pytest.mark.asyncio
    async def test_text_only_basemodel_response_with_tool_calls_prepends_comment(self):
        """BaseModel content alongside execute_python tool calls is prepended as a comment.

        Exercises the model_dump_json() branch in the content+tool_calls path.
        """
        from pydantic import BaseModel as PydanticBaseModel

        class ThoughtModel(PydanticBaseModel):
            thought: str

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def think_and_answer(self) -> str:
                """A task requiring thought."""
                ...

        model_response = LLMResponse(
            raw_response=None,
            content=ThoughtModel(thought="I should calculate this."),
            tool_calls=[_tool_call("x = 6 * 7", call_id="c1")],
            finish_reason="tool_calls",
            assistant_message={"role": "assistant", "content": ""},
        )
        fake_llm = FakeLLMClient(
            scripted_responses=[
                model_response,
                _resp("", tool_calls=[_return_result(result="done")]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.think_and_answer()
        assert result == "done"

        events = agent_instance.event_manager.values()
        exec_calls = [
            e for e in events if e.event_type == "ToolCallEvent" and e.name == "execute_python"
        ]
        assert len(exec_calls) == 1
        code = exec_calls[0].arguments["code"]
        assert code.startswith("# "), f"Expected comment prepended, got: {code!r}"
        assert "I should calculate this." in code, f"BaseModel JSON should appear in code: {code!r}"
        assert "x = 6 * 7" in code, f"Original code should follow: {code!r}"

    @pytest.mark.asyncio
    async def test_empty_stop_response_routes_through_return_result(self):
        """Empty LLM response with finish_reason='stop' routes through return_result(None).

        When the LLM emits finish_reason='stop' with no content, it signals completion.
        This is routed through return_result(None) validation. For non-None return types,
        validation fails and the LLM gets feedback to self-correct.
        """

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_retries=2)))
            async def empty_task(self) -> str:
                """A task where LLM returns stop with no content then self-corrects."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(""),  # empty stop - routes through return_result(None), fails for -> str
                _resp("", tool_calls=[_return_result(result="corrected answer")]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.empty_task()

        assert result == "corrected answer"

        # The synthetic return_result(None) should have failed validation
        all_events = agent_instance.event_manager.values()
        tool_call_events = [e for e in all_events if e.event_type == "ToolCallEvent"]
        assert len(tool_call_events) == 2
        # First is the failed synthetic return_result(None)
        assert tool_call_events[0].name == "return_result"
        assert tool_call_events[0].result.result_status == ResultStatus.ERROR
        # Second is the successful self-correction
        assert tool_call_events[1].name == "return_result"
        assert tool_call_events[1].result.result_status == ResultStatus.COMPLETE

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_event_sequence(self):
        """Multiple tool calls produce ToolCallEvent with nested result + PythonOutput.

        Architecture: ToolResult is nested in ToolCallEvent.result (no separate tool_result events)
        """

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def multi_step(self) -> int:
                """Multi-step computation."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call("x = 1", call_id="call_1")]),
                _resp("", tool_calls=[_tool_call("y = 2", call_id="call_2")]),
                _resp("", tool_calls=[_return_result(call_id="call_return", result=3)]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.multi_step()

        assert result == 3

        events = agent_instance.event_manager.values()
        event_types = [e.event_type for e in events]

        # Architecture: ToolResult is nested in ToolCallEvent.result (no separate tool_result events)
        # Sequence: Task -> (ToolCallEvent -> PythonOutput) x2 -> ToolCallEvent
        assert event_types == [
            "Task",
            "ToolCallEvent",
            "PythonOutput",
            "ToolCallEvent",
            "PythonOutput",
            "ToolCallEvent",
        ], f"Expected correct sequence with nested results, got {event_types}"

        # Verify each ToolCallEvent has nested result
        tool_call_events = [e for e in events if e.event_type == "ToolCallEvent"]
        for tc in tool_call_events:
            assert tc.result is not None, (
                f"ToolCallEvent {tc.tool_call_id} should have nested result"
            )

    @pytest.mark.asyncio
    async def test_content_plus_tool_calls_prepends_comment(self):
        """When LLM returns both content and execute_python tool calls, the content
        is prepended as a comment at the top of the first execute_python code.

        This preserves any explanatory text the LLM produced alongside its tool
        call without creating a separate synthetic event.
        """

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def think_and_answer(self) -> str:
                """A task that requires thinking."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                # LLM returns text content AND a tool call in the same response
                _resp(
                    "Let me work through this step by step.",
                    tool_calls=[_tool_call("x = 42", call_id="c1")],
                ),
                _resp("", tool_calls=[_return_result(result="done")]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.think_and_answer()

        assert result == "done"

        events = agent_instance.event_manager.values()
        tool_calls = [e for e in events if e.event_type == "ToolCallEvent"]

        # First tool call should have the content prepended as a comment
        first_tc = tool_calls[0]
        code = first_tc.arguments["code"]
        assert code.startswith("# "), f"Expected comment prepended, got: {code!r}"
        assert "Let me work through this step by step." in code, (
            f"Original content should appear in the comment, got: {code!r}"
        )
        assert "x = 42" in code, f"Original code should follow the comment, got: {code!r}"

    @pytest.mark.asyncio
    async def test_content_plus_tool_calls_empty_content_not_prepended(self):
        """Whitespace-only content alongside tool calls is ignored (not prepended)."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute(self) -> int:
                """Compute."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("   ", tool_calls=[_return_result(result=7)]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.compute()

        assert result == 7

        # The return_result tool call should have no comment prepended
        events = agent_instance.event_manager.values()
        tool_calls = [e for e in events if e.event_type == "ToolCallEvent"]
        final_tc = tool_calls[0]
        code = final_tc.arguments.get("code", "")
        assert not code.startswith("# "), (
            f"Whitespace-only content should not be prepended, got: {code!r}"
        )

    @pytest.mark.asyncio
    async def test_text_only_loop_aborts_after_threshold(self):
        """Repeated text-only responses abort with GenerationError (issue 185).

        When the LLM keeps emitting plain-text summaries instead of calling
        return_result(...), the strategy routes each through return_result()
        validation. For non-matching return types, validation fails and the
        consecutive_text_only counter increments. After max_consecutive_text_only
        consecutive failures, the guard aborts with a clear error message.
        """
        from nooa.errors import GenerationError

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(
                CodeActStrategy(config=CodeActConfig(max_consecutive_text_only=3, max_retries=10))
            )
            async def stuck_task(self) -> dict:
                """Task where the LLM repeatedly outputs text instead of return_result."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("I have successfully completed the task!"),
                _resp("I have successfully completed the task!"),
                _resp("I have successfully completed the task!"),
                _resp("I have successfully completed the task!"),
                _resp("I have successfully completed the task!"),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)

        with pytest.raises(GenerationError) as exc_info:
            await agent_instance.stuck_task()

        msg = str(exc_info.value)
        assert "max_consecutive_text_only" in msg, f"Expected guard message, got: {msg!r}"
        assert "plain text" in msg, f"Expected 'plain text' in message, got: {msg!r}"
        # Last text should be embedded for debuggability
        assert "successfully completed" in msg, (
            f"Expected last text preview in message, got: {msg!r}"
        )

        # Should have exactly 3 synthetic return_result events (all failed validation)
        # then abort fires before the 4th
        events = agent_instance.event_manager.values()
        return_result_calls = [
            e for e in events if e.event_type == "ToolCallEvent" and e.name == "return_result"
        ]
        assert len(return_result_calls) == 3, (
            f"Expected exactly 3 return_result attempts before abort, got {len(return_result_calls)}"
        )

    @pytest.mark.asyncio
    async def test_text_only_counter_resets_on_real_tool_call(self):
        """A real execute_python tool call resets the text-only counter."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(
                CodeActStrategy(config=CodeActConfig(max_consecutive_text_only=3, max_retries=10))
            )
            async def mixed(self) -> int:
                """Task that alternates text-only and real tool calls."""
                ...

        # 2 text → 1 real exec → 2 text → return_result. Counter resets at the exec,
        # so neither text run reaches the 3-strike abort threshold.
        # Each text-only "stop" routes through return_result() which fails validation
        # for -> int, but the counter tracks them separately.
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("Thinking..."),
                _resp("Still thinking..."),
                _resp("", tool_calls=[_tool_call("x = 1", call_id="c1")]),
                _resp("Almost there..."),
                _resp("Just one more step..."),
                _resp("", tool_calls=[_return_result(result=42)]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.mixed()
        assert result == 42

    @pytest.mark.asyncio
    async def test_text_only_loop_disabled_when_threshold_zero(self):
        """max_consecutive_text_only=0 disables the abort guard."""
        from nooa.errors import GenerationError

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(
                CodeActStrategy(
                    config=CodeActConfig(
                        max_consecutive_text_only=0, max_iterations=4, max_retries=10
                    )
                )
            )
            async def stuck_task(self) -> dict:
                """Task where the LLM repeatedly outputs text-only."""
                ...

        # 5 text-only responses; with abort disabled, max_iterations=4 cuts off
        # via the iteration-exhaustion path with a *different* error message.
        # Uses -> dict so return_result("text") fails validation (doesn't terminate early).
        fake_llm = FakeLLMClient(
            scripted_responses=[_resp("text") for _ in range(5)],
        )

        agent_instance = TestAgent(llm=fake_llm)
        with pytest.raises(GenerationError) as exc_info:
            await agent_instance.stuck_task()
        # When disabled, the abort message should not appear
        msg = str(exc_info.value)
        assert "max_consecutive_text_only" not in msg, (
            f"Disabled guard should not produce its message, got: {msg!r}"
        )

    @pytest.mark.asyncio
    async def test_text_only_loop_abort_records_after_turn_event(self):
        """The aborting turn emits AfterTurn with is_final=True and exception_type set.

        AfterTurn uses record=False (runtime event, not persisted in backend),
        so we capture it via the event-manager subscription channel.
        """
        from nooa.errors import GenerationError

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(
                CodeActStrategy(
                    config=CodeActConfig(
                        max_consecutive_text_only=2,
                        text_only_stop_behavior="synthetic_comment",
                    )
                )
            )
            async def stuck(self) -> str:
                """Task."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[_resp("text") for _ in range(4)],
        )

        agent_instance = TestAgent(llm=fake_llm)
        captured: list[Any] = []
        agent_instance.event_manager.on("AfterTurn", lambda e: captured.append(e))

        with pytest.raises(GenerationError):
            await agent_instance.stuck()

        assert captured, "Expected at least one AfterTurn event"
        last = captured[-1]
        assert last.is_final is True, f"Final AfterTurn should be is_final=True, got {last}"
        assert last.success is False
        assert last.exception_type == "GenerationError"

    @pytest.mark.asyncio
    async def test_content_plus_tool_calls_prepends_first_execute_python_only(self):
        """The comment is prepended to the first execute_python; later ones are untouched."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute(self) -> int:
                """Compute."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "Thinking aloud.",
                    tool_calls=[
                        _tool_call("x = 1", call_id="c1"),
                        _tool_call("y = 2", call_id="c2"),
                    ],
                ),
                _resp("", tool_calls=[_return_result(result=3)]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.compute()
        assert result == 3

        events = agent_instance.event_manager.values()
        exec_calls = [
            e for e in events if e.event_type == "ToolCallEvent" and e.name == "execute_python"
        ]
        # First execute_python should have the comment prepended
        assert exec_calls[0].arguments["code"].startswith("# "), (
            f"First execute_python should have the comment prepended, got: {exec_calls[0].arguments['code']!r}"
        )
        # Second execute_python should be unchanged
        assert exec_calls[1].arguments["code"] == "y = 2", (
            f"Second execute_python should be unchanged, got: {exec_calls[1].arguments['code']!r}"
        )

    def test_prepend_comment_skips_to_next_on_invalid_json(self):
        """If the first execute_python has invalid JSON arguments, skip it and prepend to next."""
        from nooa.strategies.codeact import _prepend_comment
        from nooa.unifiedllm import ToolCall

        bad_tc = ToolCall(id="bad", name="execute_python", arguments="NOT VALID JSON")
        good_tc = ToolCall(id="c2", name="execute_python", arguments=json.dumps({"code": "x = 42"}))
        result = _prepend_comment([bad_tc, good_tc], "Thinking aloud.")

        # First tool call unchanged (bad JSON)
        assert result[0].arguments == "NOT VALID JSON"
        # Second tool call should have the comment prepended
        args = json.loads(result[1].arguments)
        assert args["code"].startswith("# "), (
            f"Second execute_python should have the comment prepended, got: {args['code']!r}"
        )
        assert "x = 42" in args["code"]

    def test_prepend_comment_no_execute_python_unchanged(self):
        """If there's no execute_python in the list, all tool calls are returned unchanged."""
        from nooa.strategies.codeact import _prepend_comment
        from nooa.unifiedllm import ToolCall

        rr = ToolCall(id="ret", name="return_result", arguments=json.dumps({"result": 7}))
        result = _prepend_comment([rr], "some content")

        assert len(result) == 1
        assert result[0].arguments == rr.arguments


class TestCodeActStrategyPersistentState:
    """Tests for persistent state across tool calls."""

    @pytest.mark.asyncio
    async def test_variables_persist_across_calls(self):
        """Variables defined in one tool call should be available in the next.

        Uses `return x * 2` to auto-complete with a computed value that depends
        on the persisted variable. If x doesn't persist, this fails with NameError.
        """

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def multi_step(self) -> int:
                """A multi-step computation."""
                ...

        # First: define a variable
        # Second: use the variable in a return - if x doesn't persist, this fails
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call("x = 10")]),
                _resp(
                    "", tool_calls=[_tool_call("return x * 2")]
                ),  # Auto-completes with computed value
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.multi_step()

        # Result is computed from persisted x, not hardcoded
        assert result == 20

    @pytest.mark.asyncio
    async def test_multiple_variables_persist_across_calls(self):
        """Multiple variables persist across multiple tool calls."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute(self) -> int:
                """Compute using multiple persisted variables."""
                ...

        # Define variables across 3 calls, then use them all
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call("a = 5")]),
                _resp("", tool_calls=[_tool_call("b = 10")]),
                _resp("", tool_calls=[_tool_call("c = 3")]),
                _resp("", tool_calls=[_tool_call("return a + b * c")]),  # 5 + 10*3 = 35
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.compute()

        assert result == 35

    @pytest.mark.asyncio
    async def test_variable_overwrites_persist(self):
        """Reassigning a variable persists the new value."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute(self) -> int:
                """Test variable reassignment."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call("x = 10")]),
                _resp("", tool_calls=[_tool_call("x = x + 5")]),  # Reassign x to 15
                _resp("", tool_calls=[_tool_call("return x * 2")]),  # Should be 30, not 20
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.compute()

        assert result == 30

    @pytest.mark.asyncio
    async def test_helper_methods_still_persist(self):
        """Helper methods (def with self) should still persist after variable persistence is added."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute(self) -> int:
                """Test helper method persistence."""
                ...

        # helpers are plain callables — call double(self, n), not self.double(n).
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call("def double(self, n):\n    return n * 2")]),
                _resp("", tool_calls=[_tool_call("return_result(double(self, 21))")]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.compute()

        assert result == 42


class TestCodeActStrategyTypedDict:
    """Tests for TypedDict return type support."""

    @pytest.mark.asyncio
    async def test_typeddict_return_type(self):
        """TypedDict return types should be converted to proper Pydantic models."""
        from typing import TypedDict

        class Result(TypedDict):
            name: str
            count: int
            items: list[str]

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def process(self) -> Result:
                """Process and return a Result."""
                ...

        # LLM returns via return_result with all fields
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "",
                    tool_calls=[
                        _return_result(
                            result={"name": "test", "count": 42, "items": ["a", "b", "c"]}
                        )
                    ],
                ),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.process()

        # Should return a dict (TypedDict at runtime is dict)
        assert isinstance(result, dict)
        assert result == {"name": "test", "count": 42, "items": ["a", "b", "c"]}

    @pytest.mark.asyncio
    async def test_typeddict_validation_error(self):
        """TypedDict validation errors should be reported properly."""
        from typing import TypedDict

        class UserData(TypedDict):
            username: str
            age: int
            email: str

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_retries=3)))
            async def get_user(self) -> UserData:
                """Get user data."""
                ...

        # First attempt: missing required field
        # Second attempt: valid with all fields
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "", tool_calls=[_return_result(result={"username": "alice", "age": 25})]
                ),  # Missing email
                _resp(
                    "",
                    tool_calls=[
                        _return_result(
                            result={"username": "bob", "age": 30, "email": "bob@example.com"}
                        )
                    ],
                ),  # Complete
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.get_user()

        assert isinstance(result, dict)
        assert result == {"username": "bob", "age": 30, "email": "bob@example.com"}
        assert fake_llm.call_count == 2  # Retry after validation error

    def test_missing_fields_error_shows_expected_schema(self):
        """Test that missing fields error shows expected schema and actual value."""
        from typing import TypedDict

        from pydantic import ValidationError

        # Test with TypedDict
        class RouterResult(TypedDict):
            agents_called: list[str]
            results: dict

        # Simulate a Pydantic validation error for missing fields
        from pydantic import create_model

        TestModel = create_model("RouterResultModel", result=(RouterResult, ...))

        try:
            # This will raise because we're missing required fields
            TestModel(result={"all_positive": False, "is_sorted": True})
        except ValidationError as e:
            # Format the error
            actual_value = {"all_positive": False, "is_sorted": True}
            error_msg = format_validation_error(e, RouterResult, actual_value)

            # Check that error message contains expected schema
            assert "Expected:" in error_msg
            assert "RouterResult" in error_msg
            assert "agents_called" in error_msg
            assert "results" in error_msg

            # Check that error shows what was received
            assert "Got:" in error_msg
            assert "all_positive" in error_msg or "is_sorted" in error_msg

            # Check that missing fields are listed
            assert "Missing:" in error_msg

    @pytest.mark.parametrize(
        "return_type,wrong_value,expected_type_str,got_type_str",
        [
            (list, "some string value", "list", "str"),
            (int, "not a number", "int", "str"),
            (str, [1, 2, 3], "str", "list"),
        ],
    )
    def test_wrong_type_error_shows_expected_and_got(
        self, return_type, wrong_value, expected_type_str, got_type_str
    ):
        """Wrong-type validation error shows Expected type, Got type, and value (not raw Pydantic)."""
        from pydantic import ValidationError, create_model

        model = create_model("ResultModel", result=(return_type, ...))
        try:
            model(result=wrong_value)
        except ValidationError as e:
            error_msg = format_validation_error(e, return_type, wrong_value)

            # Our formatted message: clear type names
            assert f"Expected: {expected_type_str}" in error_msg
            assert f"Got: {got_type_str}" in error_msg
            # Truncated value for debugging
            assert "(value:" in error_msg
            assert "has wrong type" in error_msg
            assert "return_result(" in error_msg and "'result'" in error_msg

            # Raw Pydantic message must not appear (we replace it)
            assert "Input should be a valid" not in error_msg
            assert "For further information visit" not in error_msg

    def test_wrong_type_error_pydantic_model_expected_got_str(self):
        """Wrong-type for Pydantic model (e.g. UserResponse): expected model, got str."""
        from pydantic import BaseModel, ValidationError, create_model

        class UserResponse(BaseModel):
            message: str
            session_complete: bool = False

        model = create_model("HandleUserMessageReturnResult", result=(UserResponse, ...))
        wrong_value = 'UserResponse(\n    message="Hello",\n    session_complete=False\n)'
        try:
            model(result=wrong_value)
        except ValidationError as e:
            error_msg = format_validation_error(e, UserResponse, wrong_value)

            assert "Expected: UserResponse" in error_msg
            assert "Got: str" in error_msg
            assert "(value:" in error_msg
            assert "has wrong type" in error_msg
            assert "Input should be a valid" not in error_msg


class TestCodeActStrategyExport:
    """Tests for CodeActStrategy exports."""

    def test_exported_from_strategies_module(self):
        """CodeActStrategy should be exported from strategies module."""
        from nooa.strategies import CodeActStrategy

        assert CodeActStrategy is not None

    def test_exported_from_main_module(self):
        """CodeActStrategy should be exported from main nooa module."""
        from nooa import CodeActStrategy

        assert CodeActStrategy is not None


class TestCodeActInlineReturnResult:
    """Tests for inline return_result() functionality.

    These tests verify that LLMs can call return_result() from within
    execute_python code to compute and return the final answer in one step.
    """

    @pytest.mark.asyncio
    async def test_inline_return_result_simple_int(self):
        """return_result() called from within execute_python with simple int."""

        class TestAgent(Agent, llm=_TEST_LLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.value = 42

            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def get_value(self) -> int:
                """Return the value."""
                ...

        # Single tool call: execute_python that calls return_result inline
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "",
                    tool_calls=[
                        _tool_call("result = self.value\nreturn_result(result=result)", "call_1")
                    ],
                ),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.get_value()

        assert result == 42
        # Only one tool call should be needed (not two)
        assert fake_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_inline_return_result_with_computation(self):
        """return_result() called inline after computation."""

        class TestAgent(Agent, llm=_TEST_LLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.data = [1, 2, 3, 4, 5]

            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute_sum(self) -> int:
                """Compute the sum."""
                ...

        # Execute Python and return result inline
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "",
                    tool_calls=[
                        _tool_call(
                            "total = sum(self.data)\nprint(f'Sum is {total}')\nreturn_result(result=total)",
                            "call_1",
                        )
                    ],
                ),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.compute_sum()

        assert result == 15
        assert fake_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_inline_return_result_dict(self):
        """return_result() called inline with dict result."""

        class TestAgent(Agent, llm=_TEST_LLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.x = 10
                self.y = 20

            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute(self) -> dict:
                """Compute sum and product."""
                ...

        # Single tool call computing both values and returning inline
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "",
                    tool_calls=[
                        _tool_call(
                            "s = self.x + self.y\np = self.x * self.y\nreturn_result(result={'sum': s, 'product': p})",
                            "call_1",
                        )
                    ],
                ),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.compute()

        assert result == {"sum": 30, "product": 200}
        assert fake_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_inline_return_result_validation_error_retry(self):
        """Inline return_result() with validation error should allow retry."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_retries=3)))
            async def get_number(self) -> int:
                """Get a number."""
                ...

        # First attempt: wrong type (string instead of int)
        # Second attempt: correct type
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call("return_result(result='not an int')", "call_1")]),
                _resp("", tool_calls=[_tool_call("return_result(result=42)", "call_2")]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.get_number()

        assert result == 42
        assert fake_llm.call_count == 2  # Retry after validation error

    @pytest.mark.asyncio
    async def test_inline_return_result_with_pydantic_model(self):
        """return_result() called inline with Pydantic model result."""
        from pydantic import BaseModel

        class Result(BaseModel):
            success: bool
            value: int

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute(self) -> Result:
                """Compute a result."""
                ...

        # Execute and return inline with Pydantic model
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "",
                    tool_calls=[
                        _tool_call(
                            "x = 10 + 20\nreturn_result(result={'success': True, 'value': x})",
                            "call_1",
                        )
                    ],
                ),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.compute()

        # Should complete in one tool call
        assert fake_llm.call_count == 1
        assert isinstance(result, Result)
        assert result.success is True
        assert result.value == 30

    @pytest.mark.asyncio
    async def test_mixed_usage_patterns(self):
        """Test that both usage patterns work (inline and separate tool calls)."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute(self, use_inline: bool) -> int:
                """Compute a value."""
                ...

        # First call: use inline return_result
        inline_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "", tool_calls=[_tool_call("x = 10 + 20\nreturn_result(result=x)", "call_1")]
                ),
            ]
        )

        agent = TestAgent(llm=inline_llm)
        result1 = await agent.compute(use_inline=True)
        assert result1 == 30

        # Second call: use separate tool calls (traditional approach)
        separate_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call("x = 10 + 20\nprint(x)", "call_1")]),
                _resp("", tool_calls=[_return_result(result=30, call_id="call_2")]),
            ]
        )

        agent2 = TestAgent(llm=separate_llm)
        result2 = await agent2.compute(use_inline=False)
        assert result2 == 30

        # Both approaches should work and produce the same result
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_inline_return_result_positional_arg(self):
        """Test return_result() with positional argument."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute(self) -> int:
                """Compute a value."""
                ...

        # Use positional arg: return_result(42) instead of return_result(result=42)
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call("x = 10 + 32\nreturn_result(x)", "call_1")]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.compute()

        assert result == 42
        assert fake_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_inline_return_result_positional_arg_dict(self):
        """Test return_result() with positional argument that is a dict."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute(self) -> dict:
                """Compute a value."""
                ...

        # Use positional arg with dict
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "",
                    tool_calls=[
                        _tool_call(
                            "data = {'key': 'value', 'count': 42}\nreturn_result(data)", "call_1"
                        )
                    ],
                ),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.compute()

        assert result == {"key": "value", "count": 42}
        assert fake_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_inline_return_result_not_caught_by_except_exception(self):
        """return_result() inside try block should not be caught by 'except Exception'.

        The _ReturnResultSignal inherits from BaseException (not Exception), so it
        should NOT be caught by 'except Exception:' blocks. This ensures that
        when the LLM writes defensive code like:

            try:
                result = await self.some_method()
                return_result(result)  # Should succeed and exit
            except Exception as e:
                return_result(fallback)  # Should NOT be reached

        The first return_result() succeeds instead of being caught by the except block.
        """

        class TestAgent(Agent, llm=_TEST_LLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.correct_value = 42
                self.fallback_value = -1

            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def get_value(self) -> int:
                """Get the correct value."""
                ...

        # LLM generates code with try/except that WOULD catch the signal
        # if it inherited from Exception (which it no longer does)
        code = """
try:
    result = self.correct_value
    return_result(result)  # This should succeed and exit
except Exception as e:
    print(f"Caught exception: {e}")
    return_result(self.fallback_value)  # This should NOT be reached
"""
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call(code.strip(), "call_1")]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.get_value()

        # The correct value (42) should be returned, NOT the fallback (-1)
        assert result == 42
        assert fake_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_inline_return_result_in_nested_try_except(self):
        """return_result() works correctly in nested try/except blocks.

        Related to issue #55 - verifies the fix works in more complex scenarios.
        """

        class TestAgent(Agent, llm=_TEST_LLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.value = 100

            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute(self) -> int:
                """Compute a value."""
                ...

        # Nested try/except blocks
        code = """
try:
    try:
        x = self.value * 2
        return_result(x)  # Should succeed: 200
    except Exception as inner_e:
        return_result(-1)  # Should NOT be reached
except Exception as outer_e:
    return_result(-2)  # Should NOT be reached
"""
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call(code.strip(), "call_1")]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.compute()

        # Should return 200, not -1 or -2
        assert result == 200
        assert fake_llm.call_count == 1


# Module-level models for TestCodeActReturnTypeInspection (must be in module for exec_globals)
class _AnalyzerResultForInspection(BaseModel):
    """Analysis result with statistics (module-level so in exec_globals)."""

    mean: float
    median: float
    count: int


class _UserProfileForInspection(BaseModel):
    """User profile data (module-level so in exec_globals)."""

    username: str
    email: str
    age: int


class TestCodeActReturnTypeInspection:
    """Tests for inspecting return types from within execute_python.

    These tests verify that when return types are defined at module level they
    are in exec_globals, so LLMs can inspect them (e.g. print(AnalyzerResult)).
    """

    @pytest.mark.asyncio
    async def test_inspect_pydantic_return_type_current_behavior(self):
        """When return type is at module level, LLM can print it to inspect structure."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def analyze(self, values: list[float]) -> _AnalyzerResultForInspection:
                """Analyze the values and return statistics."""
                ...

        # LLM tries to inspect the return type (in exec_globals because module-level), then returns result
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call("print(_AnalyzerResultForInspection)", "call_1")]),
                _resp(
                    "",
                    tool_calls=[
                        _return_result(
                            result={"mean": 2.5, "median": 2.5, "count": 4}, call_id="call_2"
                        )
                    ],
                ),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.analyze([1.0, 2.0, 3.0, 4.0])

        assert isinstance(result, _AnalyzerResultForInspection)
        assert result.mean == 2.5
        assert result.count == 4

        # First tool call should succeed: type is at module level so in exec_globals
        history_events = agent_instance.event_manager.values()
        execute_python_events = [e for e in history_events if e.event_type == "PythonOutput"]
        assert len(execute_python_events) >= 2
        llm_first_event = execute_python_events[1]
        assert isinstance(llm_first_event, PythonOutput)
        assert llm_first_event.execution_status != ResultStatus.ERROR

    @pytest.mark.asyncio
    async def test_help_is_aliased_to_doc(self):
        """Test that help() is aliased to doc() to prevent blocking on stdin.

        LLMs sometimes call help() instead of doc(). Python's built-in help()
        launches an interactive pager that blocks on stdin, hanging evaluations.
        We shadow help() with doc() in the execution namespace.
        """
        from pydantic import BaseModel

        class DataResult(BaseModel):
            """Result of data processing."""

            total: int
            average: float

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def process_data(self, values: list[int]) -> DataResult:
                """Process the data."""
                ...

        # LLM calls help(self) - should work via doc() alias without blocking
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call("print(help(self))", "call_1")]),
                # After seeing documentation, provide correct result
                _resp(
                    "",
                    tool_calls=[
                        _return_result(result={"total": 10, "average": 2.5}, call_id="call_2")
                    ],
                ),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.process_data([1, 2, 3, 4])

        # Should succeed
        assert isinstance(result, DataResult)
        assert result.total == 10

        # Verify help() returned doc-like output (not an error)
        # With deferred output pattern, actual content is in execute_python events
        history_events = agent_instance.event_manager.values()
        execute_python_events = [e for e in history_events if e.event_type == "PythonOutput"]
        first_event = execute_python_events[0]
        assert isinstance(first_event, PythonOutput)
        # Should not be an error status
        assert first_event.execution_status == ResultStatus.COMPLETE
        # Value should contain documentation (string output from doc())
        first_output = str(first_event.value or first_event.stdout or "")
        # doc(self) returns agent documentation including class name
        assert "TestAgent" in first_output or "process_data" in first_output

    @pytest.mark.asyncio
    async def test_inspect_return_type_fields(self):
        """When return type is at module level, LLM can inspect model_fields."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def get_profile(self) -> _UserProfileForInspection:
                """Get user profile."""
                ...

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "",
                    tool_calls=[
                        _tool_call("print(_UserProfileForInspection.model_fields)", "call_1")
                    ],
                ),
                _resp(
                    "",
                    tool_calls=[
                        _return_result(
                            result={"username": "alice", "email": "alice@example.com", "age": 30},
                            call_id="call_2",
                        )
                    ],
                ),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.get_profile()

        assert isinstance(result, _UserProfileForInspection)
        assert result.username == "alice"

        # First execution should succeed: type is at module level so in exec_globals
        history_events = agent_instance.event_manager.values()
        execute_python_events = [e for e in history_events if e.event_type == "PythonOutput"]
        first_exec = execute_python_events[0]
        assert isinstance(first_exec, PythonOutput)
        assert first_exec.execution_status != ResultStatus.ERROR


@pytest.fixture
def mock_runtime():
    """Create mock runtime for strategy tests."""
    from nooa.events import ExecutionResult

    class MockRuntime:
        def __init__(self):
            self._agent = MagicMock()
            self._agent.agent_id = "test_agent"
            self._agent.__class__.__name__ = "TestAgent"
            self._agent.event_manager = MagicMock()
            self._agent.events = []
            self._events = MagicMock()
            self._events.add = MagicMock(return_value="event_123")
            self._events.get = MagicMock(return_value=None)
            self._events.update = MagicMock(return_value=True)

        @property
        def agent(self):
            return self._agent

        @property
        def events(self):
            """Event manager."""
            return self._events

        async def generate(self, *, tools=None, output_model=None, tool_choice=None, **kwargs):
            # Mock returning a return_result tool call
            response = MagicMock(
                content="",
                reasoning=None,
                usage={},
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(id="call_mock", name="return_result", arguments='{"value": 42}')
                ],
            )
            return response, "event_123"

        async def execute_code(self, code, builtins=None, validate=True, wrap_in_function=False):
            return ExecutionResult(
                stdout="",
                error=None,
                defined_methods={},
            )

        async def execute_nested(self, strategy, call):
            """Execute nested strategy (for @strategy methods)."""
            return await strategy.execute(self, call)

        async def expand_variables(self, template, extra_context=None, error_mode="raise"):
            """Simple variable expansion for templates."""
            context = extra_context or {}
            result = template

            for key, value in context.items():
                result = result.replace(f"{{{key}}}", str(value))

            return result

    return MockRuntime()


class TestCodeActNoneReturnType:
    """Tests for CodeActStrategy with None return type.

    These tests verify that methods with `-> None` return type work correctly,
    completing the task without requiring a result value.
    """

    @pytest.mark.asyncio
    async def test_none_return_type_direct_return_result(self):
        """Method with -> None should complete when LLM calls return_result()."""

        class TestAgent(Agent, llm=_TEST_LLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.work_done = False

            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def do_work(self) -> None:
                """Perform some task (LLM-generated)."""
                ...

        # LLM does work, then calls return_result with no arguments
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call("self.work_done = True", call_id="call_1")]),
                # return_result with empty or None result
                _resp(
                    "",
                    tool_calls=[
                        ToolCall(
                            id="call_return",
                            name="return_result",
                            arguments="{}",  # No result argument
                        )
                    ],
                ),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.do_work()

        # Should complete successfully and return None
        assert result is None
        assert agent_instance.work_done is True

    @pytest.mark.asyncio
    async def test_none_return_type_with_explicit_none(self):
        """Method with -> None should accept return_result(result=None)."""

        class TestAgent(Agent, llm=_TEST_LLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.processed = False

            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def process(self) -> None:
                """Process something."""
                ...

        # LLM explicitly passes result=None
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_tool_call("self.processed = True", call_id="call_1")]),
                _resp("", tool_calls=[_return_result(result=None, call_id="call_return")]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.process()

        assert result is None
        assert agent_instance.processed is True

    @pytest.mark.asyncio
    async def test_none_return_type_inline_return_result(self):
        """Inline return_result() with no args should work for -> None methods."""

        class TestAgent(Agent, llm=_TEST_LLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.items = []

            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def add_item(self, item: str) -> None:
                """Add an item to the list."""
                ...

        # LLM uses inline return_result() within execute_python
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "",
                    tool_calls=[
                        _tool_call("self.items.append('apple')\nreturn_result()", "call_1")
                    ],
                ),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.add_item("apple")

        assert result is None
        assert agent_instance.items == ["apple"]
        assert fake_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_none_return_type_explicit_return_none_auto_completes(self):
        """Explicit `return None` should auto-complete for -> None methods."""

        class TestAgent(Agent, llm=_TEST_LLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.counter = 0

            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def increment(self) -> None:
                """Increment the counter."""
                ...

        # LLM uses explicit return None
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "",
                    tool_calls=[_tool_call("self.counter += 1\nreturn None", "call_1")],
                ),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.increment()

        # Should auto-complete from explicit return None
        assert result is None
        assert agent_instance.counter == 1
        assert fake_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_none_return_type_rejects_non_none_value(self):
        """Method with -> None should reject return_result with a non-None value."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(max_retries=3)))
            async def do_nothing(self) -> None:
                """Do nothing and return None."""
                ...

        # First: incorrectly return a value
        # Second: correctly return None
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp("", tool_calls=[_return_result(result=42, call_id="call_1")]),  # Wrong!
                _resp(
                    "",
                    tool_calls=[
                        ToolCall(
                            id="call_return",
                            name="return_result",
                            arguments="{}",  # Correct - no result
                        )
                    ],
                ),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.do_nothing()

        assert result is None
        assert fake_llm.call_count == 2  # Retry after validation error


class TestCodeActMultiToolCallsPerResponse:
    """Tests for LLM returning multiple tool calls in a single response.

    Some LLMs (e.g., gpt-oss-120b) return multiple tool calls in one response
    even when parallel_tool_calls=false. These tests verify we process all
    tool calls sequentially and stop at the first error.
    """

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_single_response_processed_sequentially(self):
        """All tool calls in a single response should be processed in order."""

        class TestAgent(Agent, llm=_TEST_LLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.trace = []

            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute(self) -> int:
                """Compute a value."""
                ...

        # Single response with 3 tool calls, then return_result
        fake_llm = FakeLLMClient(
            scripted_responses=[
                # Single response with multiple tool calls
                _resp(
                    "",
                    tool_calls=[
                        _tool_call("self.trace.append('cell1')", call_id="call_1"),
                        _tool_call("self.trace.append('cell2')", call_id="call_2"),
                        _tool_call("self.trace.append('cell3')", call_id="call_3"),
                    ],
                ),
                # Then return result
                _resp("", tool_calls=[_return_result(result=42)]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.compute()

        # All 3 cells should have been executed in order and assigned distinct
        # cell numbers even though they came from one model response.
        assert agent_instance.trace == ["cell1", "cell2", "cell3"]
        assert result == 42
        outputs = [
            event
            for event in agent_instance.event_manager.values()
            if isinstance(event, PythonOutput)
        ]
        assert [event.execution_count for event in outputs] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_multi_tool_calls_stop_on_first_error(self):
        """If one tool call fails, remaining tool calls in batch should not run."""

        class TestAgent(Agent, llm=_TEST_LLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.trace = []

            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute(self) -> str:
                """Compute something."""
                ...

        # Single response with 3 tool calls where cell 2 has an error
        fake_llm = FakeLLMClient(
            scripted_responses=[
                # Single response: cell1 ok, cell2 error, cell3 should not run
                _resp(
                    "",
                    tool_calls=[
                        _tool_call("self.trace.append('cell1')", call_id="call_1"),
                        _tool_call("undefined_variable_xyz", call_id="call_2"),  # Error!
                        _tool_call("self.trace.append('cell3')", call_id="call_3"),
                    ],
                ),
                # After error, LLM fixes it and returns
                _resp("", tool_calls=[_return_result(result="done")]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.compute()

        # Cell1 ran, cell2 failed, cell3 should NOT have run. Each attempted
        # cell keeps its own count even though both came from one LLM response.
        assert agent_instance.trace == ["cell1"]
        assert result == "done"
        outputs = [
            event
            for event in agent_instance.event_manager.values()
            if isinstance(event, PythonOutput)
        ]
        assert [event.execution_count for event in outputs] == [1, 2]
        assert "Cell In[2]" in outputs[1].error

    @pytest.mark.asyncio
    async def test_multi_tool_calls_with_return_result_at_end(self):
        """Multi-tool batch with return_result as last call should complete."""

        class TestAgent(Agent, llm=_TEST_LLM):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.x = 5
                self.y = 3

            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute(self) -> dict:
                """Compute sum and product."""
                ...

        # Single response: compute, then return_result
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "",
                    tool_calls=[
                        _tool_call("s = self.x + self.y", call_id="call_1"),
                        _tool_call("p = self.x * self.y", call_id="call_2"),
                        _return_result(call_id="call_return", result={"sum": 8, "product": 15}),
                    ],
                ),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.compute()

        assert result == {"sum": 8, "product": 15}

    @pytest.mark.asyncio
    async def test_multi_tool_calls_variables_persist_within_batch(self):
        """Variables from earlier cells should be available to later cells in same batch."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute(self) -> int:
                """Compute a value using variables."""
                ...

        # Single response: define x, define y using x, return using both
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "",
                    tool_calls=[
                        _tool_call("x = 10", call_id="call_1"),
                        _tool_call("y = x * 2", call_id="call_2"),  # Uses x from cell 1
                        _tool_call("return x + y", call_id="call_3"),  # Uses both: 10 + 20 = 30
                    ],
                ),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.compute()

        assert result == 30


class TestCodeActTurnEvents:
    """Tests for turn event emission in CodeActStrategy.

    Note: Turn events have Role.RUNTIME_EVENT and are emitted but not stored in event_manager.
    We capture them via event_manager.on() event handlers.
    """

    @pytest.mark.asyncio
    async def test_emits_before_and_after_turn_events_on_success(self):
        """Strategy should emit BeforeTurn and AfterTurn on successful execution."""
        from nooa.events import AfterTurn, BeforeTurn

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute(self) -> int:
                """Compute a value."""
                ...

        # Single turn: return_result directly
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "",
                    tool_calls=[
                        _return_result(call_id="call_1", result=42),
                    ],
                ),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)

        # Capture turn events via event handler
        captured_events = []

        def capture_event(event):
            if isinstance(event, (BeforeTurn, AfterTurn)):
                captured_events.append(event)

        agent_instance.event_manager.on("*", capture_event)

        result = await agent_instance.compute()
        assert result == 42

        # Filter captured events
        before_events = [e for e in captured_events if isinstance(e, BeforeTurn)]
        after_events = [e for e in captured_events if isinstance(e, AfterTurn)]

        # Should have exactly one turn (before + after)
        assert len(before_events) == 1, f"Expected 1 BeforeTurn, got {len(before_events)}"
        assert len(after_events) == 1, f"Expected 1 AfterTurn, got {len(after_events)}"

        # Verify BeforeTurn
        assert before_events[0].method_name == "compute"
        assert before_events[0].strategy == "CODEACT"
        assert before_events[0].turn_number == 1

        # Verify AfterTurn
        assert after_events[0].method_name == "compute"
        assert after_events[0].strategy == "CODEACT"
        assert after_events[0].turn_number == 1
        assert after_events[0].is_final is True
        assert after_events[0].success is True
        assert after_events[0].exception_type is None

    @pytest.mark.asyncio
    async def test_turn_number_increments_across_iterations(self):
        """Turn number should increment with each iteration."""
        from nooa.events import BeforeTurn

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute(self) -> int:
                """Compute a value."""
                ...

        # Multiple turns: execute_python then return_result
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "",
                    tool_calls=[
                        _tool_call("x = 21 * 2", call_id="call_1"),
                    ],
                ),
                _resp(
                    "",
                    tool_calls=[
                        _return_result(call_id="call_2", result=42),
                    ],
                ),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)

        # Capture turn events via event handler
        captured_events = []

        def capture_event(event):
            if isinstance(event, BeforeTurn):
                captured_events.append(event)

        agent_instance.event_manager.on("*", capture_event)

        result = await agent_instance.compute()
        assert result == 42

        # Filter captured events
        before_events = [e for e in captured_events if isinstance(e, BeforeTurn)]

        # Should have two turns
        assert len(before_events) == 2
        assert before_events[0].turn_number == 1
        assert before_events[1].turn_number == 2

    @pytest.mark.asyncio
    async def test_inline_return_result_emits_success_event(self):
        """Inline return_result() should emit AfterTurn with success=True."""
        from nooa.events import AfterTurn, BeforeTurn

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute(self) -> int:
                """Compute a value."""
                ...

        # Single turn with inline return_result
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "",
                    tool_calls=[
                        _tool_call("return_result(42)", call_id="call_1"),
                    ],
                ),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)

        # Capture turn events via event handler
        captured_events = []

        def capture_event(event):
            if isinstance(event, (BeforeTurn, AfterTurn)):
                captured_events.append(event)

        agent_instance.event_manager.on("*", capture_event)

        result = await agent_instance.compute()
        assert result == 42

        # Filter captured events
        after_events = [e for e in captured_events if isinstance(e, AfterTurn)]

        # Should have exactly one turn
        assert len(after_events) == 1

        # Verify success
        assert after_events[0].is_final is True
        assert after_events[0].success is True


# =============================================================================
# Tests for get_type_hint_str and get_type_example
# =============================================================================


class TestGetTypeHintStr:
    """Tests for get_type_hint_str() in codeact_errors.

    This function produces human-readable type names for error messages
    shown back to the LLM. Edge cases matter because confusing type names
    lead to confusing error feedback.
    """

    def test_none_returns_value(self):
        assert get_type_hint_str(None) == "value"

    def test_basic_str(self):
        assert get_type_hint_str(str) == "str"

    def test_basic_int(self):
        assert get_type_hint_str(int) == "int"

    def test_basic_float(self):
        assert get_type_hint_str(float) == "float"

    def test_basic_bool(self):
        assert get_type_hint_str(bool) == "bool"

    def test_list_of_str(self):
        assert get_type_hint_str(list[str]) == "list[str]"

    def test_list_of_int(self):
        assert get_type_hint_str(list[int]) == "list[int]"

    def test_list_of_float(self):
        assert get_type_hint_str(list[float]) == "list[float]"

    def test_nested_list(self):
        assert get_type_hint_str(list[list[str]]) == "list[list[str]]"

    def test_bare_list_class(self):
        # Plain `list` (not generic) uses __name__
        assert get_type_hint_str(list) == "list"

    def test_dict_with_args(self):
        assert get_type_hint_str(dict[str, int]) == "dict[str, int]"

    def test_dict_str_str(self):
        assert get_type_hint_str(dict[str, str]) == "dict[str, str]"

    def test_bare_dict_class(self):
        assert get_type_hint_str(dict) == "dict"

    def test_tuple_with_args(self):
        assert get_type_hint_str(tuple[str, int]) == "tuple[str, int]"

    def test_tuple_single_arg(self):
        assert get_type_hint_str(tuple[str]) == "tuple[str]"

    def test_bare_tuple_class(self):
        assert get_type_hint_str(tuple) == "tuple"

    def test_custom_class(self):
        class MyModel:
            pass

        assert get_type_hint_str(MyModel) == "MyModel"

    def test_optional_falls_back_to_str_repr(self):
        # str | None == Optional[str]; the Union origin is not explicitly
        # handled, so we fall through to str(). Document current behaviour.
        result = get_type_hint_str(str | None)
        assert "str" in result  # At minimum the base type is visible

    def test_union_falls_back_to_str_repr(self):
        result = get_type_hint_str(str | int)
        assert "str" in result
        assert "int" in result


class TestGetTypeExample:
    """Tests for get_type_example() in codeact_errors.

    This function produces the example snippet shown in return_result() error
    messages. The LLM uses it to understand the expected call shape, so
    correctness of example content matters.
    """

    def test_list_of_str(self):
        assert get_type_example(list[str]) == 'result=["item1", "item2", "item3"]'

    def test_list_of_int(self):
        assert get_type_example(list[int]) == "result=[1, 2, 3]"

    def test_list_of_float(self):
        assert get_type_example(list[float]) == "result=[1.0, 2.5, 3.14]"

    def test_list_of_unrecognised_inner_type_returns_generic(self):
        # list[bool] inner → not str/int/float → falls to generic
        assert get_type_example(list[bool]) == "result=[...]"

    def test_list_without_args_returns_none(self):
        # bare list (no type args) has no __origin__, so no example is generated
        assert get_type_example(list) is None

    def test_dict(self):
        assert get_type_example(dict[str, str]) == 'result={"key": "value"}'

    def test_dict_any_args(self):
        assert get_type_example(dict[str, int]) == 'result={"key": "value"}'

    def test_str(self):
        assert get_type_example(str) == 'result="your answer"'

    def test_int(self):
        assert get_type_example(int) == "result=42"

    def test_float(self):
        assert get_type_example(float) == "result=3.14"

    def test_bool(self):
        assert get_type_example(bool) == "result=True"

    def test_custom_class_returns_none(self):
        class MyModel:
            pass

        assert get_type_example(MyModel) is None

    def test_none_returns_none(self):
        assert get_type_example(None) is None

    def test_optional_returns_none(self):
        # str | None is not handled; returns None (no example generated)
        assert get_type_example(str | None) is None


# ===========================================================================
# Tool Call Translation (weak model support)
# ===========================================================================


class TestToolCallTranslation:
    """Test that unknown tool calls matching agent methods get translated to execute_python."""

    @pytest.mark.asyncio
    async def test_translate_agent_method_to_execute_python(self):
        """LLM calls an agent method directly as a tool — should be translated."""

        class TestAgent(Agent, llm=_TEST_LLM):
            def get_stock(self, item: str) -> int:
                return {"apples": 10, "oranges": 5}.get(item, 0)

            @strategy(CodeActStrategy(config=CodeActConfig(translate_tool_calls=True)))
            async def check_stock(self, item: str) -> int:
                """Check stock for {item}."""
                ...

        # Turn 1: LLM tries to call get_stock directly as a tool (wrong!)
        # Turn 2: After seeing the execute_python result, LLM returns properly
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="get_stock",
                            arguments=json.dumps({"item": "apples"}),
                        )
                    ],
                ),
                _resp("", tool_calls=[_return_result(result=10)]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.check_stock("apples")
        assert result == 10

    @pytest.mark.asyncio
    async def test_disabled_by_default(self):
        """Without translate_tool_calls=True, unknown tools should error normally."""

        class TestAgent(Agent, llm=_TEST_LLM):
            def get_stock(self, item: str) -> int:
                return {"apples": 10, "oranges": 5}.get(item, 0)

            @strategy(CodeActStrategy(config=CodeActConfig(max_retries=2)))
            async def check_stock(self, item: str) -> int:
                """Check stock for {item}."""
                ...

        # Turn 1: LLM calls get_stock as a tool — should error (not translate)
        # Turn 2: After error, LLM returns properly
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="get_stock",
                            arguments=json.dumps({"item": "apples"}),
                        )
                    ],
                ),
                _resp("", tool_calls=[_return_result(result=10)]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.check_stock("apples")
        assert result == 10

    @pytest.mark.asyncio
    async def test_truly_unknown_tool_still_errors(self):
        """Tool name that doesn't match any method should still error even with translation enabled."""

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(
                CodeActStrategy(config=CodeActConfig(max_retries=2, translate_tool_calls=True))
            )
            async def answer(self) -> int:
                """Return 42."""
                ...

        # Turn 1: LLM calls a completely unknown tool
        # Turn 2: After error feedback, LLM returns properly
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="nonexistent_tool",
                            arguments=json.dumps({"x": 1}),
                        )
                    ],
                ),
                _resp("", tool_calls=[_return_result(result=42)]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.answer()
        assert result == 42

    @pytest.mark.asyncio
    async def test_translate_module_level_function(self):
        """Module-level functions visible in builtins should be translatable."""
        import inspect

        def helper_func(x: int) -> int:
            return x * 2

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(translate_tool_calls=True)))
            async def compute(self, n: int) -> int:
                """Compute double of {n}."""
                ...

        # Inject helper_func into the agent's module so _extract_module_context
        # includes it in the builtins dict (locally-defined functions wouldn't
        # appear in module globals otherwise).
        agent_module = inspect.getmodule(TestAgent)
        agent_module.helper_func = helper_func  # type: ignore[union-attr]

        try:
            # Turn 1: LLM calls helper_func as a tool — translation should
            #   rewrite it to execute_python("result = helper_func(x=21)\nprint(result)")
            # Turn 2: Returns result
            fake_llm = FakeLLMClient(
                scripted_responses=[
                    _resp(
                        "",
                        tool_calls=[
                            ToolCall(
                                id="call_1",
                                name="helper_func",
                                arguments=json.dumps({"x": 21}),
                            )
                        ],
                    ),
                    _resp("", tool_calls=[_return_result(result=42)]),
                ]
            )

            agent_instance = TestAgent(llm=fake_llm)
            result = await agent_instance.compute(21)
            assert result == 42
        finally:
            # Clean up injected function to avoid polluting other tests
            if hasattr(agent_module, "helper_func"):
                delattr(agent_module, "helper_func")  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_translate_async_agent_method_uses_await(self):
        """Async agent methods should be translated with await."""

        class TestAgent(Agent, llm=_TEST_LLM):
            async def fetch_data(self, key: str) -> str:
                return {"a": "alpha", "b": "beta"}.get(key, "unknown")

            @strategy(CodeActStrategy(config=CodeActConfig(translate_tool_calls=True)))
            async def lookup(self, key: str) -> str:
                """Look up {key}."""
                ...

        # Turn 1: LLM calls async fetch_data directly as a tool
        # Turn 2: After seeing the execute_python result, LLM returns properly
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="fetch_data",
                            arguments=json.dumps({"key": "a"}),
                        )
                    ],
                ),
                _resp("", tool_calls=[_return_result(result="alpha")]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.lookup("a")
        assert result == "alpha"

    @pytest.mark.asyncio
    async def test_translate_async_module_function_uses_await(self):
        """Async module-level functions should be translated with await."""

        async def async_helper(x: int) -> int:
            return x * 3

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig(translate_tool_calls=True)))
            async def compute(self, n: int) -> int:
                """Compute triple of {n}."""
                ...

        # Turn 1: LLM calls async_helper as a tool
        # Turn 2: Returns result
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "",
                    tool_calls=[
                        ToolCall(
                            id="call_1",
                            name="async_helper",
                            arguments=json.dumps({"x": 7}),
                        )
                    ],
                ),
                _resp("", tool_calls=[_return_result(result=21)]),
            ]
        )

        agent_instance = TestAgent(llm=fake_llm)
        result = await agent_instance.compute(7)
        assert result == 21
