# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for return type validation in PurePythonStrategy.

Regression tests for parameterized generic return type validation.
See: experiments/capability_eval/agents/capability_wrappers.py RouterTestWrapper.process()
which expects list[str] but LLM may return dict.

The fix validates composed types like list[str], dict[str, int] properly.
"""

from dataclasses import dataclass

import pytest
from pydantic import BaseModel

from nooa import Agent, strategy
from nooa.errors import GenerationError
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse


def _resp(content: str) -> LLMResponse:
    """Create a test LLM response with the given content."""
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": content},
    )


class TestReturnTypeValidation:
    """Tests for return type validation with parameterized generics."""

    @pytest.mark.asyncio
    async def test_rejects_dict_when_list_expected(self):
        """
        Regression test: process() expects list[str] but LLM returns dict.

        The framework should reject this and retry until max_retries is exhausted.

        Found in RouterTestWrapper.process() where LLM returned:
        {
          "agent": "Analyzer",
          "params": {"values": [1, 2, 3, 4, 5], "type": "numbers"},
          "result": {"sum": 15, "mean": 3, "max": 5, "min": 1, ...}
        }

        When it should have returned: ["Analyzer"]
        """
        # The LLM generates code that returns a dict instead of list[str]
        wrong_return_code = """
result = {
    "agent": "Analyzer",
    "params": {
        "values": [1, 2, 3, 4, 5],
        "type": "numbers"
    },
    "result": {
        "sum": 15,
        "mean": 3,
        "max": 5,
        "min": 1,
        "count": 5,
        "type": "numbers"
    }
}
return result
"""

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(wrong_return_code.strip()),
                _resp(wrong_return_code.strip()),
                _resp(wrong_return_code.strip()),
            ]
        )

        class RouterTestAgent(Agent, llm=fake_llm):
            """Wrapper for router tests."""

            @strategy(PurePythonStrategy())
            async def process(self, user_message: str, data: dict) -> list[str]:
                """Route the request to appropriate specialist agent(s).

                Return a list of agent names called: ["Analyzer"] or ["Transformer", "Validator"]
                """
                ...

        agent_instance = RouterTestAgent()

        # Should raise GenerationError after exhausting retries due to type mismatch
        with pytest.raises(GenerationError) as exc_info:
            await agent_instance.process("analyze this", {"values": [1, 2, 3, 4, 5]})

        assert "errors" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_rejects_wrong_element_type_in_list(self):
        """list[str] should reject list containing integers."""
        wrong_element_code = """
return [1, 2, 3]  # Should be list[str], not list[int]
"""

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(wrong_element_code.strip()),
                _resp(wrong_element_code.strip()),
                _resp(wrong_element_code.strip()),
            ]
        )

        class TestAgent(Agent, llm=fake_llm):
            @strategy(PurePythonStrategy())
            async def get_names(self) -> list[str]:
                """Return a list of names."""
                ...

        agent_instance = TestAgent()

        with pytest.raises(GenerationError):
            await agent_instance.get_names()

    @pytest.mark.asyncio
    async def test_correct_list_return_passes(self):
        """Verify correct return type works fine."""
        correct_return_code = """
# Route to Analyzer
return ["Analyzer"]
"""

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(correct_return_code.strip()),
            ]
        )

        class RouterTestAgent(Agent, llm=fake_llm):
            """Wrapper for router tests."""

            @strategy(PurePythonStrategy())
            async def process(self, user_message: str, data: dict) -> list[str]:
                """Route the request to appropriate specialist agent(s).

                Return a list of agent names called: ["Analyzer"] or ["Transformer", "Validator"]
                """
                ...

        agent_instance = RouterTestAgent()
        result = await agent_instance.process("analyze this", {"values": [1, 2, 3, 4, 5]})

        assert isinstance(result, list)
        assert result == ["Analyzer"]
        for item in result:
            assert isinstance(item, str)


class TestNoneReturnTypeMismatch:
    """Tests for None return value when type doesn't allow None.

    Generation methods that have a non-optional return type (e.g., `-> str`)
    must reject `None` as a return value. Only types that include None
    (e.g., `str | None`, `Optional[str]`) should accept None.

    The validation is performed by ReturnValueValidator.validate() in generated_code.py.
    """

    @pytest.mark.asyncio
    async def test_rejects_none_when_str_expected(self):
        """
        Method declares `-> str` but LLM returns `None`.

        The framework should reject this and retry, raising GenerationError
        after exhausting retries.
        """
        # LLM generates code that returns None instead of str
        wrong_return_code = "return None"

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(wrong_return_code),
                _resp(wrong_return_code),
                _resp(wrong_return_code),
            ]
        )

        class TestAgent(Agent, llm=fake_llm):
            """Agent with str return type."""

            @strategy(PurePythonStrategy())
            async def get_name(self) -> str:
                """Return a name string."""
                ...

        agent_instance = TestAgent()

        # Should raise GenerationError due to None not being allowed for str return type
        with pytest.raises(GenerationError):
            await agent_instance.get_name()

    @pytest.mark.asyncio
    async def test_rejects_none_when_int_expected(self):
        """
        Method declares `-> int` but LLM returns `None`.

        The framework should reject this and retry.
        """
        wrong_return_code = "return None"

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(wrong_return_code),
                _resp(wrong_return_code),
                _resp(wrong_return_code),
            ]
        )

        class TestAgent(Agent, llm=fake_llm):
            @strategy(PurePythonStrategy())
            async def get_count(self) -> int:
                """Return a count."""
                ...

        agent_instance = TestAgent()

        with pytest.raises(GenerationError):
            await agent_instance.get_count()

    @pytest.mark.asyncio
    async def test_rejects_none_when_list_expected(self):
        """
        Method declares `-> list[str]` but LLM returns `None`.

        The framework should reject this and retry.
        """
        wrong_return_code = "return None"

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(wrong_return_code),
                _resp(wrong_return_code),
                _resp(wrong_return_code),
            ]
        )

        class TestAgent(Agent, llm=fake_llm):
            @strategy(PurePythonStrategy())
            async def get_items(self) -> list[str]:
                """Return a list of items."""
                ...

        agent_instance = TestAgent()

        with pytest.raises(GenerationError):
            await agent_instance.get_items()

    @pytest.mark.asyncio
    async def test_allows_none_when_optional_str(self):
        """
        Method declares `-> str | None` and LLM returns `None`.

        This should be allowed since Optional types accept None.
        """
        correct_return_code = "return None"

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(correct_return_code),
            ]
        )

        class TestAgent(Agent, llm=fake_llm):
            @strategy(PurePythonStrategy())
            async def maybe_name(self) -> str | None:
                """Return a name or None."""
                ...

        agent_instance = TestAgent()
        result = await agent_instance.maybe_name()

        # Should succeed and return None
        assert result is None

    @pytest.mark.asyncio
    async def test_rejects_none_when_pydantic_model_expected(self):
        """
        Method declares `-> PydanticModel` but LLM returns `None`.

        The framework should reject this and retry.
        """

        class UserProfile(BaseModel):
            name: str
            age: int

        wrong_return_code = "return None"

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(wrong_return_code),
                _resp(wrong_return_code),
                _resp(wrong_return_code),
            ]
        )

        class TestAgent(Agent, llm=fake_llm):
            @strategy(PurePythonStrategy())
            async def get_user(self) -> UserProfile:
                """Return a user profile."""
                ...

        agent_instance = TestAgent()

        with pytest.raises(GenerationError):
            await agent_instance.get_user()

    @pytest.mark.asyncio
    async def test_rejects_none_when_dataclass_expected(self):
        """
        Method declares `-> Dataclass` but LLM returns `None`.

        The framework should reject this and retry.
        """

        @dataclass
        class Point:
            x: int
            y: int

        wrong_return_code = "return None"

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(wrong_return_code),
                _resp(wrong_return_code),
                _resp(wrong_return_code),
            ]
        )

        class TestAgent(Agent, llm=fake_llm):
            @strategy(PurePythonStrategy())
            async def get_point(self) -> Point:
                """Return a point."""
                ...

        agent_instance = TestAgent()

        with pytest.raises(GenerationError):
            await agent_instance.get_point()

    @pytest.mark.asyncio
    async def test_rejects_none_when_dict_with_types_expected(self):
        """
        Method declares `-> dict[str, int]` but LLM returns `None`.

        The framework should reject this and retry.
        """
        wrong_return_code = "return None"

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(wrong_return_code),
                _resp(wrong_return_code),
                _resp(wrong_return_code),
            ]
        )

        class TestAgent(Agent, llm=fake_llm):
            @strategy(PurePythonStrategy())
            async def get_counts(self) -> dict[str, int]:
                """Return counts by name."""
                ...

        agent_instance = TestAgent()

        with pytest.raises(GenerationError):
            await agent_instance.get_counts()

    @pytest.mark.asyncio
    async def test_rejects_none_when_bool_expected(self):
        """
        Method declares `-> bool` but LLM returns `None`.

        The framework should reject this and retry.
        Note: This is important because None is falsy like False.
        """
        wrong_return_code = "return None"

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(wrong_return_code),
                _resp(wrong_return_code),
                _resp(wrong_return_code),
            ]
        )

        class TestAgent(Agent, llm=fake_llm):
            @strategy(PurePythonStrategy())
            async def is_valid(self) -> bool:
                """Return whether something is valid."""
                ...

        agent_instance = TestAgent()

        with pytest.raises(GenerationError):
            await agent_instance.is_valid()

    @pytest.mark.asyncio
    async def test_rejects_none_when_tuple_expected(self):
        """
        Method declares `-> tuple[int, str]` but LLM returns `None`.

        The framework should reject this and retry.
        """
        wrong_return_code = "return None"

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(wrong_return_code),
                _resp(wrong_return_code),
                _resp(wrong_return_code),
            ]
        )

        class TestAgent(Agent, llm=fake_llm):
            @strategy(PurePythonStrategy())
            async def get_pair(self) -> tuple[int, str]:
                """Return an id-name pair."""
                ...

        agent_instance = TestAgent()

        with pytest.raises(GenerationError):
            await agent_instance.get_pair()

    @pytest.mark.asyncio
    async def test_allows_none_when_optional_pydantic_model(self):
        """
        Method declares `-> PydanticModel | None` and LLM returns `None`.

        This should be allowed since Optional types accept None.
        """

        class UserProfile(BaseModel):
            name: str
            age: int

        correct_return_code = "return None"

        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(correct_return_code),
            ]
        )

        class TestAgent(Agent, llm=fake_llm):
            @strategy(PurePythonStrategy())
            async def maybe_user(self) -> UserProfile | None:
                """Return a user profile or None."""
                ...

        agent_instance = TestAgent()
        result = await agent_instance.maybe_user()

        # Should succeed and return None
        assert result is None
