# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for return type validation in PurePythonStrategy."""

import pytest
from pydantic import BaseModel

from nooa import Agent, strategy
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient


class UserProfile(BaseModel):
    """Test Pydantic model."""

    username: str
    email: str
    age: int


@pytest.mark.asyncio
async def test_pure_python_validation_pydantic_dict_conversion():
    """Test that dict returns are converted to Pydantic models."""
    # LLM returns code that produces a dict (REPL-style)
    code = 'return {"username": "alice", "email": "alice@example.com", "age": 30}'
    llm = FakeLLMClient.with_code_responses([code])

    class TestAgent(Agent, llm=llm):
        @strategy(PurePythonStrategy())
        async def get_user(self) -> UserProfile:
            """Return user profile."""
            ...

    agent_instance = TestAgent()
    result = await agent_instance.get_user()

    # Result should be converted to UserProfile
    assert isinstance(result, UserProfile)
    assert result.username == "alice"
    assert result.email == "alice@example.com"
    assert result.age == 30


@pytest.mark.asyncio
async def test_pure_python_validation_pydantic_error_triggers_retry():
    """Test that Pydantic validation errors trigger regeneration."""
    # First attempt: missing required field, Second attempt: valid (REPL-style)
    llm = FakeLLMClient.with_code_responses(
        [
            'return {"username": "alice", "email": "alice@example.com"}',
            'return {"username": "bob", "email": "bob@example.com", "age": 25}',
        ]
    )

    class TestAgent(Agent, llm=llm):
        @strategy(PurePythonStrategy())
        async def get_user(self) -> UserProfile:
            """Return user profile."""
            ...

    agent_instance = TestAgent()
    result = await agent_instance.get_user()

    assert isinstance(result, UserProfile)
    assert result.username == "bob"
    assert llm.call_count == 2


@pytest.mark.asyncio
async def test_pure_python_validation_string_type():
    """Test that string return types are validated."""
    llm = FakeLLMClient.with_code_responses(['return "Alice"'])

    class TestAgent(Agent, llm=llm):
        @strategy(PurePythonStrategy())
        async def get_name(self) -> str:
            """Return a name."""
            ...

    agent_instance = TestAgent()
    result = await agent_instance.get_name()

    assert isinstance(result, str)
    assert result == "Alice"


@pytest.mark.asyncio
async def test_pure_python_validation_int_type():
    """Test that int return types are validated."""
    llm = FakeLLMClient.with_code_responses(["return 42"])

    class TestAgent(Agent, llm=llm):
        @strategy(PurePythonStrategy())
        async def get_count(self) -> int:
            """Return a count."""
            ...

    agent_instance = TestAgent()
    result = await agent_instance.get_count()

    assert isinstance(result, int)
    assert result == 42


@pytest.mark.asyncio
async def test_pure_python_validation_list_type():
    """Test that list return types are validated."""
    llm = FakeLLMClient.with_code_responses(['return ["apple", "banana", "cherry"]'])

    class TestAgent(Agent, llm=llm):
        @strategy(PurePythonStrategy())
        async def get_items(self) -> list:
            """Return a list of items."""
            ...

    agent_instance = TestAgent()
    result = await agent_instance.get_items()

    assert isinstance(result, list)
    assert len(result) == 3


@pytest.mark.asyncio
async def test_pure_python_validation_dict_type():
    """Test that dict return types are validated."""
    llm = FakeLLMClient.with_code_responses(['return {"key": "value", "enabled": True}'])

    class TestAgent(Agent, llm=llm):
        @strategy(PurePythonStrategy())
        async def get_config(self) -> dict:
            """Return a config dict."""
            ...

    agent_instance = TestAgent()
    result = await agent_instance.get_config()

    assert isinstance(result, dict)
    assert result["key"] == "value"


@pytest.mark.asyncio
async def test_pure_python_validation_wrong_basic_type_triggers_retry():
    """Test that wrong basic types trigger regeneration."""
    # First attempt: returns string instead of int, Second attempt: correct type (REPL-style)
    llm = FakeLLMClient.with_code_responses(['return "not an int"', "return 42"])

    class TestAgent(Agent, llm=llm):
        @strategy(PurePythonStrategy())
        async def get_count(self) -> int:
            """Return a count."""
            ...

    agent_instance = TestAgent()
    result = await agent_instance.get_count()

    assert isinstance(result, int)
    assert result == 42
    assert llm.call_count == 2


@pytest.mark.asyncio
async def test_pure_python_validation_no_annotation_skips():
    """Test that methods without annotations skip validation."""
    llm = FakeLLMClient.with_code_responses(['return {"anything": "goes"}'])

    class TestAgent(Agent, llm=llm):
        @strategy(PurePythonStrategy())
        async def do_something(self):  # No return annotation
            """Do something."""
            ...

    agent_instance = TestAgent()
    result = await agent_instance.do_something()

    # Should return dict without validation
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_pure_python_validation_optional_type():
    """Test that Optional[T] types work correctly."""
    # Test returning None (REPL-style)
    llm = FakeLLMClient.with_code_responses(["return None"])

    class TestAgent(Agent, llm=llm):
        @strategy(PurePythonStrategy())
        async def maybe_name(self) -> str | None:
            """Return name or None."""
            ...

    agent_instance = TestAgent()
    result = await agent_instance.maybe_name()
    assert result is None

    # Test returning actual value (REPL-style)
    llm2 = FakeLLMClient.with_code_responses(['return "Alice"'])

    class TestAgent2(Agent, llm=llm2):
        @strategy(PurePythonStrategy())
        async def maybe_name(self) -> str | None:
            """Return name or None."""
            ...

    agent_instance2 = TestAgent2()
    result2 = await agent_instance2.maybe_name()
    assert result2 == "Alice"
