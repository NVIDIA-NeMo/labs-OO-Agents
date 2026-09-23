# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Python defaults on generation-method calls."""

from __future__ import annotations

import json
from typing import Any

import pytest

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.strategies import CodeActStrategy
from nooa.strategies.current_call import CurrentCall
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


def _execute_result(expression: str) -> LLMResponse:
    """Return one deterministic CodeAct response for the supplied expression."""
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[
            ToolCall(
                id="call-default",
                name="execute_python",
                arguments=json.dumps({"code": f"return_result(result={expression})"}),
            )
        ],
        finish_reason="tool_calls",
    )


class _CaptureCallStrategy(PurePythonStrategy):
    """Capture the effective call while returning its expanded docstring."""

    def __init__(self) -> None:
        super().__init__()
        self.call: CurrentCall | None = None

    async def execute(self, runtime: Any, call: CurrentCall) -> str:
        """Record the call supplied by the runtime."""
        self.call = call
        return call.docstring or ""


def test_current_call_materializes_positional_and_keyword_only_defaults() -> None:
    """CurrentCall exposes omitted defaults while preserving explicit values."""

    def summarize(
        self: object,
        text: str,
        limit: int = 10,
        *,
        style: str = "brief",
        **metadata: Any,
    ) -> str:
        """Summarize text."""
        return text

    omitted = CurrentCall.from_method(summarize, args=("input",), kwargs={})
    explicit = CurrentCall.from_method(
        summarize,
        args=("input", 3),
        kwargs={"style": "detailed", "source": "test"},
    )

    assert omitted.kwargs == {
        "text": "input",
        "limit": 10,
        "style": "brief",
        "metadata": {},
    }
    assert explicit.kwargs == {
        "text": "input",
        "limit": 3,
        "style": "detailed",
        "source": "test",
        "metadata": {"source": "test"},
    }


def test_variadic_positionals_do_not_replace_keyword_only_defaults() -> None:
    """Variadic values bind as a tuple and cannot replace keyword-only defaults."""

    def render(self: object, value: str, *parts: str, suffix: str = "!") -> str:
        """Render a value."""
        return value

    call = CurrentCall.from_method(render, args=("a", "b", "c"), kwargs={})

    assert call.args == ("a", "b", "c")
    assert call.kwargs == {"value": "a", "parts": ("b", "c"), "suffix": "!"}
    assert call.bound_parameters() == {
        "value": "a",
        "parts": ("b", "c"),
        "suffix": "!",
    }


def test_variadic_keyword_collision_preserves_both_values() -> None:
    """A keyword matching *args remains available through the **kwargs local."""

    def render(
        self: object,
        value: str,
        *parts: str,
        suffix: str = "!",
        **metadata: str,
    ) -> str:
        """Render a value."""
        return value

    call = CurrentCall.from_method(
        render,
        args=("a", "b"),
        kwargs={"parts": "keyword", "source": "test"},
    )

    assert call.kwargs == {
        "value": "a",
        "parts": ("b",),
        "suffix": "!",
        "source": "test",
        "metadata": {"parts": "keyword", "source": "test"},
    }
    assert call.bound_parameters() == {
        "value": "a",
        "parts": ("b",),
        "suffix": "!",
        "metadata": {"parts": "keyword", "source": "test"},
    }


def test_positional_only_keyword_collision_preserves_default_and_metadata() -> None:
    """A same-named keyword cannot replace an omitted positional-only default."""

    def render(self: object, value: str = "default", /, **metadata: str) -> str:
        """Render a value."""
        return value

    call = CurrentCall.from_method(render, kwargs={"value": "keyword"})

    assert call.kwargs == {
        "value": "default",
        "metadata": {"value": "keyword"},
    }
    assert call.bound_parameters() == {
        "value": "default",
        "metadata": {"value": "keyword"},
    }


@pytest.mark.asyncio
async def test_agent_default_reaches_docstring_and_call_context() -> None:
    """Agent calls use an omitted default for expansion and strategy inputs."""
    capture = _CaptureCallStrategy()

    class GreetingAgent(Agent, llm=FakeLLMClient()):
        @strategy(capture)
        async def greet(self, greeting: str = "hello") -> str:
            """Return {greeting}."""
            ...

    result = await GreetingAgent().greet()

    assert result == "Return hello."
    assert capture.call is not None
    assert capture.call.kwargs == {"greeting": "hello"}


@pytest.mark.asyncio
async def test_agent_default_is_available_to_generated_code() -> None:
    """CodeAct can reference an omitted default without a recovery turn."""
    llm = FakeLLMClient(scripted_responses=[_execute_result("greeting")])

    class GreetingAgent(Agent, llm=llm):
        @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=1)))
        async def greet(self, greeting: str = "hello") -> str:
            """Return the supplied greeting."""
            ...

    assert await GreetingAgent().greet() == "hello"
    assert llm.call_count == 1


@pytest.mark.asyncio
async def test_variadic_agent_keeps_keyword_only_default_in_generated_code() -> None:
    """CodeAct sees every variadic input and the following keyword-only default."""
    llm = FakeLLMClient(scripted_responses=[_execute_result("''.join(parts) + suffix")])

    class RenderAgent(Agent, llm=llm):
        @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=1)))
        async def render(self, value: str, *parts: str, suffix: str = "!") -> str:
            """Join the requested parts and suffix."""
            ...

    assert await RenderAgent().render("a", "b", "c") == "bc!"
    assert llm.call_count == 1


@pytest.mark.asyncio
async def test_variadic_keyword_collision_is_available_to_generated_code() -> None:
    """CodeAct receives both *args and a same-named value captured by **kwargs."""
    llm = FakeLLMClient(
        scripted_responses=[_execute_result("parts[0] + metadata['parts'] + source + suffix")]
    )

    class RenderAgent(Agent, llm=llm):
        @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=1)))
        async def render(
            self,
            value: str,
            *parts: str,
            suffix: str = "!",
            **metadata: str,
        ) -> str:
            """Join the variadic inputs and metadata."""
            ...

    assert await RenderAgent().render("a", "b", parts="c", source="d") == "bcd!"
    assert llm.call_count == 1


@pytest.mark.asyncio
async def test_standalone_default_is_available_to_generated_code() -> None:
    """Standalone generation functions receive defaults through their adapter."""
    llm = FakeLLMClient(scripted_responses=[_execute_result("greeting")])

    @strategy(
        CodeActStrategy(config=CodeActConfig(max_iterations=1)),
        llm=llm,
    )
    async def greet(greeting: str = "hello") -> str:
        """Return the supplied greeting."""
        ...

    assert await greet() == "hello"
    assert llm.call_count == 1


@pytest.mark.asyncio
async def test_standalone_variadic_inputs_are_available_to_generated_code() -> None:
    """Standalone CodeAct receives the complete variadic tuple and later default."""
    llm = FakeLLMClient(scripted_responses=[_execute_result("''.join(parts) + suffix")])

    @strategy(
        CodeActStrategy(config=CodeActConfig(max_iterations=1)),
        llm=llm,
    )
    async def render(value: str, *parts: str, suffix: str = "!") -> str:
        """Join the requested parts and suffix."""
        ...

    assert await render("a", "b", "c") == "bc!"
    assert llm.call_count == 1
