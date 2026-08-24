# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the experimental single-tool CodeAct strategy."""

import json
from typing import Any, cast

import pytest

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.context_blocks import ToolCallEvent
from nooa.events import PythonOutput
from nooa.strategies.codeact_experimental import CodeActExperimental
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


def _execute_python(code: str, call_id: str = "call_1") -> ToolCall:
    return ToolCall(
        id=call_id,
        name="execute_python",
        arguments=json.dumps({"code": code}),
    )


def _response(code: str, call_id: str = "call_1") -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[_execute_python(code, call_id)],
        finish_reason="tool_calls",
        assistant_message={"role": "assistant", "content": ""},
    )


@pytest.mark.asyncio
async def test_explicit_return_completes_with_only_execute_python_tool():
    fake_llm = FakeLLMClient(scripted_responses=[_response("return 42")])

    class TestAgent(Agent, llm=fake_llm):
        @strategy(CodeActExperimental(config=CodeActConfig(prefill=None)))
        async def answer(self) -> int:
            """Return an integer."""
            ...

    agent = TestAgent()
    assert await agent.answer() == 42
    assert [tool.name for tool in fake_llm.last_tools or []] == ["execute_python"]
    system_prompt = "\n".join(
        str(message.get("content", ""))
        for message in fake_llm.last_messages
        if message.get("role") == "system"
    )
    assert "`return_result()`" in system_prompt
    assert "`return_result(value)`" in system_prompt
    assert "prefer it for independent work" in system_prompt
    assert "Await `delegate(...)` only" in system_prompt
    assert "method defines a background-wait result" in system_prompt
    assert "Your two tools" not in system_prompt
    completion_events = [
        event
        for event in agent.event_manager.values()
        if isinstance(event, ToolCallEvent) and event.name == "return_result"
    ]
    assert len(completion_events) == 1
    assert completion_events[0].metadata["synthetic_type"] == "codeact_inline_return"


@pytest.mark.asyncio
async def test_trailing_string_is_suppressed_and_does_not_complete():
    fake_llm = FakeLLMClient(
        scripted_responses=[
            _response("'working notes'", "call_1"),
            _response("return 'done'", "call_2"),
        ]
    )

    class TestAgent(Agent, llm=fake_llm):
        @strategy(CodeActExperimental(config=CodeActConfig(prefill=None)))
        async def answer(self) -> str:
            """Return a string."""
            ...

    agent = TestAgent()
    assert await agent.answer() == "done"
    outputs = [event for event in agent.event_manager.values() if isinstance(event, PythonOutput)]
    assert len(outputs) == 2
    assert outputs[0].value is None
    assert outputs[0].explicit_return is False
    assert outputs[1].value == "done"
    assert outputs[1].explicit_return is True


def test_prompt_and_execution_context_advertise_inline_return_result():
    strategy_instance = CodeActExperimental(config=CodeActConfig(prefill=None))
    assert "return_result" in strategy_instance._always_available_text()
    sentinel = object()
    assert strategy_instance._strategy_builtins(sentinel) == {"return_result": sentinel}
    assert strategy_instance._available_tool_names() == "execute_python"


def test_compatibility_factory_returns_supported_strategy_without_warning():
    from nooa.experimental import CodeActExperimental as factory
    from nooa.strategies import CodeActExperimental as supported

    instance = factory(config=CodeActConfig(prefill=None))
    assert isinstance(instance, supported)


@pytest.mark.asyncio
async def test_return_result_is_available_inside_execute_python_cells():
    fake_llm = FakeLLMClient(scripted_responses=[_response("return_result(41)", "call_1")])

    class TestAgent(Agent, llm=fake_llm):
        @strategy(CodeActExperimental(config=CodeActConfig(prefill=None)))
        async def answer(self) -> int:
            """Return an integer."""
            ...

    agent = TestAgent()
    assert await agent.answer() == 41
    outputs = [event for event in agent.event_manager.values() if isinstance(event, PythonOutput)]
    assert len(outputs) == 1
    # The signal carries the submitted value to the method result; unlike an
    # explicit Python return, it is not also a cell display value.
    assert outputs[0].value is None
    assert outputs[0].error == ""
    completion_events = [
        event
        for event in agent.event_manager.values()
        if isinstance(event, ToolCallEvent) and event.name == "return_result"
    ]
    assert len(completion_events) == 1
    assert completion_events[0].metadata["synthetic_type"] == "codeact_inline_return"


@pytest.mark.asyncio
async def test_execution_context_lists_initial_locals():
    fake_llm = FakeLLMClient(scripted_responses=[_response("return_result(question)")])

    class TestAgent(Agent, llm=fake_llm):
        @strategy(CodeActExperimental(config=CodeActConfig(prefill=None)))
        async def answer(self, question: str) -> str:
            """Return the question."""
            ...

    agent = TestAgent()
    session_locals = {"prior_value": 7}
    answer = cast(Any, agent.answer)
    assert await answer("hello", _session_locals=session_locals) == "hello"
    rendered_context = "\n".join(
        str(message.get("content", "")) for message in fake_llm.last_messages
    )
    locals_block = rendered_context.split("## Locals", 1)[1]
    assert "Available in the next cell:" in locals_block
    assert "- `_call`" not in locals_block
    assert "- `Out`" in locals_block
    assert "- `prior_value`" in locals_block
    assert "- `question`" in locals_block


@pytest.mark.asyncio
async def test_execution_context_lists_locals_created_by_previous_cells():
    fake_llm = FakeLLMClient(
        scripted_responses=[
            _response("working_value = question.upper()", "call_1"),
            _response("return_result(working_value)", "call_2"),
        ]
    )

    class TestAgent(Agent, llm=fake_llm):
        @strategy(CodeActExperimental(config=CodeActConfig(prefill=None)))
        async def answer(self, question: str) -> str:
            """Uppercase the question."""
            ...

    agent = TestAgent()
    assert await agent.answer("hello") == "HELLO"
    rendered_context = "\n".join(
        str(message.get("content", "")) for message in fake_llm.last_messages
    )
    locals_block = rendered_context.split("## Locals", 1)[1]
    assert "- `question`" in locals_block
    assert "- `working_value`" in locals_block
