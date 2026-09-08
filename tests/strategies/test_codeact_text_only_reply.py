# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for append-only, extensible CodeAct text-only recovery."""

import json

import pytest

from nooa import (
    Agent,
    CodeActStrategy,
    TextOnlyResponseAction,
    return_text_as_result,
    strategy,
)
from nooa.config import CodeActConfig
from nooa.context_blocks import ToolCallEvent
from nooa.errors import GenerationError
from nooa.events import LLMOutput, PythonOutput, TextOnlyReply
from nooa.runtime.event_manager import EventManager
from nooa.runtime.harness_metrics import HarnessMetrics
from nooa.storage import SQLiteStorageManager
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

_TEST_LLM = FakeLLMClient()


def _resp(content="", tool_calls=None, finish_reason=None):
    if finish_reason is None:
        finish_reason = "tool_calls" if tool_calls else "stop"
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        assistant_message={"role": "assistant", "content": content},
    )


def _ret(value, call_id="c_ret"):
    return ToolCall(
        id=call_id,
        name="return_result",
        arguments=json.dumps({"result": value}),
    )


def _events(agent, event_type):
    return [event for event in agent.event_manager.values() if isinstance(event, event_type)]


@pytest.mark.asyncio
async def test_default_preserves_text_adds_error_and_retries():
    class TestAgent(Agent, llm=_TEST_LLM):
        @strategy(CodeActStrategy(config=CodeActConfig(max_retries=5, max_iterations=10)))
        async def my_task(self) -> dict:
            """Return a dict."""
            ...

    fake_llm = FakeLLMClient(
        scripted_responses=[
            _resp("I think the answer is ready."),
            _resp(tool_calls=[_ret({"ok": True})]),
        ]
    )
    agent = TestAgent(llm=fake_llm)

    assert await agent.my_task() == {"ok": True}

    outputs = _events(agent, LLMOutput)
    assert [event.content for event in outputs] == ["I think the answer is ready."]

    diagnostics = _events(agent, TextOnlyReply)
    assert len(diagnostics) == 1
    assert diagnostics[0].content == "I think the answer is ready."
    assert diagnostics[0].finish_reason == "stop"
    assert diagnostics[0].handler == "retry_text_only_response"
    assert diagnostics[0].action == "retry"

    corrections = [
        event
        for event in agent.event_manager.values()
        if event.event_type == "Error" and "no tool call" in event.content
    ]
    assert len(corrections) == 1
    events = agent.event_manager.values()
    assert events.index(outputs[0]) < events.index(corrections[0])

    # The provider sees the exact assistant text followed by user feedback.
    assert any(
        message.get("role") == "assistant"
        and message.get("content") == "I think the answer is ready."
        for message in fake_llm.last_messages
    )
    assert not any(message.get("tool_calls") for message in fake_llm.last_messages)


@pytest.mark.asyncio
async def test_return_text_as_result_is_opt_in():
    class TestAgent(Agent, llm=_TEST_LLM):
        @strategy(CodeActStrategy(on_text_only=return_text_as_result))
        async def my_task(self) -> str:
            """Return a string."""
            ...

    agent = TestAgent(llm=FakeLLMClient(scripted_responses=[_resp("done")]))

    assert await agent.my_task() == "done"
    assert [event.content for event in _events(agent, LLMOutput)] == ["done"]
    assert _events(agent, ToolCallEvent) == []
    diagnostic = _events(agent, TextOnlyReply)[0]
    assert diagnostic.handler == "return_text_as_result"
    assert diagnostic.action == "return_result"


@pytest.mark.asyncio
async def test_default_does_not_treat_valid_string_as_result():
    class TestAgent(Agent, llm=_TEST_LLM):
        @strategy(CodeActStrategy())
        async def my_task(self) -> str:
            """Return a string."""
            ...

    fake_llm = FakeLLMClient(scripted_responses=[_resp("prose"), _resp(tool_calls=[_ret("done")])])
    agent = TestAgent(llm=fake_llm)

    assert await agent.my_task() == "done"
    assert [event.content for event in _events(agent, LLMOutput)] == ["prose"]
    assert any(event.event_type == "Error" for event in agent.event_manager.values())


@pytest.mark.asyncio
async def test_default_preserves_empty_stop_without_replaying_empty_assistant_message():
    class TestAgent(Agent, llm=_TEST_LLM):
        @strategy(CodeActStrategy())
        async def my_task(self) -> dict:
            """Return a dict."""
            ...

    fake_llm = FakeLLMClient(scripted_responses=[_resp(""), _resp(tool_calls=[_ret({"ok": True})])])
    agent = TestAgent(llm=fake_llm)

    assert await agent.my_task() == {"ok": True}
    assert [event.content for event in _events(agent, LLMOutput)] == [""]
    assert [event.content for event in _events(agent, TextOnlyReply)] == [""]
    assert not any(
        message.get("role") == "assistant" and not message.get("content")
        for message in fake_llm.last_messages
    )


@pytest.mark.asyncio
async def test_callback_can_synthesize_execute_python_without_replacing_output():
    def execute_text(context):
        return TextOnlyResponseAction.tool_calls(
            ToolCall(
                id="synthetic_cell",
                name="execute_python",
                arguments=json.dumps({"code": f"print({context.content!r})"}),
            )
        )

    class TestAgent(Agent, llm=_TEST_LLM):
        @strategy(CodeActStrategy(on_text_only=execute_text))
        async def my_task(self) -> str:
            """Return a string."""
            ...

    fake_llm = FakeLLMClient(scripted_responses=[_resp("hello"), _resp(tool_calls=[_ret("done")])])
    agent = TestAgent(llm=fake_llm)

    assert await agent.my_task() == "done"
    assert [event.content for event in _events(agent, LLMOutput)] == ["hello"]
    assert [event.tool_call_id for event in _events(agent, ToolCallEvent)] == [
        "synthetic_cell",
        "c_ret",
    ]
    python_output = _events(agent, PythonOutput)[0]
    assert python_output.tool_call_id == "synthetic_cell"
    assert python_output.stdout == "hello\n"
    diagnostic = _events(agent, TextOnlyReply)[0]
    assert diagnostic.handler.endswith("execute_text")
    assert diagnostic.action == "tool_calls"


@pytest.mark.asyncio
async def test_async_callback_is_supported():
    async def return_text(context):
        return TextOnlyResponseAction.return_result(context.content.upper())

    class TestAgent(Agent, llm=_TEST_LLM):
        @strategy(CodeActStrategy(on_text_only=return_text))
        async def my_task(self) -> str:
            """Return a string."""
            ...

    agent = TestAgent(llm=FakeLLMClient(scripted_responses=[_resp("done")]))
    assert await agent.my_task() == "DONE"


@pytest.mark.asyncio
async def test_callback_can_return_non_string_result(monkeypatch):
    def return_integer(context):
        return TextOnlyResponseAction.return_result(42)

    class TestAgent(Agent, llm=_TEST_LLM):
        @strategy(CodeActStrategy(on_text_only=return_integer))
        async def my_task(self) -> int:
            """Return an integer."""
            ...

    metrics = HarnessMetrics()
    monkeypatch.setattr("nooa.strategies.codeact.get_harness_metrics", lambda: metrics)
    agent = TestAgent(llm=FakeLLMClient(scripted_responses=[_resp("forty-two")]))

    assert await agent.my_task() == 42
    assert metrics.stop_to_return_result_count == 1
    assert metrics.stop_to_return_result_previews == []


@pytest.mark.asyncio
async def test_text_only_backstop_still_aborts():
    class TestAgent(Agent, llm=_TEST_LLM):
        @strategy(
            CodeActStrategy(
                config=CodeActConfig(
                    max_retries=10,
                    max_iterations=10,
                    max_consecutive_text_only=3,
                )
            )
        )
        async def my_task(self) -> dict:
            """Return a dict."""
            ...

    agent = TestAgent(
        llm=FakeLLMClient(scripted_responses=[_resp("still chatting") for _ in range(4)])
    )
    with pytest.raises(GenerationError, match="plain text without a tool call"):
        await agent.my_task()

    diagnostics = _events(agent, TextOnlyReply)
    assert len(diagnostics) == 3
    assert [event.consecutive_text_only for event in diagnostics] == [1, 2, 3]


def test_text_only_diagnostic_is_not_model_visible():
    from nooa.context_blocks.roles import Role

    assert TextOnlyReply._role == Role.METADATA


@pytest.mark.asyncio
async def test_text_only_output_survives_sqlite_resume(tmp_path):
    class TestAgent(Agent, llm=_TEST_LLM):
        @strategy(CodeActStrategy())
        async def my_task(self) -> dict:
            """Return a dict."""
            ...

    db_path = tmp_path / "text-only-recovery.db"
    storage = SQLiteStorageManager(db_path)
    fake_llm = FakeLLMClient(
        scripted_responses=[
            _resp("I should have used a tool."),
            _resp(tool_calls=[_ret({"ok": True})]),
        ]
    )
    agent = TestAgent(llm=fake_llm, storage=storage)

    assert await agent.my_task() == {"ok": True}
    storage.close()

    reopened = SQLiteStorageManager(db_path)
    try:
        resumed = EventManager(backend=reopened.event_backend).values()
        assert [event.content for event in resumed if isinstance(event, LLMOutput)] == [
            "I should have used a tool."
        ]
        diagnostics = [event for event in resumed if isinstance(event, TextOnlyReply)]
        assert len(diagnostics) == 1
        assert diagnostics[0].action == "retry"
    finally:
        reopened.close()
