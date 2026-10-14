# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recovery feedback for tool calls that do not use the Python cell correctly."""

import json
from types import SimpleNamespace

import pytest

from nooa import Agent, hidden, strategy
from nooa.config import CodeActConfig
from nooa.context_blocks import ToolCallEvent
from nooa.events import Error, PythonOutput
from nooa.strategies.codeact import CodeActStrategy
from nooa.strategies.codeact_v2 import CodeActV2
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


class Notes:
    """A small tool object attached to the agent."""

    async def add(self, text: str) -> str:
        """Record a note."""
        return f"added {text}"

    def count(self) -> int:
        """Return the number of notes."""
        return 0


class NotesAgent(Agent, llm=FakeLLMClient()):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.notes = Notes()

    def lookup(self, key: str) -> str:
        """Look up a key."""
        return key

    @hidden
    def secret(self) -> str:
        return "hidden"


def _feedback(name: str, args: dict | None = None, strategy_obj=None) -> str:
    strategy_obj = strategy_obj or CodeActV2()
    runtime = SimpleNamespace(agent=NotesAgent())
    return strategy_obj._unknown_tool_feedback(name, args or {}, runtime)


def _arguments(feedback: str) -> dict:
    line = next(line for line in feedback.splitlines() if line.startswith("Arguments: "))
    return json.loads(line.removeprefix("Arguments: "))


def test_object_attribute_points_to_python_cell_and_doc():
    feedback = _feedback("notes", {"action": "add"})

    assert feedback.startswith("Unknown tool `notes`. Available tools: python_cell.")
    assert "Tool name: python_cell" in feedback
    assert _arguments(feedback) == {"code": "print(doc(self.notes))"}
    assert "self.notes.method_name(...)" in feedback
    assert "Do not call `notes` as a tool again." in feedback


@pytest.mark.parametrize("name", ["notes.add", "self.notes.add"])
def test_async_method_path_shows_equivalent_awaited_call(name):
    feedback = _feedback(name, {"text": "first"})

    assert _arguments(feedback) == {
        "code": "result = await self.notes.add(text='first')\nprint(result)"
    }
    assert "print(doc(self.notes.add))" in feedback


def test_sync_agent_method_is_called_without_await():
    feedback = _feedback("lookup", {"key": "k"})

    assert _arguments(feedback) == {"code": "result = self.lookup(key='k')\nprint(result)"}


def test_large_arguments_are_replaced_with_placeholder():
    feedback = _feedback("lookup", {"key": "x" * 1000})

    assert _arguments(feedback) == {"code": "result = self.lookup(...)\nprint(result)"}


@pytest.mark.parametrize("name", ["bash", "secret", "_private", "notes.missing", "a b"])
def test_unresolved_or_hidden_names_suggest_inspecting_self(name):
    feedback = _feedback(name)

    assert _arguments(feedback) == {"code": "print(doc(self))"}
    assert "does not exist" in feedback
    assert "hidden" not in feedback.split("\n\n", 1)[1]


def test_feedback_uses_strategy_tool_names():
    feedback = _feedback("notes", strategy_obj=CodeActStrategy())

    assert "Available tools: execute_python, return_result." in feedback
    assert "Tool name: execute_python" in feedback


def test_v2_return_result_guidance_is_preserved():
    feedback = _feedback("return_result", {"result": 1})

    assert "return_result is a Python builtin, not a provider tool" in feedback


def test_long_tool_names_are_shortened():
    feedback = _feedback("n" * 500)

    assert "n" * 81 not in feedback


def _call(name: str, arguments: str, call_id: str) -> LLMResponse:
    return LLMResponse(
        parts=(ToolCall(id=call_id, name=name, arguments=arguments),),
        finish_reason="tool_calls",
    )


def _cell(code: str, call_id: str) -> LLMResponse:
    return _call("python_cell", json.dumps({"code": code}), call_id)


@pytest.mark.asyncio
async def test_agent_recovers_after_unknown_tool_feedback():
    llm = FakeLLMClient(
        scripted_responses=[
            _call("notes.add", json.dumps({"text": "first"}), "bad"),
            _cell("return_result(await self.notes.add(text='first'))", "fixed"),
        ]
    )

    class FeedbackAgent(NotesAgent, llm=llm):
        @strategy(CodeActV2(config=CodeActConfig(prefill=None)))
        async def answer(self) -> str:
            """Add a note and return the result."""
            ...

    agent = FeedbackAgent()
    try:
        assert await agent.answer() == "added first"
        bad = next(
            e
            for e in agent.event_manager.values()
            if isinstance(e, ToolCallEvent) and e.tool_call_id == "bad"
        )
        assert bad.result is not None
        assert (
            "result = await self.notes.add(text='first')"
            in json.loads(bad.result.content.split("Arguments: ", 1)[1].split("\n", 1)[0])["code"]
        )
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_misplaced_cell_arguments_name_the_code_parameter():
    llm = FakeLLMClient(
        scripted_responses=[
            _call("python_cell", json.dumps({"command": "ls"}), "bad"),
            _cell("return_result('ok')", "fixed"),
        ]
    )

    class FeedbackAgent(Agent, llm=llm):
        @strategy(CodeActV2(config=CodeActConfig(prefill=None)))
        async def answer(self) -> str:
            """Return ok."""
            ...

    agent = FeedbackAgent()
    try:
        assert await agent.answer() == "ok"
        errors = [e.error for e in agent.event_manager.values() if isinstance(e, PythonOutput)]
        assert any(
            error and "takes a single `code` argument" in error and "`command`" in error
            for error in errors
        )
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_invalid_json_arguments_show_expected_shape():
    llm = FakeLLMClient(
        scripted_responses=[
            _call("python_cell", "print(1)", "bad"),
            _cell("return_result('ok')", "fixed"),
        ]
    )

    class FeedbackAgent(Agent, llm=llm):
        @strategy(CodeActV2(config=CodeActConfig(prefill=None)))
        async def answer(self) -> str:
            """Return ok."""
            ...

    agent = FeedbackAgent()
    try:
        assert await agent.answer() == "ok"
        errors = [e.content for e in agent.event_manager.values() if isinstance(e, Error)]
        assert any(
            'call python_cell with arguments like {"code": "print(1)"}' in error for error in errors
        )
    finally:
        await agent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    [
        {"add": '[{"id": "1", "title": "Tune model"}]'},
        {"title": "Install ffmpeg and tesseract-ocr", "deps": '["2b12c3ab"]'},
    ],
)
async def test_rejected_todo_arguments_reach_model_as_tool_feedback(arguments):
    """Valid JSON with an imagined todo schema must not disappear silently."""
    from nooa.tools.todo import TodoManager

    llm = FakeLLMClient(
        scripted_responses=[
            _call("todo", json.dumps(arguments), "bad-todo"),
            _cell("return_result('recovered')", "fixed"),
        ]
    )

    class FeedbackAgent(Agent, llm=llm):
        def __init__(self):
            super().__init__()
            self.todo = TodoManager()

        @strategy(CodeActV2(config=CodeActConfig(prefill=None)))
        async def answer(self) -> str:
            """Return recovered."""
            ...

    agent = FeedbackAgent()
    try:
        assert await agent.answer() == "recovered"
        feedback = next(
            message
            for message in llm.last_messages
            if message.get("role") == "tool" and message.get("tool_call_id") == "bad-todo"
        )
        assert "Unknown tool `todo`" in feedback["content"]
        assert "print(doc(self.todo))" in feedback["content"]
        assert "self.todo.method_name(...)" in feedback["content"]
        assert "Do not call `todo` as a tool again." in feedback["content"]
    finally:
        await agent.aclose()


@pytest.mark.asyncio
async def test_retry_exhaustion_retains_feedback_for_every_rejected_todo_call():
    """The final rejection must have an error result even at the retry limit."""
    from nooa.errors import GenerationError

    llm = FakeLLMClient(
        scripted_responses=[
            _call("todo", json.dumps({"add": "[]"}), f"bad-{index}") for index in range(10)
        ]
    )

    class FeedbackAgent(Agent, llm=llm):
        @strategy(CodeActV2(config=CodeActConfig(prefill=None, max_retries=10)))
        async def answer(self) -> str:
            """Return a result."""
            ...

    agent = FeedbackAgent()
    try:
        with pytest.raises(GenerationError, match="after 10 errors"):
            await agent.answer()
        rejected = [
            event for event in agent.event_manager.values() if isinstance(event, ToolCallEvent)
        ]
        assert len(rejected) == 10
        assert all(
            event.result is not None and "Unknown tool `todo`" in event.result.content
            for event in rejected
        )
        visible = {
            message["tool_call_id"]
            for message in llm.last_messages
            if message.get("role") == "tool"
        }
        assert visible == {f"bad-{index}" for index in range(9)}
    finally:
        await agent.aclose()
