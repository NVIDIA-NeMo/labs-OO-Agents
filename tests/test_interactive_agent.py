# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for nooa.interactive — the dispatcher-driven agent base.

Deeper behavioral coverage (dispatcher loop, snapshot restore, echo hooks)
lives with the TUI package, whose BaseTUIAgent subclasses this. These tests
pin the core contract the ARC-AGI-3 example and other hosts rely on.
"""

import json

import pytest
from pydantic import ValidationError

from nooa import hidden, strategy
from nooa.events import PythonOutput, ResultStatus
from nooa.interactive import (
    AgentMessage,
    AgentVars,
    Done,
    InteractiveAgent,
    NeedInput,
    RespondReason,
    RespondResult,
    SummarizationConfig,
    Waiting,
    install_summarizer,
)
from nooa.strategies import CodeActStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


class _Host(InteractiveAgent, llm=FakeLLMClient()):
    """Minimal InteractiveAgent subclass standing in for a host-driven agent."""


@pytest.fixture
def agent():
    return _Host(llm=FakeLLMClient())


def test_declares_only_the_user_channel(agent):
    """Being dispatcher-driven implies a human feeding it, and nothing more.

    Hosts declare whatever else they need. slash_commands and system_messages
    are coding-host concepts and live on CodingAgent — see
    packages/nooa-cli/tests/test_coding_agent.py.
    """
    assert agent.queue_manager.channels().keys() == {"user_messages"}
    # Reader facade exposed under the public name; producer side hidden.
    assert agent.user_messages is agent._user_messages_in.reader


async def test_queue_roundtrip(agent):
    agent._user_messages_in.put("hello")
    assert await agent.user_messages.get() == "hello"


def test_persistent_vars_proxy(agent):
    agent.v.cursor = 3
    assert agent.v.cursor == 3
    assert "cursor" in agent.v
    assert agent.vars["cursor"] == 3
    del agent.v.cursor
    with pytest.raises(AttributeError):
        _ = agent.v.cursor
    assert isinstance(agent.v, AgentVars)


def test_message_records_event_and_renders(agent):
    rendered: list[str] = []
    agent._render_message = lambda text, **kw: rendered.append(text)
    agent.message("**hi**")
    assert rendered == ["**hi**"]
    events = [e for e in agent.event_manager.values() if isinstance(e, AgentMessage)]
    assert len(events) == 1
    assert events[0].content == "**hi**"


def test_respond_result_requires_explanation():
    result = RespondResult(kind=RespondReason.DONE, explanation="did the thing")
    assert result.kind is RespondReason.DONE
    with pytest.raises(ValidationError):
        RespondResult(kind=RespondReason.DONE, explanation="   ")


def test_turn_results_require_their_text():
    assert Done(explanation=" finished ").explanation == "finished"
    assert Waiting(explanation="job ci-42").explanation == "job ci-42"
    assert NeedInput(question="Which branch?", options=["main", "dev"]).options == ["main", "dev"]
    with pytest.raises(ValidationError):
        Done(explanation="  ")
    with pytest.raises(ValidationError):
        Waiting(explanation="")
    with pytest.raises(ValidationError):
        NeedInput(question=" ")


def test_need_input_takes_options_or_schema_not_both():
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}}
    assert NeedInput(question="How many?", schema_json=schema).schema_json == schema
    with pytest.raises(ValidationError):
        NeedInput(question="How many?", options=["1", "2"], schema_json=schema)


def _cell(code: str, call_id: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[
            ToolCall(id=call_id, name="execute_python", arguments=json.dumps({"code": code}))
        ],
        finish_reason="tool_calls",
    )


def _cell_errors(agent: InteractiveAgent) -> list[str]:
    return [
        e.stderr + e.error
        for e in agent.event_manager.values()
        if isinstance(e, PythonOutput) and e.execution_status is ResultStatus.ERROR
    ]


_NOTIFICATION = {"user_messages": ["hi"]}


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ('return_result(Done(explanation="finished"))', Done),
        ('return_result(NeedInput(question="Which branch?"))', NeedInput),
        ('return_result(Waiting(explanation="waiting for job ci-42"))', Waiting),
        (
            'return_result(RespondResult(kind=RespondReason.DONE, explanation="finished"))',
            RespondResult,
        ),
    ],
)
async def test_handle_accepts_each_turn_result(code, expected):
    agent = _Host(llm=FakeLLMClient([_cell(code, "c1")]))
    result = await agent.handle(_NOTIFICATION)
    assert type(result) is expected


async def test_handle_rejects_other_results():
    llm = FakeLLMClient(
        [
            _cell('return_result("just text")', "c1"),
            _cell('return_result(Done(explanation="finished"))', "c2"),
        ]
    )
    agent = _Host(llm=llm)
    assert isinstance(await agent.handle(_NOTIFICATION), Done)
    [error] = _cell_errors(agent)
    assert "return_result validation error" in error


async def test_handle_batch_rejects_need_input_and_names_allowed_types():
    llm = FakeLLMClient(
        [
            _cell('return_result(NeedInput(question="Which branch?"))', "c1"),
            _cell('return_result(Done(explanation="blocked: branch not given"))', "c2"),
        ]
    )
    agent = _Host(llm=llm)
    result = await agent.handle_batch(_NOTIFICATION)
    assert result == Done(explanation="blocked: branch not given")
    [error] = _cell_errors(agent)
    assert "return_result validation error" in error
    assert "Done | " in error and "Waiting" in error
    assert "NeedInput" in error  # names what was returned


class _NarrowHost(InteractiveAgent, llm=FakeLLMClient()):
    """Host whose turns always finish."""

    @hidden
    @strategy(CodeActStrategy())
    async def handle(self, notification: dict[str, list]) -> Done:
        """Answer the question in one turn; SUBCLASS_HANDLE_DOC."""
        ...


async def test_model_sees_the_subclass_handle_docstring_and_annotation():
    llm = FakeLLMClient([_cell('return_result(Done(explanation="answered"))', "c1")])
    agent = _NarrowHost(llm=llm)
    assert isinstance(await agent.handle(_NOTIFICATION), Done)

    prompt = "\n".join(str(m.get("content", "")) for m in llm.calls[0].messages)
    assert "SUBCLASS_HANDLE_DOC" in prompt
    assert "Handle one interactive turn." not in prompt
    [return_tool] = [t for t in llm.calls[0].tools or [] if t.name == "return_result"]
    assert "Expected return type: Done." in return_tool.description


async def test_model_sees_the_base_handle_batch_annotation():
    llm = FakeLLMClient([_cell('return_result(Done(explanation="finished"))', "c1")])
    agent = _Host(llm=llm)
    await agent.handle_batch(_NOTIFICATION)

    prompt = "\n".join(str(m.get("content", "")) for m in llm.calls[0].messages)
    assert "Handle one unattended turn." in prompt
    [return_tool] = [t for t in llm.calls[0].tools or [] if t.name == "return_result"]
    assert "Expected return type: Done | Waiting." in return_tool.description


def test_install_summarizer_none_policy_is_noop(agent):
    install_summarizer(SummarizationConfig(policy="none"), agent=agent)
    assert not getattr(agent, "_summarizers", [])


def test_install_summarizer_attaches(agent):
    install_summarizer(SummarizationConfig(max_tokens=50_000), agent=agent)
    summarizers = getattr(agent, "_summarizers", [])
    assert len(summarizers) == 1
    assert summarizers[0].config.max_tokens == 50_000


@pytest.mark.parametrize("fraction", [0, -0.1, 1, 1.1])
def test_summarization_threshold_fraction_must_be_between_zero_and_one(fraction):
    with pytest.raises(ValidationError):
        SummarizationConfig(threshold_fraction=fraction)
