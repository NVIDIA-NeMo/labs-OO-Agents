# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CodeAct: calls after a failed or finishing call get results; text-only
replies that wrote a tool call as text are told so."""

import json
from collections.abc import Mapping

import pytest

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.context_blocks import ResultStatus, ToolCallEvent
from nooa.events import Error
from nooa.runtime.harness_metrics import HarnessMetrics
from nooa.strategies.codeact import (
    TEXT_ONLY_TOOL_MARKUP_NOTE,
    CodeActStrategy,
    text_only_tool_call_markup,
)
from nooa.strategies.codeact_v2 import CodeActV2
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

_STRATEGIES = [(CodeActV2, "python_cell"), (CodeActStrategy, "execute_python")]
_FAILED = (
    "Not executed: an earlier call in this response ({tool} call 1) failed, "
    "so nothing after it ran."
)
_FINISHED = "Not executed: an earlier call in this response already finished the turn."


def _cells(tool: str, *codes: str, prefix: str = "c") -> LLMResponse:
    return LLMResponse(
        parts=tuple(
            ToolCall(id=f"{prefix}{i}", name=tool, arguments=json.dumps({"code": code}))
            for i, code in enumerate(codes)
        ),
        finish_reason="tool_calls",
    )


def _text(content: str, reasoning: str | None = None) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=[],
        finish_reason="stop",
        reasoning=reasoning,
    )


@pytest.fixture
def metrics(monkeypatch):
    metrics = HarnessMetrics()
    monkeypatch.setattr("nooa.strategies.codeact.get_harness_metrics", lambda: metrics)
    return metrics


def _agent(responses, strategy_type):
    llm = FakeLLMClient(scripted_responses=responses)
    config = CodeActConfig(prefill=None, max_retries=5, max_iterations=10)

    class ToolAgent(Agent, llm=llm):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.runs = 0

        def bump(self) -> int:
            """Count a run."""
            self.runs += 1
            return self.runs

        @strategy(strategy_type(config=config))
        async def answer(self) -> str:
            """Return an answer."""
            ...

    return ToolAgent(), llm


def _results(agent) -> dict[str, ToolCallEvent]:
    return {e.tool_call_id: e for e in agent.event_manager.values() if isinstance(e, ToolCallEvent)}


def _tool_messages(llm: FakeLLMClient, call_index: int) -> dict[str, str]:
    return {
        m["tool_call_id"]: m["content"]
        for m in llm.calls[call_index].messages
        if isinstance(m, Mapping) and m.get("role") == "tool"
    }


@pytest.mark.parametrize("strategy_type,tool", _STRATEGIES)
async def test_calls_after_a_failed_call_get_not_executed_results(strategy_type, tool, metrics):
    agent, llm = _agent(
        [
            _cells(tool, "1/0", "self.bump()", "self.bump()"),
            _cells(tool, "return_result('ok')", prefix="r"),
        ],
        strategy_type,
    )
    try:
        assert await agent.answer() == "ok"
        assert agent.runs == 0
        results = _results(agent)
        assert results["c0"].result.result_status == ResultStatus.ERROR
        for call_id in ("c1", "c2"):
            assert results[call_id].result.content == _FAILED.format(tool=tool)
            assert results[call_id].result.result_status == ResultStatus.ERROR
            assert results[call_id].arguments == {"code": "self.bump()"}
        assert metrics.tool_call_skipped_count == 2
        # The next request replays the whole batch: every call has a result.
        replayed = _tool_messages(llm, 1)
        assert set(replayed) == {"c0", "c1", "c2"}
        assert replayed["c1"] == _FAILED.format(tool=tool)
    finally:
        await agent.aclose()


@pytest.mark.parametrize("strategy_type,tool", _STRATEGIES)
async def test_calls_after_a_finishing_call_get_not_executed_results(strategy_type, tool, metrics):
    agent, _ = _agent([_cells(tool, "return_result('done')", "self.bump()")], strategy_type)
    try:
        assert await agent.answer() == "done"
        assert agent.runs == 0
        skipped = _results(agent)["c1"]
        assert skipped.result.content == _FINISHED
        assert metrics.tool_call_skipped_count == 1
    finally:
        await agent.aclose()


async def test_v1_return_result_tool_then_cell_gets_finished_result():
    tool = "execute_python"
    response = LLMResponse(
        parts=(
            ToolCall(id="c0", name="return_result", arguments=json.dumps({"result": "x"})),
            ToolCall(id="c1", name=tool, arguments=json.dumps({"code": "self.bump()"})),
        ),
        finish_reason="tool_calls",
    )
    agent, _ = _agent([response], CodeActStrategy)
    try:
        assert await agent.answer() == "x"
        assert agent.runs == 0
        assert _results(agent)["c1"].result.content == _FINISHED
    finally:
        await agent.aclose()


@pytest.mark.parametrize("strategy_type,tool", _STRATEGIES)
async def test_all_calls_that_succeed_get_no_skip_results(strategy_type, tool, metrics):
    agent, _ = _agent(
        [_cells(tool, "self.bump()", "self.bump()", "return_result('ok')")], strategy_type
    )
    try:
        assert await agent.answer() == "ok"
        assert agent.runs == 2
        assert metrics.tool_call_skipped_count == 0
    finally:
        await agent.aclose()


@pytest.mark.parametrize(
    "text,marker",
    [
        (
            "<think>run it\n<tool_call>\n<function=python_cell>\n<parameter=code>\n"
            "print(1)\n</parameter>\n</function>\n</tool_call></think>",
            "<tool_call>",
        ),
        ("<function=python_cell>", "<function="),
        ("<parameter=code>x</parameter>", "<parameter="),
        ("return_result(Done(explanation='ok'))</arg_value>", "</arg_value>"),
        ('<invoke name="python_cell">', "<invoke"),
        (
            '{"name": "python_cell", "arguments": {"code": "1"}}',
            '{"name": "python_cell", "arguments":',
        ),
    ],
)
def test_tool_call_markup_is_detected(text, marker):
    assert text_only_tool_call_markup(text) == marker


@pytest.mark.parametrize(
    "text",
    [None, "", "I am done.", "Call `return_result(value)` next.", '{"name": "x"}', "a <b> c"],
)
def test_plain_text_has_no_tool_call_markup(text):
    assert text_only_tool_call_markup(text) is None


_NEMOTRON = (
    "<think>I will run it.\n<tool_call>\n<function=python_cell>\n<parameter=code>\n"
    "return_result('ok')\n</parameter>\n</function>\n</tool_call>\n</think>"
)


def _corrections(agent) -> list[str]:
    return [e.content for e in agent.event_manager.values() if isinstance(e, Error)]


@pytest.mark.parametrize("strategy_type,tool", _STRATEGIES)
@pytest.mark.parametrize("where", ["content", "reasoning"])
async def test_text_only_reminder_says_tool_call_was_sent_as_text(
    strategy_type, tool, where, metrics
):
    reply = _text(_NEMOTRON) if where == "content" else _text("Done.", reasoning=_NEMOTRON)
    agent, _ = _agent([reply, _cells(tool, "return_result('ok')")], strategy_type)
    try:
        assert await agent.answer() == "ok"
        (correction,) = _corrections(agent)
        assert correction.startswith(TEXT_ONLY_TOOL_MARKUP_NOTE + " Your last reply was plain text")
        assert metrics.text_only_tool_markup_count == 1
        # The recovered call is not executed: only the real call ran.
        calls = [i for i in _results(agent) if not i.startswith("codeact_inline")]
        assert calls == ["c0"]
    finally:
        await agent.aclose()


@pytest.mark.parametrize("strategy_type,tool", _STRATEGIES)
async def test_text_only_reminder_without_markup_is_unchanged(strategy_type, tool, metrics):
    agent, _ = _agent([_text("I am done."), _cells(tool, "return_result('ok')")], strategy_type)
    try:
        assert await agent.answer() == "ok"
        (correction,) = _corrections(agent)
        assert correction.startswith("Your last reply was plain text with no tool call.")
        assert "written as text" not in correction
        assert metrics.text_only_tool_markup_count == 0
    finally:
        await agent.aclose()


def test_text_only_reminder_texts():
    assert TEXT_ONLY_TOOL_MARKUP_NOTE == (
        "Your reply contained a tool call written as text (inside <think>, or as "
        "XML/markup), so it was not run. Tool calls only run when sent as real tool "
        "calls, after your reasoning."
    )
