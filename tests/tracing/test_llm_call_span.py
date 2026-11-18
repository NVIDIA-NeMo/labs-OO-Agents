# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Each LLM call is one ``llm_call`` span: model, tokens, finish, reasoning, tool calls, latency.

The span is NOOA's own (from the runtime's ``on_llm_call`` hook), so native
exports carry per-call data even when the LiteLLM OpenInference instrumentor
emits nothing. It carries no prompt or completion text.
"""

from __future__ import annotations

import json
import logging

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from nooa import Agent, strategy
from nooa.llm_types import LLMUsage
from nooa.runtime.hooks import call_on_llm_call, hooks_scope, set_hooks
from nooa.strategies.predict import PredictStrategy
from nooa.tracing._hooks_impl import OpenInferenceHooks
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

SECRET_PROMPT = "PROMPT-TEXT-MUST-NOT-LEAK"
SECRET_REPLY = "REPLY-TEXT-MUST-NOT-LEAK"


@pytest.fixture
def traced():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    hooks = OpenInferenceHooks(tracer=provider.get_tracer("test"))
    set_hooks(hooks)
    try:
        yield hooks, exporter
    finally:
        set_hooks(None)
        exporter.clear()


class Echo(Agent):
    @strategy(PredictStrategy())
    async def answer(self, question: str) -> str:
        """Answer the question."""
        ...


def _reply(**kwargs) -> LLMResponse:
    content = json.dumps(SECRET_REPLY)
    return LLMResponse(raw_response=None, content=content, finish_reason="stop", **kwargs)


async def test_a_predict_call_is_one_llm_call_span_under_its_generation(traced):
    _, exporter = traced
    usage = LLMUsage(
        input_tokens=120,
        output_tokens=30,
        cached_input_tokens=64,
        reasoning_tokens=11,
        total_tokens=150,
    )
    agent = Echo(llm=FakeLLMClient([_reply(usage=usage)]))
    assert await agent.answer(SECRET_PROMPT) == SECRET_REPLY
    finished = exporter.get_finished_spans()
    [span] = [s for s in finished if s.name == "llm_call"]
    [generation] = [s for s in finished if s.name == "generation"]
    assert span.parent.span_id == generation.context.span_id
    assert span.context.trace_id == generation.context.trace_id
    attrs = dict(span.attributes)
    assert attrs["openinference.span.kind"] == "CHAIN"
    assert attrs["agent.name"] == "Echo" and attrs["agent.method"] == "answer"
    assert attrs["generation.id"] == generation.attributes["generation.id"]
    assert attrs["agent.call_id"] == generation.attributes["agent.call_id"]
    assert attrs["nooa.llm.turn_index"] == 1
    assert attrs["llm.model_name"] == attrs["gen_ai.request.model"]
    assert attrs["llm.token_count.prompt"] == 120 == attrs["gen_ai.usage.input_tokens"]
    assert attrs["llm.token_count.completion"] == 30 == attrs["gen_ai.usage.output_tokens"]
    assert attrs["llm.token_count.total"] == 150
    assert attrs["llm.token_count.prompt_details.cache_read"] == 64
    assert attrs["llm.token_count.completion_details.reasoning"] == 11
    assert attrs["nooa.llm.usage_reported"] is True
    assert attrs["nooa.llm.finish_reason"] == "stop"
    assert tuple(attrs["gen_ai.response.finish_reasons"]) == ("stop",)
    assert attrs["nooa.llm.reasoning.present"] is True
    assert attrs["nooa.llm.reasoning.enabled"] is True
    assert attrs["nooa.llm.reasoning.level"] == "default"
    assert attrs["nooa.llm.tool_calls.count"] == 0
    assert attrs["nooa.llm.output.chars"] == len(json.dumps(SECRET_REPLY))
    assert attrs["nooa.llm.success"] is True
    assert attrs["nooa.llm.latency_ms"] >= 0
    assert span.end_time >= span.start_time
    # Metadata only: neither the prompt nor the completion text is on the span.
    dumped = json.dumps({k: str(v) for k, v in attrs.items()})
    assert SECRET_PROMPT not in dumped and SECRET_REPLY not in dumped


def _hooks_and_generation(traced, generation_id="gen-1"):
    hooks, exporter = traced
    agent = Echo(llm=FakeLLMClient([]))
    context = hooks.before_generation(
        agent=agent,
        method_name="answer",
        strategy="CODEACT",
        generation_id=generation_id,
        parent_generation_id=None,
        agent_call_id=None,
    )
    return hooks, exporter, agent, context


def test_tool_calls_and_reasoning_level_are_recorded_without_arguments(traced):
    hooks, exporter, agent, context = _hooks_and_generation(traced)
    response = LLMResponse(
        raw_response=None,
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            ToolCall(id="call_7", name="python_cell", arguments=json.dumps({"code": "SECRET"})),
        ],
    )

    class Client:
        model = "openai/policy"
        reasoning_level = "off"

    hooks.on_llm_call(
        agent=agent,
        method_name="answer",
        generation_id="gen-1",
        start_time_ns=1_000_000_000,
        end_time_ns=1_250_000_000,
        response=response,
        turn_number=3,
        llm_client=Client(),
    )
    hooks.after_generation(
        agent=agent,
        method_name="answer",
        result="x",
        exception=None,
        context=context,
        generation_id="gen-1",
    )
    [span] = [s for s in exporter.get_finished_spans() if s.name == "llm_call"]
    attrs = dict(span.attributes)
    assert tuple(attrs["nooa.llm.tool_calls.names"]) == ("python_cell",)
    assert tuple(attrs["nooa.llm.tool_calls.ids"]) == ("call_7",)
    assert attrs["nooa.llm.tool_calls.count"] == 1
    assert attrs["nooa.llm.latency_ms"] == 250.0
    assert attrs["nooa.llm.turn_index"] == 3
    assert attrs["llm.model_name"] == "openai/policy"
    assert attrs["nooa.llm.reasoning.level"] == "off"
    assert attrs["nooa.llm.reasoning.enabled"] is False
    # No usage reported: say so rather than record zeros.
    assert attrs["nooa.llm.usage_reported"] is False
    assert "llm.token_count.prompt" not in attrs
    assert "SECRET" not in json.dumps({k: str(v) for k, v in attrs.items()})
    assert (span.start_time, span.end_time) == (1_000_000_000, 1_250_000_000)


def test_a_failed_call_is_an_error_span(traced):
    hooks, exporter, agent, context = _hooks_and_generation(traced)
    hooks.on_llm_call(
        agent=agent,
        method_name="answer",
        generation_id="gen-1",
        start_time_ns=10,
        end_time_ns=20,
        exception_type="RateLimitError",
    )
    hooks.after_generation(
        agent=agent,
        method_name="answer",
        result=None,
        exception=RuntimeError("x"),
        context=context,
        generation_id="gen-1",
    )
    [span] = [s for s in exporter.get_finished_spans() if s.name == "llm_call"]
    assert span.status.status_code.name == "ERROR"
    assert span.attributes["error.type"] == "RateLimitError"
    assert span.attributes["nooa.llm.success"] is False


async def test_a_failing_llm_call_still_emits_its_span(traced):
    _, exporter = traced

    class Boom(FakeLLMClient):
        async def acall(self, *args, **kwargs):
            raise RuntimeError("provider down")

    agent = Echo(llm=Boom([]))
    with pytest.raises(Exception, match="provider down|LLM|failed"):  # noqa: B017
        await agent.answer("q")
    spans = [s for s in exporter.get_finished_spans() if s.name == "llm_call"]
    assert spans and all(s.attributes["nooa.llm.success"] is False for s in spans)


def _no_op_hook(name: str):
    if name.startswith(("before_", "after_", "on_")):
        return lambda **_: None
    raise AttributeError(name)


class _Recorder:
    """A hook that records LLM calls and ignores every other callback."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __getattr__(self, name: str):
        return _no_op_hook(name)

    def on_llm_call(self, **kwargs) -> None:
        self.calls.append(kwargs)


async def test_hooks_without_on_llm_call_are_skipped_quietly(caplog):
    class Legacy:  # an older hook implementation: every callback but on_llm_call
        def __getattr__(self, name: str):
            if name == "on_llm_call":
                raise AttributeError(name)
            return _no_op_hook(name)

    recorder = _Recorder()
    caplog.set_level(logging.WARNING, logger="nooa.runtime.hooks")
    with hooks_scope(Legacy(), compose=False):
        call_on_llm_call(
            agent=None, method_name="m", generation_id="", start_time_ns=0, end_time_ns=0
        )
    with hooks_scope(Legacy(), compose=False), hooks_scope(recorder):
        agent = Echo(llm=FakeLLMClient([_reply()]))
        await agent.answer("q")
    assert not [r for r in caplog.records if "on_llm_call" in r.getMessage()]
    [call] = recorder.calls
    assert call["method_name"] == "answer" and SECRET_REPLY in call["response"].content
    assert call["end_time_ns"] >= call["start_time_ns"] > 0
