# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read real UnifiedLLM OTLP exports through the explorer and viewer contract."""

import json
from contextlib import nullcontext

import httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from nooa.trace_explorer.explorer import (
    ExecutionTurn,
    LLMTurn,
    TraceExplorer,
    _load_spans,
    _parse_trace_from_spans,
)
from nooa.tracing import _llm_hooks
from nooa.tracing._otlp_file_exporter import OtlpJsonFileExporter
from nooa.unifiedllm import CompletionClient, RetryConfig, Tool


@pytest.mark.parametrize("transport", ["direct", "litellm"])
@pytest.mark.parametrize("agent_span", [False, True])
async def test_exported_calls_preserve_turns_and_execution_context(
    tmp_path, monkeypatch, transport, agent_span
):
    exporter = OtlpJsonFileExporter(tmp_path)
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("explorer-contract")
    monkeypatch.setattr(_llm_hooks, "_tracer", lambda: tracer)
    monkeypatch.setattr(_llm_hooks, "callbacks", [])
    code = "print('wire result')"
    calls = []

    async def respond(self, request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "response-fixture",
                "object": "chat.completion",
                "created": 1,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Running the requested check.",
                            "tool_calls": [
                                {
                                    "id": "call-fixture",
                                    "type": "function",
                                    "function": {
                                        "name": "python_cell",
                                        "arguments": json.dumps({"code": code}),
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
            },
        )

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", respond)

    def python_cell(code: str) -> str:
        raise AssertionError("Only HTTP is mocked; model tool calls are not executed")

    agent_attrs = {
        "openinference.span.kind": "AGENT",
        "agent.name": "FixtureAgent",
        "agent.method": "answer",
        "agent.call_id": "fixture-call",
    }
    generation_attrs = {
        "openinference.span.kind": "CHAIN",
        "agent.name": "FixtureAgent",
        "agent.method": "answer",
        "agent.call_id": "fixture-call",
        "generation.id": "generation-fixture",
    }
    try:
        outer = (
            tracer.start_as_current_span("FixtureAgent.answer", attributes=agent_attrs)
            if agent_span
            else nullcontext()
        )
        with outer, tracer.start_as_current_span("generation", attributes=generation_attrs):
            async with CompletionClient(
                "openai/test",
                transport=transport,
                api_key="fixture",
                api_base="https://models.example/v1",
                retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
            ) as client:
                await client.acall(
                    [{"role": "user", "content": "Inspect this exact request."}],
                    tools=[
                        Tool(name="python_cell", description="Run a cell", callable=python_cell)
                    ],
                )
            with tracer.start_as_current_span(
                "code_execution",
                attributes={
                    "agent.name": "FixtureAgent",
                    "generation.id": "generation-fixture",
                    "tool_call_id": "call-fixture",
                    "input.value": json.dumps({"code": code}),
                    "output.value": json.dumps({"stdout": "wire result\n", "returned_value": None}),
                },
            ):
                pass
        provider.force_flush()
        assert len(calls) == 1
        assert exporter.default_file is not None
        trace = await TraceExplorer.from_file(exporter.default_file)
        assert len(trace.sessions) == 1
        session = trace.sessions[0]
        assert [type(turn) for turn in session.turns] == [LLMTurn, ExecutionTurn]
        llm, execution = session.turns
        assert llm.model == "test"
        assert llm.token_counts == {"prompt": 12, "completion": 4, "total": 16}
        assert llm.tool_calls[0].function_name == "python_cell"
        assert llm.tool_calls[0].arguments == json.dumps({"code": code})
        assert llm.tool_calls[0].tool_call_id == execution.tool_call_id == "call-fixture"
        rendered = await trace.get_turn(session.session_id, 1)
        assert "Inspect this exact request." in rendered
        assert '<tool_call name="python_cell" id="call-fixture">' in rendered
        assert code in rendered and "wire result" in rendered
        spans = _load_spans(exporter.default_file)
        exported_call = next(span for span in spans if span["name"] == "llm.call")
        # The existing React adapter selects span.llm_call from this canonical hint.
        assert exported_call["attributes"]["nooa.viewer.plugin"] == "llm_call"
        # Old saved traces have several SDK span names and may lack semantic kind.
        # A different provider's semantically classified LLM span also works.
        for name, kind in (
            ("acompletion", None),
            ("completion", None),
            ("aresponses", None),
            ("responses", None),
            ("provider.request", "LLM"),
        ):
            attributes = dict(exported_call["attributes"])
            attributes.pop("openinference.span.kind", None)
            if kind:
                attributes["openinference.span.kind"] = kind
            renamed = {**exported_call, "name": name, "attributes": attributes}
            compatible = _parse_trace_from_spans(
                [renamed if span is exported_call else span for span in spans]
            )
            assert [type(turn) for turn in compatible[0].turns] == [LLMTurn, ExecutionTurn]
            assert compatible[0].turns[0].tool_calls[0].tool_call_id == "call-fixture"
    finally:
        provider.shutdown()
