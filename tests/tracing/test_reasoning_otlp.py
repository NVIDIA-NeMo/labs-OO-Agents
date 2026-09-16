# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Readable reasoning survives scrubbing and the journal-mode OTLP strip."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from nooa.tracing import _llm_hooks
from nooa.tracing._otlp_http_exporter import OtlpJsonHttpExporter
from nooa.tracing._secret_scrubber import SecretScrubSpanProcessor
from nooa.unifiedllm import (
    CompletionClient,
    ReasoningCompletionClient,
    ResponsesClient,
    RetryConfig,
)

REASONING = "Check the evidence before answering."
OPAQUE = "opaque-provider-state-must-not-leave"


@pytest.mark.parametrize("transport", ["litellm", "direct"])
@pytest.mark.parametrize("strip_messages", [False, True])
@pytest.mark.parametrize("shape", ["reasoning_content", "reasoning", "think_tags", "responses"])
def test_plain_reasoning_reaches_serialized_otlp_without_opaque_state(
    shape, strip_messages, transport, monkeypatch
):
    monkeypatch.delenv("NEMO_TRACE_KEEP_LLM_VALUES", raising=False)
    captured = []

    def send(request, timeout):
        captured.append(json.loads(request.data))
        return nullcontext(SimpleNamespace(status=200))

    monkeypatch.setattr("urllib.request.urlopen", send)
    exporter = OtlpJsonHttpExporter(strip_llm_messages=strip_messages)
    provider = TracerProvider()
    provider.add_span_processor(SecretScrubSpanProcessor(SimpleSpanProcessor(exporter)))
    monkeypatch.setattr(_llm_hooks, "_tracer", lambda: provider.get_tracer("reasoning-test"))

    if shape == "responses":
        response = {
            "id": "resp_test",
            "created_at": 1,
            "model": "gpt-5.6",
            "object": "response",
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "id": "rs_test",
                    "encrypted_content": OPAQUE,
                    "summary": [{"type": "summary_text", "text": REASONING}],
                }
            ],
        }
        cls = ResponsesClient
    else:
        message = {"role": "assistant", "content": "answer"}
        if shape == "think_tags":
            message["content"] = f"<think>{REASONING}</think>answer"
        else:
            message[shape] = REASONING
        response = {
            "id": "chat_test",
            "created": 1,
            "object": "chat.completion",
            "model": "test",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        }
        cls = ReasoningCompletionClient if shape == "think_tags" else CompletionClient

    monkeypatch.setattr(
        httpx.HTTPTransport,
        "handle_request",
        lambda self, request: httpx.Response(200, json=response),
    )
    try:
        with cls(
            "openai/test",
            transport=transport,
            api_key="test",
            api_base="https://models.example/v1",
            retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
        ) as client:
            turn = client.call([{"role": "user", "content": "Check"}])
            client.call([turn, {"role": "user", "content": "Continue"}])
    finally:
        provider.shutdown()

    assert len(captured) == 2
    wire = json.dumps(captured)
    assert OPAQUE not in wire
    exported_span = captured[1]["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    attributes = {item["key"]: item["value"] for item in exported_span["attributes"]}
    assert attributes["llm.reasoning_content"]["stringValue"] == REASONING
