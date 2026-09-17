# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Readable reasoning survives scrubbing and the journal-mode OTLP strip."""

import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import ModelResponse
from openinference.instrumentation import litellm as instrumentation
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from nooa.tracing._litellm_patch import apply_litellm_patch
from nooa.tracing._otlp_http_exporter import OtlpJsonHttpExporter
from nooa.tracing._secret_scrubber import SecretScrubSpanProcessor

REASONING = "Check the evidence before answering."
OPAQUE = "opaque-provider-state-must-not-leave"


@pytest.mark.parametrize("strip_messages", [False, True])
@pytest.mark.parametrize("shape", ["reasoning_content", "reasoning", "think_tags", "responses"])
def test_plain_reasoning_reaches_serialized_otlp_without_opaque_state(
    shape, strip_messages, monkeypatch
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
    apply_litellm_patch()

    if shape == "responses":
        response = ResponsesAPIResponse.model_validate(
            {
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
        )
    else:
        message = {"role": "assistant", "content": "answer"}
        if shape == "think_tags":
            message["content"] = f"<think>{REASONING}</think>answer"
        else:
            message[shape] = REASONING
        response = ModelResponse(
            model="test",
            choices=[{"index": 0, "finish_reason": "stop", "message": message}],
        )

    try:
        with provider.get_tracer("reasoning-test").start_as_current_span("llm") as span:
            span.set_attribute("openinference.span.kind", "LLM")
            # Exercise the same redaction boundary for all three opaque forms.
            span.set_attribute(
                "input.value",
                json.dumps(
                    [
                        {"encrypted_content": OPAQUE},
                        {"type": "thinking", "thinking": REASONING, "signature": OPAQUE},
                        {"thought_signature": OPAQUE},
                    ]
                ),
            )
            instrumentation._finalize_span(span, response)
    finally:
        provider.shutdown()

    assert len(captured) == 1
    wire = json.dumps(captured)
    assert OPAQUE not in wire
    exported_span = captured[0]["resourceSpans"][0]["scopeSpans"][0]["spans"][0]
    attributes = {item["key"]: item["value"] for item in exported_span["attributes"]}
    assert attributes["llm.reasoning_content"]["stringValue"] == REASONING
