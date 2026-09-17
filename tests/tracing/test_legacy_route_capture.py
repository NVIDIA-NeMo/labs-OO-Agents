# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Legacy overrides and handler-backed routes must not lose observed input."""

import json

import httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from nooa.tracing import _llm_hooks
from nooa.unifiedllm import CompletionClient, RetryConfig
from tests.unifiedllm.test_direct_review_contracts import reply


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "route",
    [
        "override",
        "override-base-url",
        "hosted_vllm/test",
        "deepseek/deepseek-chat",
        "gemini/gemini-2.5-flash",
    ],
)
async def test_legacy_actual_request_and_journal_survive_route_selection(
    monkeypatch, route, asynchronous
):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(_llm_hooks, "_tracer", lambda: provider.get_tracer("test"))
    bodies, journal = [], []

    class Journal:
        def log_pre_api_call(self, model, messages, metadata):
            journal.append(messages)

        def log_success_event(self, *args):
            pass

    monkeypatch.setattr(_llm_hooks, "callbacks", [Journal()])

    def send(request):
        bodies.append(json.loads(request.content))
        if route.startswith("override"):
            assert request.url.host == "override.example"
            assert request.headers["authorization"] == "Bearer override-test-key"
            assert bodies[-1]["model"] == "changed"
        data = reply("chat")
        if route.startswith("gemini/"):
            data = {
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "42"}]},
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 12,
                },
            }
        return httpx.Response(200, json=data)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, request: send(request))

    async def async_send(self, request):
        return send(request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", async_send)
    model = "openai/test" if route.startswith("override") else route
    async with CompletionClient(
        model,
        transport="litellm",
        api_base="https://models.example/v1",
        api_key="test-key",
        max_tokens=100,
        retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
    ) as llm:
        overrides = (
            {
                "model": "openai/changed",
                "api_base": "https://override.example/v1",
                "api_key": "override-test-key",
            }
            if route.startswith("override")
            else {}
        )
        if route == "override-base-url":
            overrides["base_url"] = overrides.pop("api_base")
        messages = [{"role": "user", "content": "test input"}]
        response = (
            await llm.acall(messages, **overrides)
            if asynchronous
            else llm.call(messages, **overrides)
        )
        assert response.content == "42"
    spans = exporter.get_finished_spans()
    assert len(spans) == len(bodies) == len(journal) == 1
    assert json.loads(spans[0].attributes["input.value"]) == bodies[0]
    assert "test input" in json.dumps(journal[0])
    provider.shutdown()
