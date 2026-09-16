# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from nooa.llm_types import LLMResponse, LLMUsage


@pytest.mark.asyncio
async def test_playground_uses_shared_client_and_public_response(monkeypatch):
    from nooa import unifiedllm
    from nooa.viewer import trace_routes

    client = AsyncMock()
    client.acall.return_value = LLMResponse(
        content="answer",
        reasoning="readable",
        usage=LLMUsage(input_tokens=3, output_tokens=2, total_tokens=5),
    )
    monkeypatch.setattr(unifiedllm, "CompletionClient", lambda **kwargs: client)
    monkeypatch.setattr(trace_routes, "get_model_config", lambda model: {})
    result = await trace_routes.run_inference(
        trace_routes.InferenceRequest(
            model="test", messages=[{"role": "user", "content": "question"}]
        )
    )
    assert result["response"]["content"] == "answer"
    assert result["response"]["reasoning_content"] == "readable"
    assert result["usage"]["prompt_tokens"] == 3
    assert client.acall.await_count == client.aclose.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["litellm", "direct"])
@pytest.mark.parametrize(
    "model,wire_model",
    [("openai/test", "test"), ("test", "test"), ("company/test", "company/test")],
)
async def test_playground_custom_endpoint_preserves_wire_model(
    monkeypatch, transport, model, wire_model
):
    from nooa.viewer import trace_routes

    monkeypatch.setenv("NOOA_LLM_TRANSPORT", transport)
    monkeypatch.setattr(
        trace_routes, "get_model_config", lambda model: {"endpoint": "https://custom.example/v1"}
    )
    monkeypatch.setattr(
        trace_routes, "resolve_api_key_from_config", lambda *args, **kwargs: "test-key"
    )
    sent = []

    async def respond(self, request):
        sent.append(request)
        return httpx.Response(
            200,
            json={
                "id": "r1",
                "object": "chat.completion",
                "created": 0,
                "model": "reported-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            },
        )

    def forbid_sync(*args, **kwargs):
        raise AssertionError("Playground escaped its mocked async HTTP transport")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", respond)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbid_sync)
    result = await trace_routes.run_inference(
        trace_routes.InferenceRequest(
            model=model,
            messages=[{"role": "user", "content": "question"}],
            temperature=0.7,
            max_tokens=123,
        )
    )
    assert len(sent) == 1
    assert str(sent[0].url) == "https://custom.example/v1/chat/completions"
    assert sent[0].headers["authorization"] == "Bearer test-key"
    body = json.loads(sent[0].content)
    assert body["model"] == wire_model
    assert body["max_tokens"] == 123
    assert body["temperature"] == 0.7
    assert body["messages"] == [{"role": "user", "content": "question"}]
    assert result["response"]["content"] == "answer"
    assert result["usage"]["total_tokens"] == 5
