# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pin Chat tool-choice defaults at the HTTP boundary, without live calls."""

import copy
import json

import httpx
import pytest

from nooa.unifiedllm import CompletionClient, RetryConfig, Tool
from nooa.unifiedllm.unifiedllm import _ClientHttp


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("configured", [False, True], ids=["per-call", "constructor"])
@pytest.mark.parametrize(
    "choice",
    [None, "auto", "required", "none", {"type": "function", "function": {"name": "lookup"}}],
    ids=["omitted", "auto", "required", "none", "named"],
)
async def test_chat_omits_only_default_tool_choice(monkeypatch, is_async, configured, choice):
    """Omit auto, retain sequential tools and explicit choices, without changing config."""
    bodies = []

    def respond(request):
        assert request.url == "https://gateway.example/v1/chat/completions"
        body = json.loads(request.content)
        bodies.append(body)
        # Model the server interaction: each field works alone, but together
        # an explicit default and parallel=False produce conflicting settings.
        if body.get("tool_choice") == "auto" and body.get("parallel_tool_calls") is False:
            return httpx.Response(
                400, json={"error": {"message": "conflicting tool-choice settings"}}
            )
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
        )

    def forbid_network(*args, **kwargs):
        raise AssertionError("Test escaped the mock HTTP transport")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbid_network)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbid_network)
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        _ClientHttp, "_httpx_hardening", staticmethod(lambda: {"transport": transport})
    )
    params = {} if choice is None else {"tool_choice": copy.deepcopy(choice)}
    original_params = copy.deepcopy(params)
    client = CompletionClient(
        "openai/gpt-4o-mini",
        api_base="https://gateway.example/v1",
        api_key="test-key",
        max_tokens=16,
        retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
        **(params if configured else {}),
    )
    config = copy.deepcopy(client.config)
    messages = [{"role": "user", "content": "Look it up."}]
    tools = [Tool(name="lookup", description="Look up a value", callable=lambda: "value")]
    try:
        kwargs = {} if configured else params
        response = (
            await client.acall(messages, tools=tools, **kwargs)
            if is_async
            else client.call(messages, tools=tools, **kwargs)
        )
        assert response.content == "ok"
        assert len(bodies) == 1
        body = bodies[0]
        if choice is None or choice == "auto":
            assert "tool_choice" not in body
        else:
            assert body["tool_choice"] == choice
        assert body["parallel_tool_calls"] is False
        assert body["tools"][0]["function"]["name"] == "lookup"
        assert body["max_tokens"] == 16
        assert body["messages"] == messages
        assert client.config == config
        assert params == original_params
    finally:
        await client.aclose()
