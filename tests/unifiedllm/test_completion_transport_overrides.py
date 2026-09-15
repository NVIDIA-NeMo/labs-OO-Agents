# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise request routing through LiteLLM and the OpenAI SDK, without network I/O."""

import httpx
import pytest
from litellm.llms.openai.openai import OpenAIChatCompletion

from nooa.unifiedllm import CompletionClient
from nooa.unifiedllm.unifiedllm import _ClientHttp


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    "overrides",
    [
        {"api_base": "https://override.example/v1"},
        {"api_key": "override-key"},
        {"api_base": "https://override.example/v1", "api_key": "override-key"},
    ],
)
async def test_completion_uses_per_call_route_without_changing_defaults(
    monkeypatch, is_async, overrides
):
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
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
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            },
        )

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        _ClientHttp, "_httpx_hardening", staticmethod(lambda: {"transport": transport})
    )
    client = CompletionClient(
        "openai/gpt-4o-mini",
        api_base="https://original.example/v1",
        api_key="original-key",
    )
    client_http = client._http
    assert client_http is not None
    # Only overridden routes may use LiteLLM's default pool. An unchanged route
    # must use NOOA's owned client, not merely reach an equivalent mock transport.
    allow_default_pool = False

    def default_pool(*, is_async=False, **kwargs):
        assert allow_default_pool, "Unchanged routing must reuse the owned HTTP client"
        return client_http.httpx_async if is_async else client_http.httpx_sync

    monkeypatch.setattr(
        OpenAIChatCompletion,
        "_get_sync_http_client",
        staticmethod(default_pool),
    )
    monkeypatch.setattr(
        OpenAIChatCompletion,
        "_get_async_http_client",
        staticmethod(lambda **kwargs: default_pool(is_async=True)),
    )
    try:
        owned_client = client_http.async_client if is_async else client_http.sync_client
        assert client._completion_http_client(client.config, is_async=is_async) is owned_client
        unchanged = {**client.config, "temperature": 0.1}
        assert client._completion_http_client(unchanged, is_async=is_async) is owned_client

        for params in ({}, overrides, {}):
            allow_default_pool = bool(params)
            messages = [{"role": "user", "content": "Hello"}]
            result = (
                await client.acall(messages, **params)
                if is_async
                else client.call(messages, **params)
            )
            assert result.content == "done"

        assert [str(request.url) for request in requests] == [
            "https://original.example/v1/chat/completions",
            overrides.get("api_base", "https://original.example/v1") + "/chat/completions",
            "https://original.example/v1/chat/completions",
        ]
        assert [request.headers["Authorization"] for request in requests] == [
            "Bearer original-key",
            "Bearer " + overrides.get("api_key", "original-key"),
            "Bearer original-key",
        ]
    finally:
        await client.aclose()
