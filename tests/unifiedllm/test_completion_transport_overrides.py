# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise request routing through LiteLLM and the OpenAI SDK, without network I/O."""

import httpx
import pytest
from litellm.llms.openai.openai import OpenAIChatCompletion

from nooa.unifiedllm import CompletionClient, registry
from nooa.unifiedllm.unifiedllm import _ClientHttp


def _chat_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": "opaque-model",
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
        owned_pool = client_http.httpx_async if is_async else client_http.httpx_sync
        unchanged = {**client.config, "temperature": 0.1}
        for config in (client.config, unchanged):
            sdk_client = client._completion_http_client(config, is_async=is_async, max_retries=0)
            # Each call gets its own SDK client, but over the owned HTTP pool.
            assert sdk_client._client is owned_pool

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


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("factory", ["direct", "registry"])
async def test_base_url_provider_and_timeout_reach_owned_transport(monkeypatch, is_async, factory):
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _chat_response(request)

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        _ClientHttp,
        "_httpx_hardening",
        staticmethod(lambda: {"transport": transport, "trust_env": False}),
    )

    def no_default_pool(**kwargs):
        raise AssertionError("A declared base_url/custom_llm_provider must use the owned pool")

    monkeypatch.setattr(
        OpenAIChatCompletion, "_get_sync_http_client", staticmethod(no_default_pool)
    )
    monkeypatch.setattr(
        OpenAIChatCompletion, "_get_async_http_client", staticmethod(no_default_pool)
    )
    route = {
        "base_url": "https://gateway.example/v1",
        "custom_llm_provider": "openai",
        "timeout": 7.5,
    }
    if factory == "registry":
        entry = {"model_name": "opaque-model", "retry_config": False, **route}
        client = registry.client_from_config("gateway", entry, api_key="gateway-key")
    else:
        client = CompletionClient("opaque-model", api_key="gateway-key", **route)
    try:
        assert {key: client.config[key] for key in route} == route
        assert client._http is not None
        # The owned SDK client must target the declared gateway, not be skipped.
        sdk_client = client._completion_http_client(client.config, is_async=is_async, max_retries=0)
        assert sdk_client is not None
        assert str(sdk_client.base_url) == "https://gateway.example/v1/"
        owned_pool = client._http.httpx_async if is_async else client._http.httpx_sync
        assert sdk_client._client is owned_pool
        messages = [{"role": "user", "content": "Hello"}]
        result = await client.acall(messages) if is_async else client.call(messages)
        assert result.content == "done"
        assert [str(request.url) for request in requests] == [
            "https://gateway.example/v1/chat/completions"
        ]
        assert requests[0].headers["Authorization"] == "Bearer gateway-key"
        assert requests[0].extensions["timeout"]["read"] == 7.5
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    ("configured", "override"),
    [("api_base", "base_url"), ("base_url", "api_base")],
)
def test_endpoint_override_replaces_both_configured_aliases(configured, override):
    """Overriding either endpoint spelling must not leave the other configured one."""
    raw = {
        "model_name": "opaque-model",
        "retry_config": False,
        configured: "https://old.example/v1",
    }
    client = registry.client_from_config(
        "gateway", raw, api_key="key", **{override: "https://new.example/v1"}
    )
    assert configured not in client.config or configured == override
    assert client.config[override] == "https://new.example/v1"
    sdk_client = client._completion_http_client(client.config, is_async=False, max_retries=0)
    if sdk_client is not None:
        assert str(sdk_client.base_url) == "https://new.example/v1/"


def test_unforwarded_endpoint_key_keeps_context_window():
    """A completion entry's ``endpoint`` key is not sent, so it must not count as a route change."""
    raw = {
        "model_name": "openai/opaque-model",
        "api_base": "https://gateway.example/v1",
        "endpoint": "https://gateway.example/v1/decisions",
        "context_window": 8192,
        "retry_config": False,
    }
    client = registry.client_from_config("gateway", raw)
    assert client.context_window == 8192
