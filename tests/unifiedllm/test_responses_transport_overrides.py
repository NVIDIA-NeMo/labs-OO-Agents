# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Responses HTTP handlers reuse their pool without binding a URL, key, or model."""

import json

import httpx
import pytest

from nooa.unifiedllm import ResponsesClient
from nooa.unifiedllm.unifiedllm import _ClientHttp


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"api_base": "https://override.example/v1"},
        {"base_url": "https://override.example/v1"},
        {"api_key": "override-key"},
        {"model": "openai/gpt-5.4"},
        {"api_base": "https://override.example/v1", "api_key": "override-key"},
    ],
)
async def test_responses_transport_honors_overrides_without_changing_defaults(
    monkeypatch, is_async, overrides
):
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "resp_test",
                "created_at": 0,
                "model": "gpt-5.6",
                "status": "completed",
                "output": [
                    {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "done", "annotations": []}],
                    }
                ],
            },
        )

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        _ClientHttp, "_httpx_hardening", staticmethod(lambda: {"transport": transport})
    )

    # Fail locally if a regression discards our pool and tries the network.
    def unexpected_network(*args, **kwargs):
        raise AssertionError("Responses must reuse the configured HTTP transport")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", unexpected_network)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", unexpected_network)
    client = ResponsesClient(
        "openai/gpt-5.6",
        api_base="https://original.example/v1",
        api_key="original-key",
    )
    try:
        # Exercise real LiteLLM serialization and dispatch, not mocked call kwargs.
        for params in ({}, overrides, {}):
            messages = [{"role": "user", "content": "hello"}]
            result = (
                await client.acall(messages, **params)
                if is_async
                else client.call(messages, **params)
            )
            assert result.content == "done"

        override_base = overrides.get(
            "base_url", overrides.get("api_base", "https://original.example/v1")
        )
        assert [str(request.url) for request in requests] == [
            "https://original.example/v1/responses",
            override_base + "/responses",
            "https://original.example/v1/responses",
        ]
        assert [request.headers["Authorization"] for request in requests] == [
            "Bearer original-key",
            "Bearer " + overrides.get("api_key", "original-key"),
            "Bearer original-key",
        ]
        assert [json.loads(request.content)["model"] for request in requests] == [
            "gpt-5.6",
            overrides.get("model", "openai/gpt-5.6").removeprefix("openai/"),
            "gpt-5.6",
        ]
    finally:
        await client.aclose()
