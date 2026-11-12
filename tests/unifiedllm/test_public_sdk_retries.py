# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real SDK/HTTP calls over MockTransport: strict sends override legacy opt-ins."""

import asyncio

import httpx
import pytest
from openai import AsyncOpenAI, OpenAI

from nooa.unifiedllm import LLM, CompletionClient, LLMRequest, ResponsesClient, RetryConfig
from nooa.unifiedllm.unifiedllm import _ClientHttp

MODE = pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])


@MODE
@pytest.mark.parametrize(
    "client_type, supplied",
    [(CompletionClient, False), (CompletionClient, True), (ResponsesClient, False)],
    ids=["chat-owned", "chat-supplied-sdk", "responses-owned-handler"],
)
@pytest.mark.parametrize("nooa_retries", [0, 1])
async def test_public_builtin_has_no_hidden_sdk_retries(
    monkeypatch, is_async, client_type, supplied, nooa_retries
):
    attempts = []

    def respond(request):
        attempts.append(request)
        return httpx.Response(500, json={"error": {"message": "internal error"}})

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        _ClientHttp, "_httpx_hardening", staticmethod(lambda: {"transport": transport})
    )

    def network_forbidden(*args, **kwargs):
        raise AssertionError("unexpected network")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", network_forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", network_forbidden)
    sdk = None
    config = {}
    if supplied:
        sdk = (
            AsyncOpenAI(
                api_key="test",
                base_url="https://endpoint.test/v1",
                http_client=httpx.AsyncClient(transport=transport),
                max_retries=2,
            )
            if is_async
            else OpenAI(
                api_key="test",
                base_url="https://endpoint.test/v1",
                http_client=httpx.Client(transport=transport),
                max_retries=2,
            )
        )
        config["client"] = sdk
    client = client_type(
        "openai/gpt-4o-mini",
        api_key="test",
        api_base="https://endpoint.test/v1",
        max_retries=2,
        **config,
    )
    retry = RetryConfig(
        max_retries=nooa_retries, rate_limit_extra_retries=0, base_delay=0, jitter_factor=0
    )
    llm = LLM(client._adapter, retry_config=retry)
    try:
        with pytest.raises(Exception, match="(?i)internal"):
            request = LLMRequest(messages=[{"role": "user", "content": "hi"}])
            if is_async:
                await llm.acall(request)
            else:
                llm.call(request)
        assert len(attempts) == nooa_retries + 1
        assert client.config["max_retries"] == 2
        if sdk is not None:
            assert sdk.max_retries == 2
    finally:
        await client.aclose()
        if sdk is not None:
            if is_async:
                await sdk.close()
            else:
                sdk.close()


@pytest.mark.asyncio
async def test_public_and_legacy_calls_keep_separate_retry_budgets(monkeypatch):
    attempts = {"legacy": 0, "public": 0}

    async def respond(request):
        caller = "legacy" if b"legacy" in request.content else "public"
        attempts[caller] += 1
        await asyncio.sleep(0.01)
        return httpx.Response(500, json={"error": {"message": "internal error"}})

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        _ClientHttp, "_httpx_hardening", staticmethod(lambda: {"transport": transport})
    )
    client = CompletionClient(
        "openai/gpt-4o-mini",
        api_key="test",
        api_base="https://endpoint.test/v1",
        retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
    )
    llm = LLM(client._adapter)
    try:
        results = await asyncio.gather(
            client.acall([{"role": "user", "content": "legacy"}], max_retries=1),
            llm.acall(LLMRequest(messages=[{"role": "user", "content": "public"}])),
            return_exceptions=True,
        )
        assert all(isinstance(result, Exception) for result in results)
        assert attempts == {"legacy": 2, "public": 1}
    finally:
        await client.aclose()
