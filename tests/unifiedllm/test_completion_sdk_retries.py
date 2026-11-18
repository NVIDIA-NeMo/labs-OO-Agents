# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RetryConfig is the only endpoint retry budget for OpenAI-compatible chat routes.

LiteLLM defaults the OpenAI SDK to two retries of its own, so each NOOA attempt
became three HTTP requests. These tests count requests at the HTTP transport.
"""

import asyncio
import json
from typing import Any

import httpx
import litellm
import pytest
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.llms.openai.openai import OpenAIChatCompletion

from nooa.unifiedllm import CompletionClient, RetryConfig
from nooa.unifiedllm.unifiedllm import _ClientHttp

OK_BODY = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "created": 0,
    "model": "m",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "done"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
MESSAGES: list[Any] = [{"role": "user", "content": "hi"}]


def _retry(max_retries: int) -> RetryConfig:
    return RetryConfig(
        max_retries=max_retries, rate_limit_extra_retries=0, base_delay=0.0, jitter_factor=0.0
    )


def _install(monkeypatch, transport: Any) -> None:
    """Route NOOA's owned clients and LiteLLM's default pool to ``transport``."""
    # LiteLLM caches SDK clients per event-loop id, which a later test's loop
    # can reuse; a cached client would still hold an earlier test's transport.
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", LLMClientCache())
    monkeypatch.setattr(
        _ClientHttp, "_httpx_hardening", staticmethod(lambda: {"transport": transport})
    )
    monkeypatch.setattr(
        OpenAIChatCompletion,
        "_get_sync_http_client",
        staticmethod(lambda **kwargs: httpx.Client(transport=transport)),
    )
    monkeypatch.setattr(
        OpenAIChatCompletion,
        "_get_async_http_client",
        staticmethod(lambda **kwargs: httpx.AsyncClient(transport=transport)),
    )


def _client(nooa_retries: int, **config) -> CompletionClient:
    return CompletionClient(
        "openai/m",
        api_base="https://endpoint.example/v1",
        api_key="key",
        retry_config=_retry(nooa_retries),
        **config,
    )


async def _call(client: CompletionClient, is_async: bool, **kwargs):
    if is_async:
        return await client.acall(MESSAGES, **kwargs)
    return client.call(MESSAGES, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("max_retries", [0, 2])
@pytest.mark.parametrize(
    "overrides",
    [{}, {"api_key": "override-key"}],
    ids=["owned-client", "litellm-client"],
)
async def test_persistent_error_uses_exactly_the_configured_attempts(
    monkeypatch, is_async, max_retries, overrides
):
    attempts = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(500, json={"error": {"message": "boom"}})

    _install(monkeypatch, httpx.MockTransport(respond))
    client = _client(max_retries)
    try:
        with pytest.raises(Exception, match="(?i)internal"):
            await _call(client, is_async, **overrides)
    finally:
        await client.aclose()

    assert attempts == max_retries + 1


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_owned_sdk_client_has_no_hidden_retries(monkeypatch, is_async):
    _install(monkeypatch, httpx.MockTransport(lambda request: httpx.Response(200, json=OK_BODY)))
    client = _client(2)
    try:
        await _call(client, is_async)
        http = client._http
        assert http is not None
        sdk_client = http.async_client if is_async else http.sync_client
        assert sdk_client.max_retries == 0
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_transient_error_recovers_on_nooa_retry(monkeypatch, is_async):
    statuses: list[int] = []

    def respond(request: httpx.Request) -> httpx.Response:
        status = 503 if not statuses else 200
        statuses.append(status)
        if status == 503:
            return httpx.Response(503, json={"error": {"message": "unavailable"}})
        return httpx.Response(200, json=OK_BODY)

    _install(monkeypatch, httpx.MockTransport(respond))
    client = _client(2)
    try:
        response = await _call(client, is_async)
    finally:
        await client.aclose()

    assert response.content == "done"
    assert statuses == [503, 200]


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_explicit_sdk_retries_remain_opt_in(monkeypatch, is_async):
    attempts = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(500, json={"error": {"message": "boom"}})

    _install(monkeypatch, httpx.MockTransport(respond))
    client = _client(0, max_retries=1)
    try:
        with pytest.raises(Exception, match="(?i)internal"):
            await _call(client, is_async)
    finally:
        await client.aclose()

    assert attempts == 2


@pytest.mark.asyncio
async def test_cancellation_stops_without_further_attempts(monkeypatch):
    attempts = 0
    started = asyncio.Event()

    async def respond(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        started.set()
        await asyncio.sleep(60)
        return httpx.Response(200, json=OK_BODY)

    _install(monkeypatch, httpx.MockTransport(respond))
    client = _client(2)
    try:
        task = asyncio.create_task(client.acall(MESSAGES))
        await asyncio.wait_for(started.wait(), timeout=10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=10)
        # Neither NOOA nor the SDK may treat cancellation as a retryable failure.
        await asyncio.sleep(0.2)
    finally:
        await client.aclose()

    assert attempts == 1


@pytest.mark.asyncio
async def test_concurrent_calls_keep_their_own_sdk_retry_setting(monkeypatch):
    # LiteLLM writes max_retries onto the SDK client it is given. Concurrent
    # calls on one NOOA client must not read each other's value.
    attempts = {"explicit": 0, "default": 0}

    async def respond(request: httpx.Request) -> httpx.Response:
        caller = "explicit" if b"explicit" in request.content else "default"
        attempts[caller] += 1
        await asyncio.sleep(0.05)
        return httpx.Response(500, json={"error": {"message": "boom"}})

    _install(monkeypatch, httpx.MockTransport(respond))
    client = _client(0)
    try:
        explicit = client.acall([{"role": "user", "content": "explicit"}], max_retries=2)
        default = client.acall([{"role": "user", "content": "default"}])
        results = await asyncio.gather(explicit, default, return_exceptions=True)
    finally:
        await client.aclose()

    assert all(isinstance(result, Exception) for result in results)
    assert attempts == {"explicit": 3, "default": 1}


@pytest.mark.asyncio
async def test_caller_supplied_sdk_client_is_not_shared_between_calls(monkeypatch):
    from openai import AsyncOpenAI

    attempts = {"explicit": 0, "default": 0}

    async def respond(request: httpx.Request) -> httpx.Response:
        caller = "explicit" if b"explicit" in request.content else "default"
        attempts[caller] += 1
        await asyncio.sleep(0.05)
        return httpx.Response(500, json={"error": {"message": "boom"}})

    transport = httpx.MockTransport(respond)
    _install(monkeypatch, transport)
    supplied = AsyncOpenAI(
        api_key="key",
        base_url="https://endpoint.example/v1",
        http_client=httpx.AsyncClient(transport=transport),
    )
    client = _client(0, client=supplied)
    try:
        explicit = client.acall([{"role": "user", "content": "explicit"}], max_retries=2)
        default = client.acall([{"role": "user", "content": "default"}])
        results = await asyncio.gather(explicit, default, return_exceptions=True)
    finally:
        await client.aclose()
        await supplied.close()

    assert all(isinstance(result, Exception) for result in results)
    assert attempts == {"explicit": 3, "default": 1}
    # The caller's own client is left as configured.
    assert supplied.max_retries == 2


def test_sync_call_copies_caller_supplied_sdk_client(monkeypatch):
    from openai import OpenAI

    attempts = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(500, json={"error": {"message": "boom"}})

    transport = httpx.MockTransport(respond)
    _install(monkeypatch, transport)
    supplied = OpenAI(
        api_key="key",
        base_url="https://endpoint.example/v1",
        http_client=httpx.Client(transport=transport),
    )
    client = _client(0, client=supplied)
    try:
        with pytest.raises(Exception, match="(?i)internal"):
            client.call(MESSAGES)
    finally:
        client.close()
        supplied.close()

    assert attempts == 1
    assert supplied.max_retries == 2


def _image_count(request: httpx.Request) -> int:
    body = json.loads(request.content)
    return sum(
        1
        for message in body["messages"]
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "image_url"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("sdk_client", [False, True], ids=["owned-client", "caller-client"])
async def test_media_recovery_requests_get_no_sdk_retries(monkeypatch, is_async, sdk_client):
    """#597 re-review: an explicit SDK retry override must not multiply the recovery budget.

    Every request body the endpoint has not seen three times gets a generic
    500 first (so SDK retries would show up as extra requests); then it
    rejects any request still carrying the bad image.
    """
    import openai

    images = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,aW1n{i:04d}"}}
        for i in range(8)
    ]
    bad = images[3]["image_url"]["url"]
    seen: dict[bytes, int] = {}
    counts: list[int] = []
    first_rejection: list[int] = []

    def respond(request: httpx.Request) -> httpx.Response:
        counts.append(_image_count(request))
        seen[request.content] = seen.get(request.content, 0) + 1
        if seen[request.content] <= 3:
            return httpx.Response(500, json={"error": {"message": "boom"}})
        if bad in request.content.decode():
            first_rejection.append(len(counts))
            return httpx.Response(
                400, json={"error": {"message": "Unsupported image format: svg+xml"}}
            )
        return httpx.Response(200, json=OK_BODY)

    transport = httpx.MockTransport(respond)
    _install(monkeypatch, transport)
    config: dict[str, Any] = {"media_inputs": ["image"], "cache_breakpoint": None}
    if sdk_client:
        sdk_type = openai.AsyncOpenAI if is_async else openai.OpenAI
        http_type = httpx.AsyncClient if is_async else httpx.Client
        config["client"] = sdk_type(
            api_key="key",
            base_url="https://endpoint.example/v1",
            http_client=http_type(transport=transport),
            max_retries=3,
        )
    client = _client(1, **config)
    budget = client._media_recovery_attempts()
    messages = [{"role": "user", "content": images}]
    try:
        if is_async:
            await client.acall(messages, max_retries=3)
        else:
            client.call(messages, max_retries=3)
    except litellm.BadRequestError:
        pass  # the budget may end before the search does; the bound is what matters
    finally:
        await client.aclose()

    assert budget == 4
    # The original request may use the caller's SDK retries (3 x 500, then the 400).
    original = first_rejection[0]
    assert counts[:original] == [8] * original
    print("wire image counts:", counts)
    assert len(counts) - original <= budget, counts
    assert all(count < 8 for count in counts[original:])
