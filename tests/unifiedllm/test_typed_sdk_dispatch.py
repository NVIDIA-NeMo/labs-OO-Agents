# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict failure policy across real installed SDK dispatch over HTTPX only."""

import httpx
import pytest

from nooa.unifiedllm import (
    LLM,
    CompletionClient,
    LLMFailure,
    LLMRequest,
    ResponsesClient,
    RetryConfig,
)
from nooa.unifiedllm import FailureKind as K
from nooa.unifiedllm.unifiedllm import _ClientHttp


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("client_type", [CompletionClient, ResponsesClient])
@pytest.mark.parametrize(
    "status, body, kind, attempts",
    [
        (429, {"message": "slow down"}, K.RATE_LIMIT, 2),
        (502, {"message": "gateway"}, K.TRANSIENT, 2),
        (408, {"message": "timeout"}, K.TIMEOUT, 2),
        (401, {"message": "rate limit status 503 timeout"}, K.AUTH, 1),
        (
            400,
            {
                "message": "context exceeded",
                "code": "context_length_exceeded",
                "prompt_tokens": 0,
                "context_window": 4096,
            },
            K.CONTEXT_WINDOW,
            1,
        ),
        (422, {"message": "rate limit status 503", "param": "messages"}, K.BAD_REQUEST, 1),
        (400, {"message": "invalid model", "code": "model_not_found"}, K.CONFIG, 1),
        (
            400,
            {
                "message": "too many images",
                "code": "image_limit_exceeded",
                "media_count": 2,
                "media_count_limit": 0,
            },
            K.MEDIA_LIMIT,
            1,
        ),
        (400, {"message": "invalid image", "code": "invalid_image"}, K.MEDIA_REJECTED, 1),
    ],
)
async def test_real_dispatch_failure(
    monkeypatch, is_async, client_type, status, body, kind, attempts
):
    requests = []
    delays = []

    def respond(request):
        requests.append(request)
        return httpx.Response(status, json={"error": body}, headers={"Retry-After": "3.5"})

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        _ClientHttp, "_httpx_hardening", staticmethod(lambda: {"transport": transport})
    )
    import nooa.unifiedllm.retry as retry

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(retry.asyncio, "sleep", sleep)
    monkeypatch.setattr(retry.time, "sleep", lambda delay: delays.append(delay))
    client = client_type(
        "openai/gpt-4o-mini", api_key="test", api_base="https://endpoint.test/v1", max_retries=9
    )
    llm = LLM(
        client._adapter,
        retry_config=RetryConfig(
            max_retries=1, rate_limit_extra_retries=0, base_delay=0, max_delay=0, jitter_factor=0
        ),
    )
    try:
        with pytest.raises(LLMFailure) as caught:
            request = LLMRequest(messages=[{"role": "user", "content": "hi"}])
            if is_async:
                await llm.acall(request)
            else:
                llm.call(request)
        error = caught.value
        assert error.kind == kind
        assert error.__cause__ is not None
        assert error.status_code == status
        assert len(requests) == attempts
        assert client.config["max_retries"] == 9
        if attempts == 2:
            assert error.retry_after == 3.5
            assert delays == [3.5]
        else:
            assert delays == []
        if kind == K.CONTEXT_WINDOW:
            assert error.prompt_tokens == 0 and error.context_window == 4096
        if kind == K.MEDIA_LIMIT:
            assert error.media_count == 2 and error.media_count_limit == 0
    finally:
        await client.aclose()


def success(client_type, content="ok"):
    if client_type is CompletionClient:
        return {
            "id": "c",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        }
    return {
        "id": "r",
        "object": "response",
        "created_at": 1,
        "model": "gpt-4o-mini",
        "status": "completed",
        "output": [
            {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": content, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
    }


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("client_type", [CompletionClient, ResponsesClient])
async def test_real_transient_then_success(monkeypatch, is_async, client_type):
    requests = []

    def respond(request):
        requests.append(request)
        return (
            httpx.Response(502, json={"error": {"message": "gateway"}})
            if len(requests) == 1
            else httpx.Response(200, json=success(client_type))
        )

    monkeypatch.setattr(
        _ClientHttp,
        "_httpx_hardening",
        staticmethod(lambda: {"transport": httpx.MockTransport(respond)}),
    )
    client = client_type(
        "openai/gpt-4o-mini", api_key="test", api_base="https://endpoint.test/v1", max_retries=9
    )
    llm = LLM(
        client._adapter,
        retry_config=RetryConfig(
            max_retries=1, rate_limit_extra_retries=0, base_delay=0, jitter_factor=0
        ),
    )
    try:
        request = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        result = await llm.acall(request) if is_async else llm.call(request)
        assert result.content == "ok"
        assert len(requests) == 2
        assert client.config["max_retries"] == 9
    finally:
        await client.aclose()


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("client_type", [CompletionClient, ResponsesClient])
@pytest.mark.parametrize("truncated", [False, True])
async def test_real_paid_invalid_output_is_terminal(monkeypatch, is_async, client_type, truncated):
    from pydantic import BaseModel, ValidationError

    class Answer(BaseModel):
        value: int

    requests = []

    def respond(request):
        requests.append(request)
        body = success(client_type, '{"value": "rate limit 503"}')
        if truncated:
            if client_type is CompletionClient:
                body["choices"][0]["finish_reason"] = "length"
            else:
                body["status"] = "incomplete"
                body["incomplete_details"] = {"reason": "max_output_tokens"}
        return httpx.Response(200, json=body)

    monkeypatch.setattr(
        _ClientHttp,
        "_httpx_hardening",
        staticmethod(lambda: {"transport": httpx.MockTransport(respond)}),
    )
    client = client_type(
        "openai/gpt-4o-mini", api_key="test", api_base="https://endpoint.test/v1", max_retries=9
    )
    llm = LLM(
        client._adapter,
        retry_config=RetryConfig(
            max_retries=2, rate_limit_extra_retries=0, base_delay=0, jitter_factor=0
        ),
    )
    try:
        request = LLMRequest(messages=[{"role": "user", "content": "hi"}], output_type=Answer)
        with pytest.raises(LLMFailure) as caught:
            if is_async:
                await llm.acall(request)
            else:
                llm.call(request)
        assert caught.value.kind == (K.OUTPUT_LIMIT if truncated else K.BAD_REQUEST)
        assert isinstance(caught.value.__cause__, (ValidationError, ValueError))
        assert len(requests) == 1
    finally:
        await client.aclose()


class Disconnect(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __iter__(self):
        yield b""
        raise httpx.ReadError("wire stream disconnected")

    async def __aiter__(self):
        yield b""
        raise httpx.ReadError("wire stream disconnected")


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("client_type", [CompletionClient, ResponsesClient])
async def test_real_stream_read_failure_is_transient(monkeypatch, is_async, client_type):
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200, headers={"Content-Type": "text/event-stream"}, stream=Disconnect()
        )

    monkeypatch.setattr(
        _ClientHttp,
        "_httpx_hardening",
        staticmethod(lambda: {"transport": httpx.MockTransport(respond)}),
    )
    client = client_type(
        "openai/gpt-4o-mini",
        api_key="test",
        api_base="https://endpoint.test/v1",
        stream=client_type is CompletionClient,
        max_retries=9,
    )
    llm = LLM(
        client._adapter,
        retry_config=RetryConfig(
            max_retries=1, rate_limit_extra_retries=0, base_delay=0, jitter_factor=0
        ),
    )
    try:
        request = LLMRequest(messages=[{"role": "user", "content": "hi"}])
        with pytest.raises(LLMFailure) as caught:
            if is_async:
                await llm.acall(request)
            else:
                llm.call(request)
        assert caught.value.kind == K.TRANSIENT
        assert len(requests) == 2
    finally:
        await client.aclose()
