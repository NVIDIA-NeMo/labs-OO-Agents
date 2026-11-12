# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real core SDK transport/serialization evidence; no patched create methods."""

import json

import anthropic
import httpx
import openai
import pytest

from nooa.unifiedllm.adapters._native_failures import _failure
from nooa.unifiedllm.adapters._native_transport import _NativeTransport, _resource
from nooa.unifiedllm.errors import FailureKind, LLMFailure


@pytest.mark.parametrize("family", ["openai-chat", "openai-responses", "anthropic-messages"])
@pytest.mark.parametrize("version_root", [False, True])
async def test_real_native_binding_and_wire(family, version_root):
    seen = []
    is_anthropic = family == "anthropic-messages"
    base = "https://fixture.invalid" + ("" if is_anthropic else "/v1")
    declared = base + ("/v1" if is_anthropic and version_root else "")
    path = {
        "openai-chat": "/v1/chat/completions",
        "openai-responses": "/v1/responses",
        "anthropic-messages": "/v1/messages",
    }[family]

    def handler(request):
        assert request.method == "POST" and request.url.path == path
        assert request.headers["x-private-host"] == "fixture"
        assert request.headers["x-api-key" if is_anthropic else "authorization"] == (
            "private-fixture-key" if is_anthropic else "Bearer private-fixture-key"
        )
        assert str(request.url) == base.split("/v1")[0] + path
        body = json.loads(request.content)
        assert body["model"] == "fixture"
        assert "api_key" not in body and "private-fixture-key" not in request.content.decode()
        seen.append(body)
        if family == "openai-chat":
            payload = {
                "id": "chat-1",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
            }
        elif family == "openai-responses":
            payload = {
                "id": "resp-1",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "fixture",
                "output": [
                    {
                        "id": "msg-1",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9},
            }
        else:
            payload = {
                "id": "msg-1",
                "type": "message",
                "role": "assistant",
                "model": "fixture",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 7,
                    "output_tokens": 2,
                    "cache_read_input_tokens": 3,
                    "cache_creation_input_tokens": 4,
                },
            }
        return httpx.Response(200, json=payload, headers={"x-fixture-cost": "0.03"})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    ahttp = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    cls, acls = (
        (anthropic.Anthropic, anthropic.AsyncAnthropic)
        if is_anthropic
        else (openai.OpenAI, openai.AsyncOpenAI)
    )
    original = cls(
        api_key="private-fixture-key",
        base_url=base,
        http_client=http,
        max_retries=9,
        default_headers={"x-private-host": "fixture"},
    )
    aoriginal = acls(
        api_key="private-fixture-key",
        base_url=base,
        http_client=ahttp,
        max_retries=9,
        default_headers={"x-private-host": "fixture"},
    )
    bound = _NativeTransport(
        family,
        declared,
        "private-fixture-key",
        client={"sync": original, "async": aoriginal},
        headers={"x-private-host": "fixture"},
    )
    kwargs = {"model": "fixture"}
    if family == "openai-responses":
        kwargs["input"] = [{"role": "user", "content": "hello"}]
    else:
        kwargs["messages"] = [{"role": "user", "content": "hello"}]
    if is_anthropic:
        kwargs["max_tokens"] = 100
    raw = _resource(bound._sdk("sync"), family).with_raw_response.create(**kwargs)
    reply = raw.parse()
    araw = await _resource(bound._sdk("async"), family).with_raw_response.create(**kwargs)
    areply = araw.parse()
    expected = {
        "openai-chat": openai.types.chat.ChatCompletion,
        "openai-responses": openai.types.responses.Response,
        "anthropic-messages": anthropic.types.Message,
    }[family]
    assert type(reply) is type(areply) is expected
    assert raw.headers["x-fixture-cost"] == araw.headers["x-fixture-cost"] == "0.03"
    assert len(seen) == 2
    assert original.max_retries == aoriginal.max_retries == 9
    assert bound._sdk("sync").max_retries == bound._sdk("async").max_retries == 0
    assert bound._sdk("sync")._client is http and bound._sdk("async")._client is ahttp
    await bound.aclose()
    await bound.aclose()
    assert not http.is_closed and not ahttp.is_closed
    http.close()
    await ahttp.aclose()


@pytest.mark.parametrize("family", ["openai-chat", "openai-responses", "anthropic-messages"])
@pytest.mark.parametrize(
    "status,kind",
    [
        (429, FailureKind.RATE_LIMIT),
        (408, FailureKind.TIMEOUT),
        (503, FailureKind.TRANSIENT),
        (529, FailureKind.TRANSIENT),
        (401, FailureKind.AUTH),
    ],
)
def test_actual_sdk_http_failure_identity(family, status, kind):
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        return httpx.Response(
            status,
            json={
                "error": {
                    "type": "rate_limit_error",
                    "message": "misleading inner timeout status 503",
                }
            },
            headers={"retry-after": "0.25"},
        )

    http = httpx.Client(transport=httpx.MockTransport(handler))
    cls = anthropic.Anthropic if family == "anthropic-messages" else openai.OpenAI
    sdk = cls(
        api_key="fixture", base_url="https://fixture.invalid", http_client=http, max_retries=9
    )
    transport = _NativeTransport(family, "https://fixture.invalid", "fixture", client=sdk)
    kwargs = {"model": "fixture"}
    kwargs["input" if family == "openai-responses" else "messages"] = [
        {"role": "user", "content": "hi"}
    ]
    if family == "anthropic-messages":
        kwargs["max_tokens"] = 100
    with pytest.raises((openai.APIStatusError, anthropic.APIStatusError)) as caught:
        _resource(transport._sdk("sync"), family).create(**kwargs)
    failure = _failure(caught.value)
    assert failure.kind == kind and failure.status_code == status and failure.retry_after == 0.25
    assert failure.__cause__ is caught.value and count == 1 and sdk.max_retries == 9
    http.close()


async def test_owned_roots_close_once_and_missing_borrowed_mode_fails():
    transport = _NativeTransport("anthropic-messages", "https://fixture.invalid/v1", "fixture")
    assert transport.endpoint == "https://fixture.invalid"
    transport._initialize()
    sync, async_ = transport._owned_sync, transport._owned_async
    assert sync is not async_ and not sync.is_closed and not async_.is_closed
    await transport.aclose()
    await transport.aclose()
    assert sync.is_closed and async_.is_closed
    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    sdk = openai.OpenAI(api_key="fixture", base_url="https://fixture.invalid", http_client=http)
    borrowed = _NativeTransport("openai-chat", "https://fixture.invalid", "fixture", client=sdk)
    with pytest.raises(LLMFailure) as caught:
        borrowed._sdk("async")
    assert caught.value.kind == FailureKind.CONFIG
    await borrowed.aclose()
    assert not http.is_closed
    http.close()


def test_borrowed_bad_effective_anthropic_root_rejected_without_mutation():
    http = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    sdk = anthropic.Anthropic(
        api_key="fixture", base_url="https://fixture.invalid/v1", http_client=http
    )
    with pytest.raises(LLMFailure) as caught:
        _NativeTransport("anthropic-messages", "https://fixture.invalid/v1", "fixture", client=sdk)
    assert caught.value.kind == FailureKind.CONFIG and str(sdk.base_url).endswith("/v1/")
    assert not http.is_closed
    http.close()


@pytest.mark.parametrize("family", ["openai-chat", "openai-responses", "anthropic-messages"])
@pytest.mark.parametrize("borrowed", [False, True])
async def test_actual_timeout_extensions(family, borrowed, monkeypatch):
    from nooa.unifiedllm.http_config import HttpConfig

    seen = []

    def handler(request):
        seen.append(request.extensions["timeout"])
        return httpx.Response(
            401, json={"error": {"message": "fixture", "type": "authentication_error"}}
        )

    real_sync, real_async = httpx.Client, httpx.AsyncClient
    http = real_sync(transport=httpx.MockTransport(handler))
    ahttp = real_async(transport=httpx.MockTransport(handler))
    config = HttpConfig(connect_timeout=1, read_timeout=2, write_timeout=3, pool_timeout=4)
    if borrowed:
        cls, acls = (
            (anthropic.Anthropic, anthropic.AsyncAnthropic)
            if family == "anthropic-messages"
            else (openai.OpenAI, openai.AsyncOpenAI)
        )
        clients = {
            "sync": cls(
                api_key="fixture",
                base_url="https://fixture.invalid",
                http_client=http,
                timeout=99,
                max_retries=9,
            ),
            "async": acls(
                api_key="fixture",
                base_url="https://fixture.invalid",
                http_client=ahttp,
                timeout=99,
                max_retries=9,
            ),
        }
    else:
        clients = None

        class MockSync(real_sync):
            def __init__(self, **kw):
                super().__init__(transport=httpx.MockTransport(handler), **kw)

        class MockAsync(real_async):
            def __init__(self, **kw):
                super().__init__(transport=httpx.MockTransport(handler), **kw)

        monkeypatch.setattr(httpx, "Client", MockSync)
        monkeypatch.setattr(httpx, "AsyncClient", MockAsync)
    root = _NativeTransport(
        family, "https://fixture.invalid", "fixture", client=clients, http_config=config
    )
    root._initialize()
    kwargs = {
        "model": "fixture",
        "input" if family == "openai-responses" else "messages": [
            {"role": "user", "content": "hi"}
        ],
    }
    if family == "anthropic-messages":
        kwargs["max_tokens"] = 20
    for timeout in [None, 11]:
        with pytest.raises((openai.APIStatusError, anthropic.APIStatusError)):
            _resource(root._sdk("sync", timeout), family).create(**kwargs)
        with pytest.raises((openai.APIStatusError, anthropic.APIStatusError)):
            await _resource(root._sdk("async", timeout), family).create(**kwargs)
    assert (
        seen
        == [{"connect": 1, "read": 2, "write": 3, "pool": 4}] * 2
        + [{"connect": 11, "read": 11, "write": 11, "pool": 11}] * 2
    )
    if borrowed:
        assert all(sdk.timeout == 99 and sdk.max_retries == 9 for sdk in clients.values())
    await root.aclose()
    http.close()
    await ahttp.aclose()


async def test_partial_construct_pool_retains_cleanup_and_cancel_join(monkeypatch):
    import asyncio

    from nooa.unifiedllm.adapters._loader import _Pool

    closed = []
    entered, release = asyncio.Event(), asyncio.Event()
    real_sync, real_async = httpx.Client, httpx.AsyncClient

    class CountingSync(real_sync):
        def close(self):
            closed.append("sync")
            super().close()

    class GatedAsync(real_async):
        async def aclose(self):
            closed.append("async-enter")
            entered.set()
            await release.wait()
            await super().aclose()
            closed.append("async-exit")

    monkeypatch.setattr(httpx, "Client", CountingSync)
    monkeypatch.setattr(httpx, "AsyncClient", GatedAsync)

    class Abort(BaseException):
        pass

    abort = Abort()
    original = _NativeTransport._construct

    def construct(self, mode, http):
        if mode == "async":
            raise abort
        return original(self, mode, http)

    monkeypatch.setattr(_NativeTransport, "_construct", construct)
    root = _NativeTransport("openai-chat", "https://fixture.invalid", "fixture")
    pool = _Pool(root)
    pool.acquire()
    with pytest.raises(Abort) as caught:
        root._initialize()
    assert caught.value is abort and not root._owned_sync.is_closed
    pool.release()
    await entered.wait()
    joiner = asyncio.create_task(pool.join_disposal())
    await asyncio.sleep(0)
    joiner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await joiner
    assert not pool._disposal.done() and pool.retired
    release.set()
    await asyncio.gather(pool.join_disposal(), pool.join_disposal())
    assert closed == ["sync", "async-enter", "async-exit"]
    assert root._owned_sync.is_closed and root._owned_async.is_closed
