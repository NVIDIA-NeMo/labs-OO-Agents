# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native feature matrix through actual SDK HTTP, never patched create methods."""

import json

import httpx
import pytest

from nooa.llm_types import CacheBoundary, LLMResponse
from nooa.unifiedllm import load_llm, registry
from nooa.unifiedllm._accounting import _accounting_receiver
from nooa.unifiedllm.adapters._native_failures import _failure
from nooa.unifiedllm.errors import FailureKind, LLMFailure
from nooa.unifiedllm.request import LLMRequest
from tests.unifiedllm.test_native_load_llm import borrowed, dispose, load, zero_delay
from tests.unifiedllm.test_native_projection import FAMILIES, Answer, payload


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(registry, "_loaded", True)
    monkeypatch.setattr(registry, "MODELS", {})


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_actual_media_reasoning_cache_tools_and_native_roundtrip(family, mode):
    arguments = '{ "query" : "x" }'
    data = payload(family, "answer")
    if family == "openai-chat":
        data["choices"][0]["message"].update(
            reasoning_content="thought",
            opaque="private-native",
            tool_calls=[
                {
                    "id": "call_bad-id",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": arguments},
                }
            ],
        )
        data["choices"][0]["finish_reason"] = "tool_calls"
        options = {"reasoning_effort": "low", "extra_body": {"host_extension": True}}
    elif family == "openai-responses":
        data["output"] = [
            {
                "id": "r",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "thought"}],
                "encrypted_content": "private-native",
            },
            {
                "id": "c",
                "type": "function_call",
                "call_id": "call_bad-id",
                "name": "lookup",
                "arguments": arguments,
                "status": "completed",
            },
            data["output"][0],
        ]
        options = {
            "reasoning": {"effort": "low"},
            "include": ["reasoning.encrypted_content"],
            "cache_breakpoint": "openai",
        }
    else:
        data["content"] = [
            {"type": "thinking", "thinking": "thought", "signature": "private-native"},
            {"type": "redacted_thinking", "data": "private-redacted"},
            {"type": "text", "text": " "},
            {"type": "tool_use", "id": "call_bad-id", "name": "lookup", "input": {"query": "x"}},
            {"type": "text", "text": "answer"},
        ]
        data["stop_reason"] = "tool_use"
        options = {
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "max_completion_tokens": 2048,
            "cache_breakpoint": "anthropic",
        }
    seen = []

    def handler(request):
        assert request.method == "POST" and request.url.host == "fixture.invalid"
        assert request.headers["x-private-host"] == "fixture"
        if family == "anthropic-messages":
            assert request.headers["anthropic-version"] == "2023-06-01"
        assert b"fixture-key" not in request.content
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=data)

    sdk, http = borrowed(family, mode, handler)
    if family == "anthropic-messages":
        llm = load_llm(
            "native",
            adapter=family,
            model="model",
            base_url="https://fixture.invalid/v1",
            api_key="fixture-key",
            client=sdk,
            headers={"x-private-host": "fixture"},
            media_inputs=["image", "pdf"],
            retry_config=zero_delay(),
            **options,
        )
    else:
        llm = load(
            family,
            sdk,
            media_inputs=["image"] if family == "openai-chat" else ["image", "pdf"],
            **options,
        )
    blocks = [
        {"type": "text", "text": "question"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
    ]
    if family != "openai-chat":
        blocks.append(
            {
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": "cGRm"},
            }
        )
    request = LLMRequest(
        messages=[
            {"role": "system", "content": "stable"},
            CacheBoundary(),
            {"role": "user", "content": blocks},
        ]
    )
    response = llm.call(request) if mode == "sync" else await llm.acall(request)
    assert response.content == (" answer" if family == "anthropic-messages" else "answer")
    assert response.reasoning == "thought" and response.tool_calls[0].arguments == (
        '{"query":"x"}' if family == "anthropic-messages" else arguments
    )
    restored = LLMResponse.model_validate_json(response.model_dump_json())
    before = restored.model_dump_json()
    follow = LLMRequest(
        messages=[restored, {"role": "tool", "tool_call_id": "call_bad-id", "content": "result"}]
    )
    if mode == "sync":
        llm.call(follow)
    else:
        await llm.acall(follow)
    body = seen[0]
    assert "CacheBoundary" not in json.dumps(body)
    if family == "openai-chat":
        assert body["reasoning_effort"] == "low" and body["host_extension"] is True
        assert body["messages"][1]["content"][1]["type"] == "image_url"
        assert seen[1]["messages"][0] == data["choices"][0]["message"]
        assert seen[1]["messages"][1]["tool_call_id"] == "call_bad-id"
    elif family == "openai-responses":
        assert body["reasoning"] == {"effort": "low"} and body["include"] == [
            "reasoning.encrypted_content"
        ]
        assert body["input"][1]["content"][1]["type"] == "input_image"
        assert body["input"][1]["content"][2]["file_data"] == "data:application/pdf;base64,cGRm"
        assert seen[1]["input"][:-1] == data["output"]
        assert seen[1]["input"][-1]["call_id"] == "call_bad-id"
    else:
        assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in body["messages"][0]["content"][0]
        assert body["messages"][0]["content"][1]["type"] == "image"
        assert body["messages"][0]["content"][2]["type"] == "document"
        assert seen[1]["messages"][0]["content"] == data["content"]
        assert seen[1]["messages"][1]["content"][0]["tool_use_id"] == "call_bad-id"
    assert restored.model_dump_json() == before
    # Different model drops opaque replay, same model restores it unchanged.
    other = load_llm(
        "native",
        adapter=family,
        model_name="other",
        base_url="https://fixture.invalid/v1",
        api_key="fixture-key",
        client=sdk,
        headers={"x-private-host": "fixture"},
        max_tokens=100,
        retry_config=False,
    )
    if mode == "sync":
        other.call(LLMRequest(messages=[restored]))
    else:
        await other.acall(LLMRequest(messages=[restored]))
    assert "private-native" not in json.dumps(seen[-1])
    assert restored.model_dump_json() == before
    await other.aclose()
    await llm.aclose()
    assert not http.is_closed
    await dispose(sdk, mode)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("finish", ["length", "refusal"])
async def test_actual_paid_length_vs_refusal(family, mode, finish):
    data = payload(family, "bad json", "length" if finish == "length" else None)
    if finish == "refusal":
        if family == "openai-chat":
            data["choices"][0]["message"]["refusal"] = "refused"
        elif family == "openai-responses":
            data["output"][0]["content"] = [{"type": "refusal", "refusal": "refused"}]
        else:
            data["stop_reason"] = "refusal"
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=data, headers={"x-response-cost": "0.03"})

    sdk, http = borrowed(family, mode, handler)
    llm = load(family, sdk, retry_config=zero_delay(3))
    receipts = []
    with _accounting_receiver(receipts.append), pytest.raises(LLMFailure) as caught:
        request = LLMRequest(messages=[{"role": "user", "content": "q"}], output_type=Answer)
        llm.call(request) if mode == "sync" else await llm.acall(request)
    assert caught.value.kind == (
        FailureKind.OUTPUT_LIMIT if finish == "length" else FailureKind.BAD_REQUEST
    )
    assert len(seen) == len(receipts) == 1
    assert receipts[0].subtotal("cost_usd").known == 0.03
    await llm.aclose()
    assert not http.is_closed
    await dispose(sdk, mode)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    "code,kind",
    [
        ("context_length_exceeded", FailureKind.CONTEXT_WINDOW),
        ("image_count_exceeded", FailureKind.MEDIA_LIMIT),
        ("invalid_image", FailureKind.MEDIA_REJECTED),
        ("max_tokens_exceeded", FailureKind.OUTPUT_LIMIT),
        ("invalid_request_error", FailureKind.BAD_REQUEST),
    ],
)
async def test_actual_structured_failure_facts_and_terminal_outer_status(family, code, kind):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            400,
            json={
                "error": {
                    "code": code,
                    "type": "invalid_request_error",
                    "message": "private 503 timeout",
                    "prompt_tokens": 0,
                    "context_window": 100,
                    "media_kind": "image",
                    "media_count": 3,
                    "media_count_limit": 2,
                }
            },
        )

    sdk, http = borrowed(family, "async", handler)
    llm = load(family, sdk)
    with pytest.raises(LLMFailure) as caught:
        await llm.acall(LLMRequest(messages=[{"role": "user", "content": "q"}]))
    error = caught.value
    assert error.kind == kind and len(seen) == 1
    assert error.status_code == 400
    if kind != FailureKind.BAD_REQUEST:
        assert error.prompt_tokens == 0 and error.context_window == 100
        assert error.media_count == 3 and error.media_count_limit == 2
    error.__cause__.__cause__ = TimeoutError("misleading")
    error.__cause__.status_code = 401
    assert _failure(error.__cause__).kind == FailureKind.AUTH
    assert "private" not in str(error)
    await llm.aclose()
    assert not http.is_closed
    await sdk.close()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("error_type", [httpx.ReadTimeout, httpx.ConnectError])
async def test_actual_sdk_transport_timeout_connection(family, mode, error_type):
    seen = []

    def handler(request):
        seen.append(request)
        raise error_type("private-route", request=request)

    sdk, http = borrowed(family, mode, handler)
    llm = load(family, sdk, retry_config=False)
    with pytest.raises(LLMFailure) as caught:
        request = LLMRequest(messages=[{"role": "user", "content": "q"}])
        llm.call(request) if mode == "sync" else await llm.acall(request)
    assert caught.value.kind == (
        FailureKind.TIMEOUT if error_type is httpx.ReadTimeout else FailureKind.TRANSIENT
    )
    assert len(seen) == 1 and "private-route" not in str(caught.value)
    await llm.aclose()
    assert not http.is_closed
    await dispose(sdk, mode)
