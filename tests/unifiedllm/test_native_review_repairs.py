# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded independent-review regressions using actual SDK HTTP and pure loads."""

import json

import anthropic
import httpx
import openai
import pytest

from nooa.unifiedllm import load_llm, registry
from nooa.unifiedllm._accounting import _accounting_receiver
from nooa.unifiedllm.adapters._binding import _bind_adapter
from nooa.unifiedllm.adapters._native_settings import _resolve_native
from nooa.unifiedllm.errors import FailureKind, LLMFailure
from nooa.unifiedllm.request import LLMRequest
from tests.unifiedllm.test_native_projection import FAMILIES, payload


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(registry, "_loaded", True)
    monkeypatch.setattr(registry, "MODELS", {})


def resolve(family, **overrides):
    return _resolve_native(
        family,
        "native",
        {
            "model": "model",
            "base_url": "https://route-a.invalid/v1",
            "api_key": "private-key-a",
            "max_tokens": 100,
        },
        overrides,
    )[0]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    "change",
    [
        {"base_url": "https://route-b.invalid/v1"},
        {"api_key": "private-key-b"},
        {"headers": {"x-private-host": "other"}},
        {"default_query": {"tenant": "other"}},
        {"host_profile": {"cost_header": "x-response-cost"}},
        {"extra_headers": {"x-request-tenant": "other"}},
        {"model": "other-model"},
    ],
)
def test_route_only_scope_protects_each_identity_component(family, change):
    a = resolve(family)
    b = resolve(family, **change)
    assert a.replay_scope != b.replay_scope
    assert a.replay_scope == resolve(family).replay_scope
    assert "private-key" not in a.replay_scope and "route-a" not in a.replay_scope
    assert len(a.replay_scope.rsplit(":", 1)[1]) == 64


@pytest.mark.parametrize("family", FAMILIES)
def test_scope_excludes_defaults_facts_and_isolates_unprojectable_identity(family):
    a = resolve(family)
    b = resolve(family, max_tokens=200, temperature=0.5, context_window=9999)
    assert a.replay_scope == b.replay_scope
    huge = {"x-host-state": "x" * 9000}
    assert resolve(family, headers=huge).replay_scope != resolve(family, headers=huge).replay_scope


@pytest.mark.parametrize("family", ["openai-chat", "openai-responses"])
@pytest.mark.parametrize("change", [{"organization": "org-b"}, {"project": "project-b"}])
def test_scope_protects_organization_and_project(family, change):
    assert resolve(family).replay_scope != resolve(family, **change).replay_scope


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("change", ["endpoint", "key"])
async def test_actual_sdk_a_b_a_preserves_opaque_state_without_cross_route_leak(family, change):
    seen = [[], []]
    data = payload(family, "answer")
    if family == "openai-chat":
        data["choices"][0]["message"]["reasoning_content"] = "private-state-a"
    elif family == "openai-responses":
        data["output"].insert(
            0,
            {
                "id": "rs",
                "type": "reasoning",
                "summary": [],
                "encrypted_content": "private-state-a",
            },
        )
    else:
        data["content"].insert(
            0,
            {
                "type": "thinking",
                "thinking": "thought",
                "signature": "private-state-a",
            },
        )
    roots, llms = [], []
    for index in range(2):
        endpoint = (
            "https://route-b.invalid/v1"
            if index and change == "endpoint"
            else "https://route-a.invalid/v1"
        )
        key = "private-key-b" if index and change == "key" else "private-key-a"

        def handler(request, index=index):
            seen[index].append(json.loads(request.content))
            return httpx.Response(200, json=data)

        http = httpx.Client(transport=httpx.MockTransport(handler))
        cls = anthropic.Anthropic if family == "anthropic-messages" else openai.OpenAI
        sdk = cls(
            api_key=key,
            base_url=endpoint.removesuffix("/v1") if family == "anthropic-messages" else endpoint,
            max_retries=9,
            http_client=http,
        )
        roots.append((sdk, http))
        llms.append(
            load_llm(
                "native",
                adapter=family,
                model="model",
                base_url=endpoint,
                api_key=key,
                client=sdk,
                max_tokens=100,
                retry_config=False,
            )
        )
    try:
        a, b = llms
        response = a.call(LLMRequest(messages=[{"role": "user", "content": "first"}]))
        before = response.model_dump_json()
        history = LLMRequest(messages=[response, {"role": "user", "content": "continue"}])
        b.call(history)
        assert "private-state-a" not in json.dumps(seen[1][0])
        view = _bind_adapter(a.adapter, max_tokens=200, temperature=0.5)
        assert view._settings.replay_scope == a.adapter._settings.replay_scope
        a.call(history)
        assert "private-state-a" in json.dumps(seen[0][1])
        assert response.model_dump_json() == before
        # A distinct borrowed SDK at otherwise identical route is also isolated.
        same = resolve(family, client=roots[0][0])
        other = resolve(family, client=roots[1][0])
        assert same.replay_scope != other.replay_scope
    finally:
        for llm, (sdk, http) in zip(llms, roots, strict=True):
            llm.close()
            assert not http.is_closed and sdk.max_retries == 9
            sdk.close()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    "knob", ["temperature", "max_tokens", "stream", "thinking", "reasoning", "prediction"]
)
def test_recognized_extra_body_controls_reject_even_other_family_or_none(family, knob):
    with pytest.raises(LLMFailure) as caught:
        resolve(family, extra_body={knob: None})
    assert caught.value.kind == FailureKind.CONFIG
    assert resolve(family, extra_body={"host_extension": {"enabled": True}}).selected["extra_body"][
        "host_extension"
    ]["enabled"]


@pytest.mark.parametrize(
    "block",
    [
        {"type": "output_audio", "data": "AAAA", "transcript": "semantic output"},
        {"type": "output_text", "text": None},
        {"type": "refusal", "refusal": None},
    ],
)
def test_malformed_or_unsupported_responses_block_terminal_after_paid_receipt(block):
    data = payload("openai-responses", "answer")
    data["output"][0]["content"].append(block)
    seen, receipts = [], []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=data, headers={"x-response-cost": "0.03"})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    sdk = openai.OpenAI(
        api_key="private-key-a",
        base_url="https://route-a.invalid/v1",
        http_client=http,
        max_retries=9,
    )
    llm = load_llm(
        "native",
        adapter="openai-responses",
        model="model",
        base_url="https://route-a.invalid/v1",
        api_key="private-key-a",
        client=sdk,
        host_profile={"cost_header": "x-response-cost"},
        max_tokens=100,
    )
    try:
        with _accounting_receiver(receipts.append), pytest.raises(LLMFailure) as caught:
            llm.call(LLMRequest(messages=[{"role": "user", "content": "question"}]))
        assert caught.value.kind == FailureKind.BAD_REQUEST
        assert len(seen) == len(receipts) == 1
        assert receipts[0].subtotal("cost_usd").known == pytest.approx(0.03)
        assert receipts[0].subtotal("cost_usd").complete
        assert receipts[0].attempts[0].response_received
    finally:
        llm.close()
        assert not http.is_closed
        sdk.close()


@pytest.mark.parametrize("family", ["openai-responses", "anthropic-messages"])
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "spelling", ["file_uri", "source_raw", "file_raw", "file_url", "source_url"]
)
@pytest.mark.parametrize("role", ["user", "tool"])
async def test_actual_sdk_pdf_carriers_preserve_bytes_and_tool_result(family, mode, spelling, role):
    seen = []
    opaque = "AbC+/09=="
    uri = "data:application/pdf;base64," + opaque
    url = "https://files.invalid/sample.pdf"
    block = {
        "file_uri": {"type": "input_file", "file_data": uri, "filename": "sample.pdf"},
        "source_raw": {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf", "data": opaque},
        },
        "file_raw": {
            "type": "input_file",
            "file_data": opaque,
            "mime_type": "application/pdf",
            "filename": "sample.pdf",
        },
        "file_url": {"type": "input_file", "file_url": url, "mime_type": "application/pdf"},
        "source_url": {
            "type": "document",
            "source": {"type": "url", "url": url, "media_type": "application/pdf"},
        },
    }[spelling]
    content = [{"type": "text", "text": "PDF"}, block]
    messages = [{"role": role, "content": content}]
    if role == "tool":
        messages[0]["tool_call_id"] = "call_1"
    before = json.dumps(messages)

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=payload(family, "answer"))

    http = (httpx.Client if mode == "sync" else httpx.AsyncClient)(
        transport=httpx.MockTransport(handler)
    )
    cls = (
        (anthropic.Anthropic if mode == "sync" else anthropic.AsyncAnthropic)
        if family == "anthropic-messages"
        else (openai.OpenAI if mode == "sync" else openai.AsyncOpenAI)
    )
    endpoint = (
        "https://route-a.invalid"
        if family == "anthropic-messages"
        else "https://route-a.invalid/v1"
    )
    sdk = cls(api_key="private-key-a", base_url=endpoint, http_client=http, max_retries=9)
    llm = load_llm(
        "native",
        adapter=family,
        model="model",
        base_url=endpoint,
        api_key="private-key-a",
        client=sdk,
        max_tokens=100,
        media_inputs=["pdf"],
        retry_config=False,
    )
    try:
        request = LLMRequest(messages=messages)
        response = llm.call(request) if mode == "sync" else await llm.acall(request)
        assert response.content == "answer"
        assert len(seen) == 1 and json.dumps(messages) == before
        is_url = spelling in {"file_url", "source_url"}
        if family == "openai-responses":
            item = seen[0]["input"][0]
            blocks = item["output"] if role == "tool" else item["content"]
            expected = (
                {"type": "input_file", "file_url": url}
                if is_url
                else {
                    "type": "input_file",
                    "file_data": uri,
                    "filename": block.get("filename", "document.pdf"),
                }
            )
            assert blocks == [{"type": "input_text", "text": "PDF"}, expected]
            if role == "tool":
                assert item["type"] == "function_call_output" and item["call_id"] == "call_1"
        else:
            blocks = seen[0]["messages"][0]["content"]
            if role == "tool":
                assert blocks[0]["type"] == "tool_result" and blocks[0]["tool_use_id"] == "call_1"
                blocks = blocks[0]["content"]
            source = (
                {"type": "url", "url": url}
                if is_url
                else {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": opaque,
                }
            )
            assert blocks == [
                {"type": "text", "text": "PDF"},
                {"type": "document", "source": source},
            ]
    finally:
        await llm.aclose()
        assert not http.is_closed and sdk.max_retries == 9
        if mode == "sync":
            sdk.close()
        else:
            await sdk.close()


@pytest.mark.parametrize("family", ["openai-responses", "anthropic-messages"])
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "uri,mime",
    [
        ("data:application/pdf;base64", None),
        ("data:application/pdf;base64,", None),
        ("data:application/pdf,AAAA", None),
        ("data:image/png;base64,AAAA", None),
        ("data:application/pdf;charset=utf-8;base64,AAAA", None),
        ("data:application/pdf;base64,AAAA", "image/png"),
    ],
)
async def test_explicit_pdf_uri_malformed_or_incompatible_is_local(family, mode, uri, mime):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=payload(family, "answer"))

    http = (httpx.Client if mode == "sync" else httpx.AsyncClient)(
        transport=httpx.MockTransport(handler)
    )
    cls = (
        (anthropic.Anthropic if mode == "sync" else anthropic.AsyncAnthropic)
        if family == "anthropic-messages"
        else (openai.OpenAI if mode == "sync" else openai.AsyncOpenAI)
    )
    endpoint = (
        "https://route-a.invalid"
        if family == "anthropic-messages"
        else "https://route-a.invalid/v1"
    )
    sdk = cls(api_key="private-key-a", base_url=endpoint, http_client=http, max_retries=9)
    llm = load_llm(
        "native",
        adapter=family,
        model="model",
        base_url=endpoint,
        api_key="private-key-a",
        client=sdk,
        max_tokens=100,
        media_inputs=["pdf"],
        retry_config=False,
    )
    block = {"type": "input_file", "file_data": uri, "filename": "sample.pdf"}
    if mime is not None:
        block["mime_type"] = mime
    before = dict(block)
    try:
        request = LLMRequest(messages=[{"role": "user", "content": [block]}])
        unsupported_file = uri.startswith("data:image/") or mime == "image/png"
        if unsupported_file:
            # Existing shared policy filters conflicting/non-PDF declarations
            # before native projection; do not misreport this as BAD_REQUEST.
            with pytest.raises(LLMFailure) as caught:
                if mode == "sync":
                    llm.adapter.send_sync(request)
                else:
                    await llm.adapter.send(request)
            assert caught.value.kind == FailureKind.MEDIA_REJECTED and not seen
            response = llm.call(request) if mode == "sync" else await llm.acall(request)
            assert response.content == "answer" and len(seen) == 1
            body = json.loads(seen[0].content)
            assert uri not in json.dumps(body) and "file_data" not in json.dumps(body)
            assert "A file was removed here" in json.dumps(body)
        else:
            with pytest.raises(LLMFailure) as caught:
                if mode == "sync":
                    llm.call(request)
                else:
                    await llm.acall(request)
            assert caught.value.kind == FailureKind.BAD_REQUEST and not seen
        assert block == before
    finally:
        await llm.aclose()
        assert not http.is_closed and sdk.max_retries == 9
        if mode == "sync":
            sdk.close()
        else:
            await sdk.close()
