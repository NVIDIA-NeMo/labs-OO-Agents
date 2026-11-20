# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public native loader and shared ledger through actual core SDK HTTP."""

import builtins
import json
import sys

import anthropic
import httpx
import openai
import pytest

from nooa.unifiedllm import load_llm, registry
from nooa.unifiedllm._accounting import _accounting_receiver
from nooa.unifiedllm._observations import _collect_observations
from nooa.unifiedllm.adapters._binding import _bind_adapter
from nooa.unifiedllm.errors import FailureKind, LLMFailure
from nooa.unifiedllm.request import LLMRequest
from nooa.unifiedllm.retry_config import RetryConfig
from tests.unifiedllm.test_native_projection import FAMILIES, Answer, payload


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(registry, "_loaded", True)
    monkeypatch.setattr(registry, "MODELS", {})


def zero_delay(retries=1, empty=False):
    return RetryConfig(
        max_retries=retries,
        rate_limit_extra_retries=0,
        base_delay=0,
        max_delay=0,
        jitter_factor=0,
        rate_limit_base_delay=0,
        retry_on_empty_content=empty,
    )


def borrowed(family, mode, handler):
    cls = (
        (anthropic.Anthropic if mode == "sync" else anthropic.AsyncAnthropic)
        if family == "anthropic-messages"
        else (openai.OpenAI if mode == "sync" else openai.AsyncOpenAI)
    )
    http = (
        httpx.Client(transport=httpx.MockTransport(handler))
        if mode == "sync"
        else httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    sdk = cls(
        api_key="fixture-key",
        base_url="https://fixture.invalid"
        if family == "anthropic-messages"
        else "https://fixture.invalid/v1",
        http_client=http,
        max_retries=9,
        default_headers={"x-private-host": "fixture"},
    )
    return sdk, http


def load(family, sdk, **knobs):
    return load_llm(
        "native",
        adapter=family,
        model="model",
        base_url="https://fixture.invalid/v1",
        api_key="fixture-key",
        client=sdk,
        headers={"x-private-host": "fixture"},
        max_tokens=100,
        host_profile={"cost_header": "x-response-cost"},
        retry_config=knobs.pop("retry_config", zero_delay()),
        **knobs,
    )


async def dispose(sdk, mode):
    if mode == "sync":
        sdk.close()
    else:
        await sdk.close()


def paid_payload(family, *, empty=False, input_tokens=3, output_tokens=1):
    data = payload(family, "answer")
    usage = data["usage"]
    usage["completion_tokens" if family == "openai-chat" else "output_tokens"] = output_tokens
    usage["prompt_tokens" if family == "openai-chat" else "input_tokens"] = input_tokens
    usage["total_tokens"] = input_tokens + output_tokens
    if family == "anthropic-messages":
        usage.update(cache_read_input_tokens=0, cache_creation_input_tokens=0)
    if empty:
        if family == "openai-chat":
            data["choices"][0]["message"].update(content="", reasoning_content="readable thought")
        elif family == "openai-responses":
            data["output"] = [
                {
                    "id": "thought",
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "readable thought"}],
                    "encrypted_content": "private-encrypted",
                }
            ]
        else:
            data["content"] = [
                {"type": "thinking", "thinking": "readable thought", "signature": "private-signed"}
            ]
    return data


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_public_load_actual_sdk_typed_and_borrowed_retry_contract(family, mode):
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        assert request.headers["x-private-host"] == "fixture"
        assert b"fixture-key" not in request.content
        return httpx.Response(200, json=payload(family), headers={"x-response-cost": "0.03"})

    sdk, http = borrowed(family, mode, handler)
    llm = load(family, sdk)
    assert not seen and sdk.max_retries == 9
    clone = llm.adapter._transport._sdk(mode)
    assert clone.max_retries == 0 and clone._client is http
    request = LLMRequest(messages=[{"role": "user", "content": "question"}], output_type=Answer)
    with _collect_observations() as collector:
        response = llm.call(request) if mode == "sync" else await llm.acall(request)
    assert response.parsed == Answer(value=42) and response.usage.cost_usd == 0.03
    assert len(seen) == 1 and len(collector.records) == 1
    assert collector.records[0].response_received and collector.records[0].finished
    assert collector.records[0].provider_entered and not collector.records[0].hidden_retries
    await llm.aclose()
    assert not http.is_closed and sdk.max_retries == 9
    await dispose(sdk, mode)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("terminal", ["success", "exhausted", "parse"])
async def test_paid_receipts_through_load_llm(family, mode, terminal):
    seen = []

    def handler(request):
        seen.append(request)
        first = len(seen) == 1
        data = paid_payload(
            family,
            empty=(terminal == "exhausted" or (first and terminal == "success")),
            input_tokens=7 if first else 3,
            output_tokens=2 if first else 1,
        )
        if terminal == "parse":
            data = payload(family, "not json")
            if family == "anthropic-messages":
                data["usage"].update(cache_read_input_tokens=0, cache_creation_input_tokens=0)
        return httpx.Response(
            200, json=data, headers={"x-response-cost": "0.03" if first else "0.01"}
        )

    sdk, http = borrowed(family, mode, handler)
    llm = load(family, sdk)
    llm._retry_config = zero_delay(empty=True)
    snapshots = []
    request = LLMRequest(
        messages=[{"role": "user", "content": "question"}],
        output_type=Answer if terminal == "parse" else None,
    )
    with _accounting_receiver(snapshots.append):
        if terminal == "success":
            response = llm.call(request) if mode == "sync" else await llm.acall(request)
            assert (
                response.usage.input_tokens == 3
                and response.usage.output_tokens == 1
                and response.usage.cost_usd == 0.01
            )
        else:
            with pytest.raises(LLMFailure) as caught:
                if mode == "sync":
                    llm.call(request)
                else:
                    await llm.acall(request)
            assert caught.value.kind == (
                FailureKind.BAD_REQUEST if terminal == "parse" else FailureKind.OUTPUT_LIMIT
            )
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.attempt_count == len(seen) == (1 if terminal == "parse" else 2)
    assert snapshot.subtotal("input_tokens").known == (7 if terminal == "parse" else 10)
    assert snapshot.subtotal("output_tokens").known == (2 if terminal == "parse" else 3)
    assert snapshot.subtotal("cost_usd").known == pytest.approx(
        0.03 if terminal == "parse" else 0.04
    )
    assert snapshot.subtotal("cost_usd").complete
    assert all(attempt.finished and attempt.response_received for attempt in snapshot.attempts)
    await llm.aclose()
    assert not http.is_closed
    await dispose(sdk, mode)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("status", [429, 503, 529, 401])
async def test_actual_http_errors_shared_retry_not_sdk_retry(family, status):
    seen = []

    def handler(request):
        seen.append(request)
        if len(seen) == 1:
            return httpx.Response(
                status,
                json={"error": {"type": "api_error", "message": "private-body"}},
                headers={"retry-after": "0"},
            )
        return httpx.Response(200, json=payload(family))

    sdk, http = borrowed(family, "async", handler)
    policy = zero_delay()
    policy = policy.model_copy(
        update={"retryable_status_codes": policy.retryable_status_codes | {529}}
    )
    llm = load(family, sdk, retry_config=policy)
    snapshots = []
    with _accounting_receiver(snapshots.append):
        if status == 401:
            with pytest.raises(LLMFailure) as caught:
                await llm.acall(LLMRequest(messages=[{"role": "user", "content": "hi"}]))
            assert caught.value.kind == FailureKind.AUTH
            assert isinstance(
                caught.value.__cause__, (openai.APIStatusError, anthropic.APIStatusError)
            )
        else:
            await llm.acall(LLMRequest(messages=[{"role": "user", "content": "hi"}]))
    assert len(seen) == (1 if status == 401 else 2) and sdk.max_retries == 9
    assert snapshots[0].attempts[0].status_code == status
    assert snapshots[0].attempts[0].retry_after == 0
    assert "private-body" not in str(snapshots[0])
    await llm.aclose()
    assert not http.is_closed
    await sdk.close()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_native_actual_path_no_legacy_helpers_and_cheap_views(family, mode, monkeypatch):
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=payload(family))

    sdk, http = borrowed(family, mode, handler)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if (
            name == "litellm"
            or name.startswith("litellm.")
            or name
            in {
                "nooa.unifiedllm.unifiedllm",
                "nooa.unifiedllm.adapters.legacy",
                "nooa.unifiedllm.adapters.litellm_chat",
                "nooa.unifiedllm.adapters.litellm_responses",
            }
        ):
            raise AssertionError("Native path imported legacy provider implementation: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    # Import guards alone miss already-loaded helpers and relative imports.
    # Profile the actual SDK path transitively, including binding/estimation,
    # without patching any SDK method or legacy helper implementation.
    forbidden_calls = []
    previous_profile = sys.getprofile()

    def profile(frame, event, arg):
        if event != "call":
            return
        filename = frame.f_code.co_filename.replace("\\", "/")
        if "/litellm/" in filename or filename.endswith(
            (
                "/nooa/unifiedllm/unifiedllm.py",
                "/adapters/legacy.py",
                "/adapters/litellm_chat.py",
                "/adapters/litellm_responses.py",
            )
        ):
            forbidden_calls.append((filename, frame.f_code.co_name))

    sys.setprofile(profile)
    try:
        await _exercise_native_path(family, mode, sdk, http, seen)
    finally:
        sys.setprofile(previous_profile)
    assert not forbidden_calls
    await dispose(sdk, mode)


async def _exercise_native_path(family, mode, sdk, http, seen):
    first = load(family, sdk, temperature=0.2)
    second = load(family, sdk, temperature=0.7)
    assert not seen and first._lease is second._lease
    assert first.adapter._route_calibration_record is second.adapter._route_calibration_record
    derived = _bind_adapter(first.adapter, temperature=0.4)
    assert (
        derived._transport is first.adapter._transport
        and derived.capabilities is first.adapter.capabilities
    )
    request = LLMRequest(messages=[{"role": "user", "content": "question"}])
    first.call(request) if mode == "sync" else await first.acall(request)
    assert seen[0]["temperature"] == 0.2
    assert first.adapter._route_calibration_record.ratio() != 1.0
    await first.aclose()
    await second.aclose()
    assert not http.is_closed


@pytest.mark.parametrize("family", FAMILIES)
async def test_default_529_terminal_status_policy_unchanged(family):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(529, json={"error": {"type": "overloaded_error", "message": "busy"}})

    sdk, http = borrowed(family, "async", handler)
    llm = load(family, sdk)
    with pytest.raises(LLMFailure) as caught:
        await llm.acall(LLMRequest(messages=[{"role": "user", "content": "hi"}]))
    assert caught.value.kind == FailureKind.TRANSIENT and caught.value.status_code == 529
    assert len(seen) == 1 and 529 not in RetryConfig().retryable_status_codes
    await llm.aclose()
    assert not http.is_closed
    await sdk.close()


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("malformed", ["json", "normalized"])
async def test_paid_raw_acknowledgement_survives_parser_failure(family, mode, malformed):
    seen = []

    def handler(request):
        seen.append(request)
        headers = {"x-response-cost": "0.03", "content-type": "application/json"}
        if malformed == "json":
            return httpx.Response(200, content=b"private-invalid-json fixture-key", headers=headers)
        data = paid_payload(family)
        if family == "openai-chat":
            data["choices"][0]["message"]["tool_calls"] = [
                {"id": "broken", "type": "function", "function": {"name": "tool"}}
            ]
        elif family == "openai-responses":
            data["output"] = [{"type": "function_call", "call_id": "broken", "name": "tool"}]
        else:
            data["content"] = [{"type": "tool_use", "id": "broken", "name": "tool"}]
        return httpx.Response(200, json=data, headers=headers)

    sdk, http = borrowed(family, mode, handler)
    llm = load(family, sdk)
    snapshots = []
    with _accounting_receiver(snapshots.append), _collect_observations() as collector:
        with pytest.raises(LLMFailure) as caught:
            request = LLMRequest(messages=[{"role": "user", "content": "hi"}])
            if mode == "sync":
                llm.call(request)
            else:
                await llm.acall(request)
    assert caught.value.kind == FailureKind.BAD_REQUEST
    assert caught.value.__cause__ is not None
    assert len(seen) == len(snapshots) == len(collector.records) == 1
    snapshot = snapshots[0]
    assert snapshot.attempt_count == 1
    assert snapshot.subtotal("cost_usd").known == pytest.approx(0.03)
    assert snapshot.subtotal("cost_usd").complete
    attempt = collector.records[0]
    assert attempt.response_received and attempt.finished
    assert attempt.usage.cost_usd == 0.03
    if malformed == "json":
        assert isinstance(caught.value.__cause__, json.JSONDecodeError)
        assert attempt.usage.input_tokens is None and attempt.usage.output_tokens is None
        assert not snapshot.subtotal("input_tokens").complete
    else:
        assert attempt.usage.input_tokens == 3 and attempt.usage.output_tokens == 1
    assert "fixture-key" not in str(snapshot) and "private-invalid" not in str(snapshot)
    await llm.aclose()
    assert not http.is_closed
    await dispose(sdk, mode)
