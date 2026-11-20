# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Installed SDK objects at strict adapters; no hypothetical Anthropic SDK."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import openai
import pytest
from litellm import exceptions as sdk
from pydantic import BaseModel, ValidationError

from nooa.unifiedllm import LLM, CompletionClient, LLMRequest, ResponsesClient
from nooa.unifiedllm.adapters import _legacy_error_text as legacy
from nooa.unifiedllm.adapters._failures import normalize, retry_after
from nooa.unifiedllm.admission import AdmissionError
from nooa.unifiedllm.errors import FailureKind as K
from nooa.unifiedllm.errors import LLMFailure
from nooa.unifiedllm.retry_config import RetryConfig

RETRY = RetryConfig(max_retries=2, base_delay=0, rate_limit_base_delay=0, jitter_factor=0)
REQUEST = LLMRequest(messages=[{"role": "user", "content": "hello"}])


def response(status, headers=None):
    return httpx.Response(
        status, request=httpx.Request("POST", "https://offline.example/v1"), headers=headers
    )


def api_error(status=400, body=None, message="timeout rate limit status 503", sdk_family="openai"):
    if sdk_family == "litellm":
        return sdk.BadRequestError(message, "model", "openai", response=response(status), body=body)
    return openai.APIStatusError(message, response=response(status), body=body)


def case(name, sdk_family):
    if name == "rate_limit":
        return (
            sdk.RateLimitError("plain", "openai", "model")
            if sdk_family == "litellm"
            else api_error(429, message="plain")
        )
    if name == "transient":
        return (
            sdk.APIConnectionError("plain", "openai", "model")
            if sdk_family == "litellm"
            else openai.APIConnectionError(request=response(502).request)
        )
    if name == "timeout":
        return (
            sdk.Timeout("plain", "model", "openai")
            if sdk_family == "litellm"
            else openai.APITimeoutError(request=response(408).request)
        )
    if name == "auth":
        return (
            sdk.AuthenticationError("timeout rate limit", "openai", "model")
            if sdk_family == "litellm"
            else api_error(401)
        )
    if name == "config":
        return (
            sdk.BudgetExceededError(2, 1)
            if sdk_family == "litellm"
            else api_error(404, {"code": "model_not_found"})
        )
    if name == "context_window":
        return (
            sdk.ContextWindowExceededError("prompt contains 0 tokens", "model", "openai")
            if sdk_family == "litellm"
            else api_error(
                body={"code": "context_length_exceeded", "prompt_tokens": 0, "context_window": 4096}
            )
        )
    if name == "media_limit":
        return api_error(
            body={"code": "too_many_images", "media_count": 3, "media_count_limit": 0},
            sdk_family=sdk_family,
        )
    if name == "media_rejected":
        return api_error(body={"code": "invalid_image"}, sdk_family=sdk_family)
    if name == "output_limit":
        return openai.LengthFinishReasonError(completion=SimpleNamespace(usage=None))
    if name == "bad_request":
        return (
            sdk.APIResponseValidationError("status 503", "openai", "model")
            if sdk_family == "litellm"
            else openai.APIResponseValidationError(response=response(500), body={})
        )
    raise AssertionError(name)


@pytest.mark.parametrize("sdk_family", ["openai", "litellm"])
@pytest.mark.parametrize("kind", list(K))
@pytest.mark.parametrize("family", [CompletionClient, ResponsesClient])
@pytest.mark.parametrize("is_async", [False, True])
async def test_real_sdk_types_direct_send(monkeypatch, sdk_family, kind, family, is_async):
    client = family("openai/gpt-4o-mini", api_key="offline")
    error = case(kind.value, sdk_family)
    seen = []

    def fail(*args, **kwargs):
        seen.append(True)
        raise error

    async def afail(*args, **kwargs):
        return fail(*args, **kwargs)

    monkeypatch.setattr(client._adapter, "transmit_sync", fail)
    monkeypatch.setattr(client._adapter, "transmit", afail)
    try:
        with pytest.raises(LLMFailure) as caught:
            if is_async:
                await client._adapter.send(REQUEST)
            else:
                client._adapter.send_sync(REQUEST)
        assert caught.value.kind == kind
        assert caught.value.__cause__ is error
        assert seen == [True]
        if kind == K.MEDIA_LIMIT:
            assert caught.value.media_count == 3
            assert caught.value.media_count_limit == 0
        if kind == K.CONTEXT_WINDOW:
            assert caught.value.prompt_tokens == 0
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    "body",
    [
        None,
        [],
        "junk",
        {"error": []},
        {"code": []},
        {"code": "context_length_exceeded", "prompt_tokens": True, "context_window": 0},
        {"code": "context_length_exceeded", "prompt_tokens": -1, "context_window": "4000"},
    ],
)
def test_malformed_facts_are_ignored(body):
    failure = normalize(api_error(body=body, message="plain"))
    assert failure.prompt_tokens is None
    assert failure.context_window is None
    assert failure.media_count is None
    assert failure.media_count_limit is None


@pytest.mark.parametrize("status", [401, 403, 404, 422])
def test_outer_terminal_status_dominates_cause_and_timeout_class(status):
    error = sdk.Timeout("timeout", "model", "openai", exception_status_code=status)
    error.__cause__ = case("transient", "openai")
    failure = normalize(error)
    assert failure.kind == (K.AUTH if status in (401, 403) else K.BAD_REQUEST)
    assert failure.__cause__ is error


@pytest.mark.parametrize(
    "error",
    [
        sdk.UnsupportedParamsError("rate limit"),
        sdk.LiteLLMUnknownProvider("unknown"),
        sdk.RateLimitError("timeout", "openai", "m", rate_limit_type="budget"),
        sdk.RateLimitError("timeout", "openai", "m", rate_limit_type="max_iterations"),
    ],
)
def test_synthetic_429_and_configuration_failures_are_terminal(error):
    failure = normalize(error)
    assert failure.kind == K.CONFIG
    assert failure.status_code is None


@pytest.mark.parametrize(
    "value, expected",
    [
        ("3.5", 3.5),
        ("0", 0),
        ("-1", None),
        ("nan", None),
        ("inf", None),
        ("nonsense", None),
        ("Thu, 01 Jan 1970 00:01:50 GMT", 10),
        ("Thu, 01 Jan 1970 00:00:50 GMT", 0),
    ],
)
def test_retry_after_numeric_and_date_frozen_clock(value, expected, monkeypatch):
    monkeypatch.setattr("nooa.unifiedllm.adapters._failures.time.time", lambda: 100)
    error = openai.RateLimitError(
        "plain", response=response(429, {"Retry-After": value}), body=None
    )
    assert retry_after(error) == expected
    assert normalize(error).retry_after == expected
    local = sdk.RateLimitError("plain", "openai", "model", headers={"Retry-After": value})
    assert retry_after(local) == expected


def test_typed_identity_and_raw_cycle_safe_fallback():
    failure = LLMFailure(K.AUTH, "context window", cause=Exception("prompt contains 50 tokens"))
    assert normalize(failure) is failure
    one, two = Exception("outer"), Exception("context window prompt contains 50 tokens")
    one.__cause__ = two
    two.__cause__ = one
    assert legacy._is_context_window_error(one)
    assert legacy._parse_prompt_tokens(one) == 50
    assert legacy.image_limit(one) is None
    assert not legacy.explicitly_rejected(one, "image")


@pytest.mark.parametrize("family", [CompletionClient, ResponsesClient])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    "error",
    [
        asyncio.CancelledError(),
        KeyboardInterrupt(),
        SystemExit(),
        LLMFailure(K.AUTH, "original"),
        AdmissionError("timeout"),
    ],
)
async def test_direct_control_flow_and_typed_identity(monkeypatch, family, is_async, error):
    client = family("openai/gpt-4o-mini", api_key="offline")

    def fail(*args, **kwargs):
        raise error

    async def afail(*args, **kwargs):
        fail()

    monkeypatch.setattr(client._adapter, "transmit_sync", fail)
    monkeypatch.setattr(client._adapter, "transmit", afail)
    try:
        with pytest.raises(type(error)) as caught:
            if is_async:
                await client._adapter.send(REQUEST)
            else:
                client._adapter.send_sync(REQUEST)
        assert caught.value is error
    finally:
        await client.aclose()


class Answer(BaseModel):
    value: int


@pytest.mark.parametrize("family", [CompletionClient, ResponsesClient])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("parse_type", ["json", "validation"])
@pytest.mark.parametrize("truncation", [False, True])
async def test_strict_paid_parse_vs_legacy_original(
    monkeypatch, family, is_async, parse_type, truncation
):
    client = family("openai/gpt-4o-mini", api_key="offline", retry_config=RETRY)
    error = json.JSONDecodeError("rate limit", "{", 0)
    if parse_type == "validation":
        try:
            Answer.model_validate({"value": "rate limit"})
        except ValidationError as exc:
            error = exc
    raw = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="length" if truncation else "stop")],
        status="incomplete" if truncation else "completed",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        usage=None,
    )
    seen = []

    def transmit(*args, **kwargs):
        seen.append(True)
        return raw

    async def atransmit(*args, **kwargs):
        return transmit()

    def parse(*args, **kwargs):
        raise error

    monkeypatch.setattr(client._adapter, "transmit_sync", transmit)
    monkeypatch.setattr(client._adapter, "transmit", atransmit)
    monkeypatch.setattr(client._adapter, "parse", parse)
    monkeypatch.setattr(
        client._adapter, "_finish", parse if family is CompletionClient else lambda *a, **k: None
    )
    try:
        llm = LLM(client._adapter, retry_config=RETRY)
        with pytest.raises(LLMFailure) as caught:
            if is_async:
                await llm.acall(REQUEST)
            else:
                llm.call(REQUEST)
        assert caught.value.kind == (K.OUTPUT_LIMIT if truncation else K.BAD_REQUEST)
        assert caught.value.__cause__ is error
        assert seen == [True]
        # Configured compatibility view retains exact original parse object.
        with pytest.raises(type(error)) as caught:
            if is_async:
                await client.acall(REQUEST.messages)
            else:
                client.call(REQUEST.messages)
        assert caught.value is error
        assert seen == [True, True]
    finally:
        await client.aclose()


@pytest.mark.parametrize("family", [CompletionClient, ResponsesClient])
@pytest.mark.parametrize("phase, kind", [("config", K.CONFIG), ("prepare", K.BAD_REQUEST)])
@pytest.mark.parametrize("is_async", [False, True])
async def test_preprovider_local_failures(monkeypatch, family, phase, kind, is_async):
    client = family("openai/gpt-4o-mini", api_key="offline")
    error = ValueError("status 503")

    def fail(*args, **kwargs):
        raise error

    adapter = client._adapter
    target = (
        "prepare"
        if phase == "prepare"
        else ("_settings" if family is CompletionClient else "_call_config")
    )
    monkeypatch.setattr(adapter, target, fail)
    monkeypatch.setattr(adapter, "transmit_sync", lambda *a: pytest.fail("preprovider"))
    try:
        with pytest.raises(LLMFailure) as caught:
            if is_async:
                await adapter.send(REQUEST)
            else:
                adapter.send_sync(REQUEST)
        assert caught.value.kind == kind
        assert caught.value.__cause__ is error
    finally:
        await client.aclose()


@pytest.mark.parametrize("outer_status", [401, 403, 404, 422])
def test_terminal_outer_status_beats_concrete_sdk_context(outer_status):
    error = sdk.Timeout("plain", "model", "openai", exception_status_code=outer_status)
    error.__context__ = openai.RateLimitError("plain", response=response(429), body=None)
    failure = normalize(error)
    assert failure.status_code == outer_status
    assert failure.kind == (K.AUTH if outer_status in (401, 403) else K.BAD_REQUEST)
    assert failure.__cause__ is error


def test_concrete_litellm_context_preserves_wire_facts_without_cause_replacement():
    from litellm.llms.base_llm.chat.transformation import BaseLLMException

    wire_response = httpx.Response(
        422,
        request=response(422).request,
        json={"error": {"code": "image_limit_exceeded", "media_count_limit": 0}},
    )
    wire = httpx.HTTPStatusError("wire", request=wire_response.request, response=wire_response)
    wrapper = BaseLLMException(status_code=422, message="plain")
    wrapper.__context__ = wire
    error = sdk.BadRequestError("plain", "model", "openai")
    error.__context__ = wrapper
    failure = normalize(error)
    assert failure.kind == K.MEDIA_LIMIT
    assert failure.media_count_limit == 0 and failure.media_count is None
    assert failure.status_code == 400  # explicitly terminal outer status wins
    assert failure.__cause__ is error


def test_no_arbitrary_context_walk_for_openai_or_unknown_errors():
    auth = openai.AuthenticationError("plain", response=response(401), body=None)
    transient = openai.InternalServerError("plain", response=response(503), body=None)
    transient.__context__ = auth
    auth.__context__ = transient
    assert normalize(transient).kind == K.TRANSIENT
    raw = RuntimeError("status 503 rate limit")
    raw.__context__ = transient
    assert normalize(raw).kind == K.CONFIG


def test_installed_litellm_vendor_headers_and_invalid_values():
    for value, expected in [("4", 4), ("inf", None), ("-2", None), ("bad", None)]:
        error = sdk.RateLimitError("plain", "openai", "model")
        error.litellm_response_headers = {"Retry-After": value}
        assert normalize(error).retry_after == expected


@pytest.mark.parametrize("status", [401, 429, 503])
def test_unknown_raw_status_attributes_do_not_grant_sdk_policy(status):
    error = RuntimeError("timeout rate limit")
    error.status_code = status
    error.body = {"code": "context_length_exceeded", "prompt_tokens": 100}
    failure = normalize(error)
    assert failure.kind == K.CONFIG
    assert failure.status_code is None and failure.prompt_tokens is None
    assert failure.__cause__ is error


@pytest.mark.parametrize("phase", ["wire", "parse"])
def test_sdk_length_finish_is_trusted_at_wire_or_parse_boundary(phase):
    error = openai.LengthFinishReasonError(completion=SimpleNamespace(usage=None))
    failure = normalize(error, phase)
    assert failure.kind == K.OUTPUT_LIMIT
    assert failure.status_code is None and failure.__cause__ is error
