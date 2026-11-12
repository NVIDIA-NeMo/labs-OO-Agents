# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Neutral typed retry policy, strict custom errors and bounded empty policy."""

import asyncio

import pytest

from nooa.llm_types import AssistantText, LLMResponse
from nooa.unifiedllm import LLM, LLMRequest
from nooa.unifiedllm.adapters.base import Capabilities
from nooa.unifiedllm.admission import AdmissionError
from nooa.unifiedllm.errors import FailureKind as K
from nooa.unifiedllm.errors import LLMFailure
from nooa.unifiedllm.retry import sync_retry, with_retry
from nooa.unifiedllm.retry_config import RetryConfig


class Adapter:
    capabilities = Capabilities()

    def __init__(self, errors):
        self.errors = list(errors)
        self.attempts = 0

    def send_sync(self, request):
        self.attempts += 1
        if self.errors:
            raise self.errors.pop(0)
        return LLMResponse(parts=(AssistantText(text="ok"),), finish_reason="stop")

    async def send(self, request):
        return self.send_sync(request)


def sleeps(monkeypatch):
    seen = []

    async def asleep(delay):
        seen.append(delay)

    monkeypatch.setattr("nooa.unifiedllm.retry.time.sleep", seen.append)
    monkeypatch.setattr("nooa.unifiedllm.retry.asyncio.sleep", asleep)
    return seen


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("kind", list(K))
async def test_all_typed_kinds_authoritative(monkeypatch, is_async, kind):
    waits = sleeps(monkeypatch)
    error = LLMFailure(
        kind, "rate limit timeout status 503", status_code=503, cause=ConnectionError("timeout")
    )
    adapter = Adapter([error])
    callbacks = []
    config = RetryConfig(
        max_retries=1,
        base_delay=2,
        rate_limit_base_delay=3,
        jitter_factor=0,
        on_retry=lambda *args: callbacks.append(args),
    )
    llm = LLM(adapter, retry_config=config)
    retryable = kind in (K.RATE_LIMIT, K.TRANSIENT, K.TIMEOUT)
    if retryable:
        result = await llm.acall(LLMRequest()) if is_async else llm.call(LLMRequest())
        assert result.content == "ok"
        assert adapter.attempts == 2
        assert callbacks == [(1, error, 3 if kind == K.RATE_LIMIT else 2)]
        assert waits == [callbacks[0][2]]
    else:
        with pytest.raises(LLMFailure) as caught:
            if is_async:
                await llm.acall(LLMRequest())
            else:
                llm.call(LLMRequest())
        assert caught.value is error
        assert adapter.attempts == 1
        assert callbacks == waits == []


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("kind", list(K))
async def test_status_allowlist_never_expands_terminal_kind(monkeypatch, is_async, kind):
    waits = sleeps(monkeypatch)
    error = LLMFailure(kind, "plain", status_code=400)
    adapter = Adapter([error])
    config = RetryConfig(
        max_retries=1, retryable_status_codes=frozenset({400}), base_delay=0, jitter_factor=0
    )
    llm = LLM(adapter, retry_config=config)
    if kind in (K.RATE_LIMIT, K.TRANSIENT, K.TIMEOUT):
        assert (
            await llm.acall(LLMRequest()) if is_async else llm.call(LLMRequest())
        ).content == "ok"
        assert adapter.attempts == 2
    else:
        with pytest.raises(LLMFailure) as caught:
            if is_async:
                await llm.acall(LLMRequest())
            else:
                llm.call(LLMRequest())
        assert caught.value is error
        assert waits == [] and adapter.attempts == 1
    # Excluding a transient HTTP status narrows eligibility.
    adapter = Adapter([LLMFailure(K.TRANSIENT, "plain", status_code=503)])
    with pytest.raises(LLMFailure):
        if is_async:
            await LLM(adapter, retry_config=config).acall(LLMRequest())
        else:
            LLM(adapter, retry_config=config).call(LLMRequest())
    assert adapter.attempts == 1


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize(
    "raw",
    [
        RuntimeError("rate limit status 503 timeout"),
        ConnectionError("timeout"),
        AdmissionError("timeout controller unavailable"),
    ],
)
async def test_strict_unknown_and_admission_no_resend(monkeypatch, is_async, raw):
    waits = sleeps(monkeypatch)
    adapter = Adapter([raw])
    config = RetryConfig(max_retries=3)
    with pytest.raises(LLMFailure) as caught:
        if is_async:
            await LLM(adapter, retry_config=config).acall(LLMRequest())
        else:
            LLM(adapter, retry_config=config).call(LLMRequest())
    assert caught.value.kind == K.CONFIG
    assert caught.value.__cause__ is raw
    assert adapter.attempts == 1 and waits == []


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("hint, expected", [(0, 2), (1, 2), (12, 12)])
async def test_retry_after_is_floor_after_max_delay_and_jitter(
    monkeypatch, is_async, hint, expected
):
    waits = sleeps(monkeypatch)
    monkeypatch.setattr("nooa.unifiedllm.retry.random.random", lambda: 1)
    error = LLMFailure(K.RATE_LIMIT, "plain", retry_after=hint)
    adapter = Adapter([error, error, error])
    callbacks = []
    config = RetryConfig(
        max_retries=1,
        rate_limit_extra_retries=1,
        rate_limit_base_delay=9,
        max_delay=1,
        jitter_factor=1,
        on_retry=lambda *a: callbacks.append(a),
    )
    with pytest.raises(LLMFailure) as caught:
        if is_async:
            await LLM(adapter, retry_config=config).acall(LLMRequest())
        else:
            LLM(adapter, retry_config=config).call(LLMRequest())
    assert caught.value is error
    assert adapter.attempts == 3
    assert waits == [expected, expected]
    assert callbacks == [(1, error, expected), (2, error, expected)]


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("sequence, expected", [("rrt", 3), ("trrr", 4), ("rtrrr", 2)])
async def test_mixed_rate_and_normal_budget(monkeypatch, is_async, sequence, expected):
    waits = sleeps(monkeypatch)
    errors = [LLMFailure(K.RATE_LIMIT if c == "r" else K.TRANSIENT, "plain") for c in sequence]
    adapter = Adapter(errors)
    config = RetryConfig(
        max_retries=1,
        rate_limit_extra_retries=2,
        base_delay=0,
        rate_limit_base_delay=0,
        jitter_factor=0,
    )
    with pytest.raises(LLMFailure) as caught:
        if is_async:
            await LLM(adapter, retry_config=config).acall(LLMRequest())
        else:
            LLM(adapter, retry_config=config).call(LLMRequest())
    assert caught.value is errors[expected - 1]
    assert adapter.attempts == expected and len(waits) == expected - 1


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("error", [asyncio.CancelledError(), KeyboardInterrupt(), SystemExit()])
async def test_control_flow_never_wrapped_or_retried(monkeypatch, is_async, error):
    waits = sleeps(monkeypatch)
    adapter = Adapter([error])
    with pytest.raises(type(error)) as caught:
        if is_async:
            await LLM(adapter, retry_config=RetryConfig()).acall(LLMRequest())
        else:
            LLM(adapter, retry_config=RetryConfig()).call(LLMRequest())
    assert caught.value is error
    assert adapter.attempts == 1 and waits == []


@pytest.mark.parametrize("is_async", [False, True])
async def test_actual_timeout_is_typed_and_retried(monkeypatch, is_async):
    sleeps(monkeypatch)
    original = TimeoutError("attempt deadline")
    adapter = Adapter([original, original])
    config = RetryConfig(max_retries=1, base_delay=0)
    with pytest.raises(LLMFailure) as caught:
        if is_async:
            await LLM(adapter, retry_config=config).acall(LLMRequest())
        else:
            LLM(adapter, retry_config=config).call(LLMRequest())
    assert caught.value.kind == K.TIMEOUT
    assert caught.value.__cause__ is original
    assert adapter.attempts == 2


@pytest.mark.parametrize("is_async", [False, True])
async def test_legacy_raw_retry_function_contract(monkeypatch, is_async):
    sleeps(monkeypatch)
    error = RuntimeError("status 503")
    adapter = Adapter([error, error])
    config = RetryConfig(max_retries=1, base_delay=0)
    with pytest.raises(RuntimeError) as caught:
        if is_async:
            await with_retry(adapter.send, LLMRequest(), config=config)
        else:
            sync_retry(adapter.send_sync, LLMRequest(), config=config)
    assert caught.value is error and adapter.attempts == 2
