# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A total deadline for one logical LLM call, across all endpoint retries.

The per-attempt ``request_timeout`` (600 s) times ``max_retries`` let one call
outlive an 870 s task budget, and the retries left no trace but a stderr line.
These tests drive a real ``CompletionClient`` through a fake HTTP transport
that answers 500 and then stalls.
"""

import asyncio
import logging
import time
from typing import Any

import httpx
import pytest
from opentelemetry import trace as otel_trace

from nooa.runtime.actor import _make_llm_metrics_bridge
from nooa.runtime.harness_metrics import HarnessMetrics
from nooa.unifiedllm import (
    CompletionClient,
    LLMDeadlineExceeded,
    RetryConfig,
    llm_deadline,
    llm_time_remaining,
    with_retry,
)
from nooa.unifiedllm._metrics import _llm_metrics_callback
from nooa.unifiedllm.retry import clip_attempt_timeout

from .test_completion_sdk_retries import MESSAGES, OK_BODY, _install


class _Span(otel_trace.NonRecordingSpan):
    """The active LLM span: records events; a full Span to tracing code around it."""

    def __init__(self) -> None:
        super().__init__(otel_trace.INVALID_SPAN_CONTEXT)
        self.events: list[tuple[str, dict[str, Any]]] = []

    def is_recording(self) -> bool:
        return True

    def add_event(self, name: str, attributes: Any = None, timestamp: Any = None) -> None:
        self.events.append((name, dict(attributes or {})))


@pytest.fixture
def span(monkeypatch) -> _Span:
    fake = _Span()
    monkeypatch.setattr(otel_trace, "get_current_span", lambda *a, **k: fake)
    return fake


@pytest.fixture
def metrics():
    hm = HarnessMetrics()
    token = _llm_metrics_callback.set(_make_llm_metrics_bridge(hm))
    try:
        yield hm
    finally:
        _llm_metrics_callback.reset(token)


def _500_then_stall() -> tuple[httpx.MockTransport, list[float]]:
    """A 200 (the warm-up call: LiteLLM's first call is slow), a 500, then a stall."""
    requests: list[float] = []

    async def respond(request: httpx.Request) -> httpx.Response:
        requests.append(time.monotonic())
        if len(requests) == 1:
            return httpx.Response(200, json=OK_BODY)
        if len(requests) == 2:
            return httpx.Response(500, json={"error": {"message": "upstream finish_reason error"}})
        await asyncio.sleep(3600)
        return httpx.Response(200, json=OK_BODY)

    return httpx.MockTransport(respond), requests


async def _warm_up(client: CompletionClient) -> None:
    assert (await client.acall(MESSAGES)).content == "done"


def _client(**retry) -> CompletionClient:
    return CompletionClient(
        "openai/m",
        api_base="https://endpoint.example/v1",
        api_key="key",
        retry_config=RetryConfig(
            max_retries=3,
            rate_limit_extra_retries=0,
            base_delay=0.0,
            jitter_factor=0.0,
            min_attempt_seconds=0.1,
            **retry,
        ),
    )


@pytest.mark.asyncio
async def test_500_then_stall_fails_within_total_timeout(monkeypatch, span, metrics, caplog):
    transport, requests = _500_then_stall()
    _install(monkeypatch, transport)
    await _warm_up(_client())  # same loop and transport; no deadline on the slow first call
    client = _client(total_timeout=1.5)  # request_timeout stays at its 600 s default
    try:
        started = time.monotonic()
        with caplog.at_level(logging.WARNING, logger="nooa.unifiedllm.retry"):
            with pytest.raises(LLMDeadlineExceeded, match="deadline"):
                await client.acall(MESSAGES)
    finally:
        await client.aclose()
    elapsed = time.monotonic() - started

    assert 1.3 < elapsed < 5.0
    assert (
        len(requests) == 3
    )  # warm-up, the 500, then the 500, then the stalled retry cut off at the deadline
    retry_lines = [r.getMessage() for r in caplog.records if "LLM retry" in r.getMessage()]
    assert len(retry_lines) == 1
    assert "attempt 1/4" in retry_lines[0] and "status 500" in retry_lines[0]
    assert (
        "retrying in 0.0s" in retry_lines[0] and "left before the call deadline" in retry_lines[0]
    )
    retry_events = [attrs for name, attrs in span.events if name == "llm.retry"]
    assert retry_events and retry_events[0]["status_code"] == 500
    assert retry_events[0]["attempt"] == 1 and retry_events[0]["max_attempts"] == 4
    assert metrics.llm_retries == 1
    assert metrics.llm_retry_errors and "500" in metrics.llm_retry_errors[0]


@pytest.mark.asyncio
async def test_caller_deadline_bounds_a_call_without_total_timeout(monkeypatch, span):
    transport, requests = _500_then_stall()
    _install(monkeypatch, transport)
    client = _client()
    try:
        await _warm_up(client)
        started = time.monotonic()
        with llm_deadline(1.0), pytest.raises(LLMDeadlineExceeded):
            await client.acall(MESSAGES)
        elapsed = time.monotonic() - started
    finally:
        await client.aclose()
    assert 0.8 < elapsed < 4.0
    assert len(requests) == 3
    assert llm_time_remaining() is None  # the context restored it


@pytest.mark.asyncio
async def test_no_retry_starts_without_time_for_it(span, metrics):
    calls = 0

    class Upstream500(Exception):
        status_code = 500

    async def fail():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.3)
        raise Upstream500("internal server error")

    config = RetryConfig(
        max_retries=3, base_delay=0.0, jitter_factor=0.0, total_timeout=0.5, min_attempt_seconds=1
    )
    with pytest.raises(LLMDeadlineExceeded, match=r"total_timeout=0.5s") as info:
        await with_retry(fail, config=config)

    assert calls == 1
    assert info.value.attempts == 1
    assert isinstance(info.value.__cause__, Upstream500)
    assert [name for name, _ in span.events] == ["llm.deadline_exceeded"]
    assert metrics.llm_deadline_exceeded == 1 and metrics.llm_retries == 0


@pytest.mark.asyncio
async def test_without_a_deadline_retries_are_unchanged(span):
    calls = 0

    async def flaky():
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ConnectionError("reset")
        return "ok"

    config = RetryConfig(max_retries=3, base_delay=0.0, jitter_factor=0.0)
    assert await with_retry(flaky, config=config) == "ok"
    assert calls == 3
    assert [name for name, _ in span.events] == ["llm.retry", "llm.retry"]


def test_clip_attempt_timeout():
    assert clip_attempt_timeout(600.0) == (600.0, False)
    assert clip_attempt_timeout(None) == (None, False)
    with llm_deadline(10):
        timeout, clipped = clip_attempt_timeout(600.0)
        assert clipped and 9 < timeout <= 10
        timeout, clipped = clip_attempt_timeout(None)
        assert clipped and 9 < timeout <= 10
        assert clip_attempt_timeout(1.0) == (1.0, False)
        with llm_deadline(100):  # nested deadlines keep the earliest
            remaining = llm_time_remaining()
            assert remaining is not None and remaining <= 10
    with llm_deadline(0), pytest.raises(LLMDeadlineExceeded):
        clip_attempt_timeout(600.0)


def test_deadline_error_is_never_retried():
    from nooa.unifiedllm.retry import _is_retryable_error

    assert _is_retryable_error(LLMDeadlineExceeded("spent"), RetryConfig()) == (False, False)


def test_llm_retry_events_reach_harness_metrics():
    from nooa.runtime.actor import _make_llm_metrics_bridge
    from nooa.runtime.harness_metrics import HarnessMetrics

    hm = HarnessMetrics()
    bridge = _make_llm_metrics_bridge(hm)
    bridge("llm_retry", {"error_type": "InternalServerError", "status_code": 500, "attempt": 1})
    bridge("llm_retry", {"error_type": "ReadTimeout", "status_code": None, "attempt": 2})
    bridge("llm_deadline_exceeded", {"attempts": 3})
    assert hm.llm_retries == 2
    assert hm.llm_retry_errors == ["InternalServerError 500", "ReadTimeout"]
    assert hm.llm_deadline_exceeded == 1
