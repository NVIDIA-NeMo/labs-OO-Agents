# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared executor policies and legacy injection, with no provider calls."""

import ast
import asyncio
from contextlib import contextmanager
from pathlib import Path

import pytest

from nooa.llm_types import AssistantReasoning, AssistantText, LLMResponse, LLMUsage
from nooa.unifiedllm import CompletionClient, ResponsesClient
from nooa.unifiedllm import unifiedllm as legacy
from nooa.unifiedllm.adapters.base import Capabilities
from nooa.unifiedllm.admission import AdmissionError
from nooa.unifiedllm.errors import EmptyContentError, _ReplyParseError
from nooa.unifiedllm.execution import execute_async, execute_sync
from nooa.unifiedllm.request import LLMRequest
from nooa.unifiedllm.retry_config import RetryConfig

RETRY = RetryConfig(max_retries=2, base_delay=0, max_delay=0, jitter_factor=0)
MODE = pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])


def turn(text="ok", reasoning=None, usage=True):
    parts = (AssistantText(text=text),) if text else ()
    if reasoning:
        parts += (AssistantReasoning(text=reasoning),)
    return LLMResponse(
        parts=parts,
        finish_reason="stop",
        usage=LLMUsage(input_tokens=3, output_tokens=1) if usage else None,
    )


class StrictAdapter:
    capabilities = Capabilities()

    def __init__(self, replies):
        self.replies = iter(replies)
        self.seen = []

    def send_sync(self, request):
        self.seen.append(request)
        reply = next(self.replies)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    async def send(self, request):
        return self.send_sync(request)


async def run(is_async, adapter, request, config=RETRY, **policies):
    if is_async:
        return await execute_async(adapter, request, config, **policies)
    return execute_sync(adapter, request, config, **policies)


@contextmanager
def tracking(events):
    events.append("enter")
    try:
        yield
    finally:
        events.append("exit")


@MODE
@pytest.mark.parametrize("usage", [False, True])
@pytest.mark.asyncio
async def test_attempts_are_strict_and_usage_is_once_after_tracking(is_async, usage):
    response = turn(usage=usage)
    adapter = StrictAdapter([ConnectionError("gone"), response])
    request = LLMRequest()
    events = []
    result = await run(
        is_async,
        adapter,
        request,
        tracking=tracking(events),
        record_usage=lambda detail: events.append(detail),
    )
    assert result is response
    assert len(adapter.seen) == 2 and all(r is request for r in adapter.seen)
    assert events == ["enter", "exit"] + ([response.usage] if usage else [])


@MODE
@pytest.mark.parametrize("failure", ["parse", "admission", "cancel", "exhausted"])
@pytest.mark.asyncio
async def test_terminal_errors_unwrap_and_balance_tracking(is_async, failure):
    error = {
        "parse": ValueError("rate limit status 503"),
        "admission": AdmissionError("controller unavailable"),
        "cancel": asyncio.CancelledError(),
        "exhausted": ConnectionError("gone"),
    }[failure]
    wrapped = _ReplyParseError(error) if failure == "parse" else error
    adapter = StrictAdapter([wrapped] * 3)
    events = []
    with pytest.raises(type(error)) as caught:
        await run(
            is_async,
            adapter,
            LLMRequest(),
            tracking=tracking(events),
            record_usage=lambda usage: events.append(usage),
        )
    assert caught.value is error
    assert len(adapter.seen) == (3 if failure == "exhausted" else 1)
    assert events == ["enter", "exit"]


@MODE
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.asyncio
async def test_empty_policy_and_no_config_toggle(is_async, enabled, configured):
    empty = turn(text="", reasoning="thinking")
    final = turn()
    adapter = StrictAdapter([empty, final])
    config = RETRY.model_copy(update={"retry_on_empty_content": enabled}) if configured else None
    checked = []

    def check(response, retry_empty):
        checked.append(retry_empty)
        if retry_empty and not response.content and response.reasoning:
            raise EmptyContentError(response.reasoning)

    metrics = []
    result = await run(
        is_async, adapter, LLMRequest(), config, check_empty=check, record_usage=metrics.append
    )
    retries = enabled and configured
    assert result is (final if retries else empty)
    assert checked == ([True, True] if retries else [False])
    assert len(adapter.seen) == (2 if retries else 1)
    assert metrics == [result.usage]


@pytest.mark.asyncio
async def test_cancellation_while_send_is_pending_balances_tracking():
    arrived = asyncio.Event()
    events = []

    class Pending(StrictAdapter):
        async def send(self, request):
            self.seen.append(request)
            arrived.set()
            await asyncio.Future()

    adapter = Pending([])
    task = asyncio.create_task(
        execute_async(
            adapter,
            LLMRequest(),
            RETRY,
            tracking=tracking(events),
            record_usage=lambda detail: events.append(detail),
        )
    )
    await asyncio.wait_for(arrived.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(adapter.seen) == 1 and events == ["enter", "exit"]


@MODE
@pytest.mark.parametrize("client_type", [CompletionClient, ResponsesClient])
@pytest.mark.parametrize("empty_enabled", [False, True])
@pytest.mark.asyncio
async def test_legacy_retry_tracking_empty_and_metric_hooks(
    monkeypatch, is_async, client_type, empty_enabled
):
    events = []
    response = turn()
    config = RETRY.model_copy(update={"retry_on_empty_content": empty_enabled})

    def retry(call, **kwargs):
        assert kwargs == {"config": config}
        events.append("retry")
        return call()

    async def aretry(call, **kwargs):
        assert kwargs == {"config": config}
        events.append("retry")
        return await call()

    monkeypatch.setattr(legacy, "sync_retry", retry)
    monkeypatch.setattr(legacy, "with_retry", aretry)
    monkeypatch.setattr(legacy, "_track_llm_call", lambda **kwargs: tracking(events))
    monkeypatch.setattr(
        legacy, "_raise_if_empty_content", lambda reply, enabled: events.append((reply, enabled))
    )
    monkeypatch.setattr(legacy, "_record_llm_metric", lambda event, detail: events.append(event))
    client = client_type("openai/test", api_key="offline", retry_config=config)
    adapter = StrictAdapter([response])
    client._adapter = adapter
    try:
        result = await client.acall([]) if is_async else client.call([])
        assert result is response and len(adapter.seen) == 1
        expected = ["enter", "retry"]
        if client_type is CompletionClient:
            expected.append((response, empty_enabled))
        expected.append("exit")
        if client_type is CompletionClient:
            expected.append("token_usage")
        assert events == expected
    finally:
        await client.aclose()


@MODE
@pytest.mark.asyncio
async def test_absent_retry_config_never_calls_retry_hook(is_async):
    error = ConnectionError("gone")
    adapter = StrictAdapter([error])

    def forbidden(*args, **kwargs):
        raise AssertionError("retry disabled")

    with pytest.raises(ConnectionError) as caught:
        await run(is_async, adapter, LLMRequest(), None, retry=forbidden)
    assert caught.value is error and len(adapter.seen) == 1


@pytest.mark.asyncio
async def test_concurrent_executions_keep_callbacks_and_requests_local():
    arrived = asyncio.Event()
    release = asyncio.Event()
    one, two = LLMRequest(), LLMRequest()
    response = turn()
    events_one, events_two = [], []

    class Pending(StrictAdapter):
        async def send(self, request):
            self.seen.append(request)
            if len(self.seen) == 2:
                arrived.set()
            await release.wait()
            return response

    adapter = Pending([])
    tasks = [
        asyncio.create_task(
            execute_async(
                adapter,
                request,
                RETRY,
                tracking=tracking(events),
                record_usage=events.append,
            )
        )
        for request, events in [(one, events_one), (two, events_two)]
    ]
    try:
        await asyncio.wait_for(arrived.wait(), timeout=2)
        assert events_one == events_two == ["enter"]
        assert adapter.seen[0] is one and adapter.seen[1] is two
    finally:
        release.set()
        results = await asyncio.gather(*tasks)
    assert all(result is response for result in results)
    assert events_one == events_two == ["enter", "exit", response.usage]


def test_executor_imports_no_provider_runtime_or_monolith():
    source = Path(__file__).resolve().parents[2] / "src/nooa/unifiedllm/execution.py"
    tree = ast.parse(source.read_text())
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    assert not any(
        name == "unifiedllm"
        or any(part in name for part in ("litellm", "runtime", "tracing", "opentelemetry"))
        for name in modules
    )
