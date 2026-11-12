# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Terminal charge sources across the installed SDK, without provider mocks."""

from dataclasses import asdict
from types import SimpleNamespace

import httpx
import pytest
from pydantic import BaseModel

from nooa.unifiedllm import (
    LLM,
    CompletionClient,
    LLMFailure,
    LLMRequest,
    ResponsesClient,
    RetryConfig,
)
from nooa.unifiedllm._accounting import (
    _accounting_receiver,
    _logical_account_scope,
    _LogicalAccount,
)
from nooa.unifiedllm._observations import _collect_observations, _NumericUsage
from nooa.unifiedllm.adapters._numeric_usage import _numeric_usage
from nooa.unifiedllm.unifiedllm import _ClientHttp


def wire(client_type, text="ok", *, empty=False, input_tokens=3, output_tokens=1):
    if client_type is CompletionClient:
        message = {"role": "assistant", "content": text}
        if empty:
            message.update(content="", reasoning_content="readable thought")
        return {
            "id": "chat-id",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
            "usage": {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }
    output = [
        {
            "id": "message",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
    ]
    if empty:
        output = [
            {
                "id": "thought",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "readable thought"}],
            }
        ]
    return {
        "id": "response-id",
        "object": "response",
        "created_at": 1,
        "model": "gpt-4o-mini",
        "status": "completed",
        "output": output,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def retry(retries=1):
    return RetryConfig(
        max_retries=retries,
        rate_limit_extra_retries=0,
        base_delay=0,
        max_delay=0,
        jitter_factor=0,
        retry_on_empty_content=True,
    )


def client(monkeypatch, client_type, respond):
    monkeypatch.setattr(
        _ClientHttp,
        "_httpx_hardening",
        staticmethod(lambda: {"transport": httpx.MockTransport(respond)}),
    )
    return client_type(
        "openai/gpt-4o-mini",
        api_key="not-a-secret",
        api_base="https://endpoint.test/v1",
        max_retries=9,
    )


@pytest.mark.parametrize("client_type", [CompletionClient, ResponsesClient])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("exhausted", [False, True])
async def test_paid_empty_terminal_sources(monkeypatch, client_type, is_async, exhausted):
    from opentelemetry.context import _SUPPRESS_INSTRUMENTATION_KEY, get_value

    before = get_value(_SUPPRESS_INSTRUMENTATION_KEY)
    requests, snapshots = [], []

    def respond(request):
        assert get_value(_SUPPRESS_INSTRUMENTATION_KEY) is True
        requests.append(request)
        first = len(requests) == 1
        return httpx.Response(
            200,
            json=wire(
                client_type,
                empty=first or exhausted,
                input_tokens=7 if first else 3,
                output_tokens=2 if first else 1,
            ),
            headers={"x-litellm-response-cost": ".03" if first else ".01"},
        )

    c = client(monkeypatch, client_type, respond)
    llm = LLM(c._adapter, retry_config=retry())
    request = LLMRequest(messages=[{"role": "user", "content": "hello"}])
    try:
        with _accounting_receiver(snapshots.append), _collect_observations(limit=1) as collector:
            if exhausted:
                with pytest.raises(LLMFailure):
                    if is_async:
                        await llm.acall(request)
                    else:
                        llm.call(request)
            elif is_async:
                final = await llm.acall(request)
            else:
                final = llm.call(request)
        [snapshot] = snapshots
        assert len(requests) == snapshot.attempt_count == 2
        assert snapshot.outcome == ("error" if exhausted else "success")
        assert snapshot.subtotal("input_tokens").known == 10
        assert snapshot.subtotal("output_tokens").known == 3
        assert snapshot.subtotal("total_tokens").known == 13
        assert snapshot.subtotal("input_tokens").complete
        assert snapshot.subtotal("cost_usd").known == pytest.approx(0.04)
        assert snapshot.subtotal("cost_usd").complete
        assert snapshot.cost_sources == ("gateway_header",)
        assert get_value(_SUPPRESS_INSTRUMENTATION_KEY) is before
        assert [a.attempt_index for a in snapshot.attempts] == [0, 1]
        assert all(
            a.response_received and a.provider_entered and a.finished for a in snapshot.attempts
        )
        assert len(collector.records) == 1 and collector.dropped_attempts == 1
        if not exhausted:
            assert (final.usage.input_tokens, final.usage.output_tokens) == (3, 1)
        # Neutral evidence never includes prompts, routes, provider IDs or errors.
        serialized = repr(asdict(snapshot))
        assert "hello" not in serialized and "endpoint.test" not in serialized
    finally:
        await c.aclose()


@pytest.mark.parametrize("client_type", [CompletionClient, ResponsesClient])
@pytest.mark.parametrize("is_async", [False, True])
async def test_paid_parse_failure_no_fabricated_success(monkeypatch, client_type, is_async):
    class Answer(BaseModel):
        answer: int

    requests, snapshots = [], []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200, json=wire(client_type, "not JSON", input_tokens=7, output_tokens=2)
        )

    c = client(monkeypatch, client_type, respond)
    try:
        with _accounting_receiver(snapshots.append), pytest.raises(LLMFailure):
            llm = LLM(c._adapter, retry_config=retry(0))
            request = LLMRequest(messages=[{"role": "user", "content": "hi"}], output_type=Answer)
            if is_async:
                await llm.acall(request)
            else:
                llm.call(request)
        [snapshot] = snapshots
        assert len(requests) == 1 and snapshot.outcome == "error"
        assert snapshot.subtotal("input_tokens").known == 7
        assert snapshot.subtotal("output_tokens").known == 2
        assert snapshot.attempts[0].response_received
    finally:
        await c.aclose()


def test_partial_zero_subtotals_cost_provenance_and_bounded_diagnostics():
    account = _LogicalAccount(detail_limit=1)
    for usage in [
        _NumericUsage(input_tokens=0, cost_usd=0, cost_source="provider"),
        _NumericUsage(
            input_tokens=7,
            output_tokens=2,
            total_tokens=9,
            cost_usd=0.03,
            cost_source="gateway_header",
        ),
    ]:
        index, owner = account.attempt()
        owner.writer.entering_provider()
        owner.writer.delivered(usage)
        account.fold(index, owner)
        account.fold(index, owner)
    snapshot = account.freeze("error")
    assert snapshot is account.freeze("success")
    assert (
        snapshot.subtotal("input_tokens").known == 7 and snapshot.subtotal("input_tokens").complete
    )
    assert (
        snapshot.subtotal("output_tokens").known == 2
        and not snapshot.subtotal("output_tokens").complete
    )
    assert snapshot.subtotal("reasoning_tokens").known is None
    assert snapshot.subtotal("cost_usd").known == 0.03 and snapshot.subtotal("cost_usd").complete
    assert snapshot.cost_sources == ("gateway_header", "provider")
    assert snapshot.attempt_count == 2 and snapshot.dropped_attempts == 1
    owner.writer.delivered(_NumericUsage(input_tokens=1000))
    assert account.snapshot is snapshot and owner.writer.record.usage.input_tokens == 7


@pytest.mark.parametrize("cost", ["nan", "inf", "-1", "bad", True])
def test_invalid_gateway_cost_not_known(cost):
    raw = SimpleNamespace(
        usage={"input_tokens": 0},
        _hidden_params={"additional_headers": {"x-litellm-response-cost": cost}},
    )
    usage = _numeric_usage(raw)
    assert usage.input_tokens == 0 and usage.cost_usd is None
    assert usage.cost_source == "unknown"


def test_gateway_cost_zero_priority_and_breakdowns_not_added_twice():
    raw = SimpleNamespace(
        usage={
            "input_tokens": 7,
            "output_tokens": 2,
            "input_tokens_details": {"cached_tokens": 4},
            "output_tokens_details": {"reasoning_tokens": 1},
        },
        _hidden_params={
            "response_cost": 0.03,
            "additional_headers": {"provider-X-LiteLLM-Response-Cost": "0"},
        },
    )
    usage = _numeric_usage(raw)
    assert usage.cost_usd == 0 and usage.cost_source == "gateway_header"
    assert (
        usage.input_tokens,
        usage.output_tokens,
        usage.total_tokens,
        usage.cached_input_tokens,
        usage.reasoning_tokens,
    ) == (7, 2, 9, 4, 1)


def test_receiver_failure_baseexception_and_unknown_custom_dispatch():
    snapshots = []

    def receive(snapshot):
        snapshots.append(snapshot)
        raise KeyboardInterrupt()

    with _accounting_receiver(receive), pytest.raises(SystemExit):
        with _logical_account_scope() as account:
            index, owner = account.attempt()
            owner.writer.transmitting()  # No actual dispatch proof for custom adapter.
            account.fold(index, owner)
            raise SystemExit(3)
    [snapshot] = snapshots
    assert snapshot.outcome == "error" and snapshot.subtotal("input_tokens").known is None
    assert not snapshot.subtotal("input_tokens").complete


def test_zero_send_terminal_evidence():
    snapshots = []
    with _accounting_receiver(snapshots.append), pytest.raises(ValueError):
        with _logical_account_scope():
            raise ValueError("preflight")
    [snapshot] = snapshots
    assert snapshot.attempt_count == 0
    assert all(total.known == 0 and total.complete for _, total in snapshot.totals)


@pytest.mark.parametrize("client_type", [CompletionClient, ResponsesClient])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("cost", ["0", ".03", "nan", "inf", "-1", "invalid", None])
async def test_installed_sdk_gateway_cost_vs_local_estimate(
    monkeypatch, client_type, is_async, cost
):
    snapshots, requests = [], []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json=wire(client_type),
            headers={} if cost is None else {"x-litellm-response-cost": cost},
        )

    c = client(monkeypatch, client_type, respond)
    try:
        with _accounting_receiver(snapshots.append):
            llm = LLM(c._adapter, retry_config=retry(0))
            request = LLMRequest(messages=[{"role": "user", "content": "hi"}])
            final = await llm.acall(request) if is_async else llm.call(request)
        [snapshot] = snapshots
        assert len(requests) == 1
        assert snapshot.subtotal("input_tokens").known == 3
        valid = cost in {"0", ".03"}
        subtotal = snapshot.subtotal("cost_usd")
        assert subtotal.known == (float(cost) if valid else None)
        assert subtotal.complete is valid
        # The SDK itself rejects malformed billed headers as cost metadata;
        # only an absent header falls back to its local calculator.
        assert snapshot.cost_sources == (
            ("gateway_header",) if valid else (("response_metadata",) if cost is None else ())
        )
        # SDK-calculated final-response usage is left intact; not a paid sum.
        assert final.usage.input_tokens == 3 and final.usage.output_tokens == 1
    finally:
        await c.aclose()


def test_estimated_metadata_not_certified_and_cache_hit_is_known_free():
    account = _LogicalAccount()
    for raw in [
        SimpleNamespace(usage={"input_tokens": 7}, _hidden_params={"response_cost": 0}),
        SimpleNamespace(
            usage={"input_tokens": 3},
            _hidden_params={"additional_headers": {"x-litellm-response-cost": ".03"}},
        ),
    ]:
        index, owner = account.attempt()
        owner.writer.entering_provider()
        owner.writer.delivered(_numeric_usage(raw))
        account.fold(index, owner)
    snapshot = account.freeze("success")
    assert snapshot.subtotal("input_tokens").known == 10
    assert (
        snapshot.subtotal("cost_usd").known == 0.03 and not snapshot.subtotal("cost_usd").complete
    )
    assert snapshot.cost_sources == ("gateway_header", "response_metadata")
    account = _LogicalAccount()
    index, owner = account.attempt()
    owner.writer.entering_provider()
    owner.writer.delivered(
        _numeric_usage(
            SimpleNamespace(usage={"input_tokens": 999}, _hidden_params={"cache_hit": True})
        )
    )
    account.fold(index, owner)
    snapshot = account.freeze("success")
    assert snapshot.subtotal("input_tokens").known == snapshot.subtotal("cost_usd").known == 0
    assert snapshot.subtotal("cost_usd").complete


@pytest.mark.parametrize("is_async", [False, True])
async def test_invalid_request_keeps_typed_config_and_zero_send_receipt(is_async):
    from nooa.unifiedllm import FailureKind
    from nooa.unifiedllm.adapters import Capabilities

    class Adapter:
        capabilities = Capabilities()

        def send_sync(self, request):
            pytest.fail("invalid request dispatched")

        async def send(self, request):
            pytest.fail("invalid request dispatched")

    snapshots = []
    with _accounting_receiver(snapshots.append), pytest.raises(LLMFailure) as caught:
        llm = LLM(Adapter())
        if is_async:
            await llm.acall([])
        else:
            llm.call([])
    assert caught.value.kind is FailureKind.BAD_REQUEST
    [snapshot] = snapshots
    assert snapshot.attempt_count == 0 and snapshot.outcome == "error"
    assert all(total.known == 0 and total.complete for _, total in snapshot.totals)


@pytest.mark.parametrize("client_type", [CompletionClient, ResponsesClient])
async def test_actual_cancel_late_success_cannot_settle_frozen_account(monkeypatch, client_type):
    import asyncio

    entered, release, exited = asyncio.Event(), asyncio.Event(), asyncio.Event()
    requests, snapshots = [], []

    async def respond(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                200,
                json=wire(client_type, empty=True, input_tokens=7, output_tokens=2),
                headers={"x-litellm-response-cost": ".03"},
            )
        if len(requests) == 2:
            entered.set()
            await release.wait()
            return httpx.Response(
                200,
                json=wire(client_type, input_tokens=1000),
                headers={"x-litellm-response-cost": "99"},
            )
        return httpx.Response(
            200, json=wire(client_type), headers={"x-litellm-response-cost": ".01"}
        )

    c = client(monkeypatch, client_type, respond)
    from contextlib import contextmanager

    from nooa.unifiedllm.unifiedllm import _provider_lifetime

    @contextmanager
    def lifetime():
        try:
            yield
        finally:
            if len(requests) == 2:
                exited.set()

    token = _provider_lifetime.set(lifetime)
    try:
        with _accounting_receiver(snapshots.append):
            llm = LLM(c._adapter, retry_config=retry())
            request = LLMRequest(messages=[{"role": "user", "content": "hi"}])
            task = asyncio.create_task(llm.acall(request))
            await asyncio.wait_for(entered.wait(), 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            [snapshot] = snapshots
            assert snapshot.outcome == "cancelled" and snapshot.attempt_count == 2
            assert snapshot.subtotal("input_tokens").known == 7
            assert not snapshot.subtotal("input_tokens").complete
            assert snapshot.subtotal("cost_usd").known == 0.03
            release.set()
            await asyncio.wait_for(exited.wait(), 5)
            assert snapshots == [snapshot]
            assert snapshot.subtotal("input_tokens").known == 7
            final = await llm.acall(request)
            assert len(snapshots) == 2 and snapshots[1].logical_id != snapshot.logical_id
            assert snapshots[1].subtotal("input_tokens").known == final.usage.input_tokens == 3
            assert snapshots[1].subtotal("cost_usd").known == 0.01
            assert len(requests) == 3
    finally:
        release.set()
        _provider_lifetime.reset(token)
        await c.aclose()


@pytest.mark.parametrize("client_type", [CompletionClient, ResponsesClient])
async def test_paid_empty_broker_call_cap_before_second_http(monkeypatch, client_type):
    import asyncio

    from nooa.unifiedllm import AdmissionBroker, AdmissionCallCapError
    from nooa.unifiedllm.admission import _admission_controller_scope

    requests, snapshots, observations = [], [], []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json=wire(client_type, empty=True, input_tokens=7, output_tokens=2),
            headers={"x-litellm-response-cost": ".03"},
        )

    c = client(monkeypatch, client_type, respond)
    with AdmissionBroker(max_in_flight=1, max_calls=1) as broker:
        controller = broker.controller(queue_timeout=2)
        original_acquire = controller.acquire

        async def observed_acquire(observer):
            def record(detail):
                observations.append(detail)
                observer(detail)

            return await original_acquire(record)

        monkeypatch.setattr(controller, "acquire", observed_acquire)
        try:
            with _admission_controller_scope(controller), _accounting_receiver(snapshots.append):
                with pytest.raises(LLMFailure) as caught:
                    await LLM(c._adapter, retry_config=retry()).acall(
                        LLMRequest(messages=[{"role": "user", "content": "hi"}])
                    )
            assert isinstance(caught.value.__cause__, AdmissionCallCapError)
            for _ in range(200):
                state = broker.snapshot()
                if state.active == state.queued == 0:
                    break
                await asyncio.sleep(0.005)
            [snapshot] = snapshots
            assert len(requests) == state.admitted_calls == 1
            assert state.active == state.queued == 0
            assert [d["outcome"] for d in observations] == ["immediate", "call_cap"]
            assert snapshot.attempt_count == 2 and snapshot.outcome == "error"
            paid, refused = snapshot.attempts
            assert paid.provider_entered and paid.response_received
            assert paid.failure_kind == "output_limit"
            assert refused.dispatch_known and not refused.provider_entered
            assert not refused.response_received
            assert snapshot.subtotal("input_tokens").known == 7
            assert snapshot.subtotal("output_tokens").known == 2
            assert snapshot.subtotal("cost_usd").known == pytest.approx(0.03)
            assert snapshot.subtotal("input_tokens").complete
            assert snapshot.subtotal("cost_usd").complete
        finally:
            await c.aclose()
            controller.close()


@pytest.mark.parametrize("failure_site", ["description", "span", "render", "sink"])
async def test_logical_tracing_failure_cannot_prevent_send_or_leak_scope(monkeypatch, failure_site):
    from opentelemetry import trace

    from nooa.tracing import _journal_api
    from nooa.unifiedllm._logical_trace import _LogicalJournal

    before = trace.get_current_span()
    requests, snapshots, spans = [], [], []
    original_init = _LogicalJournal.__init__

    def fail(*args, **kwargs):
        raise ValueError("diagnostic failure")

    def initialize(journal, *args, **kwargs):
        original_init(journal, *args, **kwargs)
        spans.append(journal.span)
        if failure_site == "description":
            journal.describe_request = fail
        elif failure_site == "span":
            journal.span = SimpleNamespace(set_attribute=fail, end=journal.span.end)
        elif failure_site == "render":
            journal.effective_messages = [{"role": "user", "content": "effective"}]

    monkeypatch.setattr(_LogicalJournal, "__init__", initialize)
    if failure_site == "render":
        monkeypatch.setattr(_journal_api, "_input_message", fail)
    elif failure_site == "sink":
        monkeypatch.setattr(_journal_api, "_installed_sinks", fail)

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=wire(CompletionClient))

    c = client(monkeypatch, CompletionClient, respond)
    try:
        with _accounting_receiver(snapshots.append):
            final = await LLM(c._adapter).acall(
                LLMRequest(messages=[{"role": "user", "content": "hi"}])
            )
        assert final.content == "ok" and len(requests) == len(snapshots) == 1
        assert snapshots[0].outcome == "success"
        assert snapshots[0].subtotal("input_tokens").known == 3
        assert trace.get_current_span() is before
        assert all(not span.is_recording() for span in spans)
    finally:
        await c.aclose()


async def test_logical_span_scope_construction_failure_ends_span_and_settles(monkeypatch):
    from opentelemetry import trace

    before = trace.get_current_span()
    spans, requests, snapshots = [], [], []
    tracer = trace.get_tracer("scope-construction-regression")

    def start_span(*args, **kwargs):
        span = tracer.start_span(*args, **kwargs)
        original_end = span.end

        def end(*args, **kwargs):
            spans.append(span)
            original_end(*args, **kwargs)

        span.end = end
        return span

    def fail(*args, **kwargs):
        raise ValueError("scope construction failed")

    monkeypatch.setattr(trace, "get_tracer", lambda *a, **k: SimpleNamespace(start_span=start_span))
    monkeypatch.setattr(trace, "use_span", fail)

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=wire(CompletionClient))

    c = client(monkeypatch, CompletionClient, respond)
    try:
        with _accounting_receiver(snapshots.append):
            final = await LLM(c._adapter).acall(
                LLMRequest(messages=[{"role": "user", "content": "hi"}])
            )
        assert final.content == "ok" and len(requests) == len(snapshots) == 1
        assert snapshots[0].outcome == "success"
        assert snapshots[0].subtotal("input_tokens").known == 3
        assert len(spans) == 1 and not spans[0].is_recording()
        assert trace.get_current_span() is before
    finally:
        await c.aclose()


@pytest.mark.parametrize("cancelled", [False, True])
async def test_logical_span_scope_control_signal_freezes_zero_send(monkeypatch, cancelled):
    from asyncio import CancelledError
    from dataclasses import FrozenInstanceError

    from opentelemetry import trace

    from nooa.unifiedllm._accounting import _current_account

    class TraceControlSignal(BaseException):
        pass

    signal = CancelledError() if cancelled else TraceControlSignal()
    before = trace.get_current_span()
    before_account = _current_account.get()
    spans, requests, snapshots = [], [], []
    tracer = trace.get_tracer("scope-control-signal-regression")

    def start_span(*args, **kwargs):
        span = tracer.start_span(*args, **kwargs)
        original_end = span.end

        def end(*args, **kwargs):
            spans.append(span)
            original_end(*args, **kwargs)

        span.end = end
        return span

    def fail(*args, **kwargs):
        raise signal

    monkeypatch.setattr(trace, "get_tracer", lambda *a, **k: SimpleNamespace(start_span=start_span))
    monkeypatch.setattr(trace, "use_span", fail)

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json=wire(CompletionClient))

    c = client(monkeypatch, CompletionClient, respond)
    try:
        with _accounting_receiver(snapshots.append):
            with pytest.raises(BaseException) as caught:
                await LLM(c._adapter).acall(
                    LLMRequest(messages=[{"role": "user", "content": "hi"}])
                )
        assert caught.value is signal
        assert requests == []
        [snapshot] = snapshots
        assert snapshot.outcome == ("cancelled" if cancelled else "error")
        assert snapshot.attempt_count == snapshot.dropped_attempts == 0
        assert snapshot.attempts == ()
        with pytest.raises(FrozenInstanceError):
            snapshot.outcome = "success"
        assert all(total.known == 0 and total.complete for _, total in snapshot.totals)
        assert len(spans) == 1 and not spans[0].is_recording()
        assert trace.get_current_span() is before
        assert _current_account.get() is before_account
    finally:
        await c.aclose()


def test_logical_account_without_renderer_still_settles(monkeypatch):
    from nooa.unifiedllm import _logical_trace

    monkeypatch.setattr(_logical_trace, "_journal_renderer", None)
    snapshots = []
    with _accounting_receiver(snapshots.append):
        with _logical_account_scope() as first:
            assert first._journal is not None
            assert first._journal._writer is None
        with _logical_account_scope() as second:
            assert second._journal is not None
    assert len(snapshots) == 2
    assert snapshots[0].logical_id != snapshots[1].logical_id
    assert all(s.outcome == "success" and s.attempt_count == 0 for s in snapshots)


def test_logical_renderer_binding_is_idempotent_and_captured(monkeypatch):
    from opentelemetry import trace

    from nooa.unifiedllm import _logical_trace

    monkeypatch.setattr(_logical_trace, "_journal_renderer", None)
    calls = []
    before = trace.get_current_span()

    def first(*args):
        calls.append(("first", args[1].logical_id))

    def second(*args):
        calls.append(("second", args[1].logical_id))

    _logical_trace._bind_journal_renderer(first)
    _logical_trace._bind_journal_renderer(first)
    with _logical_account_scope() as account_a:
        _logical_trace._bind_journal_renderer(second)
        assert account_a._journal._writer is first
    with _logical_account_scope() as account_b:
        assert account_b._journal._writer is second
    assert calls == [("first", account_a.logical_id), ("second", account_b.logical_id)]
    assert trace.get_current_span() is before
    with pytest.raises(TypeError):
        _logical_trace._bind_journal_renderer(None)
    assert _logical_trace._journal_renderer is second
