# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public calibration authority without changing legacy hooks or response usage."""

import asyncio
import json

import litellm
import pytest

from nooa.llm_types import LLMResponse, LLMUsage
from nooa.unifiedllm import LLM, LLMFailure, LLMRequest, RetryConfig
from nooa.unifiedllm import unifiedllm as core
from nooa.unifiedllm._observations import _collect_observations, _current_collector
from nooa.unifiedllm.adapters import Capabilities
from nooa.unifiedllm.adapters import _calibration as routes
from nooa.unifiedllm.adapters._binding import _bind_adapter
from tests.unifiedllm.test_private_wire_observations import (
    MESSAGES,
    Answer,
    make_client,
    reply,
    stub_provider,
)

MODES = pytest.mark.parametrize("is_async", [False, True])
FAMILIES = pytest.mark.parametrize("family", ["chat", "responses"])


@pytest.fixture(autouse=True)
def calibration(monkeypatch):
    class Legacy(core.TokenCalibration):
        def __init__(self):
            super().__init__()
            self.updates = []

        def update(self, *args):
            self.updates.append(args)
            super().update(*args)

    legacy = Legacy()
    monkeypatch.setattr(core, "_token_calibration", legacy)
    monkeypatch.setattr(routes, "_records", {})
    updates, estimates = [], []
    update = routes._RouteCalibration.update

    def counted(record, estimated, actual):
        updates.append((record, estimated, actual))
        update(record, estimated, actual)

    def estimate(**kwargs):
        estimates.append(kwargs)
        return 20

    monkeypatch.setattr(routes._RouteCalibration, "update", counted)
    monkeypatch.setattr(litellm, "token_counter", estimate)
    return legacy, updates, estimates


def raw_reply(family, actual=40, *, empty=False, malformed=False):
    raw = reply(
        family,
        content="" if empty else "not JSON 429 503" if malformed else "ok",
        usage=None if actual is None else {"input_tokens": actual, "output_tokens": 2},
    )
    if empty:
        if family == "chat":
            raw.choices[0].message.reasoning_content = "thinking"
        else:
            raw.output = [
                {
                    "type": "reasoning",
                    "id": "reason",
                    "summary": [{"type": "summary_text", "text": "thinking"}],
                }
            ]
    return raw


def retry_config(max_retries=2):
    return RetryConfig(
        max_retries=max_retries,
        base_delay=0,
        max_delay=0,
        jitter_factor=0,
        retry_on_empty_content=True,
    )


async def call(llm, request, is_async):
    return await llm.acall(request) if is_async else llm.call(request)


@FAMILIES
@MODES
@pytest.mark.parametrize(
    "outcome",
    [
        "success",
        "malformed",
        "empty",
        "exhausted",
        "unknown",
        "zero",
        "empty_success",
        "unknown_success",
        "transport_success",
        "no_response",
    ],
)
async def test_public_paid_attempt_count_parity(
    monkeypatch, family, is_async, outcome, calibration
):
    legacy, updates, estimates = calibration
    client = make_client(family)
    sent = []

    def provider(**kwargs):
        sent.append(kwargs)
        if outcome == "no_response" or (outcome == "transport_success" and len(sent) == 1):
            raise ConnectionError("offline transient")
        empty = outcome in {"empty", "exhausted"} or (
            outcome in {"empty_success", "unknown_success"} and len(sent) == 1
        )
        actual = (
            None
            if outcome == "unknown" or (outcome == "unknown_success" and len(sent) == 1)
            else 0
            if outcome == "zero"
            else 40
            if len(sent) == 1
            else 60
        )
        return raw_reply(family, actual, empty=empty, malformed=outcome == "malformed")

    stub_provider(monkeypatch, family, provider)
    config = retry_config(2) if outcome != "empty" else None
    llm = LLM(client._adapter, retry_config=config)
    request = LLMRequest(messages=MESSAGES, output_type=Answer if outcome == "malformed" else None)
    expected_sends = (
        3
        if outcome in {"exhausted", "no_response"}
        else 2
        if outcome in {"empty_success", "unknown_success", "transport_success"}
        else 1
    )
    try:
        with _collect_observations() as collector:
            if outcome in {"malformed", "exhausted", "no_response"}:
                error = (
                    json.JSONDecodeError
                    if outcome == "malformed"
                    else (core.EmptyContentError if outcome == "exhausted" else ConnectionError)
                )
                with pytest.raises(error):
                    await call(llm, request, is_async)
            else:
                response = await call(llm, request, is_async)
                assert response.content == ("" if outcome == "empty" else "ok")
                if outcome not in {"unknown", "zero", "unknown_success"}:
                    assert response.usage.input_tokens == (60 if expected_sends == 2 else 40)
            assert len(collector.records) == expected_sends
            positive = [
                r
                for r in collector.records
                if r.response_received
                and r.usage is not None
                and r.usage.input_tokens is not None
                and r.usage.input_tokens > 0
            ]
            assert len(updates) == len(positive)
            assert all(r.finished for r in collector.records)
        assert len(sent) == len(estimates) == expected_sends  # no second estimate/success hook
        assert legacy.updates == []
        record = routes._route_calibration(client._adapter)
        expected = 1.0
        for observed in positive:
            expected = 0.3 * (observed.usage.input_tokens / 20) + 0.7 * expected
        assert record.ratio() == pytest.approx(expected)
        assert all(r is record and estimate == 20 for r, estimate, actual in updates)
        assert _current_collector.get() is None
    finally:
        await client.aclose()


@FAMILIES
@MODES
async def test_buffer_limit_does_not_limit_paid_calibration(
    monkeypatch, family, is_async, calibration
):
    legacy, updates, estimates = calibration
    client = make_client(family)
    stub_provider(monkeypatch, family, lambda **kwargs: raw_reply(family, empty=True))
    try:
        with _collect_observations(limit=1) as collector:
            with pytest.raises(core.EmptyContentError):
                await call(
                    LLM(client._adapter, retry_config=retry_config(260)),
                    LLMRequest(messages=MESSAGES),
                    is_async,
                )
            assert len(collector.records) == 1 and collector.dropped_attempts == 260
        assert len(updates) == len(estimates) == 261
        assert legacy.updates == []
    finally:
        await client.aclose()


@FAMILIES
@MODES
async def test_warm_seed_shared_views_and_legacy_estimate_boundary(
    monkeypatch, family, is_async, calibration
):
    legacy, updates, estimates = calibration
    client, other = make_client(family), make_client(family)
    other.config["api_base"] = "https://different.test/v1"
    legacy.update(client.model, 20, 60)
    stub_provider(monkeypatch, family, lambda **kwargs: raw_reply(family, 100))
    try:
        assert client._adapter._route_calibration_record is None
        first = _bind_adapter(client._adapter, temperature=0.1)
        second = _bind_adapter(first, temperature=0.2)
        record = routes._route_calibration(first)
        different = routes._route_calibration(other._adapter)
        legacy.updates.clear()
        for adapter in (first, second, client._adapter):
            await call(LLM(adapter), LLMRequest(messages=MESSAGES), is_async)
            assert routes._route_calibration(adapter) is record
        assert record.ratio() == pytest.approx(5 - 2 * 0.7**3)
        assert different.ratio() == 3
        assert legacy.ratio(client.model) == 3 and legacy.updates == []
        assert client.count_tokens("still legacy") == 60
        assert first._client._http is second._client._http is client._http
        assert len(updates) == 3
    finally:
        await client.aclose()
        await other.aclose()


@FAMILIES
@MODES
async def test_nested_same_backing_legacy_send_is_not_owned(
    monkeypatch, family, is_async, calibration
):
    legacy, updates, estimates = calibration
    client = make_client(family)
    entered = False
    name = "completion" if family == "chat" else "responses"

    def sync_provider(**kwargs):
        nonlocal entered
        if not entered:
            entered = True
            client.call(MESSAGES)
        return raw_reply(family)

    async def async_provider(**kwargs):
        nonlocal entered
        if not entered:
            entered = True
            await client.acall(MESSAGES)
        return raw_reply(family)

    monkeypatch.setattr(litellm, name, sync_provider)
    monkeypatch.setattr(litellm, "a" + name, async_provider)
    try:
        with _collect_observations() as collector:
            await call(LLM(client._adapter), LLMRequest(messages=MESSAGES), is_async)
            assert len(collector.records) == 2
        assert len(updates) == len(legacy.updates) == 1
        assert len(estimates) == 3  # public once, observation-only legacy plus legacy hook
        assert client._adapter._legacy_options is None
    finally:
        await client.aclose()


@FAMILIES
async def test_simultaneous_public_and_legacy_same_adapter(monkeypatch, family, calibration):
    legacy, updates, estimates = calibration
    client = make_client(family)
    name = "acompletion" if family == "chat" else "aresponses"
    entered, release = asyncio.Event(), asyncio.Event()
    count = 0

    async def provider(**kwargs):
        nonlocal count
        count += 1
        if count == 2:
            entered.set()
        await release.wait()
        return raw_reply(family)

    monkeypatch.setattr(litellm, name, provider)
    try:
        public = asyncio.create_task(LLM(client._adapter).acall(LLMRequest(messages=MESSAGES)))
        old = asyncio.create_task(client.acall(MESSAGES))
        await entered.wait()
        release.set()
        await asyncio.gather(public, old)
        assert count == len(estimates) == 2
        assert len(updates) == len(legacy.updates) == 1
    finally:
        release.set()
        await client.aclose()


@FAMILIES
async def test_cancelled_public_late_result_never_calibrates(monkeypatch, family, calibration):
    legacy, updates, estimates = calibration
    client = make_client(family)
    name = "acompletion" if family == "chat" else "aresponses"
    entered, release = asyncio.Event(), asyncio.Event()
    tasks, collectors = [], []

    async def provider(**kwargs):
        tasks.append(asyncio.current_task())
        entered.set()
        await release.wait()
        return raw_reply(family)

    monkeypatch.setattr(litellm, name, provider)

    async def caller():
        try:
            with _collect_observations() as collector:
                collectors.append(collector)
                await LLM(client._adapter).acall(LLMRequest(messages=MESSAGES))
        finally:
            assert _current_collector.get() is None

    task = asyncio.create_task(caller())
    try:
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        before = collectors[0].records
        assert before[0].finished and not before[0].response_received
        assert before[0].usage is None and not tasks[0].done()
        with _collect_observations() as unrelated:
            release.set()
            await tasks[0]
            assert unrelated.records == ()
        assert updates == legacy.updates == []
        assert collectors[0].records == before
        # A subsequent direct legacy call retains the normal authority.
        await client.acall(MESSAGES)
        assert updates == [] and len(legacy.updates) == 1
    finally:
        release.set()
        await client.aclose()


@MODES
async def test_strict_custom_has_only_original_protocol(is_async, calibration):
    class Strict:
        __slots__ = ()
        capabilities = Capabilities()

        def __copy__(self):
            pytest.fail("custom must not be copied")

        def send_sync(self, request):
            return LLMResponse(usage=LLMUsage(input_tokens=9))

        async def send(self, request):
            return self.send_sync(request)

    legacy, updates, estimates = calibration
    with _collect_observations() as collector:
        response = await call(LLM(Strict()), LLMRequest(messages=MESSAGES), is_async)
        assert response.usage.input_tokens == 9 and collector.records == ()
    assert updates == legacy.updates == estimates == []


@FAMILIES
@MODES
@pytest.mark.parametrize("estimate", [None, 0])
async def test_absent_or_zero_estimate_no_fabricated_update(
    monkeypatch, family, is_async, estimate, calibration
):
    legacy, updates, estimates = calibration
    client = make_client(family)
    module = "nooa.unifiedllm.adapters.litellm_" + family
    if family == "chat":
        monkeypatch.setattr(
            module + "._estimate_projected_input_tokens", lambda *args, **kw: (estimate, "full")
        )
    else:
        monkeypatch.setattr(
            core, "_estimate_projected_input_tokens", lambda *args, **kw: (estimate, "full")
        )
    stub_provider(monkeypatch, family, lambda **kwargs: raw_reply(family))
    try:
        await call(LLM(client._adapter), LLMRequest(messages=MESSAGES), is_async)
        assert updates == legacy.updates == estimates == []
    finally:
        await client.aclose()


@FAMILIES
@MODES
async def test_preflight_does_not_seed_or_observe(family, is_async, calibration):
    legacy, updates, estimates = calibration
    client = make_client(family)
    try:
        llm = LLM(client._adapter)
        assert client._adapter._route_calibration_record is None
        with _collect_observations() as collector:
            with pytest.raises(LLMFailure):
                await call(llm, LLMRequest(messages=MESSAGES, tool_choice="missing"), is_async)
            assert collector.records == ()
        assert routes._records == {} and client._adapter._route_calibration_record is None
        assert updates == legacy.updates == estimates == []
    finally:
        await client.aclose()


@FAMILIES
@MODES
async def test_nested_unrelated_route_keeps_legacy_authority(
    monkeypatch, family, is_async, calibration
):
    legacy, updates, estimates = calibration
    client, other = make_client(family), make_client(family)
    other.config["api_base"] = "https://unrelated.test/v1"
    unrelated = routes._route_calibration(other._adapter)
    name = "completion" if family == "chat" else "responses"

    def sync_provider(**kwargs):
        if kwargs["api_base"] != other.config["api_base"]:
            other.call(MESSAGES)
        return raw_reply(family, 80 if kwargs["api_base"] == other.config["api_base"] else 40)

    async def async_provider(**kwargs):
        if kwargs["api_base"] != other.config["api_base"]:
            await other.acall(MESSAGES)
        return raw_reply(family, 80 if kwargs["api_base"] == other.config["api_base"] else 40)

    monkeypatch.setattr(litellm, name, sync_provider)
    monkeypatch.setattr(litellm, "a" + name, async_provider)
    try:
        await call(LLM(client._adapter), LLMRequest(messages=MESSAGES), is_async)
        assert len(updates) == len(legacy.updates) == 1
        assert updates[0][0] is routes._route_calibration(client._adapter)
        assert updates[0][2] == 40 and legacy.updates[0][2] == 80
        assert unrelated.ratio() == 1 and len(estimates) == 2
    finally:
        await client.aclose()
        await other.aclose()


@FAMILIES
@MODES
@pytest.mark.parametrize("override", ["subclass", "instance"])
async def test_private_participation_never_bypasses_custom_send(
    family, is_async, override, calibration
):
    client = make_client(family)
    seen = []

    def send_sync(request):
        seen.append(request)
        return LLMResponse(usage=LLMUsage(input_tokens=9))

    async def send(request):
        return send_sync(request)

    adapter = client._adapter
    if override == "subclass":

        class Custom(type(adapter)):
            def send_sync(self, request):
                return send_sync(request)

            async def send(self, request):
                return await send(request)

        adapter = Custom(client)
    else:
        adapter.send_sync = send_sync
        adapter.send = send
    legacy, updates, estimates = calibration
    try:
        with _collect_observations() as collector:
            result = await call(LLM(adapter), LLMRequest(messages=MESSAGES), is_async)
        assert result.usage.input_tokens == 9 and len(seen) == 1
        assert collector.records == ()
        assert updates == legacy.updates == estimates == []
    finally:
        await client.aclose()
