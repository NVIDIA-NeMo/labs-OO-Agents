# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual HTTP admission/paid retry and shielded exit reservation evidence."""

import asyncio

import httpx
import pytest
from nooa_bench.gym import PolicyCallBudgetExceeded, _classify, _RecordingClient

from nooa.unifiedllm import RetryConfig
from nooa.unifiedllm.unifiedllm import _ClientHttp


def response(text="ok"):
    return {
        "id": "r",
        "object": "response",
        "created_at": 1,
        "model": "gpt-4o-mini",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "m",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9},
    }


def recording(monkeypatch, handler):
    monkeypatch.setattr(
        _ClientHttp,
        "_httpx_hardening",
        staticmethod(lambda: {"transport": httpx.MockTransport(handler)}),
    )
    records = []
    client = _RecordingClient(
        "openai/gpt-4o-mini",
        api_key="test",
        api_base="https://endpoint.test/v1",
        max_policy_calls=1,
        records=records,
        max_retries=0,
        retry_config=RetryConfig(
            max_retries=2,
            rate_limit_extra_retries=0,
            base_delay=0,
            max_delay=0,
            jitter_factor=0,
            retry_on_empty_content=True,
        ),
    )
    return client, records


@pytest.mark.parametrize("invalid", [False, True])
async def test_paid_retry_cannot_dispatch_second_http(monkeypatch, invalid):
    import json

    from nooa_bench.gym import _current_slot, _PolicySlot
    from pydantic import BaseModel

    from nooa.unifiedllm import ResponsesClient
    from nooa.unifiedllm.unifiedllm import _provider_lifetime

    class Answer(BaseModel):
        answer: int

    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=response("not JSON" if invalid else ""))

    client, records = recording(monkeypatch, handler)
    # Legacy Responses deliberately neither retries parse errors nor checks empty
    # replies. Adversarial retry inside the SAME admitted reservation tests the
    # actual HTTP hook without changing that success policy or retry classifier.
    slot = _PolicySlot()
    client._slots.add(slot)
    slot_token = _current_slot.set(slot)
    lifetime_token = _provider_lifetime.set(lambda: client._provider_scope(slot))
    try:
        if invalid:
            with pytest.raises(json.JSONDecodeError):
                await ResponsesClient.acall(
                    client, [{"role": "user", "content": "hi"}], output_model=Answer
                )
        else:
            result = await ResponsesClient.acall(client, [{"role": "user", "content": "hi"}])
            assert result.content == ""
        assert slot.answered
        with pytest.raises(PolicyCallBudgetExceeded) as caught:
            await ResponsesClient.acall(client, [{"role": "user", "content": "hi"}])
        slot.closed = True
        client._cleanup_slot(slot)
        assert _classify(caught.value)[0] == "policy_budget_exceeded"
        assert len(requests) == len(records) == 1 and records[0].status_code == 200
        assert requests[0].headers["x-session-id"] == "root"
        assert not client._slots and not client._slot_of and not client._pending
    finally:
        _provider_lifetime.reset(lifetime_token)
        _current_slot.reset(slot_token)
        slot.closed = True
        client._cleanup_slot(slot)
        await client.aclose()


async def test_http_errors_reuse_unanswered_reservation(monkeypatch):
    requests = []

    def handler(request):
        requests.append(request)
        status = [429, 503, 200][len(requests) - 1]
        return httpx.Response(
            status,
            json=response()
            if status == 200
            else {"error": {"message": "retry", "type": "server_error"}},
        )

    client, records = recording(monkeypatch, handler)
    try:
        result = await client.acall([{"role": "user", "content": "hi"}])
        assert result.content == "ok"
        assert len(requests) == len(records) == 3
        assert [r.status_code for r in records] == [429, 503, 200]
        assert not client._slots and not client._pending and not client._slot_of
        with pytest.raises(PolicyCallBudgetExceeded):
            await client.acall([{"role": "user", "content": "again"}])
        assert len(requests) == 3
    finally:
        await client.aclose()


async def test_cancelled_http_keeps_slot_until_actual_exit(monkeypatch):
    entered, release, exited = asyncio.Event(), asyncio.Event(), asyncio.Event()
    requests = []

    async def handler(request):
        requests.append(request)
        entered.set()
        await release.wait()
        return httpx.Response(200, json=response())

    client, records = recording(monkeypatch, handler)
    original = client._provider_scope
    from contextlib import contextmanager

    @contextmanager
    def scope(slot):
        try:
            with original(slot):
                yield
        finally:
            exited.set()

    monkeypatch.setattr(client, "_provider_scope", scope)
    task = asyncio.create_task(client.acall([{"role": "user", "content": "hi"}]))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(client._slots) == len(client._pending) == len(client._slot_of) == 1
        with pytest.raises(PolicyCallBudgetExceeded):
            await client.acall([{"role": "user", "content": "child"}])
        assert len(requests) == 1
        release.set()
        await asyncio.wait_for(exited.wait(), 5)
        assert records[0].status_code == 200
        assert not client._slots and not client._pending and not client._slot_of
    finally:
        release.set()
        await client.aclose()


@pytest.mark.parametrize("kind", ["empty", "parse", "truncated"])
async def test_strict_public_retry_respects_existing_http_slot(monkeypatch, kind):
    """HOOK/COMPAT scaffold, not a claim of current Gym public-factory reachability."""
    from nooa_bench.gym import _current_slot, _PolicySlot
    from pydantic import BaseModel

    from nooa.unifiedllm import LLM, LLMFailure, LLMRequest
    from nooa.unifiedllm._accounting import _accounting_receiver
    from nooa.unifiedllm.unifiedllm import _provider_lifetime

    class Answer(BaseModel):
        answer: int

    requests, snapshots, refusals = [], [], []

    def handler(request):
        requests.append(request)
        body = response("not JSON" if kind == "parse" else "")
        if kind != "parse":
            body["output"] = [
                {
                    "id": "thought",
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "readable thought"}],
                }
            ]
        if kind == "truncated":
            body.update(status="incomplete", incomplete_details={"reason": "max_output_tokens"})
        return httpx.Response(200, json=body, headers={"x-litellm-response-cost": ".03"})

    client, records = recording(monkeypatch, handler)
    original_check = client._check_budget

    def check():
        try:
            original_check()
        except PolicyCallBudgetExceeded as error:
            refusals.append(error)
            raise

    monkeypatch.setattr(client, "_check_budget", check)
    slot = _PolicySlot()
    client._slots.add(slot)
    slot_token = _current_slot.set(slot)
    lifetime_token = _provider_lifetime.set(lambda: client._provider_scope(slot))
    llm = LLM(client._adapter, retry_config=client.retry_config)
    try:
        with _accounting_receiver(snapshots.append):
            if kind == "parse":
                # Parse failure remains terminal: a subsequent call in this
                # reservation still cannot dispatch another paid HTTP request.
                with pytest.raises(LLMFailure):
                    await llm.acall(
                        LLMRequest(
                            messages=[{"role": "user", "content": "hi"}],
                            output_type=Answer,
                        )
                    )
            with pytest.raises(LLMFailure) as caught:
                await llm.acall(LLMRequest(messages=[{"role": "user", "content": "hi"}]))
        # The public typed boundary has the exact restored raw policy object as
        # its cause; no cause probing or SDK-error retry may dispatch again.
        assert len(refusals) == 1
        assert caught.value.__cause__ is refusals[0]
        assert _classify(refusals[0])[0] == "policy_budget_exceeded"
        assert len(requests) == len(records) == 1
        assert snapshots[0].subtotal("input_tokens").known == 7
        assert snapshots[0].subtotal("cost_usd").known == pytest.approx(0.03)
        assert slot.refusal is None and slot.active == 0
        slot.closed = True
        client._cleanup_slot(slot)
        assert not client._slots and not client._pending and not client._slot_of
    finally:
        _provider_lifetime.reset(lifetime_token)
        _current_slot.reset(slot_token)
        slot.closed = True
        client._cleanup_slot(slot)
        await client.aclose()
