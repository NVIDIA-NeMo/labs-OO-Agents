# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Event-gated actual SDK tasks, caller terminal freeze and owned pool leases."""

import asyncio
from contextlib import contextmanager

import httpx
import pytest

from nooa.unifiedllm import load_llm, registry
from nooa.unifiedllm._accounting import _accounting_receiver
from nooa.unifiedllm._observations import _collect_observations
from nooa.unifiedllm._transport_lifetime import _provider_lifetime
from nooa.unifiedllm.admission import _admission_controller_scope
from nooa.unifiedllm.request import LLMRequest
from tests.unifiedllm.test_native_load_llm import paid_payload, zero_delay
from tests.unifiedllm.test_native_projection import FAMILIES


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    monkeypatch.setattr(registry, "_loaded", True)
    monkeypatch.setattr(registry, "MODELS", {})


@pytest.mark.parametrize("family", FAMILIES)
async def test_actual_queued_and_entered_cancel_retains_owned_pool_until_exit(family, monkeypatch):
    from nooa.unifiedllm.adapters import _native_transport

    entered, release, exited, queued = (asyncio.Event() for _ in range(4))
    wire_tasks, roots, closes = [], [], []
    lifetime_active = 0
    active = 0
    slot = asyncio.Semaphore(1)

    class Permit:
        def release(self):
            nonlocal active
            active -= 1
            slot.release()

    class Controller:
        async def acquire(self, observer):
            nonlocal active
            if slot.locked():
                queued.set()
            await slot.acquire()
            active += 1
            observer({"outcome": "immediate"})
            return Permit()

    @contextmanager
    def lifetime():
        nonlocal lifetime_active
        lifetime_active += 1
        try:
            yield
        finally:
            lifetime_active -= 1
            exited.set()

    async def handler(request):
        wire_tasks.append(asyncio.current_task())
        entered.set()
        await release.wait()
        return httpx.Response(
            200,
            json=paid_payload(family, input_tokens=900000, output_tokens=900000),
            headers={"x-response-cost": "99999"},
        )

    sync_cls, async_cls = httpx.Client, httpx.AsyncClient

    class SyncRoot(sync_cls):
        def __init__(self, **kwargs):
            assert kwargs["verify"] is True
            super().__init__(
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, json=paid_payload(family))
                ),
                **kwargs,
            )
            roots.append(self)

        def close(self):
            closes.append("sync")
            super().close()

    class AsyncRoot(async_cls):
        def __init__(self, **kwargs):
            assert kwargs["verify"] is True
            super().__init__(transport=httpx.MockTransport(handler), **kwargs)
            roots.append(self)

        async def aclose(self):
            closes.append("async")
            await super().aclose()

    monkeypatch.setattr(_native_transport.httpx, "Client", SyncRoot)
    monkeypatch.setattr(_native_transport.httpx, "AsyncClient", AsyncRoot)
    llm = load_llm(
        "native",
        adapter=family,
        model="model",
        api_key="fixture-key",
        base_url="https://fixture.invalid/v1",
        max_tokens=100,
        retry_config=zero_delay(),
        host_profile={"cost_header": "x-response-cost"},
    )
    pool, calibration = llm._lease, llm.adapter._route_calibration_record
    assert pool.leases == 1 and not wire_tasks
    snapshots = []
    token = _provider_lifetime.set(lifetime)
    try:
        with (
            _accounting_receiver(snapshots.append),
            _collect_observations() as collector,
            _admission_controller_scope(Controller()),
        ):
            request = LLMRequest(messages=[{"role": "user", "content": "hi"}])
            first = asyncio.create_task(llm.acall(request))
            await asyncio.wait_for(entered.wait(), 2)
            assert active == lifetime_active == 1 and pool.leases == 3
            second = asyncio.create_task(llm.acall(request))
            await asyncio.wait_for(queued.wait(), 2)
            second.cancel()
            with pytest.raises(asyncio.CancelledError):
                await second
            assert len(wire_tasks) == 1 and active == lifetime_active == 1
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert len(snapshots) == 2 and active == lifetime_active == 1
            frozen_snapshots, frozen_observations = tuple(snapshots), collector.records
            frozen_ratio = calibration.ratio()
            assert all(
                attempt.finished and not attempt.response_received
                for snapshot in snapshots
                for attempt in snapshot.attempts
            )
            assert pool.leases == 2
            await asyncio.gather(llm.aclose(), llm.aclose())
            assert pool.leases == 1 and not pool.retired and not closes
            assert not wire_tasks[0].done()
            task_resources_released = asyncio.Event()
            wire_tasks[0].add_done_callback(lambda task: task_resources_released.set())
            release.set()
            await asyncio.wait_for(exited.wait(), 2)
            await asyncio.wait_for(task_resources_released.wait(), 2)
            await wire_tasks[0]
            await pool.join_disposal()
            assert pool.leases == active == lifetime_active == 0 and pool.retired
            assert closes.count("sync") == closes.count("async") == 1
            assert all(root.is_closed for root in roots)
            assert tuple(snapshots) == frozen_snapshots and collector.records == frozen_observations
            assert calibration.ratio() == frozen_ratio
            await asyncio.gather(llm.aclose(), llm.aclose())
            assert len(closes) == 2
    finally:
        release.set()
        _provider_lifetime.reset(token)
    # A new load cannot inherit the cancelled receipt or retired pool.
    next_llm = load_llm(
        "native",
        adapter=family,
        model="model",
        api_key="fixture-key",
        base_url="https://fixture.invalid/v1",
        max_tokens=100,
        retry_config=zero_delay(),
        host_profile={"cost_header": "x-response-cost"},
    )
    assert next_llm._lease is not pool
    fresh = []
    with _accounting_receiver(fresh.append):
        response = next_llm.call(request)
    assert response.usage.input_tokens == 3 and response.usage.output_tokens == 1
    assert len(fresh) == 1 and fresh[0].attempt_count == 1
    assert fresh[0].subtotal("input_tokens").known == 3
    await next_llm.aclose()
