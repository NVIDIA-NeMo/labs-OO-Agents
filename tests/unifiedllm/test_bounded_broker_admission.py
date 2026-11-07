# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded broker queues over real sockets and across spawned processes."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import time
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import litellm
import pytest

from nooa.unifiedllm import (
    AdmissionBroker,
    AdmissionCallCapError,
    AdmissionControl,
    AdmissionControlConfig,
    AdmissionQueueFullError,
    AdmissionTimeoutError,
    BrokerAdmissionConfig,
    CompletionClient,
    RetryConfig,
)
from nooa.unifiedllm import broker_admission as broker_module
from nooa.unifiedllm.broker_admission import _PROTOCOL_VERSION, _BrokerState


async def _queued(broker: AdmissionBroker, count: int) -> None:
    """Synchronize with broker state instead of depending on network sleeps."""
    async with asyncio.timeout(5):
        while broker.snapshot().queued != count:
            await asyncio.sleep(0.005)


@pytest.mark.parametrize("phase", ["offered", "ready"])
@pytest.mark.asyncio
async def test_expired_handshake_never_dispatches_and_bounded_owner_recovers(
    phase: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Elapsed deadlines are terminal even with free slots, not a deadlock or retry."""
    real_clock = time.perf_counter
    offset = 0.0
    expired = False
    monkeypatch.setattr(
        broker_module, "time", SimpleNamespace(perf_counter=lambda: real_clock() + offset)
    )
    with AdmissionBroker(max_in_flight=1, max_pending=1, max_calls=2) as broker:
        controller = broker.controller(queue_timeout=5)
        original_readline = controller._readline

        async def pause_after_response(reader, started):
            """Advance only this module's elapsed clock at a causal protocol boundary."""
            nonlocal offset, expired
            data = await original_readline(reader, started)
            if not expired and data and json.loads(data).get("status") == phase:
                offset += 10
                expired = True
            return data

        monkeypatch.setattr(controller, "_readline", pause_after_response)
        llm = AdmissionControl(
            CompletionClient(
                "openai/gpt-4o-mini",
                api_base="http://127.0.0.1:1/v1",
                api_key="inert-test-key",  # noqa: S106
                retry_config=RetryConfig(max_retries=3, rate_limit_extra_retries=0),
            ),
            AdmissionControlConfig(controller=controller),
        )
        response = litellm.ModelResponse(
            model="gpt-4o-mini",
            choices=[
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
        )
        messages = [{"role": "user", "content": "synthetic deadline test"}]
        try:
            with patch("litellm.acompletion", AsyncMock(return_value=response)) as provider:
                with patch.object(
                    controller, "acquire", AsyncMock(wraps=controller.acquire)
                ) as acquire:
                    with pytest.raises(AdmissionTimeoutError):
                        await llm.acall(messages)
                    assert expired
                    assert acquire.await_count == 1
                    assert provider.await_count == 0
                offset = 0.0
                async with asyncio.timeout(5):
                    while broker.snapshot().active or broker.snapshot().queued:
                        await asyncio.sleep(0.005)
                # Keep the same owner and controller; do not hide leaks by restarting.
                assert (await llm.acall(messages)).content == "ok"
                assert provider.await_count == 1
                async with asyncio.timeout(5):
                    while broker.snapshot().active or broker.snapshot().queued:
                        await asyncio.sleep(0.005)
                # A final acknowledgement may race expiry and conservatively commit
                # a budget unit. No provider call is permitted for that expired lease.
                assert 1 <= broker.snapshot().admitted_calls <= 2
        finally:
            offset = 0.0
            await llm.aclose()
            controller.close()
            await asyncio.sleep(0.02)


@pytest.mark.parametrize("pending", [None, 0, 1, 8])
@pytest.mark.asyncio
async def test_broker_enforces_bound_and_preserves_fifo(pending: int | None) -> None:
    """Only waiting leases count, and rejection never consumes the call budget."""
    with AdmissionBroker(max_in_flight=1, max_pending=pending, max_calls=21) as broker:
        controller = broker.controller(queue_timeout=10)
        first = await controller.acquire(lambda _: None)
        observations = []
        tasks = [asyncio.create_task(controller.acquire(observations.append)) for _ in range(20)]
        keep = 20 if pending is None else pending
        try:
            await _queued(broker, keep)
            async with asyncio.timeout(5):
                while sum(t.done() for t in tasks) != 20 - keep:
                    await asyncio.sleep(0.005)
            snapshot = broker.snapshot()
            assert snapshot.active == snapshot.admitted_calls == 1
            assert snapshot.max_pending == pending
            first.release()
            accepted = 0
            for task in tasks:
                try:
                    permit = await task
                except AdmissionQueueFullError:
                    continue
                accepted += 1
                permit.release()
            assert accepted == keep
            await _queued(broker, 0)
            assert broker.snapshot().admitted_calls == 1 + keep
            assert broker.snapshot().peak_active == 1
            full = [o for o in observations if o["outcome"] == "queue_full"]
            assert len(full) == 20 - keep
            assert all(not o["queued"] and o["queue_depth"] == keep for o in full)
        finally:
            first.release()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            controller.close()


@pytest.mark.parametrize("cause", ["cancel", "timeout"])
@pytest.mark.asyncio
async def test_broker_abandoned_wait_reclaims_pending_capacity(cause: str) -> None:
    """Cancelled or expired clients must not permanently fill the bounded queue."""
    with AdmissionBroker(max_in_flight=1, max_pending=1) as broker:
        controller = broker.controller()
        expiring = broker.controller(queue_timeout=0.2 if cause == "timeout" else None)
        first = await controller.acquire(lambda _: None)
        waiter = asyncio.create_task(expiring.acquire(lambda _: None))
        replacement = None
        try:
            await _queued(broker, 1)
            if cause == "cancel":
                waiter.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await waiter
            else:
                with pytest.raises(AdmissionTimeoutError):
                    await waiter
            await _queued(broker, 0)
            replacement = asyncio.create_task(controller.acquire(lambda _: None))
            await _queued(broker, 1)
            first.release()
            (await replacement).release()
            await _queued(broker, 0)
        finally:
            first.release()
            for task in (waiter, replacement):
                if task is not None:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            controller.close()
            expiring.close()


@pytest.mark.parametrize("gate", ["pacing", "cooldown", "reservation"])
@pytest.mark.asyncio
async def test_broker_bounds_waiting_for_all_gates(gate: str) -> None:
    """Unused concurrency must not bypass a pacing, cooldown, or budget gate."""
    state = _BrokerState(
        group="bounded",
        max_in_flight=4,
        max_calls=1 if gate == "reservation" else None,
        max_pending=0,
        requests_per_second=0.1 if gate == "pacing" else None,
        max_cooldown=10,
    )
    try:
        offered = state.enqueue("first")
        assert (await offered.future)["status"] == "offered"
        if gate != "reservation":
            state.confirm(offered)
        if gate == "cooldown":
            state.cooldown(10)
        rejected = state.enqueue("second")
        assert (await rejected.future)["status"] == "queue_full"
        assert not state.waiters
        assert state.admitted_calls == (0 if gate == "reservation" else 1)
        state.disconnect(offered)
        if gate == "reservation":
            # An abandoned offer refunds the reservation, not a pending slot.
            next_offer = state.enqueue("replacement")
            assert (await next_offer.future)["status"] == "offered"
            state.disconnect(next_offer)
    finally:
        state.close()


@pytest.mark.asyncio
async def test_committed_call_cap_takes_precedence_over_full_queue() -> None:
    """Permanent exhaustion remains distinct from transient queue pressure."""
    with AdmissionBroker(max_in_flight=1, max_pending=0, max_calls=1) as broker:
        controller = broker.controller()
        first = await controller.acquire(lambda _: None)
        try:
            with pytest.raises(AdmissionCallCapError):
                await controller.acquire(lambda _: None)
            assert broker.snapshot().admitted_calls == 1
        finally:
            first.release()
            controller.close()


@pytest.mark.asyncio
async def test_fifo_and_granted_slots_are_distinct_from_pending() -> None:
    """Transferred offers free queue positions without creating extra capacity."""
    state = _BrokerState(group="bounded", max_in_flight=1, max_calls=None, max_pending=2)
    try:
        first = state.enqueue("active")
        state.confirm(first)
        a, b = state.enqueue("a"), state.enqueue("b")
        rejected = state.enqueue("overflow")
        assert (await rejected.future)["status"] == "queue_full"
        assert [w.ticket for w in state.waiters] == ["a", "b"]
        state.disconnect(first)
        assert a.state == "offered" and b.state == "queued"
        # The grant is active even though the client has not acknowledged it.
        replacement = state.enqueue("replacement")
        assert replacement.state == "queued"
        assert [w.ticket for w in state.waiters] == ["b", "replacement"]
        state.disconnect(a)
        assert b.state == "offered" and replacement.state == "queued"
        state.confirm(b)
        state.disconnect(b)
        assert replacement.state == "offered"
        state.disconnect(replacement)
        assert state.active == state.reserved_calls == len(state.waiters) == 0
        assert state.admitted_calls == 2
    finally:
        state.close()


@pytest.mark.asyncio
async def test_disconnected_socket_returns_a_pending_position() -> None:
    """Abrupt worker connection loss removes its bounded waiter without a lease."""
    with AdmissionBroker(max_in_flight=1, max_pending=1) as broker:
        controller = broker.controller()
        first = await controller.acquire(lambda _: None)
        config = broker.controller_config()
        reader, writer = await asyncio.open_connection(config.host, config.port)
        del reader
        try:
            writer.write(
                json.dumps(
                    {
                        "version": _PROTOCOL_VERSION,
                        "auth_token": config.auth_token,
                        "group": config.group,
                        "ticket": "departing-worker",
                    }
                ).encode()
                + b"\n"
            )
            await writer.drain()
            await _queued(broker, 1)
            writer.close()
            await writer.wait_closed()
            await _queued(broker, 0)
            assert broker.snapshot().admitted_calls == 1
            replacement = asyncio.create_task(controller.acquire(lambda _: None))
            try:
                await _queued(broker, 1)
                first.release()
                (await replacement).release()
            finally:
                replacement.cancel()
                await asyncio.gather(replacement, return_exceptions=True)
        finally:
            writer.close()
            await writer.wait_closed()
            first.release()
            controller.close()


@pytest.mark.asyncio
async def test_queue_full_observer_cannot_replace_rejection() -> None:
    """Instrumentation failure must not disguise overload or leak a lease."""

    def broken(_detail: dict[str, Any]) -> None:
        raise RuntimeError("observer failure")

    with AdmissionBroker(max_in_flight=1, max_pending=0) as broker:
        controller = broker.controller()
        first = await controller.acquire(lambda _: None)
        try:
            with pytest.raises(AdmissionQueueFullError):
                await controller.acquire(broken)
            assert broker.snapshot().active == broker.snapshot().admitted_calls == 1
        finally:
            first.release()
            controller.close()


@pytest.mark.parametrize("bad", [-1, True, 1.5, "2"])
def test_broker_validates_pending_configuration(bad: Any) -> None:
    """Both owner and serialized connection settings reject invalid bounds."""
    with pytest.raises((TypeError, ValueError), match="max_pending"):
        AdmissionBroker(max_in_flight=1, max_pending=bad)
    config = BrokerAdmissionConfig("127.0.0.1", 1, "inert", "bounded", 1, None, None)
    with pytest.raises((TypeError, ValueError), match="max_pending"):
        replace(config, max_pending=bad)


@pytest.mark.asyncio
async def test_broker_config_roundtrip_and_old_protocol_fail_closed() -> None:
    """Bounds survive transport to children, and mixed-version peers dispatch nothing."""
    with AdmissionBroker(max_in_flight=1, max_pending=2) as broker:
        config = broker.controller_config()
        assert BrokerAdmissionConfig(**asdict(config)) == config
        assert config.max_pending == 2
        reader, writer = await asyncio.open_connection(config.host, config.port)
        try:
            writer.write(
                json.dumps(
                    {
                        "version": _PROTOCOL_VERSION - 1,
                        "auth_token": config.auth_token,
                        "group": config.group,
                        "ticket": "old-worker",
                    }
                ).encode()
                + b"\n"
            )
            await writer.drain()
            assert json.loads(await reader.readline())["status"] == "unauthorized"
            assert broker.snapshot().admitted_calls == 0
        finally:
            writer.close()
            await writer.wait_closed()


def _contend_in_child(config: BrokerAdmissionConfig, ready: Any, start: Any, results: Any) -> None:
    """Start contenders only after all spawned interpreters have imported."""
    ready.put("ready")
    assert start.wait(60)

    async def run() -> None:
        controller = config.controller()

        async def one() -> None:
            try:
                permit = await controller.acquire(lambda _: None)
            except AdmissionQueueFullError:
                results.put("full")
            else:
                permit.release()
                results.put("ok")

        try:
            await asyncio.gather(*(one() for _ in range(10)))
        finally:
            controller.close()
            await asyncio.sleep(0)

    asyncio.run(run())


@pytest.mark.timeout(180)
def test_spawned_workers_share_one_pending_bound() -> None:
    """Two OS processes share three queue positions, not three per controller."""
    ctx = multiprocessing.get_context("spawn")
    ready, results, start = ctx.Queue(), ctx.Queue(), ctx.Event()
    processes = []
    with AdmissionBroker(max_in_flight=1, max_pending=3, max_calls=4) as broker:
        controller = broker.controller()

        async def run() -> None:
            first = await controller.acquire(lambda _: None)
            try:
                for _ in range(2):
                    process = ctx.Process(
                        target=_contend_in_child,
                        args=(
                            broker.controller_config(queue_timeout=120),
                            ready,
                            start,
                            results,
                        ),
                    )
                    process.start()
                    processes.append(process)
                for _ in processes:
                    assert await asyncio.to_thread(ready.get, True, 90) == "ready"
                start.set()
                rejected = [await asyncio.to_thread(results.get, True, 30) for _ in range(17)]
                assert rejected == ["full"] * 17
                assert broker.snapshot().queued == 3
                assert broker.snapshot().admitted_calls == 1
                first.release()
                accepted = [await asyncio.to_thread(results.get, True, 30) for _ in range(3)]
                assert accepted == ["ok"] * 3
                for process in processes:
                    await asyncio.to_thread(process.join, 30)
                    assert process.exitcode == 0
                assert broker.snapshot().admitted_calls == 4
                assert broker.snapshot().peak_active == 1
                await _queued(broker, 0)
            finally:
                first.release()
                controller.close()

        try:
            asyncio.run(run())
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(5)
                process.close()
            for channel in (ready, results):
                channel.close()
                channel.join_thread()
