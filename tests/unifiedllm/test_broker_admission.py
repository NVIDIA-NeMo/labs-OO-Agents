# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-socket and spawned-process tests for parent-broker admission."""

from __future__ import annotations

import asyncio
import multiprocessing
import os
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest

from nooa.unifiedllm import (
    AdmissionBroker,
    AdmissionCallCapError,
    AdmissionTimeoutError,
    AdmissionUnavailableError,
    BrokerAdmissionConfig,
)


async def _wait_for_snapshot(
    broker: AdmissionBroker,
    predicate: Callable[[Any], bool],
    *,
    timeout: float = 2,
) -> Any:
    deadline = time.monotonic() + timeout
    snapshot = broker.snapshot()
    while not predicate(snapshot) and time.monotonic() < deadline:
        await asyncio.sleep(0.005)
        snapshot = broker.snapshot()
    return snapshot


def _join_spawned_processes(processes: list[Any], *, timeout: float = 60) -> None:
    """Join slow-importing spawn children and always clean up on failure."""
    deadline = time.monotonic() + timeout
    try:
        for process in processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
        stalled = [process.pid for process in processes if process.is_alive()]
        failures = [
            (process.pid, process.exitcode)
            for process in processes
            if not process.is_alive() and process.exitcode != 0
        ]
        assert not stalled, f"spawned processes did not exit within {timeout}s: {stalled}"
        assert not failures, f"spawned processes exited unsuccessfully: {failures}"
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)


def _run_calls_in_child(
    config: BrokerAdmissionConfig,
    calls: int,
    start: Any,
    results: Any,
) -> None:
    async def run() -> tuple[int, int, int]:
        controller = config.controller()

        async def one_call() -> str:
            try:
                permit = await controller.acquire(lambda _detail: None)
            except AdmissionCallCapError:
                return "capped"
            except Exception:
                return "error"
            try:
                await asyncio.sleep(0.01)
                return "ok"
            finally:
                permit.release()

        outcomes = await asyncio.gather(*(one_call() for _ in range(calls)))
        return outcomes.count("ok"), outcomes.count("capped"), outcomes.count("error")

    start.wait()
    results.put(asyncio.run(run()))


def _acquire_and_exit(config: BrokerAdmissionConfig, ready: Any) -> None:
    async def run() -> None:
        permit = await config.controller().acquire(lambda _detail: None)
        del permit
        ready.set()
        os._exit(23)

    asyncio.run(run())


@pytest.mark.parametrize("value", [0, -1, True])
def test_broker_rejects_invalid_concurrency_limit(value: Any):
    with pytest.raises((TypeError, ValueError)):
        AdmissionBroker(max_in_flight=value)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_broker_rejects_invalid_call_cap(value: Any):
    with pytest.raises(ValueError, match="max_calls"):
        AdmissionBroker(max_in_flight=1, max_calls=value)


@pytest.mark.asyncio
async def test_broker_queues_and_admits_in_fifo_order():
    with AdmissionBroker(max_in_flight=1, group="fifo-test") as broker:
        controller = broker.controller(queue_timeout=1)
        first = await controller.acquire(lambda _detail: None)
        order: list[int] = []
        observations: list[dict[str, Any]] = []

        async def waiter(index: int) -> None:
            permit = await controller.acquire(observations.append)
            order.append(index)
            permit.release()

        waiters = []
        for index in range(4):
            waiters.append(asyncio.create_task(waiter(index)))
            snapshot = await _wait_for_snapshot(
                broker,
                lambda current, expected=index + 1: current.queued == expected,
            )
            assert snapshot.queued == index + 1

        first.release()
        await asyncio.wait_for(asyncio.gather(*waiters), timeout=2)
        snapshot = await _wait_for_snapshot(
            broker, lambda current: current.active == 0 and current.queued == 0
        )

    assert order == [0, 1, 2, 3]
    assert snapshot.active == 0
    assert snapshot.admitted_calls == 5
    assert all(item["outcome"] == "admitted_after_wait" for item in observations)


@pytest.mark.asyncio
async def test_broker_queue_timeout_removes_waiter_before_provider_dispatch():
    with AdmissionBroker(max_in_flight=1, group="timeout-test") as broker:
        holder = await broker.controller().acquire(lambda _detail: None)
        observations: list[dict[str, Any]] = []
        with pytest.raises(AdmissionTimeoutError):
            await broker.controller(queue_timeout=0.05).acquire(observations.append)

        snapshot = await _wait_for_snapshot(broker, lambda current: current.queued == 0)
        holder.release()

    assert snapshot.queued == 0
    assert observations[0]["outcome"] == "timeout"


@pytest.mark.asyncio
async def test_broker_queued_cancellation_removes_waiter_and_preserves_capacity():
    with AdmissionBroker(max_in_flight=1, group="cancel-test") as broker:
        controller = broker.controller()
        holder = await controller.acquire(lambda _detail: None)
        observations: list[dict[str, Any]] = []
        waiting = asyncio.create_task(controller.acquire(observations.append))
        snapshot = await _wait_for_snapshot(broker, lambda current: current.queued == 1)
        assert snapshot.queued == 1

        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        snapshot = await _wait_for_snapshot(broker, lambda current: current.queued == 0)
        assert snapshot.queued == 0

        holder.release()
        probe = await asyncio.wait_for(controller.acquire(lambda _detail: None), timeout=1)
        probe.release()

    assert observations[0]["outcome"] == "cancelled"


@pytest.mark.asyncio
async def test_broker_call_cap_rejects_before_dispatch():
    with AdmissionBroker(max_in_flight=2, max_calls=3, group="cap-test") as broker:
        controller = broker.controller()
        for _ in range(3):
            permit = await controller.acquire(lambda _detail: None)
            permit.release()

        observations: list[dict[str, Any]] = []
        with pytest.raises(AdmissionCallCapError, match="max_calls=3"):
            await controller.acquire(observations.append)

        snapshot = await _wait_for_snapshot(
            broker, lambda current: current.active == 0 and current.queued == 0
        )

    assert snapshot.admitted_calls == 3
    assert observations[0]["outcome"] == "call_cap"


@pytest.mark.asyncio
async def test_controller_reuses_a_connection_for_sequential_attempts(monkeypatch):
    opened = 0
    real_open_connection = asyncio.open_connection

    async def counted_open_connection(*args: Any, **kwargs: Any):
        nonlocal opened
        opened += 1
        return await real_open_connection(*args, **kwargs)

    monkeypatch.setattr(asyncio, "open_connection", counted_open_connection)
    with AdmissionBroker(max_in_flight=4, group="connection-reuse") as broker:
        controller = broker.controller(queue_timeout=2)
        for _ in range(500):
            permit = await controller.acquire(lambda _detail: None)
            permit.release()
        snapshot = await _wait_for_snapshot(
            broker, lambda current: current.active == 0 and current.queued == 0
        )
        controller.close()

    assert opened == 1
    assert snapshot.admitted_calls == 500


@pytest.mark.asyncio
async def test_invalid_external_capability_is_rejected_without_consuming_capacity():
    with AdmissionBroker(max_in_flight=2, group="credential-test") as broker:
        invalid = replace(
            broker.controller_config(queue_timeout=1),
            auth_token="not-the-broker-token",  # noqa: S106 -- deliberately invalid test token
        )
        with pytest.raises(AdmissionUnavailableError, match="credentials"):
            await invalid.controller().acquire(lambda _detail: None)
        snapshot = broker.snapshot()

    assert snapshot.active == 0
    assert snapshot.queued == 0
    assert snapshot.admitted_calls == 0


@pytest.mark.asyncio
async def test_broker_shutdown_fails_active_and_queued_clients_without_hanging():
    broker = AdmissionBroker(max_in_flight=1, group="owner-disappeared").start()
    controller = broker.controller(queue_timeout=5)
    holder = await controller.acquire(lambda _detail: None)
    waiting = asyncio.create_task(controller.acquire(lambda _detail: None))
    snapshot = await _wait_for_snapshot(broker, lambda current: current.queued == 1)
    assert snapshot.active == 1
    assert snapshot.queued == 1

    broker.close()
    with pytest.raises(AdmissionUnavailableError, match="closed before granting"):
        await asyncio.wait_for(waiting, timeout=1)
    holder.release()  # idempotent even though the owner already closed the lease


@pytest.mark.asyncio
async def test_stale_connection_fails_closed_after_broker_shutdown():
    broker = AdmissionBroker(max_in_flight=1, group="stale-config").start()
    stale = broker.controller_config(queue_timeout=0.5)
    broker.close()

    with pytest.raises(AdmissionUnavailableError, match="unavailable"):
        await stale.controller().acquire(lambda _detail: None)


@pytest.mark.asyncio
async def test_broker_owner_can_restart_and_publish_a_fresh_connection():
    broker = AdmissionBroker(max_in_flight=1, group="owner-restart")
    broker.start()
    first = await broker.controller(queue_timeout=1).acquire(lambda _detail: None)
    first.release()
    broker.close()

    broker.start()
    try:
        second = await broker.controller(queue_timeout=1).acquire(lambda _detail: None)
        second.release()
        snapshot = await _wait_for_snapshot(
            broker, lambda current: current.active == 0 and current.queued == 0
        )
    finally:
        broker.close()

    assert snapshot.admitted_calls == 1
    assert snapshot.peak_active == 1


@pytest.mark.asyncio
async def test_large_waiter_burst_uses_bounded_threads():
    baseline_threads = threading.active_count()
    peak_threads = baseline_threads

    with AdmissionBroker(max_in_flight=8, group="large-burst") as broker:
        config = broker.controller_config(queue_timeout=10)

        async def attempt() -> None:
            nonlocal peak_threads
            permit = await config.controller().acquire(lambda _detail: None)
            try:
                peak_threads = max(peak_threads, threading.active_count())
                await asyncio.sleep(0.005)
            finally:
                permit.release()

        await asyncio.wait_for(
            asyncio.gather(*(attempt() for _ in range(300))),
            timeout=15,
        )
        snapshot = await _wait_for_snapshot(
            broker,
            lambda current: current.active == 0 and current.queued == 0,
        )

    assert peak_threads <= baseline_threads + 2
    assert snapshot.peak_active == 8
    assert snapshot.admitted_calls == 300


def test_spawned_processes_share_one_concurrency_ceiling():
    ctx = multiprocessing.get_context("spawn")
    with AdmissionBroker(max_in_flight=4, group="spawn-concurrency") as broker:
        config = broker.controller_config(queue_timeout=10)
        start = ctx.Event()
        results = ctx.Queue()
        processes = [
            ctx.Process(target=_run_calls_in_child, args=(config, 12, start, results))
            for _ in range(4)
        ]
        for process in processes:
            process.start()
        start.set()
        _join_spawned_processes(processes)

        outcomes = [results.get(timeout=2) for _ in processes]
        deadline = time.monotonic() + 2
        snapshot = broker.snapshot()
        while (snapshot.active or snapshot.queued) and time.monotonic() < deadline:
            time.sleep(0.005)
            snapshot = broker.snapshot()

    assert sum(result[0] for result in outcomes) == 48
    assert sum(result[1] for result in outcomes) == 0
    assert sum(result[2] for result in outcomes) == 0
    assert snapshot.peak_active == 4
    assert snapshot.admitted_calls == 48


def test_spawned_processes_share_one_exact_call_cap():
    ctx = multiprocessing.get_context("spawn")
    with AdmissionBroker(max_in_flight=4, max_calls=19, group="spawn-cap") as broker:
        config = broker.controller_config(queue_timeout=10)
        start = ctx.Event()
        results = ctx.Queue()
        processes = [
            ctx.Process(target=_run_calls_in_child, args=(config, 8, start, results))
            for _ in range(4)
        ]
        for process in processes:
            process.start()
        start.set()
        _join_spawned_processes(processes)

        outcomes: list[tuple[int, int, int]] = []
        for _ in processes:
            try:
                outcomes.append(results.get(timeout=2))
            except queue.Empty:
                pytest.fail("child did not publish its admission results")
        snapshot = broker.snapshot()

    assert sum(result[0] for result in outcomes) == 19
    assert sum(result[1] for result in outcomes) == 13
    assert sum(result[2] for result in outcomes) == 0
    assert snapshot.admitted_calls == 19


def test_connection_lease_recovers_after_abrupt_child_exit():
    ctx = multiprocessing.get_context("spawn")
    with AdmissionBroker(max_in_flight=1, group="crash-recovery") as broker:
        ready = ctx.Event()
        child = ctx.Process(
            target=_acquire_and_exit,
            args=(broker.controller_config(queue_timeout=5), ready),
        )
        child.start()
        try:
            assert ready.wait(timeout=30)
            child.join(timeout=30)
            assert child.exitcode == 23
        finally:
            if child.is_alive():
                child.terminate()
            child.join(timeout=5)

        deadline = time.monotonic() + 2
        snapshot = broker.snapshot()
        while snapshot.active and time.monotonic() < deadline:
            time.sleep(0.005)
            snapshot = broker.snapshot()
        assert snapshot.active == 0

        async def probe() -> None:
            permit = await broker.controller(queue_timeout=1).acquire(lambda _detail: None)
            permit.release()

        asyncio.run(probe())
