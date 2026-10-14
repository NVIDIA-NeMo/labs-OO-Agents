# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-socket and spawned-process tests for parent-broker admission."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import queue
import socket
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
    broker_admission,
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


def _ipv6_loopback_available() -> bool:
    """Return whether this host can bind a TCP socket on IPv6 loopback."""
    if not socket.has_ipv6:
        return False
    probe = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    try:
        probe.bind(("::1", 0))
    except OSError:
        return False
    finally:
        probe.close()
    return True


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


@pytest.mark.parametrize("host", ["127.0.0.1", "127.42.0.9", "::1"])
def test_broker_accepts_numeric_loopback_hosts(host: str):
    broker = AdmissionBroker(max_in_flight=1, host=host)
    config = BrokerAdmissionConfig(
        host=host,
        port=1,
        auth_token="test-token",  # noqa: S106 -- inert test credential
        group="loopback-test",
        max_in_flight=1,
        max_calls=None,
        queue_timeout=None,
    )

    assert broker.host == host
    assert config.host == host


@pytest.mark.skipif(not _ipv6_loopback_available(), reason="IPv6 loopback is unavailable")
@pytest.mark.asyncio
async def test_ipv6_loopback_broker_starts_and_serves_controller():
    observations: list[dict[str, Any]] = []
    with AdmissionBroker(max_in_flight=1, group="ipv6-test", host="::1") as broker:
        config = broker.controller_config(queue_timeout=1)
        controller = config.controller()
        permit = await asyncio.wait_for(controller.acquire(observations.append), timeout=2)
        permit.release()
        snapshot = await _wait_for_snapshot(broker, lambda current: current.active == 0)
        controller.close()

    assert config.host == "::1"
    assert observations[0]["outcome"] == "immediate"
    assert snapshot.admitted_calls == 1


@pytest.mark.parametrize(
    "host",
    ["0.0.0.0", "192.0.2.1", "::", "2001:db8::1", "localhost", "", None, 1234],
)
def test_broker_rejects_non_loopback_or_non_numeric_hosts(host: Any):
    with pytest.raises(ValueError, match="numeric loopback"):
        AdmissionBroker(max_in_flight=1, host=host)
    with pytest.raises(ValueError, match="numeric loopback"):
        BrokerAdmissionConfig(
            host=host,
            port=1,
            auth_token="test-token",  # noqa: S106 -- inert test credential
            group="loopback-test",
            max_in_flight=1,
            max_calls=None,
            queue_timeout=None,
        )


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
async def test_broker_queue_timeout_bounds_connection_establishment(monkeypatch):
    async def stalled_connection(*_args: Any, **_kwargs: Any):
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    monkeypatch.setattr(asyncio, "open_connection", stalled_connection)
    config = BrokerAdmissionConfig(
        host="127.0.0.1",
        port=443,
        auth_token="test-token",  # noqa: S106 -- inert test credential
        group="connect-timeout",
        max_in_flight=1,
        max_calls=None,
        queue_timeout=0.01,
    )
    observations: list[dict[str, Any]] = []

    with pytest.raises(AdmissionTimeoutError):
        await asyncio.wait_for(config.controller().acquire(observations.append), timeout=1)

    assert observations[0]["outcome"] == "timeout"


@pytest.mark.asyncio
async def test_broker_queue_timeout_bounds_handshake_writes_and_cleanup(monkeypatch):
    class StalledWriter:
        closed = False

        def write(self, _data: bytes) -> None:
            pass

        async def drain(self) -> None:
            await asyncio.Event().wait()

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            await asyncio.Event().wait()

    writer = StalledWriter()

    async def stalled_connection(*_args: Any, **_kwargs: Any):
        return asyncio.StreamReader(), writer

    monkeypatch.setattr(asyncio, "open_connection", stalled_connection)
    config = BrokerAdmissionConfig(
        host="127.0.0.1",
        port=443,
        auth_token="test-token",  # noqa: S106 -- inert test credential
        group="write-timeout",
        max_in_flight=1,
        max_calls=None,
        queue_timeout=0.01,
    )
    observations: list[dict[str, Any]] = []

    with pytest.raises(AdmissionTimeoutError):
        await asyncio.wait_for(config.controller().acquire(observations.append), timeout=1)

    assert writer.closed
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
async def test_terminal_observer_failures_do_not_replace_broker_timeout_or_cancellation():
    with AdmissionBroker(max_in_flight=1, group="terminal-observer") as broker:
        holder = await broker.controller().acquire(lambda _detail: None)

        def fail_observation(_detail: dict[str, Any]) -> None:
            raise RuntimeError("observer failed")

        with pytest.raises(AdmissionTimeoutError):
            await broker.controller(queue_timeout=0.01).acquire(fail_observation)

        cancelled = asyncio.create_task(broker.controller().acquire(fail_observation))
        snapshot = await _wait_for_snapshot(broker, lambda current: current.queued == 1)
        assert snapshot.queued == 1
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        holder.release()


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
async def test_unacknowledged_offer_does_not_consume_call_cap():
    with AdmissionBroker(max_in_flight=1, max_calls=1, group="offer-disconnect") as broker:
        config = broker.controller_config(queue_timeout=1)
        reader, writer = await asyncio.open_connection(config.host, config.port)
        writer.write(
            json.dumps(
                {
                    "version": broker_admission._PROTOCOL_VERSION,
                    "auth_token": config.auth_token,
                    "group": config.group,
                    "ticket": "disconnect-before-acquired",
                }
            ).encode()
            + b"\n"
        )
        await writer.drain()
        assert json.loads(await reader.readline())["status"] == "offered"
        writer.write(b'{"status":"accept"}\n')
        await writer.drain()
        assert json.loads(await reader.readline())["status"] == "ready"
        writer.close()
        await writer.wait_closed()

        snapshot = await _wait_for_snapshot(
            broker,
            lambda current: current.active == 0 and current.queued == 0,
        )
        assert snapshot.admitted_calls == 0

        permit = await broker.controller(queue_timeout=1).acquire(lambda _detail: None)
        permit.release()
        snapshot = await _wait_for_snapshot(broker, lambda current: current.active == 0)

    assert snapshot.admitted_calls == 1


def test_broker_close_retains_owner_state_when_server_close_fails(monkeypatch):
    broker = AdmissionBroker(max_in_flight=1, group="failed-close").start()
    server = broker._server
    assert server is not None
    real_close = server.close
    monkeypatch.setattr(server, "close", lambda: (_ for _ in ()).throw(RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        broker.close()

    assert broker._server is server
    assert broker.controller_config().auth_token

    monkeypatch.setattr(server, "close", real_close)
    broker.close()


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
        observations: list[dict[str, Any]] = []
        with pytest.raises(AdmissionUnavailableError, match="credentials"):
            await invalid.controller().acquire(observations.append)
        snapshot = broker.snapshot()

    assert snapshot.active == 0
    assert snapshot.queued == 0
    assert snapshot.admitted_calls == 0
    assert [item["outcome"] for item in observations] == ["unavailable"]


@pytest.mark.asyncio
async def test_broker_shutdown_fails_active_and_queued_clients_without_hanging():
    broker = AdmissionBroker(max_in_flight=1, group="owner-disappeared").start()
    controller = broker.controller(queue_timeout=5)
    holder = await controller.acquire(lambda _detail: None)
    observations: list[dict[str, Any]] = []
    waiting = asyncio.create_task(controller.acquire(observations.append))
    snapshot = await _wait_for_snapshot(broker, lambda current: current.queued == 1)
    assert snapshot.active == 1
    assert snapshot.queued == 1

    broker.close()
    with pytest.raises(AdmissionUnavailableError, match="closed before granting"):
        await asyncio.wait_for(waiting, timeout=1)
    holder.release()  # idempotent even though the owner already closed the lease
    assert [item["outcome"] for item in observations] == ["unavailable"]


@pytest.mark.asyncio
async def test_stale_connection_fails_closed_after_broker_shutdown():
    broker = AdmissionBroker(max_in_flight=1, group="stale-config").start()
    stale = broker.controller_config(queue_timeout=0.5)
    broker.close()
    observations: list[dict[str, Any]] = []

    with pytest.raises(AdmissionUnavailableError, match="unavailable"):
        await stale.controller().acquire(observations.append)

    assert [item["outcome"] for item in observations] == ["unavailable"]


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
async def test_broker_restart_rotates_credentials_at_the_same_address():
    broker = AdmissionBroker(max_in_flight=1, group="credential-rotation").start()
    stale = broker.controller_config(queue_timeout=1)
    broker.close()

    broker.port = stale.port
    broker.start()
    try:
        fresh = broker.controller_config(queue_timeout=1)
        observations: list[dict[str, Any]] = []
        with pytest.raises(AdmissionUnavailableError, match="credentials"):
            await stale.controller().acquire(observations.append)

        permit = await fresh.controller().acquire(lambda _detail: None)
        permit.release()
        snapshot = await _wait_for_snapshot(broker, lambda current: current.active == 0)
    finally:
        broker.close()

    assert stale.port == fresh.port
    assert stale.auth_token != fresh.auth_token
    assert observations[0]["outcome"] == "unavailable"
    assert snapshot.admitted_calls == 1


@pytest.mark.parametrize(
    "error_type",
    [RuntimeError, TimeoutError, OSError, asyncio.CancelledError],
)
@pytest.mark.asyncio
async def test_broker_observer_failure_is_not_translated_or_reobserved(
    error_type: type[BaseException],
):
    observations: list[dict[str, Any]] = []

    def fail_observation(detail: dict[str, Any]) -> None:
        observations.append(detail)
        raise error_type("observer failed")

    with AdmissionBroker(max_in_flight=1, group="observer-failure") as broker:
        controller = broker.controller(queue_timeout=1)
        with pytest.raises(error_type, match="observer failed"):
            await controller.acquire(fail_observation)

        snapshot = await _wait_for_snapshot(broker, lambda current: current.active == 0)
        probe = await asyncio.wait_for(controller.acquire(lambda _detail: None), timeout=1)
        probe.release()
        controller.close()

    assert snapshot.active == 0
    assert [item["outcome"] for item in observations] == ["immediate"]


@pytest.mark.asyncio
async def test_call_cap_observer_failure_is_not_reclassified():
    observations: list[dict[str, Any]] = []

    def fail_observation(detail: dict[str, Any]) -> None:
        observations.append(detail)
        raise OSError("observer failed")

    with AdmissionBroker(max_in_flight=1, max_calls=1, group="cap-observer-failure") as broker:
        controller = broker.controller(queue_timeout=1)
        permit = await controller.acquire(lambda _detail: None)
        permit.release()

        with pytest.raises(OSError, match="observer failed"):
            await controller.acquire(fail_observation)
        snapshot = await _wait_for_snapshot(broker, lambda current: current.active == 0)
        controller.close()

    assert snapshot.admitted_calls == 1
    assert [item["outcome"] for item in observations] == ["call_cap"]


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
