# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Socket, process and lifecycle checks for shared broker pacing and cooldown."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
import os
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from typing import Any

import pytest

from nooa.unifiedllm import (
    AdmissionBroker,
    AdmissionCallCapError,
    AdmissionTimeoutError,
    AdmissionUnavailableError,
    BrokerAdmissionConfig,
    BrokerAdmissionController,
    broker_admission,
)


@pytest.fixture(autouse=True)
async def _close_test_controllers(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[None]:
    """Close convenience clients before pytest shuts their short-lived event loop."""
    controllers: list[BrokerAdmissionController] = []
    original = BrokerAdmissionConfig.controller

    def tracked(config: BrokerAdmissionConfig) -> BrokerAdmissionController:
        """Keep ownership even when a test only retains the acquired permit."""
        controller = original(config)
        controllers.append(controller)
        return controller

    monkeypatch.setattr(BrokerAdmissionConfig, "controller", tracked)
    try:
        yield
    finally:
        for controller in controllers:
            controller.close()
        # close() intentionally stays synchronous. Allow transport-close callbacks to
        # complete before pytest destroys the loop instead of waiting for idle expiry.
        await asyncio.sleep(0)
        await asyncio.sleep(0)


async def _snapshot_when(
    broker: AdmissionBroker,
    predicate: Callable[[Any], bool],
) -> Any:
    """Wait briefly for a real broker event, accounting for its server thread."""
    deadline = time.monotonic() + 2
    snapshot = broker.snapshot()
    while not predicate(snapshot) and time.monotonic() < deadline:
        await asyncio.sleep(0.002)
        snapshot = broker.snapshot()
    assert predicate(snapshot), snapshot
    return snapshot


def _paced_child(config: BrokerAdmissionConfig, start: Any, results: Any) -> None:
    """Send acquisition times from independent spawned event loops."""

    async def run() -> list[float]:
        """Acquire and release a few permits using one child's controller."""
        controller = config.controller()
        times = []
        try:
            for _ in range(3):
                permit = await controller.acquire(lambda _detail: None)
                times.append(time.monotonic())
                permit.release()
            return times
        finally:
            controller.close()

    start.wait(timeout=30)
    results.put(asyncio.run(run()))


def _paced_crash(config: BrokerAdmissionConfig, ready: Any) -> None:
    """Abruptly exit while holding a paced lease, relying on socket EOF."""

    async def run() -> None:
        """Signal successful acquisition before immediately ending the process."""
        await config.controller().acquire(lambda _detail: None)
        ready.set()
        os._exit(23)

    asyncio.run(run())


@pytest.mark.parametrize("option", ["requests_per_second", "max_cooldown"])
@pytest.mark.parametrize("value", [0, -1, True, "1", float("nan"), float("inf")])
def test_broker_validates_new_policy_options(option: str, value: Any):
    """Reject invalid settings both at ownership and serialized-client boundaries."""
    with pytest.raises((TypeError, ValueError), match=option):
        AdmissionBroker(max_in_flight=2, **{option: value})
    config = BrokerAdmissionConfig("127.0.0.1", 1, "token", "policy", 2, None, None)
    with pytest.raises((TypeError, ValueError), match=option):
        replace(config, **{option: value})


def test_broker_preserves_old_positional_config_and_propagates_policy():
    """Keep old serialized construction valid and transfer owner settings unchanged."""
    config = BrokerAdmissionConfig("127.0.0.1", 1, "token", "policy", 2, None, None)
    assert config.requests_per_second is None
    assert config.max_cooldown is None
    with AdmissionBroker(max_in_flight=2, requests_per_second=20, max_cooldown=0.25) as broker:
        configured = broker.controller_config()
        assert configured.requests_per_second == 20
        assert configured.max_cooldown == 0.25
        assert configured.controller().max_cooldown == 0.25


def test_broker_rejects_nonfinite_interval_from_tiny_positive_rate():
    """Finite positive rates still need a representable monotonic pacing interval."""
    with pytest.raises(ValueError, match="requests_per_second"):
        AdmissionBroker(max_in_flight=2, requests_per_second=5e-324)
    config = BrokerAdmissionConfig("127.0.0.1", 1, "token", "policy", 2, None, None)
    with pytest.raises(ValueError, match="requests_per_second"):
        replace(config, requests_per_second=5e-324)


@pytest.mark.asyncio
async def test_broker_paces_fifo_without_reserving_concurrency_for_waiters():
    """One FIFO gate separates starts even when several concurrency slots are free."""
    with AdmissionBroker(max_in_flight=4, requests_per_second=20) as broker:
        controller = broker.controller(queue_timeout=2)
        first = await controller.acquire(lambda _detail: None)
        started = time.monotonic()
        first.release()
        await _snapshot_when(broker, lambda snapshot: snapshot.active == 0)
        order = []
        times = [started]
        observations = []

        async def attempt(index: int) -> None:
            """Record one provider-ready acquisition and return the lease."""
            permit = await controller.acquire(observations.append)
            order.append(index)
            times.append(time.monotonic())
            permit.release()

        tasks = []
        for index in range(3):
            tasks.append(asyncio.create_task(attempt(index)))
            snapshot = await _snapshot_when(
                broker, lambda current, depth=index + 1: current.queued == depth
            )
            assert snapshot.active == 0
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)
        snapshot = await _snapshot_when(broker, lambda current: current.active == 0)
        controller.close()

    assert order == [0, 1, 2]
    assert all(right - left >= 0.045 for left, right in zip(times, times[1:], strict=False))
    assert snapshot.admitted_calls == 4
    assert all(detail["outcome"] == "admitted_after_wait" for detail in observations)


@pytest.mark.asyncio
async def test_broker_pacing_keeps_independent_concurrency_limit():
    """Paced starts can overlap while active provider work stays within the ceiling."""
    with AdmissionBroker(max_in_flight=2, requests_per_second=40) as broker:
        controller = broker.controller(queue_timeout=2)
        times = []

        async def attempt() -> None:
            """Hold a real lease long enough to make the independent slot limit visible."""
            permit = await controller.acquire(lambda _detail: None)
            times.append(time.monotonic())
            try:
                await asyncio.sleep(0.1)
            finally:
                permit.release()

        await asyncio.wait_for(asyncio.gather(*(attempt() for _ in range(4))), timeout=2)
        snapshot = await _snapshot_when(broker, lambda current: current.active == 0)
        controller.close()

    assert snapshot.peak_active == 2
    assert snapshot.admitted_calls == 4
    assert all(right - left >= 0.02 for left, right in zip(times, times[1:], strict=False))
    assert times[2] - times[0] >= 0.095


@pytest.mark.asyncio
async def test_broker_cooldown_is_shared_and_shorter_reports_do_not_shorten_it():
    """All clients obey the longest outstanding report from a still-held lease."""
    with AdmissionBroker(max_in_flight=2, max_cooldown=0.3) as broker:
        first = await broker.controller().acquire(lambda _detail: None)
        second = await broker.controller().acquire(lambda _detail: None)
        await first.cooldown(0.15)
        reported = time.monotonic()
        first.release()
        await asyncio.sleep(0.025)
        await second.cooldown(0.01)
        second.release()
        await _snapshot_when(broker, lambda current: current.active == 0)
        permit = await asyncio.wait_for(
            broker.controller().acquire(lambda _detail: None), timeout=1
        )
        elapsed = time.monotonic() - reported
        permit.release()

    assert elapsed >= 0.14


@pytest.mark.asyncio
async def test_broker_cooldown_extends_a_scheduled_wakeup():
    """A later report reschedules the queued head instead of admitting at the old time."""
    with AdmissionBroker(max_in_flight=2, max_cooldown=0.3) as broker:
        reporter = await broker.controller().acquire(lambda _detail: None)
        await reporter.cooldown(0.05)
        waiting = asyncio.create_task(broker.controller().acquire(lambda _detail: None))
        await _snapshot_when(broker, lambda current: current.queued == 1)
        await asyncio.sleep(0.015)
        await reporter.cooldown(0.12)
        extended = time.monotonic()
        reporter.release()
        permit = await asyncio.wait_for(waiting, timeout=1)
        elapsed = time.monotonic() - extended
        permit.release()

    assert elapsed >= 0.11


@pytest.mark.asyncio
async def test_broker_caps_cooldown_before_acknowledging_it():
    """Untrusted retry guidance cannot pause this configured group indefinitely."""
    with AdmissionBroker(max_in_flight=2, max_cooldown=0.05) as broker:
        controller = broker.controller()
        permit = await controller.acquire(lambda _detail: None)
        await permit.cooldown(10_000)
        reported = time.monotonic()
        permit.release()
        probe = await asyncio.wait_for(controller.acquire(lambda _detail: None), timeout=0.3)
        elapsed = time.monotonic() - reported
        probe.release()
        controller.close()

    assert 0.04 <= elapsed < 0.3


@pytest.mark.asyncio
async def test_broker_default_cooldown_is_noop_and_released_feedback_is_noop():
    """Disabled and already-returned permits do not change another client's admission."""
    with AdmissionBroker(max_in_flight=1) as broker:
        controller = broker.controller(queue_timeout=0.1)
        permit = await controller.acquire(lambda _detail: None)
        await permit.cooldown(5)
        permit.release()
        await permit.cooldown(5)
        probe = await controller.acquire(lambda _detail: None)
        probe.release()
        controller.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("delay", [0, -1, True, "1", None, float("nan"), float("inf")])
async def test_broker_validates_cooldown_even_when_disabled(delay: Any):
    """Validate the extension contract without leaking the permit on invalid input."""
    with AdmissionBroker(max_in_flight=1) as broker:
        permit = await broker.controller().acquire(lambda _detail: None)
        try:
            with pytest.raises((TypeError, ValueError), match="delay_s"):
                await permit.cooldown(delay)
        finally:
            permit.release()


@pytest.mark.asyncio
async def test_broker_pacing_wait_timeout_and_cancellation_do_not_dispatch():
    """Removing paced waiters leaves the call budget untouched and no active slots."""
    with AdmissionBroker(max_in_flight=3, requests_per_second=4) as broker:
        permit = await broker.controller().acquire(lambda _detail: None)
        permit.release()
        with pytest.raises(AdmissionTimeoutError):
            await broker.controller(queue_timeout=0.015).acquire(lambda _detail: None)
        waiting = asyncio.create_task(broker.controller().acquire(lambda _detail: None))
        await _snapshot_when(broker, lambda current: current.queued == 1)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        snapshot = await _snapshot_when(
            broker, lambda current: current.active == 0 and current.queued == 0
        )

    assert snapshot.admitted_calls == 1


@pytest.mark.asyncio
async def test_broker_call_cap_rejects_without_waiting_for_pacing_or_cooldown():
    """Terminal budget exhaustion cannot unnecessarily occupy a pacing queue."""
    with AdmissionBroker(
        max_in_flight=2, max_calls=1, requests_per_second=1, max_cooldown=1
    ) as broker:
        permit = await broker.controller().acquire(lambda _detail: None)
        await permit.cooldown(1)
        permit.release()
        with pytest.raises(AdmissionCallCapError):
            await asyncio.wait_for(broker.controller().acquire(lambda _detail: None), timeout=0.1)
        snapshot = await _snapshot_when(broker, lambda current: current.queued == 0)

    assert snapshot.admitted_calls == 1


@pytest.mark.asyncio
async def test_broker_old_protocol_fails_closed_before_admission():
    """A client predating feedback cannot accidentally use unsupported wire commands."""
    with AdmissionBroker(max_in_flight=1, max_cooldown=1) as broker:
        config = broker.controller_config()
        reader, writer = await asyncio.open_connection(config.host, config.port)
        writer.write(
            json.dumps(
                {
                    "version": broker_admission._PROTOCOL_VERSION - 1,
                    "auth_token": config.auth_token,
                    "group": config.group,
                    "ticket": "old-client",
                }
            ).encode()
            + b"\n"
        )
        await writer.drain()
        assert json.loads(await reader.readline())["status"] == "unauthorized"
        writer.close()
        await writer.wait_closed()
        snapshot = broker.snapshot()

    assert snapshot.admitted_calls == 0
    assert snapshot.active == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("delay", [0, -1, True, "1", None, float("nan")])
async def test_malformed_feedback_reclaims_lease_without_server_error(
    delay: Any, caplog: pytest.LogCaptureFixture
):
    """Authenticated invalid feedback is rejected rather than leaking its socket lease."""
    with AdmissionBroker(max_in_flight=1, max_cooldown=0.1) as broker:
        controller = broker.controller()
        permit = await controller.acquire(lambda _detail: None)
        permit._writer.write(json.dumps({"status": "cooldown", "delay_s": delay}).encode() + b"\n")
        await permit._writer.drain()
        assert await asyncio.wait_for(permit._reader.readline(), timeout=1) == b""
        permit.release()
        snapshot = await _snapshot_when(broker, lambda current: current.active == 0)
        controller.close()
    assert snapshot.admitted_calls == 1
    assert "Unhandled exception" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "unexpected", "cancel", "release"])
async def test_failed_feedback_closes_socket_and_never_reuses_unread_ack(
    monkeypatch: pytest.MonkeyPatch, failure: str
):
    """Dirty feedback connections must return capacity without entering the idle pool."""
    original = broker_admission._AdmissionBrokerServer._write
    reached = asyncio.Event()
    loop = asyncio.get_running_loop()

    async def fail_ack(writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
        """Inject failure only after the server has processed a cooldown command."""
        if payload["status"] == "cooldown_applied":
            loop.call_soon_threadsafe(reached.set)
            if failure == "unexpected":
                await original(writer, {"status": "wrong_ack"})
        else:
            await original(writer, payload)

    monkeypatch.setattr(broker_admission._AdmissionBrokerServer, "_write", staticmethod(fail_ack))
    monkeypatch.setattr(broker_admission, "_FEEDBACK_TIMEOUT_SECONDS", 0.03)
    with AdmissionBroker(max_in_flight=1, max_cooldown=0.1) as broker:
        controller = broker.controller()
        permit = await controller.acquire(lambda _detail: None)
        feedback = asyncio.create_task(permit.cooldown(0.01))
        await asyncio.wait_for(reached.wait(), timeout=1)
        if failure == "cancel":
            feedback.cancel()
            with pytest.raises(asyncio.CancelledError):
                await feedback
        else:
            if failure == "release":
                permit.release()
            with pytest.raises(AdmissionUnavailableError):
                await feedback
        permit.release()
        assert not controller._idle
        snapshot = await _snapshot_when(broker, lambda current: current.active == 0)
        controller.close()

    assert snapshot.admitted_calls == 1


@pytest.mark.asyncio
async def test_foreign_loop_feedback_does_not_contend_with_owner_feedback(
    monkeypatch: pytest.MonkeyPatch,
):
    """Reject another event loop before it can attach a waiter to the protocol lock."""
    original = broker_admission._AdmissionBrokerServer._write
    reached = asyncio.Event()
    loop = asyncio.get_running_loop()

    async def delayed_ack(writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
        """Keep valid feedback active long enough for a concurrent foreign call."""
        if payload["status"] == "cooldown_applied":
            loop.call_soon_threadsafe(reached.set)
            await asyncio.sleep(0.1)
        await original(writer, payload)

    monkeypatch.setattr(
        broker_admission._AdmissionBrokerServer, "_write", staticmethod(delayed_ack)
    )
    with AdmissionBroker(max_in_flight=1, max_cooldown=0.05) as broker:
        controller = broker.controller()
        permit = await controller.acquire(lambda _detail: None)
        feedback = asyncio.create_task(permit.cooldown(0.01))
        await asyncio.wait_for(reached.wait(), timeout=1)

        def foreign_feedback() -> None:
            """Call from a new loop while the permit's valid feedback owns the reader."""
            with pytest.raises(AdmissionUnavailableError, match="permit's event loop"):
                asyncio.run(permit.cooldown(0.01))

        await asyncio.wait_for(asyncio.to_thread(foreign_feedback), timeout=0.5)
        await asyncio.wait_for(feedback, timeout=1)
        permit.release()
        probe = await asyncio.wait_for(controller.acquire(lambda _detail: None), timeout=1)
        probe.release()
        controller.close()


@pytest.mark.asyncio
async def test_broker_pacing_timer_disappears_when_queue_is_empty_or_shutdown():
    """Cancelling the final paced waiter and closing the state retain no timer."""
    state = broker_admission._BrokerState(
        group="timers", max_in_flight=2, max_calls=None, requests_per_second=1
    )
    first = state.enqueue("first")
    state.confirm(first)
    state.disconnect(first)
    second = state.enqueue("second")
    assert state._wake is not None
    state.disconnect(second)
    assert state._wake is None
    third = state.enqueue("third")
    assert state._wake is not None
    state.close()
    assert state._wake is None
    state.disconnect(third)
    assert state.active == 0


@pytest.mark.asyncio
async def test_abandoned_pacing_offer_does_not_refund_start_spacing():
    """Even an unconfirmed offer burns its interval before its successor can start."""
    state = broker_admission._BrokerState(
        group="offers", max_in_flight=3, max_calls=None, requests_per_second=20
    )
    abandoned = state.enqueue("abandoned")
    offered = time.monotonic()
    state.disconnect(abandoned)
    next_waiter = state.enqueue("next")
    await asyncio.wait_for(next_waiter.future, timeout=0.3)
    elapsed = time.monotonic() - offered
    state.confirm(next_waiter)
    state.disconnect(next_waiter)
    state.close()

    assert elapsed >= 0.045
    assert state.admitted_calls == 1


@pytest.mark.asyncio
async def test_delayed_handshake_cannot_accumulate_burst_of_paced_offers():
    """Space the next offer from confirmation even when its predecessor was slow."""
    state = broker_admission._BrokerState(
        group="handshakes", max_in_flight=3, max_calls=None, requests_per_second=20
    )
    slow = state.enqueue("slow")
    next_waiter = state.enqueue("next")
    await asyncio.sleep(0.06)
    assert not next_waiter.future.done()
    assert state.active == 1
    state.confirm(slow)
    confirmed = time.monotonic()
    await asyncio.wait_for(next_waiter.future, timeout=0.3)
    elapsed = time.monotonic() - confirmed
    state.confirm(next_waiter)
    state.disconnect(next_waiter)
    state.disconnect(slow)
    state.close()

    assert elapsed >= 0.045
    assert state.admitted_calls == 2


@pytest.mark.asyncio
async def test_cooldown_preserves_existing_offer_but_delays_new_work():
    """Feedback gates future offers without revoking a lease already being accepted."""
    state = broker_admission._BrokerState(
        group="grace", max_in_flight=3, max_calls=None, max_cooldown=0.1
    )
    reporting = state.enqueue("reporting")
    state.confirm(reporting)
    already_offered = state.enqueue("offered-before-feedback")
    assert already_offered.future.done()
    state.cooldown(0.08)
    reported = time.monotonic()
    waiting = state.enqueue("queued-after-feedback")
    assert not waiting.future.done()
    state.confirm(already_offered)
    state.disconnect(reporting)
    state.disconnect(already_offered)
    assert state.active == 0
    assert not waiting.future.done()
    await asyncio.wait_for(waiting.future, timeout=0.3)
    elapsed = time.monotonic() - reported
    state.confirm(waiting)
    state.disconnect(waiting)
    state.close()

    assert elapsed >= 0.07
    assert state.admitted_calls == 3


def test_broker_pacing_is_shared_across_spawned_processes():
    """Independent clients cannot multiply the owner-configured admission rate."""
    spawn = multiprocessing.get_context("spawn")
    start = spawn.Event()
    results = spawn.Queue()
    processes = []
    try:
        with AdmissionBroker(max_in_flight=4, requests_per_second=20) as broker:
            config = broker.controller_config(queue_timeout=10)
            processes = [
                spawn.Process(target=_paced_child, args=(config, start, results)) for _ in range(3)
            ]
            for process in processes:
                process.start()
            start.set()
            times = sorted(value for _ in processes for value in results.get(timeout=60))
            for process in processes:
                process.join(timeout=5)
                assert process.exitcode == 0
            snapshot = broker.snapshot()
        assert len(times) == 9
        assert all(right - left >= 0.045 for left, right in zip(times, times[1:], strict=False))
        assert snapshot.admitted_calls == 9
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
        results.close()
        results.join_thread()


@pytest.mark.asyncio
async def test_paced_child_crash_returns_capacity_without_refunding_spacing():
    """A process exit releases its socket lease but preserves the last start's rate debt."""
    spawn = multiprocessing.get_context("spawn")
    ready = spawn.Event()
    process = None
    try:
        with AdmissionBroker(max_in_flight=1, requests_per_second=2) as broker:
            process = spawn.Process(target=_paced_crash, args=(broker.controller_config(), ready))
            process.start()
            assert await asyncio.to_thread(ready.wait, 60)
            crashed = time.monotonic()
            process.join(timeout=5)
            assert process.exitcode == 23
            await _snapshot_when(broker, lambda current: current.active == 0)
            permit = await asyncio.wait_for(
                broker.controller().acquire(lambda _detail: None), timeout=2
            )
            elapsed = time.monotonic() - crashed
            permit.release()
        assert elapsed >= 0.4
    finally:
        if process is not None:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)


@pytest.mark.asyncio
async def test_broker_paced_start_stop_cycles_release_queued_connections():
    """Repeated shutdown with a long pacing debt has bounded teardown and clean restarts."""
    broker = AdmissionBroker(max_in_flight=2, requests_per_second=0.1)
    for _ in range(5):
        broker.start()
        controller = broker.controller()
        permit = await controller.acquire(lambda _detail: None)
        permit.release()
        waiting = asyncio.create_task(controller.acquire(lambda _detail: None))
        await _snapshot_when(broker, lambda current: current.queued == 1)
        broker.close()
        with pytest.raises(AdmissionUnavailableError):
            await asyncio.wait_for(waiting, timeout=1)
        controller.close()
