# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional broker deadlines bound unconfirmed exchanges, never provider leases."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from typing import Any

import pytest

from nooa.unifiedllm import (
    AdmissionBroker,
    AdmissionCallCapError,
    AdmissionTimeoutError,
    AdmissionUnavailableError,
)
from nooa.unifiedllm import (
    broker_admission as module,
)


async def _until(broker: AdmissionBroker, predicate: Callable[[Any], bool]) -> Any:
    """Synchronize on broker state, with a bound on failed test progress."""
    async with asyncio.timeout(5):
        while True:
            snapshot = broker.snapshot()
            if predicate(snapshot):
                return snapshot
            await asyncio.sleep(0.002)


@asynccontextmanager
async def _owner(**settings: Any) -> AsyncIterator[tuple[AdmissionBroker, Any]]:
    """Close controller transports while their loop and owner are still alive."""
    broker = AdmissionBroker(**settings).start()
    controller = broker.controller(queue_timeout=3)
    try:
        yield broker, controller
    finally:
        controller.close()
        await asyncio.sleep(0.02)
        broker.close()


@asynccontextmanager
async def _stalled(
    broker: AdmissionBroker, phase: str, *, malformed: bytes | None = None
) -> AsyncIterator[asyncio.StreamReader]:
    """Leave a real authenticated connection open before its final acknowledgement."""
    config = broker.controller_config()
    reader, writer = await asyncio.open_connection(config.host, config.port)
    try:
        writer.write(
            json.dumps(
                {
                    "version": module._PROTOCOL_VERSION,
                    "auth_token": config.auth_token,
                    "group": config.group,
                    "ticket": "synthetic-stalled-worker",
                }
            ).encode()
            + b"\n"
        )
        await writer.drain()
        assert json.loads(await reader.readline())["status"] == "offered"
        if phase in ("ready", "partial-acquired"):
            writer.write(b'{"status":"accept"}\n')
            await writer.drain()
            assert json.loads(await reader.readline())["status"] == "ready"
        if phase.startswith("partial-"):
            writer.write(b'{"status":')
            await writer.drain()
        if malformed:
            writer.write(malformed + b"\n")
            await writer.drain()
        yield reader
    finally:
        writer.close()
        with suppress(ConnectionError):
            await writer.wait_closed()


@pytest.mark.parametrize("value", [True, False, "1", object()])
def test_handshake_timeout_rejects_nonnumeric_values(value: Any) -> None:
    """Reject policy mistakes before creating any owner resources."""
    with pytest.raises(TypeError, match="handshake_timeout"):
        AdmissionBroker(max_in_flight=1, handshake_timeout=value)


@pytest.mark.parametrize(
    "value",
    [0, -1, float("nan"), float("inf"), -float("inf"), 10**1000],
    ids=["zero", "negative", "nan", "infinity", "negative-infinity", "overflow"],
)
def test_handshake_timeout_rejects_nonpositive_or_nonfinite_values(value: Any) -> None:
    """Use the existing finite-positive policy validator, including overflow."""
    with pytest.raises(ValueError, match="handshake_timeout"):
        AdmissionBroker(max_in_flight=1, handshake_timeout=value)


@pytest.mark.parametrize("value", [None, 1, 0.25])
def test_handshake_timeout_is_owner_only(value: float | None) -> None:
    """Leave serialized worker settings and wire compatibility unchanged."""
    with AdmissionBroker(max_in_flight=1, handshake_timeout=value) as broker:
        assert broker.handshake_timeout == value
        assert broker._server is not None
        assert broker._server._handshake_timeout == value
        assert "handshake_timeout" not in broker.controller_config().__dataclass_fields__


@pytest.mark.parametrize("phase", ["offered", "ready", "partial-accept", "partial-acquired"])
@pytest.mark.parametrize("gate", ["capacity", "pacing", "call-cap"])
async def test_stalled_handshake_expires_and_same_owner_recovers(phase: str, gate: str) -> None:
    """Reclaim only uncommitted reservations, including paced and budget-blocking offers."""
    async with _owner(
        max_in_flight=1 if gate == "capacity" else 4,
        requests_per_second=50 if gate == "pacing" else None,
        max_calls=1 if gate == "call-cap" else None,
        handshake_timeout=0.1,
    ) as (broker, controller):
        async with _stalled(broker, phase) as stalled:
            before = broker.snapshot()
            assert before.active == 1 and before.admitted_calls == 0
            permit = await controller.acquire(lambda _: None)
            try:
                assert await asyncio.wait_for(stalled.read(), 2) == b""
                assert broker.snapshot().admitted_calls == 1
            finally:
                permit.release()
            end = await _until(broker, lambda s: s.active == 0 and s.queued == 0)
            assert end.admitted_calls == 1
            if gate == "call-cap":
                with pytest.raises(AdmissionCallCapError):
                    await controller.acquire(lambda _: None)


@pytest.mark.parametrize("phase", ["offered", "ready"])
async def test_default_does_not_expire_an_unconfirmed_offer(phase: str) -> None:
    """Preserve today's unbounded owner policy unless the application opts in."""
    async with _owner(max_in_flight=1) as (broker, controller):
        short_wait = broker.controller(queue_timeout=0.08)
        try:
            async with _stalled(broker, phase):
                with pytest.raises(AdmissionTimeoutError):
                    await short_wait.acquire(lambda _: None)
                assert broker.snapshot().active == 1
                assert broker.snapshot().admitted_calls == 0
            await _until(broker, lambda s: s.active == 0)
            permit = await controller.acquire(lambda _: None)
            permit.release()
            assert (await _until(broker, lambda s: s.active == 0)).admitted_calls == 1
        finally:
            short_wait.close()


@pytest.mark.parametrize("phase", ["offered", "ready"])
@pytest.mark.parametrize(
    "malformed",
    [None, b"not-json", b"[]", b'"accept"', b"null", b"1", b"true", b"{}"],
    ids=[
        "disconnect",
        "invalid-json",
        "array",
        "string",
        "null",
        "number",
        "boolean",
        "missing-status",
    ],
)
async def test_failed_exchange_reclaims_capacity_before_its_long_deadline(
    phase: str, malformed: bytes | None, caplog
) -> None:
    """Reject malformed frames without an unhandled callback or waiting for expiry."""
    async with _owner(max_in_flight=1, max_calls=1, handshake_timeout=10) as (broker, controller):
        async with _stalled(broker, phase, malformed=malformed) as peer:
            if malformed is not None:
                assert await asyncio.wait_for(peer.read(), 2) == b""
        await _until(broker, lambda s: s.active == 0)
        permit = await controller.acquire(lambda _: None)
        permit.release()
        assert (await _until(broker, lambda s: s.active == 0)).admitted_calls == 1
    assert "Unhandled exception in client_connected_cb" not in caplog.text


async def test_expiry_preserves_existing_fifo_waiters() -> None:
    """Returning an abandoned offer serves older queued work before later arrivals."""
    async with _owner(max_in_flight=1, max_calls=3, handshake_timeout=1) as (broker, controller):
        pending = []
        try:
            async with _stalled(broker, "ready") as peer:
                for expected in range(1, 4):
                    pending.append(asyncio.create_task(controller.acquire(lambda _: None)))
                    await _until(broker, lambda s, count=expected: s.queued == count)
                for index, call in enumerate(pending):
                    permit = await call
                    assert all(not later.done() for later in pending[index + 1 :])
                    permit.release()
                assert await asyncio.wait_for(peer.read(), 2) == b""
            end = await _until(broker, lambda s: s.active == 0 and s.queued == 0)
            assert end.admitted_calls == 3 and end.peak_active == 1
        finally:
            for call in pending:
                if call.done() and not call.cancelled() and call.exception() is None:
                    call.result().release()
                else:
                    call.cancel()
            await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.parametrize("gate", ["capacity", "pacing", "cooldown"])
async def test_queue_waits_and_confirmed_leases_outlive_the_handshake_window(gate: str) -> None:
    """Start the deadline only after offering, never while waiting for policy gates."""
    async with _owner(
        max_in_flight=1,
        requests_per_second=4 if gate == "pacing" else None,
        max_cooldown=1 if gate == "cooldown" else None,
        handshake_timeout=0.04,
    ) as (broker, controller):
        held = await controller.acquire(lambda _: None)
        pending = None
        try:
            if gate == "cooldown":
                await held.cooldown(0.25)
            if gate != "capacity":
                held.release()
                held = None
            pending = asyncio.create_task(controller.acquire(lambda _: None))
            await _until(broker, lambda s: s.queued == 1)
            await asyncio.sleep(0.12)
            assert not pending.done()
            if held:
                assert broker.snapshot().active == 1
                held.release()
                held = None
            permit = await pending
            permit.release()
            assert (await _until(broker, lambda s: s.active == 0)).admitted_calls == 2
        finally:
            if held:
                held.release()
            if pending is not None:
                if pending.done() and not pending.cancelled() and pending.exception() is None:
                    pending.result().release()
                else:
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)


async def test_idle_authenticated_connection_can_be_reused_after_the_deadline(monkeypatch) -> None:
    """Rearm per offer without treating idle pooled sockets as stalled handshakes."""
    original = asyncio.open_connection
    connections = 0

    async def counted_connect(*args, **kwargs):
        """Count actual controller transport creation."""
        nonlocal connections
        connections += 1
        return await original(*args, **kwargs)

    monkeypatch.setattr(asyncio, "open_connection", counted_connect)
    async with _owner(max_in_flight=1, handshake_timeout=0.04) as (broker, controller):
        for _ in range(3):
            permit = await controller.acquire(lambda _: None)
            permit.release()
            await _until(broker, lambda s: s.active == 0)
            await asyncio.sleep(0.1)
        assert connections == 1
        assert broker.snapshot().admitted_calls == 3


async def test_confirmed_reply_is_outside_the_handshake_deadline(monkeypatch) -> None:
    """Never retract committed capacity while the final confirmation is in flight."""
    original = module._AdmissionBrokerServer._write
    committed, finish = threading.Event(), threading.Event()

    async def delayed_confirmation(writer, payload):
        """Hold only the post-commit write on the broker's owner loop."""
        if payload.get("status") == "confirmed":
            committed.set()
            while not finish.is_set():
                await asyncio.sleep(0.002)
        await original(writer, payload)

    monkeypatch.setattr(module._AdmissionBrokerServer, "_write", staticmethod(delayed_confirmation))
    async with _owner(max_in_flight=1, max_calls=1, handshake_timeout=0.04) as (broker, controller):
        pending = asyncio.create_task(controller.acquire(lambda _: None))
        try:
            assert await asyncio.to_thread(committed.wait, 3)
            await asyncio.sleep(0.12)
            assert not pending.done()
            assert broker.snapshot().active == broker.snapshot().admitted_calls == 1
            finish.set()
            permit = await pending
            permit.release()
            assert (await _until(broker, lambda s: s.active == 0)).admitted_calls == 1
        finally:
            finish.set()
            if pending.done() and not pending.cancelled() and pending.exception() is None:
                pending.result().release()
            else:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)


async def test_one_total_window_covers_both_offer_and_ready_writes(monkeypatch) -> None:
    """Do not reset the budget for the second acknowledgement or exclude writes."""
    original = module._AdmissionBrokerServer._write
    ready_started = threading.Event()
    delay = True

    async def delayed_write(writer, payload):
        """Each individual write fits the window, but their sum does not."""
        if delay and payload.get("status") in ("offered", "ready"):
            if payload["status"] == "ready":
                ready_started.set()
            await asyncio.sleep(0.6)
        await original(writer, payload)

    monkeypatch.setattr(module._AdmissionBrokerServer, "_write", staticmethod(delayed_write))
    async with _owner(max_in_flight=1, max_calls=1, handshake_timeout=1) as (broker, controller):
        with pytest.raises(AdmissionUnavailableError):
            await controller.acquire(lambda _: None)
        assert ready_started.is_set()
        assert broker.snapshot().admitted_calls == 0
        delay = False
        permit = await controller.acquire(lambda _: None)
        permit.release()
        assert (await _until(broker, lambda s: s.active == 0)).admitted_calls == 1


@pytest.mark.parametrize("phase", ["offered", "ready"])
async def test_shutdown_cancels_handshake_deadline_and_wakes_queued_clients(phase: str) -> None:
    """No expiry callback or uncommitted reservation survives owner shutdown."""
    async with _owner(max_in_flight=1, handshake_timeout=5) as (broker, controller):
        async with _stalled(broker, phase):
            pending = asyncio.create_task(controller.acquire(lambda _: None))
            server = broker._server
            assert server is not None
            try:
                await _until(broker, lambda s: s.queued == 1)
                broker.close()
                with pytest.raises(AdmissionUnavailableError):
                    await pending
                assert server.state.active == server.state.reserved_calls == 0
                assert not server.state.waiters
            finally:
                if not pending.done():
                    pending.cancel()
                    await asyncio.gather(pending, return_exceptions=True)


async def test_repeated_expiry_and_restart_do_not_retain_owner_threads() -> None:
    """Exercise timer cleanup and retain the configured policy across restart."""
    before = {t.ident for t in threading.enumerate() if t.name.startswith("nooa-admission-")}
    broker = AdmissionBroker(max_in_flight=1, handshake_timeout=0.03)
    for _ in range(10):
        broker.start()
        controller = broker.controller(queue_timeout=3)
        try:
            async with _stalled(broker, "ready") as peer:
                assert await asyncio.wait_for(peer.read(), 2) == b""
                permit = await controller.acquire(lambda _: None)
                permit.release()
                assert (await _until(broker, lambda s: s.active == 0)).admitted_calls == 1
        finally:
            controller.close()
            await asyncio.sleep(0.02)
            broker.close()
    after = {t.ident for t in threading.enumerate() if t.name.startswith("nooa-admission-")}
    assert after == before


async def test_buffered_final_ack_cannot_commit_after_deadline_before_timer_runs(
    monkeypatch,
) -> None:
    """Check the monotonic deadline even when awaits finish without an event-loop yield."""
    reader = asyncio.StreamReader()
    request = {
        "version": module._PROTOCOL_VERSION,
        "group": "buffered",
        "auth_token": "synthetic-token",
        "ticket": "buffered-final-ack",
    }
    reader.feed_data(
        json.dumps(request).encode() + b'\n{"status":"accept"}\n{"status":"acquired"}\n'
    )
    reader.feed_eof()
    written = []

    class Writer:
        """Keep protocol writes immediately awaitable without scheduling the timer."""

        def write(self, data):
            """Record wire outcomes, not controller credentials."""
            written.append(json.loads(data))

        async def drain(self):
            """Simulate an already-writable transport."""

        def close(self):
            """No OS transport is owned by this deterministic fixture."""

        async def wait_closed(self):
            """Simulate completed transport cleanup."""

    class ExpiredWindow:
        """Model elapsed monotonic time before the timeout callback has executed."""

        async def __aenter__(self):
            """Keep the event loop unmodified; only the broker sees this window."""
            return self

        async def __aexit__(self, *args):
            """Let the broker's normal error cleanup handle expiry."""
            return False

        def when(self):
            """Use a deadline that elapsed before the buffered acknowledgement."""
            return asyncio.get_running_loop().time() - 1

    class AsyncioProxy:
        """Override only this module's deadline context, not pytest's event loop."""

        def __getattr__(self, name):
            """Delegate ordinary broker scheduling operations unchanged."""
            return getattr(asyncio, name)

        @staticmethod
        def timeout(delay):
            """Supply the deterministic expired-window fixture."""
            assert delay == 1
            return ExpiredWindow()

    state = module._BrokerState(group="buffered", max_in_flight=1, max_calls=1)
    server = module._AdmissionBrokerServer(
        ("127.0.0.1", 0), state=state, auth_token="synthetic-token", handshake_timeout=1
    )

    async def offered(_reader, waiter):
        """Skip only queued EOF detection to feed already-buffered handshake frames."""
        return await waiter.future

    monkeypatch.setattr(server, "_wait_for_outcome", offered)
    monkeypatch.setattr(module, "asyncio", AsyncioProxy())
    await server._handle(reader, Writer())
    assert [item["status"] for item in written] == ["offered", "ready"]
    assert state.admitted_calls == state.active == state.reserved_calls == 0
    assert not state.waiters and not server._handlers and not server._connections
