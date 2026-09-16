# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parent-owned admission broker for local multiprocess applications.

The broker coordinates asynchronous provider attempts from child processes on
one host. Each granted slot is tied to a TCP connection, so the operating
system closes the lease and returns capacity if a child exits abruptly.

This is application-level backpressure, not a replacement for gateway-side
capacity enforcement or a multi-host distributed rate limiter.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import ipaddress
import json
import secrets
import socket
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future as ThreadFuture
from dataclasses import dataclass, field
from typing import Any, Literal

from nooa.unifiedllm.admission import (
    AdmissionCallCapError,
    AdmissionObserver,
    AdmissionOutcome,
    AdmissionTimeoutError,
    AdmissionUnavailableError,
    _named_group_identity,
    _validated_limit,
    _validated_timeout,
)

_PROTOCOL_VERSION = 1
_MAX_MESSAGE_BYTES = 16 * 1024
_IDLE_CONNECTION_SECONDS = 1.0
_WRITER_CLOSE_TIMEOUT_SECONDS = 0.5


def _validated_loopback_host(host: str) -> str:
    """Return a numeric loopback host or reject cleartext off-host transport."""
    if not isinstance(host, str):
        raise ValueError("host must be a numeric loopback IP address")
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise ValueError("host must be a numeric loopback IP address") from error
    if not address.is_loopback:
        raise ValueError("host must be a numeric loopback IP address")
    return host


def _socket_family(host: str) -> socket.AddressFamily:
    """Return the socket family for a validated numeric IP address."""
    return socket.AF_INET6 if ipaddress.ip_address(host).version == 6 else socket.AF_INET


@dataclass(frozen=True, slots=True)
class BrokerAdmissionConfig:
    """Serializable connection settings passed from a parent to its children."""

    host: str
    port: int
    auth_token: str = field(repr=False)
    group: str
    max_in_flight: int
    max_calls: int | None
    queue_timeout: float | None

    def __post_init__(self) -> None:
        """Prevent controller credentials from being sent over a non-loopback socket."""
        _validated_loopback_host(self.host)

    def controller(self) -> BrokerAdmissionController:
        """Create an independent controller client for the current process."""
        return BrokerAdmissionController(self)


@dataclass(frozen=True, slots=True)
class BrokerAdmissionSnapshot:
    """Current broker counters for diagnostics and comparison runs."""

    active: int
    peak_active: int
    queued: int
    admitted_calls: int
    max_in_flight: int
    max_calls: int | None


@dataclass(slots=True)
class _BrokerWaiter:
    ticket: str
    queued_at: float
    queue_depth: int
    queued: bool
    future: asyncio.Future[dict[str, Any]]
    state: Literal["queued", "offered", "leased", "cancelled", "terminal"] = "queued"


class _BrokerState:
    def __init__(self, *, group: str, max_in_flight: int, max_calls: int | None):
        self.group = group
        self.max_in_flight = max_in_flight
        self.max_calls = max_calls
        self.active = 0
        self.peak_active = 0
        self.admitted_calls = 0
        self.waiters: deque[_BrokerWaiter] = deque()

    def enqueue(self, ticket: str) -> _BrokerWaiter:
        """Append one FIFO waiter and offer any currently available capacity."""
        loop = asyncio.get_running_loop()
        queued = bool(self.waiters) or self.active >= self.max_in_flight
        waiter = _BrokerWaiter(
            ticket=ticket,
            queued_at=time.perf_counter(),
            queue_depth=len(self.waiters) + 1,
            queued=queued,
            future=loop.create_future(),
        )
        self.waiters.append(waiter)
        self._offer_available()
        return waiter

    def disconnect(self, waiter: _BrokerWaiter) -> None:
        """Remove a queued client or return its offered/leased capacity."""
        if waiter.state == "queued":
            waiter.state = "cancelled"
            with contextlib.suppress(ValueError):
                self.waiters.remove(waiter)
            waiter.future.cancel()
        elif waiter.state == "offered":
            waiter.state = "cancelled"
            self._decrement_active(retract_call=True)
        elif waiter.state == "leased":
            waiter.state = "terminal"
            self._decrement_active(retract_call=False)
        self._offer_available()

    def confirm(self, waiter: _BrokerWaiter) -> None:
        """Mark an offered slot ready for provider dispatch."""
        if waiter.state != "offered":
            raise RuntimeError(f"Cannot confirm admission waiter in state {waiter.state!r}")
        waiter.state = "leased"

    def _decrement_active(self, *, retract_call: bool) -> None:
        if self.active <= 0 or (retract_call and self.admitted_calls <= 0):
            raise RuntimeError("Admission broker accounting underflow")
        self.active -= 1
        if retract_call:
            self.admitted_calls -= 1

    def _offer_available(self) -> None:
        while self.waiters:
            waiter = self.waiters[0]
            if self.max_calls is not None and self.admitted_calls >= self.max_calls:
                self.waiters.popleft()
                waiter.state = "terminal"
                waiter.future.set_result(
                    {
                        "status": "call_cap",
                        "queued": waiter.queued,
                        "queue_depth": waiter.queue_depth,
                        "wait_s": time.perf_counter() - waiter.queued_at,
                        "admitted_calls": self.admitted_calls,
                    }
                )
                continue
            if self.active >= self.max_in_flight:
                return

            self.waiters.popleft()
            if waiter.state != "queued":
                continue
            waiter.state = "offered"
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
            self.admitted_calls += 1
            wait_s = time.perf_counter() - waiter.queued_at
            queued = waiter.queued or wait_s >= 0.001
            waiter.future.set_result(
                {
                    "status": "offered",
                    "queued": queued,
                    "queue_depth": waiter.queue_depth if queued else 0,
                    "wait_s": wait_s,
                    "admitted_calls": self.admitted_calls,
                }
            )

    def snapshot(self) -> BrokerAdmissionSnapshot:
        return BrokerAdmissionSnapshot(
            active=self.active,
            peak_active=self.peak_active,
            queued=len(self.waiters),
            admitted_calls=self.admitted_calls,
            max_in_flight=self.max_in_flight,
            max_calls=self.max_calls,
        )


class _AdmissionBrokerServer:
    """One background event loop serving all active and queued leases."""

    def __init__(
        self,
        address: tuple[str, int],
        *,
        state: _BrokerState,
        auth_token: str,
    ):
        self.host, self.port = address
        self.state = state
        self.auth_token = auth_token
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: asyncio.Server | None = None
        self._thread: threading.Thread | None = None
        self._ready: ThreadFuture[None] = ThreadFuture()
        self._connections: set[asyncio.StreamWriter] = set()
        self._handlers: set[asyncio.Task[None]] = set()

    @property
    def server_address(self) -> tuple[str, int]:
        if self._server is None or not self._server.sockets:
            raise RuntimeError("Admission broker server is not bound")
        host, port = self._server.sockets[0].getsockname()[:2]
        return str(host), int(port)

    def start(self) -> None:
        thread = threading.Thread(
            target=self._run,
            name=f"nooa-admission-{self.state.group}",
            daemon=True,
        )
        self._thread = thread
        thread.start()
        self._ready.result(timeout=5)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.run_until_complete(self._bind())
        except BaseException as error:
            self._ready.set_exception(error)
            loop.close()
            return
        self._ready.set_result(None)
        loop.run_forever()
        loop.close()

    async def _bind(self) -> None:
        self._server = await asyncio.start_server(
            self._handle,
            self.host,
            self.port,
            family=_socket_family(self.host),
            backlog=2048,
            limit=_MAX_MESSAGE_BYTES + 1,
        )

    async def _handle(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        task = asyncio.current_task()
        assert task is not None
        self._handlers.add(task)
        self._connections.add(writer)
        waiter: _BrokerWaiter | None = None
        try:
            # A controller reuses a small pool of connections. The lease still belongs to this
            # connection, so process exit or a broken socket returns capacity exactly as before;
            # reuse only avoids consuming one ephemeral TCP port per provider attempt.
            while True:
                raw = await reader.readline()
                if not raw or len(raw) > _MAX_MESSAGE_BYTES:
                    return
                request = json.loads(raw)
                if not self._authorized(request):
                    await self._write(writer, {"status": "unauthorized"})
                    return

                ticket = str(request.get("ticket", ""))
                if not ticket:
                    return
                waiter = self.state.enqueue(ticket)
                outcome = await self._wait_for_outcome(reader, waiter)
                if outcome is None:
                    return
                if outcome["status"] == "call_cap":
                    await self._write(writer, outcome)
                    return

                await self._write(writer, outcome)
                acknowledgement = await reader.readline()
                if not acknowledgement or len(acknowledgement) > _MAX_MESSAGE_BYTES:
                    return
                if json.loads(acknowledgement).get("status") != "accept":
                    return
                await self._write(writer, {"status": "ready"})
                self.state.confirm(waiter)
                # Explicit release and EOF are equivalent. EOF also covers an abrupt child-process
                # exit and makes the grant a crash-safe lease. A well-formed release keeps the
                # connection alive for the controller's next attempt.
                released = await reader.readline()
                if not released or len(released) > _MAX_MESSAGE_BYTES:
                    return
                if json.loads(released).get("status") != "release":
                    return
                self.state.disconnect(waiter)
                waiter = None
        except asyncio.CancelledError:
            raise
        except (BrokenPipeError, ConnectionResetError, json.JSONDecodeError, OSError, ValueError):
            pass
        finally:
            if waiter is not None:
                self.state.disconnect(waiter)
            self._handlers.discard(task)
            self._connections.discard(writer)
            writer.close()
            # A peer holding a lease does not read again until release. On some transports,
            # wait_closed() can consequently outlive broker shutdown even after close() has been
            # requested. Broker ownership must remain bounded: the socket is already closing and
            # the handler has returned its lease, so do not let transport teardown hold the
            # server's shutdown gather indefinitely.
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await asyncio.wait_for(writer.wait_closed(), timeout=_WRITER_CLOSE_TIMEOUT_SECONDS)

    async def _wait_for_outcome(
        self,
        reader: asyncio.StreamReader,
        waiter: _BrokerWaiter,
    ) -> dict[str, Any] | None:
        disconnected = asyncio.create_task(reader.read(1))
        try:
            done, _ = await asyncio.wait(
                (waiter.future, disconnected),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if disconnected in done:
                self.state.disconnect(waiter)
                return None
            return waiter.future.result()
        finally:
            if not disconnected.done():
                disconnected.cancel()
            await asyncio.gather(disconnected, return_exceptions=True)

    def _authorized(self, request: Any) -> bool:
        if not isinstance(request, dict):
            return False
        token = request.get("auth_token")
        return (
            request.get("version") == _PROTOCOL_VERSION
            and request.get("group") == self.state.group
            and isinstance(token, str)
            and hmac.compare_digest(token, self.auth_token)
        )

    @staticmethod
    async def _write(writer: asyncio.StreamWriter, payload: dict[str, Any]) -> None:
        writer.write(json.dumps(payload).encode() + b"\n")
        await writer.drain()

    def snapshot(self) -> BrokerAdmissionSnapshot:
        if self._loop is None:
            raise RuntimeError("Admission broker server is not running")

        async def capture() -> BrokerAdmissionSnapshot:
            return self.state.snapshot()

        return asyncio.run_coroutine_threadsafe(capture(), self._loop).result(timeout=5)

    def close(self) -> None:
        loop, thread = self._loop, self._thread
        if loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._shutdown(), loop).result(timeout=5)
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5)
        self._loop = None
        self._thread = None

    async def _shutdown(self) -> None:
        server = self._server
        self._server = None
        if server is not None:
            # Stop accepting first, then close established connections before waiting for the
            # listening server to finish. Some asyncio transports include live client handlers
            # in wait_closed(), so waiting before closing those clients can deadlock shutdown.
            server.close()
        for writer in tuple(self._connections):
            writer.close()
        for task in tuple(self._handlers):
            task.cancel()
        if server is not None:
            await server.wait_closed()
        if self._handlers:
            await asyncio.gather(*self._handlers, return_exceptions=True)


class _BrokerAdmissionPermit:
    __slots__ = ("_controller", "_loop", "_reader", "_release_lock", "_released", "_writer")

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        controller: BrokerAdmissionController,
        loop: asyncio.AbstractEventLoop,
    ):
        self._reader = reader
        self._writer = writer
        self._controller = controller
        self._loop = loop
        self._released = False
        self._release_lock = threading.Lock()

    def release(self) -> None:
        with self._release_lock:
            if self._released:
                return
            self._released = True
        self._controller._release_connection(self._reader, self._writer, self._loop)


@dataclass(slots=True)
class _IdleBrokerConnection:
    loop: asyncio.AbstractEventLoop
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    expiry: asyncio.TimerHandle | None = None


class BrokerAdmissionController:
    """Admission-controller client connected to a parent-owned broker."""

    def __init__(self, config: BrokerAdmissionConfig):
        self.config = config
        self.queue_timeout = _validated_timeout(config.queue_timeout)
        self._idle: deque[_IdleBrokerConnection] = deque()
        self._idle_lock = threading.Lock()
        self._closed = False

    async def acquire(self, observer: AdmissionObserver) -> _BrokerAdmissionPermit:
        started = time.perf_counter()
        loop = asyncio.get_running_loop()
        writer: asyncio.StreamWriter | None = None
        leased = False
        try:
            pooled = self._take_idle_connection(loop)
            if pooled is None:
                remaining = self._remaining_timeout(started)
                connection = asyncio.open_connection(
                    self.config.host,
                    self.config.port,
                    family=_socket_family(self.config.host),
                )
                if remaining is None:
                    reader, writer = await connection
                else:
                    reader, writer = await asyncio.wait_for(connection, remaining)
            else:
                reader, writer = pooled.reader, pooled.writer
            request = {
                "version": _PROTOCOL_VERSION,
                "auth_token": self.config.auth_token,
                "group": self.config.group,
                "ticket": uuid.uuid4().hex,
            }
            writer.write(json.dumps(request).encode() + b"\n")
            await self._drain(writer, started)
            first = await self._readline(reader, started)
            if not first:
                raise AdmissionUnavailableError("Admission broker closed before granting a lease")
            response = json.loads(first)
            status = response.get("status")
            if status == "call_cap":
                self._observe(observer, "call_cap", response, started)
                raise AdmissionCallCapError(
                    f"LLM admission call cap reached for group {self.config.group!r} "
                    f"(max_calls={self.config.max_calls})"
                )
            if status == "unauthorized":
                raise AdmissionUnavailableError("Admission broker rejected the client credentials")
            if status != "offered":
                raise AdmissionUnavailableError(f"Unexpected admission broker response: {status!r}")

            writer.write(b'{"status":"accept"}\n')
            await self._drain(writer, started)
            second = await self._readline(reader, started)
            if not second or json.loads(second).get("status") != "ready":
                raise AdmissionUnavailableError("Admission broker did not confirm the lease")

            outcome: Literal["immediate", "admitted_after_wait"] = (
                "admitted_after_wait" if response.get("queued") else "immediate"
            )
            self._observe(observer, outcome, response, started)
            leased = True
            return _BrokerAdmissionPermit(reader, writer, self, loop)
        except TimeoutError as error:
            self._observe(observer, "timeout", {}, started)
            timeout_label = (
                f"{self.queue_timeout:g}s"
                if self.queue_timeout is not None
                else "the admission deadline"
            )
            raise AdmissionTimeoutError(
                f"Timed out after {timeout_label} waiting for LLM admission "
                f"group {self.config.group!r} "
                f"(max_in_flight={self.config.max_in_flight})"
            ) from error
        except asyncio.CancelledError:
            self._observe(observer, "cancelled", {}, started)
            raise
        except AdmissionCallCapError:
            raise
        except AdmissionUnavailableError:
            self._observe(observer, "unavailable", {}, started)
            raise
        except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
            self._observe(observer, "unavailable", {}, started)
            raise AdmissionUnavailableError(
                f"Admission broker unavailable at {self.config.host}:{self.config.port}"
            ) from error
        finally:
            if writer is not None and not leased:
                await self._close_writer(writer)

    async def _readline(self, reader: asyncio.StreamReader, started: float) -> bytes:
        remaining = self._remaining_timeout(started)
        if remaining is None:
            return await reader.readline()
        return await asyncio.wait_for(reader.readline(), remaining)

    async def _drain(self, writer: asyncio.StreamWriter, started: float) -> None:
        remaining = self._remaining_timeout(started)
        if remaining is None:
            await writer.drain()
        else:
            await asyncio.wait_for(writer.drain(), remaining)

    def _remaining_timeout(self, started: float) -> float | None:
        """Return the queue deadline remainder or raise when it has elapsed."""
        if self.queue_timeout is None:
            return None
        remaining = self.queue_timeout - (time.perf_counter() - started)
        if remaining <= 0:
            raise TimeoutError
        return remaining

    def _observe(
        self,
        observer: AdmissionObserver,
        outcome: AdmissionOutcome,
        response: dict[str, Any],
        started: float,
    ) -> None:
        queued = bool(response.get("queued", outcome in ("timeout", "cancelled")))
        observer(
            {
                "group": self.config.group,
                "outcome": outcome,
                "queued": queued,
                "wait_s": round(time.perf_counter() - started, 6),
                "queue_depth": int(response.get("queue_depth", 0)),
                "max_in_flight": self.config.max_in_flight,
                "max_calls": self.config.max_calls,
                "admitted_calls": int(response.get("admitted_calls", 0)),
                "backend": "parent_broker",
            }
        )

    @staticmethod
    async def _close_writer(writer: asyncio.StreamWriter) -> None:
        writer.close()
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await asyncio.wait_for(writer.wait_closed(), timeout=_WRITER_CLOSE_TIMEOUT_SECONDS)

    def close(self) -> None:
        """Close this controller's idle connections; active permits close when released."""
        with self._idle_lock:
            if self._closed:
                return
            self._closed = True
            idle = tuple(self._idle)
            self._idle.clear()
        for connection in idle:
            if connection.expiry is not None:
                connection.expiry.cancel()
            self._close_connection_on_owner_loop(connection)

    def _take_idle_connection(
        self, loop: asyncio.AbstractEventLoop
    ) -> _IdleBrokerConnection | None:
        stale: list[_IdleBrokerConnection] = []
        selected: _IdleBrokerConnection | None = None
        with self._idle_lock:
            if self._closed:
                raise AdmissionUnavailableError("Admission controller is closed")
            while self._idle:
                candidate = self._idle.pop()
                if candidate.expiry is not None:
                    candidate.expiry.cancel()
                if (
                    candidate.loop is loop
                    and not candidate.writer.is_closing()
                    and not candidate.reader.at_eof()
                ):
                    selected = candidate
                    break
                stale.append(candidate)
        for connection in stale:
            self._close_connection_on_owner_loop(connection)
        return selected

    def _release_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        def release_and_pool() -> None:
            if writer.is_closing():
                return
            try:
                writer.write(b'{"status":"release"}\n')
            except Exception:
                writer.close()
                return

            connection = _IdleBrokerConnection(loop, reader, writer)
            connection.expiry = loop.call_later(
                _IDLE_CONNECTION_SECONDS,
                self._expire_idle_connection,
                connection,
            )
            with self._idle_lock:
                keep = not self._closed and len(self._idle) < self.config.max_in_flight
                if keep:
                    self._idle.append(connection)
            if not keep:
                assert connection.expiry is not None
                connection.expiry.cancel()
                writer.close()

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            release_and_pool()
        elif not loop.is_closed():
            loop.call_soon_threadsafe(release_and_pool)
        else:
            with contextlib.suppress(Exception):
                writer.close()

    def _expire_idle_connection(self, connection: _IdleBrokerConnection) -> None:
        with self._idle_lock:
            with contextlib.suppress(ValueError):
                self._idle.remove(connection)
        connection.writer.close()

    @staticmethod
    def _close_connection_on_owner_loop(connection: _IdleBrokerConnection) -> None:
        def close() -> None:
            connection.writer.close()

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is connection.loop:
            close()
        elif not connection.loop.is_closed():
            connection.loop.call_soon_threadsafe(close)
        else:
            with contextlib.suppress(Exception):
                connection.writer.close()


class AdmissionBroker:
    """Lifecycle owner for one host-local, application-scoped admission group."""

    def __init__(
        self,
        *,
        max_in_flight: int,
        max_calls: int | None = None,
        group: str = "application",
        host: str = "127.0.0.1",
        port: int = 0,
    ):
        limit = _validated_limit(max_in_flight)
        assert limit is not None
        if isinstance(max_calls, bool) or (
            max_calls is not None and (not isinstance(max_calls, int) or max_calls <= 0)
        ):
            raise ValueError("max_calls must be a positive integer or None")
        _, validated_group = _named_group_identity(group)
        if not isinstance(port, int) or isinstance(port, bool) or not 0 <= port <= 65535:
            raise ValueError("port must be an integer between 0 and 65535")

        self.max_in_flight = limit
        self.max_calls = max_calls
        self.group = validated_group
        self.host = _validated_loopback_host(host)
        self.port = port
        self._auth_token: str | None = None
        self._server: _AdmissionBrokerServer | None = None

    def start(self) -> AdmissionBroker:
        """Bind the broker and start its background server thread."""
        if self._server is not None:
            return self
        auth_token = secrets.token_urlsafe(32)
        server = _AdmissionBrokerServer(
            (self.host, self.port),
            state=_BrokerState(
                group=self.group,
                max_in_flight=self.max_in_flight,
                max_calls=self.max_calls,
            ),
            auth_token=auth_token,
        )
        server.start()
        self._auth_token = auth_token
        self._server = server
        return self

    def controller_config(self, *, queue_timeout: float | None = None) -> BrokerAdmissionConfig:
        """Return serializable settings for child-process controllers."""
        auth_token = self._auth_token
        if self._server is None or auth_token is None:
            raise RuntimeError("AdmissionBroker.start() must be called first")
        host = str(self._server.server_address[0])
        port = int(self._server.server_address[1])
        return BrokerAdmissionConfig(
            host=host,
            port=port,
            auth_token=auth_token,
            group=self.group,
            max_in_flight=self.max_in_flight,
            max_calls=self.max_calls,
            queue_timeout=_validated_timeout(queue_timeout),
        )

    def controller(self, *, queue_timeout: float | None = None) -> BrokerAdmissionController:
        """Create a controller client for the current or a child process."""
        return self.controller_config(queue_timeout=queue_timeout).controller()

    def snapshot(self) -> BrokerAdmissionSnapshot:
        """Return active, queued, and run-call counters."""
        if self._server is None:
            raise RuntimeError("AdmissionBroker.start() must be called first")
        return self._server.snapshot()

    def close(self) -> None:
        """Stop the broker after its application run has finished."""
        server = self._server
        if server is None:
            return
        self._server = None
        self._auth_token = None
        server.close()

    def __enter__(self) -> AdmissionBroker:
        return self.start()

    def __exit__(self, *_exc_info: Any) -> None:
        self.close()


__all__ = [
    "AdmissionBroker",
    "BrokerAdmissionConfig",
    "BrokerAdmissionController",
    "BrokerAdmissionSnapshot",
]
