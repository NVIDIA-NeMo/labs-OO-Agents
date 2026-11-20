# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private in-process ownership for live ACP sessions.

This is an adapter implementation detail, not part of the durable session
model. A future agent daemon can replace this registry with daemon handles
without changing :mod:`nooa_cli.sessions` or the ACP protocol surface.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum
from types import TracebackType
from typing import Any

type CloseCallback[T] = Callable[[T], Awaitable[None] | None]


@dataclass(slots=True)
class _ACPOutcome:
    value: Any = None
    failure: BaseException | None = None
    traceback: TracebackType | None = None

    def unwrap(self):
        if self.failure is not None:
            raise self.failure.with_traceback(self.traceback)
        return self.value


async def _acp_capture(operation) -> _ACPOutcome:
    # All ACP-owned workers finish with DATA, including control exceptions.
    try:
        return _ACPOutcome(value=await operation)
    except BaseException as error:
        return _ACPOutcome(failure=error, traceback=error.__traceback__)


def _acp_task(operation, *, name):
    worker = _acp_capture(operation)
    try:
        return asyncio.create_task(worker, name=name)
    except BaseException:
        worker.close()
        operation.close()
        raise


async def _acp_join(task) -> _ACPOutcome:
    """Join the actual ACP worker despite cancellation of its public waiter."""
    cancellation = None
    while True:
        try:
            outcome = await asyncio.shield(task)
            break
        except asyncio.CancelledError as error:
            if task.cancelled():
                return _ACPOutcome(failure=error, traceback=error.__traceback__)
            if cancellation is None:
                cancellation = _ACPOutcome(failure=error, traceback=error.__traceback__)
    if cancellation is not None and outcome.failure is None:
        return cancellation
    return outcome


class SessionBusyError(RuntimeError):
    """Raised when a second foreground turn targets a busy ACP session."""


class SessionRuntimeClosedError(RuntimeError):
    """Raised when a caller targets a closing or closed ACP session."""


class _RuntimeState(StrEnum):
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


class SessionRuntime[T]:
    """One adapter-owned live session with serialized foreground turns."""

    def __init__(
        self,
        session_id: str,
        value: T,
        *,
        close: CloseCallback[T] | None = None,
    ) -> None:
        self.session_id = session_id
        self.value = value
        self._close_callback = close
        self._turn_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._state = _RuntimeState.OPEN
        self._turn_claimed = False
        self._close_task: asyncio.Task[_ACPOutcome] | None = None

    @property
    def busy(self) -> bool:
        """Whether a foreground turn currently owns or is entering this session."""
        return self._turn_claimed

    @property
    def is_closed(self) -> bool:
        return self._state is _RuntimeState.CLOSED

    @asynccontextmanager
    async def turn(self) -> AsyncIterator[T]:
        """Own the foreground turn, failing immediately if it is already claimed."""
        async with self._state_lock:
            if self._state is not _RuntimeState.OPEN:
                raise SessionRuntimeClosedError(f"Session {self.session_id!r} is not open")
            if self._turn_claimed:
                raise SessionBusyError(f"Session {self.session_id!r} already has an active turn")
            self._turn_claimed = True

        acquired = False
        try:
            await self._turn_lock.acquire()
            acquired = True
            async with self._state_lock:
                if self._state is not _RuntimeState.OPEN:
                    raise SessionRuntimeClosedError(f"Session {self.session_id!r} is not open")
            yield self.value
        finally:
            if acquired:
                self._turn_lock.release()
            async with self._state_lock:
                self._turn_claimed = False

    async def close(self) -> None:
        """Wait for the foreground turn, release resources once, and mark closed."""
        async with self._state_lock:
            retryable = getattr(self.value, "_close_retryable", False)
            if self._close_task is None or (self._close_task.done() and retryable):
                if self._state is _RuntimeState.CLOSED and not retryable:
                    return
                self._state = _RuntimeState.CLOSING
                self._close_task = _acp_task(
                    self._close_once(),
                    name=f"nooa-acp-close-{self.session_id}",
                )
            close_task = self._close_task

        # The owned worker survives cancellation, but this request returns promptly.
        (await asyncio.shield(close_task)).unwrap()

    async def _close_once(self) -> None:
        await self._turn_lock.acquire()
        try:
            await self._close_value()
        finally:
            self._turn_lock.release()
            async with self._state_lock:
                self._state = _RuntimeState.CLOSED

    async def _close_value(self) -> None:
        callback = self._close_callback
        if callback is not None:
            result = callback(self.value)
        else:
            close = getattr(self.value, "close", None)
            if close is None:
                return
            result = close()
        if inspect.isawaitable(result):
            await result


class SessionRuntimePool[T]:
    """Concurrency-safe, ACP-private registry of live in-process sessions."""

    def __init__(self) -> None:
        self._runtimes: dict[str, SessionRuntime[T]] = {}
        self._available: set[str] = set()
        self._retired: dict[str, SessionRuntime[T]] = {}
        self._remove_tasks: dict[str, asyncio.Task[_ACPOutcome]] = {}
        self._lock = asyncio.Lock()
        self._closed = False
        self._close_task: asyncio.Task[_ACPOutcome] | None = None

    async def add(
        self,
        session_id: str,
        value: T,
        *,
        close: CloseCallback[T] | None = None,
        available: bool = True,
    ) -> SessionRuntime[T]:
        async with self._lock:
            if self._closed:
                raise SessionRuntimeClosedError("ACP session runtime pool is closed")
            if session_id in self._runtimes or session_id in self._retired:
                raise ValueError(f"Session {session_id!r} is already registered")
            runtime = SessionRuntime(session_id, value, close=close)
            self._runtimes[session_id] = runtime
            if available:
                self._available.add(session_id)
            return runtime

    async def owns(self, session_id: str) -> bool:
        """Include unpublished and failed-close owners in the writer fence."""
        async with self._lock:
            return session_id in self._runtimes or session_id in self._retired

    async def publish(self, session_id: str) -> None:
        """Make a fully initialized runtime available to protocol requests."""
        async with self._lock:
            if session_id not in self._runtimes:
                raise KeyError(f"Unknown live ACP session {session_id!r}")
            self._available.add(session_id)

    async def get(self, session_id: str) -> SessionRuntime[T]:
        async with self._lock:
            if session_id not in self._available:
                raise KeyError(f"Unknown live ACP session {session_id!r}")
            return self._runtimes[session_id]

    async def ids(self) -> tuple[str, ...]:
        async with self._lock:
            return tuple(
                session_id for session_id in self._runtimes if session_id in self._available
            )

    async def remove(self, session_id: str, *, include_unavailable: bool = False) -> T:
        """Close and unregister one runtime, returning its adapter value."""
        async with self._lock:
            runtime = self._retired.get(session_id)
            if runtime is None:
                if session_id not in self._runtimes or (
                    not include_unavailable and session_id not in self._available
                ):
                    raise KeyError(f"Unknown live ACP session {session_id!r}")
                runtime = self._runtimes[session_id]
            remove_task = self._remove_tasks.get(session_id)
            if remove_task is None:
                remove_task = _acp_task(
                    self._remove_once(session_id, runtime),
                    name=f"nooa-acp-remove-{session_id}",
                )
                self._remove_tasks[session_id] = remove_task
        # Cancellation ends this request, not the worker or its ID reservation.
        return (await asyncio.shield(remove_task)).unwrap()

    async def _remove_if_owned(self, session_id: str, value: T) -> None:
        """Retire a failed registration only when its exact typed owner is ours."""
        async with self._lock:
            runtime = self._runtimes.get(session_id) or self._retired.get(session_id)
            if runtime is None or runtime.value is not value:
                return
            remove_task = self._remove_tasks.get(session_id)
            if remove_task is None:
                remove_task = _acp_task(
                    self._remove_once(session_id, runtime),
                    name=f"nooa-acp-remove-{session_id}",
                )
                self._remove_tasks[session_id] = remove_task
        (await _acp_join(remove_task)).unwrap()

    async def _remove_once(self, session_id: str, runtime: SessionRuntime[T]) -> T:
        result = await _acp_capture(runtime.close())
        async with self._lock:
            # First-error DATA is not an outstanding resource obligation.
            # Untyped owners retain the conservative failed-close contract.
            if result.failure is not None and getattr(runtime.value, "_close_retained", True):
                self._retired[session_id] = runtime
            else:
                self._retired.pop(session_id, None)
            self._remove_tasks.pop(session_id, None)
            if self._runtimes.get(session_id) is runtime:
                del self._runtimes[session_id]
                self._available.discard(session_id)
        result.unwrap()
        return runtime.value

    async def close(self) -> None:
        """Close every registered runtime and reject future registrations."""
        async with self._lock:
            if self._close_task is None or (
                self._close_task.done()
                and any(
                    getattr(runtime.value, "_close_retryable", False)
                    for runtime in self._retired.values()
                )
            ):
                self._closed = True
                runtimes = list(self._runtimes.values()) + list(self._retired.values())
                self._close_task = _acp_task(
                    self._close_all(runtimes),
                    name="nooa-acp-close-sessions",
                )
            close_task = self._close_task

        (await _acp_join(close_task)).unwrap()

    async def _close_all(self, runtimes: list[SessionRuntime[T]]) -> None:
        results = await asyncio.gather(
            *(_acp_capture(runtime.close()) for runtime in runtimes),
            return_exceptions=True,
        )
        failures = []
        async with self._lock:
            self._runtimes.clear()
            self._available.clear()
            for runtime, result in zip(runtimes, results, strict=True):
                if result.failure is not None:
                    failures.append(result.failure)
                if result.failure is not None and getattr(runtime.value, "_close_retained", True):
                    self._retired[runtime.session_id] = runtime
                else:
                    self._retired.pop(runtime.session_id, None)
        if failures:
            raise BaseExceptionGroup("Failed to close one or more ACP sessions", failures)
