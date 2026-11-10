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
from enum import StrEnum

type CloseCallback[T] = Callable[[T], Awaitable[None] | None]


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
        self._close_task: asyncio.Task[None] | None = None

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
            if self._close_task is None:
                if self._state is _RuntimeState.CLOSED:
                    return
                self._state = _RuntimeState.CLOSING
                self._close_task = asyncio.create_task(
                    self._close_once(),
                    name=f"nooa-acp-close-{self.session_id}",
                )
            close_task = self._close_task

        # Cleanup belongs to the adapter, not to the lifetime of one request.
        await asyncio.shield(close_task)

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
        self._remove_tasks: dict[str, asyncio.Task[T]] = {}
        self._lock = asyncio.Lock()
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

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
            if session_id in self._runtimes:
                raise ValueError(f"Session {session_id!r} is already registered")
            runtime = SessionRuntime(session_id, value, close=close)
            self._runtimes[session_id] = runtime
            if available:
                self._available.add(session_id)
            return runtime

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
            if session_id not in self._runtimes or (
                not include_unavailable and session_id not in self._available
            ):
                raise KeyError(f"Unknown live ACP session {session_id!r}")
            runtime = self._runtimes[session_id]
            remove_task = self._remove_tasks.get(session_id)
            if remove_task is None:
                remove_task = asyncio.create_task(
                    self._remove_once(session_id, runtime),
                    name=f"nooa-acp-remove-{session_id}",
                )
                self._remove_tasks[session_id] = remove_task

                def _finished(done: asyncio.Task[T]) -> None:
                    if not done.cancelled():
                        done.exception()

                remove_task.add_done_callback(_finished)

        # Teardown and unregistration belong to the adapter, not to the request
        # that happened to initiate them. Cancellation must not make the id
        # reusable while its old runtime is still active.
        return await asyncio.shield(remove_task)

    async def _remove_once(self, session_id: str, runtime: SessionRuntime[T]) -> T:
        try:
            await runtime.close()
        finally:
            async with self._lock:
                self._remove_tasks.pop(session_id, None)
                if self._runtimes.get(session_id) is runtime:
                    del self._runtimes[session_id]
                    self._available.discard(session_id)
        return runtime.value

    async def close(self) -> None:
        """Close every registered runtime and reject future registrations."""
        async with self._lock:
            if self._close_task is None:
                self._closed = True
                runtimes = list(self._runtimes.values())
                self._close_task = asyncio.create_task(
                    self._close_all(runtimes),
                    name="nooa-acp-close-sessions",
                )
            close_task = self._close_task

        await asyncio.shield(close_task)

    async def _close_all(self, runtimes: list[SessionRuntime[T]]) -> None:
        results = await asyncio.gather(
            *(runtime.close() for runtime in runtimes),
            return_exceptions=True,
        )
        async with self._lock:
            self._runtimes.clear()
            self._available.clear()

        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise BaseExceptionGroup("Failed to close one or more ACP sessions", failures)
