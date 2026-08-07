# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lifecycle and foreground-turn ownership for live sessions.

The runtime value is intentionally generic. A native terminal may store its
dispatcher while an ACP adapter stores a bundle containing an agent,
dispatcher, and event bridge. Core owns only the concurrency invariant: one
foreground turn per session, independent sessions may run concurrently.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from enum import StrEnum

type CloseCallback[T] = Callable[[T], Awaitable[None] | None]


class SessionBusyError(RuntimeError):
    """Raised when a second foreground turn targets a busy session."""


class SessionRuntimeClosedError(RuntimeError):
    """Raised when a caller targets a closing or closed session runtime."""


class _RuntimeState(StrEnum):
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


class SessionRuntime[T]:
    """A host-specific runtime value with host-neutral turn and close semantics."""

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
        self._close_task: asyncio.Task[None] | None = None

    @property
    def busy(self) -> bool:
        """Whether a foreground turn currently owns this session."""
        return self._turn_lock.locked()

    @property
    def is_closed(self) -> bool:
        return self._state is _RuntimeState.CLOSED

    @asynccontextmanager
    async def turn(self, *, wait: bool = False) -> AsyncIterator[T]:
        """Own the session's foreground turn for the duration of the context.

        Args:
            wait: Queue behind an existing turn when true. When false, fail
                immediately with :class:`SessionBusyError`.
        """
        async with self._state_lock:
            if self._state is not _RuntimeState.OPEN:
                raise SessionRuntimeClosedError(f"Session {self.session_id!r} is not open")
            if not wait and self._turn_lock.locked():
                raise SessionBusyError(f"Session {self.session_id!r} already has an active turn")

        await self._turn_lock.acquire()
        try:
            async with self._state_lock:
                if self._state is not _RuntimeState.OPEN:
                    raise SessionRuntimeClosedError(f"Session {self.session_id!r} is not open")
            yield self.value
        finally:
            self._turn_lock.release()

    async def close(self) -> None:
        """Wait for the active turn, release resources once, and mark closed."""
        async with self._state_lock:
            if self._close_task is None:
                if self._state is _RuntimeState.CLOSED:
                    return
                self._state = _RuntimeState.CLOSING
                self._close_task = asyncio.create_task(self._close_once())
            close_task = self._close_task

        # Resource cleanup must survive cancellation of an individual host
        # request. Later close() calls await the same task and see its result.
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
    """Concurrency-safe registry of live session runtimes."""

    def __init__(self) -> None:
        self._runtimes: dict[str, SessionRuntime[T]] = {}
        self._lock = asyncio.Lock()
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    async def add(
        self,
        session_id: str,
        value: T,
        *,
        close: CloseCallback[T] | None = None,
    ) -> SessionRuntime[T]:
        async with self._lock:
            if self._closed:
                raise SessionRuntimeClosedError("Session runtime pool is closed")
            if session_id in self._runtimes:
                raise ValueError(f"Session {session_id!r} is already registered")
            runtime = SessionRuntime(session_id, value, close=close)
            self._runtimes[session_id] = runtime
            return runtime

    async def get(self, session_id: str) -> SessionRuntime[T]:
        async with self._lock:
            try:
                return self._runtimes[session_id]
            except KeyError:
                raise KeyError(f"Unknown live session {session_id!r}") from None

    async def ids(self) -> tuple[str, ...]:
        async with self._lock:
            return tuple(self._runtimes)

    async def remove(self, session_id: str) -> T:
        """Close and unregister one runtime, returning its host value."""
        runtime = await self.get(session_id)
        await runtime.close()
        async with self._lock:
            if self._runtimes.get(session_id) is runtime:
                del self._runtimes[session_id]
        return runtime.value

    async def close(self) -> None:
        """Close every registered runtime and reject future registrations."""
        async with self._lock:
            if self._close_task is None:
                self._closed = True
                runtimes = list(self._runtimes.values())
                self._close_task = asyncio.create_task(self._close_all(runtimes))
            close_task = self._close_task

        await asyncio.shield(close_task)

    async def _close_all(self, runtimes: list[SessionRuntime[T]]) -> None:
        results = await asyncio.gather(
            *(runtime.close() for runtime in runtimes),
            return_exceptions=True,
        )
        async with self._lock:
            self._runtimes.clear()

        failures = [result for result in results if isinstance(result, BaseException)]
        if failures:
            raise BaseExceptionGroup("Failed to close one or more session runtimes", failures)
