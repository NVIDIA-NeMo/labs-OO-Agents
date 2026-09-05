# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Process-local admission control for asynchronous LLM provider calls.

The registry in this module is deliberately process local.  It coordinates
``UnifiedLLM`` clients and event loops in one Python process, but it is not a
distributed rate limiter or inference scheduler.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import re
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable
from urllib.parse import urlsplit, urlunsplit

_GROUP_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,127}$")

AdmissionOutcome = Literal[
    "immediate",
    "admitted_after_wait",
    "call_cap",
    "timeout",
    "cancelled",
]
AdmissionObserver = Callable[[dict[str, Any]], None]


class AdmissionError(Exception):
    """Base class for failures that happen before provider dispatch."""


class AdmissionTimeoutError(AdmissionError):
    """Raised when a provider attempt times out before admission.

    This intentionally does not inherit from :class:`TimeoutError`: the normal
    provider retry policy treats timeouts as retryable, while an admission
    timeout must be terminal for that attempt chain to avoid feeding an already
    congested queue.
    """


class AdmissionCallCapError(AdmissionError):
    """Raised before provider dispatch when an admission call budget is exhausted."""


class AdmissionUnavailableError(AdmissionError):
    """Raised when a configured external admission controller is unavailable."""


@runtime_checkable
class AdmissionPermit(Protocol):
    """An idempotently releasable slot returned by an admission controller."""

    def release(self) -> None:
        """Return temporary concurrency capacity to the controller."""


@runtime_checkable
class AdmissionController(Protocol):
    """Pluggable admission boundary for one asynchronous provider attempt.

    Implementations may coordinate one process, a process family, or a remote
    service. They report one observation per acquisition outcome and return a
    permit only when the provider attempt may be dispatched.
    """

    async def acquire(self, observer: AdmissionObserver) -> AdmissionPermit | None:
        """Acquire capacity or raise before provider dispatch."""


@dataclass(slots=True)
class _Waiter:
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future[None]
    queued_at: float
    queue_depth: int
    state: Literal["queued", "granted", "acquired", "cancelled"] = "queued"


class _Permit:
    """One idempotently releasable admission permit."""

    __slots__ = ("_group", "_released", "_release_lock")

    def __init__(self, group: _AdmissionGroup):
        self._group = group
        self._released = False
        self._release_lock = threading.Lock()

    def release(self) -> None:
        with self._release_lock:
            if self._released:
                return
            self._released = True
        self._group.release()


class _AdmissionGroup:
    """A FIFO permit pool that can serve waiters from multiple event loops."""

    def __init__(self, identity: str, display_name: str, max_in_flight: int):
        self.identity = identity
        self.display_name = display_name
        self.max_in_flight = max_in_flight
        self._lock = threading.Lock()
        self._active = 0
        self._waiters: deque[_Waiter] = deque()

    @property
    def active(self) -> int:
        with self._lock:
            return self._active

    @property
    def queued(self) -> int:
        with self._lock:
            return self._queued_count_locked()

    def _queued_count_locked(self) -> int:
        return sum(waiter.state == "queued" for waiter in self._waiters)

    async def acquire(
        self,
        *,
        queue_timeout: float | None,
        observer: AdmissionObserver,
    ) -> _Permit:
        started = time.perf_counter()
        loop = asyncio.get_running_loop()

        with self._lock:
            if self._active < self.max_in_flight and not self._waiters:
                self._active += 1
                immediate = True
                waiter = None
                queue_depth = 0
            else:
                immediate = False
                future = loop.create_future()
                queue_depth = self._queued_count_locked() + 1
                waiter = _Waiter(loop, future, started, queue_depth)
                self._waiters.append(waiter)

        if immediate:
            observer(self._observation("immediate", False, 0.0, queue_depth))
            return _Permit(self)

        assert waiter is not None
        try:
            if queue_timeout is None:
                await asyncio.shield(waiter.future)
            else:
                async with asyncio.timeout(queue_timeout):
                    await asyncio.shield(waiter.future)
        except TimeoutError as error:
            self._cancel_waiter(waiter)
            wait_s = time.perf_counter() - started
            observer(self._observation("timeout", True, wait_s, waiter.queue_depth))
            raise AdmissionTimeoutError(
                f"Timed out after {queue_timeout:g}s waiting for LLM admission "
                f"group {self.display_name!r} (max_in_flight={self.max_in_flight})"
            ) from error
        except asyncio.CancelledError:
            self._cancel_waiter(waiter)
            wait_s = time.perf_counter() - started
            observer(self._observation("cancelled", True, wait_s, waiter.queue_depth))
            raise

        with self._lock:
            if waiter.state != "granted":
                raise RuntimeError(
                    f"Invalid admission waiter state after wake-up: {waiter.state!r}"
                )
            waiter.state = "acquired"

        wait_s = time.perf_counter() - started
        observer(self._observation("admitted_after_wait", True, wait_s, waiter.queue_depth))
        return _Permit(self)

    def _observation(
        self,
        outcome: AdmissionOutcome,
        queued: bool,
        wait_s: float,
        queue_depth: int,
    ) -> dict[str, Any]:
        return {
            "group": self.display_name,
            "outcome": outcome,
            "queued": queued,
            "wait_s": round(wait_s, 6),
            "queue_depth": queue_depth,
            "max_in_flight": self.max_in_flight,
        }

    def _cancel_waiter(self, waiter: _Waiter) -> None:
        release_grant = False
        with self._lock:
            if waiter.state == "queued":
                waiter.state = "cancelled"
                # Do not retain cancelled/expired waiters behind a long-lived
                # provider call.  ``remove`` is bounded by the queue length and
                # keeps cancellation storms from becoming a memory backlog.
                try:
                    self._waiters.remove(waiter)
                except ValueError:
                    pass
            elif waiter.state == "granted":
                # Capacity was transferred to this waiter, but cancellation or
                # timeout won before its task claimed the grant.
                waiter.state = "cancelled"
                release_grant = True
            elif waiter.state in ("acquired", "cancelled"):
                return
        if release_grant:
            self.release()

    def release(self) -> None:
        with self._lock:
            if self._active <= 0:
                raise RuntimeError("Admission permit accounting underflow")

            while self._waiters:
                waiter = self._waiters.popleft()
                if waiter.state != "queued":
                    continue
                waiter.state = "granted"
                # The active slot transfers to the waiter, so _active does not
                # change. Delivery happens on the waiter's own event loop.
                try:
                    waiter.loop.call_soon_threadsafe(self._deliver_grant, waiter)
                except RuntimeError:
                    # A closed loop cannot accept the grant. Skip this waiter
                    # and transfer the same slot to the next one.
                    waiter.state = "cancelled"
                    continue
                return

            self._active -= 1

    def _deliver_grant(self, waiter: _Waiter) -> None:
        release_grant = False
        with self._lock:
            if waiter.state != "granted":
                return
            if waiter.future.cancelled():
                waiter.state = "cancelled"
                release_grant = True
            elif not waiter.future.done():
                waiter.future.set_result(None)
        if release_grant:
            self.release()


_groups_lock = threading.RLock()
_groups: dict[str, _AdmissionGroup] = {}


def _reset_admission_groups_after_fork() -> None:
    """Give a forked child clean locks and process-local capacity accounting."""
    global _groups_lock, _groups
    _groups_lock = threading.RLock()
    _groups = {}


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_admission_groups_after_fork)


def _validated_limit(max_in_flight: int | None) -> int | None:
    if max_in_flight is None:
        return None
    if isinstance(max_in_flight, bool) or not isinstance(max_in_flight, int):
        raise TypeError("max_in_flight must be a positive integer or None")
    if max_in_flight <= 0:
        raise ValueError("max_in_flight must be greater than zero")
    return max_in_flight


def _validated_timeout(queue_timeout: float | None) -> float | None:
    if queue_timeout is None:
        return None
    if isinstance(queue_timeout, bool) or not isinstance(queue_timeout, int | float):
        raise TypeError("queue_timeout must be a positive number or None")
    value = float(queue_timeout)
    if value <= 0 or not math.isfinite(value):
        raise ValueError("queue_timeout must be finite and greater than zero")
    return value


def _named_group_identity(label: str) -> tuple[str, str]:
    if not isinstance(label, str):
        raise TypeError("concurrency_group must be a string or None")
    if not _GROUP_LABEL_RE.fullmatch(label):
        raise ValueError(
            "concurrency_group must be 1-128 characters and contain only "
            "letters, digits, '.', '_', ':', '/', or '-'"
        )
    return f"named:{label}", label


def _endpoint_group_identity(api_base: str) -> tuple[str, str]:
    if not isinstance(api_base, str) or not api_base:
        raise TypeError("api_base must be a non-empty string for inferred admission groups")

    parts = urlsplit(api_base)
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        raise ValueError(
            "max_in_flight without concurrency_group requires an absolute HTTP(S) api_base"
        )

    host = parts.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    port = parts.port
    if port is not None and not (
        (parts.scheme.lower() == "http" and port == 80)
        or (parts.scheme.lower() == "https" and port == 443)
    ):
        host = f"{host}:{port}"
    path = parts.path.rstrip("/")
    normalized = urlunsplit((parts.scheme.lower(), host, path, "", ""))
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    return f"endpoint:{digest}", f"endpoint:{digest[:12]}"


def _get_or_create_group(
    identity: str,
    display_name: str,
    max_in_flight: int | None,
) -> _AdmissionGroup | None:
    with _groups_lock:
        existing = _groups.get(identity)
        if max_in_flight is None:
            return existing
        if existing is not None:
            if existing.max_in_flight != max_in_flight:
                raise ValueError(
                    f"Conflicting max_in_flight for admission group {display_name!r}: "
                    f"existing={existing.max_in_flight}, requested={max_in_flight}"
                )
            return existing
        group = _AdmissionGroup(identity, display_name, max_in_flight)
        _groups[identity] = group
        return group


class AdmissionPolicy:
    """Per-client view of a shared, process-local admission group."""

    def __init__(
        self,
        *,
        max_in_flight: int | None,
        concurrency_group: str | None,
        queue_timeout: float | None,
        api_base: str | None,
    ):
        self.max_in_flight = _validated_limit(max_in_flight)
        self.queue_timeout = _validated_timeout(queue_timeout)

        if concurrency_group is not None:
            self.identity, self.display_name = _named_group_identity(concurrency_group)
        elif api_base is not None:
            try:
                self.identity, self.display_name = _endpoint_group_identity(api_base)
            except (TypeError, ValueError):
                if self.max_in_flight is not None or self.queue_timeout is not None:
                    raise
                # Preserve existing behavior for an unbounded client with an
                # unusual provider-specific base URL. It cannot participate in
                # endpoint-inferred admission until given a normal HTTP(S) URL.
                self.identity = None
                self.display_name = None
        elif self.max_in_flight is not None:
            raise ValueError("max_in_flight requires concurrency_group or api_base")
        elif self.queue_timeout is not None:
            raise ValueError("queue_timeout requires concurrency_group or api_base")
        else:
            self.identity = None
            self.display_name = None

        if self.identity is not None and self.max_in_flight is not None:
            _get_or_create_group(self.identity, self.display_name or "", self.max_in_flight)

    async def acquire(self, observer: AdmissionObserver) -> _Permit | None:
        """Acquire capacity, or return ``None`` when no group has a configured limit."""
        if self.identity is None:
            return None
        group = _get_or_create_group(
            self.identity,
            self.display_name or "",
            self.max_in_flight,
        )
        if group is None:
            return None
        return await group.acquire(queue_timeout=self.queue_timeout, observer=observer)


def _reset_admission_groups_for_tests() -> None:
    """Clear process-global groups; private helper for isolated unit tests."""
    with _groups_lock:
        if any(group.active or group.queued for group in _groups.values()):
            raise RuntimeError("Cannot reset admission groups while calls are active or queued")
        _groups.clear()


__all__ = [
    "AdmissionCallCapError",
    "AdmissionController",
    "AdmissionError",
    "AdmissionPermit",
    "AdmissionPolicy",
    "AdmissionTimeoutError",
    "AdmissionUnavailableError",
]
