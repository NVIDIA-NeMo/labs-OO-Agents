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
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
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
    "unavailable",
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


_current_controller: ContextVar[AdmissionController | None] = ContextVar(
    "nooa_current_admission_controller",
    default=None,
)


@contextmanager
def _admission_controller_scope(controller: AdmissionController) -> Iterator[None]:
    """Select a controller for provider attempts made in this async context."""
    token = _current_controller.set(controller)
    try:
        yield
    finally:
        _current_controller.reset(token)


def _current_admission_controller() -> AdmissionController | None:
    """Return the controller selected by the nearest admission wrapper."""
    return _current_controller.get()


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
        self._queued = 0
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
        return self._queued

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
                self._queued += 1

        if immediate:
            return self._permit_with_observation(
                observer,
                self._observation("immediate", False, 0.0, queue_depth),
            )

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
            self._observe_terminal(
                observer,
                self._observation("timeout", True, wait_s, waiter.queue_depth),
            )
            raise AdmissionTimeoutError(
                f"Timed out after {queue_timeout:g}s waiting for LLM admission "
                f"group {self.display_name!r} (max_in_flight={self.max_in_flight})"
            ) from error
        except asyncio.CancelledError:
            self._cancel_waiter(waiter)
            wait_s = time.perf_counter() - started
            self._observe_terminal(
                observer,
                self._observation("cancelled", True, wait_s, waiter.queue_depth),
            )
            raise

        with self._lock:
            if waiter.state != "granted":
                raise RuntimeError(
                    f"Invalid admission waiter state after wake-up: {waiter.state!r}"
                )
            waiter.state = "acquired"

        wait_s = time.perf_counter() - started
        return self._permit_with_observation(
            observer,
            self._observation("admitted_after_wait", True, wait_s, waiter.queue_depth),
        )

    def _permit_with_observation(
        self,
        observer: AdmissionObserver,
        observation: dict[str, Any],
    ) -> _Permit:
        """Return a permit without leaking its slot if observation fails."""
        permit = _Permit(self)
        try:
            observer(observation)
        except BaseException:
            permit.release()
            raise
        return permit

    @staticmethod
    def _observe_terminal(
        observer: AdmissionObserver,
        observation: dict[str, Any],
    ) -> None:
        """Report a terminal wait outcome without replacing its exception."""
        try:
            observer(observation)
        except BaseException:
            pass

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
                self._queued -= 1
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
                self._queued -= 1
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
    """Per-wrapper view of a shared, process-local admission group."""

    def __init__(
        self,
        *,
        max_in_flight: int,
        concurrency_group: str | None,
        queue_timeout: float | None,
        api_base: str | None,
    ):
        self.max_in_flight = _validated_limit(max_in_flight)
        if self.max_in_flight is None:
            raise ValueError("max_in_flight is required for process-local admission")
        self.queue_timeout = _validated_timeout(queue_timeout)

        if concurrency_group is not None:
            self.identity, self.display_name = _named_group_identity(concurrency_group)
        elif api_base is not None:
            self.identity, self.display_name = _endpoint_group_identity(api_base)
        else:
            raise ValueError("max_in_flight requires concurrency_group or api_base")

        _get_or_create_group(self.identity, self.display_name, self.max_in_flight)

    async def acquire(self, observer: AdmissionObserver) -> _Permit:
        """Acquire process-local capacity for one provider attempt."""
        group = _get_or_create_group(
            self.identity,
            self.display_name,
            self.max_in_flight,
        )
        assert group is not None
        return await group.acquire(queue_timeout=self.queue_timeout, observer=observer)


@dataclass(frozen=True, slots=True)
class AdmissionControlConfig:
    """Explicit per-use admission configuration.

    Supply either an application-owned ``controller`` or a positive
    ``max_in_flight`` for the built-in process-local controller. Local groups
    may be named explicitly; otherwise the wrapped client's HTTP endpoint is
    normalized into an opaque group identity when the wrapper is created.
    """

    max_in_flight: int | None = None
    concurrency_group: str | None = None
    queue_timeout: float | None = None
    controller: AdmissionController | None = None

    def __post_init__(self) -> None:
        if self.controller is not None:
            if any(
                value is not None
                for value in (self.max_in_flight, self.concurrency_group, self.queue_timeout)
            ):
                raise ValueError(
                    "controller cannot be combined with max_in_flight, "
                    "concurrency_group, or queue_timeout"
                )
            if not isinstance(self.controller, AdmissionController):
                raise TypeError("controller must implement AdmissionController")
            return

        if self.max_in_flight is None:
            raise ValueError(
                "max_in_flight is required when an admission controller is not supplied"
            )
        _validated_limit(self.max_in_flight)
        _validated_timeout(self.queue_timeout)
        if self.concurrency_group is not None:
            _named_group_identity(self.concurrency_group)

    def _controller_for(self, llm: Any) -> AdmissionController:
        """Build or return the controller applied to one wrapped LLM."""
        if self.controller is not None:
            return self.controller

        client_config = getattr(llm, "config", None)
        api_base = None
        if isinstance(client_config, dict):
            api_base = client_config.get("api_base") or client_config.get("base_url")
        assert self.max_in_flight is not None
        return AdmissionPolicy(
            max_in_flight=self.max_in_flight,
            concurrency_group=self.concurrency_group,
            queue_timeout=self.queue_timeout,
            api_base=api_base,
        )


def _reset_admission_groups_for_tests() -> None:
    """Clear process-global groups; private helper for isolated unit tests."""
    with _groups_lock:
        if any(group.active or group.queued for group in _groups.values()):
            raise RuntimeError("Cannot reset admission groups while calls are active or queued")
        _groups.clear()


__all__ = [
    "AdmissionCallCapError",
    "AdmissionControlConfig",
    "AdmissionController",
    "AdmissionError",
    "AdmissionPermit",
    "AdmissionPolicy",
    "AdmissionTimeoutError",
    "AdmissionUnavailableError",
]
