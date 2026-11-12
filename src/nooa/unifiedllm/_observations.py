# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private opt-in numeric handoff, not accounting or an Adapter protocol.

Each scope owns a bounded collector. Nested scopes restore the parent; independent
async callers must enter their own scope (ContextVars intentionally inherit into
child tasks). Attempts capture their owner before preparation/awaits, never look
up the current collector at delivery, and replace one immutable record in place.
No callbacks, payloads, route identities or exceptions are retained.

``transmission_started`` means entry into the adapter's transmit step, not proof
of admission, a wire send or billing. Cancellation may leave shielded provider
work alive: a finished record with no delivered response remains unknown. This
foundation does not settle late usage or make a billing-completeness guarantee.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Literal


@dataclass(frozen=True, slots=True)
class _NumericUsage:
    # None is unknown, distinct from a provider's explicit zero. Breakdowns
    # are not additions to input/output; total may be normalized from both.
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    cost_source: Literal["unknown", "provider", "response_metadata"] = "unknown"


@dataclass(frozen=True, slots=True)
class _AttemptObservation:
    raw_input_estimate: int | None = None
    transmission_started: bool = False
    response_received: bool = False
    usage: _NumericUsage | None = None
    finished: bool = False


class _ObservationCollector:
    """Invocation-local storage; snapshots never expose the mutable list."""

    def __init__(self, limit: int):
        self._limit = limit
        self._records: list[_AttemptObservation] = []
        self._dropped = 0

    @property
    def records(self) -> tuple[_AttemptObservation, ...]:
        return tuple(self._records)

    @property
    def dropped_attempts(self) -> int:
        return self._dropped

    def _start(self, writer=None):
        if len(self._records) >= self._limit:
            self._dropped += 1
            return None
        index = len(self._records)
        writer = writer if writer is not None else _AttemptWriter()
        self._records.append(writer.record)
        writer._collector, writer._index = self, index
        return writer


class _AttemptWriter:
    def __init__(self):
        self._collector = None
        self._index = None
        self.record = _AttemptObservation()

    def _replace(self, **changes):
        self.record = replace(self.record, **changes)
        if self._collector is not None:
            self._collector._records[self._index] = self.record

    def projected(self, estimate: int | None):
        self._replace(
            raw_input_estimate=estimate if type(estimate) is int and estimate >= 0 else None
        )

    def transmitting(self):
        self._replace(transmission_started=True)

    def delivered(self, usage: _NumericUsage | None):
        self._replace(response_received=True, usage=usage)

    def finish(self):
        self._replace(finished=True)


@dataclass(frozen=True, slots=True)
class _ObservationOwner:
    """Explicit one-send ownership, never inherited through a ContextVar.

    The shell consumes this one numeric slot after the send exits. An external
    collector retains the SAME immutable records if it has room; its cap cannot
    discard the owner's slot. No arbitrary callback or adapter protocol hook.
    """

    writer: _AttemptWriter


_current_collector: ContextVar[_ObservationCollector | None] = ContextVar(
    "_numeric_wire_observations", default=None
)


@contextmanager
def _collect_observations(*, limit: int = 256):
    """Explicit private scope, bounded even if a caller retries indefinitely."""
    if type(limit) is not int or not 1 <= limit <= 256:
        raise ValueError("observation limit must be an integer in [1, 256]")
    collector = _ObservationCollector(limit)
    token = _current_collector.set(collector)
    try:
        yield collector
    finally:
        _current_collector.reset(token)


@contextmanager
def _observe_attempt(owner: _ObservationOwner | None = None):
    collector = _current_collector.get()
    writer = owner.writer if owner is not None else None
    if collector is not None:
        collected = collector._start(writer)
        if owner is None:
            writer = collected
    try:
        yield writer
    finally:
        if writer is not None:
            writer.finish()
