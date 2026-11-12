# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shell-owned numeric calibration and reply decisions, no concrete imports.

Association and eligibility belong to adapters. The private participation factory
is not part of Adapter: exact built-ins opt in; subclasses retain send overrides.
"""

import math
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from ._observations import _AttemptWriter, _ObservationOwner
from .errors import FailureKind, LLMFailure


@dataclass(eq=False)
class _RouteCalibration:
    """Numeric EMA state. Seeding counts as a baseline, not a new observation."""

    _ratio: float
    _alpha: float
    _lock: Lock = field(default_factory=Lock, repr=False)

    def ratio(self) -> float:
        return self._ratio

    def calibrate(self, estimated: int) -> int:
        return int(estimated * self._ratio)

    def update(self, estimated: int, actual: int) -> None:
        """Private shell observation hook; never updates the legacy singleton."""
        if estimated <= 0 or actual <= 0:
            return
        observed = actual / estimated
        with self._lock:  # Numeric-only critical section, no I/O or callbacks.
            self._ratio = self._alpha * observed + (1 - self._alpha) * self._ratio


@dataclass(frozen=True, slots=True)
class _CalibrationAdapter:
    adapter: Any
    record: _RouteCalibration
    wire_owner: Any = None

    @property
    def capabilities(self):
        return self.adapter.capabilities

    def _decision(self, estimated, quality, reply_cap):
        """Numeric only. A full projection is an estimate, not tokenizer proof."""
        if type(reply_cap) is not int or reply_cap <= 0:
            return None  # An absent cap does not mean the server ceiling.
        facts = self.capabilities
        cap = min(reply_cap, facts.max_output_tokens or reply_cap)
        if (
            facts.context_window is not None
            and quality == "full"
            and type(estimated) is int
            and estimated >= 0
        ):
            ratio = self.record.ratio()
            try:
                if estimated >= facts.context_window:
                    raise LLMFailure(FailureKind.CONTEXT_WINDOW, "No estimated room for a reply")
                calibrated = estimated * max(1.0, ratio)
                if not math.isfinite(calibrated):
                    raise LLMFailure(FailureKind.CONTEXT_WINDOW, "No estimated room for a reply")
                room = facts.context_window - math.ceil(calibrated)
            except (OverflowError, ValueError):
                raise LLMFailure(
                    FailureKind.CONTEXT_WINDOW, "No estimated room for a reply"
                ) from None
            if room <= 0:
                raise LLMFailure(FailureKind.CONTEXT_WINDOW, "No estimated room for a reply")
            cap = min(cap, room)
        return cap

    def _update(self, owner):
        observation = owner.writer.record
        estimated = observation.raw_input_estimate
        usage = observation.usage
        actual = usage.input_tokens if usage is not None else None
        if (
            observation.finished
            and observation.response_received
            and estimated is not None
            and actual is not None
            and estimated > 0
            and actual > 0
        ):
            self.record.update(estimated, actual)

    def send_sync(self, request):
        owner = _ObservationOwner(_AttemptWriter())
        try:
            return self.adapter._send_sync(request, _owner=owner, _decision=self._decision)
        finally:
            self._update(owner)

    async def send(self, request):
        owner = _ObservationOwner(_AttemptWriter())
        try:
            return await self.adapter._send(
                request, _owner=owner, _decision=self._decision, _wire_owner=self.wire_owner
            )
        finally:
            self._update(owner)


def _calibration_adapter(adapter, wire_owner=None):
    factory = getattr(adapter, "_numeric_participation", None)
    participant = factory() if callable(factory) else None
    if isinstance(participant, _CalibrationAdapter) and wire_owner is not None:
        from dataclasses import replace

        participant = replace(participant, wire_owner=wire_owner)
    return participant if participant is not None else adapter
