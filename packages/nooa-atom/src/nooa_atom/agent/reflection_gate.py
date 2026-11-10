# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private, in-memory admission policy for automatic post-Done reflection.

No timers or background work. The session supplies its increasing successful
settlement ordinal; nested generations/cells/maintenance never call this gate.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable


def reflection_min_turns(value: object) -> int:
    """Positive exact integer; reject bool and coercions (including 5.0)."""
    if type(value) is not int or value < 1:
        raise ValueError("experimental_reflection_min_turns must be an integer >= 1 (not bool)")
    return value


def reflection_min_seconds(value: object) -> float:
    """Finite nonnegative int/float seconds; reject bool, strings and coercions."""
    if type(value) not in (int, float):
        raise ValueError(
            "experimental_reflection_min_seconds must be finite seconds >= 0 (not bool)"
        )
    try:
        seconds = float(value)
    except OverflowError:
        raise ValueError(
            "experimental_reflection_min_seconds must be finite seconds >= 0"
        ) from None
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("experimental_reflection_min_seconds must be finite seconds >= 0")
    return seconds


class _ReflectionGate:
    """Count AND elapsed time, reset on admission rather than successful completion.

    Absolute monotonic anchors/counters are process-local and never snapshotted.
    start_session is only the owner lifecycle hook (creation/restore), not a
    model-facing reset facility. A clock can be injected for deterministic tests.
    """

    __nosnapshot__ = True

    def __init__(
        self, *, min_turns=50, min_seconds=3600.0, clock: Callable[[], float] | None = None
    ):
        self.min_turns = reflection_min_turns(min_turns)
        self.min_seconds = reflection_min_seconds(min_seconds)
        self._clock = time.monotonic if clock is None else clock
        self.start_session()

    def _now(self) -> float:
        value = self._clock()
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("reflection clock must return finite monotonic seconds")
        return float(value)

    def start_session(self) -> None:
        self.anchor = self._now()
        self.turns = 0
        self._last_sequence = 0
        self._pending_done: int | None = None

    def settled(
        self, sequence: int, *, kind: str, method: str | None, committed: bool, ran: bool
    ) -> bool:
        """Record one ordered owner settlement; True identifies a Done opportunity.

        Duplicate/older ordinals cannot count again or invalidate a newer Done.
        Non-qualifying outcomes preserve the turn/time window, but cannot admit
        reflection based on a previous unconsumed Done opportunity.
        """
        if sequence <= self._last_sequence:
            return False
        self._last_sequence = sequence
        if (
            method != "handle"
            or not committed
            or not ran
            or kind not in {"done", "waiting", "need_input"}
        ):
            self._pending_done = None
            return False
        self.turns += 1
        self._pending_done = sequence if kind == "done" else None
        return kind == "done"

    def admit(self, sequence: int) -> bool:
        """Consume this Done opportunity once; atomically reset if both bounds hold.

        Skipped Done does not reset the window. It also cannot be retried later
        merely because time passed: another accepted Done is needed. Failure,
        timeout, cancellation and no-op after admission all consume the window.
        """
        if self._pending_done is None or self._pending_done != sequence:
            return False
        self._pending_done = None
        now = self._now()
        if self.turns < self.min_turns or now - self.anchor < self.min_seconds:
            return False
        self.turns = 0
        self.anchor = now
        return True
