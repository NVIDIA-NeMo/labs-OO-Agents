# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefix-only writer that aborts once a serialization budget is exhausted."""

from __future__ import annotations


class SerializationLimitReached(Exception):
    """Raised when a :class:`LimitedWriter` cannot accept a complete write."""


class LimitedWriter:
    """Collect at most ``limit`` characters, then abort the producer.

    Unlike a truncating stream that retains a rolling tail, this writer raises
    immediately on overflow. Producers such as :func:`json.dump` therefore stop
    traversing the source object as soon as the trace preview is full.
    """

    def __init__(self, limit: int) -> None:
        if limit <= 0:
            raise ValueError(f"LimitedWriter limit must be > 0, got {limit}")
        self._limit = limit
        self._parts: list[str] = []
        self._chars_written = 0

    def write(self, value: str) -> int:
        """Store ``value`` or its remaining prefix, raising on overflow."""
        remaining = self.remaining
        if len(value) <= remaining:
            self._parts.append(value)
            self._chars_written += len(value)
            return len(value)

        if remaining:
            self._parts.append(value[:remaining])
            self._chars_written += remaining
        raise SerializationLimitReached

    @property
    def remaining(self) -> int:
        """Number of characters that can still be retained."""
        return self._limit - self._chars_written

    @property
    def chars_written(self) -> int:
        """Number of prefix characters retained so far."""
        return self._chars_written

    def getvalue(self) -> str:
        """Return the retained serialization prefix."""
        return "".join(self._parts)
