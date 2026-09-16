# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefix-only writer that aborts once a serialization budget is exhausted."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

_STRING_CHUNK_CHARS = 256


class SerializationLimitReached(Exception):
    """Raised when a :class:`LimitedWriter` cannot accept a complete write."""


class LimitedWriter:
    """Collect at most ``limit`` characters, then abort the producer.

    Unlike a truncating stream that retains a rolling tail, this writer raises
    immediately on overflow. Streaming producers therefore stop traversing the
    source object as soon as the trace preview is full.
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


def dump_json_bounded(
    value: Any,
    writer: LimitedWriter,
    *,
    default: Callable[[Any], Any],
) -> None:
    """Write default-format JSON without materializing unbounded string tokens.

    The output matches :func:`json.dumps` with its default options. String values
    are escaped in small fragments so the largest temporary allocation is bounded;
    container traversal still stops immediately when ``writer`` reaches its limit.
    """
    markers: dict[int, Any] = {}

    def write_string(string: str) -> None:
        """Write one JSON string as independently escaped bounded fragments."""
        writer.write('"')
        for start in range(0, len(string), _STRING_CHUNK_CHARS):
            # JSON string escaping is character-local, so concatenating the escaped
            # interiors of separately quoted fragments is identical to encoding the
            # complete string at once.
            encoded = json.dumps(string[start : start + _STRING_CHUNK_CHARS])
            writer.write(encoded[1:-1])
        writer.write('"')

    def mark(container: Any) -> int:
        """Track a value being traversed and reject circular references."""
        marker = id(container)
        if marker in markers:
            raise ValueError("Circular reference detected")
        markers[marker] = container
        return marker

    def write_value(item: Any) -> None:
        """Write one value using the standard encoder's default representation."""
        if isinstance(item, str):
            write_string(item)
        elif item is None:
            writer.write("null")
        elif item is True:
            writer.write("true")
        elif item is False:
            writer.write("false")
        elif isinstance(item, int):
            writer.write(int.__repr__(item))
        elif isinstance(item, float):
            writer.write(json.dumps(item))
        elif isinstance(item, (list, tuple)):
            marker = mark(item)
            try:
                writer.write("[")
                for index, child in enumerate(item):
                    if index:
                        writer.write(", ")
                    write_value(child)
                writer.write("]")
            finally:
                markers.pop(marker, None)
        elif isinstance(item, dict):
            marker = mark(item)
            try:
                writer.write("{")
                first = True
                for key, child in item.items():
                    if isinstance(key, str):
                        string_key = key
                    elif isinstance(key, float):
                        string_key = json.dumps(key)
                    elif key is True:
                        string_key = "true"
                    elif key is False:
                        string_key = "false"
                    elif key is None:
                        string_key = "null"
                    elif isinstance(key, int):
                        string_key = int.__repr__(key)
                    else:
                        raise TypeError(
                            "keys must be str, int, float, bool or None, "
                            f"not {key.__class__.__name__}"
                        )
                    if first:
                        first = False
                    else:
                        writer.write(", ")
                    write_string(string_key)
                    writer.write(": ")
                    write_value(child)
                writer.write("}")
            finally:
                markers.pop(marker, None)
        else:
            marker = mark(item)
            try:
                write_value(default(item))
            finally:
                markers.pop(marker, None)

    write_value(value)
