# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure functions on ``Text``: ``cut`` a whole string or bytes, ``merge`` two ``Text``s.

They read no stream state and no live objects; they share only the nodes
(``_model``) and the UTF-8 cut points (``_codec``).
"""

from __future__ import annotations

from typing import Literal

from nooa.snapshot._codec import complete_prefix, encode_text, safe_end, safe_start
from nooa.snapshot._model import Text

__all__ = ["cut", "merge"]


def cut(
    data: str | bytes,
    *,
    head: int,
    tail: int,
    unit: Literal["chars", "bytes"] | None = None,
) -> Text:
    """``data``, already whole in memory, as a ``Text`` keeping its first ``head`` and last ``tail`` units.

    - ``data``: a ``str`` or ``bytes`` (``bytearray`` and ``memoryview``
      are read as bytes).
    - ``head``, ``tail``: the units kept from the start and from the end.
    - ``unit`` (default: ``"chars"`` for a ``str``, ``"bytes"`` for bytes):
      ``"bytes"`` on a ``str`` counts and keeps its UTF-8 encoding (as a
      bytes ``TextStream`` would); ``"chars"`` on bytes raises ``TypeError``.

    The same ``Text`` a ``TextStream(unit=unit, head=head, tail=tail)``
    returns after one ``write(data)`` and ``close()``, without the stream:
    complete when ``data`` fits in ``head + tail``; else cut with reason
    ``"max_length"``, a bytes cut on whole code points. Nothing is spilled
    (``full`` is ``None``). Render it with ``render_output`` (or
    ``render_python``). Raises ``ValueError`` for a negative ``head`` or
    ``tail`` or an unknown ``unit``.

    >>> from nooa.snapshot import cut, render_output
    >>> print(render_output(cut("0123456789", head=3, tail=2)))
    012
    [... 5 chars omitted; full output not kept]
    89
    """
    if head < 0 or tail < 0:
        raise ValueError("head and tail must be non-negative")
    if isinstance(data, str):
        if unit is None or unit == "chars":
            return _cut(data, head, tail, "chars")
        if unit == "bytes":
            return _cut(encode_text(data), head, tail, "bytes")
    elif isinstance(data, (bytes, bytearray, memoryview)):
        if unit is None or unit == "bytes":
            return _cut(bytes(data), head, tail, "bytes")
        if unit == "chars":
            raise TypeError("cut(unit='chars') takes str; cut bytes with unit='bytes'")
    else:
        raise TypeError(f"cut takes str or bytes, not {type(data).__name__}")
    raise ValueError(f"unit must be 'chars' or 'bytes', got {unit!r}")


def _cut(data: str | bytes, head: int, tail: int, unit: Literal["chars", "bytes"]) -> Text:
    if len(data) <= head + tail:
        return Text(head=data, length=len(data), unit=unit)
    kept_head, kept_tail = data[:head], data[len(data) - tail :] if tail else data[:0]
    if isinstance(kept_head, bytes) and isinstance(kept_tail, bytes):  # whole code points only
        kept_head = kept_head[: complete_prefix(kept_head)]
        kept_tail = kept_tail[safe_start(kept_tail, 0) :]
    return Text(head=kept_head, tail=kept_tail, length=len(data), unit=unit, reason="max_length")


def merge(a: Text, b: Text, head: int, tail: int) -> Text:
    """``a`` followed by ``b`` as one ``Text`` keeping at most ``head`` + ``tail`` units.

    For two batches of output read one after the other (progress deltas).

    - ``a``, ``b``: two ``Text`` of the same unit.
    - ``head``, ``tail``: the units the result may keep from its start and
      its end.

    The result's ``length`` is ``a.length + b.length``. What either input
    already omitted stays omitted: the head comes only from what was kept
    before the first gap, the tail only from what was kept after the last
    one. Counts are exact in the inputs' unit (bytes stay bytes). A bytes
    cut keeps whole code points and never moves across the joint of two
    inputs' parts. ``reason``: ``None`` when everything fits, else the
    first input's reason, or ``"max_length"``. ``full``: ``b.full`` when
    both inputs spilled to the same path, else ``None``.

    Raises ``ValueError`` when the units differ or ``head`` / ``tail`` is
    negative.

    >>> from nooa.snapshot import Text, merge
    >>> merge(Text(head="hello", length=5, unit="chars"), Text(head=" world", length=6, unit="chars"), 3, 3)
    Text(head='hel', tail='rld', length=11, unit='chars', reason='max_length', full=None)
    """
    if a.unit != b.unit:
        raise ValueError(f"cannot merge {a.unit} with {b.unit}")
    if head < 0 or tail < 0:
        raise ValueError("head and tail must be non-negative")
    length = a.length + b.length
    same_file = a.full is not None and b.full is not None and a.full.path == b.full.path
    full = b.full if same_file else None
    # The joints: where one input's part ends and the next begins. A bytes
    # cut never moves across one, so each part is cut on its own code points.
    gapless = a.complete and b.complete
    if gapless:
        whole = a.head + b.head  # type: ignore[operator]
        if len(whole) <= head + tail:
            return Text(head=whole, length=length, unit=a.unit, full=full)
        left = right = whole
        left_joints = right_joints = (len(a.head),)
    else:
        left = a.head if not a.complete else a.head + b.head  # type: ignore[operator]
        right = b.tail if not b.complete else a.tail + b.head  # type: ignore[operator]
        left_joints = () if not a.complete else (len(a.head),)
        right_joints = () if not b.complete else (len(a.tail),)
    kept_head = left[: _head_end(left, head, left_joints)]
    low = len(kept_head) if gapless else 0  # the tail starts after the kept head
    start = _tail_start(right, max(low, len(right) - tail), (*right_joints, low))
    return Text(
        head=kept_head,
        tail=right[start:],
        length=length,
        unit=a.unit,
        reason=a.reason or b.reason or "max_length",
        full=full,
    )


def _head_end(text: str | bytes, end: int, joints: tuple[int, ...]) -> int:
    """Where a head of at most ``end`` units of ``text`` stops (bytes: on a code point, not before a joint)."""
    if isinstance(text, str):
        return min(end, len(text))
    return max([safe_end(text, end), *(joint for joint in joints if joint <= end)])


def _tail_start(text: str | bytes, start: int, joints: tuple[int, ...]) -> int:
    """Where a tail from ``start`` begins (bytes: on a code point, not past a joint)."""
    if isinstance(text, str):
        return start
    return min([safe_start(text, start), *(joint for joint in joints if joint >= start)])
