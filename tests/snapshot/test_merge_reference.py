# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``merge`` equals a direct two-loop reference implementation (below) on random deltas."""

from __future__ import annotations

import random

from nooa.snapshot import Spill, Text, merge
from nooa.snapshot._codec import safe_end, safe_start

_BYTES = (
    b"a",
    b"z\n",
    "\u00e9".encode(),
    "\u20ac".encode(),
    "\U0001f600".encode(),
    b"\xff",
    b"\x80",
)
_CHARS = ("a", "z\n", "\u00e9", "\u20ac", "\U0001f600", "\ud800")


def _reference(a: Text, b: Text, head: int, tail: int) -> Text:
    """The pre-slicing ``merge`` (8da0139b2): walk the kept parts, ``None`` at each gap."""
    unit = a.unit
    pieces: list = []
    for text in (a, b):
        pieces += [text.head, None, text.tail] if not text.complete else [text.head, text.tail]
    pieces = [piece for piece in pieces if piece is None or len(piece)]
    length = a.length + b.length
    same = a.full is not None and b.full is not None and a.full.path == b.full.path
    full = b.full if same else None
    empty = b"" if unit == "bytes" else ""
    if None not in pieces and sum(len(piece) for piece in pieces) <= head + tail:
        return Text(head=empty.join(pieces), length=length, unit=unit, full=full)
    head_parts: list = []
    room, index = head, 0
    while index < len(pieces) and room > 0:
        piece = pieces[index]
        if piece is None:
            break
        if len(piece) <= room:
            head_parts.append(piece)
            room -= len(piece)
            index += 1
            continue
        cut = safe_end(piece, room) if isinstance(piece, bytes) else room
        head_parts.append(piece[:cut])
        pieces[index] = piece[cut:]
        break
    tail_parts: list = []
    room, last = tail, len(pieces) - 1
    while last >= index and room > 0:
        piece = pieces[last]
        if piece is None:
            break
        if len(piece) <= room:
            tail_parts.insert(0, piece)
            room -= len(piece)
            last -= 1
            continue
        start = len(piece) - room
        tail_parts.insert(
            0, piece[safe_start(piece, start) if isinstance(piece, bytes) else start :]
        )
        break
    return Text(
        head=empty.join(head_parts),
        tail=empty.join(tail_parts),
        length=length,
        unit=unit,
        reason=a.reason or b.reason or "max_length",
        full=full,
    )


def _random_text(rng: random.Random, unit: str) -> Text:
    """A consistent delta: whole or cut, with any fragments (stray continuation bytes too)."""
    frags = _BYTES if unit == "bytes" else _CHARS
    empty = b"" if unit == "bytes" else ""

    def part() -> str | bytes:
        return empty.join(rng.choice(frags) for _ in range(rng.randrange(0, 5)))  # type: ignore[arg-type]

    full = rng.choice((None, None, Spill("/tmp/a", 4, 9), Spill("/tmp/b", 9, 9)))
    if rng.random() < 0.5:
        whole = part()
        return Text(head=whole, length=len(whole), unit=unit, full=full)  # type: ignore[arg-type]
    head, tail = part(), part()
    length = len(head) + len(tail) + rng.randrange(0, 4)
    reason = rng.choice(("max_length", "max_length", "error"))
    return Text(head=head, tail=tail, length=length, unit=unit, reason=reason, full=full)  # type: ignore[arg-type]


def test_merge_equals_the_reference_on_random_deltas() -> None:
    rng = random.Random(515)
    for _ in range(20_000):
        unit = rng.choice(("bytes", "chars"))
        a, b = _random_text(rng, unit), _random_text(rng, unit)
        head, tail = rng.randrange(0, 11), rng.randrange(0, 11)
        assert merge(a, b, head, tail) == _reference(a, b, head, tail), (a, b, head, tail)
