# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Random snapshot trees for render checks: every node type, tails, wrappers and overflow files."""

from __future__ import annotations

import random

from nooa.snapshot import (
    Mapping,
    Object,
    Overflow,
    Ref,
    Scalar,
    Sequence,
    Snapshot,
    Text,
    Unavailable,
)

FILES = tuple(f"/tmp/out-{n}.txt" for n in range(6))


def random_text(rng: random.Random) -> Text:
    """A complete or cut ``Text`` (chars or bytes), sometimes with an overflow file."""
    unit = rng.choice(("chars", "bytes"))
    head = "".join(rng.choice('ab\u00e9\u20ac\n"') for _ in range(rng.randint(0, 12)))
    tail = "".join(rng.choice("xy\u00e9") for _ in range(rng.randint(0, 6)))
    if unit == "bytes":
        head_kept, tail_kept = head.encode(), tail.encode()
    else:
        head_kept, tail_kept = head, tail
    if rng.random() < 0.3:
        whole = head_kept + tail_kept  # type: ignore[operator]
        return Text(head=whole, tail=whole[:0], length=len(whole), unit=unit, reason=None)
    omitted = rng.randint(1, 500)
    length = len(head_kept) + len(tail_kept) + omitted
    full = None
    if rng.random() < 0.5:
        total = length
        full = Overflow(rng.choice(FILES), rng.choice((total, total // 2)), total)
    return Text(
        head=head_kept,
        tail=tail_kept,
        length=length,
        unit=unit,
        reason="max_length",
        full=full,
    )


def random_tree(rng: random.Random, depth: int = 0) -> Snapshot:
    """A random detached snapshot of at most about five levels."""
    leaf = depth >= 4 or rng.random() < 0.35
    if leaf:
        choice = rng.randrange(7)
        if choice == 0:
            return Scalar(rng.choice((None, True, 7, -1.5, 10**30, b"ab", b"\xff")))
        if choice <= 2:
            return random_text(rng)
        if choice == 3:
            module = rng.choice(("builtins", "pkg"))
            return Unavailable("opaque", rng.choice(("int", "Thing")), module == "builtins")
        if choice == 4:
            return Unavailable(rng.choice(("error", "max_nodes")), rng.choice(("", "list", "int")))
        if choice == 5:
            return Ref("r1", rng.choice(("sequence", "mapping", "object")))
        return Scalar(3)
    kind = rng.randrange(3)
    count = rng.randint(0, 4)
    children = [random_tree(rng, depth + 1) for _ in range(count)]
    tail = tuple(random_tree(rng, depth + 1) for _ in range(rng.randint(0, 2)))
    cut = bool(tail) or rng.random() < 0.2
    reason = "max_items" if cut else None
    if kind == 0:
        type_name = rng.choice(("list", "tuple"))
        length = len(children) + len(tail) + (5 if cut else 0)
        return Sequence(type_name, length, tuple(children), reason, tail)
    if kind == 1:
        keys = [rng.choice(("a", "b", "k" * 30, 1, 2.5, True, None, "~/")) for _ in children]
        items = tuple(zip(keys, children, strict=True))
        tail_items = tuple((f"t{n}", child) for n, child in enumerate(tail))
        wrapper = not tail_items and rng.random() < 0.3
        length = len(items) + len(tail_items) + (5 if cut else 0)
        return Mapping("dict", length, items, reason, tail_items, wrapper)
    fields = tuple((f"f{n}", child) for n, child in enumerate(children + list(tail)))
    return Object("Thing", fields, reason, rng.choice((None, "fake.thing")))
