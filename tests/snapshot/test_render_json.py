# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""render_json and preview_attributes: valid, bounded JSON previews and their span attributes."""

from __future__ import annotations

import json
import math
import random
from typing import Any

import pytest
from pydantic import BaseModel

from nooa.snapshot import (
    Limits,
    Mapping,
    Opaque,
    Scalar,
    Spill,
    Text,
    capture,
    capture_fields,
    preview_attributes,
    render_json,
)

# --- render_json -----------------------------------------------------------------------


class _Model(BaseModel):
    text: str = "t" * 30
    items: list[Any] = [1, "two", None]


def _shapes(rng: random.Random, depth: int = 0) -> Any:
    choice = rng.randrange(10 if depth < 4 else 5)
    if choice == 0:
        return rng.choice(["", "x" * rng.randrange(300), '"\\\n' * rng.randrange(40), "😀" * 30])
    if choice == 1:
        return rng.choice([0, -1, 10**40, 1 << 5000, 1.5, math.nan, None, True])
    if choice == 2:
        return rng.choice([b"", b"\xff" * 10, "é".encode() * 50])
    if choice == 3:
        return rng.choice([_Model(), object(), Opaque])
    if choice == 4:
        return "😀" * rng.randrange(5)
    if choice in (5, 6):
        return [_shapes(rng, depth + 1) for _ in range(rng.randrange(8))]
    if choice == 7:
        return tuple(_shapes(rng, depth + 1) for _ in range(rng.randrange(4)))
    return {
        rng.choice(["k", "a/b", "~", "q" * 40, 7, None, 1.5]): _shapes(rng, depth + 1)
        for _ in range(rng.randrange(6))
    }


def test_render_json_is_valid_and_bounded_for_many_shapes() -> None:
    rng = random.Random(342)
    for _ in range(1_500):
        value = _shapes(rng)
        limits = Limits(
            max_chars=rng.randrange(4, 300),
            max_nodes=rng.randrange(1, 80),
            max_depth=rng.randrange(0, 6),
            max_items=rng.choice([None, 1, 3]),
            max_string=rng.choice([None, 5, 50]),
            tail_share=rng.choice([0.0, 0.5]),
        )
        snapshot = capture(value, limits)
        # Rendering at the capture budget, and at smaller ones, always fits.
        for max_chars in (limits.max_chars, max(1, limits.max_chars // 3), 1):
            text = render_json(snapshot, max_chars=max_chars)
            json.loads(text)
            assert len(text) <= max_chars
        attributes = preview_attributes("output", snapshot, max_chars=limits.max_chars)
        assert attributes["nooa.output.preview.version"] == 1
        assert attributes["nooa.output.preview.incomplete"] is (
            "nooa.output.preview.paths" in attributes
        )


def test_render_json_at_capture_budget_cuts_nothing_further() -> None:
    value = {"a": "x" * 1_000, "b": list(range(100))}
    limits = Limits(max_chars=200)
    snapshot = capture(value, limits)

    assert preview_attributes("output", snapshot, max_chars=200) == preview_attributes(
        "output", snapshot
    )


def test_render_json_smaller_budget_reports_new_cuts() -> None:
    snapshot = capture({"a": "x" * 50, "b": "y" * 50})

    text = render_json(snapshot, max_chars=30)
    attributes = preview_attributes("input", snapshot, max_chars=30)

    assert len(text) <= 30 and json.loads(text)
    assert attributes["nooa.input.preview.incomplete"] is True
    assert attributes["nooa.input.preview.reasons"] == ("max_length",)
    assert preview_attributes("input", snapshot)["nooa.input.preview.incomplete"] is False


def test_render_json_non_string_keys_follow_json() -> None:
    snapshot = capture({2: "a", None: "b", True: "c", 1.5: "d"})

    assert json.loads(render_json(snapshot, max_chars=100)) == {
        "2": "a",
        "null": "b",
        "true": "c",
        "1.5": "d",
    }


def test_render_json_colliding_keys_stop_the_mapping() -> None:
    snapshot = capture({1: "a", "1": "b"})

    assert json.loads(render_json(snapshot, max_chars=100)) == {"1": "a"}
    assert preview_attributes("output", snapshot)["nooa.output.preview.paths"] == ("",)


def test_render_json_bytes_and_cut_text_keep_a_prefix() -> None:
    assert json.loads(render_json(capture(b"abc"), max_chars=10)) == "abc"
    cut = capture("abcdefghij", Limits(max_string=4, tail_share=0.5))
    assert json.loads(render_json(cut, max_chars=100)) == "ab"  # the tail is not joined


def test_render_json_rejects_nonpositive_budget() -> None:
    with pytest.raises(ValueError):
        render_json(Scalar(1), max_chars=0)


def test_capture_fields_keep_every_root_field() -> None:
    snapshot = capture_fields(
        stdout="o" * 1_000, stderr="", returned_value=_Model(), limits=Limits(max_chars=60)
    )

    assert isinstance(snapshot, Mapping) and snapshot.wrapper
    parsed = json.loads(render_json(snapshot, max_chars=60))
    assert set(parsed) == {"stdout", "stderr", "returned_value"}
    assert isinstance(parsed["returned_value"], dict)


# --- preview_attributes ------------------------------------------------------------------


def test_preview_attributes_complete() -> None:
    assert preview_attributes("input", capture({"a": 1})) == {
        "nooa.input.preview.version": 1,
        "nooa.input.preview.incomplete": False,
    }


def test_preview_attributes_reasons_are_sorted_and_unique() -> None:
    cycle: list[Any] = []
    cycle.append(cycle)
    snapshot = capture(
        {"c": cycle, "o": object(), "o2": object(), "s": "x" * 100}, Limits(max_string=5)
    )

    attributes = preview_attributes("output", snapshot)

    assert attributes["nooa.output.preview.reasons"] == ("cycle", "max_length", "opaque")
    assert attributes["nooa.output.preview.paths"] == ("/c/0", "/o", "/o2", "/s")


def test_preview_attributes_full_output() -> None:
    spilled = Text(
        "h", "t", 100, "bytes", False, "max_length", Spill("/tmp/a", 100, 100), omitted=98
    )
    snapshot = Mapping("dict", 2, (("stdout", spilled), ("again", spilled)), True, None)

    attributes = preview_attributes("output", snapshot)

    assert attributes["nooa.output.preview.full_output"] == ("/tmp/a",)
    assert attributes["nooa.output.preview.paths"] == ("/stdout", "/again")
