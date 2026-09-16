# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for prefix-only serialization limiting."""

import json

import pytest

from nooa.tracing._limited_writer import (
    LimitedWriter,
    SerializationLimitReached,
    dump_json_bounded,
)


def test_limited_writer_returns_complete_content_within_limit() -> None:
    writer = LimitedWriter(5)

    assert writer.write("ab") == 2
    assert writer.write("cde") == 3

    assert writer.getvalue() == "abcde"
    assert writer.chars_written == 5
    assert writer.remaining == 0


def test_limited_writer_retains_prefix_and_aborts_on_overflow() -> None:
    writer = LimitedWriter(5)
    writer.write("abc")

    with pytest.raises(SerializationLimitReached):
        writer.write("defg")

    assert writer.getvalue() == "abcde"
    assert writer.chars_written == 5
    assert writer.remaining == 0


@pytest.mark.parametrize("limit", [0, -1])
def test_limited_writer_requires_positive_limit(limit: int) -> None:
    with pytest.raises(ValueError, match="must be > 0"):
        LimitedWriter(limit)


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        0,
        -42,
        1.25,
        float("inf"),
        float("-inf"),
        float("nan"),
        'quotes: " slash: \\ newline:\n unicode: ☃ 🚀',
        "x" * 255 + "\n" + "🚀" + "y" * 300,
        [1, "two", None, {"nested": (True, 3.5)}],
        {2: "int", 2.5: "float", False: "bool", None: "none"},
    ],
)
def test_bounded_json_matches_standard_encoder(value: object) -> None:
    writer = LimitedWriter(10_000)

    dump_json_bounded(value, writer, default=repr)

    assert writer.getvalue() == json.dumps(value, default=repr)
