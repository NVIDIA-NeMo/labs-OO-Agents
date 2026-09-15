# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for prefix-only serialization limiting."""

import pytest

from nooa.tracing._limited_writer import LimitedWriter, SerializationLimitReached


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
