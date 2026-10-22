# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TruncatingStringIO.

Covers: head fill, tail roll, was_truncated, chars_written, getvalue format,
empty writes, exact-limit writes, custom tail_chars, and re-export path.
"""

from nooa.agentdoc._truncating_stream import TruncatingStringIO


class TestHeadFill:
    def test_small_write_stays_in_head(self):
        buf = TruncatingStringIO(limit=100)
        buf.write("hello")
        assert buf.getvalue() == "hello"
        assert not buf.was_truncated

    def test_multiple_small_writes_accumulate(self):
        buf = TruncatingStringIO(limit=100)
        buf.write("abc")
        buf.write("def")
        assert buf.getvalue() == "abcdef"

    def test_write_exactly_at_limit_not_truncated(self):
        buf = TruncatingStringIO(limit=10)
        buf.write("x" * 10)
        assert not buf.was_truncated
        assert buf.chars_written == 10
        assert buf.getvalue() == "x" * 10

    def test_write_one_over_limit_is_truncated(self):
        buf = TruncatingStringIO(limit=10)
        buf.write("x" * 11)
        assert buf.was_truncated
        assert buf.chars_written == 11

    def test_empty_write_does_nothing(self):
        buf = TruncatingStringIO(limit=10)
        buf.write("")
        assert buf.getvalue() == ""
        assert not buf.was_truncated
        assert buf.chars_written == 0


class TestTailRoll:
    def test_overflow_goes_to_tail(self):
        # limit=10, so head=5, tail=5
        buf = TruncatingStringIO(limit=10)
        buf.write("A" * 5)  # fills head
        buf.write("B" * 10)  # overflow: tail keeps last 5 = "BBBBB"
        assert buf.was_truncated
        value = buf.getvalue()
        assert value.startswith("<truncated-output>")
        assert "AAAAA" in value  # head
        assert "BBBBB" in value  # tail

    def test_tail_rolls_oldest_first(self):
        buf = TruncatingStringIO(limit=10)  # head=5, tail=5
        buf.write("A" * 5)  # fills head
        buf.write("B" * 5)  # fills tail
        buf.write("C" * 5)  # evicts Bs, tail = "CCCCC"
        assert buf.was_truncated
        value = buf.getvalue()
        assert "CCCCC" in value
        assert "BBBBB" not in value

    def test_custom_tail_chars(self):
        buf = TruncatingStringIO(limit=20, tail_chars=15)
        buf.write("A" * 5)  # head_limit = 5
        buf.write("B" * 100)  # tail keeps last 15
        assert buf.was_truncated
        value = buf.getvalue()
        assert value.count("B") == 15


class TestWasTruncated:
    def test_false_when_under_limit(self):
        buf = TruncatingStringIO(limit=50)
        buf.write("x" * 30)
        assert not buf.was_truncated

    def test_false_at_exactly_limit(self):
        buf = TruncatingStringIO(limit=50)
        buf.write("x" * 50)
        assert not buf.was_truncated

    def test_true_when_over_limit(self):
        buf = TruncatingStringIO(limit=50)
        buf.write("x" * 51)
        assert buf.was_truncated

    def test_multiple_writes_cumulative(self):
        buf = TruncatingStringIO(limit=10)
        buf.write("x" * 6)
        buf.write("y" * 6)  # total 12 > 10
        assert buf.was_truncated


class TestCharsWritten:
    def test_counts_all_chars_including_dropped(self):
        buf = TruncatingStringIO(limit=10)
        buf.write("x" * 200)
        assert buf.chars_written == 200

    def test_accumulates_across_writes(self):
        buf = TruncatingStringIO(limit=1000)
        buf.write("a" * 100)
        buf.write("b" * 200)
        assert buf.chars_written == 300

    def test_zero_initially(self):
        buf = TruncatingStringIO(limit=100)
        assert buf.chars_written == 0


class TestGetvalueFormat:
    def test_not_truncated_returns_exact_content(self):
        buf = TruncatingStringIO(limit=100)
        buf.write("hello world")
        assert buf.getvalue() == "hello world"

    def test_truncated_starts_with_truncated_output_tag(self):
        buf = TruncatingStringIO(limit=10)
        buf.write("x" * 100)
        assert buf.getvalue().startswith("<truncated-output>")

    def test_truncated_notice_includes_total_chars(self):
        buf = TruncatingStringIO(limit=10)
        buf.write("x" * 100)
        assert "100" in buf.getvalue()

    def test_truncated_notice_includes_not_shown(self):
        buf = TruncatingStringIO(limit=10)
        buf.write("x" * 100)
        assert "not shown" in buf.getvalue()

    def test_truncated_head_and_tail_both_present(self):
        buf = TruncatingStringIO(limit=10)  # head=5, tail=5
        buf.write("A" * 5 + "." * 90 + "Z" * 5)
        value = buf.getvalue()
        assert "AAAAA" in value
        assert "ZZZZZ" in value


class TestAgentdocPublicExport:
    def test_importable_from_agentdoc(self):
        from nooa.agentdoc import TruncatingStringIO as T

        assert T is TruncatingStringIO

    def test_has_chars_written(self):
        buf = TruncatingStringIO(limit=5)
        buf.write("hello world")
        assert buf.chars_written == 11
        assert buf.was_truncated
