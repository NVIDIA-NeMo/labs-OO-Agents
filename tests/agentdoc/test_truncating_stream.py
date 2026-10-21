# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TruncatingStringIO.

Covers: head fill, tail roll, was_truncated, chars_written, getvalue format,
empty writes, exact-limit writes, custom tail_chars, and re-export path;
the tail-only notice; LineSplitter and bounded_lines.
"""

import asyncio

from nooa.agentdoc._truncating_stream import LineSplitter, TruncatingStringIO, bounded_lines


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


class TestTailOnly:
    """tail_chars == limit: keep only the end, behind one short notice line."""

    def test_untruncated_is_verbatim(self):
        buf = TruncatingStringIO(limit=10, tail_chars=10)
        buf.write("hello")
        assert buf.getvalue() == "hello"

    def test_notice_is_one_line_before_the_tail(self):
        buf = TruncatingStringIO(limit=10, tail_chars=10)
        for _ in range(3):
            buf.write("abcdefghij")
        assert buf.getvalue() == "[... 20 chars omitted]\nabcdefghij"

    def test_notice_names_the_full_output_path(self):
        buf = TruncatingStringIO(limit=4, tail_chars=4, full_output_path="/tmp/x.log")
        buf.write("x" * 52_004)
        assert buf.getvalue() == "[... 52,000 chars omitted; full output: /tmp/x.log]\nxxxx"

    def test_head_and_tail_notice_with_a_path(self):
        buf = TruncatingStringIO(limit=10, full_output_path="/tmp/x.log")
        buf.write("a" * 30)
        assert "The full untruncated output (30 chars) is in: /tmp/x.log" in buf.getvalue()


class TestLineSplitter:
    def test_splits_lines_across_chunks(self):
        splitter = LineSplitter(100)
        assert splitter.feed(b"one\ntw") == ["one\n"]
        assert splitter.feed(b"o\nthree") == ["two\n"]
        assert splitter.flush() == "three"
        assert splitter.flush() is None

    def test_caps_an_overlong_line_and_passes_every_byte_on(self):
        seen = bytearray()
        splitter = LineSplitter(5, on_bytes=seen.extend)
        lines = []
        for _ in range(10):
            lines += splitter.feed(b"x" * 1_000)
        lines += splitter.feed(b"\nnext\n")
        assert lines == ["xxxxx[... 9,995 chars of this line omitted]\n", "next\n"]
        assert bytes(seen) == b"x" * 10_000 + b"\nnext\n"

    def test_a_character_split_across_chunks_decodes_whole(self):
        data = "é\n".encode()
        splitter = LineSplitter(10)
        assert splitter.feed(data[:1]) == []
        assert splitter.feed(data[1:]) == ["é\n"]

    def test_undecodable_bytes_are_replaced(self):
        assert LineSplitter(10).feed(b"\xff\n") == ["\ufffd\n"]


class _Chunks:
    """A reader whose ``read(n)`` makes ``total`` bytes of one line on demand, then ``tail``."""

    def __init__(self, total: int, tail: bytes):
        self.left, self.tail = total, tail

    async def read(self, n: int) -> bytes:
        if self.left:
            size = min(n, self.left)
            self.left -= size
            return b"z" * size
        data, self.tail = self.tail[:n], self.tail[n:]
        return data


class TestBoundedLines:
    async def test_reads_a_huge_line_in_bounded_memory(self):
        import tracemalloc

        total = 0

        def count(data: bytes) -> None:
            nonlocal total
            total += len(data)

        tracemalloc.start()
        try:
            baseline = tracemalloc.get_traced_memory()[0]
            lines = [
                line
                async for line in bounded_lines(
                    _Chunks(3_000_000, b"\nend"),
                    max_line_chars=100,
                    chunk_bytes=4096,
                    on_bytes=count,
                )
            ]
            peak = tracemalloc.get_traced_memory()[1] - baseline
        finally:
            tracemalloc.stop()
        assert lines == ["z" * 100 + "[... 2,999,900 chars of this line omitted]\n", "end"]
        assert total == 3_000_004
        assert peak < 100_000

    async def test_reads_an_asyncio_stream_reader(self):
        reader = asyncio.StreamReader()
        reader.feed_data(b"a\n" + b"b" * 70_000 + b"\nc")
        reader.feed_eof()
        lines = [line async for line in bounded_lines(reader, max_line_chars=10)]
        assert lines == ["a\n", "b" * 10 + "[... 69,990 chars of this line omitted]\n", "c"]
