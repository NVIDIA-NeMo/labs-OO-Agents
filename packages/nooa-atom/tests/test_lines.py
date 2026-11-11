# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LineSplitter and bounded_lines (moved from nooa.agentdoc._truncating_stream)."""

import asyncio

from nooa_atom.agent.lines import LineSplitter, bounded_lines


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
