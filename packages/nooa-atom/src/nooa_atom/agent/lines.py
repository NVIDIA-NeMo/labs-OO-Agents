# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unbounded byte streams read line by line in bounded memory (job logs).

- :class:`LineSplitter` — feed it byte chunks, get complete lines back, each
  capped at ``max_line_chars`` with a ``[... N chars of this line omitted]``
  marker; usable from polling code (e.g. following a file).
- :func:`bounded_lines` — the same over an ``asyncio.StreamReader``, reading
  fixed-size chunks.
"""

import codecs
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncio


_LINE_OMITTED = "[... {omitted:,} chars of this line omitted]"


class LineSplitter:
    """Split byte chunks into text lines, each capped at ``max_line_chars``.

    ``feed(chunk)`` returns the lines the chunk completes (each ending in
    ``"\\n"``); ``flush()`` returns the last unterminated line, if any. Only
    the first ``max_line_chars`` characters of a line are kept; the rest is
    counted and replaced by ``[... N chars of this line omitted]``, so memory
    stays at one capped line plus the chunk being split. ``on_bytes``, if
    given, receives every input byte, in order, one line piece at a time (e.g.
    for a full log). Text is decoded as UTF-8 with replacement characters.
    """

    def __init__(self, max_line_chars: int, on_bytes: Callable[[bytes], object] | None = None):
        if max_line_chars < 1:
            raise ValueError("max_line_chars must be at least 1")
        self._max = max_line_chars
        self._on_bytes = on_bytes
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._kept: list[str] = []
        self._kept_chars = 0
        self._omitted = 0
        self._started = False

    def _add(self, text: str) -> None:
        room = self._max - self._kept_chars
        if room > 0 and text:
            piece = text[:room]
            self._kept.append(piece)
            self._kept_chars += len(piece)
            self._omitted += len(text) - len(piece)
        else:
            self._omitted += len(text)

    def _take(self, newline: str) -> str:
        line = "".join(self._kept)
        if self._omitted:
            line += _LINE_OMITTED.format(omitted=self._omitted)
        self._kept, self._kept_chars, self._omitted, self._started = [], 0, 0, False
        return line + newline

    def feed(self, data: bytes) -> list[str]:
        """Split ``data``; return the lines it completes (each ending in a newline)."""
        lines: list[str] = []
        start = 0
        while True:
            end = data.find(b"\n", start)
            piece = data[start:] if end < 0 else data[start : end + 1]
            if piece:
                self._started = True
                if self._on_bytes is not None:
                    self._on_bytes(piece)
            if end < 0:
                self._add(self._decoder.decode(piece))
                return lines
            self._add(self._decoder.decode(piece[:-1], final=True))
            lines.append(self._take("\n"))
            start = end + 1

    def flush(self) -> str | None:
        """The pending unterminated line (no newline), or None if there is none."""
        self._add(self._decoder.decode(b"", final=True))
        if not self._started:
            return None
        return self._take("")


async def bounded_lines(
    reader: "asyncio.StreamReader",
    *,
    max_line_chars: int,
    chunk_bytes: int = 64 * 1024,
    on_bytes: Callable[[bytes], object] | None = None,
) -> AsyncIterator[str]:
    """Yield ``reader``'s lines in bounded memory, reading ``chunk_bytes`` at a time.

    Lines end in ``"\\n"`` (the last one may not) and are capped as in
    :class:`LineSplitter`; ``on_bytes`` receives every byte read. Unlike
    ``StreamReader.readline`` a line longer than the reader's limit neither
    raises nor loses data.
    """
    splitter = LineSplitter(max_line_chars, on_bytes)
    while True:
        data = await reader.read(chunk_bytes)
        if not data:
            break
        for line in splitter.feed(data):
            yield line
    last = splitter.flush()
    if last is not None:
        yield last
