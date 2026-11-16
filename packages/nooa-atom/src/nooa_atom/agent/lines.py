# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unbounded byte streams read line by line in bounded memory (job logs).

- :class:`LineSplitter` — feed it byte chunks, get complete lines back, each
  bounded to its first ``max_line_chars`` by a chars ``nooa.snapshot.TextStream``
  and rendered by ``render_output``; usable from polling code (e.g. following
  a file).
- :func:`bounded_lines` — the same over an ``asyncio.StreamReader``, reading
  fixed-size chunks.
"""

import codecs
import dataclasses
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING

from nooa.snapshot import Overflow, TextStream, render_output

if TYPE_CHECKING:
    import asyncio


class LineSplitter:
    """Split byte chunks into text lines, each bounded to its first ``max_line_chars``.

    ``feed(chunk)`` returns the lines the chunk completes (each ending in
    ``"\\n"``); ``flush()`` returns the last unterminated line, if any. Each
    line is written to a chars ``nooa.snapshot.TextStream(head=max_line_chars,
    tail=0)`` and returned as ``render_output`` of it: a line within the
    limit is itself; a longer one is its first ``max_line_chars`` characters,
    then, on its own line, ``[... N chars omitted; full output not kept]``.
    Memory stays at one bounded line plus the chunk being split. ``on_bytes``,
    if given, receives every input byte, in order, one line piece at a time
    (e.g. for a full log). ``full_path`` (keyword-only), if given, names the
    file that holds the whole input (that log): a cut line then says
    ``[... N chars omitted; full output: PATH]``. Text is decoded as UTF-8
    with replacement characters.
    """

    def __init__(
        self,
        max_line_chars: int,
        on_bytes: Callable[[bytes], object] | None = None,
        *,
        full_path: str | None = None,
    ):
        if max_line_chars < 1:
            raise ValueError("max_line_chars must be at least 1")
        self._max = max_line_chars
        self._on_bytes = on_bytes
        self._full_path = full_path
        self._line_bytes = 0  # bytes of the current line, for its Overflow
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._line = self._new_line()
        self._started = False

    def _new_line(self) -> TextStream:
        return TextStream(unit="chars", head=self._max, tail=0)

    def _add(self, text: str) -> None:
        if text:
            self._line.write(text)

    def _take(self, newline: str) -> str:
        text = self._line.close()
        if not text.complete and self._full_path is not None:
            # The whole line is in the file that receives every byte (on_bytes).
            size = self._line_bytes
            text = dataclasses.replace(text, full=Overflow(self._full_path, size, size))
        line = render_output(text)
        self._line, self._started, self._line_bytes = self._new_line(), False, 0
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
                self._line_bytes += len(piece)
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
    full_path: str | None = None,
) -> AsyncIterator[str]:
    """Yield ``reader``'s lines in bounded memory, reading ``chunk_bytes`` at a time.

    Lines end in ``"\\n"`` (the last one may not) and are capped as in
    :class:`LineSplitter`; ``on_bytes`` receives every byte read, and a cut
    line names ``full_path`` when given. Unlike
    ``StreamReader.readline`` a line longer than the reader's limit neither
    raises nor loses data.
    """
    splitter = LineSplitter(max_line_chars, on_bytes, full_path=full_path)
    while True:
        data = await reader.read(chunk_bytes)
        if not data:
            break
        for line in splitter.feed(data):
            yield line
    last = splitter.flush()
    if last is not None:
        yield last
