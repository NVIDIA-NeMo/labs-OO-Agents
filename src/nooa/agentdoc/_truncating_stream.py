# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""StringIO with hard character limit and truncation notices.

Used by pformat/truncating_pformat to bound memory during formatting of large objects.

- :class:`TruncatingStringIO` — pure in-memory head/tail buffer.

With ``tail_chars == limit`` (no head) the buffer keeps only the end of the
output and the notice is one short line before it.

For reading unbounded byte streams line by line in bounded memory:

- :class:`LineSplitter` — feed it byte chunks, get complete lines back, each
  bounded to its first ``max_line_chars`` by a chars ``nooa.snapshot.TextStream``
  and rendered by ``render_output``; usable from polling code (e.g. following
  a file).
- :func:`bounded_lines` — the same over an ``asyncio.StreamReader``, reading
  fixed-size chunks.
"""

import codecs
import collections
import io
import logging
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING

from nooa.snapshot import TextStream, render_output

if TYPE_CHECKING:
    import asyncio

_log = logging.getLogger(__name__)


class TruncatingStringIO(io.StringIO):
    """StringIO with hard character limit and head+tail truncation.

    Keeps the first ``head_limit`` chars verbatim and a rolling tail of the
    last ``tail_limit`` chars.  When truncated, ``getvalue()`` returns the
    head and tail joined by a prose notice.

    ``was_truncated`` is True only when ``_chars_written > limit`` (i.e. the
    total content actually exceeded the budget — not merely when the head
    buffer filled up).  When not truncated, ``getvalue()`` returns all content
    verbatim (head concatenated with any tail overflow).

    Example:
        buffer = TruncatingStringIO(limit=100)
        buffer.write("x" * 200)
        content = buffer.getvalue()
        # Returns prose head+tail format with a "chars not shown" notice.
    """

    DEFAULT_LIMIT = 50_000  # 50KB per execution

    def __init__(
        self,
        limit: int = DEFAULT_LIMIT,
        tail_chars: int | None = None,
        *,
        unrecoverable_hint: str | None = None,
    ):
        """Initialize truncating buffer.

        Args:
            limit: Maximum characters to store (default: 50,000).
            tail_chars: Characters reserved for the tail window.
                        None = half of limit (50/50 split). Equal to
                        ``limit``: keep only the end (tail-only notice).
            unrecoverable_hint: A sentence after the notice's "not recoverable"
                        line (e.g. how to keep large output next time).
        """
        super().__init__()
        self._unrecoverable_hint = unrecoverable_hint
        self._limit = limit
        self._tail_limit = tail_chars if tail_chars is not None else limit // 2
        self._head_limit = limit - self._tail_limit
        self._head_full = False
        # Rolling tail buffer — each element is a string chunk written after
        # the head was full.  We keep enough chars to reconstruct the last
        # _tail_limit characters.
        self._tail_chunks: collections.deque[str] = collections.deque()
        self._tail_chars = 0
        self._chars_written = 0

    def write(self, s: str) -> int:
        """Write string, filling head then rolling tail."""
        n = len(s)
        self._chars_written += n

        if not self._head_full:
            current_head = len(super().getvalue())
            remaining_head = self._head_limit - current_head

            if n <= remaining_head:
                super().write(s)
                return n

            if remaining_head > 0:
                super().write(s[:remaining_head])
            self._head_full = True
            overflow = s[remaining_head:]
            if overflow:
                self._add_to_tail(overflow)
        else:
            self._add_to_tail(s)

        return n

    def _add_to_tail(self, s: str) -> None:
        """Add a chunk to the rolling tail buffer, evicting old chars as needed."""
        self._tail_chunks.append(s)
        self._tail_chars += len(s)
        while self._tail_chars > self._tail_limit and self._tail_chunks:
            oldest = self._tail_chunks[0]
            excess = self._tail_chars - self._tail_limit
            if len(oldest) <= excess:
                self._tail_chunks.popleft()
                self._tail_chars -= len(oldest)
            else:
                self._tail_chunks[0] = oldest[excess:]
                self._tail_chars -= excess

    def _get_tail(self) -> str:
        """Return the current tail buffer as a single string."""
        return "".join(self._tail_chunks)

    def getvalue(self) -> str:
        """Get buffer contents, with prose head+tail notice if truncated."""
        head = super().getvalue()

        if not self.was_truncated:
            tail = self._get_tail()
            return head + tail

        tail = self._get_tail()
        total = self._chars_written
        head_chars = len(head)
        tail_chars = len(tail)
        dropped = total - head_chars - tail_chars

        if self._head_limit <= 0:
            return self._format_tail_notice(tail, dropped)
        return self._format_truncation_notice(head, tail, total, head_chars, tail_chars, dropped)

    def _output_path(self) -> str | None:
        """Where the full output is, if anywhere (named in the notices).

        None here: an in-memory buffer keeps no full copy. File-backed
        subclasses return their file.
        """
        return None

    def _not_recoverable(self, dropped: int) -> str:
        """The notice line for a cut with no full copy, plus the configured hint."""
        hint = f" {self._unrecoverable_hint}" if self._unrecoverable_hint else ""
        return f"The {dropped:,} chars in the middle are not recoverable.{hint}\n"

    def _format_tail_notice(self, tail: str, dropped: int) -> str:
        """Tail-only buffers: one short line, then the end of the output."""
        path = self._output_path()
        where = f"; full output: {path}" if path else ""
        return f"[... {dropped:,} chars omitted{where}]\n{tail}"

    def _format_truncation_notice(
        self, head: str, tail: str, total: int, head_chars: int, tail_chars: int, dropped: int
    ) -> str:
        """Build the truncated-output wrapper.

        Subclasses override this to inject extra information (e.g. a file path).
        """
        path = self._output_path()
        if path:
            where = f"The full untruncated output ({total:,} chars) is in: {path}\n"
        else:
            where = self._not_recoverable(dropped)
        return (
            f"<truncated-output>\n"
            f"Output too large ({total:,} chars). "
            f"Showing first {head_chars:,} and last {tail_chars:,} chars.\n"
            f"{where}\n"
            f"{head}\n\n"
            f"... {dropped:,} chars not shown ...\n\n"
            f"{tail}\n"
            f"</truncated-output>"
        )

    @property
    def was_truncated(self) -> bool:
        """True if total chars written exceeded the limit."""
        return self._chars_written > self._limit

    @property
    def chars_written(self) -> int:
        """Total characters written (including chars that were dropped)."""
        return self._chars_written


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
    (e.g. for a full log). Text is decoded as UTF-8 with replacement characters.
    """

    def __init__(self, max_line_chars: int, on_bytes: Callable[[bytes], object] | None = None):
        if max_line_chars < 1:
            raise ValueError("max_line_chars must be at least 1")
        self._max = max_line_chars
        self._on_bytes = on_bytes
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._line = self._new_line()
        self._started = False

    def _new_line(self) -> TextStream:
        return TextStream(unit="chars", head=self._max, tail=0)

    def _add(self, text: str) -> None:
        if text:
            self._line.write(text)

    def _take(self, newline: str) -> str:
        line = render_output(self._line.close())
        self._line, self._started = self._new_line(), False
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
