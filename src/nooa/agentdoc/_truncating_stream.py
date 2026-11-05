# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""StringIO with hard character limit and truncation notices.

Used by pformat/truncating_pformat to bound memory during formatting of large objects.

Two flavors:

- :class:`TruncatingStringIO` — pure in-memory head/tail buffer (default).
- :class:`FileBackedTruncatingStringIO` — same truncation, but also streams
  all output to a temp file and includes the file path in the truncation
  notice, so the full untruncated output remains retrievable.

With ``tail_chars == limit`` (no head) the buffer keeps only the end of the
output and the notice is one short line before it.

For reading unbounded byte streams line by line in bounded memory:

- :class:`LineSplitter` — feed it byte chunks, get complete lines back, each
  capped at ``max_line_chars`` with a ``[... N chars of this line omitted]``
  marker; usable from polling code (e.g. following a file).
- :func:`bounded_lines` — the same over an ``asyncio.StreamReader``, reading
  fixed-size chunks.
"""

import codecs
import collections
import io
import logging
import os
import tempfile
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING

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


class FileBackedTruncatingStringIO(TruncatingStringIO):
    """TruncatingStringIO that also writes full output to a file.

    Behaves identically to :class:`TruncatingStringIO` for in-memory
    head/tail truncation, but additionally streams all written content to a
    file on disk (UTF-8; lone surrogates are written as ``\\udXXX`` escapes):
    a new temp file. When truncated, ``getvalue()`` includes the file path
    in the notice so the user can inspect the full output.

    File I/O errors are handled gracefully — if the file cannot be
    created or written to, the stream falls back to pure in-memory behavior
    (identical to the parent class).

    The caller is responsible for cleanup via :meth:`cleanup` (which removes
    the file) or :meth:`close` (which closes the file handle but leaves
    the file on disk for later reference).
    """

    def __init__(
        self,
        limit: int = TruncatingStringIO.DEFAULT_LIMIT,
        tail_chars: int | None = None,
        *,
        dir: str | None = None,
        prefix: str = "nemo_output_",
        suffix: str = ".txt",
        unrecoverable_hint: str | None = None,
    ):
        super().__init__(limit=limit, tail_chars=tail_chars, unrecoverable_hint=unrecoverable_hint)
        self._file_failed = False
        self._file_path: str | None = None
        self._file: io.BufferedWriter | None = None
        try:
            fd, temp_path = tempfile.mkstemp(dir=dir, prefix=prefix, suffix=suffix)
            self._file = os.fdopen(fd, "wb")
            self._file_path = temp_path
        except OSError:
            _log.warning(
                "Failed to create file for output capture; falling back to in-memory",
                exc_info=True,
            )
            self._file_failed = True

    def write(self, s: str) -> int:
        """Write to both the file and the in-memory truncating buffer."""
        if self._file is not None and not self._file_failed and not self._file.closed:
            try:
                self._file.write(s.encode("utf-8", errors="backslashreplace"))
                self._file.flush()
            except OSError:
                _log.warning("Failed to write to file; disabling file backing", exc_info=True)
                self._file_failed = True
        return super().write(s)

    def _output_path(self) -> str | None:
        if self._file_path and not self._file_failed:
            self._flush_file()
            return self._file_path
        return super()._output_path()

    def _format_truncation_notice(
        self, head: str, tail: str, total: int, head_chars: int, tail_chars: int, dropped: int
    ) -> str:
        """Include the file path in the truncation notice."""
        path = self._output_path()
        if path:
            file_notice = (
                f"The full untruncated output ({total:,} chars) is in: {path}\n"
                f"Read that file to see the {dropped:,} chars not shown here.\n"
            )
        else:
            file_notice = self._not_recoverable(dropped)

        return (
            f"<truncated-output>\n"
            f"Output too large ({total:,} chars). "
            f"Showing first {head_chars:,} and last {tail_chars:,} chars.\n"
            f"{file_notice}\n"
            f"{head}\n\n"
            f"... {dropped:,} chars not shown ...\n\n"
            f"{tail}\n"
            f"</truncated-output>"
        )

    def _flush_file(self) -> None:
        """Flush the file, ignoring errors."""
        if self._file is not None and not self._file.closed:
            try:
                self._file.flush()
            except OSError:
                pass

    @property
    def file_path(self) -> str | None:
        """Path to the file, or None if file creation failed."""
        return self._file_path

    def close(self) -> None:
        """Close the file handle (file remains on disk).

        Intentionally does NOT call ``super().close()`` — the underlying
        ``io.StringIO`` must stay open so ``getvalue()`` remains usable
        after the file handle is closed.
        """
        if self._file is not None and not self._file.closed:
            try:
                self._file.close()
            except OSError:
                pass

    def cleanup(self) -> None:
        """Close the file handle and remove the file from disk."""
        self.close()
        if self._file_path is not None:
            try:
                os.unlink(self._file_path)
            except OSError:
                pass


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
