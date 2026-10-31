# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""StringIO with hard character limit and truncation notices.

Used by pformat/truncating_pformat to bound memory during formatting of large objects.

Two flavors:

- :class:`TruncatingStringIO` — pure in-memory head/tail buffer (default).
- :class:`FileBackedTruncatingStringIO` — same truncation, but also streams
  all output (text, or raw bytes such as a process's output) to a file (a
  temp file unless ``path=`` is given, optionally capped by
  ``max_file_bytes=``) and names the file in the truncation notice
  (caller-phrased via ``notice=``), so the full untruncated output remains
  retrievable.

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
import dataclasses
import io
import json
import logging
import os
import tempfile
import threading
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
        full_output_path: str | None = None,
    ):
        """Initialize truncating buffer.

        Args:
            limit: Maximum characters to store (default: 50,000).
            tail_chars: Characters reserved for the tail window.
                        None = half of limit (50/50 split). Equal to
                        ``limit``: keep only the end (tail-only notice).
            full_output_path: Where the caller keeps the full output, named
                        in the truncation notice (None: not recoverable).
        """
        super().__init__()
        self._full_output_path = full_output_path
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
        self._head_written = 0

    def write(self, s: str) -> int:
        """Write string, filling head then rolling tail."""
        n = len(s)
        self._chars_written += n

        if not self._head_full:
            # A counter, not len(getvalue()): getvalue() copies the head, which
            # made many small writes quadratic in the head size.
            remaining_head = self._head_limit - self._head_written

            if n <= remaining_head:
                super().write(s)
                self._head_written += n
                return n

            if remaining_head > 0:
                super().write(s[:remaining_head])
                self._head_written += remaining_head
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
        """Where the full output is, if anywhere (named in the notices)."""
        return self._full_output_path

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
            where = f"The {dropped:,} chars in the middle are not recoverable.\n"
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


@dataclasses.dataclass(frozen=True)
class CaptureStats:
    """What a :class:`FileBackedTruncatingStringIO` received and kept.

    Also the fields of a ``notice`` template (``str.format`` keywords, plus
    ``quoted_path``: the path as a JSON string literal).
    """

    path: str | None  # the file, or None when no file is kept (creation/write failed)
    chars_written: int  # characters fed to the in-memory view (decoded bytes included)
    bytes_written: int  # bytes received (text writes count their UTF-8 encoding)
    file_bytes: int  # bytes in the file
    file_bytes_dropped: int  # bytes past ``max_file_bytes``: counted, not written
    truncated: bool  # the in-memory view is cut (``chars_written > limit``)
    dropped_chars: int  # characters the cut view leaves out


class FileBackedTruncatingStringIO(TruncatingStringIO):
    """TruncatingStringIO that also writes the full output to a file.

    In memory it behaves like :class:`TruncatingStringIO` (head/tail view of
    at most ``limit`` chars); everything written also goes to a file on disk:
    a new temp file (``dir``/``prefix``/``suffix``), or ``path`` if given.
    When the view is cut, ``getvalue()`` names the file in the notice.

    Input:

    - ``write(text)`` — text (``sys.stdout`` style); the file gets its UTF-8.
    - ``write_bytes(data)`` — raw bytes (e.g. a process's output): written to
      the file unchanged (byte-exact), and decoded for the in-memory view
      with an incremental UTF-8 decoder, so a character split across chunks
      decodes correctly (invalid bytes become U+FFFD in the view only).
      :meth:`close` flushes a trailing incomplete character into the view.

    ``max_file_bytes`` caps the file: bytes past it are counted
    (``file_bytes_dropped``) but not written; the file holds exactly the
    first ``max_file_bytes`` bytes, and the notice says it is incomplete.

    ``notice`` phrases where the full output is, in the cut view (default:
    a generic "is in: <path>" sentence): a ``str.format`` template over the
    :class:`CaptureStats` fields plus ``quoted_path``, e.g.
    ``'Full output: read_file({quoted_path})'``, or a callable taking the
    :class:`CaptureStats` and returning the text.

    Lifecycle: the caller owns the file. :meth:`close` closes the handle and
    keeps the file (the view stays readable); :meth:`cleanup` closes and
    removes it. Both are idempotent; ``with`` closes on exit (keeps the
    file). :attr:`path` and :attr:`stats` stay valid after either.

    File I/O errors are handled gracefully: if the file cannot be created or
    written, the stream carries on in memory only (``path`` is then None and
    the notice says the middle is not recoverable).

    Thread safety: writes, ``getvalue()``, ``stats`` and ``close()`` take an
    internal lock, so a reader thread may write while another thread reads.

    Cost: memory is bounded by ``limit`` chars; ``getvalue()`` copies the
    head and joins the tail, O(limit) (a few ms at a 1 MiB limit).
    """

    def __init__(
        self,
        limit: int = TruncatingStringIO.DEFAULT_LIMIT,
        tail_chars: int | None = None,
        *,
        dir: str | None = None,
        prefix: str = "nemo_output_",
        suffix: str = ".txt",
        path: str | os.PathLike[str] | None = None,
        max_file_bytes: int | None = None,
        notice: str | Callable[[CaptureStats], str] | None = None,
    ):
        # Set first: io.StringIO's finalizer calls close() even if __init__ raises.
        self._lock = threading.RLock()
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._file: io.BufferedWriter | None = None
        super().__init__(limit=limit, tail_chars=tail_chars)
        if max_file_bytes is not None and max_file_bytes < 0:
            raise ValueError("max_file_bytes must be >= 0")
        self._notice = notice
        self._max_file_bytes = max_file_bytes
        self._bytes_written = 0
        self._file_bytes = 0
        self._file_dropped = 0
        self._file_failed = False
        self._removed = False
        self._file_path: str | None = None
        try:
            if path is not None:
                self._file = open(path, "wb")  # noqa: SIM115 - closed in close()
                self._file_path = os.fspath(path)
            else:
                fd, temp_path = tempfile.mkstemp(dir=dir, prefix=prefix, suffix=suffix)
                self._file = os.fdopen(fd, "wb")
                self._file_path = temp_path
        except OSError:
            _log.warning(
                "Failed to create file for output capture; falling back to in-memory",
                exc_info=True,
            )
            self._file_failed = True

    # ---- input ------------------------------------------------------------

    def write(self, s: str) -> int:
        """Write text to the file (as UTF-8) and to the in-memory view."""
        with self._lock:
            data = s.encode("utf-8", errors="replace")
            self._bytes_written += len(data)
            self._write_file(data)
            return super().write(s)

    def write_bytes(self, data: bytes) -> int:
        """Write raw bytes: unchanged to the file, decoded (incrementally) to the view."""
        with self._lock:
            self._bytes_written += len(data)
            self._write_file(data)
            text = self._decoder.decode(data)
            if text:
                super().write(text)
            return len(data)

    def _write_file(self, data: bytes) -> None:
        """Append ``data`` to the file, up to ``max_file_bytes`` (the rest is counted)."""
        if not data:
            return
        if self._file is None or self._file_failed or self._file.closed:
            return
        room = len(data)
        if self._max_file_bytes is not None:
            room = max(0, min(room, self._max_file_bytes - self._file_bytes))
        self._file_dropped += len(data) - room
        if not room:
            return
        try:
            self._file.write(data if room == len(data) else data[:room])
            self._file.flush()
            self._file_bytes += room
        except OSError:
            _log.warning("Failed to write to file; disabling file backing", exc_info=True)
            self._file_failed = True

    # ---- output -----------------------------------------------------------

    def getvalue(self) -> str:
        """The view: all of it, or head and tail with a notice naming the file."""
        with self._lock:
            return super().getvalue()

    def _output_path(self) -> str | None:
        """Where the full output is: the file, unless it failed or was removed."""
        if self._file_path and not self._file_failed and not self._removed:
            return self._file_path
        return None

    def _stats(self, dropped_chars: int | None = None) -> CaptureStats:
        if dropped_chars is None:
            kept = self._head_written + self._tail_chars
            dropped_chars = max(0, self._chars_written - kept) if self.was_truncated else 0
        return CaptureStats(
            path=self._output_path(),
            chars_written=self._chars_written,
            bytes_written=self._bytes_written,
            file_bytes=self._file_bytes,
            file_bytes_dropped=self._file_dropped,
            truncated=self.was_truncated,
            dropped_chars=dropped_chars,
        )

    def _where(self, stats: CaptureStats) -> str:
        """The notice's lines on where the full output is (each ends in a newline)."""
        if stats.path is None:
            return f"The {stats.dropped_chars:,} chars in the middle are not recoverable.\n"
        if self._notice is None:
            text = (
                f"The full untruncated output ({stats.chars_written:,} chars) is in: "
                f"{stats.path}\n"
                f"Read that file to see the {stats.dropped_chars:,} chars not shown here."
            )
        elif callable(self._notice):
            text = self._notice(stats)
        else:
            text = self._notice.format(
                quoted_path=json.dumps(stats.path), **dataclasses.asdict(stats)
            )
        text = text.rstrip("\n") + "\n"
        if stats.file_bytes_dropped:
            text += (
                f"The file is incomplete: it holds the first {stats.file_bytes:,} bytes; "
                f"{stats.file_bytes_dropped:,} more bytes were over its "
                f"{self._max_file_bytes:,}-byte limit.\n"
            )
        return text

    def _format_truncation_notice(
        self, head: str, tail: str, total: int, head_chars: int, tail_chars: int, dropped: int
    ) -> str:
        """Name the file (in the caller's ``notice`` wording) in the truncation notice."""
        self._flush_file()
        return (
            f"<truncated-output>\n"
            f"Output too large ({total:,} chars). "
            f"Showing first {head_chars:,} and last {tail_chars:,} chars.\n"
            f"{self._where(self._stats(dropped))}\n"
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
    def path(self) -> str | None:
        """The file holding the full output; None if none is kept (failed or removed)."""
        return self._output_path()

    @property
    def file_path(self) -> str | None:
        """The file's path as created (None if creation failed); see also :attr:`path`."""
        return self._file_path

    @property
    def stats(self) -> CaptureStats:
        """Counts so far: chars and bytes received, file bytes, dropped bytes, truncated."""
        with self._lock:
            return self._stats()

    @property
    def bytes_written(self) -> int:
        """Bytes received (text writes count their UTF-8 encoding)."""
        return self._bytes_written

    @property
    def file_bytes(self) -> int:
        """Bytes in the file."""
        return self._file_bytes

    @property
    def file_bytes_dropped(self) -> int:
        """Bytes not written because of ``max_file_bytes``."""
        return self._file_dropped

    # ---- lifecycle --------------------------------------------------------

    def __enter__(self) -> "FileBackedTruncatingStringIO":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Close the file handle (the file stays on disk); idempotent.

        Flushes an incomplete trailing UTF-8 sequence from ``write_bytes``
        into the view. Intentionally does NOT call ``super().close()`` — the
        underlying ``io.StringIO`` must stay open so ``getvalue()`` remains
        usable after the file handle is closed.
        """
        with self._lock:
            text = self._decoder.decode(b"", final=True)
            if text:
                TruncatingStringIO.write(self, text)
            if self._file is not None and not self._file.closed:
                try:
                    self._file.close()
                except OSError:
                    pass

    def cleanup(self) -> None:
        """Close the file handle and remove the file from disk; idempotent."""
        with self._lock:
            self.close()
            if self._file_path is not None and not self._removed:
                self._removed = True
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
