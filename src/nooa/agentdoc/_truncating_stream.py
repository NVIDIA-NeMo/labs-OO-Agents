# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""StringIO with hard character limit and truncation notices.

Used by pformat/truncating_pformat to bound memory during formatting of large objects.

Two flavors:

- :class:`TruncatingStringIO` — pure in-memory head/tail buffer (default).
- :class:`FileBackedTruncatingStringIO` — same truncation, but also streams
  all output to a temp file and includes the file path in the truncation
  notice, so the full untruncated output remains retrievable.
"""

import collections
import io
import logging
import os
import tempfile

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

    def __init__(self, limit: int = DEFAULT_LIMIT, tail_chars: int | None = None):
        """Initialize truncating buffer.

        Args:
            limit: Maximum characters to store (default: 50,000).
            tail_chars: Characters reserved for the tail window.
                        None = half of limit (50/50 split).
        """
        super().__init__()
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

        return self._format_truncation_notice(head, tail, total, head_chars, tail_chars, dropped)

    def _format_truncation_notice(
        self, head: str, tail: str, total: int, head_chars: int, tail_chars: int, dropped: int
    ) -> str:
        """Build the truncated-output wrapper.

        Subclasses override this to inject extra information (e.g. a file path).
        """
        return (
            f"<truncated-output>\n"
            f"Output too large ({total:,} chars). "
            f"Showing first {head_chars:,} and last {tail_chars:,} chars.\n"
            f"The {dropped:,} chars in the middle are not recoverable.\n\n"
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
    """TruncatingStringIO that also writes full output to a temp file.

    Behaves identically to :class:`TruncatingStringIO` for in-memory
    head/tail truncation, but additionally streams all written content to a
    temporary file on disk.  When truncated, ``getvalue()`` includes the
    file path in the notice so the user can inspect the full output.

    File I/O errors are handled gracefully — if the temp file cannot be
    created or written to, the stream falls back to pure in-memory behavior
    (identical to the parent class).

    The caller is responsible for cleanup via :meth:`cleanup` (which removes
    the temp file) or :meth:`close` (which closes the file handle but leaves
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
    ):
        super().__init__(limit=limit, tail_chars=tail_chars)
        self._file_failed = False
        self._file_path: str | None = None
        self._file: io.TextIOWrapper | None = None
        try:
            fd, path = tempfile.mkstemp(dir=dir, prefix=prefix, suffix=suffix)
            self._file = os.fdopen(fd, "w")
            self._file_path = path
        except OSError:
            _log.warning(
                "Failed to create temp file for output capture; falling back to in-memory",
                exc_info=True,
            )
            self._file_failed = True

    def write(self, s: str) -> int:
        """Write to both the temp file and the in-memory truncating buffer."""
        if self._file is not None and not self._file_failed:
            try:
                self._file.write(s)
                self._file.flush()
            except OSError:
                _log.warning("Failed to write to temp file; disabling file backing", exc_info=True)
                self._file_failed = True
        return super().write(s)

    def _format_truncation_notice(
        self, head: str, tail: str, total: int, head_chars: int, tail_chars: int, dropped: int
    ) -> str:
        """Include the temp file path in the truncation notice."""
        if self._file_path and not self._file_failed:
            self._flush_file()
            file_notice = (
                f"The full untruncated output ({total:,} chars) is in: {self._file_path}\n"
                f"Read that file to see the {dropped:,} chars not shown here.\n"
            )
        else:
            file_notice = f"The {dropped:,} chars in the middle are not recoverable.\n"

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
        """Flush the temp file, ignoring errors."""
        if self._file is not None and not self._file.closed:
            try:
                self._file.flush()
            except OSError:
                pass

    @property
    def file_path(self) -> str | None:
        """Path to the temp file, or None if file creation failed."""
        return self._file_path

    def close(self) -> None:
        """Close the temp file handle (file remains on disk).

        Intentionally does NOT call ``super().close()`` — the underlying
        ``io.StringIO`` must stay open so ``getvalue()`` remains usable
        after the temp file handle is closed.
        """
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass

    def cleanup(self) -> None:
        """Close the file handle and remove the temp file from disk."""
        self.close()
        if self._file_path is not None:
            try:
                os.unlink(self._file_path)
            except OSError:
                pass
