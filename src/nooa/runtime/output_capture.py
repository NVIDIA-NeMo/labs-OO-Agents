# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A cell's stdout / stderr capture: a bounded ``nooa.snapshot.TextStream`` of characters.

``CaptureBuffer`` replaces ``TruncatingStringIO`` for ``execute_code``'s
streams. It keeps the head and the tail (``CaptureConfig.max_stdout`` /
``max_stderr`` characters together, ``CaptureConfig.tail`` of them, or half,
from the end) and, when ``CaptureConfig.file_backed`` is set, also writes the
whole stream to an overflow file (``nooa.overflow``). ``getvalue()`` is ``render_output`` of the stream::

    HEAD
    [... N chars omitted; full output: PATH]
    TAIL

The file comes from the ``OverflowDir`` in ``capture_overflow`` when a
session set one (Atom sets its own: the files then follow the session's
lifetime, and the folder keeps the newest, so an older path may no longer
exist), else from a standalone ``OverflowFile()`` (a temporary file kept
when the output was cut).

``TruncationConfig.overflow_files`` False (the off switch) makes no file at
all, whatever ``file_backed`` says (``nooa.runtime.actor._make_capture_buffer``):
a cut stream keeps its head and tail, and its line says ``full output not
kept``.

``capture_overflow`` is a context variable, not a parameter, because the
capture is made inside the core cell execution (``nooa.runtime.actor``) from
the static ``CaptureConfig``, which knows nothing of a session; a session
sets it for its cells next to the other cell context (``shell_owner``, the
working directory).
"""

from __future__ import annotations

import contextvars

from nooa.overflow import OverflowDir, OverflowFile
from nooa.snapshot import Text, TextStream, render_output

__all__ = ["CaptureBuffer", "capture_overflow"]

capture_overflow: contextvars.ContextVar[OverflowDir | None] = contextvars.ContextVar(
    "nooa_capture_overflow", default=None
)
"""The ``OverflowDir`` a file-backed cell capture writes into (a session sets it); None: a temp file."""


class CaptureBuffer:
    """One captured stream: ``write``, ``getvalue`` (``render_output``), bounded memory.

    ``CaptureBuffer(limit, tail_chars=None, *, file_backed=False, unrecoverable_hint=None)``:

    - ``limit``: the characters kept (head and tail together).
    - ``tail_chars``: the characters of those kept from the end; ``None``:
      half (``limit // 2``).
    - ``file_backed``: also write the whole stream to a file (see the module).
    - ``unrecoverable_hint``: a sentence put after the cut line when the
      output was cut and no file keeps it.

    ``was_truncated``: more than ``limit`` characters were written.
    ``close()`` ends the stream (its file is kept only when cut);
    ``getvalue()`` still works after it, and later writes are ignored.
    """

    def __init__(
        self,
        limit: int,
        tail_chars: int | None = None,
        *,
        file_backed: bool = False,
        unrecoverable_hint: str | None = None,
    ) -> None:
        tail = tail_chars if tail_chars is not None else limit // 2
        file: OverflowFile | None = None
        if file_backed:
            folder = capture_overflow.get()
            file = folder.file() if folder is not None else OverflowFile()
        self._hint = unrecoverable_hint
        try:
            self._stream = TextStream(unit="chars", head=limit - tail, tail=tail, overflow=file)
        except BaseException:  # bad sizes: give the folder's unopened file back
            if file is not None:
                file.release()
            raise

    def write(self, s: str) -> int:
        """Append ``s``; returns ``len(s)``. Raises ``TypeError`` for a non-``str``."""
        if not isinstance(s, str):
            raise TypeError(f"string argument expected, got {type(s).__name__!r}")
        try:
            self._stream.write(s)
        except ValueError:
            pass  # closed: the cell's output is final
        return len(s)

    def writelines(self, lines: object) -> None:
        for line in lines:  # type: ignore[attr-defined]
            self.write(line)

    def flush(self) -> None:
        """Nothing to flush: ``getvalue`` reads the stream (and flushes its file)."""

    def text(self) -> Text:
        """The stream so far as a ``Text``."""
        return self._stream.snapshot()

    def getvalue(self) -> str:
        """The stream in ``render_output``'s notation (whole when nothing was cut)."""
        text = self._stream.snapshot()
        rendered = render_output(text)
        if text.complete or text.full is not None or not self._hint:
            return rendered
        # The cut line alone, to find it after the head (which may hold any text).
        line = render_output(Text(head="", length=text.omitted, unit="chars", reason=text.reason))
        line_end = (len(text.head) + 1 if text.head else 0) + len(line)
        return f"{rendered[:line_end]} {self._hint}{rendered[line_end:]}"

    @property
    def was_truncated(self) -> bool:
        """Whether more than ``limit`` characters were written."""
        return not self._stream.snapshot().complete

    @property
    def chars_written(self) -> int:
        """The characters written, cut or not."""
        return self._stream.snapshot().length

    @property
    def file_path(self) -> str | None:
        """The overflow file while the stream is cut, else ``None``."""
        full = self._stream.snapshot().full
        return full.path if full is not None else None

    def close(self) -> None:
        """End the stream; its overflow file stays only when the output was cut."""
        self._stream.close()

    def isatty(self) -> bool:
        return False

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True
