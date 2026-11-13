# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A cell's stdout / stderr capture: a bounded ``nooa.snapshot.TextStream`` of characters.

``CaptureBuffer`` holds ``execute_code``'s
streams. It keeps the head and the tail (``CaptureConfig.max_stdout`` /
``max_stderr`` characters together, ``CaptureConfig.tail`` of them, or half,
from the end) and, when ``CaptureConfig.file_backed`` is set, also writes the
whole stream to an overflow file (``nooa.overflow``). ``getvalue()`` is ``render_output`` of the stream::

    HEAD
    [... N chars omitted; full output: PATH]
    TAIL

The stream opens its file only once it is cut, from the ``OverflowDir``
in ``capture_overflow`` when a session set one (Atom sets its own: the
files then follow the session's lifetime), else from one private
``OverflowDir`` of this process, made on first use (its files are removed
at interpreter exit). Either folder keeps the newest files, so an older
path may no longer exist.

``TruncationConfig.overflow_files`` False (the off switch) makes no file at
all, whatever ``file_backed`` says: the actor clears ``file_backed``
(``nooa.runtime.actor._make_capture_buffer``), and a session whose files are
off sets ``capture_overflow`` to ``NO_OVERFLOW_FILES``, which the buffer
itself honours, so a capture made by another entry point writes no file
either. A cut stream keeps its head and tail, and its line says ``full
output not kept``.

``capture_overflow`` is a context variable, not a parameter, because the
capture is made inside the core cell execution (``nooa.runtime.actor``) from
the static ``CaptureConfig``, which knows nothing of a session; a session
sets it for its cells next to the other cell context (``shell_owner``, the
working directory).
"""

from __future__ import annotations

import contextvars
import threading

from nooa.overflow import OverflowDir
from nooa.snapshot import Text, TextStream, render_output

__all__ = ["NO_OVERFLOW_FILES", "CaptureBuffer", "capture_overflow"]


class _NoOverflowFiles:
    """The type of ``NO_OVERFLOW_FILES`` (one instance)."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "NO_OVERFLOW_FILES"


NO_OVERFLOW_FILES = _NoOverflowFiles()
"""``capture_overflow``'s value for a session whose output files are off: captures keep memory only."""

capture_overflow: contextvars.ContextVar[OverflowDir | _NoOverflowFiles | None] = (
    contextvars.ContextVar("nooa_capture_overflow", default=None)
)
"""Where a file-backed cell capture writes, as the session set it.

- an ``OverflowDir``: the session's folder;
- ``NO_OVERFLOW_FILES``: the session's output files are off: no file, whatever
  ``file_backed`` says;
- ``None`` (unset, no session): the process's own folder.
"""

_own_folder: OverflowDir | None = None
_own_folder_lock = threading.Lock()


def _process_folder() -> OverflowDir:
    """The process's private ``OverflowDir`` for captures outside a session (made on first use)."""
    global _own_folder
    with _own_folder_lock:
        if _own_folder is None:
            _own_folder = OverflowDir()
        return _own_folder


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
        folder: OverflowDir | None = None
        if file_backed:
            session = capture_overflow.get()
            if session is None:
                folder = _process_folder()
            elif isinstance(session, OverflowDir):
                folder = session
            # NO_OVERFLOW_FILES: the session's files are off, memory only.
        self._hint = unrecoverable_hint
        self._stream = TextStream(unit="chars", head=limit - tail, tail=tail, overflow=folder)

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
        """The stream in ``render_output``'s notation (whole when nothing was cut), with the hint as its note."""
        return render_output(self._stream.snapshot(), note=self._hint)

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
        """The overflow file while the stream is cut, else ``None`` (``Text.full_path``)."""
        return self._stream.snapshot().full_path

    def close(self) -> None:
        """End the stream; its overflow file stays only when the output was cut."""
        self._stream.close()

    def isatty(self) -> bool:
        return False

    def readable(self) -> bool:
        return False

    def writable(self) -> bool:
        return True
