# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Output that arrives over time: ``TextStream``.

The second producer of snapshots, next to ``capture``: it builds a ``Text``
incrementally, so the whole output is never held in memory. It reads no
live objects and does not import the capture layers; it shares only the
nodes (``_model``) and the UTF-8 cut points (``_codec``).

A ``TextStream`` given an ``OverflowDir`` (``nooa.overflow``) opens its
overflow file only once its output is cut, and bounds it.
"""

from __future__ import annotations

import collections
import logging
import re
from typing import Literal

from nooa.overflow import OverflowDir, OverflowWriter
from nooa.snapshot._codec import complete_prefix, encode_text, safe_end, safe_start
from nooa.snapshot._model import Overflow, Text

__all__ = ["TextStream"]

_log = logging.getLogger(__name__)
_SURROGATE = re.compile("[\ud800-\udfff]")


class TextStream:
    """One output stream as a bounded ``Text``: head, rolling tail, exact count, optional overflow file.

    ``TextStream(*, unit, head, tail, overflow=None, overflow_bytes=None)``, all keyword-only:

    - ``unit``: ``"chars"`` (``write`` takes ``str``) or ``"bytes"``
      (``write`` takes ``bytes``, ``bytearray``, ``memoryview`` or ``str``,
      which is encoded as UTF-8).
    - ``head``: the units kept from the start.
    - ``tail``: the units kept from the end (a rolling window).
    - ``overflow`` (default ``None``): an ``OverflowDir`` to also write the
      whole output to a file; ``None``: memory only. The stream opens its
      file (``overflow.open(max_bytes=overflow_bytes)``) only when its
      output first exceeds ``head + tail`` units, and then writes all of it
      from the start; an output that is never cut makes no file.
    - ``overflow_bytes`` (default ``None``: the folder's ``file_bytes``):
      the cap of this stream's file, at most the folder's ``file_bytes``.

    Methods: ``write(data)``, ``snapshot()``, ``close()``.
    A ``TextStream`` is also a context manager that closes on exit.

    Guarantees:

    - Bounded memory: the stream retains at most ``head + tail`` units
      between writes, whatever was written. During one ``write`` it may
      also need memory proportional to that chunk (a ``bytearray`` or
      ``memoryview`` is copied to ``bytes``, a ``str`` written to a bytes
      stream or to a chars stream with a file is encoded), released when the
      call returns. ``write`` is O(len(data)).
    - Exact accounting: ``Text.length`` counts every unit written, and
      ``Text.omitted`` every unit not kept.
    - UTF-8-safe cuts: a bytes stream keeps the raw bytes, and a cut
      ``Text`` of it starts and ends on code point boundaries. Bytes are
      never decoded here; a chars stream keeps its ``str`` as written.
    - The overflow file holds the first ``overflow_bytes`` bytes of the
      output (``Text.full``; a chars stream's file is UTF-8, cut on a code
      point boundary, except that a ``surrogateescape`` character is written
      as its original byte, so such text need not give valid UTF-8); a file
      error never raises (see ``OverflowDir``), and when the folder gives
      no file (its budget is in use, it is closed) the stream keeps memory
      only.

    Threads: one writer. ``snapshot()`` only reads memory but is not
    synchronised with ``write``: call it from
    the writer's thread, or serialise it with writes.
    There is no lock.

    Raises ``ValueError`` for an unknown ``unit``, or a negative ``head``,
    ``tail`` or ``overflow_bytes``.

    >>> from nooa.snapshot import TextStream, render_output
    >>> stream = TextStream(unit="bytes", head=4, tail=4)
    >>> stream.write(b"0123456789abcdef")
    >>> print(render_output(stream.close()))
    0123
    [... 8 bytes omitted; full output not kept]
    cdef
    """

    unit: Literal["chars", "bytes"]
    """``"chars"`` or ``"bytes"``, as given."""
    head: int
    """The units kept from the start, as given."""
    tail: int
    """The units kept from the end, as given."""

    def __init__(
        self,
        *,
        unit: Literal["chars", "bytes"],
        head: int,
        tail: int,
        overflow: OverflowDir | None = None,
        overflow_bytes: int | None = None,
    ) -> None:
        if unit not in ("chars", "bytes"):
            raise ValueError(f"unit must be 'chars' or 'bytes', got {unit!r}")
        if head < 0 or tail < 0:
            raise ValueError("head and tail must be non-negative")
        if overflow_bytes is not None and overflow_bytes < 0:
            raise ValueError("overflow_bytes must be non-negative")
        self.unit: Literal["chars", "bytes"] = unit
        self.head = head
        self.tail = tail
        self._folder = overflow  # until the output is first cut: then its file opens
        self._file_bytes = overflow_bytes
        self._file_max = 0
        self._total = 0  # units written
        self._window = _Window(head, tail, b"" if unit == "bytes" else "")
        self._closed: Text | None = None
        self._file: OverflowWriter | None = None  # the open overflow file, if any
        self._path: str | None = None  # its path, for ``Text.full``
        self._file_failed = False
        self._kept = 0  # bytes written to the file
        self._produced = 0  # bytes produced
        # A chars stream, until its file opens: the (start, end) of each write holding a
        # surrogate. Its bytes, and where a cut file ends in them, depend on the write
        # as a whole (``encode_text``, ``safe_end``), so the file gets it on its own.
        self._spans: list[tuple[int, int]] = []

    # --- the overflow file ---------------------------------------------------------

    def _open(self) -> None:
        """The output is cut for the first time: open the overflow file and write it all so far.

        Until now memory held the whole output (at most ``head + tail``
        units), so the file gets exactly what writing it from the start
        would have given.
        """
        folder, self._folder = self._folder, None
        writer = folder.open(max_bytes=self._file_bytes) if folder is not None else None
        if writer is None:
            self._file_failed = True
            return
        self._file, self._path, self._file_max = writer, writer.path, writer.max_bytes
        so_far = self._window.head() + self._window.tail()  # type: ignore[operator]
        if isinstance(so_far, bytes):
            self._to_file(so_far)
            return
        start = 0  # the writes between spans are surrogate-free: valid UTF-8, joined as is
        for span_start, span_end in [*self._spans, (len(so_far), len(so_far))]:
            for part in (so_far[start:span_start], so_far[span_start:span_end]):
                if part:
                    data = encode_text(part)
                    self._produced += len(data)
                    self._to_file(data)
            start = span_end
        self._spans = []

    def _fail(self) -> None:
        """Give up the overflow file after a file error: remove it and keep memory only (``Text.full`` is ``None``)."""
        self._file_failed = True
        file, self._file = self._file, None
        if file is not None:
            file.finish(delete=True)

    def _to_file(self, data: bytes) -> None:
        """Write what fits of ``data`` to the overflow file.

        The file keeps a prefix of the output, at most ``max_bytes``; a chars
        stream's prefix ends on a code point boundary. A write error is
        logged and ends the file (``_fail``); it never raises.
        """
        writer = self._file
        if writer is None:
            return
        room = self._file_max - self._kept
        if room <= 0:
            return
        part = data
        if len(data) > room:
            # A chars stream's file is never cut inside a code point: it ends
            # before one that does not fit whole, and takes nothing after it (the
            # file is a prefix of the output).
            part = data[: safe_end(data, room)] if self.unit == "chars" else data[:room]
            self._file_max = self._kept + len(part)
        try:
            self._kept += writer.write(part)
        except OSError:
            _log.warning("could not write the overflow file; keeping memory only", exc_info=True)
            self._fail()

    # --- writing ----------------------------------------------------------

    def write(self, data: str | bytes | bytearray | memoryview) -> None:
        """Append ``data`` to the stream, in O(len(data)) time.

        Afterwards the stream retains at most ``head + tail`` units. While it
        runs, the call may hold one more copy of ``data`` (a ``bytes`` copy of a
        ``bytearray`` / ``memoryview``, or the UTF-8 encoding of a ``str`` for a
        bytes stream or an overflow file).

            A chars stream takes ``str`` only and raises ``TypeError`` for
            anything else (write raw output to a bytes stream, which keeps it as
            written). A bytes stream takes ``bytes``, ``bytearray`` or
            ``memoryview`` (its raw bytes) and raises ``TypeError`` for anything
            else but ``str``; a ``str`` is encoded as UTF-8 (surrogate escapes as
            their raw bytes, a lone surrogate as ``\\udXXXX`` text). While an overflow file is active,
            ``data`` is also written to it, up to ``overflow_bytes``
            (a chars stream encodes it as a bytes stream would). Raises
            ``ValueError`` after ``close()``.
        """
        if self._closed is not None:
            raise ValueError("write to a closed TextStream")
        if self.unit == "bytes":
            if isinstance(data, str):
                data = encode_text(data)
            elif isinstance(data, (bytes, bytearray, memoryview)):
                data = bytes(data)
            else:  # ``bytes(10)`` would be ten NUL bytes, ``bytes([1])`` a made-up byte
                raise TypeError(
                    f"a bytes TextStream takes bytes, bytearray, memoryview or str, "
                    f"not {type(data).__name__}"
                )
        elif not isinstance(data, str):
            raise TypeError(
                f"a chars TextStream takes str, not {type(data).__name__}; "
                "write bytes to a bytes stream"
            )
        if self._folder is not None:
            if self._total + len(data) > self.head + self.tail:
                self._open()  # this write cuts the output: write what came before first
            elif isinstance(data, str) and not data.isascii() and _SURROGATE.search(data):
                self._spans.append((self._total, self._total + len(data)))
        if isinstance(data, bytes):
            self._produced += len(data)
            self._to_file(data)
        elif self._file is not None:
            encoded = encode_text(data)
            self._produced += len(encoded)
            self._to_file(encoded)
        self._total += len(data)
        self._window.write(data)

    # --- describing -------------------------------------------------------

    def _full(self, cut: bool) -> Overflow | None:
        """The ``Overflow`` for ``Text.full``: only for a ``cut`` text whose overflow file is still good."""
        if not cut or self._path is None or self._file_failed:
            return None
        return Overflow(self._path, self._kept, self._produced)

    def _text(self) -> Text:
        """What memory holds of the whole stream, as a ``Text`` (reads only)."""
        length = self._total
        head, tail = self._window.head(), self._window.tail()
        if length <= len(head) + len(tail):
            whole = head + tail  # type: ignore[operator]
            return Text(head=whole, length=length, unit=self.unit, full=self._full(False))
        if isinstance(head, bytes) and isinstance(tail, bytes):  # whole code points only
            head = head[: complete_prefix(head)]
            tail = tail[safe_start(tail, 0) :]
        return Text(
            head=head,
            tail=tail,
            length=length,
            unit=self.unit,
            reason="max_length",
            full=self._full(True),
        )

    def snapshot(self) -> Text:
        """The whole stream so far, as a ``Text``; once closed, the final one.

        Callable at any time while the stream runs. The overflow file is
        unbuffered, so its ``Text.full.kept`` bytes are already on disk.
        """
        if self._closed is not None:
            return self._closed
        return self._text()

    def close(self) -> Text:
        """Finish the stream and return the final ``Text``; later calls return the same one.

        Closes the overflow file and deletes it when the ``Text`` is complete.
        After ``close()``, ``write`` raises ``ValueError``; ``snapshot()``
        still works.
        """
        if self._closed is not None:
            return self._closed
        text = self._text()
        writer, self._file = self._file, None
        if writer is not None and not writer.finish(delete=text.complete) and not text.complete:
            self._file_failed = True  # the file did not close cleanly (logged, deleted)
            text = self._text()
        self._closed = text
        return text

    def __enter__(self) -> TextStream:
        """Return the stream itself, for ``with TextStream(...) as stream:``."""
        return self

    def __exit__(self, *exc: object) -> None:
        """``close()`` the stream on leaving the ``with`` block."""
        self.close()


class _Window:
    """The first ``head`` and the last ``tail`` units of one stream, as written parts.

    The same for both units: a part is ``bytes`` in a bytes stream and
    ``str`` in a chars stream. Memory stays at ``head + tail`` units; a read
    joins the parts without storing the result back, so reads and writes
    can interleave.

    Args:
        head: the units kept from the start.
        tail: the units kept from the end.
        empty: ``""`` or ``b""``, the empty part of the stream's unit.
    """

    def __init__(self, head: int, tail: int, empty: str | bytes) -> None:
        self._empty = empty
        self._head_room = head
        self._tail_room = tail
        self._head: list[str | bytes] = []
        self._tail: collections.deque[str | bytes] = collections.deque()
        self.head_len = 0  # units kept in the head
        self.tail_len = 0  # units kept in the tail

    def write(self, data: str | bytes) -> None:
        """Keep what ``data`` adds to the head, and its last units in the tail.

        Copies at most ``head + tail`` units of ``data``; the oldest tail
        units are dropped to keep the tail at ``tail`` units.
        """
        room = self._head_room - self.head_len
        if room > 0 and data:
            part = data[:room]
            self._head.append(part)
            self.head_len += len(part)
        else:
            room = 0
        if self._tail_room == 0 or len(data) <= room:
            return
        # One slice: copying ``data[room:]`` first would copy everything past
        # the head of a large write only to keep its last ``tail`` units.
        data = data[max(room, len(data) - self._tail_room) :]
        self._tail.append(data)
        self.tail_len += len(data)
        while self.tail_len > self._tail_room:
            oldest = self._tail[0]
            excess = self.tail_len - self._tail_room
            if len(oldest) <= excess:
                self._tail.popleft()
                self.tail_len -= len(oldest)
            else:
                self._tail[0] = oldest[excess:]
                self.tail_len -= excess

    def head(self) -> str | bytes:
        """The kept head, joined (at most ``head`` units)."""
        return self._empty.join(self._head)  # type: ignore[arg-type]

    def tail(self) -> str | bytes:
        """The kept tail, joined (at most ``tail`` units, the last ones written after the head)."""
        return self._empty.join(self._tail)  # type: ignore[arg-type]
