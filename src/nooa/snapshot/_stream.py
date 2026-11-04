# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Output that arrives over time: ``TextStream``, ``SpillTarget`` and ``merge``.

The second producer of snapshots, next to ``capture``: it builds a ``Text``
incrementally, so the whole output is never held in memory. It reads no
live objects and does not import the capture layers; it shares only the
nodes (``_model``) and the UTF-8 cut points (``_codec``).
"""

from __future__ import annotations

import collections
import contextlib
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import IO, Literal

from nooa.snapshot._codec import complete_prefix, safe_end, safe_start
from nooa.snapshot._model import Spill, Text

__all__ = ["SpillTarget", "TextStream", "merge"]

_log = logging.getLogger(__name__)

DEFAULT_SPILL_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class SpillTarget:
    """Where and how much a ``TextStream`` spills: the file that keeps its whole output.

    Fields:

    - ``path`` (default ``None``): the file to write; an existing file is
      truncated. ``None``: a new temporary file (``nooa-output-*.txt`` in
      the system temporary directory).
    - ``max_bytes`` (default 64 MiB): only the first ``max_bytes`` bytes are
      written; ``Spill.cut`` then says the file is cut. Non-negative.
    - ``always`` (default ``False``): keep the file even when nothing was
      cut. ``False``: ``close()`` deletes it when the final ``Text`` is
      complete (the snapshot then holds everything).

    The stream owns the file: it never raises for a file error. When the
    file cannot be created, written or flushed, the stream logs a warning,
    removes the partial file and keeps memory only (``Text.full`` is
    ``None``).
    """

    path: str | None = None
    max_bytes: int = DEFAULT_SPILL_BYTES
    always: bool = False


class TextStream:
    """One output stream as a bounded ``Text``: head, rolling tail, exact count, optional spill file.

    ``TextStream(*, unit, head, tail, spill=None)``, all keyword-only:

    - ``unit``: ``"chars"`` (``write`` takes ``str``) or ``"bytes"``
      (``write`` takes ``bytes``, ``bytearray`` or ``str``, which is encoded
      as UTF-8).
    - ``head``: the units kept from the start.
    - ``tail``: the units kept from the end (a rolling window).
    - ``spill`` (default ``None``): a ``SpillTarget`` to also write the
      whole output to a file while the stream runs; ``None``: memory only.

    Methods: ``write(data)``, ``snapshot()``, ``since(mark)``, ``close()``.
    A ``TextStream`` is also a context manager that closes on exit.

    Guarantees:

    - Bounded memory: at most ``head + tail`` units plus the chunk being
      written; ``write`` is O(len(data)).
    - Exact accounting: ``Text.length`` counts every unit written, and
      ``Text.omitted`` every unit not kept.
    - UTF-8-safe cuts: a bytes stream keeps the raw bytes, and a cut
      ``Text`` of it starts and ends on code point boundaries. Bytes are
      never decoded here; a chars stream keeps its ``str`` as written.
    - The spill file holds the first ``SpillTarget.max_bytes`` bytes of the
      output (``Text.full``); a file error never raises (see
      ``SpillTarget``).

    Threads: one writer. ``snapshot()`` and ``since()`` only read memory
    (and flush the spill file) but are not synchronised with ``write``:
    call them from the writer's thread, or serialise them with writes.
    There is no lock.

    Raises ``ValueError`` for an unknown ``unit``, a negative ``head`` or
    ``tail``, or a negative ``spill.max_bytes``.

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
        spill: SpillTarget | None = None,
    ) -> None:
        if unit not in ("chars", "bytes"):
            raise ValueError(f"unit must be 'chars' or 'bytes', got {unit!r}")
        if head < 0 or tail < 0:
            raise ValueError("head and tail must be non-negative")
        if spill is not None and spill.max_bytes < 0:
            raise ValueError("spill max_bytes must be non-negative")
        self.unit: Literal["chars", "bytes"] = unit
        self.head = head
        self.tail = tail
        self._spill_max = spill.max_bytes if spill is not None else 0
        self._keep_file = spill is not None and spill.always  # kept even when nothing was cut
        self._total = 0  # units written
        self._window = _Window(head, tail, b"" if unit == "bytes" else "")
        # The last bytes (up to 4) of a bytes stream, whatever head and tail
        # keep: ``since()`` stops before an incomplete UTF-8 sequence in them.
        self._last = b""
        self._closed: Text | None = None
        self._file: IO[bytes] | None = None
        self._path: str | None = None
        self._spill_failed = False
        self._kept = 0  # bytes written to the file
        self._produced = 0  # bytes produced
        if spill is not None:
            self._open(spill)

    # --- spilling ---------------------------------------------------------

    def _open(self, spill: SpillTarget) -> None:
        try:
            if spill.path is not None:
                self._file = open(spill.path, "wb")  # noqa: SIM115 - closed in close()
                self._path = spill.path
            else:
                fd, path = tempfile.mkstemp(prefix="nooa-output-", suffix=".txt")
                self._file = os.fdopen(fd, "wb")
                self._path = path
        except OSError:
            _log.warning("could not create the spill file; keeping memory only", exc_info=True)
            self._fail()

    def _fail(self) -> None:
        self._spill_failed = True
        file, self._file = self._file, None
        if file is not None:
            with contextlib.suppress(OSError):
                file.close()
        if self._path is not None:
            with contextlib.suppress(OSError):
                os.unlink(self._path)

    def _spill(self, data: bytes) -> None:
        self._produced += len(data)
        file = self._file
        if file is None:
            return
        room = self._spill_max - self._kept
        if room <= 0:
            return
        part = data if len(data) <= room else data[:room]
        try:
            file.write(part)
        except OSError:
            _log.warning("could not write the spill file; keeping memory only", exc_info=True)
            self._fail()
            return
        self._kept += len(part)

    def _flush(self) -> None:
        """Push buffered spill bytes to the file, so it holds what ``Spill.kept`` says."""
        file = self._file
        if file is None:
            return
        try:
            file.flush()
        except OSError:
            _log.warning("could not flush the spill file; keeping memory only", exc_info=True)
            self._fail()

    # --- writing ----------------------------------------------------------

    def write(self, data: str | bytes | bytearray) -> None:
        """Append ``data`` to the stream: O(len(data)), memory bounded by ``head + tail``.

        A chars stream takes ``str`` only and raises ``TypeError`` for
        anything else (write raw output to a bytes stream, which keeps it as
        written). A bytes stream takes ``bytes`` or ``bytearray``; a ``str``
        is encoded as UTF-8 (surrogate escapes as their raw bytes, a lone
        surrogate as ``\\udXXXX`` text). While a spill file is active,
        ``data`` is also written to it, up to ``SpillTarget.max_bytes``
        (a chars stream encodes it as a bytes stream would). Raises
        ``ValueError`` after ``close()``.
        """
        if self._closed is not None:
            raise ValueError("write to a closed TextStream")
        if self.unit == "bytes":
            data = _encode(data) if isinstance(data, str) else bytes(data)
            self._spill(data)
            self._last = (self._last + data[-4:])[-4:]
        elif not isinstance(data, str):
            raise TypeError(
                f"a chars TextStream takes str, not {type(data).__name__}; "
                "write bytes to a bytes stream"
            )
        elif self._file is not None:
            self._spill(_encode(data))
        self._total += len(data)
        self._window.write(data)

    # --- describing -------------------------------------------------------

    def _full(self, cut: bool) -> Spill | None:
        if self._path is None or self._spill_failed:
            return None
        if not cut and not self._keep_file:
            return None
        return Spill(self._path, self._kept, self._produced)

    def _text(self, start: int, end: int) -> Text:
        """What memory holds of units ``[start, end)``, as a ``Text`` (reads only)."""
        window = self._window
        tail_from = self._total - window.tail_len
        length = max(0, end - start)
        first = max(start, tail_from)
        head = window.head()[min(start, window.head_len) : min(end, window.head_len)]
        tail = window.tail()[first - tail_from : max(first, end) - tail_from]
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

        Callable at any time while the stream runs. It first flushes the
        spill file, so ``Text.full.kept`` bytes are on disk.
        """
        if self._closed is not None:
            return self._closed
        self._flush()
        return self._text(0, self._total)

    def _safe_total(self) -> int:
        """The units written, without a trailing incomplete UTF-8 sequence (bytes)."""
        if self.unit != "bytes":
            return self._total
        # ``_last`` holds the stream's last bytes even when head and tail keep
        # fewer than a code point's width (a short tail, ``tail=0``).
        last = self._last
        return self._total - (len(last) - complete_prefix(last))

    def since(self, mark: int = 0) -> tuple[Text, int]:
        """What arrived after ``mark``, and the mark to pass next time: ``(Text, mark)``.

        ``mark`` (default 0, the start) is a unit count returned by an
        earlier ``since``; it is clamped to what was written. The ``Text``
        has ``length`` = the units after ``mark`` and keeps what memory
        still holds of them (in the head or the rolling tail); the rest is
        omitted. A bytes stream's new mark stops before an incomplete UTF-8
        sequence at the end, so the next delta starts with it. Join unread
        deltas with ``merge``.

        >>> from nooa.snapshot import TextStream
        >>> stream = TextStream(unit="chars", head=3, tail=3)
        >>> stream.write("hello")
        >>> first, mark = stream.since()
        >>> stream.write(" world")
        >>> stream.since(mark)
        (Text(head='', tail='rld', length=6, unit='chars', reason='max_length', full=None), 11)
        """
        if self._closed is not None:
            end = self._total
        else:
            self._flush()
            end = self._safe_total()
        mark = min(max(0, mark), end)
        return self._text(mark, end), end

    def close(self) -> Text:
        """Finish the stream and return the final ``Text``; later calls return the same one.

        Closes the spill file and deletes it when the ``Text`` is complete,
        unless ``SpillTarget.always`` is set. After ``close()``, ``write``
        raises ``ValueError``; ``snapshot()`` and ``since()`` still work.
        """
        if self._closed is not None:
            return self._closed
        file, self._file = self._file, None
        if file is not None:
            try:
                file.close()
            except OSError:
                self._fail()
        text = self._text(0, self._total)
        if (
            text.complete
            and not self._keep_file
            and self._path is not None
            and not self._spill_failed
        ):
            with contextlib.suppress(OSError):
                os.unlink(self._path)
        self._closed = text
        return text

    def __enter__(self) -> TextStream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class _Window:
    """The first ``head`` and the last ``tail`` units of one stream, as written parts.

    The same for both units: a part is ``bytes`` in a bytes stream and
    ``str`` in a chars stream. Memory stays at ``head + tail`` units; a read
    joins the parts without storing the result back.
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
        room = self._head_room - self.head_len
        if room > 0 and data:
            part = data[:room]
            self._head.append(part)
            self.head_len += len(part)
            data = data[room:]
        if not data or self._tail_room == 0:
            return
        data = data[-self._tail_room :]
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
        return self._empty.join(self._head)  # type: ignore[arg-type]

    def tail(self) -> str | bytes:
        return self._empty.join(self._tail)  # type: ignore[arg-type]


def _encode(text: str) -> bytes:
    """UTF-8 for the file: raw bytes for surrogate escapes, else ``\\udXXX`` text."""
    try:
        return text.encode("utf-8", errors="surrogateescape")
    except UnicodeEncodeError:
        return text.encode("utf-8", errors="backslashreplace")


# --- merging deltas ----------------------------------------------------------------


def merge(a: Text, b: Text, head: int, tail: int) -> Text:
    """``a`` followed by ``b`` as one ``Text`` keeping at most ``head`` + ``tail`` units.

    For unread progress deltas (two results of ``TextStream.since``).

    - ``a``, ``b``: two ``Text`` of the same unit.
    - ``head``, ``tail``: the units the result may keep from its start and
      its end.

    The result's ``length`` is ``a.length + b.length``. What either input
    already omitted stays omitted: the head comes only from what was kept
    before the first gap, the tail only from what was kept after the last
    one. Counts are exact in the inputs' unit (bytes stay bytes). A bytes
    cut keeps whole code points and never moves across the joint of two
    inputs' parts. ``reason``: ``None`` when everything fits, else the
    first input's reason, or ``"max_length"``. ``full``: ``b.full`` when
    both inputs spilled to the same path, else ``None``.

    Raises ``ValueError`` when the units differ or ``head`` / ``tail`` is
    negative.

    >>> from nooa.snapshot import Text, merge
    >>> merge(Text(head="hello", length=5, unit="chars"), Text(head=" world", length=6, unit="chars"), 3, 3)
    Text(head='hel', tail='rld', length=11, unit='chars', reason='max_length', full=None)
    """
    if a.unit != b.unit:
        raise ValueError(f"cannot merge {a.unit} with {b.unit}")
    if head < 0 or tail < 0:
        raise ValueError("head and tail must be non-negative")
    length = a.length + b.length
    same_file = a.full is not None and b.full is not None and a.full.path == b.full.path
    full = b.full if same_file else None
    # The joints: where one input's part ends and the next begins. A bytes
    # cut never moves across one, so each part is cut on its own code points.
    gapless = a.complete and b.complete
    if gapless:
        whole = a.head + b.head  # type: ignore[operator]
        if len(whole) <= head + tail:
            return Text(head=whole, length=length, unit=a.unit, full=full)
        left = right = whole
        left_joints = right_joints = (len(a.head),)
    else:
        left = a.head if not a.complete else a.head + b.head  # type: ignore[operator]
        right = b.tail if not b.complete else a.tail + b.head  # type: ignore[operator]
        left_joints = () if not a.complete else (len(a.head),)
        right_joints = () if not b.complete else (len(a.tail),)
    kept_head = left[: _head_end(left, head, left_joints)]
    low = len(kept_head) if gapless else 0  # the tail starts after the kept head
    start = _tail_start(right, max(low, len(right) - tail), (*right_joints, low))
    return Text(
        head=kept_head,
        tail=right[start:],
        length=length,
        unit=a.unit,
        reason=a.reason or b.reason or "max_length",
        full=full,
    )


def _head_end(text: str | bytes, end: int, joints: tuple[int, ...]) -> int:
    """Where a head of at most ``end`` units of ``text`` stops (bytes: on a code point, not before a joint)."""
    if isinstance(text, str):
        return min(end, len(text))
    return max([safe_end(text, end), *(joint for joint in joints if joint <= end)])


def _tail_start(text: str | bytes, start: int, joints: tuple[int, ...]) -> int:
    """Where a tail from ``start`` begins (bytes: on a code point, not past a joint)."""
    if isinstance(text, str):
        return start
    return min([safe_start(text, start), *(joint for joint in joints if joint >= start)])
