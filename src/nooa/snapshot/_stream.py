# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``TextStream``: output that arrives over time, as one bounded ``Text`` snapshot.

A stream keeps its first ``head`` and last ``tail`` units (characters or
bytes) in memory, counts everything, and can spill every byte, up to
``SpillTarget.max_bytes``, to a file while it runs. ``snapshot()`` describes
it at any time, ``since(mark)`` describes what arrived after a mark (progress)
and ``merge`` joins two unread deltas. Byte streams keep the raw bytes, cut
on UTF-8 code point boundaries, so every count is exact.
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
from nooa.snapshot._model import Spill, Text, cut_text, whole_text

__all__ = ["DEFAULT_SPILL_BYTES", "SpillTarget", "TextStream", "merge"]

_log = logging.getLogger(__name__)

DEFAULT_SPILL_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class SpillTarget:
    """Where a stream writes its whole output while it runs.

    ``path=None``: a new temporary file. Only the first ``max_bytes`` bytes are
    written. ``always=False``: the file is deleted on ``close()`` when nothing
    was cut from memory (the snapshot then holds everything).
    """

    path: str | None = None
    max_bytes: int = DEFAULT_SPILL_BYTES
    always: bool = False


class TextStream:
    """Head, rolling tail, counter and optional spill file of one output stream.

    ``unit="bytes"``: ``write`` takes bytes (``str`` is encoded as UTF-8);
    ``unit="chars"``: it takes ``str`` only and raises ``TypeError`` for
    bytes (raw output belongs in a bytes stream, which keeps it as written).
    Memory stays at ``head + tail`` units plus the chunk being written.

    Threads: one writer. ``snapshot()`` and ``since()`` only read memory (and
    flush the spill file), but they are not synchronised with ``write``:
    call them from the writer's thread, or have the caller serialise them
    with writes. There is no lock (PR 1).
    """

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
        """Add ``data``: O(len(data)), memory bounded by ``head + tail``."""
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
            return whole_text(head + tail, length, self.unit, self._full(False))  # type: ignore[operator]
        if isinstance(head, bytes) and isinstance(tail, bytes):  # whole code points only
            head = head[: complete_prefix(head)]
            tail = tail[safe_start(tail, 0) :]
        return cut_text(head, tail, length, self.unit, "max_length", self._full(True))

    def snapshot(self) -> Text:
        """The whole stream so far (the final ``Text`` once closed)."""
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
        """What arrived after ``mark`` (0: the start), and the mark to pass next time.

        Bounded like the stream: what memory no longer holds is omitted. A
        byte stream's mark stops before an incomplete UTF-8 sequence.
        """
        if self._closed is not None:
            end = self._total
        else:
            self._flush()
            end = self._safe_total()
        mark = min(max(0, mark), end)
        return self._text(mark, end), end

    def close(self) -> Text:
        """The final ``Text``; closes the spill file (deleting it if nothing was cut)."""
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
    """``a`` then ``b`` as one ``Text`` of at most ``head`` + ``tail`` kept units.

    For unread progress deltas: ``length`` is ``a.length + b.length``, and
    what either delta had already omitted stays omitted. Each input keeps its
    own units (bytes stay bytes), so the counts are exact:
    ``length == len(head) + len(tail) + omitted``. A bytes cut keeps whole
    code points, as ``TextStream`` does.
    """
    if a.unit != b.unit:
        raise ValueError(f"cannot merge {a.unit} with {b.unit}")
    if head < 0 or tail < 0:
        raise ValueError("head and tail must be non-negative")
    unit = a.unit
    # The kept parts in order, ``None`` where an input omitted units.
    pieces: list[str | bytes | None] = []
    for text in (a, b):
        pieces += [text.head, None, text.tail] if not text.complete else [text.head, text.tail]
    pieces = [piece for piece in pieces if piece is None or len(piece)]
    length = a.length + b.length
    full = (
        b.full if a.full is not None and b.full is not None and a.full.path == b.full.path else None
    )
    empty = b"" if unit == "bytes" else ""

    def joined(parts: list[str | bytes | None]) -> str | bytes:
        return empty.join(parts)  # type: ignore[arg-type]

    if None not in pieces and sum(len(piece) for piece in pieces) <= head + tail:  # type: ignore[arg-type]
        return whole_text(joined(pieces), length, unit, full)

    head_parts: list[str | bytes | None] = []
    room, index = head, 0
    while index < len(pieces) and room > 0:
        piece = pieces[index]
        if piece is None:
            break
        if len(piece) <= room:
            head_parts.append(piece)
            room -= len(piece)
            index += 1
            continue
        cut = safe_end(piece, room) if isinstance(piece, bytes) else room
        head_parts.append(piece[:cut])
        pieces[index] = piece[cut:]  # what the tail may still keep of it
        break
    tail_parts: list[str | bytes | None] = []
    room, last = tail, len(pieces) - 1
    while last >= index and room > 0:
        piece = pieces[last]
        if piece is None:
            break
        if len(piece) <= room:
            tail_parts.insert(0, piece)
            room -= len(piece)
            last -= 1
            continue
        start = len(piece) - room
        tail_parts.insert(
            0, piece[safe_start(piece, start) if isinstance(piece, bytes) else start :]
        )
        break
    reason = a.reason or b.reason or "max_length"
    return cut_text(joined(head_parts), joined(tail_parts), length, unit, reason, full)
