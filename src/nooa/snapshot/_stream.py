# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``TextStream``: output that arrives over time, as one bounded ``Text`` snapshot.

A stream keeps its first ``head`` and last ``tail`` units (characters or
bytes) in memory, counts everything, and can spill every byte, up to
``SpillTarget.max_bytes``, to a file while it runs. ``snapshot()`` describes
it at any time, ``since(mark)`` describes what arrived after a mark (progress)
and ``merge`` joins two unread deltas. Byte streams are cut on UTF-8 code
point boundaries and count the replacement characters decoding introduced.
"""

from __future__ import annotations

import codecs
import collections
import contextlib
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import IO, Literal

from nooa.snapshot._capture import decode_counted, safe_start
from nooa.snapshot._model import Reason, Spill, Text

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


def complete_prefix(data: bytes | bytearray) -> int:
    """Length of ``data`` without a trailing, incomplete UTF-8 sequence."""
    size = len(data)
    index = size - 1
    continuation = 0
    while index >= 0 and continuation < 4 and data[index] & 0xC0 == 0x80:
        index -= 1
        continuation += 1
    if index < 0 or continuation >= 4:
        return size  # not a sequence that more bytes could complete
    lead = data[index]
    if lead < 0x80:
        need = 1
    elif lead >> 5 == 0b110:
        need = 2
    elif lead >> 4 == 0b1110:
        need = 3
    elif lead >> 3 == 0b11110:
        need = 4
    else:
        return size
    return index if size - index < need else size


class TextStream:
    """Head, rolling tail, counter and optional spill file of one output stream.

    ``unit="bytes"``: ``write`` takes bytes (``str`` is encoded as UTF-8);
    ``unit="chars"``: it takes ``str`` (``bytes`` are decoded as UTF-8). Memory
    stays at ``head + tail`` units plus the chunk being written.

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
        self._spill_target = spill
        self._total = 0  # units written
        self._head_bytes = bytearray()
        self._tail_bytes = bytearray()
        self._head_chars: list[str] = []
        self._head_len = 0
        self._tail_chars: collections.deque[str] = collections.deque()
        self._tail_len = 0
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._replaced = 0  # replacements while decoding bytes written to a chars stream
        # Recent marks since() returned, with ``_replaced`` at each (chars streams).
        self._replaced_at: collections.OrderedDict[int, int] = collections.OrderedDict()
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
        if file is None or self._spill_target is None:
            return
        room = self._spill_target.max_bytes - self._kept
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
            raw = _encode(data) if isinstance(data, str) else bytes(data)
            self._spill(raw)
            self._write_bytes(raw)
            return
        if isinstance(data, (bytes, bytearray)):
            decoded = self._decoder.decode(bytes(data))
            self._replaced += decoded.count("\ufffd") - bytes(data).count(b"\xef\xbf\xbd")
            data = decoded
        if self._file is not None:
            self._spill(_encode(data))
        self._write_chars(data)

    def _write_bytes(self, data: bytes) -> None:
        self._total += len(data)
        room = self.head - len(self._head_bytes)
        if room > 0:
            self._head_bytes += data[:room]
            data = data[room:]
        if not data:
            return
        if self.tail == 0:
            return
        self._tail_bytes += data[-self.tail :]
        excess = len(self._tail_bytes) - self.tail
        if excess > 0:
            del self._tail_bytes[:excess]

    def _write_chars(self, data: str) -> None:
        self._total += len(data)
        room = self.head - self._head_len
        if room > 0:
            part = data[:room]
            self._head_chars.append(part)
            self._head_len += len(part)
            data = data[room:]
        if not data or self.tail == 0:
            return
        data = data[-self.tail :]
        self._tail_chars.append(data)
        self._tail_len += len(data)
        while self._tail_len > self.tail:
            oldest = self._tail_chars[0]
            excess = self._tail_len - self.tail
            if len(oldest) <= excess:
                self._tail_chars.popleft()
                self._tail_len -= len(oldest)
            else:
                self._tail_chars[0] = oldest[excess:]
                self._tail_len -= excess

    # --- describing -------------------------------------------------------

    def _memory(self) -> tuple[int, int]:
        """(units in head, units in tail)."""
        if self.unit == "bytes":
            return len(self._head_bytes), len(self._tail_bytes)
        return self._head_len, self._tail_len

    def _full(self, cut: bool) -> Spill | None:
        if self._path is None or self._spill_failed:
            return None
        if not cut and not (self._spill_target is not None and self._spill_target.always):
            return None
        return Spill(self._path, self._kept, self._produced)

    def _reason(self) -> Reason:
        if self._spill_target is not None and self._spill_failed:
            return "not_kept"
        if self._path is not None and self._kept < self._produced:
            return "file_limit"
        return "max_length"

    def _text(self, start: int, end: int, replaced: int | None = None) -> Text:
        """What memory holds of units ``[start, end)``, as a ``Text`` (reads only).

        ``replaced``: a chars stream's replacements in the window (default: all).
        """
        held_head, held_tail = self._memory()
        tail_from = self._total - held_tail
        length = max(0, end - start)
        first = max(start, tail_from)
        head_slice = slice(min(start, held_head), min(end, held_head))
        tail_slice = slice(first - tail_from, max(first, end) - tail_from)
        if self.unit == "bytes":
            head_bytes = bytes(self._head_bytes[head_slice])
            tail_bytes = bytes(self._tail_bytes[tail_slice])
            gap = length - len(head_bytes) - len(tail_bytes)
            if gap <= 0:
                text, replaced = decode_counted(head_bytes + tail_bytes)
                return Text(text, "", length, "bytes", True, None, self._full(False), replaced)
            head_bytes = head_bytes[: complete_prefix(head_bytes)]
            tail_bytes = tail_bytes[safe_start(tail_bytes, 0) :]
            head, replaced_head = decode_counted(head_bytes)
            tail, replaced_tail = decode_counted(tail_bytes)
            omitted = length - len(head_bytes) - len(tail_bytes)
            return Text(
                head,
                tail,
                length,
                "bytes",
                False,
                self._reason(),
                self._full(True),
                replaced_head + replaced_tail,
                omitted,
            )
        if replaced is None:
            replaced = self._replaced
        # Join without storing back: a read never rebinds the writer's state.
        head_all = "".join(self._head_chars)
        tail_all = "".join(self._tail_chars)
        head, tail = head_all[head_slice], tail_all[tail_slice]
        gap = length - len(head) - len(tail)
        if gap <= 0:
            return Text(head + tail, "", length, "chars", True, None, self._full(False), replaced)
        return Text(
            head,
            tail,
            length,
            "chars",
            False,
            self._reason(),
            self._full(True),
            replaced,
            gap,
        )

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
        held_head, held_tail = self._memory()
        if held_head + held_tail == self._total:
            last = (bytes(self._head_bytes[-4:]) + bytes(self._tail_bytes))[-4:]
        else:
            last = bytes(self._tail_bytes[-4:])
        return self._total - (len(last) - complete_prefix(last))

    def since(self, mark: int = 0) -> tuple[Text, int]:
        """What arrived after ``mark`` (0: the start), and the mark to pass next time.

        Bounded like the stream: what memory no longer holds is omitted. A
        byte stream's mark stops before an incomplete UTF-8 sequence.

        ``replaced`` counts the window only. A byte stream decodes the window
        itself; a chars stream remembers its count at the last
        ``_REMEMBERED_MARKS`` marks returned here. For an older or foreign
        mark it reports the count since the nearest remembered mark after it
        (0 if none): a lower bound, never more than the window holds.
        """
        if self._closed is not None:
            end = self._total
        else:
            self._flush()
            end = self._safe_total()
        mark = min(max(0, mark), end)
        if self.unit == "bytes":
            return self._text(mark, end), end
        text = self._text(mark, end, self._replaced - self._replaced_before(mark))
        remembered = self._replaced_at
        remembered[end] = self._replaced
        remembered.move_to_end(end)
        while len(remembered) > _REMEMBERED_MARKS:
            remembered.popitem(last=False)
        return text, end

    def _replaced_before(self, mark: int) -> int:
        """Replacements before ``mark`` (an upper bound when it isn't remembered)."""
        if mark == 0:
            return 0
        known = self._replaced_at.get(mark)
        if known is not None:
            return known
        later = [count for at, count in self._replaced_at.items() if at >= mark]
        return min(later) if later else self._replaced

    def close(self) -> Text:
        """The final ``Text``; closes the spill file (deleting it if nothing was cut)."""
        if self._closed is not None:
            return self._closed
        if self.unit == "chars":
            rest = self._decoder.decode(b"", final=True)
            if rest:
                self._replaced += rest.count("\ufffd")
                self._write_chars(rest)
                self._spill(rest.encode("utf-8"))
        file, self._file = self._file, None
        if file is not None:
            try:
                file.close()
            except OSError:
                self._fail()
        text = self._text(0, self._total)
        keep = self._spill_target is not None and self._spill_target.always
        if text.complete and not keep and self._path is not None and not self._spill_failed:
            with contextlib.suppress(OSError):
                os.unlink(self._path)
        self._closed = text
        return text

    def __enter__(self) -> TextStream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


_REMEMBERED_MARKS = 16
"""How many recent ``since()`` marks a chars stream keeps its replacement count for."""


def _encode(text: str) -> bytes:
    """UTF-8 for the file: raw bytes for surrogate escapes, else ``\\udXXX`` text."""
    try:
        return text.encode("utf-8", errors="surrogateescape")
    except UnicodeEncodeError:
        return text.encode("utf-8", errors="backslashreplace")


# --- merging deltas ----------------------------------------------------------------


def _units(text: str, unit: str) -> int:
    return len(text) if unit == "chars" else len(text.encode("utf-8", errors="surrogateescape"))


def _take_front(text: str, budget: int, unit: str) -> str:
    """The longest prefix of ``text`` of at most ``budget`` units (whole code points)."""
    if unit == "chars":
        return text[:budget]
    data = text.encode("utf-8", errors="surrogateescape")
    if len(data) <= budget:
        return text
    return data[: complete_prefix(data[:budget])].decode("utf-8", errors="surrogateescape")


def _take_back(text: str, budget: int, unit: str) -> str:
    """The longest suffix of ``text`` of at most ``budget`` units (whole code points)."""
    if budget <= 0:
        return ""
    if unit == "chars":
        return text[-budget:]
    data = text.encode("utf-8", errors="surrogateescape")
    if len(data) <= budget:
        return text
    return data[safe_start(data, len(data) - budget) :].decode("utf-8", errors="surrogateescape")


def merge(a: Text, b: Text, head: int, tail: int) -> Text:
    """``a`` then ``b`` as one ``Text`` of at most ``head`` + ``tail`` kept units.

    For unread progress deltas: ``length`` is ``a.length + b.length``, and
    what either delta had already omitted stays omitted.
    """
    if a.unit != b.unit:
        raise ValueError(f"cannot merge {a.unit} with {b.unit}")
    if head < 0 or tail < 0:
        raise ValueError("head and tail must be non-negative")
    unit = a.unit
    pieces: list[str | int] = []
    for text in (a, b):
        pieces.append(text.head)
        if not text.complete:
            pieces.append(max(1, text.omitted))
        pieces.append(text.tail)
    pieces = [piece for piece in pieces if piece != ""]
    length = a.length + b.length
    known = sum(_units(piece, unit) for piece in pieces if isinstance(piece, str))
    gaps = any(isinstance(piece, int) for piece in pieces)
    full = (
        b.full if a.full is not None and b.full is not None and a.full.path == b.full.path else None
    )
    replaced = a.replaced + b.replaced
    if not gaps and known <= head + tail:
        joined = "".join(piece for piece in pieces if isinstance(piece, str))
        return Text(joined, "", length, unit, True, None, full, replaced)

    head_parts: list[str] = []
    room = head
    while pieces and room > 0 and isinstance(pieces[0], str):
        piece = pieces[0]
        assert isinstance(piece, str)
        part = _take_front(piece, room, unit)
        if not part:
            break
        head_parts.append(part)
        room -= _units(part, unit)
        if part == piece:
            pieces.pop(0)
        else:
            pieces[0] = piece[len(part) :]
            break
    tail_parts: list[str] = []
    room = tail
    while pieces and room > 0 and isinstance(pieces[-1], str):
        piece = pieces[-1]
        assert isinstance(piece, str)
        part = _take_back(piece, room, unit)
        if not part:
            break
        tail_parts.insert(0, part)
        room -= _units(part, unit)
        if part == piece:
            pieces.pop()
        else:
            break
    new_head, new_tail = "".join(head_parts), "".join(tail_parts)
    omitted = max(0, length - _units(new_head, unit) - _units(new_tail, unit))
    reason = a.reason or b.reason or "max_length"
    return Text(new_head, new_tail, length, unit, False, reason, full, replaced, max(1, omitted))
