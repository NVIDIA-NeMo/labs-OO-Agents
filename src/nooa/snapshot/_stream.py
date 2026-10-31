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

from nooa.snapshot._codec import (
    REPLACEMENT,
    REPLACEMENT_BYTES,
    complete_prefix,
    decode_counted,
    safe_start,
)
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
        # The last bytes (up to 2) written to a chars stream: a genuine U+FFFD
        # (EF BF BD) split across writes is still recognised as genuine.
        self._carry = b""
        # The last bytes (up to 4) of a bytes stream, whatever head and tail
        # keep: ``since()`` stops before an incomplete UTF-8 sequence in them.
        self._last = b""
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
            raw = bytes(data)
            decoded = self._decoder.decode(raw)
            # Genuine U+FFFD in the input, counting one that began in an
            # earlier write (the 2-byte carry cannot hold a whole one).
            joined = self._carry + raw
            self._carry = joined[-2:]
            self._replaced += decoded.count(REPLACEMENT) - joined.count(REPLACEMENT_BYTES)
            data = decoded
        else:
            # A str chunk is a boundary: bytes before it can't complete with
            # bytes after it. An incomplete sequence becomes U+FFFD first.
            pending = self._decoder.decode(b"", final=True)
            self._carry = b""
            if pending:
                self._replaced += pending.count(REPLACEMENT)
                data = pending + data
        if self._file is not None:
            self._spill(_encode(data))
        self._write_chars(data)

    def _write_bytes(self, data: bytes) -> None:
        self._total += len(data)
        self._last = (self._last + data[-4:])[-4:]
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
                return Text(
                    text,
                    "",
                    length,
                    "bytes",
                    True,
                    None,
                    self._full(False),
                    replaced,
                    head_units=len(head_bytes) + len(tail_bytes),
                    tail_units=0,
                )
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
                len(head_bytes),
                len(tail_bytes),
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
        # ``_last`` holds the stream's last bytes even when head and tail keep
        # fewer than a code point's width (a short tail, ``tail=0``).
        last = self._last
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
                self._replaced += rest.count(REPLACEMENT)
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


@dataclass(frozen=True)
class _Piece:
    """A kept head or tail of a merge input, with the original units it stands for."""

    text: str
    units: int
    source: int  # 0: ``a``, 1: ``b``


def _cut(piece: _Piece, budget: int, unit: str, *, back: bool, limit: int) -> tuple[int, int]:
    """The longest prefix (``back``: suffix) of ``piece`` within ``budget`` original units.

    Returns (characters kept, at most ``limit``; an upper bound of their
    units). Chars and bytes without U+FFFD are exact. A bytes piece knows its
    total units (``units``) but not each U+FFFD's width (1 to 3 original
    bytes): the bound follows from the total and from what the rest of the
    piece holds, and is exact when the part keeps none or all of the
    piece's U+FFFD, or when all of them have one width.
    """
    if budget <= 0 or limit <= 0:
        return 0, 0
    text = piece.text
    if unit == "chars":
        kept = min(budget, limit, len(text))
        return kept, kept
    count = text.count(REPLACEMENT)
    other = _units(text, unit) - 3 * count  # bytes of the characters that are not U+FFFD
    other_part = count_part = 0
    best = (0, 0)
    for kept, char in enumerate(reversed(text) if back else text, 1):
        if kept > limit:
            break
        if char == REPLACEMENT:
            count_part += 1
        else:
            other_part += _units(char, unit)
        rest_other, rest_count = other - other_part, count - count_part
        high = min(other_part + 3 * count_part, piece.units - rest_other - rest_count)
        if high > budget:
            break
        best = (kept, high)
    return best


def _min_replaced(piece: _Piece) -> int:
    """Replacements ``piece`` holds at least: a genuine U+FFFD stands for 3 bytes, others 1-3."""
    count = piece.text.count(REPLACEMENT)
    if not count:
        return 0
    other = _units(piece.text, "bytes") - 3 * count
    genuine_at_most = max(0, (piece.units - other - count) // 2)
    return max(0, count - genuine_at_most)


def merge(a: Text, b: Text, head: int, tail: int) -> Text:
    """``a`` then ``b`` as one ``Text`` of at most ``head`` + ``tail`` kept units.

    For unread progress deltas: ``length`` is ``a.length + b.length``, and
    what either delta had already omitted stays omitted. Units are original
    units, as each input records them (``Text.units()``): a byte decoded to
    U+FFFD counts as the bytes it replaced. Whole parts are kept with their
    exact units. Cutting inside a bytes part with U+FFFD of unknown widths
    (see ``_cut``), the kept part counts its upper bound: the limits hold in
    original units, ``length == head_units + tail_units + omitted`` and
    ``omitted`` is then a lower bound. A bytes ``Text`` counts in
    ``replaced`` only replacements it still holds (a lower bound when that
    depends on which U+FFFD were genuine); a chars ``Text`` sums the windows.
    """
    if a.unit != b.unit:
        raise ValueError(f"cannot merge {a.unit} with {b.unit}")
    if head < 0 or tail < 0:
        raise ValueError("head and tail must be non-negative")
    unit = a.unit
    sources = (a, b)
    pieces: list[_Piece | int] = []
    for source, text in enumerate(sources):
        head_units, tail_units = text.units()
        pieces.append(_Piece(text.head, head_units, source))
        if not text.complete:
            pieces.append(max(1, text.omitted))
        pieces.append(_Piece(text.tail, tail_units, source))
    pieces = [piece for piece in pieces if not isinstance(piece, _Piece) or piece.text != ""]
    length = a.length + b.length
    known = sum(piece.units for piece in pieces if isinstance(piece, _Piece))
    gaps = any(isinstance(piece, int) for piece in pieces)
    full = (
        b.full if a.full is not None and b.full is not None and a.full.path == b.full.path else None
    )

    if not gaps and known <= head + tail:
        joined = "".join(piece.text for piece in pieces if isinstance(piece, _Piece))
        if unit == "chars":
            return Text(joined, "", length, unit, True, None, full, a.replaced + b.replaced)
        replaced = min(a.replaced + b.replaced, joined.count(REPLACEMENT))
        return Text(joined, "", length, unit, True, None, full, replaced, 0, known, 0)

    kept: list[tuple[_Piece, str]] = []  # what is kept of each piece
    head_parts: list[str] = []
    head_kept = 0
    room = head
    index = 0
    cut_at, cut_chars = -1, 0  # the piece the head cut inside, and its characters kept
    while index < len(pieces) and room > 0:
        piece = pieces[index]
        if not isinstance(piece, _Piece):
            break
        if piece.units <= room:
            head_parts.append(piece.text)
            kept.append((piece, piece.text))
            room -= piece.units
            head_kept += piece.units
            index += 1
            continue
        cut_at = index
        cut_chars, used = _cut(piece, room, unit, back=False, limit=len(piece.text))
        head_parts.append(piece.text[:cut_chars])
        kept.append((piece, piece.text[:cut_chars]))
        head_kept += used
        break
    tail_parts: list[str] = []
    tail_kept = 0
    room = tail
    last = len(pieces) - 1
    while last >= index and room > 0:
        piece = pieces[last]
        if not isinstance(piece, _Piece):
            break
        if last != cut_at and piece.units <= room:
            tail_parts.insert(0, piece.text)
            kept.append((piece, piece.text))
            room -= piece.units
            tail_kept += piece.units
            last -= 1
            continue
        limit = len(piece.text) - (cut_chars if last == cut_at else 0)
        chars, used = _cut(piece, room, unit, back=True, limit=limit)
        part = piece.text[len(piece.text) - chars :]
        tail_parts.insert(0, part)
        kept.append((piece, part))
        tail_kept += used
        break
    new_head, new_tail = "".join(head_parts), "".join(tail_parts)
    omitted = length - head_kept - tail_kept
    reason = a.reason or b.reason or "max_length"
    if unit == "chars":
        return Text(
            new_head, new_tail, length, unit, False, reason, full, a.replaced + b.replaced, omitted
        )
    return Text(
        new_head,
        new_tail,
        length,
        unit,
        False,
        reason,
        full,
        _kept_replaced(sources, pieces, kept),
        omitted,
        head_kept,
        tail_kept,
    )


def _kept_replaced(
    sources: tuple[Text, Text], pieces: list[_Piece | int], kept: list[tuple[_Piece, str]]
) -> int:
    """The replacements a cut bytes merge still holds (at least; never one it dropped)."""
    total = 0
    for source, text in enumerate(sources):
        own = [piece for piece in pieces if isinstance(piece, _Piece) and piece.source == source]
        held = sum(piece.text.count(REPLACEMENT) for piece in own)
        kept_parts = [(piece, part) for piece, part in kept if piece.source == source]
        kept_count = sum(part.count(REPLACEMENT) for _piece, part in kept_parts)
        # A replacement of this input not kept is one of the U+FFFD dropped.
        by_total = text.replaced - (held - kept_count)
        # Per piece: the replacements it must hold, minus the U+FFFD dropped from it.
        by_piece = 0
        for piece in own:
            kept_here = sum(part.count(REPLACEMENT) for other, part in kept_parts if other is piece)
            dropped = piece.text.count(REPLACEMENT) - kept_here
            by_piece += max(0, _min_replaced(piece) - dropped)
        total += min(kept_count, max(0, by_total, by_piece))
    return total
