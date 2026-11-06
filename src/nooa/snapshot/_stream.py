# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Output that arrives over time: ``TextStream``, ``SpillTarget``, ``SpillDir`` and ``merge``.

The second producer of snapshots, next to ``capture``: it builds a ``Text``
incrementally, so the whole output is never held in memory. It reads no
live objects and does not import the capture layers; it shares only the
nodes (``_model``) and the UTF-8 cut points (``_codec``).

Each ``TextStream`` bounds its own spill file. ``SpillDir`` bounds the
files of many streams together (count and total bytes), evicts the oldest
closed ones when it needs room and deletes what it created on ``close()``;
its streams tell it when they close through the ``SpillTarget`` it handed
out.
"""

from __future__ import annotations

import collections
import contextlib
import logging
import os
import stat
import tempfile
import threading
from dataclasses import dataclass, field
from typing import IO, Literal

from nooa.snapshot._codec import complete_prefix, safe_end, safe_start
from nooa.snapshot._model import Spill, Text

__all__ = ["SpillDir", "SpillTarget", "TextStream", "merge"]

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

    ``SpillDir.target()`` hands out targets inside a shared folder; such a
    target also tells its folder when its stream closes. A target built
    here belongs to no folder.
    """

    path: str | None = None
    max_bytes: int = DEFAULT_SPILL_BYTES
    always: bool = False
    # The folder that handed this target out (``SpillDir.target``), or ``None``.
    _dir: SpillDir | None = field(default=None, init=False, repr=False, compare=False)


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
        # The ``SpillDir`` that handed out ``spill``, told once when the file is done.
        self._owner = spill._dir if spill is not None else None
        self._owned_path = spill.path if spill is not None else None
        if spill is not None:
            self._open(spill)

    # --- spilling ---------------------------------------------------------

    def _open(self, spill: SpillTarget) -> None:
        try:
            if self._owner is not None and spill.path is not None:
                self._file = self._owner._open(spill.path)
                self._path = spill.path
            elif spill.path is not None:
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
        if self._owner is not None:
            self._release(delete=True)
        elif self._path is not None:
            with contextlib.suppress(OSError):
                os.unlink(self._path)

    def _release(self, *, delete: bool) -> None:
        """Tell the ``SpillDir`` the file is done (``delete``: remove it), once."""
        owner, self._owner = self._owner, None
        if owner is not None and self._owned_path is not None:
            owner._finished(self._owned_path, delete=delete)

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
        delete = (
            text.complete
            and not self._keep_file
            and self._path is not None
            and not self._spill_failed
        )
        if self._owner is not None:
            self._release(delete=delete)
        elif delete and self._path is not None:
            with contextlib.suppress(OSError):
                os.unlink(self._path)
        self._closed = text
        return text

    def __enter__(self) -> TextStream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


DEFAULT_SPILL_FILES = 20


@dataclass
class _Entry:
    """One file a ``SpillDir`` created: its identity, and its budget while open or once closed."""

    ident: tuple[int, int]  # (st_dev, st_ino) when created
    reserved: int  # bytes: the target's ``max_bytes`` while open, the size on disk once closed
    open: bool = True


class SpillDir:
    """A folder of spill files shared by many ``TextStream``s, within a file count and a byte total.

    ``SpillDir(path=None, *, max_files=20, max_bytes=64 * 2**20)``:

    - ``path`` (default ``None``): a new private temporary directory that
      the ``SpillDir`` owns and removes on ``close()``. Else an existing
      directory; the ``SpillDir`` then owns only the files it creates there
      and leaves everything else alone.
    - ``max_files``: at most this many of its files exist at once.
    - ``max_bytes``: its files hold at most this many bytes together.

    ``target(*, always=False, max_bytes=None)`` hands out a ``SpillTarget``
    for one ``TextStream``, with a new empty file in the folder. The budget
    holds without watching writes: an open stream's file counts its
    target's ``max_bytes`` (a stream never writes more), a closed one its
    size on disk. To make room for a new target, ``target()`` deletes
    closed files, oldest first, until the new one fits both limits; it
    never deletes the file of a stream still open. The new target's
    ``max_bytes`` is the one asked for (default: an equal share,
    ``max_bytes // max_files``, so ``max_files`` default-size streams fit
    at once), cut to what is left. With no file or byte left,
    ``target()`` returns ``None``: pass it on as ``TextStream(spill=None)``,
    which keeps memory only. A stream that deletes its complete output
    frees its share at once.

    A full copy in a ``SpillDir`` is not permanent: ``Text.full`` (a
    ``Spill``) of an earlier snapshot can name a file deleted since,
    because the folder needed room or closed. Readers must expect the file
    to be gone.

    ``close()`` (also on leaving a ``with`` block) deletes every file the
    ``SpillDir`` created, and the directory if it created it; later calls do
    nothing, and ``target()`` then raises ``ValueError``. It deletes only
    regular files it created, matched by path and ``(st_dev, st_ino)``,
    and never follows a symlink; a file that was replaced is left alone
    with a logged warning. A file error never raises from ``close()`` or an
    eviction: it is logged. A stream still open when the folder closes
    keeps running; its file is gone.

    Threads: ``target()``, ``close()`` and the streams' closes may run in
    any threads; the bookkeeping is locked.

    Raises ``ValueError`` for a ``path`` that is not an existing directory,
    or a negative ``max_files`` or ``max_bytes``.

    >>> from nooa.snapshot import SpillDir, TextStream
    >>> with SpillDir(max_files=2, max_bytes=8) as spills:
    ...     stream = TextStream(unit="bytes", head=2, tail=2, spill=spills.target())
    ...     stream.write(b"0123456789")
    ...     stream.close().full.kept
    4
    """

    path: str
    """The folder's directory."""
    max_files: int
    """At most this many files at once, as given."""
    max_bytes: int
    """At most this many bytes in all files, as given."""

    def __init__(
        self,
        path: str | None = None,
        *,
        max_files: int = DEFAULT_SPILL_FILES,
        max_bytes: int = DEFAULT_SPILL_BYTES,
    ) -> None:
        if max_files < 0 or max_bytes < 0:
            raise ValueError("max_files and max_bytes must be non-negative")
        if path is not None and not os.path.isdir(path):
            raise ValueError(f"spill folder {path!r} is not an existing directory")
        self.max_files = max_files
        self.max_bytes = max_bytes
        self._owns_dir = path is None
        self.path = tempfile.mkdtemp(prefix="nooa-spill-") if path is None else path
        self._lock = threading.Lock()
        self._files: dict[str, _Entry] = {}  # in creation order: the oldest first
        self._closed = False

    def target(self, *, always: bool = False, max_bytes: int | None = None) -> SpillTarget | None:
        """A ``SpillTarget`` for one new stream, within the budget; ``None``: keep memory only.

        - ``always``: as ``SpillTarget.always``.
        - ``max_bytes`` (default: an equal share, the folder's
          ``max_bytes // max_files``): the most the stream may write; cut to
          what the folder has left.

        Deletes the oldest closed files to make room (see ``SpillDir``).
        Returns ``None`` when no file or byte is left, or when the file
        cannot be created (logged). Raises ``ValueError`` after ``close()``
        or for a negative ``max_bytes``.
        """
        want = (
            (self.max_bytes // self.max_files if self.max_files else 0)
            if max_bytes is None
            else max_bytes
        )
        if want < 0:
            raise ValueError("max_bytes must be non-negative")
        with self._lock:
            if self._closed:
                raise ValueError("target() on a closed SpillDir")
            for path in [path for path, entry in self._files.items() if not entry.open]:
                if self._fits(want):
                    break
                self._delete(path, self._files.pop(path))
            room = min(want, self.max_bytes - self._used())
            if len(self._files) >= self.max_files or room <= 0:
                return None
            try:
                fd, path = tempfile.mkstemp(prefix="nooa-output-", suffix=".txt", dir=self.path)
                try:
                    info = os.fstat(fd)
                finally:
                    os.close(fd)
            except OSError:
                _log.warning("could not create a spill file; keeping memory only", exc_info=True)
                return None
            self._files[path] = _Entry((info.st_dev, info.st_ino), room)
        target = SpillTarget(path=path, max_bytes=room, always=always)
        object.__setattr__(target, "_dir", self)  # frozen: set once, before anyone sees it
        return target

    def _used(self) -> int:
        return sum(entry.reserved for entry in self._files.values())

    def _fits(self, want: int) -> bool:
        return len(self._files) < self.max_files and self._used() + want <= self.max_bytes

    def _open(self, path: str) -> IO[bytes]:
        """Open a file this folder created for writing, without following a symlink."""
        with self._lock:
            entry = self._files.get(path)
        if entry is None:
            raise OSError(f"spill file {path!r} is no longer in its folder")
        flags = os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        fd = os.open(path, flags)
        try:
            info = os.fstat(fd)
            if (info.st_dev, info.st_ino) != entry.ident or not stat.S_ISREG(info.st_mode):
                raise OSError(f"spill file {path!r} was replaced")
            os.ftruncate(fd, 0)
            return os.fdopen(fd, "wb")
        except BaseException:
            os.close(fd)
            raise

    def _finished(self, path: str, *, delete: bool) -> None:
        """A stream closed its file: delete it, or count its size on disk from now on."""
        with self._lock:
            entry = self._files.get(path)
            if entry is None or not entry.open:
                return  # evicted or closed already
            if delete:
                self._delete(path, self._files.pop(path))
                return
            try:
                info = os.lstat(path)
            except OSError:
                del self._files[path]  # gone: nothing left to count
                return
            if (info.st_dev, info.st_ino) != entry.ident or not stat.S_ISREG(info.st_mode):
                _log.warning("spill file %r was replaced; leaving it alone", path)
                del self._files[path]
                return
            entry.open = False
            entry.reserved = info.st_size

    @staticmethod
    def _delete(path: str, entry: _Entry) -> None:
        """Delete ``path`` if it is still the regular file created; log, never raise."""
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            return
        except OSError:
            _log.warning("could not check spill file %r; leaving it", path, exc_info=True)
            return
        if (info.st_dev, info.st_ino) != entry.ident or not stat.S_ISREG(info.st_mode):
            _log.warning("spill file %r was replaced; leaving it alone", path)
            return
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        except OSError:
            _log.warning("could not delete spill file %r", path, exc_info=True)

    def close(self) -> None:
        """Delete every file this folder created, and the directory if it created it; idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            files, self._files = self._files, {}
            for path, entry in files.items():
                self._delete(path, entry)
            if self._owns_dir:
                try:
                    os.rmdir(self.path)
                except OSError:
                    _log.warning("could not remove spill folder %r", self.path, exc_info=True)

    def __enter__(self) -> SpillDir:
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
