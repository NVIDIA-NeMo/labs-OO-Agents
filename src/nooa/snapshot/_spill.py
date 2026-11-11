# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spill files: ``SpillTarget`` (one stream's file) and ``SpillDir`` (a bounded folder of them).

The only module that touches the filesystem. It is built on the standard
library: ``tempfile.mkdtemp`` for a private folder (0700),
``tempfile.mkstemp`` for each file (exclusive create, 0600) and
``shutil.rmtree`` / ``os.unlink`` to remove them. What it adds is the
eviction bookkeeping: a folder keeps at most ``max_files`` files of at most
``file_bytes`` each, and deletes the oldest closed ones to make room.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import tempfile
import threading
import weakref
from dataclasses import dataclass, field
from typing import IO, Literal

__all__ = ["SpillDir", "SpillTarget"]

_log = logging.getLogger(__name__)

DEFAULT_SPILL_BYTES = 64 * 1024 * 1024  # SpillTarget.max_bytes and SpillDir.file_bytes default
DEFAULT_SPILL_FILES = 20  # SpillDir.max_files default
_PREFIX, _SUFFIX = "nooa-output-", ".txt"  # spill file names: nooa-output-*.txt
# A SpillDir file's state: "new" (handed out by target()), "open" (a stream
# writes it) or "closed" (its stream closed; it may be evicted).
_State = Literal["new", "open", "closed"]


@dataclass(frozen=True)
class SpillTarget:
    """Where and how much a ``TextStream`` spills: the file that keeps its whole output.

    Fields:

    - ``path`` (default ``None``): a new file to write, created exclusively:
      an existing path (a file, a symlink) is refused, ``TextStream``
      raises ``FileExistsError`` and never truncates it. ``None``: a new
      temporary file (``nooa-output-*.txt`` in the system temporary
      directory, mode 0600).
    - ``max_bytes`` (default 64 MiB): only the first ``max_bytes`` bytes are
      written; ``Spill.cut`` then says the file is cut. Non-negative. A
      chars stream cuts its file on a code point boundary, so the file is
      always valid UTF-8 and may hold up to 3 bytes less; ``Spill.kept``
      counts the bytes actually written. A bytes stream's file holds the
      raw bytes as written.

    ``close()`` deletes the file when the final ``Text`` is complete (the
    snapshot then holds everything). Apart from ``FileExistsError`` for an
    existing ``path``, a stream never raises for a file error: when the
    file cannot be created, written or flushed, it logs a warning, removes
    the partial file and keeps memory only (``Text.full`` is ``None``).

    ``SpillDir.target()`` hands out targets inside its folder. Such a target
    opens one stream (a second one keeps memory only, logged), never more
    than the folder's ``file_bytes`` (also after ``dataclasses.replace``),
    and tells its folder when the stream closes. A target built here
    belongs to no folder.
    """

    path: str | None = None
    max_bytes: int = DEFAULT_SPILL_BYTES
    # The folder that handed this target out (``SpillDir.target``), or ``None``.
    # Private: only ``SpillDir.target`` passes it. A real (keyword-only) field,
    # so ``dataclasses.replace`` keeps it; left out of repr and comparison.
    _dir: SpillDir | None = field(default=None, repr=False, compare=False, kw_only=True)


class _SpillFile:
    """One stream's open spill file: its path, its byte cap, and who removes it.

    Args:
        file: the open binary file (``None`` once finished).
        path: its path.
        max_bytes: the most bytes the stream writes to it.
        folder: the ``SpillDir`` that owns the file, or ``None`` for a
            standalone target (the stream then removes the file itself).
    """

    def __init__(self, file: IO[bytes], path: str, max_bytes: int, folder: SpillDir | None) -> None:
        self.file: IO[bytes] | None = file
        self.path = path
        self.max_bytes = max_bytes
        self._folder = folder

    @classmethod
    def open(cls, target: SpillTarget) -> _SpillFile | None:
        """Open the file of ``target`` for one stream.

        A standalone target's file is created: exclusively at ``target.path``
        (mode 0600), or as a new temporary file. A ``SpillDir`` target's file
        already exists; it is claimed from its folder (once only) and opened
        without following a symlink.

        Returns:
            The open ``_SpillFile``, or ``None`` (logged; the stream keeps
            memory only) when the file cannot be created or opened, or a
            folder's target was already used.

        Raises:
            FileExistsError: a standalone ``target.path`` already exists.
        """
        folder = target._dir
        try:
            if folder is not None and target.path is not None:
                if not folder._claim(target.path):
                    _log.warning(
                        "spill target %r is not open to a new stream; keeping memory only",
                        target.path,
                    )
                    return None
                path = target.path
                try:  # our new file in the folder; never through a symlink put in its place
                    fd = os.open(
                        path,
                        os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                    )
                except OSError:
                    folder._finished(path, delete=True)
                    raise
                return cls(
                    os.fdopen(fd, "wb"), path, min(target.max_bytes, folder.file_bytes), folder
                )
            if target.path is not None:
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
                fd, path = os.open(target.path, flags, 0o600), target.path
            else:
                fd, path = tempfile.mkstemp(prefix=_PREFIX, suffix=_SUFFIX)
            return cls(os.fdopen(fd, "wb"), path, target.max_bytes, None)
        except FileExistsError:
            raise  # a usage error, not a file error: the caller named an existing file
        except OSError:
            _log.warning("could not create the spill file; keeping memory only", exc_info=True)
            return None

    def finish(self, *, delete: bool) -> None:
        """Close the file and, with ``delete``, remove it; a folder's file is handed back to its folder.

        Idempotent; never raises (file errors are logged or ignored).
        """
        file, self.file = self.file, None
        if file is not None:
            with contextlib.suppress(OSError):
                file.close()
        folder, self._folder = self._folder, None
        if folder is not None:
            folder._finished(self.path, delete=delete)
        elif delete and file is not None:
            _unlink(self.path)


def _unlink(path: str) -> None:
    """Remove ``path`` (a symlink itself, never what it points to); log, never raise."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError:
        _log.warning("could not delete spill file %r", path, exc_info=True)


def _remove(path: str, owns_dir: bool, names: dict[str, _State]) -> None:
    """Clean up a ``SpillDir``: remove the whole private directory, or only the names it created.

    Args:
        path: the folder's directory.
        owns_dir: whether the folder made the directory (``tempfile.mkdtemp``).
        names: the files it created; emptied.

    Never raises.
    """
    if owns_dir:
        shutil.rmtree(path, ignore_errors=True)
    else:
        for name in names:
            _unlink(name)
    names.clear()


class SpillDir:
    """A folder of spill files shared by many ``TextStream``s: at most ``max_files`` of ``file_bytes`` each.

    ``SpillDir(path=None, *, max_files=20, file_bytes=64 * 2**20)``:

    - ``path`` (default ``None``): a new private temporary directory
      (``tempfile.mkdtemp``, mode 0700) that the ``SpillDir`` removes whole
      on ``close()``. Else an existing directory; the ``SpillDir`` then
      deletes only the names it created there and leaves everything else
      alone.
    - ``max_files``: at most this many of its files exist at once.
    - ``file_bytes``: each file holds at most this many bytes, so the
      folder never holds more than ``max_files * file_bytes``.

    ``target()`` hands out a ``SpillTarget`` for one ``TextStream``, with a
    new empty file in the folder (``tempfile.mkstemp``: exclusive, 0600).
    To make room it first deletes closed files, oldest first; it never
    deletes the file of a stream still open (or of a target not yet
    closed). When every file is still open, or the file cannot be created
    (logged), it returns ``None``: pass it on as ``TextStream(spill=None)``,
    which keeps memory only. A stream that deletes its complete output
    frees its place at once.

    A full copy in a ``SpillDir`` is not permanent: ``Text.full`` (a
    ``Spill``) of an earlier stream can name a file deleted since, because
    the folder needed room or closed ("keep the newest"). Opening that path
    then raises ``FileNotFoundError``: that is how a reader tells the full
    output is gone.

    ``close()`` (also on leaving a ``with`` block, and at interpreter exit
    through ``weakref.finalize``) removes the private directory with
    ``shutil.rmtree``, or unlinks each name it created in a given one
    (never following a symlink); later calls do nothing, and ``target()``
    then raises ``ValueError``. A file error never raises from ``close()``
    or an eviction: it is logged. A stream still open when the folder
    closes keeps running; its file is gone.

    Threads: ``target()``, ``close()`` and the streams' closes may run in
    any threads; the bookkeeping is locked.

    Raises ``ValueError`` for a ``path`` that is not an existing directory,
    or a negative ``max_files`` or ``file_bytes``.

    >>> from nooa.snapshot import SpillDir, TextStream
    >>> with SpillDir(max_files=2, file_bytes=4) as spills:
    ...     stream = TextStream(unit="bytes", head=2, tail=2, spill=spills.target())
    ...     stream.write(b"0123456789")
    ...     stream.close().full.kept
    4
    """

    path: str
    """The folder's directory."""
    max_files: int
    """At most this many files at once, as given."""
    file_bytes: int
    """At most this many bytes per file, as given."""

    def __init__(
        self,
        path: str | None = None,
        *,
        max_files: int = DEFAULT_SPILL_FILES,
        file_bytes: int = DEFAULT_SPILL_BYTES,
    ) -> None:
        if max_files < 0 or file_bytes < 0:
            raise ValueError("max_files and file_bytes must be non-negative")
        if path is not None and not os.path.isdir(path):
            raise ValueError(f"spill folder {path!r} is not an existing directory")
        self.max_files = max_files
        self.file_bytes = file_bytes
        self.path = tempfile.mkdtemp(prefix="nooa-spill-") if path is None else path
        self._lock = threading.Lock()
        # The names it created, oldest first, and their state: "new" (handed
        # out), "open" (a stream writes it), "closed" (it may be evicted now).
        self._files: dict[str, _State] = {}
        self._cleanup = weakref.finalize(self, _remove, self.path, path is None, self._files)

    def target(self) -> SpillTarget | None:
        """A ``SpillTarget`` for one new stream, within the budget; ``None``: keep memory only.

        Deletes the oldest closed files to make room (see ``SpillDir``),
        never an open stream's file. Returns ``None`` when ``max_files``
        files are still in use, ``file_bytes`` is 0, or the file cannot be
        created (logged). Raises ``ValueError`` after ``close()``.
        """
        with self._lock:
            if not self._cleanup.alive:
                raise ValueError("target() on a closed SpillDir")
            for path in [path for path, state in self._files.items() if state == "closed"]:
                if len(self._files) < self.max_files:
                    break
                del self._files[path]
                _unlink(path)
            if len(self._files) >= self.max_files or self.file_bytes == 0:
                return None
            try:
                fd, path = tempfile.mkstemp(prefix=_PREFIX, suffix=_SUFFIX, dir=self.path)
                os.close(fd)
            except OSError:
                _log.warning("could not create a spill file; keeping memory only", exc_info=True)
                return None
            self._files[path] = "new"
        return SpillTarget(path=path, max_bytes=self.file_bytes, _dir=self)

    def _claim(self, path: str) -> bool:
        """Mark ``path`` open for one stream.

        Returns ``False`` when ``path`` is not a target of this folder that no
        stream has opened yet (already used, evicted, or the folder closed).
        """
        with self._lock:
            if self._files.get(path) != "new":
                return False
            self._files[path] = "open"
            return True

    def _finished(self, path: str, *, delete: bool) -> None:
        """Record that the stream writing ``path`` closed: delete the file, or mark it evictable.

        A no-op when ``path`` is not open in this folder (the folder closed).
        """
        with self._lock:
            if self._files.get(path) != "open":
                return  # the folder closed
            if delete:
                del self._files[path]
                _unlink(path)
            else:
                self._files[path] = "closed"

    def close(self) -> None:
        """Remove every file this folder created (the whole private directory).

        Idempotent; never raises (file errors are logged or ignored).
        """
        with self._lock:
            self._cleanup()

    def __enter__(self) -> SpillDir:
        """Return the folder itself, for ``with SpillDir() as spills:``."""
        return self

    def __exit__(self, *exc: object) -> None:
        """``close()`` the folder on leaving the ``with`` block."""
        self.close()
