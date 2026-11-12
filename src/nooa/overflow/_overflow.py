# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``OverflowFile``, ``OverflowDir`` and the writer of one file; see ``nooa.overflow`` for the contract.

Built on the standard library: ``tempfile.mkdtemp`` for a private folder
(0700), ``tempfile.mkstemp`` for each file (exclusive create, 0600),
``shutil.rmtree`` / ``os.unlink`` to remove them and ``atexit`` for the
exit cleanup. What it adds is the eviction bookkeeping and the explicit
lifetime (no ``weakref.finalize``: nothing is deleted by garbage
collection).
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass, field
from typing import IO, Literal

__all__ = [
    "DEFAULT_FILE_BYTES",
    "DEFAULT_MAX_FILES",
    "OverflowDir",
    "OverflowFile",
    "OverflowWriter",
]

_log = logging.getLogger(__name__)

DEFAULT_FILE_BYTES = 64 * 1024 * 1024  # OverflowFile.max_bytes and OverflowDir.file_bytes default
DEFAULT_MAX_FILES = 20  # OverflowDir.max_files default
_PREFIX, _SUFFIX = "nooa-output-", ".txt"  # file names: nooa-output-*.txt
_DIR_PREFIX = "nooa-overflow-"  # private folders: nooa-overflow-*
# An OverflowDir file's state: "new" (handed out by file()), "open" (a
# writer writes it) or "closed" (its writer finished; it may be evicted).
_State = Literal["new", "open", "closed"]


@dataclass(frozen=True)
class OverflowFile:
    """One overflow file: where a writer puts the whole output, and how much of it.

    Fields:

    - ``path`` (default ``None``): a new file to write, created exclusively:
      an existing path (a file, a symlink) is refused, ``open()`` raises
      ``FileExistsError`` and never truncates it. ``None``: a new temporary
      file (``nooa-output-*.txt`` in the system temporary directory, mode
      0600).
    - ``max_bytes`` (default 64 MiB): the writer keeps only the first
      ``max_bytes`` bytes. Non-negative.

    ``open()`` returns the ``OverflowWriter`` of the file. Apart from
    ``FileExistsError`` for an existing ``path``, it never raises for a file
    error: when the file cannot be created it logs a warning and returns
    ``None`` (keep memory only).

    ``OverflowDir.file()`` hands out files inside its folder. Such a file
    opens one writer (a second ``open()`` returns ``None``, logged), never
    more than the folder's ``file_bytes`` (also after
    ``dataclasses.replace``), and tells its folder when the writer
    finishes. A file built here belongs to no folder: nothing deletes it
    but its writer (``finish(delete=True)``).
    """

    path: str | None = None
    max_bytes: int = DEFAULT_FILE_BYTES
    # The folder that handed this file out (``OverflowDir.file``), or ``None``.
    # Private: only ``OverflowDir.file`` passes it. A real (keyword-only) field,
    # so ``dataclasses.replace`` keeps it; left out of repr and comparison.
    _owner: OverflowDir | None = field(default=None, repr=False, compare=False, kw_only=True)

    def open(self) -> OverflowWriter | None:
        """Open the file for one writer.

        A standalone file is created: exclusively at ``path`` (mode 0600), or
        as a new temporary file. An ``OverflowDir`` file already exists; it is
        claimed from its folder (once only) and opened without following a
        symlink.

        Returns:
            The ``OverflowWriter``, or ``None`` (logged; keep memory only)
            when the file cannot be created or opened, or a folder's file was
            already opened, evicted or its folder closed.

        Raises:
            FileExistsError: a standalone ``path`` already exists.
            ValueError: a negative ``max_bytes``.
        """
        if self.max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        owner = self._owner
        try:
            if owner is not None and self.path is not None:
                if not owner._claim(self.path):
                    _log.warning(
                        "overflow file %r is not open to a new writer; keeping memory only",
                        self.path,
                    )
                    return None
                path = self.path
                try:  # our new file in the folder; never through a symlink put in its place
                    fd = os.open(
                        path,
                        os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                    )
                except OSError:
                    owner._finished(path, delete=True)
                    raise
                return OverflowWriter(
                    os.fdopen(fd, "wb"), path, min(self.max_bytes, owner.file_bytes), owner
                )
            if self.path is not None:
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
                fd, path = os.open(self.path, flags, 0o600), self.path
            else:
                fd, path = tempfile.mkstemp(prefix=_PREFIX, suffix=_SUFFIX)
            return OverflowWriter(os.fdopen(fd, "wb"), path, self.max_bytes, None)
        except FileExistsError:
            raise  # a usage error, not a file error: the caller named an existing file
        except OSError:
            _log.warning("could not create the overflow file; keeping memory only", exc_info=True)
            return None


class OverflowWriter:
    """The open file of one ``OverflowFile``: what ``OverflowFile.open()`` returns.

    - ``path``: the file.
    - ``max_bytes``: the most bytes ``write`` keeps.
    - ``written``: the bytes written so far.
    - ``file``: the open binary file, ``None`` once finished. A caller may
      ``flush()`` or ``close()`` it directly to see its ``OSError``.

    ``write(data)`` writes what fits under ``max_bytes`` and returns how
    much; it raises ``OSError`` on a file error (the caller decides to give
    up: ``finish(delete=True)``). ``finish(delete=...)`` closes the file;
    with ``delete`` it removes it, else a folder's file becomes evictable.
    """

    def __init__(
        self, file: IO[bytes], path: str, max_bytes: int, owner: OverflowDir | None
    ) -> None:
        self.file: IO[bytes] | None = file
        self.path = path
        self.max_bytes = max_bytes
        self.written = 0
        self._owner = owner

    def write(self, data: bytes) -> int:
        """Write the part of ``data`` that fits under ``max_bytes``; return its length.

        Returns 0 once finished or full. Raises ``OSError`` on a file error.
        """
        if self.file is None:
            return 0
        part = data[: max(0, self.max_bytes - self.written)]
        if part:
            self.file.write(part)
            self.written += len(part)
        return len(part)

    def finish(self, *, delete: bool) -> None:
        """Close the file and, with ``delete``, remove it; a folder's file is handed back to its folder.

        Idempotent; never raises (file errors are logged or ignored).
        """
        file, self.file = self.file, None
        if file is not None:
            with contextlib.suppress(OSError):
                file.close()
        owner, self._owner = self._owner, None
        if owner is not None:
            owner._finished(self.path, delete=delete)
        elif delete and file is not None:
            _unlink(self.path)


def _unlink(path: str) -> None:
    """Remove ``path`` (a symlink itself, never what it points to); log, never raise."""
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError:
        _log.warning("could not delete overflow file %r", path, exc_info=True)


class _Cleanup:
    """What an ``OverflowDir`` must remove, held apart from it so that exit can run it.

    It holds no reference to the ``OverflowDir``: dropping the folder object
    deletes nothing; its cleanup stays registered until ``close()`` or exit.
    """

    def __init__(self, path: str, owns_dir: bool, lock: threading.Lock) -> None:
        self.path = path
        self.owns_dir = owns_dir
        self.lock = lock
        # The names it created, oldest first, and their state.
        self.files: dict[str, _State] = {}
        self.done = False

    def run(self) -> None:
        """Remove the whole private directory, or only the names it created; once. Call under ``lock``."""
        if self.done:
            return
        self.done = True
        with _live_lock:
            _live.discard(self)
        if self.owns_dir:
            shutil.rmtree(self.path, ignore_errors=True)
        else:
            for name in self.files:
                _unlink(name)
        self.files.clear()


# Every OverflowDir not yet closed, for the exit cleanup. Strong references to
# the cleanups only, never to the folders.
_live: set[_Cleanup] = set()
_live_lock = threading.Lock()


@atexit.register
def _close_all_at_exit() -> None:
    """At interpreter exit, remove the files of every ``OverflowDir`` never closed."""
    with _live_lock:
        pending = list(_live)
    for cleanup in pending:
        # A thread still inside file() may hold the lock; do not hang the exit on it.
        locked = cleanup.lock.acquire(timeout=1.0)
        try:
            cleanup.run()
        except Exception:  # noqa: BLE001 - never fail the exit
            _log.warning("could not clean up overflow folder %r", cleanup.path, exc_info=True)
        finally:
            if locked:
                cleanup.lock.release()


class OverflowDir:
    """A folder of overflow files shared by many writers: at most ``max_files`` of ``file_bytes`` each.

    ``OverflowDir(path=None, *, max_files=20, file_bytes=64 * 2**20)``:

    - ``path`` (default ``None``): a new private temporary directory
      (``tempfile.mkdtemp``, ``nooa-overflow-*``, mode 0700) that the
      ``OverflowDir`` removes whole when closed. Else an existing directory;
      the ``OverflowDir`` then deletes only the names it created there and
      leaves everything else alone.
    - ``max_files``: at most this many of its files exist at once.
    - ``file_bytes``: each file holds at most this many bytes, so the
      folder never holds more than ``max_files * file_bytes``.

    ``file()`` hands out an ``OverflowFile`` for one writer, with a new
    empty file in the folder (``tempfile.mkstemp``: exclusive, 0600). To
    make room it first deletes closed files, oldest first; it never deletes
    the file of a writer still open (or of a file handed out and not yet
    finished). When every file is still in use, or the file cannot be
    created (logged), it returns ``None``: keep memory only. A writer that
    finishes with ``delete=True`` frees its place at once.

    A file in an ``OverflowDir`` is not permanent: a path handed out earlier
    can name a file deleted since, because the folder needed room or
    closed ("keep the newest"). Opening that path then raises
    ``FileNotFoundError``.

    Lifetime: ``close()`` (also on leaving a ``with`` block) removes the
    private directory with ``shutil.rmtree``, or unlinks each name it
    created in a given one (never following a symlink); later calls do
    nothing, and ``file()`` then raises ``ValueError``. A folder never
    closed is cleaned up at interpreter exit (``atexit``), **not** when it
    is garbage collected: its files outlive the object. A file error never
    raises from ``close()`` or an eviction: it is logged. A writer still
    open when the folder closes keeps running; its file is gone.

    Threads: ``file()``, ``close()`` and the writers' ``finish`` may run in
    any threads; the bookkeeping is locked.

    Raises ``ValueError`` for a ``path`` that is not an existing directory,
    or a negative ``max_files`` or ``file_bytes``.
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
        max_files: int = DEFAULT_MAX_FILES,
        file_bytes: int = DEFAULT_FILE_BYTES,
    ) -> None:
        if max_files < 0 or file_bytes < 0:
            raise ValueError("max_files and file_bytes must be non-negative")
        if path is not None and not os.path.isdir(path):
            raise ValueError(f"overflow folder {path!r} is not an existing directory")
        self.max_files = max_files
        self.file_bytes = file_bytes
        self.path = tempfile.mkdtemp(prefix=_DIR_PREFIX) if path is None else path
        self._lock = threading.Lock()
        self._cleanup = _Cleanup(self.path, path is None, self._lock)
        self._files = self._cleanup.files
        with _live_lock:
            _live.add(self._cleanup)

    @property
    def closed(self) -> bool:
        """Whether ``close()`` ran (its files are gone)."""
        return self._cleanup.done

    def file(self) -> OverflowFile | None:
        """An ``OverflowFile`` for one new writer, within the budget; ``None``: keep memory only.

        Deletes the oldest closed files to make room (see ``OverflowDir``),
        never an open writer's file. Returns ``None`` when ``max_files``
        files are still in use, ``file_bytes`` is 0, or the file cannot be
        created (logged). Raises ``ValueError`` after ``close()``.
        """
        with self._lock:
            if self._cleanup.done:
                raise ValueError("file() on a closed OverflowDir")
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
                _log.warning(
                    "could not create an overflow file; keeping memory only", exc_info=True
                )
                return None
            self._files[path] = "new"
        return OverflowFile(path=path, max_bytes=self.file_bytes, _owner=self)

    def _claim(self, path: str) -> bool:
        """Mark ``path`` open for one writer.

        Returns ``False`` when ``path`` is not a file of this folder that no
        writer has opened yet (already used, evicted, or the folder closed).
        """
        with self._lock:
            if self._files.get(path) != "new":
                return False
            self._files[path] = "open"
            return True

    def _finished(self, path: str, *, delete: bool) -> None:
        """Record that the writer of ``path`` finished: delete the file, or mark it evictable.

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
            self._cleanup.run()

    def __enter__(self) -> OverflowDir:
        """Return the folder itself, for ``with OverflowDir() as folder:``."""
        return self

    def __exit__(self, *exc: object) -> None:
        """``close()`` the folder on leaving the ``with`` block."""
        self.close()
