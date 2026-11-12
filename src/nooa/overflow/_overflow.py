# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``OverflowFile``, ``OverflowDir`` and the writer of one file; see ``nooa.overflow`` for the contract.

Built on the standard library: ``tempfile.mkdtemp`` for a private folder
(0700), ``os.open(..., O_CREAT | O_EXCL | O_NOFOLLOW, 0o600, dir_fd=...)``
for each file, ``os.unlink`` / ``os.rmdir`` / ``shutil.rmtree`` to remove
them and ``atexit`` for the exit cleanup. What it adds is the eviction
bookkeeping, the explicit lifetime (no ``weakref.finalize``: nothing is
deleted by garbage collection) and the identity of what it created.

Identity: an ``OverflowDir`` holds an open descriptor of its folder and
works relative to it (``dir_fd``), so renaming the folder, or putting a
symlink or another directory at its path, never redirects a create or a
delete. Each file is created exclusively with ``O_NOFOLLOW`` and written
through the descriptor it was created with (never reopened by name). A
name is deleted only after ``os.stat(name, dir_fd=..., follow_symlinks=False)``
shows the device and inode it created; a private folder is removed only
while its path still names the directory it made. Anything else at those
names is left alone and recorded (logged, ``_Cleanup.left``), never raised.
The check and the delete are two system calls: this defends against
mistakes and ordinary renames, not a same-user process racing between them.
Nor can a device and inode tell a file someone else deleted from a new
file the filesystem then gave the same inode number (only for a file no
descriptor holds open: a handed-out or open one is safe).

Platforms without ``dir_fd`` support (``os.supports_dir_fd``, e.g. Windows)
fall back to full paths with the same identity check (a renamed folder is
then followed by its old path, but a replaced entry is still left alone);
``O_NOFOLLOW`` is used where it exists.

Processes: cleanup runs only in the process that created the folder or the
file. In a forked child the inherited folders are inert (``file()`` returns
``None``, ``close()`` and exit do nothing to their files) and the exit
registry starts empty (``os.register_at_fork``); a child that makes its own
``OverflowDir`` cleans up only that one.
"""

from __future__ import annotations

import atexit
import contextlib
import logging
import os
import secrets
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
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
# Relative (dir_fd) operations; without them, full paths (see the module docstring).
_DIR_FD = (
    {os.open, os.stat, os.unlink} <= os.supports_dir_fd
    and os.scandir in os.supports_fd
    and shutil.rmtree.avoids_symlink_attacks
)
# An OverflowDir file's state: "new" (handed out by file()), "open" (a writer
# writes it), "closed" (its writer finished; it may be evicted) or "delete"
# (to delete, a delete failed: still counted, retried).
_State = Literal["new", "open", "closed", "delete"]
_Id = tuple[int, int]  # (st_dev, st_ino) of what we created


def _identity(st: os.stat_result) -> _Id:
    return (st.st_dev, st.st_ino)


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

        A standalone file is created: exclusively at ``path`` (mode 0600,
        never through a symlink), or as a new temporary file. An
        ``OverflowDir`` file already exists; it is claimed from its folder
        (once only) and written through the descriptor the folder created it
        with, never reopened by name.

        Returns:
            The ``OverflowWriter``, or ``None`` (logged; keep memory only)
            when the file cannot be created, or a folder's file was already
            opened, evicted, replaced, or its folder closed.

        Raises:
            FileExistsError: a standalone ``path`` already exists.
            ValueError: a negative ``max_bytes``.
        """
        if self.max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        owner = self._owner
        try:
            if owner is not None and self.path is not None:
                fd = owner._claim(self.path)
                if fd is None:
                    return None  # logged by _claim
                return OverflowWriter(
                    os.fdopen(fd, "wb"), self.path, min(self.max_bytes, owner.file_bytes), owner
                )
            if self.path is not None:
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | _CLOEXEC
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

    Not constructed directly: get one from ``OverflowFile.open()``.

    - ``path``: the file.
    - ``max_bytes``: the most bytes ``write`` keeps.
    - ``written``: the bytes written so far.
    - ``file``: the open binary file, ``None`` once finished. A caller may
      ``flush()`` or ``close()`` it directly to see its ``OSError``.

    ``write(data)`` writes what fits under ``max_bytes`` and returns how
    much; it raises ``OSError`` on a file error (the caller decides to give
    up: ``finish(delete=True)``). ``finish(delete=...)`` closes the file;
    with ``delete`` it removes it (only if ``path`` still names the file
    this writer created, and only in the process that created it), else a
    folder's file becomes evictable.
    """

    def __init__(
        self, file: IO[bytes], path: str, max_bytes: int, owner: OverflowDir | None
    ) -> None:
        self.file: IO[bytes] | None = file
        self.path = path
        self.max_bytes = max_bytes
        self.written = 0
        self._owner = owner
        self._pid = os.getpid()
        self._id: _Id | None = None if owner is not None else _identity(os.fstat(file.fileno()))

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
        elif delete and file is not None and self._id is not None and os.getpid() == self._pid:
            _remove(None, self.path, self.path, self._id)


def _remove(dir_fd: int | None, name: str, path: str, ident: _Id) -> bool:
    """Delete ``name`` (relative to ``dir_fd``, else ``path``) if it is still ``ident``.

    Never follows a symlink. Returns ``True`` when it is gone or is no
    longer ours (left alone, logged), ``False`` when the delete failed
    (logged; the caller keeps it to retry). Never raises.
    """
    where = {"dir_fd": dir_fd} if dir_fd is not None else {}
    target = name if dir_fd is not None else path
    try:
        if _identity(os.stat(target, follow_symlinks=False, **where)) != ident:
            _log.warning("overflow file %r was replaced; leaving it alone", path)
            return True
        os.unlink(target, **where)
    except FileNotFoundError:
        return True
    except OSError:
        _log.warning("could not delete overflow file %r", path, exc_info=True)
        return False
    return True


class _Entry:
    """One file an ``OverflowDir`` created: its state, identity and (until claimed) descriptor."""

    __slots__ = ("fd", "ident", "name", "state")

    def __init__(self, name: str, fd: int, ident: _Id) -> None:
        self.name = name
        self.fd: int | None = fd  # open from creation until a writer claims it
        self.ident = ident
        self.state: _State = "new"

    def close_fd(self) -> None:
        fd, self.fd = self.fd, None
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)


class _Cleanup:
    """What an ``OverflowDir`` must remove, held apart from it so that exit can run it.

    It holds no reference to the ``OverflowDir``: dropping the folder object
    deletes nothing; its cleanup stays registered until a ``close()`` (or the
    exit) fully succeeds. Only the process that created it runs it.
    """

    def __init__(self, path: str, owns_dir: bool, lock: threading.Lock) -> None:
        self.path = path
        self.owns_dir = owns_dir
        self.lock = lock
        self.pid = os.getpid()
        flags = os.O_RDONLY | _DIRECTORY | _CLOEXEC | (_NOFOLLOW if owns_dir else 0)
        self.dir_fd: int | None = os.open(path, flags) if _DIR_FD else None
        st = os.fstat(self.dir_fd) if self.dir_fd is not None else os.stat(path)
        self.dir_id = _identity(st)
        # The files it created, by path, oldest first.
        self.files: dict[str, _Entry] = {}
        self.left: list[str] = []  # names found replaced and left alone
        self.closing = False  # close() was called: no new files
        self.done = False  # cleanup completed

    def remove(self, path: str, entry: _Entry) -> bool:
        """Delete one file if it is still ours; ``True`` once it is no longer tracked."""
        entry.close_fd()
        removed = _remove(self.dir_fd, entry.name, path, entry.ident)
        if removed:
            del self.files[path]
        return removed

    def _remove_private_dir(self) -> bool | None:
        """Remove the whole private directory, while its path still names the one we made.

        ``True``: removed; ``False``: a removal failed (logged); ``None``:
        the path no longer names it (moved or replaced: left alone, logged).
        """
        try:
            same = _identity(os.stat(self.path, follow_symlinks=False)) == self.dir_id
        except FileNotFoundError:
            same = False
        if not same:
            if self.path not in self.left:
                self.left.append(self.path)
                _log.warning(
                    "overflow folder %r was moved or replaced; leaving it alone", self.path
                )
            return None
        if self.dir_fd is None:
            try:
                shutil.rmtree(self.path)
            except FileNotFoundError:
                pass
            except OSError:
                _log.warning("could not remove overflow folder %r", self.path, exc_info=True)
                return False
            return True
        ok = True
        for entry in list(os.scandir(self.dir_fd)):
            try:
                if entry.is_dir(follow_symlinks=False):
                    shutil.rmtree(entry.name, dir_fd=self.dir_fd)
                else:
                    os.unlink(entry.name, dir_fd=self.dir_fd)
            except FileNotFoundError:
                pass
            except OSError:
                ok = False
                _log.warning(
                    "could not remove %r in overflow folder %r",
                    entry.name,
                    self.path,
                    exc_info=True,
                )
        if ok:
            try:  # rmdir only removes an empty directory: never a replacement's contents
                os.rmdir(self.path)
            except FileNotFoundError:
                pass
            except OSError:
                _log.warning("could not remove overflow folder %r", self.path, exc_info=True)
                ok = False
        return ok

    def run(self) -> None:
        """Remove the private directory, or only the files it created; retry what failed. Call under ``lock``.

        Does nothing in another process (a forked child) or once done.
        """
        self.closing = True
        if self.done or os.getpid() != self.pid:
            return
        try:
            for entry in self.files.values():
                entry.close_fd()
            removed = self._remove_private_dir() if self.owns_dir else None
            if removed:
                self.files.clear()
            ok = removed is not False
            if removed is None:  # a given folder, or a private one moved away: our files only
                for path, entry in list(self.files.items()):
                    ok = self.remove(path, entry) and ok
        except OSError:
            _log.warning("could not clean up overflow folder %r", self.path, exc_info=True)
            ok = False
        if not ok:
            return  # still registered: the next close() or the exit retries
        self.done = True
        if self.dir_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self.dir_fd)
            self.dir_fd = None
        with _live_lock:
            _live.discard(self)


# Every OverflowDir not yet cleaned up, for the exit cleanup. Strong references to
# the cleanups only, never to the folders.
_live: set[_Cleanup] = set()
_live_lock = threading.Lock()


def _after_fork_in_child() -> None:
    """A forked child starts with no exit cleanups: the inherited ones are its parent's."""
    global _live, _live_lock  # noqa: PLW0603 - per-process registry
    _live = set()
    _live_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


@atexit.register
def _close_all_at_exit() -> None:
    """At interpreter exit, remove the files of every ``OverflowDir`` of this process not cleaned up."""
    with _live_lock:
        pending = [cleanup for cleanup in _live if cleanup.pid == os.getpid()]
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
    empty file in the folder (created exclusively, ``O_NOFOLLOW``, 0600,
    relative to the folder's open descriptor). To make room it first
    deletes closed files, oldest first; it never deletes the file of a
    writer still open (or of a file handed out and not yet finished). When
    every file is still in use, or the file cannot be created (logged), it
    returns ``None``: keep memory only. A writer that finishes with
    ``delete=True`` frees its place at once. A file whose delete failed
    stays counted against ``max_files`` and is retried by each ``file()``,
    ``close()`` and the exit.

    A file in an ``OverflowDir`` is not permanent: a path handed out earlier
    can name a file deleted since, because the folder needed room or
    closed ("keep the newest"). Opening that path then raises
    ``FileNotFoundError``.

    Lifetime: ``close()`` (also on leaving a ``with`` block) removes the
    private directory, or unlinks each name it created in a given one;
    ``file()`` then raises ``ValueError``. It deletes only what this folder
    created (the same device and inode, never through a symlink); something
    put in its place is left alone (logged). When a delete fails (logged,
    never raised), the folder stays registered and the next ``close()`` or
    the exit retries; once everything is gone, later calls do nothing. A
    folder never closed is cleaned up at interpreter exit (``atexit``),
    **not** when it is garbage collected: its files outlive the object. A
    writer still open when the folder closes keeps running; its file is
    gone.

    Processes: only the process that created the folder deletes its files.
    In a forked child the inherited folder is inert: ``file()`` returns
    ``None`` (logged), and ``close()`` and the child's exit leave the
    parent's files alone. A child makes its own ``OverflowDir``.

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
        try:
            self._cleanup = _Cleanup(self.path, path is None, self._lock)
        except BaseException:
            if path is None:
                with contextlib.suppress(OSError):
                    os.rmdir(self.path)
            raise
        self._files = self._cleanup.files
        with _live_lock:
            _live.add(self._cleanup)

    @property
    def closed(self) -> bool:
        """Whether ``close()`` ran (no new files; its files are gone, or a retry is pending)."""
        return self._cleanup.closing

    def _foreign_process(self) -> bool:
        """True in a forked child: the folder belongs to the parent (logged)."""
        if os.getpid() == self._cleanup.pid:
            return False
        _log.warning(
            "overflow folder %r belongs to the parent process; keeping memory only", self.path
        )
        return True

    def file(self) -> OverflowFile | None:
        """An ``OverflowFile`` for one new writer, within the budget; ``None``: keep memory only.

        Retries the deletes that failed, then deletes the oldest closed
        files to make room (see ``OverflowDir``), never an open writer's
        file. Returns ``None`` when ``max_files`` files are still in use
        (or could not be deleted), ``file_bytes`` is 0, the file cannot be
        created (logged), or in a forked child. Raises ``ValueError`` after
        ``close()``.
        """
        if self._foreign_process():
            return None  # never take the lock: a parent thread may have held it at fork
        with self._lock:
            cleanup = self._cleanup
            if cleanup.closing:
                raise ValueError("file() on a closed OverflowDir")
            for path, entry in list(self._files.items()):
                if entry.state == "delete":
                    cleanup.remove(path, entry)
            for path, entry in list(self._files.items()):
                if len(self._files) < self.max_files:
                    break
                if entry.state == "closed" and not cleanup.remove(path, entry):
                    entry.state = "delete"
            if len(self._files) >= self.max_files or self.file_bytes == 0:
                return None
            try:
                path, entry = self._create()
            except OSError:
                _log.warning(
                    "could not create an overflow file; keeping memory only", exc_info=True
                )
                return None
            self._files[path] = entry
        return OverflowFile(path=path, max_bytes=self.file_bytes, _owner=self)

    def _create(self) -> tuple[str, _Entry]:
        """Create a new empty file in the folder, exclusively, never through a symlink."""
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW | _CLOEXEC
        for _ in range(100):
            name = _PREFIX + secrets.token_hex(6) + _SUFFIX
            path = os.path.join(self.path, name)
            try:
                if self._cleanup.dir_fd is not None:
                    fd = os.open(name, flags, 0o600, dir_fd=self._cleanup.dir_fd)
                else:
                    fd = os.open(path, flags, 0o600)
            except FileExistsError:
                continue
            return path, _Entry(name, fd, _identity(os.fstat(fd)))
        raise FileExistsError(f"no free overflow file name in {self.path!r}")

    def _claim(self, path: str) -> int | None:
        """Hand the descriptor of ``path`` to its one writer, and mark it open.

        Returns ``None`` (logged) when ``path`` is not a file of this folder
        that no writer has opened yet (already used, evicted, the folder
        closed, or another process), or when its name no longer names the
        file this folder created (replaced: left alone, its place freed).
        """
        if self._foreign_process():
            return None
        with self._lock:
            entry = self._files.get(path)
            if entry is None or entry.state != "new" or entry.fd is None:
                _log.warning(
                    "overflow file %r is not open to a new writer; keeping memory only", path
                )
                return None
            where = {"dir_fd": self._cleanup.dir_fd} if self._cleanup.dir_fd is not None else {}
            target = entry.name if where else path
            try:
                same = _identity(os.stat(target, follow_symlinks=False, **where)) == entry.ident
            except OSError:
                same = False
            if not same:
                _log.warning("overflow file %r was replaced; keeping memory only", path)
                entry.close_fd()
                self._cleanup.left.append(path)
                del self._files[path]
                return None
            fd, entry.fd = entry.fd, None
            entry.state = "open"
            return fd

    def _finished(self, path: str, *, delete: bool) -> None:
        """Record that the writer of ``path`` finished: delete the file, or mark it evictable.

        A no-op when ``path`` is not open in this folder (the folder closed),
        or in another process. A failed delete keeps the file counted, to retry.
        """
        if os.getpid() != self._cleanup.pid:
            return
        with self._lock:
            entry = self._files.get(path)
            if entry is None or entry.state != "open":
                return  # the folder closed
            if delete and not self._cleanup.remove(path, entry):
                entry.state = "delete"
            elif not delete:
                entry.state = "closed"

    def close(self) -> None:
        """Remove every file this folder created (the whole private directory).

        Never raises: file errors are logged, and what could not be deleted
        is retried by the next ``close()`` and at exit. Does nothing in a
        forked child, or once everything is gone.
        """
        if os.getpid() != self._cleanup.pid:
            return
        with self._lock:
            self._cleanup.run()

    def __enter__(self) -> OverflowDir:
        """Return the folder itself, for ``with OverflowDir() as folder:``."""
        return self

    def __exit__(self, *exc: object) -> None:
        """``close()`` the folder on leaving the ``with`` block."""
        self.close()
