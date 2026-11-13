# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``OverflowFile``, ``OverflowDir`` and the writer of one file; see ``nooa.overflow`` for the contract.

Built on the standard library: ``tempfile.mkdtemp`` for the private folder
(0700), ``os.open(..., O_CREAT | O_EXCL | O_NOFOLLOW, 0o600, dir_fd=...)``
for each file, ``os.write`` on that descriptor, ``os.unlink`` /
``shutil.rmtree`` relative to the folder's descriptor to remove them, and
``atexit`` for the exit cleanup. What it adds is the eviction bookkeeping
and the explicit lifetime (no ``weakref.finalize``: nothing is deleted by
garbage collection).

Threat model (CPython's ``tempfile``/``shutil.rmtree`` line): the folder is
private (0700, an unpredictable name), so no other user can touch it. The
same user changing things inside their own private folder (renaming,
replacing or linking our files) is out of scope: no file API can win there.
Writes go through the descriptor each file was created with, so a name
replaced after creation is never written; deletes are by name.

Platforms without ``dir_fd`` support (``os.supports_dir_fd``, e.g. Windows)
use full paths instead of the folder's descriptor; ``O_NOFOLLOW`` is used
where it exists (not on Windows, where a symlink put in the private folder
would be followed), and ``O_BINARY`` where it exists. On Windows an open
file cannot be deleted: a folder closed while a writer is open keeps that
file until a later ``close()`` or the exit retries.

Processes: an ``os.register_at_fork`` hook makes everything inherited by a
forked child inert there (no writes, deletes or exit cleanup; fresh locks,
so the child never waits on a lock held at the fork); a child's own
``OverflowDir`` works normally.

Shutdown: the exit cleanup takes each folder's lock (never runs without
it). A folder whose lock a thread holds for more than a second is marked
for cleanup and that thread removes it when it lets go (a ``file()`` in
progress then hands out nothing). An ``OverflowDir`` made after the exit
cleanup started gets no folder at all.
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
from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar, Literal

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
_CREATE = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_BINARY", 0)
)
# Relative (dir_fd) operations; without them, full paths (see the module docstring).
_DIR_FD = (
    {os.open, os.unlink} <= os.supports_dir_fd
    and os.scandir in os.supports_fd
    and shutil.rmtree.avoids_symlink_attacks
)
# A folder file's state: "new" (handed out by file(), not opened), "open" (a writer
# writes it), "closed" (its writer finished; evictable) or "delete" (a delete
# failed: still counted, retried).
_State = Literal["new", "open", "closed", "delete"]


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
    ``None`` (keep memory only). A file built here belongs to no folder:
    nothing deletes it but its writer (``finish(delete=True)``).

    ``OverflowDir.file()`` hands out files of its folder. Only that call
    makes one: the link to the folder is not a field, so neither this
    constructor nor ``dataclasses.replace`` can make or keep it (a copy made
    by ``replace`` is a standalone file at an existing path, which
    ``open()`` refuses). A folder's file opens one writer (a second
    ``open()`` returns ``None``, logged), never more than the folder's
    ``file_bytes``. ``release()`` gives back a folder's file that will not
    be opened: until it is opened and finished, or released, it holds one
    of the folder's ``max_files`` places.
    """

    path: str | None = None
    max_bytes: int = DEFAULT_FILE_BYTES
    # The folder that handed this file out, set on the instance by ``OverflowDir.file``
    # only (not a field: no constructor argument, not copied by ``dataclasses.replace``).
    _folder: ClassVar[_Folder | None] = None

    def open(self) -> OverflowWriter | None:
        """Open the file for one writer.

        A standalone file is created: exclusively at ``path`` (mode 0600,
        never through a symlink), or as a new temporary file. A folder's
        file already exists; it is claimed from its folder (once only) and
        written through the descriptor the folder created it with, never
        reopened by name.

        Returns:
            The ``OverflowWriter``, or ``None`` (logged; keep memory only)
            when the file cannot be created, or a folder's file was already
            opened or released, or its folder closed.

        Raises:
            FileExistsError: a standalone ``path`` already exists.
            ValueError: a negative ``max_bytes``.
        """
        if self.max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        if self._folder is not None and self.path is not None:
            return self._folder.claim(os.path.basename(self.path), self.max_bytes)
        fd, path = None, self.path
        try:
            if self.path is not None:
                fd, path = os.open(self.path, _CREATE, 0o600), self.path
            else:
                fd, path = tempfile.mkstemp(prefix=_PREFIX, suffix=_SUFFIX)
            return OverflowWriter(fd, path, self.max_bytes, None)
        except FileExistsError:
            raise  # a usage error, not a file error: the caller named an existing file
        except OSError:
            _log.warning("could not create the overflow file; keeping memory only", exc_info=True)
            return None
        except BaseException:  # interrupted after creating: remove what was made
            if fd is not None and path is not None:
                with contextlib.suppress(OSError):
                    os.close(fd)
                with contextlib.suppress(OSError):
                    os.unlink(path)
            raise

    def release(self) -> None:
        """Give a folder's file that was never opened back to its folder: its place is free again.

        The empty file is deleted and a later ``open()`` returns ``None``.
        Does nothing for a standalone file, or once the file was opened
        (finish its writer instead), released or its folder closed.
        """
        if self._folder is not None and self.path is not None:
            self._folder.release(os.path.basename(self.path))


class OverflowWriter:
    """The open file of one ``OverflowFile``: what ``OverflowFile.open()`` returns.

    Not constructed directly: get one from ``OverflowFile.open()``.

    - ``path``: the file (read-only).
    - ``max_bytes``: the most bytes ``write`` keeps (read-only).
    - ``written``: the bytes written to the file so far (read-only): only
      bytes the operating system took, never bytes still in a buffer.
    - ``closed``: whether ``finish`` ran (or the writer belongs to the
      parent of a forked process).

    ``write(data)`` writes what fits under ``max_bytes`` straight to the
    file's descriptor (there is no buffer) and returns how much; it raises
    ``OSError`` on a file error (the caller decides to give up:
    ``finish(delete=True)``). ``finish(delete=...)`` closes the file; with
    ``delete`` it removes it, else a folder's file becomes evictable. Only
    ``finish`` closes the descriptor; it is never handed out.

    One thread at a time: a writer has no lock. In a forked child an
    inherited writer is inert: ``write`` returns 0 and ``finish`` only closes
    the child's copy of the descriptor.
    """

    def __init__(self, fd: int, path: str, max_bytes: int, folder: _Folder | None) -> None:
        self._fd: int | None = fd
        self._path = path
        self._max_bytes = max_bytes
        self._written = 0
        self._folder = folder
        self._kept = False  # finish(delete=False) closed it cleanly
        self._pid = os.getpid()
        st = os.fstat(fd)
        self._id = (st.st_dev, st.st_ino)

    @property
    def path(self) -> str:
        """The file."""
        return self._path

    @property
    def max_bytes(self) -> int:
        """The most bytes ``write`` keeps."""
        return self._max_bytes

    @property
    def written(self) -> int:
        """The bytes written to the file so far."""
        return self._written

    @property
    def closed(self) -> bool:
        """Whether ``finish`` ran, or the writer was inherited by a forked child."""
        return self._fd is None or os.getpid() != self._pid

    def write(self, data: bytes) -> int:
        """Write the part of ``data`` that fits under ``max_bytes``; return its length.

        Returns 0 once finished or full. Raises ``OSError`` on a file error;
        ``written`` then counts what reached the file before it.
        """
        if self.closed:
            return 0
        part = memoryview(data)[: max(0, self._max_bytes - self._written)]
        done = 0
        while done < len(part):
            n = os.write(self._fd, part[done:])  # type: ignore[arg-type]  # not closed
            if n <= 0:
                raise OSError(f"no progress writing overflow file {self._path!r}")
            done += n
            self._written += n
        return done

    def finish(self, *, delete: bool) -> bool:
        """Close the file and, with ``delete``, remove it; a folder's file is handed back to its folder.

        Returns whether the file is kept: ``True`` after ``delete=False``
        when the file closed cleanly. Never raises: a failed close is
        logged and the file, which may not hold ``written`` bytes, is
        deleted (``False``). A later ``finish(delete=True)`` deletes a kept
        file; any other later call returns the same answer. In a forked
        child it only closes the child's copy of the descriptor
        (``False``).
        """
        fd, self._fd = self._fd, None
        if os.getpid() != self._pid:
            if fd is not None:
                with contextlib.suppress(OSError):
                    os.close(fd)
            return False
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                _log.warning(
                    "could not close overflow file %r; deleting it", self._path, exc_info=True
                )
                delete = True
            if not delete:
                self._kept = True
                if self._folder is not None:
                    self._folder.finished(os.path.basename(self._path), delete=False)
                return True
        elif not (delete and self._kept):
            return self._kept  # finished before: the same answer
        self._kept = False
        if self._folder is not None:
            self._folder.finished(os.path.basename(self._path), delete=True)
        else:
            _remove_standalone(self._path, self._id)
        return False


def _remove_standalone(path: str, ident: tuple[int, int]) -> None:
    """Delete a standalone file if ``path`` still names it (never follows a symlink); log failures."""
    try:
        st = os.stat(path, follow_symlinks=False)
        if (st.st_dev, st.st_ino) != ident:
            _log.warning("overflow file %r was replaced; leaving it alone", path)
            return
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError:
        _log.warning("could not delete overflow file %r", path, exc_info=True)


class _Entry:
    """One file a folder created: its state and, until a writer claims it, its descriptor."""

    __slots__ = ("fd", "state")

    def __init__(self) -> None:
        self.fd: int | None = None
        self.state: _State = "new"


class _Folder:
    """An ``OverflowDir``'s state, held apart from it so that the exit cleanup can reach it.

    It holds no reference to the ``OverflowDir``: dropping the folder object
    deletes nothing; it stays registered (``_live``) until a ``close()`` or
    the exit removes everything. Every method that changes it runs under
    ``lock``, in the process that created it, through ``locked()``.
    """

    def __init__(self, parent: str | None, max_files: int, file_bytes: int) -> None:
        self.max_files = max_files
        self.file_bytes = file_bytes
        self.pid = os.getpid()
        self.lock = threading.Lock()
        self.entries: dict[str, _Entry] = {}  # by name; closed ones in finish order, oldest first
        self.closing = False  # close() or the exit began: no new files
        self.at_exit = False  # the exit closed it: file() returns None instead of raising
        self.exit_pending = False  # the exit could not take the lock: its holder cleans up
        self.done = False  # everything removed
        self.path = tempfile.mkdtemp(prefix=_DIR_PREFIX, dir=parent)
        self.dir_fd: int | None = None
        if _DIR_FD:
            try:
                self.dir_fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            except BaseException:
                os.rmdir(self.path)
                raise

    def _at(self, name: str) -> tuple[str, dict[str, int]]:
        """``name`` as a path argument: relative to the folder's descriptor, else a full path."""
        if self.dir_fd is not None:
            return name, {"dir_fd": self.dir_fd}
        return os.path.join(self.path, name), {}

    def mine(self) -> bool:
        """True in the process that created the folder; in a forked child the folder is inert."""
        return os.getpid() == self.pid

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        """Hold the lock; on leaving, run a cleanup the exit asked for while it was held."""
        with self.lock:
            try:
                yield
            finally:
                if self.exit_pending and not self.done:
                    self.run()

    def remove(self, name: str) -> bool:
        """Delete one file; ``True`` once it is gone, ``False`` (logged) to keep it counted and retry."""
        entry = self.entries[name]
        fd, entry.fd = entry.fd, None
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        target, where = self._at(name)
        try:
            os.unlink(target, **where)
        except FileNotFoundError:
            pass
        except OSError:
            _log.warning("could not delete overflow file %r", target, exc_info=True)
            entry.state = "delete"
            return False
        del self.entries[name]
        return True

    def new_file(self) -> str | None:
        """Create a new empty file and make room for it; its name, or ``None`` (memory only). Under the lock.

        The place is reserved before the file is created and given up on
        any exception, so an interrupt never leaves a file uncounted.
        Closed files are evicted only after the new file exists, oldest
        finished first; when they cannot be, the new file goes instead.
        """
        for name in [n for n, e in self.entries.items() if e.state == "delete"]:
            self.remove(name)
        closed = [n for n, e in self.entries.items() if e.state == "closed"]
        if self.file_bytes == 0 or len(self.entries) - len(closed) >= self.max_files:
            return None
        name = _PREFIX + secrets.token_hex(6) + _SUFFIX
        while name in self.entries:
            name = _PREFIX + secrets.token_hex(6) + _SUFFIX
        entry = self.entries[name] = _Entry()  # reserved first: counted from here on
        target, where = self._at(name)
        try:
            entry.fd = os.open(target, _CREATE, 0o600, **where)
        except BaseException as exc:
            del self.entries[name]
            if not isinstance(exc, OSError):
                # Interrupted, perhaps after the file was made: the name is ours, remove it.
                with contextlib.suppress(OSError):
                    os.unlink(target, **where)
                raise
            _log.warning("could not create an overflow file; keeping memory only", exc_info=True)
            return None

        try:
            for old in closed:
                if len(self.entries) <= self.max_files:
                    break
                self.remove(old)
            if len(self.entries) > self.max_files or self.closing:
                self.remove(name)  # no room after all, or the exit began meanwhile
                return None
        except BaseException:
            if name in self.entries:
                self.remove(name)
            raise
        return name

    def claim(self, name: str, max_bytes: int) -> OverflowWriter | None:
        """The writer of the handed-out file ``name``; ``None`` (logged) when it cannot be opened."""
        if not self.mine():
            _log.warning(
                "overflow file %r belongs to the parent process; keeping memory only", name
            )
            return None
        with self.locked():
            entry = self.entries.get(name)
            if self.closing or entry is None or entry.state != "new" or entry.fd is None:
                _log.warning(
                    "overflow file %r is not open to a new writer; keeping memory only", name
                )
                return None
            fd, entry.fd, entry.state = entry.fd, None, "open"
            try:
                return OverflowWriter(
                    fd, os.path.join(self.path, name), min(max_bytes, self.file_bytes), self
                )
            except BaseException:  # interrupted: the file stays handed out, unopened
                entry.fd, entry.state = fd, "new"
                raise

    def release(self, name: str) -> None:
        """Delete the handed-out file ``name`` if no writer opened it."""
        if not self.mine():
            return
        with self.locked():
            entry = self.entries.get(name)
            if not self.closing and entry is not None and entry.state == "new":
                self.remove(name)

    def finished(self, name: str, *, delete: bool) -> None:
        """The writer of ``name`` finished: delete it, or make it the newest closed file."""
        if not self.mine():
            return
        with self.locked():
            entry = self.entries.get(name)
            if self.closing or entry is None or entry.state not in ("open", "closed"):
                return  # the folder closed, or evicted it
            if delete:
                self.remove(name)
            else:
                entry.state = "closed"
                self.entries[name] = self.entries.pop(name)  # newest finished: evicted last

    def run(self) -> None:
        """Remove the whole private folder; on failure (logged) stay registered to retry. Under the lock."""
        self.closing = True
        if self.done or not self.mine():
            return
        for entry in self.entries.values():
            fd, entry.fd = entry.fd, None
            if fd is not None:
                with contextlib.suppress(OSError):
                    os.close(fd)
        ok = True

        def failed(_func: object, path: object, exc: BaseException) -> None:
            nonlocal ok
            if not isinstance(exc, FileNotFoundError):
                ok = False
                _log.warning(
                    "could not remove %r in overflow folder %r", path, self.path, exc_info=exc
                )

        try:
            if self.dir_fd is None:
                shutil.rmtree(self.path, onexc=failed)
            else:
                for item in list(os.scandir(self.dir_fd)):
                    try:
                        if item.is_dir(follow_symlinks=False):
                            shutil.rmtree(item.name, dir_fd=self.dir_fd, onexc=failed)
                        else:
                            os.unlink(item.name, dir_fd=self.dir_fd)
                    except OSError as exc:
                        failed(None, item.name, exc)
                if ok:
                    os.rmdir(self.path)
        except FileNotFoundError:
            pass
        except OSError:
            ok = False
            _log.warning("could not remove overflow folder %r", self.path, exc_info=True)
        if not ok:
            return  # still registered: the next close() or the exit retries
        self.done = True
        self.entries.clear()
        if self.dir_fd is not None:
            with contextlib.suppress(OSError):
                os.close(self.dir_fd)
            self.dir_fd = None
        with _live_lock:
            _live.discard(self)

    def close_at_exit(self) -> None:
        """The exit cleanup: under the lock, or by the thread holding it when it lets go."""
        self.closing = self.at_exit = self.exit_pending = True
        if self.lock.acquire(timeout=1.0):
            try:
                self.run()
            finally:
                self.lock.release()
        elif not self.done:
            _log.warning(
                "overflow folder %r is busy at exit; the thread using it removes it", self.path
            )

    def forked(self) -> None:
        """In a forked child: inert, with a fresh lock, and without the parent's descriptors."""
        self.lock = threading.Lock()
        for fd in [e.fd for e in self.entries.values()] + [self.dir_fd]:
            if fd is not None:
                with contextlib.suppress(OSError):
                    os.close(fd)


# Every folder not yet removed, for the exit cleanup: strong references to the
# ``_Folder`` states only, never to the ``OverflowDir`` objects. ``_exiting``:
# the exit cleanup started, so a new ``OverflowDir`` gets no folder.
_live: set[_Folder] = set()
_live_lock = threading.Lock()  # held for set updates only, never across I/O
_exiting = False


def _after_fork_in_child() -> None:
    """A forked child starts with no exit cleanups: the inherited folders are its parent's (inert here)."""
    global _live, _live_lock  # noqa: PLW0603 - per-process registry
    for folder in _live:
        folder.forked()
    _live = set()
    _live_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


@atexit.register
def _close_all_at_exit() -> None:
    """At interpreter exit, remove every folder of this process not yet removed."""
    global _exiting  # noqa: PLW0603 - set once, at exit
    with _live_lock:
        _exiting = True
        pending = list(_live)
    for folder in pending:
        try:
            folder.close_at_exit()
        except Exception:  # noqa: BLE001 - never fail the exit
            _log.warning("could not clean up overflow folder %r", folder.path, exc_info=True)


class OverflowDir:
    """A private folder of overflow files shared by many writers: at most ``max_files`` of ``file_bytes`` each.

    ``OverflowDir(parent=None, *, max_files=20, file_bytes=64 * 2**20)``:

    - ``parent`` (default ``None``: the system temporary directory): where
      to create the folder. The ``OverflowDir`` always creates its own
      private folder there (``tempfile.mkdtemp``, ``nooa-overflow-*``,
      mode 0700, an unpredictable name) and removes it whole when closed.
    - ``max_files``: at most this many of its files exist at once.
    - ``file_bytes``: each file holds at most this many bytes, so the
      folder never holds more than ``max_files * file_bytes``.

    ``file()`` hands out an ``OverflowFile`` for one writer, with a new
    empty file in the folder (created exclusively, ``O_NOFOLLOW``, 0600,
    relative to the folder's open descriptor). It creates the file first,
    then makes room by deleting closed files, the oldest finished first
    ("keep the newest"): while it runs the new, empty file can be one more
    than ``max_files``. It never deletes the file of a writer still open, or
    of a file handed out and neither opened nor released. When every place
    is in use, or the file cannot be created (logged), it returns ``None``:
    keep memory only; nothing was evicted. A writer that finishes with
    ``delete=True`` frees its place at once, as does
    ``OverflowFile.release()`` for a file that will not be opened. A file
    whose delete failed stays counted against ``max_files`` and is retried
    by each ``file()``, ``close()`` and the exit.

    A file in an ``OverflowDir`` is not permanent: a path handed out earlier
    can name a file deleted since, because the folder needed room or
    closed. Reading that path then raises ``FileNotFoundError``.

    Lifetime: ``close()`` (also on leaving a ``with`` block) removes the
    whole folder; ``file()`` then raises ``ValueError``. When a removal
    fails (logged, never raised), the folder stays registered and the next
    ``close()`` or the exit retries; once everything is gone, later calls
    do nothing. A folder never closed is removed at interpreter exit
    (``atexit``), **not** when it is garbage collected: its files outlive
    the object. A writer still open when the folder closes keeps running;
    its file is gone. During the exit, ``file()`` returns ``None``.

    Processes: in a forked child an inherited folder is inert: ``file()``
    returns ``None`` (logged), and ``close()`` and the child's exit leave
    the parent's files alone. A child makes its own ``OverflowDir``.

    Threads: ``file()``, ``close()``, ``OverflowFile.open()`` /
    ``release()`` and the writers' ``finish`` may run in any threads; the
    bookkeeping is locked.

    Raises ``ValueError`` for a negative ``max_files`` or ``file_bytes``,
    and ``OSError`` when the folder cannot be created (``parent`` missing,
    not a directory, not writable).
    """

    def __init__(
        self,
        parent: str | None = None,
        *,
        max_files: int = DEFAULT_MAX_FILES,
        file_bytes: int = DEFAULT_FILE_BYTES,
    ) -> None:
        if max_files < 0 or file_bytes < 0:
            raise ValueError("max_files and file_bytes must be non-negative")
        folder = _Folder(parent, max_files, file_bytes)
        with _live_lock:
            exiting = _exiting
            if not exiting:
                _live.add(folder)
        if exiting:  # made after the exit cleanup started: nothing would remove it
            with folder.lock:
                folder.at_exit = True
                folder.run()
        self._folder = folder

    @property
    def path(self) -> str:
        """The private folder (gone once closed)."""
        return self._folder.path

    @property
    def max_files(self) -> int:
        """At most this many files at once, as given."""
        return self._folder.max_files

    @property
    def file_bytes(self) -> int:
        """At most this many bytes per file, as given."""
        return self._folder.file_bytes

    @property
    def closed(self) -> bool:
        """Whether ``close()`` (or the exit) ran: no new files; its files are gone, or a retry is pending."""
        return self._folder.closing

    def file(self, max_bytes: int | None = None) -> OverflowFile | None:
        """An ``OverflowFile`` for one new writer, within the budget; ``None``: keep memory only.

        ``max_bytes``: this file's cap, at most ``file_bytes`` (``None``:
        ``file_bytes``), for a folder shared by writers with smaller limits.
        Retries the deletes that failed, creates the file, then deletes the
        oldest finished closed files to make room (see ``OverflowDir``).
        Returns ``None`` when ``max_files`` files are still in use (or could
        not be deleted), ``file_bytes`` is 0, the file cannot be created
        (logged), during the exit, or in a forked child. Raises
        ``ValueError`` after ``close()``.
        """
        folder = self._folder
        if not folder.mine():  # never take the lock: a parent thread may have held it at fork
            _log.warning(
                "overflow folder %r belongs to the parent process; keeping memory only", self.path
            )
            return None
        with folder.locked():
            if folder.closing:
                if folder.at_exit:
                    return None
                raise ValueError("file() on a closed OverflowDir")
            name = folder.new_file()
        if name is None:
            return None
        cap = folder.file_bytes if max_bytes is None else min(max_bytes, folder.file_bytes)
        file = OverflowFile(path=os.path.join(folder.path, name), max_bytes=cap)
        object.__setattr__(file, "_folder", folder)  # not a field: see OverflowFile
        return file

    def close(self) -> None:
        """Remove the whole folder and every file in it.

        Never raises: failures are logged, and what could not be removed
        is retried by the next ``close()`` and at exit. Does nothing in a
        forked child, or once everything is gone.
        """
        folder = self._folder
        if folder.mine():
            with folder.locked():
                folder.run()

    def __enter__(self) -> OverflowDir:
        """Return the folder itself, for ``with OverflowDir() as folder:``."""
        return self

    def __exit__(self, *exc: object) -> None:
        """``close()`` the folder on leaving the ``with`` block."""
        self.close()
