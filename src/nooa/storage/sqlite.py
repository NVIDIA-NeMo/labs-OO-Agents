# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SQLite-backed StorageManager and EventBackend.

Provides persistent storage using stdlib sqlite3 — no new dependencies.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import sqlite3
import subprocess
import threading
import typing
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Self

from pydantic import ValidationError as PydanticValidationError

from nooa.context_blocks import Event, EventBase, EventStatus, Metadata
from nooa.context_blocks.events import _EVENT_REGISTRY
from nooa.events import (
    AfterTurn,
    BeforeTurn,
    Error,
    Feedback,
    LLMOutput,
    Message,
    PythonOutput,
    Reasoning,
    Summary,
    Task,
    TuiSessionCleared,
    TuiSessionResumed,
)
from nooa.storage.json_snapshot import snapshot_from_dict, snapshot_to_dict
from nooa.storage.snapshot import AgentSnapshot

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from nooa.agent import Agent

# Unwrap Event = UserEvent | AssistantEvent | ToolCallEvent
# to get the concrete types. This stays in sync with the Event union automatically.
_CONTEXT_BLOCKS_TYPES: tuple[type[EventBase], ...] = typing.get_args(Event)
if len(_CONTEXT_BLOCKS_TYPES) < 3:
    raise AssertionError(
        f"Failed to unwrap context_blocks Event union; got {_CONTEXT_BLOCKS_TYPES!r}. "
        "If Event's structure changed, update the get_args unwrap in sqlite.py."
    )

# All core event types for deserialization, keyed by their registry name
# (class name, or explicit event_type default if non-empty).
# These are also present in _EVENT_REGISTRY, but we keep a snapshot here
# to seed per-instance registries and to validate at import time.
_CORE_TYPES: dict[str, type[EventBase]] = {}


def _registry_key(cls: type[EventBase]) -> str:
    """Derive the registry key for an EventBase subclass.

    Uses the explicit ``event_type`` field default if non-empty,
    otherwise falls back to ``cls.__name__``.
    """
    fi = cls.model_fields.get("event_type")
    default = fi.default if fi is not None and isinstance(fi.default, str) else ""
    return default if default else cls.__name__


for _cls in (
    *_CONTEXT_BLOCKS_TYPES,
    Task,
    Message,
    Reasoning,
    Error,
    Feedback,
    LLMOutput,
    PythonOutput,
    Summary,
    BeforeTurn,
    AfterTurn,
    TuiSessionResumed,
    TuiSessionCleared,
):
    _key: str = _registry_key(_cls)
    if _key in _CORE_TYPES:
        raise AssertionError(
            f"Duplicate event_type key {_key!r} in _CORE_TYPES: "
            f"{_CORE_TYPES[_key].__name__} vs {_cls.__name__}. "
            "Each event class must have a unique event_type default."
        )
    _CORE_TYPES[_key] = _cls

_SCHEMA_VERSION = 1

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

-- event_type and status are denormalized from the JSON data blob for indexed
-- queries. All write paths must derive these columns from the blob to keep
-- them in sync. The JSON blob (data) is the source of truth.
CREATE TABLE IF NOT EXISTS events (
    tag             TEXT PRIMARY KEY,
    event_id        TEXT NOT NULL,
    event_type      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'active',
    data            TEXT NOT NULL,
    insertion_order INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_event_id ON events(event_id);
CREATE INDEX IF NOT EXISTS idx_events_insertion_order ON events(insertion_order);
CREATE INDEX IF NOT EXISTS idx_events_type_order ON events(event_type, insertion_order);

CREATE TABLE IF NOT EXISTS active_tags (
    position INTEGER NOT NULL,
    tag      TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    data        TEXT NOT NULL
);
"""


@lru_cache(maxsize=8)
def _is_virtiofs(db_path: str) -> bool:
    """Detect if *db_path* resides on a virtiofs mount (Docker Desktop file sharing)."""
    if db_path == ":memory:":
        return False

    import platform

    if platform.system() == "Linux":
        try:
            result = subprocess.run(
                ["df", "-T", "--", db_path],
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError) as e:
            logger.debug("virtiofs detection failed for %s: %s", db_path, type(e).__name__)
            return False
        return "virtiofs" in result.stdout

    # macOS / other: parse mount output for virtiofs
    try:
        result = subprocess.run(
            ["mount"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError) as e:
        logger.debug("virtiofs detection failed for %s: %s", db_path, type(e).__name__)
        return False

    import os

    resolved = os.path.realpath(db_path)
    for line in result.stdout.splitlines():
        if "virtiofs" in line:
            parts = line.split(" on ")
            if len(parts) >= 2:
                mount_point = parts[1].split(" (")[0].strip()
                if resolved.startswith(mount_point):
                    return True
    return False


def _is_corruption_error(exc: sqlite3.Error) -> bool:
    """Return True for SQLite failures caused by corrupt/unreadable DB content."""
    code = getattr(exc, "sqlite_errorcode", None)
    _CORRUPT = getattr(sqlite3, "SQLITE_CORRUPT", 11)
    _NOTADB = getattr(sqlite3, "SQLITE_NOTADB", 26)
    if code in {_CORRUPT, _NOTADB}:
        return True
    msg = str(exc).lower()
    return any(
        marker in msg
        for marker in (
            "could not decode to utf-8",
            "database disk image is malformed",
            "file is not a database",
        )
    )


class CorruptDatabaseError(sqlite3.DatabaseError):
    """Raised when a write fails because the SQLite file is corrupt.

    Subclasses sqlite3.DatabaseError so existing ``except sqlite3.Error``
    handlers still catch it, while giving callers (e.g. the TUI) a typed
    signal to surface recovery guidance instead of an opaque crash.
    """


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables if needed and verify schema version."""
    conn.executescript(_SCHEMA)
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (_SCHEMA_VERSION,))
        conn.commit()
    elif row[0] != _SCHEMA_VERSION:
        raise RuntimeError(
            f"SQLite schema version mismatch: DB has v{row[0]}, "
            f"code expects v{_SCHEMA_VERSION}. Migration required."
        )


class SQLiteEventBackend:
    """EventBackend backed by SQLite tables.

    Thread-safe: all methods acquire ``self._lock`` (a ``threading.RLock``
    shared with the owning ``SQLiteStorageManager``) before touching the
    connection. Multiple threads may call into the backend concurrently;
    access is serialized by the lock.
    """

    def __init__(self, conn: sqlite3.Connection, lock: threading.RLock | None = None) -> None:
        self._conn = conn
        self._lock = lock or threading.RLock()
        self._insertion_counter = self._max_insertion_order() + 1
        self._registry: dict[str, type[EventBase]] = dict(_CORE_TYPES)
        # Tag counter — eager-init from storage so a resumed DB picks
        # up at the right number. ``store()`` keeps it coherent when a
        # caller writes a tag higher than the current value.
        self._next_tag_num: int = self.max_tag_num() + 1
        # Optional callback invoked on OperationalError("disk I/O error")
        # to allow the owning StorageManager to reconnect before retry.
        self._on_io_error: typing.Callable[[], None] | None = None

    def register_event_type(self, cls: type[EventBase]) -> None:
        """Register a custom EventBase subclass for deserialization.

        Adds *cls* to the per-instance registry keyed by its registry name
        (class name, or explicit event_type default).  Logs a warning if the
        key already maps to a different class.
        """
        with self._lock:
            event_type = _registry_key(cls)
            existing = self._registry.get(event_type)
            if existing is not None and existing is not cls:
                logger.warning(
                    "register_event_type: %r overwrites existing %s for event_type %r",
                    cls.__name__,
                    existing.__name__,
                    event_type,
                )
            self._registry[event_type] = cls

    def _max_insertion_order(self) -> int:
        try:
            row = self._conn.execute("SELECT MAX(insertion_order) FROM events").fetchone()
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
            if not _is_corruption_error(e):
                raise
            logger.warning(
                "_max_insertion_order(): events table unreadable, starting counter at 0: %s",
                type(e).__name__,
            )
            return -1
        return row[0] if row[0] is not None else -1

    def _max_position(self) -> int:
        try:
            row = self._conn.execute("SELECT MAX(position) FROM active_tags").fetchone()
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
            if not _is_corruption_error(e):
                raise
            logger.warning(
                "_max_position(): active_tags table unreadable, starting positions at 0: %s",
                type(e).__name__,
            )
            return -1
        return row[0] if row[0] is not None else -1

    def _deserialize(self, data: str) -> EventBase:
        raw = json.loads(data)
        if not isinstance(raw, dict):
            raise TypeError("event data JSON root must be an object")
        event_type = raw.get("event_type", "")
        cls = self._registry.get(event_type)
        if cls is None:
            # Fall back to the global auto-registration registry
            cls = _EVENT_REGISTRY.get(event_type)
        if cls is None:
            logger.warning("Unknown event_type %r, falling back to Metadata", event_type)
            return Metadata.model_validate(raw)
        return cls.model_validate(raw)

    def _try_deserialize(self, data: str, *, context: str) -> EventBase | None:
        try:
            return self._deserialize(data)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError, PydanticValidationError) as e:
            logger.warning("%s: skipping corrupt event data (%s)", context, type(e).__name__)
            return None

    def _serialize(self, event: EventBase) -> str:
        return str(event.model_dump_json())

    # -- EventBackend protocol --

    def store(self, tag: str, event: EventBase) -> None:
        from nooa.runtime.event_backend import _tag_max_num

        with self._lock:
            data = self._serialize(event)
            order = self._insertion_counter
            self._insertion_counter += 1
            try:
                self._do_store(tag, event, data, order)
            except sqlite3.OperationalError as e:
                if "disk I/O error" not in str(e) or self._on_io_error is None:
                    raise
                logger.warning("store(%s): disk I/O error, attempting reconnect + retry", tag)
                self._on_io_error()
                # Retry once after reconnect. If the original write partially
                # committed before the I/O error, the retry hits IntegrityError
                # on the UNIQUE constraint — that means the data IS stored, so
                # we treat it as success.
                try:
                    self._do_store(tag, event, data, order)
                except sqlite3.IntegrityError:
                    logger.info("store(%s): row already persisted before I/O error", tag)
                except sqlite3.DatabaseError as retry_exc:
                    # A double failure (I/O error then corruption on the
                    # reconnected retry) must not escape unhandled either.
                    self._raise_for_db_error(tag, retry_exc)
            except sqlite3.DatabaseError as e:
                self._raise_for_db_error(tag, e)
            self._next_tag_num = max(self._next_tag_num, _tag_max_num(tag) + 1)

    def _raise_for_db_error(self, tag: str, exc: sqlite3.DatabaseError) -> typing.NoReturn:
        """Re-raise a write-path DatabaseError, typed if it indicates corruption.

        Read paths degrade via _is_corruption_error(); the write path must not
        crash a spawned task with an opaque, uncaught DatabaseError. Corruption
        becomes CorruptDatabaseError; anything else propagates unchanged.
        """
        if not _is_corruption_error(exc):
            raise exc
        logger.error("store(%s): database is corrupt — %s", tag, exc)
        raise CorruptDatabaseError(str(exc)) from exc

    def _do_store(self, tag: str, event: EventBase, data: str, order: int) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO events (tag, event_id, event_type, status, data, insertion_order) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (tag, event.id, event.event_type, event.status.value, data, order),
            )
            pos = self._max_position() + 1
            self._conn.execute(
                "INSERT INTO active_tags (position, tag) VALUES (?, ?)",
                (pos, tag),
            )

    def _retry_on_io_error(self, fn: typing.Callable[[], typing.Any]) -> typing.Any:
        """Run *fn*; on disk I/O error, reconnect and retry once."""
        with self._lock:
            try:
                return fn()
            except sqlite3.OperationalError as e:
                if "disk I/O error" not in str(e) or self._on_io_error is None:
                    raise
                logger.warning("disk I/O error, attempting reconnect + retry")
                self._on_io_error()
                return fn()

    def get(self, tag: str) -> EventBase | None:
        with self._lock:
            try:
                row = self._conn.execute("SELECT data FROM events WHERE tag = ?", (tag,)).fetchone()
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                if not _is_corruption_error(e):
                    raise
                logger.warning(
                    "get(%s): corrupt/unreadable row, skipping (%s)", tag, type(e).__name__
                )
                return None
            if row is None:
                return None
            return self._try_deserialize(row[0], context=f"get({tag})")

    def get_by_id(self, event_id: str) -> EventBase | None:
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT data FROM events WHERE event_id = ?", (event_id,)
                ).fetchone()
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                if not _is_corruption_error(e):
                    raise
                logger.warning(
                    "get_by_id(%s): corrupt/unreadable row, skipping (%s)",
                    event_id,
                    type(e).__name__,
                )
                return None
            if row is None:
                return None
            return self._try_deserialize(row[0], context=f"get_by_id({event_id})")

    def update(self, tag: str, **fields: object) -> bool:
        def _do_update() -> bool:
            row = self._conn.execute("SELECT data FROM events WHERE tag = ?", (tag,)).fetchone()
            if row is None:
                return False
            event = self._try_deserialize(row[0], context=f"update({tag})")
            if event is None:
                return False
            for key, value in fields.items():
                if key == "metadata":
                    event.metadata.update(typing.cast(dict, value))
                elif hasattr(event, key):
                    setattr(event, key, value)
            new_data = self._serialize(event)
            with self._conn:
                self._conn.execute(
                    "UPDATE events SET data = ?, status = ? WHERE tag = ?",
                    (new_data, event.status.value, tag),
                )
            return True

        return self._retry_on_io_error(_do_update)

    def remove(self, tag: str) -> bool:
        def _do_remove() -> bool:
            try:
                row = self._conn.execute("SELECT tag FROM events WHERE tag = ?", (tag,)).fetchone()
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                if not _is_corruption_error(e):
                    raise
                logger.warning(
                    "remove(%s): corrupt/unreadable DB, skipping (%s)", tag, type(e).__name__
                )
                return False
            if row is None:
                return False
            with self._conn:
                self._conn.execute("DELETE FROM events WHERE tag = ?", (tag,))
                self._conn.execute("DELETE FROM active_tags WHERE tag = ?", (tag,))
            return True

        return self._retry_on_io_error(_do_remove)

    def set_status(self, tag: str, status: EventStatus) -> bool:
        def _do_set_status() -> bool:
            row = self._conn.execute("SELECT data FROM events WHERE tag = ?", (tag,)).fetchone()
            if row is None:
                return False
            event = self._try_deserialize(row[0], context=f"set_status({tag})")
            if event is None:
                return False
            event.status = status
            new_data = self._serialize(event)
            with self._conn:
                self._conn.execute(
                    "UPDATE events SET data = ?, status = ? WHERE tag = ?",
                    (new_data, status.value, tag),
                )
            return True

        return self._retry_on_io_error(_do_set_status)

    def active_tags(self) -> list[str]:
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT tag FROM active_tags ORDER BY position"
                ).fetchall()
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                if not _is_corruption_error(e):
                    raise
                logger.error("active_tags(): table corrupt, returning empty (%s)", type(e).__name__)
                return []
            return [r[0] for r in rows]

    def insert_active_tag(self, tag: str, index: int) -> None:
        with self._lock:
            with self._conn:
                # Shift existing positions >= index up by 1
                self._conn.execute(
                    "UPDATE active_tags SET position = position + 1 WHERE position >= ?",
                    (index,),
                )
                self._conn.execute(
                    "INSERT INTO active_tags (position, tag) VALUES (?, ?)",
                    (index, tag),
                )

    def remove_active_tag(self, tag: str) -> bool:
        with self._lock:
            with self._conn:
                cursor = self._conn.execute("DELETE FROM active_tags WHERE tag = ?", (tag,))
            return cursor.rowcount > 0

    def all_events(self) -> Iterator[EventBase]:
        # fetchall() under lock, yield outside: avoids holding the lock during
        # potentially slow deserialization while iterating large event sets.
        with self._lock:
            try:
                rows = self._conn.execute(
                    "SELECT data FROM events ORDER BY insertion_order"
                ).fetchall()
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                if not _is_corruption_error(e):
                    raise
                logger.error("all_events(): query failed due to corruption (%s)", type(e).__name__)
                return
        for (data,) in rows:
            if event := self._try_deserialize(data, context="all_events()"):
                yield event

    def find_tag(self, event: EventBase) -> str | None:
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT tag FROM events WHERE event_id = ?", (event.id,)
                ).fetchone()
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                if not _is_corruption_error(e):
                    raise
                logger.warning("find_tag(): corrupt table, returning None (%s)", type(e).__name__)
                return None
            if row is None:
                return None
            return str(row[0])

    def clear(self) -> None:
        with self._lock:
            with self._conn:
                self._conn.execute("DELETE FROM events")
                self._conn.execute("DELETE FROM active_tags")
            self._insertion_counter = 0
            self._next_tag_num = 1

    def allocate_next_tag(self) -> str:
        with self._lock:
            tag = str(self._next_tag_num)
            self._next_tag_num += 1
            return tag

    def max_tag_num(self) -> int:
        # Extract the trailing number from each tag in SQL:
        #   - simple tags ("5")      → CAST("5" AS INTEGER) = 5
        #   - range tags ("2..40")   → substr after ".." → CAST("40" AS INTEGER) = 40
        # COALESCE handles the empty-table case where MAX returns NULL.
        with self._lock:
            try:
                row = self._conn.execute(
                    """
                    SELECT COALESCE(MAX(
                        CAST(
                            CASE WHEN instr(tag, '..') > 0
                                 THEN substr(tag, instr(tag, '..') + 2)
                                 ELSE tag
                            END AS INTEGER
                        )
                    ), 0)
                    FROM events
                    """
                ).fetchone()
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
                if not _is_corruption_error(e):
                    raise
                logger.warning(
                    "max_tag_num(): events table unreadable, starting tag counter at 1: %s",
                    type(e).__name__,
                )
                return 0
            return int(row[0])

    def __len__(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()
            return int(row[0])


class SessionAlreadyActiveError(Exception):
    """Raised when a session database is already open in another process.

    Attributes:
        session_id: The session identifier (db filename stem), if known.
        owner_pid: PID read from the lock file, or None if unavailable/unparseable.
    """

    def __init__(
        self,
        message: str,
        *,
        session_id: str | None = None,
        owner_pid: int | None = None,
    ) -> None:
        super().__init__(message)
        self.session_id = session_id
        self.owner_pid = owner_pid


def _read_lock_pid(lock_path: str) -> int | None:
    """Read the owner PID written by whichever process currently holds the lock.

    The lock file content is just ASCII digits (see ``_acquire_session_lock``).
    Returns None if the file is missing, empty, or not parseable.
    """
    try:
        with open(lock_path, "rb") as f:
            raw = f.read(32).strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _acquire_session_lock(lock_path: str) -> int:
    """Acquire an exclusive flock on *lock_path*, returning the held fd.

    Raises SessionAlreadyActiveError, annotated with the recorded owner PID
    so the caller can tell the user exactly which process is holding the
    session. On Linux/macOS flock is released by the kernel when the owning
    process dies, so a genuine crash doesn't wedge the next run — no
    stale-file cleanup needed here.

    Lock file contents: the ASCII decimal PID of the current owner. We
    truncate before writing so a shorter PID can't leave trailing digits
    from a longer predecessor.
    """
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        owner_pid = _read_lock_pid(lock_path)
        session_id = Path(lock_path).stem
        if owner_pid is not None:
            msg = (
                f"Session {session_id!r} is already active in another process "
                f"(pid {owner_pid}). If that process is gone, remove "
                f"{lock_path!r} to reclaim the session."
            )
        else:
            msg = (
                f"Session {session_id!r} is already active in another process "
                f"(owner pid unavailable — lock file {lock_path!r} is empty or unreadable)."
            )
        raise SessionAlreadyActiveError(msg, session_id=session_id, owner_pid=owner_pid) from None

    # Holder now; replace any stale predecessor PID with ours.
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    os.write(fd, str(os.getpid()).encode())
    return fd


def delete_sqlite_database(db_path: str | Path) -> bool:
    """Delete an inactive SQLite database and its WAL/SHM sidecars.

    The same file lock used by :class:`SQLiteStorageManager` is held for the
    whole deletion. Attempting to delete a live session therefore raises
    :class:`SessionAlreadyActiveError` instead of unlinking an open database.
    The lock file itself is intentionally retained: unlinking a flock target
    creates a race where another process can lock a different inode at the
    same path.

    Returns:
        True when the main database existed, otherwise false.
    """
    path = Path(db_path)
    if str(db_path) == ":memory:":
        raise ValueError("An in-memory SQLite database cannot be deleted")
    if not path.parent.exists():
        return False

    lock_path = str(path.with_suffix(".lock"))
    lock_fd = _acquire_session_lock(lock_path)
    try:
        existed = path.exists()
        for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
            try:
                candidate.unlink()
            except FileNotFoundError:
                pass
        return existed
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


class SQLiteStorageManager:
    """StorageManager backed by a SQLite database.

    Provides persistent event storage and agent snapshots.
    Supports use as a context manager for safe resource cleanup.

    Security: Snapshot restore executes stored Python source code via
    ``exec()``. The database file must be treated as trusted input —
    an attacker who can modify it gains arbitrary code execution on
    restore. Protect the file with appropriate OS-level permissions.

    Args:
        db_path: Path to SQLite database file. Use ":memory:" for in-memory
                 (useful for testing).

    Raises:
        SessionAlreadyActiveError: If ``db_path`` is already open in another
            process.  The caller should start a fresh session instead of
            resuming this one.
    """

    def __init__(self, db_path: str | Path = ":memory:", *, check_same_thread: bool = True) -> None:
        # Safety invariant for check_same_thread=False: callers (the TUI)
        # guarantee that all DB access is serialized through a single
        # asyncio event loop on the agent thread. No concurrent writes.
        self._db_path = str(db_path)
        self._check_same_thread = check_same_thread
        self._lock_fd: int | None = None
        self._closed = False

        if self._db_path != ":memory:":
            lock_path = str(Path(self._db_path).with_suffix(".lock"))
            self._lock_fd = _acquire_session_lock(lock_path)

        self._db_lock = threading.RLock()

        try:
            self._conn = self._open_connection()
            _ensure_schema(self._conn)
            self._backend = SQLiteEventBackend(self._conn, lock=self._db_lock)
            self._backend._on_io_error = self._reconnect
        except Exception:
            self.close()
            raise

    def _open_connection(self) -> sqlite3.Connection:
        """Create and configure a new SQLite connection."""
        conn = sqlite3.connect(self._db_path, check_same_thread=self._check_same_thread)
        # Retry up to 5 s on SQLITE_BUSY before raising, giving concurrent
        # readers time to release shared locks on virtiofs/FUSE mounts.
        conn.execute("PRAGMA busy_timeout=5000")
        if _is_virtiofs(self._db_path):
            # virtiofs (Docker Desktop file sharing) has weak fsync semantics.
            # Avoid WAL entirely: its checkpoint step can lose pages on crash,
            # producing zeroed pages. DELETE journal + FULL sync preserves the
            # original page until the replacement is durably committed.
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.execute("PRAGMA synchronous=FULL")
            logger.info(
                "Detected virtiofs at %s — using journal_mode=DELETE + synchronous=FULL",
                self._db_path,
            )
        else:
            conn.execute("PRAGMA journal_mode=WAL")
            # synchronous=FULL even on normal disks: with the default NORMAL,
            # a disk-full (ENOSPC) during a WAL commit/checkpoint can persist
            # partially-written or zeroed pages, surfacing later as
            # "database disk image is malformed". FULL fsyncs before the commit
            # is acknowledged, so an interrupted write rolls back cleanly.
            conn.execute("PRAGMA synchronous=FULL")
        return conn

    def _reconnect(self) -> None:
        """Close the current connection and open a fresh one.

        Used to recover from sticky SQLITE_IOERR states where the pager
        cache is poisoned but the underlying database file is healthy.
        Must only be called from the same thread that owns the connection.
        """
        logger.warning("SQLiteStorageManager: reconnecting to %s", self._db_path)
        try:
            self._conn.close()
        except Exception:
            pass
        try:
            new_conn = self._open_connection()
        except Exception:
            # Reconnect failed — mark as closed so callers get a clear error
            # rather than "cannot operate on a closed database".
            self._closed = True
            raise
        self._conn = new_conn
        self._backend._conn = self._conn

    @property
    def event_backend(self) -> SQLiteEventBackend:
        return self._backend

    @property
    def path(self) -> Path | None:
        """Filesystem path for this storage, or ``None`` for in-memory storage."""
        return None if self._db_path == ":memory:" else Path(self._db_path)

    def save_snapshot(self, agent: Agent) -> str:
        snapshot = AgentSnapshot.from_agent(agent)
        data = snapshot_to_dict(snapshot)
        snapshot_id = str(uuid.uuid4())
        created_at = datetime.now(UTC).isoformat()
        with self._db_lock:
            with self._conn:
                self._conn.execute(
                    "INSERT INTO snapshots (snapshot_id, created_at, data) VALUES (?, ?, ?)",
                    (snapshot_id, created_at, json.dumps(data)),
                )
        return snapshot_id

    def restore_snapshot(self, snapshot_id: str, agent: Agent) -> None:
        from nooa.errors.storage import SnapshotNotFoundError

        with self._db_lock:
            row = self._conn.execute(
                "SELECT data FROM snapshots WHERE snapshot_id = ?", (snapshot_id,)
            ).fetchone()
            if row is None:
                raise SnapshotNotFoundError(f"Snapshot '{snapshot_id}' not found")
            data = json.loads(row[0])
        snapshot = snapshot_from_dict(data)
        snapshot.restore(agent)

    def get_latest_snapshot_id(self) -> str | None:
        """Return the snapshot_id of the most recently saved snapshot, or None."""
        with self._db_lock:
            row = self._conn.execute(
                "SELECT snapshot_id FROM snapshots ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            return row[0] if row is not None else None

    def restore_latest_snapshot(self, agent: Agent) -> bool:
        """Restore the most recent snapshot into agent.

        Args:
            agent: A freshly constructed Agent instance to restore into.

        Returns:
            True if a snapshot was found and restored, False if no snapshots exist.
        """
        snapshot_id = self.get_latest_snapshot_id()
        if snapshot_id is None:
            return False
        self.restore_snapshot(snapshot_id, agent)
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            lock = getattr(self, "_db_lock", None)
            conn = getattr(self, "_conn", None)
            if conn is not None:
                if lock is not None:
                    with lock:
                        try:
                            conn.commit()
                        finally:
                            conn.close()
                else:
                    try:
                        conn.commit()
                    finally:
                        conn.close()
        finally:
            if self._lock_fd is not None:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                os.close(self._lock_fd)
                self._lock_fd = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
