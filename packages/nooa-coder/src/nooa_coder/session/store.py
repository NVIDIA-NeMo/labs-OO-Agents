# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SQLite-backed coding-agent sessions discoverable by interactive hosts."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import socket
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock

from nooa.runtime.event_manager import EventManager
from nooa.storage import read_lock_owner
from nooa.storage.sqlite import (
    SessionAlreadyActiveError,
    SQLiteStorageManager,
    delete_sqlite_database,
)
from nooa_coder.session.events import (
    SESSION_EVENT_TYPES,
    SessionModeChanged,
    SessionModelChanged,
    SessionStarted,
    SessionTitleUpdated,
)
from nooa_coder.session.items import USAGE_FIELDS, SessionInfo, TranscriptEntry, Usage

logger = logging.getLogger(__name__)

_START_EVENT_TYPES = frozenset(("SessionStarted", "TUISessionStart"))
_TITLE_EVENT_TYPES = frozenset(("SessionTitleUpdated", "TUISessionRename"))
_SETTING_EVENT_TYPES = ("SessionModeChanged", "SessionModelChanged")
# SessionUserMessage: written by earlier versions of this package, still read.
_USER_EVENT_TYPES = frozenset(("SessionUserMessage", "TUIUserInput"))
_AGENT_EVENT_TYPES = frozenset(("AgentMessage", "TUIAgentMessage"))
_TURN_EVENT_TYPES = _USER_EVENT_TYPES | _AGENT_EVENT_TYPES
_TRANSCRIPT_EVENT_TYPES = _TURN_EVENT_TYPES | frozenset(
    ("ItemAdmitted", "TurnEnded", "TurnCancelled", "SnapshotRestoreFailed")
)


def sessions_root(workspace: Path, override: Path | None = None) -> Path:
    """The directory a workspace's sessions live in.

    ``override`` if given, else ``$NOOA_SESSIONS_DIR`` if set (one shared
    directory for all workspaces), else ``<workspace>/.nooa/sessions``, where
    the ``nooa-acp`` server and the TUI keep them.
    """
    if override is not None:
        return Path(override)
    shared = os.environ.get("NOOA_SESSIONS_DIR")
    if shared:
        return Path(shared).expanduser()
    return Path(workspace) / ".nooa" / "sessions"


def _normalise_workspace(workspace: str | Path) -> str:
    """Compare workspaces as absolute, resolved paths; empty stays empty."""
    if not str(workspace):
        return ""
    return str(Path(workspace).expanduser().resolve())


def _lock_is_held(lock_path: Path) -> bool:
    """Whether another process holds the kernel lock on ``lock_path``, without writing to it.

    Acquiring the lock the normal way records the acquirer's own pid and host
    in the file; a probe must not, or the other side of a shared mount would
    read the prober as the owner.
    """
    try:
        fd = os.open(lock_path, os.O_RDONLY)
    except OSError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def _pid_alive(pid: int) -> bool:
    """Whether ``pid`` is a running process on this host (permission errors count as alive)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _workspace_matches(recorded: str | Path, wanted: str) -> bool:
    """Whether a session's recorded workspace is ``wanted`` (a normalised path).

    The old TUI recorded the directory as typed, so ``../`` or ``.`` is
    common; a relative record names no directory and cannot be compared, and
    hiding the session would be worse than listing it.
    """
    text = str(recorded)
    if text and not Path(text).expanduser().is_absolute():
        return True
    return _normalise_workspace(text) == wanted


def _connect_read_only(path: Path) -> sqlite3.Connection:
    """Open a session file for reading only.

    Read-only mode never creates the file, so a reader racing a delete
    fails instead of leaving an empty database behind.
    """
    # immutable: no locks and no shared-memory side file, so a reader on
    # another machine sharing the directory cannot disturb the writer.
    uri = f"{path.resolve().as_uri()}?mode=ro&immutable=1"
    return sqlite3.connect(uri, uri=True)


def _item_text(item_json: str) -> str:
    """Text of an admitted item: the string itself, else its JSON."""
    try:
        value = json.loads(item_json)
    except (TypeError, json.JSONDecodeError):
        return item_json
    return value if isinstance(value, str) else item_json


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _int(value: object, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    return default


def _utc_offset(start: dict[str, object]) -> float | None:
    """The writer's UTC offset recorded in a session's start event, if any."""
    value = start.get("utc_offset")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


_USAGE_SUMS = ", ".join(
    f"COALESCE(SUM(json_extract(data, '$.usage.{name}')), 0)" for name in USAGE_FIELDS
)


def _stored_usage(connection: sqlite3.Connection) -> Usage:
    """Usage over the whole session, from its records.

    Own totals are the sum of each ``TurnEnded``'s usage; attributed totals
    the sum of the ``UsageAttributed`` records its children's turns left.
    """
    own = connection.execute(
        f"SELECT {_USAGE_SUMS} FROM events WHERE event_type = 'TurnEnded'"
    ).fetchone()
    attributed = connection.execute(
        f"SELECT {_USAGE_SUMS} FROM events WHERE event_type = 'UsageAttributed'"
    ).fetchone()
    values: dict[str, float] = {}
    for index, name in enumerate(USAGE_FIELDS):
        cast = float if name == "cost_usd" else int
        values[name] = cast(own[index] or 0)
        values[f"attributed_{name}"] = cast(attributed[index] or 0)
    return Usage.model_validate(values)


class InvalidSessionIdError(ValueError):
    """Raised before an unsafe or empty session ID can become a file path."""


class SessionNotFoundError(FileNotFoundError):
    """Raised when a durable session does not exist or lacks start metadata."""


def _open_session_storage(path: Path, *, must_exist: bool) -> SQLiteStorageManager:
    """Storage for one session file: rollback journal, and no accidental creation.

    ``must_exist`` makes a delete racing an open fail instead of leaving a new
    empty file behind. Session files use the rollback journal everywhere, not
    only where virtiofs is detected: the directory may be shared with another
    machine (a sandbox and its host), and WAL keeps cross-process state in a
    shared-memory file that two kernels do not agree on. A file left in WAL
    mode by an older build is converted on open. ``check_same_thread=False``
    because the Session writes its checkpoint from a worker thread through
    ``save_snapshot_json``, which takes the manager's lock.
    """
    return SQLiteStorageManager(
        path, must_exist=must_exist, journal_mode="delete", check_same_thread=False
    )


class SessionHandle:
    """An agent-runtime-owned session database and metadata writer.

    Exactly one live agent runtime owns this handle. Interactive hosts attach
    to that runtime through their transport; they do not open the database for
    writing alongside it.
    """

    def __init__(
        self,
        store: SessionStore,
        storage: SQLiteStorageManager,
        info: SessionInfo,
    ) -> None:
        self._store = store
        self._storage = storage
        self._events = EventManager(backend=storage.event_backend)
        for event_type in SESSION_EVENT_TYPES:
            self._events.register_event_type(event_type)
        self._info = info
        self._metadata_lock = RLock()
        self._closed = False

    @property
    def store(self) -> SessionStore:
        return self._store

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has run; a closed handle records nothing more."""
        return self._closed

    @property
    def id(self) -> str:
        return self._info.id

    @property
    def info(self) -> SessionInfo:
        with self._metadata_lock:
            return self._info.model_copy(deep=True)

    @property
    def storage(self) -> SQLiteStorageManager:
        """Storage to pass to the agent constructed by the owning runtime."""
        return self._storage

    @property
    def events(self) -> EventManager:
        """Event manager for runtime-owned session metadata."""
        return self._events

    @property
    def path(self) -> Path:
        return self._store.path_for(self.id)

    def set_title(self, title: str, *, user_set: bool = False) -> None:
        """Persist a title and update this handle's current metadata."""
        self._ensure_open()
        event = SessionTitleUpdated(title=title, user_set=user_set)
        self._events.add(event)
        with self._metadata_lock:
            self._info = self._info.model_copy(
                update={
                    "title": title,
                    "title_is_user_set": self._info.title_is_user_set or user_set,
                    "last_active": event.timestamp.timestamp(),
                }
            )

    def set_mode(self, mode: str) -> None:
        """Persist a permission mode and update this handle's current metadata."""
        self._ensure_open()
        self._events.add(SessionModeChanged(mode=mode))
        with self._metadata_lock:
            self._info = self._info.model_copy(update={"mode": mode})

    def set_model(self, model: str) -> None:
        """Persist a model alias and update this handle's current metadata."""
        self._ensure_open()
        self._events.add(SessionModelChanged(model=model))
        with self._metadata_lock:
            self._info = self._info.model_copy(update={"model": model})

    def update_usage(self, usage: Usage) -> None:
        """Set the session's usage totals (a copy) as ``info`` reports them."""
        with self._metadata_lock:
            self._info = self._info.model_copy(update={"usage": usage.model_copy()})

    def transcript(self) -> list[TranscriptEntry]:
        """The session's transcript (see :meth:`SessionStore.load_transcript`)."""
        return self._store.load_transcript(self.id)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._storage.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"Session {self.id!r} is closed")

    def __enter__(self) -> SessionHandle:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class SessionStore:
    """Repository and factory for durable sessions.

    ``root`` is one directory of session files, normally
    ``sessions_root(workspace)``: the workspace's ``.nooa/sessions``, or one
    directory shared by all workspaces when ``NOOA_SESSIONS_DIR`` or an
    explicit override names it. Each session records the workspace it was
    created for, and :meth:`list` can filter on it (a shared directory
    needs that).

    A daemon may use read-only operations such as :meth:`list` and :meth:`get`
    for discovery. Only the process running the agent opens a
    :class:`SessionHandle` for writes.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @property
    def workspace(self) -> Path | None:
        """The workspace of a ``<workspace>/.nooa/sessions`` store; ``None`` for a shared directory."""
        if self.root.name == "sessions" and self.root.parent.name == ".nooa":
            return self.root.parent.parent
        return None

    def path_for(self, session_id: str) -> Path:
        session_id = self.validate_id(session_id)
        return self.root / f"{session_id}.db"

    def create(
        self,
        *,
        model: str = "",
        agent: str = "",
        workspace: str = "",
        host: str = "",
        parent_id: str | None = None,
        depth: int = 0,
        name: str | None = None,
        retained: bool = False,
        turn_method: str = "handle",
        mode: str = "auto",
        session_id: str | None = None,
    ) -> SessionHandle:
        session_id = self.validate_id(session_id or str(uuid.uuid4()))
        # Recorded as a resolved absolute path: a relative one ("../", as the
        # old TUI wrote) names no directory once the process has moved.
        workspace = _normalise_workspace(workspace)
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(session_id)
        if path.exists():
            raise FileExistsError(f"Session {session_id!r} already exists")

        offset = datetime.now().astimezone().utcoffset()
        storage = _open_session_storage(path, must_exist=False)
        events = EventManager(backend=storage.event_backend)
        for event_type in SESSION_EVENT_TYPES:
            events.register_event_type(event_type)
        started = SessionStarted(
            host=host,
            model=model,
            agent=agent,
            workspace=workspace,
            parent_id=parent_id,
            depth=depth,
            name=name,
            retained=retained,
            turn_method=turn_method,
            mode=mode,
            utc_offset=offset.total_seconds() if offset is not None else None,
        )
        try:
            events.add(started)
        except BaseException:
            storage.close()
            raise
        timestamp = started.timestamp.timestamp()
        return SessionHandle(
            self,
            storage,
            SessionInfo(
                id=session_id,
                model=model,
                agent=agent,
                created_at=timestamp,
                last_active=timestamp,
                workspace=workspace,
                host=host,
                parent_id=parent_id,
                depth=depth,
                name=name,
                retained=retained,
                turn_method=turn_method,
                mode=mode,
            ),
        )

    def open(self, session_id: str) -> SessionHandle:
        path = self.path_for(session_id)
        info = self._read_info(path)
        if info is None:
            raise SessionNotFoundError(f"Session {session_id!r} was not found or is invalid")
        owner = self.claim_owner(session_id)
        if owner is not None:
            raise SessionAlreadyActiveError(
                f"Session {session_id!r} is active in another process (pid {owner}, "
                "TUI claim). Close it there first.",
                session_id=session_id,
                owner_pid=owner,
            )
        foreign = self.foreign_owner(session_id)
        if foreign is not None:
            pid, host = foreign
            lock_path = self.path_for(session_id).with_suffix(".lock")
            raise SessionAlreadyActiveError(
                f"Session {session_id!r} is in use on {host} (pid {pid}). Close it there "
                f"first. If that process is gone, empty the file {str(lock_path)!r} to reclaim it.",
                session_id=session_id,
                owner_pid=pid,
            )
        try:
            storage = _open_session_storage(path, must_exist=True)
        except sqlite3.OperationalError as exc:
            if path.exists():
                raise
            raise SessionNotFoundError(f"Session {session_id!r} was not found") from exc
        return SessionHandle(self, storage, info)

    def get(self, session_id: str) -> SessionInfo:
        path = self.path_for(session_id)
        info = self._read_info(path)
        if info is None:
            raise SessionNotFoundError(f"Session {session_id!r} was not found or is invalid")
        return info

    def list(
        self,
        *,
        limit: int | None = None,
        workspace: str | Path | None = None,
        roots_only: bool = True,
    ) -> list[SessionInfo]:
        """Sessions on disk, most recently active first.

        Only root sessions by default; ``roots_only=False`` includes
        children. ``workspace`` keeps only sessions recorded for that
        directory.
        """
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        if limit == 0 or not self.root.exists():
            return []
        wanted = _normalise_workspace(workspace) if workspace is not None else None
        sessions = [
            info
            for path in self.root.glob("*.db")
            if not path.stem.endswith("-memory")
            if (info := self._read_info(path)) is not None
            if not roots_only or info.parent_id is None
            if wanted is None or _workspace_matches(info.workspace, wanted)
        ]
        sessions.sort(key=lambda info: info.last_active, reverse=True)
        return sessions if limit is None else sessions[:limit]

    def snapshot_ids(self, session_id: str) -> list[str]:
        """Snapshot ids newest first, reading past a damaged table tail.

        The ordered query needs the whole index; when the file is damaged it
        fails, and the rows are then probed one by one by rowid, so the
        snapshots that are still readable are found.
        """
        path = self.path_for(session_id)
        try:
            connection = _connect_read_only(path)
        except sqlite3.Error:
            return []
        try:
            try:
                rows = connection.execute(
                    "SELECT snapshot_id FROM snapshots ORDER BY created_at DESC"
                ).fetchall()
                return [str(row[0]) for row in rows]
            except sqlite3.DatabaseError:
                pass
            found: list[tuple[str, str]] = []
            misses = 0
            rowid = 0
            while misses < 50:
                rowid += 1
                try:
                    row = connection.execute(
                        "SELECT snapshot_id, created_at FROM snapshots WHERE rowid = ?", (rowid,)
                    ).fetchone()
                except sqlite3.DatabaseError:
                    misses += 1
                    continue
                if row is None:
                    misses += 1
                    continue
                misses = 0
                found.append((str(row[1]), str(row[0])))
            found.sort(reverse=True)
            return [snapshot_id for _created, snapshot_id in found]
        finally:
            connection.close()

    def load_rows(
        self, session_id: str, event_types: frozenset[str] | None = None
    ) -> list[tuple[str, dict[str, object]]]:
        """Raw ``(event_type, data)`` rows of a session, oldest first, read-only."""
        return self._read_rows(self.path_for(session_id), event_types=event_types)

    def load_transcript(self, session_id: str) -> list[TranscriptEntry]:
        """What a person would see: their messages, replies, questions and cancels.

        User messages are the items admitted on ``user_messages`` (and
        steers), each shown once even when a steer was admitted again as a
        message. Questions come from turns that ended with ``NeedInput``.
        A ``note`` says a load could not restore the saved agent state.
        """
        rows = self.load_rows(session_id, _TRANSCRIPT_EVENT_TYPES | _START_EVENT_TYPES)
        start = next((raw for event_type, raw in rows if event_type in _START_EVENT_TYPES), {})
        offset = _utc_offset(start)
        entries: list[TranscriptEntry] = []
        seen_items: set[str] = set()
        for event_type, raw in rows:
            timestamp = self._timestamp(raw, fallback=0.0, utc_offset=offset)
            if event_type == "ItemAdmitted":
                item_id = str(raw.get("item_id", ""))
                if raw.get("channel") not in ("user_messages", "steer") or item_id in seen_items:
                    continue
                seen_items.add(item_id)
                entries.append(
                    TranscriptEntry(
                        role="user",
                        content=_item_text(str(raw.get("item_json", ""))),
                        item_id=item_id,
                        timestamp=timestamp,
                    )
                )
            elif event_type in _USER_EVENT_TYPES:
                content = raw.get("content", raw.get("text", ""))
                entries.append(
                    TranscriptEntry(role="user", content=str(content), timestamp=timestamp)
                )
            elif event_type in _AGENT_EVENT_TYPES:
                entries.append(
                    TranscriptEntry(
                        role="agent", content=str(raw.get("content", "")), timestamp=timestamp
                    )
                )
            elif event_type == "TurnEnded" and raw.get("outcome_kind") == "need_input":
                entries.append(
                    TranscriptEntry(
                        role="question",
                        content=str(raw.get("explanation", "")),
                        timestamp=timestamp,
                    )
                )
            elif event_type == "TurnCancelled":
                entries.append(
                    TranscriptEntry(
                        role="cancelled",
                        content=f"Stopped by {raw.get('by', '')}",
                        timestamp=timestamp,
                    )
                )
            elif event_type == "SnapshotRestoreFailed":
                entries.append(
                    TranscriptEntry(
                        role="note",
                        content=(
                            f"The saved agent state could not be restored "
                            f"({raw.get('error', '')}); the session went on from an empty state."
                        ),
                        timestamp=timestamp,
                    )
                )
        return entries

    def is_active(self, session_id: str) -> bool:
        """Whether another owner holds the session: the file lock, or a live claim.

        The TUI marks a session it runs with a ``<id>.active/owner-*.json``
        claim naming its process instead of the file lock; a claim whose
        process is alive on this host counts as active.
        """
        if self.claim_owner(session_id) is not None:
            return True
        lock_path = self.path_for(session_id).with_suffix(".lock")
        if not lock_path.exists():
            return False
        if self.foreign_owner(session_id) is not None:
            return True
        return _lock_is_held(lock_path)

    def foreign_owner(self, session_id: str) -> tuple[int, str] | None:
        """``(pid, host)`` of a process on another machine holding the session, else None.

        The kernel lock is not visible across a shared mount; the owner's
        record in the lock file is. A blank file means a clean close. A
        record naming this machine is left to the kernel lock to judge, so
        a crash here never blocks a resume here.
        """
        lock_path = self.path_for(session_id).with_suffix(".lock")
        pid, host = read_lock_owner(str(lock_path))
        if pid is None or host is None or host == socket.gethostname():
            return None
        return pid, host

    def claim_owner(self, session_id: str) -> int | None:
        """The pid of a live process claiming the session through ``<id>.active/``, else None."""
        claim_dir = self.path_for(session_id).with_suffix(".active")
        try:
            owners = list(claim_dir.glob("owner-*.json"))
        except OSError:
            return None
        for owner_path in owners:
            try:
                pid = json.loads(owner_path.read_text()).get("pid")
            except (OSError, ValueError, AttributeError):
                continue
            if isinstance(pid, int) and pid > 0 and _pid_alive(pid):
                return pid
        return None

    def delete(self, session_id: str) -> bool:
        """Delete an inactive session database and its SQLite sidecars."""
        return delete_sqlite_database(self.path_for(session_id))

    def _read_info(self, path: Path) -> SessionInfo | None:
        try:
            fallback = path.stat().st_mtime
        except OSError:
            return None

        try:
            connection = _connect_read_only(path)
            try:
                start_row = connection.execute(
                    "SELECT event_type, data FROM events "
                    "WHERE event_type IN (?, ?) ORDER BY insertion_order LIMIT 1",
                    tuple(_START_EVENT_TYPES),
                ).fetchone()
                if start_row is None:
                    return None

                title_rows = connection.execute(
                    "SELECT data FROM events WHERE event_type IN (?, ?) ORDER BY insertion_order",
                    tuple(_TITLE_EVENT_TYPES),
                ).fetchall()
                turn_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM events WHERE event_type IN (?, ?) "
                        "OR (event_type = 'ItemAdmitted' "
                        "AND json_extract(data, '$.channel') = 'user_messages')",
                        tuple(_USER_EVENT_TYPES),
                    ).fetchone()[0]
                )
                reply_count = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM events WHERE event_type IN (?, ?, 'TurnEnded')",
                        tuple(_AGENT_EVENT_TYPES),
                    ).fetchone()[0]
                )
                last_row = connection.execute(
                    "SELECT data FROM events ORDER BY insertion_order DESC LIMIT 1"
                ).fetchone()
                setting_rows = connection.execute(
                    "SELECT event_type, data FROM events WHERE event_type IN (?, ?) "
                    "ORDER BY insertion_order",
                    _SETTING_EVENT_TYPES,
                ).fetchall()
                usage = _stored_usage(connection)
            finally:
                connection.close()
        except (OSError, sqlite3.Error):
            logger.debug("Could not summarize session database %s", path, exc_info=True)
            return None

        start = self._decode_data(start_row[1], path)
        if start is None:
            return None
        start_event_type = str(start_row[0])
        offset = _utc_offset(start)
        started_at = self._timestamp(start, fallback=fallback, utc_offset=offset)
        last_active = fallback
        if last_row is not None:
            last = self._decode_data(last_row[0], path)
            if last is not None:
                last_active = max(
                    last_active, self._timestamp(last, fallback=fallback, utc_offset=offset)
                )

        mode = str(start.get("mode") or "auto")
        model = str(start.get("model", ""))
        for event_type, data in setting_rows:
            raw = self._decode_data(data, path)
            if raw is None:
                continue
            if event_type == "SessionModeChanged" and raw.get("mode"):
                mode = str(raw["mode"])
            elif event_type == "SessionModelChanged" and raw.get("model"):
                model = str(raw["model"])

        title: str | None = None
        title_is_user_set = False
        for (data,) in title_rows:
            raw = self._decode_data(data, path)
            if raw is None:
                continue
            title = str(raw.get("title", raw.get("name", ""))) or None
            title_is_user_set = title_is_user_set or bool(
                raw.get("user_set", raw.get("user_named", False))
            )

        return SessionInfo(
            id=path.stem,
            model=model,
            mode=mode,
            agent=str(start.get("agent", start.get("agent_cls", ""))),
            created_at=started_at,
            last_active=last_active,
            turn_count=turn_count,
            reply_count=reply_count,
            usage=usage,
            workspace=str(
                start.get("workspace", start.get("working_directory", start.get("working_dir", "")))
            ),
            parent_id=_optional_str(start.get("parent_id")),
            depth=_int(start.get("depth"), default=0),
            name=_optional_str(start.get("name")),
            retained=bool(start.get("retained", False)),
            turn_method=str(start.get("turn_method") or "handle"),
            title=title,
            title_is_user_set=title_is_user_set,
            host=str(
                start.get(
                    "host",
                    start.get(
                        "origin",
                        "tui" if start_event_type == "TUISessionStart" else "",
                    ),
                )
            ),
        )

    @staticmethod
    def _decode_data(data: object, path: Path) -> dict[str, object] | None:
        if not isinstance(data, (str, bytes, bytearray)):
            logger.debug("Skipping invalid session event payload in %s", path)
            return None
        try:
            raw = json.loads(data)
        except (TypeError, json.JSONDecodeError):
            logger.debug("Skipping corrupt session event in %s", path, exc_info=True)
            return None
        return raw if isinstance(raw, dict) else None

    @staticmethod
    def _read_rows(
        path: Path,
        *,
        event_types: frozenset[str] | None = None,
    ) -> list[tuple[str, dict[str, object]]]:
        if not path.exists():
            return []
        try:
            connection = _connect_read_only(path)
            try:
                if event_types:
                    placeholders = ", ".join("?" for _ in event_types)
                    query = (
                        "SELECT event_type, data FROM events "
                        f"WHERE event_type IN ({placeholders}) ORDER BY insertion_order"
                    )
                    db_rows = connection.execute(query, tuple(event_types)).fetchall()
                else:
                    db_rows = connection.execute(
                        "SELECT event_type, data FROM events ORDER BY insertion_order"
                    ).fetchall()
            finally:
                connection.close()
        except (OSError, sqlite3.Error):
            logger.debug("Could not read session database %s", path, exc_info=True)
            return []

        rows: list[tuple[str, dict[str, object]]] = []
        for event_type, data in db_rows:
            raw = SessionStore._decode_data(data, path)
            if raw is not None:
                rows.append((str(event_type), raw))
        return rows

    @staticmethod
    def _timestamp(
        raw: dict[str, object], *, fallback: float, utc_offset: float | None = None
    ) -> float:
        """Epoch seconds of an event's timestamp.

        A naive timestamp is the writer's local time: it is read with the
        writer's recorded ``utc_offset``, so the result does not depend on
        the reader's time zone. Files written before the offset was
        recorded fall back to the reader's local time.
        """
        value = raw.get("timestamp")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError:
                return fallback
            if parsed.tzinfo is None and utc_offset is not None:
                parsed = parsed.replace(tzinfo=timezone(timedelta(seconds=utc_offset)))
            return parsed.timestamp()
        return fallback

    @staticmethod
    def validate_id(session_id: str) -> str:
        """``session_id`` if it can name a session file; else ``InvalidSessionIdError``."""
        if (
            not session_id
            or session_id in {".", ".."}
            or any(separator in session_id for separator in ("/", "\\", "\x00"))
        ):
            raise InvalidSessionIdError(f"Invalid session ID {session_id!r}")
        return session_id
