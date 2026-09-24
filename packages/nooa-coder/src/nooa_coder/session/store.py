# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SQLite-backed coding-agent sessions discoverable by interactive hosts."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Literal

from nooa.paths import get_user_dir
from nooa.runtime.event_manager import EventManager
from nooa.storage.sqlite import SQLiteStorageManager, _is_virtiofs, delete_sqlite_database
from nooa_coder.session.events import (
    SESSION_EVENT_TYPES,
    SessionStarted,
    SessionTitleUpdated,
    SessionUserMessage,
)
from nooa_coder.session.items import SessionInfo, TranscriptEntry

logger = logging.getLogger(__name__)

_START_EVENT_TYPES = frozenset(("SessionStarted", "TUISessionStart"))
_TITLE_EVENT_TYPES = frozenset(("SessionTitleUpdated", "TUISessionRename"))
_USER_EVENT_TYPES = frozenset(("SessionUserMessage", "TUIUserInput"))
_AGENT_EVENT_TYPES = frozenset(("AgentMessage", "TUIAgentMessage"))
_TURN_EVENT_TYPES = _USER_EVENT_TYPES | _AGENT_EVENT_TYPES
_TRANSCRIPT_EVENT_TYPES = _TURN_EVENT_TYPES | frozenset(
    ("ItemAdmitted", "TurnEnded", "TurnCancelled")
)


def _normalise_workspace(workspace: str | Path) -> str:
    """Compare workspaces as absolute, resolved paths; empty stays empty."""
    if not str(workspace):
        return ""
    return str(Path(workspace).expanduser().resolve())


def _connect_read_only(path: Path) -> sqlite3.Connection:
    """Open a session file for reading only.

    Read-only mode never creates the file, so a reader racing a delete
    fails instead of leaving an empty database behind.
    """
    uri = f"{path.resolve().as_uri()}?mode=ro"
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


class InvalidSessionIdError(ValueError):
    """Raised before an unsafe or empty session ID can become a file path."""


class SessionNotFoundError(FileNotFoundError):
    """Raised when a durable session does not exist or lacks start metadata."""


@dataclass(frozen=True, slots=True)
class SessionTurn:
    role: Literal["user", "agent"]
    content: str
    timestamp: float = field(default_factory=time.time)


class _ExistingSQLiteStorageManager(SQLiteStorageManager):
    """Storage for a session file that must already exist.

    ``SQLiteStorageManager`` opens with a plain ``sqlite3.connect(path)``,
    which creates an empty database when the file is gone, so a delete
    racing an open would leave a new empty file behind. Opening with the
    ``mode=rw`` URI fails instead. Replace this subclass with the
    ``must_exist`` option once #382 lands. The pragmas match the base
    class's ``_open_connection``.
    """

    def _open_connection(self) -> sqlite3.Connection:
        uri = f"{Path(self._db_path).resolve().as_uri()}?mode=rw"
        connection = sqlite3.connect(uri, uri=True, check_same_thread=self._check_same_thread)
        connection.execute("PRAGMA busy_timeout=5000")
        if _is_virtiofs(self._db_path):
            connection.execute("PRAGMA journal_mode=DELETE")
        else:
            connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection


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
    def id(self) -> str:
        return self._info.id

    @property
    def info(self) -> SessionInfo:
        with self._metadata_lock:
            return self._info.model_copy()

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

    def record_user_message(self, content: str) -> tuple[str, SessionUserMessage]:
        """Persist raw user text before dispatching its agent turn."""
        self._ensure_open()
        event = SessionUserMessage(content=content)
        tag = self._events.add(event)
        with self._metadata_lock:
            self._info = self._info.model_copy(
                update={
                    "turn_count": self._info.turn_count + 1,
                    "last_active": event.timestamp.timestamp(),
                }
            )
        return tag, event

    def turns(self) -> list[SessionTurn]:
        return self._store.load_turns(self.id)

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

    Sessions live in one user-level directory (``nooa.paths.get_user_dir(
    "sessions")`` by default), not per workspace; each records the
    workspace it was created for, and :meth:`list` can filter on it.

    A daemon may use read-only operations such as :meth:`list` and :meth:`get`
    for discovery. Only the process running the agent opens a
    :class:`SessionHandle` for writes.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else get_user_dir("sessions")

    def path_for(self, session_id: str) -> Path:
        session_id = self._validate_id(session_id)
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
        session_id: str | None = None,
        check_same_thread: bool = True,
    ) -> SessionHandle:
        session_id = self._validate_id(session_id or str(uuid.uuid4()))
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.path_for(session_id)
        if path.exists():
            raise FileExistsError(f"Session {session_id!r} already exists")

        storage = SQLiteStorageManager(path, check_same_thread=check_same_thread)
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
            ),
        )

    def open(self, session_id: str, *, check_same_thread: bool = True) -> SessionHandle:
        path = self.path_for(session_id)
        info = self._read_info(path)
        if info is None:
            raise SessionNotFoundError(f"Session {session_id!r} was not found or is invalid")
        try:
            storage = _ExistingSQLiteStorageManager(path, check_same_thread=check_same_thread)
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
            if wanted is None or _normalise_workspace(info.workspace) == wanted
        ]
        sessions.sort(key=lambda info: info.last_active, reverse=True)
        return sessions if limit is None else sessions[:limit]

    def load_turns(self, session_id: str) -> list[SessionTurn]:
        path = self.path_for(session_id)
        rows = self._read_rows(path, event_types=_TURN_EVENT_TYPES)
        turns: list[SessionTurn] = []
        for event_type, raw in rows:
            try:
                if event_type in _USER_EVENT_TYPES:
                    content = raw.get("content", raw.get("text", ""))
                    role: Literal["user", "agent"] = "user"
                else:
                    content = raw.get("content", "")
                    role = "agent"
                turns.append(
                    SessionTurn(
                        role=role,
                        content=str(content),
                        timestamp=self._timestamp(raw, fallback=time.time()),
                    )
                )
            except Exception:
                logger.debug("Skipping invalid turn in %s", path, exc_info=True)
        return turns

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
        """
        rows = self.load_rows(session_id, _TRANSCRIPT_EVENT_TYPES)
        entries: list[TranscriptEntry] = []
        seen_items: set[str] = set()
        for event_type, raw in rows:
            timestamp = self._timestamp(raw, fallback=0.0)
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
        return entries

    def find_by_prefix(self, prefix: str) -> list[str]:
        if not prefix or any(separator in prefix for separator in ("/", "\\", "\x00")):
            return []
        if not self.root.exists():
            return []
        matches = [
            path
            for path in self.root.glob("*.db")
            if path.stem.startswith(prefix) and not path.stem.endswith("-memory")
        ]
        matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return [path.stem for path in matches]

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
                last_row = connection.execute(
                    "SELECT data FROM events ORDER BY insertion_order DESC LIMIT 1"
                ).fetchone()
            finally:
                connection.close()
        except (OSError, sqlite3.Error):
            logger.debug("Could not summarize session database %s", path, exc_info=True)
            return None

        start = self._decode_data(start_row[1], path)
        if start is None:
            return None
        start_event_type = str(start_row[0])
        started_at = self._timestamp(start, fallback=fallback)
        last_active = fallback
        if last_row is not None:
            last = self._decode_data(last_row[0], path)
            if last is not None:
                last_active = max(last_active, self._timestamp(last, fallback=fallback))

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
            model=str(start.get("model", "")),
            agent=str(start.get("agent", start.get("agent_cls", ""))),
            created_at=started_at,
            last_active=last_active,
            turn_count=turn_count,
            workspace=str(
                start.get("workspace", start.get("working_directory", start.get("working_dir", "")))
            ),
            parent_id=_optional_str(start.get("parent_id")),
            depth=_int(start.get("depth"), default=0),
            name=_optional_str(start.get("name")),
            retained=bool(start.get("retained", False)),
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
    def _timestamp(raw: dict[str, object], *, fallback: float) -> float:
        value = raw.get("timestamp")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value).timestamp()
            except ValueError:
                pass
        return fallback

    @staticmethod
    def _validate_id(session_id: str) -> str:
        if (
            not session_id
            or session_id in {".", ".."}
            or any(separator in session_id for separator in ("/", "\\", "\x00"))
        ):
            raise InvalidSessionIdError(f"Invalid session ID {session_id!r}")
        return session_id
