# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TUI presentation adapter for the shared durable session store."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from nooa.paths import get_project_dir
from nooa.sessions import SessionHandle, SessionInfo, SessionStore
from nooa.storage.sqlite import delete_sqlite_database

SESSIONS_DIR = get_project_dir("sessions")


def _make_trace_session_name(session_id: str) -> str:
    """Build a trace name correlated with the durable session UUID."""
    return f"tui-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{session_id[:8]}"


@dataclass(frozen=True, slots=True)
class SessionMeta:
    """Compatibility view used by the TUI's session tables."""

    id: str
    model: str
    agent: str
    started_at: float
    last_active: float
    turn_count: int = 0
    working_dir: str = ""
    name: str | None = None
    user_named: bool = False

    @classmethod
    def from_info(cls, info: SessionInfo) -> SessionMeta:
        return cls(
            id=info.id,
            model=info.model,
            agent=info.agent,
            started_at=info.started_at,
            last_active=info.last_active,
            turn_count=info.turn_count,
            working_dir=info.working_directory,
            name=info.title,
            user_named=info.title_is_user_set,
        )


@dataclass(frozen=True, slots=True)
class Turn:
    role: Literal["user", "agent"]
    content: str
    ts: float


class SessionManager:
    """TUI-facing view of one :class:`nooa.sessions.SessionHandle`.

    The adapter deliberately contains no persistence implementation. Both the
    native TUI and protocol hosts read and write the same session event schema.
    """

    def __init__(self, handle: SessionHandle) -> None:
        self._handle = handle
        self._storage = handle.storage
        self.agent_cls = handle.info.agent

    @classmethod
    def create(
        cls,
        *,
        model: str = "",
        agent_cls: str = "CodingAgent",
        working_dir: str = "",
        session_id: str | None = None,
    ) -> SessionManager:
        handle = SessionStore(SESSIONS_DIR).create(
            model=model,
            agent=agent_cls,
            working_directory=working_dir,
            host="tui",
            session_id=session_id,
            check_same_thread=False,
        )
        return cls(handle)

    @classmethod
    def open(cls, session_id: str) -> SessionManager:
        return cls(SessionStore(SESSIONS_DIR).open(session_id, check_same_thread=False))

    @property
    def session_id(self) -> str:
        return self._handle.id

    @property
    def model(self) -> str:
        return self._handle.info.model

    @property
    def working_dir(self) -> str:
        return self._handle.info.working_directory

    @property
    def agent_db_path(self) -> Path:
        return self._handle.path

    @property
    def name(self) -> str | None:
        return self._handle.info.title

    @property
    def user_named(self) -> bool:
        return self._handle.info.title_is_user_set

    @property
    def turns(self) -> list[Turn]:
        return self.load_turns(self.session_id)

    def rename(self, name: str, user_named: bool = False) -> None:
        self._handle.set_title(name, user_set=user_named)

    def update_agent_cls(self, agent_cls: str) -> None:
        # Agent class is immutable start metadata. This local value is retained
        # only so a newly created session can inherit the active custom class.
        self.agent_cls = agent_cls

    def record_user(self, text: str):
        return self._handle.record_user_message(text)

    def close(self) -> None:
        self._handle.close()

    def as_markdown(self) -> str:
        lines = [f"# Session {self.session_id[:8]}\n"]
        for turn in self.turns:
            prefix = "**You:**" if turn.role == "user" else "**NeMo OO Agents:**"
            lines.append(f"{prefix}\n\n{turn.content}\n")
        return "\n---\n\n".join(lines)

    @classmethod
    def list_sessions(cls, limit: int = 20) -> list[SessionMeta]:
        return [
            SessionMeta.from_info(info) for info in SessionStore(SESSIONS_DIR).list(limit=limit)
        ]

    @classmethod
    def _read_meta(cls, path: Path) -> SessionMeta | None:
        try:
            return SessionMeta.from_info(SessionStore(path.parent).get(path.stem))
        except (OSError, ValueError):
            return None

    @classmethod
    def load_turns(cls, session_id: str) -> list[Turn]:
        return [
            Turn(role=turn.role, content=turn.content, ts=turn.timestamp)
            for turn in SessionStore(SESSIONS_DIR).load_turns(session_id)
        ]

    @classmethod
    def find_by_prefix(cls, prefix: str) -> list[str]:
        return SessionStore(SESSIONS_DIR).find_by_prefix(prefix)

    @classmethod
    def delete_session(cls, session_id: str) -> bool:
        deleted = SessionStore(SESSIONS_DIR).delete(session_id)
        # Session-scoped memory is stored in a sidecar SQLite database.
        delete_sqlite_database(SESSIONS_DIR / f"{session_id}-memory.db")
        return deleted


# Keep startup replay bounded for very long sessions.
RESUME_MAX_TURNS = 20


def build_resume_outputs(
    session_db_path: Path,
    session_id: str,
    *,
    in_nemo_term: bool = False,
    max_turns: int | None = None,
) -> list:
    """Build interleaved conversation and legacy rich-content replay output."""
    from .output import HistoryReplay, HistoryTurn, _RichReplayPayload

    if max_turns is None:
        max_turns = RESUME_MAX_TURNS

    try:
        connection = sqlite3.connect(str(session_db_path))
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT event_type, data FROM events ORDER BY insertion_order"
        ).fetchall()
        connection.close()
    except (OSError, sqlite3.Error):
        rows = []

    items: list[tuple[str, object]] = []
    pending: list[HistoryTurn] = []
    for row in rows:
        try:
            raw = json.loads(row["data"])
        except (TypeError, json.JSONDecodeError):
            continue
        event_type = row["event_type"]
        if event_type in {"SessionUserMessage", "TUIUserInput"}:
            content = raw.get("content", raw.get("text", ""))
            if content:
                pending.append(HistoryTurn(role="user", content=str(content)))
        elif event_type in {"AgentMessage", "TUIAgentMessage"} and raw.get("content"):
            pending.append(HistoryTurn(role="agent", content=str(raw["content"])))
        elif event_type == "RichOutput" and in_nemo_term and raw.get("payload"):
            if pending:
                items.append(("turns", pending))
                pending = []
            items.append(("rich", raw["payload"]))
    if pending:
        items.append(("turns", pending))
    if not items:
        return []

    total_turns = sum(len(data) for kind, data in items if kind == "turns")  # type: ignore[arg-type]
    omitted = max(0, total_turns - max_turns) if max_turns else 0
    if omitted:
        remaining = omitted
        kept: list[tuple[str, object]] = []
        keeping = False
        for kind, data in items:
            if kind == "turns":
                turns = data  # type: ignore[assignment]
                if remaining >= len(turns):
                    remaining -= len(turns)
                    continue
                if remaining:
                    turns = turns[remaining:]
                    remaining = 0
                keeping = True
                kept.append((kind, turns))
            elif keeping:
                kept.append((kind, data))
        items = kept

    turn_indices = [index for index, (kind, _) in enumerate(items) if kind == "turns"]
    if not turn_indices:
        return []
    first, last = turn_indices[0], turn_indices[-1]
    outputs: list = []
    for index, (kind, data) in enumerate(items):
        if kind == "turns":
            outputs.append(
                HistoryReplay(
                    turns=data,
                    session_id=session_id[:8] if index == first else "",
                    show_header=index == first,
                    show_footer=index == last,
                    omitted_count=omitted if index == first else 0,
                )
            )
        else:
            outputs.append(_RichReplayPayload(payload=data))
    return outputs
