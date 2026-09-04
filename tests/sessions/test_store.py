# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for host-neutral durable session storage."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import threading
from datetime import UTC, datetime

import pytest

import nooa.sessions.store as store_module
from nooa.interactive import AgentMessage
from nooa.sessions import InvalidSessionIdError, SessionNotFoundError, SessionStore
from nooa.storage import SessionAlreadyActiveError


def test_create_record_title_list_and_resume(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create(
        session_id="session-one",
        host="tui",
        model="test/model",
        agent="CodingAgent",
        working_directory="/workspace",
    )
    session.record_user_message("hello")
    session.events.add(AgentMessage(content="hi back"))
    session.set_title("First session", user_set=True)
    session.close()

    info = store.list()[0]
    assert info.id == "session-one"
    assert info.host == "tui"
    assert info.model == "test/model"
    assert info.agent == "CodingAgent"
    assert info.working_directory == "/workspace"
    assert info.title == "First session"
    assert info.title_is_user_set is True
    assert info.turn_count == 2

    resumed = store.open("session-one")
    try:
        assert resumed.info == info
        assert [(turn.role, turn.content) for turn in resumed.turns()] == [
            ("user", "hello"),
            ("agent", "hi back"),
        ]
    finally:
        resumed.close()


def test_session_summary_uses_targeted_queries(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    session = store.create(session_id="query-shape")
    session.record_user_message("hello")
    session.set_title("Targeted")
    session.close()

    queries: list[str] = []
    real_connect = sqlite3.connect

    class RecordingConnection:
        def __init__(self, *args, **kwargs):
            self._connection = real_connect(*args, **kwargs)

        def execute(self, query, parameters=()):
            queries.append(" ".join(query.split()))
            return self._connection.execute(query, parameters)

        def close(self):
            self._connection.close()

    monkeypatch.setattr(store_module.sqlite3, "connect", RecordingConnection)

    assert store.get("query-shape").title == "Targeted"
    assert not any(
        query == "SELECT event_type, data FROM events ORDER BY insertion_order" for query in queries
    )
    assert any("SELECT COUNT(*)" in query for query in queries)
    assert sum("LIMIT 1" in query for query in queries) == 2


def test_different_session_databases_can_be_open_together(tmp_path):
    store = SessionStore(tmp_path)
    first = store.create(session_id="first")
    second = store.create(session_id="second")
    try:
        first.record_user_message("one")
        second.record_user_message("two")
        assert [turn.content for turn in first.turns()] == ["one"]
        assert [turn.content for turn in second.turns()] == ["two"]
    finally:
        first.close()
        second.close()


def test_same_live_session_cannot_be_opened_twice(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create(session_id="live")
    try:
        with pytest.raises(SessionAlreadyActiveError):
            store.open("live")
    finally:
        session.close()

    resumed = store.open("live")
    resumed.close()


def test_shared_claim_blocks_open_when_flock_namespace_cannot_see_owner(tmp_path, monkeypatch):
    """The filesystem claim prevents a second writer even if flock appears free."""
    import nooa.storage.sqlite as sqlite_storage

    store = SessionStore(tmp_path)
    session = store.create(session_id="sandbox-live")
    try:
        # Simulate a host/sandbox boundary whose flock namespaces do not contend.
        monkeypatch.setattr(sqlite_storage.fcntl, "flock", lambda *_args: None)

        assert sqlite_storage.is_sqlite_database_active(session.path)
        with pytest.raises(SessionAlreadyActiveError) as exc_info:
            store.open(session.id)
        assert exc_info.value.owner_pid == os.getpid()
        with pytest.raises(SessionAlreadyActiveError):
            store.delete(session.id)
    finally:
        session.close()


def test_shared_claim_is_visible_to_an_independent_process(tmp_path):
    """A separate process observes the claim even when its flock is disabled."""
    store = SessionStore(tmp_path)
    session = store.create(session_id="process-live")
    script = """
import sys
import nooa.storage.sqlite as sqlite_storage
from nooa.sessions import SessionStore
from nooa.storage import SessionAlreadyActiveError

sqlite_storage.fcntl.flock = lambda *_args: None
store = SessionStore(sys.argv[1])
try:
    store.open("process-live")
except SessionAlreadyActiveError:
    pass
else:
    raise SystemExit("independent process opened an active session")
try:
    store.delete("process-live")
except SessionAlreadyActiveError:
    pass
else:
    raise SystemExit("independent process deleted an active session")
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path)],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stderr or result.stdout
    finally:
        session.close()


def test_forked_child_close_does_not_remove_parent_claim(tmp_path, monkeypatch):
    """A forked copy cannot release the original process's ownership marker."""
    import nooa.storage.sqlite as sqlite_storage

    path = tmp_path / "fork-safe.db"
    manager = sqlite_storage.SQLiteStorageManager(path)
    claim = manager._session_claim
    assert claim is not None
    claim_path = sqlite_storage._claim_path(path)
    parent_pid = os.getpid()

    monkeypatch.setattr(sqlite_storage.os, "getpid", lambda: parent_pid + 1)
    claim.close()
    assert claim_path.is_dir()

    monkeypatch.setattr(sqlite_storage.os, "getpid", lambda: parent_pid)
    manager.close()
    assert not claim_path.exists()


def test_orphaned_shared_claim_requires_explicit_recovery(tmp_path, monkeypatch):
    """An orphaned claim never silently admits a potentially live old writer."""
    import nooa.storage.sqlite as sqlite_storage

    store = SessionStore(tmp_path)
    session = store.create(session_id="orphaned")
    path = session.path
    session.close()

    claim_path = sqlite_storage._claim_path(path)
    claim_path.mkdir()
    owner_path = claim_path / "owner-unknown.json"
    owner_path.write_text('{"token": "unknown", "pid": 123}')
    monkeypatch.setattr(sqlite_storage.fcntl, "flock", lambda *_args: None)

    assert sqlite_storage.is_sqlite_database_active(path)
    with pytest.raises(SessionAlreadyActiveError, match="remove .*orphaned.active"):
        store.open("orphaned")

    owner_path.unlink()
    claim_path.rmdir()
    resumed = store.open("orphaned")
    resumed.close()


def test_old_owner_does_not_remove_replacement_claim(tmp_path):
    """Token checking keeps stale cleanup from deleting a successor's claim."""
    import nooa.storage.sqlite as sqlite_storage

    path = tmp_path / "replaced.db"
    manager = sqlite_storage.SQLiteStorageManager(path)
    claim = manager._session_claim
    assert claim is not None
    claim_path = sqlite_storage._claim_path(path)
    displaced = claim_path.with_name("replaced.displaced")
    claim_path.rename(displaced)
    claim_path.mkdir()
    replacement_owner = claim_path / "owner-replacement.json"
    replacement_owner.write_text('{"token": "replacement", "pid": 456}')

    manager.close()

    assert claim_path.is_dir()
    assert json.loads(replacement_owner.read_text())["token"] == "replacement"
    replacement_owner.unlink()
    claim_path.rmdir()
    next(displaced.iterdir()).unlink()
    displaced.rmdir()


def test_user_messages_are_thread_safe_when_host_opts_into_cross_thread_access(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create(session_id="threaded", check_same_thread=False)
    barrier = threading.Barrier(3)
    errors: list[Exception] = []

    def write(prefix: str) -> None:
        try:
            barrier.wait(timeout=5)
            for index in range(25):
                session.record_user_message(f"{prefix}-{index}")
        except Exception as error:
            errors.append(error)

    writers = [threading.Thread(target=write, args=(prefix,)) for prefix in ("a", "b")]
    for writer in writers:
        writer.start()
    barrier.wait(timeout=5)
    for writer in writers:
        writer.join(timeout=10)

    try:
        assert errors == []
        assert all(not writer.is_alive() for writer in writers)
        assert len(session.turns()) == 50
        assert session.info.turn_count == 50
    finally:
        session.close()


def test_open_missing_or_invalid_session(tmp_path):
    store = SessionStore(tmp_path)
    with pytest.raises(SessionNotFoundError):
        store.open("missing")
    for unsafe in ("", ".", "..", "../escape", "nested/id", "nested\\id", "bad\x00id"):
        with pytest.raises(InvalidSessionIdError):
            store.path_for(unsafe)


def test_delete_refuses_live_session_then_removes_database_and_sidecars(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create(session_id="delete-me")
    path = session.path
    with pytest.raises(SessionAlreadyActiveError):
        store.delete(session.id)

    session.close()
    path.with_name(f"{path.name}-wal").touch()
    path.with_name(f"{path.name}-shm").touch()
    assert store.delete("delete-me") is True
    assert not path.exists()
    assert not path.with_name(f"{path.name}-wal").exists()
    assert not path.with_name(f"{path.name}-shm").exists()
    assert store.delete("delete-me") is False


def test_delete_missing_session_from_missing_root_returns_false(tmp_path):
    store = SessionStore(tmp_path / "not-created")
    assert store.delete("missing") is False


def test_list_skips_corrupt_and_non_session_databases(tmp_path):
    store = SessionStore(tmp_path)
    valid = store.create(session_id="valid")
    valid.close()
    (tmp_path / "corrupt.db").write_bytes(b"not sqlite")
    connection = sqlite3.connect(tmp_path / "no-start.db")
    connection.execute("CREATE TABLE events (event_type TEXT, data TEXT, insertion_order INTEGER)")
    connection.close()

    assert [info.id for info in store.list()] == ["valid"]


def test_prefix_search_is_literal_and_sorted(tmp_path):
    store = SessionStore(tmp_path)
    for session_id in ("prefix-a", "prefix-b", "other"):
        session = store.create(session_id=session_id)
        session.close()

    assert set(store.find_by_prefix("prefix-")) == {"prefix-a", "prefix-b"}
    assert store.find_by_prefix("*") == []
    assert store.find_by_prefix("../") == []


def test_reads_legacy_tui_session_events(tmp_path):
    store = SessionStore(tmp_path)
    session = store.create(session_id="legacy-placeholder")
    session.close()
    store.delete("legacy-placeholder")

    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE events (
            tag TEXT PRIMARY KEY,
            event_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            data TEXT NOT NULL,
            insertion_order INTEGER NOT NULL
        );
        """
    )
    timestamp = datetime.now(UTC).isoformat()
    legacy_events = [
        (
            "TUISessionStart",
            {
                "event_type": "TUISessionStart",
                "timestamp": timestamp,
                "model": "legacy/model",
                "agent_cls": "TUIAgent",
                "working_dir": "/legacy",
            },
        ),
        (
            "TUIUserInput",
            {"event_type": "TUIUserInput", "timestamp": timestamp, "text": "old user"},
        ),
        (
            "TUIAgentMessage",
            {
                "event_type": "TUIAgentMessage",
                "timestamp": timestamp,
                "content": "old agent",
            },
        ),
        (
            "TUISessionRename",
            {
                "event_type": "TUISessionRename",
                "timestamp": timestamp,
                "name": "Legacy title",
                "user_named": True,
            },
        ),
    ]
    for order, (event_type, data) in enumerate(legacy_events):
        connection.execute(
            "INSERT INTO events VALUES (?, ?, ?, 'active', ?, ?)",
            (str(order + 1), f"id-{order}", event_type, json.dumps(data), order),
        )
    connection.commit()
    connection.close()

    info = store.get("legacy")
    assert info.host == "tui"
    assert info.model == "legacy/model"
    assert info.agent == "TUIAgent"
    assert info.working_directory == "/legacy"
    assert info.title == "Legacy title"
    assert info.title_is_user_set is True
    assert info.turn_count == 2
    assert [(turn.role, turn.content) for turn in store.load_turns("legacy")] == [
        ("user", "old user"),
        ("agent", "old agent"),
    ]


def test_load_recent_turns_is_bounded_and_chronological(tmp_path):
    store = SessionStore(tmp_path)
    handle = store.create(session_id="recent", working_directory=str(tmp_path))
    handle.record_user_message("first")
    handle.events.add(AgentMessage(content="middle"))
    handle.record_user_message("last")
    handle.close()

    turns = store.load_recent_turns("recent", limit=2)

    assert [(turn.role, turn.content) for turn in turns] == [
        ("agent", "middle"),
        ("user", "last"),
    ]
    assert store.load_recent_turns("recent", limit=0) == []
    with pytest.raises(ValueError, match="non-negative"):
        store.load_recent_turns("recent", limit=-1)
