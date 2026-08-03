# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for host-neutral durable session storage."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime

import pytest

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
