# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SessionStore: durable session files, their metadata and listing."""

import json
import sqlite3

import pytest
from nooa_coder.session.items import SessionInfo, Usage
from nooa_coder.session.store import SessionNotFoundError, SessionStore


def test_store_returns_the_single_session_info_model(sessions_dir):
    store = SessionStore(sessions_dir)
    with store.create(model="m", agent="pkg:Agent") as handle:
        assert isinstance(handle.info, SessionInfo)
        assert handle.info.status == "on_disk"
        assert handle.info.usage == Usage()
        session_id = handle.id

    info = store.get(session_id)
    assert isinstance(info, SessionInfo)
    assert (info.id, info.model, info.agent) == (session_id, "m", "pkg:Agent")
    assert [i.id for i in store.list()] == [session_id]


def test_default_directory_is_the_user_level_sessions_dir(_user_dir):
    store = SessionStore()
    assert store.root == _user_dir / "sessions"
    with store.create() as handle:
        assert handle.path.parent == _user_dir / "sessions"


def test_workspace_is_recorded_and_filters_the_listing(sessions_dir, tmp_path):
    store = SessionStore(sessions_dir)
    first, second = tmp_path / "one", tmp_path / "two"
    with store.create(workspace=str(first)) as a, store.create(workspace=str(second)) as b:
        ids = {a.id: first, b.id: second}
    assert {info.id: info.workspace for info in store.list()} == {
        key: str(value) for key, value in ids.items()
    }
    assert [info.id for info in store.list(workspace=first)] == [
        key for key, value in ids.items() if value == first
    ]
    assert store.list(workspace=tmp_path / "three") == []


def test_tree_position_is_recorded_at_creation(sessions_dir, tmp_path):
    store = SessionStore(sessions_dir)
    with store.create(
        model="m",
        agent="pkg:Agent",
        workspace=str(tmp_path),
        host="headless",
        parent_id="parent-1",
        depth=1,
        name="Review auth",
        retained=True,
    ) as handle:
        live = handle.info
    on_disk = store.get(live.id)
    for info in (live, on_disk):
        assert (info.parent_id, info.depth, info.name, info.retained, info.host) == (
            "parent-1",
            1,
            "Review auth",
            True,
            "headless",
        )
        assert info.workspace == str(tmp_path)


def test_records_written_before_the_tree_fields_still_load(sessions_dir):
    store = SessionStore(sessions_dir)
    with store.create(model="m", agent="a") as handle:
        session_id, path = handle.id, handle.path
    old = {
        "event_type": "SessionStarted",
        "id": "e1",
        "metadata": {},
        "timestamp": "2026-09-01T10:00:00",
        "origin": "tui",
        "model": "old-model",
        "agent": "old:Agent",
        "working_directory": "/work/old",
    }
    connection = sqlite3.connect(path)
    with connection:
        connection.execute(
            "UPDATE events SET data = ? WHERE event_type = 'SessionStarted'", (json.dumps(old),)
        )
    connection.close()

    info = store.get(session_id)
    assert (info.model, info.agent, info.host, info.workspace) == (
        "old-model",
        "old:Agent",
        "tui",
        "/work/old",
    )
    assert (info.parent_id, info.depth, info.name, info.retained) == (None, 0, None, False)
    with store.open(session_id) as handle:
        assert handle.info.host == "tui"


def test_open_of_a_deleted_session_raises_and_creates_no_file(sessions_dir):
    store = SessionStore(sessions_dir)
    with store.create() as handle:
        session_id, path = handle.id, handle.path
    assert store.delete(session_id)

    with pytest.raises(SessionNotFoundError):
        store.open(session_id)
    assert not path.exists()


def test_open_racing_a_delete_raises_and_creates_no_file(sessions_dir, monkeypatch):
    """The file disappears between the metadata read and the connection."""
    store = SessionStore(sessions_dir)
    with store.create() as handle:
        session_id, path = handle.id, handle.path
        info = handle.info
    store.delete(session_id)
    monkeypatch.setattr(store, "_read_info", lambda _path: info)

    with pytest.raises(SessionNotFoundError):
        store.open(session_id)
    assert not path.exists()
