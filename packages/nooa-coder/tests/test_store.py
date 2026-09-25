# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SessionStore: durable session files, their metadata and listing."""

import json
import sqlite3
from pathlib import Path

import pytest
from nooa_coder.session.items import SessionInfo, TurnCancelled, Usage
from nooa_coder.session.store import SessionNotFoundError, SessionStore, sessions_root

from nooa.context_blocks import Metadata
from nooa.context_blocks.roles import Role


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


def test_sessions_live_in_the_workspace_by_default(tmp_path):
    assert sessions_root(tmp_path) == tmp_path / ".nooa" / "sessions"
    store = SessionStore(sessions_root(tmp_path))
    with store.create(workspace=str(tmp_path)) as handle:
        assert handle.path.parent == tmp_path / ".nooa" / "sessions"


def test_nooa_sessions_dir_names_one_shared_directory(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    monkeypatch.setenv("NOOA_SESSIONS_DIR", str(shared))
    assert sessions_root(tmp_path / "one") == shared
    assert sessions_root(tmp_path / "two") == shared


def test_an_explicit_directory_wins_over_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("NOOA_SESSIONS_DIR", str(tmp_path / "shared"))
    assert sessions_root(tmp_path, tmp_path / "given") == tmp_path / "given"


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


def test_readers_work_while_the_session_is_open(sessions_dir):
    store = SessionStore(sessions_dir)
    with store.create(model="m") as handle:
        handle.set_title("live title")
        assert store.get(handle.id).title == "live title"
        assert [info.id for info in store.list()] == [handle.id]


def test_readers_racing_a_delete_create_no_file(sessions_dir, tmp_path, monkeypatch):
    """Readers open read-only, so a file deleted under them is not recreated."""
    store = SessionStore(sessions_dir)
    with store.create() as handle:
        session_id, path = handle.id, handle.path
    store.delete(session_id)
    other = tmp_path / "other"
    other.write_text("x")
    real_stat = Path.stat

    with monkeypatch.context() as patch:
        # Both readers check the file first; make the check pass as if the
        # delete landed just after it.
        patch.setattr(Path, "exists", lambda self: True)
        patch.setattr(
            Path, "stat", lambda self, **kw: real_stat(other if self == path else self, **kw)
        )
        assert store._read_info(path) is None
        assert store._read_rows(path) == []
    assert not path.exists()


def test_listing_shows_roots_by_default(sessions_dir):
    store = SessionStore(sessions_dir)
    with store.create() as root:
        with store.create(parent_id=root.id, depth=1, name="child") as child:
            root_id, child_id = root.id, child.id
    assert [info.id for info in store.list()] == [root_id]
    assert {info.id for info in store.list(roots_only=False)} == {root_id, child_id}


def test_turn_markers_and_turn_cancelled_reload_with_their_types(sessions_dir, monkeypatch):
    # Another class with the same name in the global registry (as nooa_cli's
    # copies are in a full test run) must not change what this store loads.
    from nooa_coder.session import events

    from nooa.context_blocks.events import _EVENT_REGISTRY

    store = SessionStore(sessions_dir)
    written = [
        events.ItemAdmitted(
            channel="user_messages",
            item_id="i1",
            item_json='"hi"',
            item_type="builtins:str",
            source="user",
        ),
        events.ItemConsumed(item_id="i1"),
        events.TurnStarted(item_ids=["i1"], item_preview="hi"),
        TurnCancelled(by="user", interrupted="7"),
        events.TurnEnded(outcome_kind="cancelled", usage=Usage(input_tokens=3)),
        events.ItemWithdrawn(item_id="i2"),
        events.ItemRequeued(item_id="i3"),
    ]
    with store.create() as handle:
        session_id = handle.id
        for event in written:
            handle.events.add(event)

    for event in written:
        monkeypatch.setitem(_EVENT_REGISTRY, type(event).__name__, SessionInfo)
    with store.open(session_id) as handle:
        loaded = [e for e in handle.events.values() if type(e).__name__ != "SessionStarted"]
    assert [type(e) for e in loaded] == [type(e) for e in written]
    assert [e.model_dump(exclude={"tag"}) for e in loaded] == [
        e.model_dump(exclude={"tag"}) for e in written
    ]
    [cancelled] = [e for e in loaded if isinstance(e, TurnCancelled)]
    assert cancelled._role is Role.USER
    assert not isinstance(cancelled, Metadata)
    assert all(isinstance(e, Metadata) for e in loaded if e is not cancelled)


def test_turn_count_counts_admitted_user_messages(sessions_dir):
    from nooa_coder.session import events

    store = SessionStore(sessions_dir)
    with store.create() as handle:
        for channel in ("user_messages", "delegates", "user_messages"):
            handle.events.add(
                events.ItemAdmitted(channel=channel, item_id=channel, item_json="1", source="x")
            )
        session_id = handle.id
    assert store.get(session_id).turn_count == 2


@pytest.fixture
def local_tz(monkeypatch):
    """Set the process's local time zone; restored after the test."""
    import time

    def set_tz(name: str) -> None:
        monkeypatch.setenv("TZ", name)
        time.tzset()

    yield set_tz
    monkeypatch.undo()
    time.tzset()


def test_timestamps_do_not_depend_on_the_readers_time_zone(sessions_dir, local_tz):
    import time

    from nooa_coder.session import events

    local_tz("America/Los_Angeles")
    store = SessionStore(sessions_dir)
    before = time.time()
    with store.create() as handle:
        handle.events.add(
            events.ItemAdmitted(channel="user_messages", item_id="i1", item_json='"hi"', source="u")
        )
        session_id = handle.id
        created = handle.info.created_at
    after = time.time()
    assert before - 1 <= created <= after + 1  # the true epoch, as the writer computed it

    info = store.get(session_id)
    [entry] = store.load_transcript(session_id)
    local_tz("Asia/Tokyo")
    moved = store.get(session_id)
    [moved_entry] = store.load_transcript(session_id)
    assert info.created_at == moved.created_at == pytest.approx(created, abs=1e-3)
    assert entry.timestamp == moved_entry.timestamp
    assert before - 1 <= moved_entry.timestamp <= after + 1
