# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SessionStore: durable session files, their metadata and listing."""

from nooa_coder.session.items import SessionInfo, Usage
from nooa_coder.session.store import SessionStore


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
