# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sessions written by the ``nooa-acp`` server are found and loaded by nooa-coder.

The old server keeps each workspace's sessions in ``<workspace>/.nooa/sessions``
and records the agent as the bare name ``CodingAgent`` (the old TUI wrote
``TUIAgent``). Only this test
imports ``nooa_cli``; the package source must not.
"""

import logging
import sqlite3
from contextlib import closing

import pytest
from nooa_cli.coding import CodingAgent as OldCodingAgent
from nooa_cli.sessions.store import SessionStore as OldSessionStore
from nooa_coder.coding.agent import CodingAgent
from nooa_coder.coding.factory import create_session_agent
from nooa_coder.session.registry import SessionRegistry
from nooa_coder.session.store import SessionStore, sessions_root

from nooa.unifiedllm import FakeLLMClient

_TODO = "carry the plan over"


def _old_session(workspace, agent_name="CodingAgent", working_directory=None):
    """Write a session the way ``packages/nooa-acp`` does: message, then snapshot."""
    store = OldSessionStore(workspace / ".nooa" / "sessions")
    with store.create(
        model="fake",
        agent=agent_name,
        working_directory=working_directory or str(workspace),
        origin="acp",
    ) as handle:
        handle.record_user_message("please remember the plan")
        agent = OldCodingAgent(
            llm=FakeLLMClient([]),
            storage=handle.storage,
            cwd=workspace,
            libs_dir=workspace / ".nooa" / "libs",
        )
        agent.todo.add(_TODO)
        handle.storage.save_snapshot(agent)
        return handle.id


def _workspace(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    return workspace


@pytest.mark.parametrize("agent_name", ["CodingAgent", "TUIAgent"])
async def test_an_old_session_is_listed_and_loads_with_its_state(tmp_path, agent_name):
    workspace = _workspace(tmp_path)
    session_id = _old_session(workspace, agent_name)
    store = SessionStore(sessions_root(workspace))

    [info] = store.list(workspace=workspace)
    assert (info.id, info.parent_id, info.host) == (session_id, None, "acp")
    assert info.turn_count > 0

    registry = SessionRegistry(store, agent_factory=create_session_agent)
    try:
        session = await registry.load(session_id, llm=FakeLLMClient([]))
        assert isinstance(session._agent, CodingAgent)
        assert session._agent.cwd == workspace.resolve()
        assert [(e.role, e.content) for e in session.transcript()] == [
            ("user", "please remember the plan")
        ]
        assert _TODO in session._agent.todo.status()
    finally:
        await registry.close_all()


async def test_a_snapshot_that_cannot_be_restored_still_loads(tmp_path, caplog):
    workspace = _workspace(tmp_path)
    session_id = _old_session(workspace)
    path = sessions_root(workspace) / f"{session_id}.db"
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("UPDATE snapshots SET data = 'damaged'")

    registry = SessionRegistry(
        SessionStore(sessions_root(workspace)), agent_factory=create_session_agent
    )
    try:
        with caplog.at_level(logging.WARNING, logger="nooa_coder.session.registry"):
            session = await registry.load(session_id, llm=FakeLLMClient([]))
        assert f"Session {session_id}: could not restore its latest saved state" in caplog.text
        [user, note] = session.transcript()
        assert (user.role, user.content) == ("user", "please remember the plan")
        assert note.role == "note"
        assert "could not be restored" in note.content
        assert _TODO not in session._agent.todo.status()
        assert registry.get(session_id) is session
    finally:
        await registry.close_all()


@pytest.mark.parametrize("recorded", ["../", ".", "../../dev/"])
async def test_an_old_session_with_a_relative_working_directory_is_listed(tmp_path, recorded):
    """The old TUI recorded the directory as typed; a relative record hides nothing."""
    workspace = _workspace(tmp_path)
    session_id = _old_session(workspace, "TUIAgent", working_directory=recorded)
    store = SessionStore(sessions_root(workspace))

    [info] = store.list(workspace=workspace)
    assert info.id == session_id
    assert store.list(workspace=tmp_path / "elsewhere") == [info]


async def test_an_unreadable_latest_snapshot_falls_back_to_the_one_before(tmp_path, caplog):
    """A damaged file loses its newest pages first; the previous snapshot still counts."""
    workspace = _workspace(tmp_path)
    session_id = _old_session(workspace)
    path = sessions_root(workspace) / f"{session_id}.db"
    with closing(sqlite3.connect(path)) as db, db:
        # A second, newer snapshot that cannot be read back.
        db.execute(
            "INSERT INTO snapshots (snapshot_id, created_at, data) VALUES (?, ?, ?)",
            ("newest", "2099-01-01T00:00:00+00:00", "damaged"),
        )

    registry = SessionRegistry(
        SessionStore(sessions_root(workspace)), agent_factory=create_session_agent
    )
    try:
        with caplog.at_level(logging.WARNING, logger="nooa_coder.session.registry"):
            session = await registry.load(session_id, llm=FakeLLMClient([]))
        assert "restored an older snapshot" in caplog.text
        [user, note] = session.transcript()
        assert user.role == "user"
        assert "restored the older snapshot" in note.content
        assert _TODO in session._agent.todo.status()
    finally:
        await registry.close_all()
