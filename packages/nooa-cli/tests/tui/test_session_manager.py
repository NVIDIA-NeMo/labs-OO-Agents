# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SessionManager and related session persistence logic."""

from __future__ import annotations

import time
from unittest.mock import patch

from nooa_cli.tui.session_manager import SessionManager


def _make_sm(tmp_path, *, model="m", agent_cls="A", working_dir="", session_id=None):
    """Create a SessionManager backed by a SQLite DB in tmp_path."""
    return SessionManager.create(
        session_id=session_id,
        model=model,
        agent_cls=agent_cls,
        working_dir=working_dir,
    )


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------


class TestSessionManagerBasic:
    def test_creates_session_in_sessions_dir(self, tmp_path):
        """A new SessionManager creates a DB file in SESSIONS_DIR."""
        with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
            sm = _make_sm(tmp_path, model="openai/gpt-4o", agent_cls="TUIAgent", working_dir="/tmp")
            sid = sm.session_id
            sm.close()

            metas = SessionManager.list_sessions()

        assert any(m.id == sid for m in metas)

    def test_record_user_persisted(self, tmp_path):
        """record_user() inserts a user turn retrievable via load_turns."""
        with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
            sm = _make_sm(tmp_path)
            sm.record_user("hello")
            sid = sm.session_id
            sm.close()

            turns = SessionManager.load_turns(sid)

        assert len(turns) == 1
        assert turns[0].role == "user"
        assert turns[0].content == "hello"

    def test_turns_returned_in_insertion_order(self, tmp_path):
        """load_turns returns turns in insertion order."""
        from nooa.interactive import AgentMessage

        with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
            sm = _make_sm(tmp_path)
            sm.record_user("first")
            sm._handle.events.add(AgentMessage(content="reply"))
            sm.record_user("second")
            sid = sm.session_id
            sm.close()

            turns = SessionManager.load_turns(sid)

        assert [t.content for t in turns] == ["first", "reply", "second"]

    def test_load_turns_empty_for_missing_session(self, tmp_path):
        """load_turns returns [] for a session_id that doesn't exist."""
        with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
            turns = SessionManager.load_turns("nonexistent-id")
        assert turns == []


# ---------------------------------------------------------------------------
# list_sessions
# ---------------------------------------------------------------------------


class TestListSessions:
    def test_returns_empty_list_when_no_sessions(self, tmp_path):
        """list_sessions returns [] when SESSIONS_DIR is empty."""
        with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
            result = SessionManager.list_sessions()
        assert result == []

    def test_returns_sessions_sorted_newest_first(self, tmp_path):
        """Sessions are returned sorted by last_active (file mtime) descending."""
        with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
            sm_old = _make_sm(tmp_path)
            sm_old.record_user("old")
            sm_old.close()

            # Ensure different mtime by bumping the new file's mtime
            time.sleep(0.01)

            sm_new = _make_sm(tmp_path)
            sm_new.record_user("new")
            sm_new.close()

            sessions = SessionManager.list_sessions()

        # newest (sm_new) should be first
        ids = [s.id for s in sessions]
        assert ids.index(sm_new.session_id) < ids.index(sm_old.session_id)

    def test_turn_count_reflects_user_turns(self, tmp_path):
        """list_sessions includes turn_count = number of user turns."""
        from nooa.interactive import AgentMessage

        with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
            sm = _make_sm(tmp_path)
            sm.record_user("one")
            sm.record_user("two")
            sm._handle.events.add(AgentMessage(content="reply"))
            sid = sm.session_id
            sm.close()

            metas = SessionManager.list_sessions()

        meta = next(m for m in metas if m.id == sid)
        assert meta.turn_count == 2  # Agent replies do not add user turns.

    def test_limit_is_respected(self, tmp_path):
        """list_sessions(limit=N) returns at most N results."""
        with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
            for _ in range(5):
                sm = _make_sm(tmp_path)
                sm.record_user("hi")
                sm.close()

            sessions = SessionManager.list_sessions(limit=2)

        assert len(sessions) == 2


# ---------------------------------------------------------------------------
# rename / naming
# ---------------------------------------------------------------------------


class TestSessionManagerRename:
    def test_rename_persists_name(self, tmp_path):
        """rename() stores the name so it survives list_sessions()."""
        with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
            sm = _make_sm(tmp_path)
            sm.rename("My Session", user_named=True)
            sid = sm.session_id
            sm.close()

            metas = SessionManager.list_sessions()

        meta = next(m for m in metas if m.id == sid)
        assert meta.name == "My Session"
        assert meta.user_named is True

    def test_rename_auto_does_not_set_user_named(self, tmp_path):
        """rename(..., user_named=False) leaves user_named=False."""
        with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
            sm = _make_sm(tmp_path)
            sm.rename("auto name", user_named=False)
            sid = sm.session_id
            sm.close()

            metas = SessionManager.list_sessions()

        meta = next(m for m in metas if m.id == sid)
        assert meta.name == "auto name"
        assert meta.user_named is False


# ---------------------------------------------------------------------------
# close() idempotency
# ---------------------------------------------------------------------------


class TestSessionManagerClose:
    def test_close_is_idempotent(self, tmp_path):
        """Calling close() twice must not raise and session should still be listable."""
        with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
            sm = _make_sm(tmp_path, model="m", agent_cls="TUIAgent", working_dir="/tmp")
            sid = sm.session_id
            sm.close()
            sm.close()  # second call must be a no-op

            metas = SessionManager.list_sessions()
        assert any(m.id == sid for m in metas)

    def test_live_sandbox_session_is_active_without_shared_flock(self, tmp_path, monkeypatch):
        """The shared claim exposes activity across independent lock namespaces."""
        import nooa.storage.sqlite as sqlite_storage

        with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
            sm = _make_sm(tmp_path, session_id="sandbox-live")
            try:
                # Model host and sandbox kernels that do not share flock state.
                monkeypatch.setattr(sqlite_storage.fcntl, "flock", lambda *_args: None)
                assert SessionManager.is_active(sm.session_id)
            finally:
                sm.close()
            assert not SessionManager.is_active(sm.session_id)

    def test_close_session_still_accessible(self, tmp_path):
        """After close(), load_turns still works."""
        before = time.time()
        with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
            sm = _make_sm(tmp_path)
            sm.record_user("test")
            sid = sm.session_id
            sm.close()

            turns = SessionManager.load_turns(sid)
        after = time.time()

        assert turns[0].ts >= before
        assert turns[0].ts <= after


# ---------------------------------------------------------------------------
# find_by_prefix
# ---------------------------------------------------------------------------


class TestFindByPrefix:
    def test_finds_session_by_short_prefix(self, tmp_path):
        """find_by_prefix returns the full ID matching a short prefix."""
        with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
            sm = _make_sm(tmp_path)
            sid = sm.session_id
            sm.close()

            matches = SessionManager.find_by_prefix(sid[:8])

        assert matches == [sid]

    def test_returns_empty_for_no_match(self, tmp_path):
        with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
            matches = SessionManager.find_by_prefix("xxxxxxxx")
        assert matches == []

    def test_returns_multiple_for_ambiguous_prefix(self, tmp_path):
        """When multiple sessions share a prefix, all are returned."""
        sid1 = "prefix00-aaaa-0000-0000-000000000001"
        sid2 = "prefix00-bbbb-0000-0000-000000000002"
        with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
            sm1 = _make_sm(tmp_path, session_id=sid1)
            sm1.close()
            sm2 = _make_sm(tmp_path, session_id=sid2)
            sm2.close()

            matches = SessionManager.find_by_prefix("prefix00")

        assert len(matches) == 2
        assert sid1 in matches
        assert sid2 in matches


# ---------------------------------------------------------------------------
# delete_session
# ---------------------------------------------------------------------------


class TestDeleteSession:
    def test_delete_removes_session_and_turns(self, tmp_path):
        """delete_session() removes the DB file so the session disappears."""
        with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
            sm = _make_sm(tmp_path)
            sm.record_user("hi")
            sid = sm.session_id
            sm.close()

            SessionManager.delete_session(sid)

            metas = SessionManager.list_sessions()
            turns = SessionManager.load_turns(sid)

        assert not any(m.id == sid for m in metas)
        assert turns == []

    def test_delete_nonexistent_returns_false(self, tmp_path):
        with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
            result = SessionManager.delete_session("does-not-exist")
        assert result is False


# ---------------------------------------------------------------------------
# list_sessions sort stability
# ---------------------------------------------------------------------------


class TestListSessionsSortStability:
    def test_sessions_sorted_newest_first(self, tmp_path):
        """list_sessions returns sessions newest-first by file mtime."""
        with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
            sm_a = _make_sm(tmp_path)
            sm_a.record_user("a")
            sid_a = sm_a.session_id
            sm_a.close()

            time.sleep(0.01)

            sm_b = _make_sm(tmp_path)
            sm_b.record_user("b")
            sid_b = sm_b.session_id
            sm_b.close()

            time.sleep(0.01)

            sm_c = _make_sm(tmp_path)
            sm_c.record_user("c")
            sid_c = sm_c.session_id
            sm_c.close()

            metas = SessionManager.list_sessions()

        ids = [m.id for m in metas]
        # c is newest, a is oldest
        assert ids.index(sid_c) < ids.index(sid_b) < ids.index(sid_a)

    def test_all_sessions_returned(self, tmp_path):
        """list_sessions returns all sessions created in SESSIONS_DIR."""
        with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
            sm1 = _make_sm(tmp_path)
            sm1.close()
            sm2 = _make_sm(tmp_path)
            sm2.close()

            metas = SessionManager.list_sessions()

        # Both sessions should be present
        assert len(metas) == 2


async def test_native_adapter_uses_shared_title_request_and_preserves_user_title(tmp_path):
    from nooa_cli.coding.experimental_agent import ExperimentalCodingAgent
    from nooa_cli.coding.factory import create_session_agent
    from nooa_cli.interactive.session_title import SessionTitleRequest
    from nooa_cli.tui.bootstrap import session_options_from_config
    from nooa_cli.tui.config import Config

    from nooa.sessions import SessionStore
    from nooa.unifiedllm import FakeLLMClient

    with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
        manager = SessionManager.create(working_dir=str(tmp_path))
        config = Config()
        config.agent.working_dir = str(tmp_path)
        agent = create_session_agent(
            llm=FakeLLMClient(),
            storage=manager._storage,
            options=session_options_from_config(config),
        )
        try:
            assert type(agent) is ExperimentalCodingAgent
            agent._session_manager = manager
            request = SessionTitleRequest()
            assert request.request(agent, "Repair the session picker")
            assert not request.request(agent, "Do not queue the same request again")
            assert agent.rename_session("  Repair   session picker  ") == "Repair session picker"
            assert manager.name == "Repair session picker"
            assert SessionStore(tmp_path).get(manager.session_id).title == manager.name
            manager.rename("My chosen title", user_named=True)
            assert agent.rename_session("Late automatic title") == "My chosen title"
            assert not SessionTitleRequest().request(agent, "Another prompt")
            assert SessionStore(tmp_path).get(manager.session_id).title_is_user_set
        finally:
            await agent.aclose()
            await agent.shell.close()
            manager.close()


def test_deleting_session_preserves_retired_memory_sidecar(tmp_path):
    with patch("nooa_cli.tui.session_manager.SESSIONS_DIR", tmp_path):
        manager = SessionManager.create()
        session_id = manager.session_id
        manager.close()
        sidecar = tmp_path / f"{session_id}-memory.db"
        sidecar.write_bytes(b"existing user memory")
        assert SessionManager.delete_session(session_id)
        assert sidecar.read_bytes() == b"existing user memory"
