# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for trace session name generation and assignment.

Covers:
- ``_make_trace_session_name`` format
- ``bootstrap()`` calls ``set_session`` with the right name after ``_session_id`` is resolved
- ``Session._swap_session_manager`` calls ``set_session`` with the right name
- Both are silent when the tracing package is not installed
"""

from __future__ import annotations

import re
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

from nooa_cli.tui.session_manager import _make_trace_session_name

# ---------------------------------------------------------------------------
# _make_trace_session_name
# ---------------------------------------------------------------------------

TRACE_NAME_RE = re.compile(r"^tui-\d{8}-\d{6}-[0-9a-f]{8}$")


class TestMakeTraceSessionName:
    def test_format(self):
        """Name must match tui-YYYYMMDD-HHMMSS-<8hex>."""
        name = _make_trace_session_name("abcdef12-0000-0000-0000-000000000000")
        assert TRACE_NAME_RE.match(name), f"Unexpected format: {name!r}"

    def test_uses_session_id_prefix(self):
        """Last 8 chars must be the first 8 chars of the session UUID."""
        sid = "12345678-aaaa-bbbb-cccc-dddddddddddd"
        name = _make_trace_session_name(sid)
        assert name.endswith("-12345678")

    def test_empty_session_id(self):
        """Should not raise on empty string — produces tui-YYYYMMDD-HHMMSS-."""
        name = _make_trace_session_name("")
        assert name.startswith("tui-")


# ---------------------------------------------------------------------------
# Session._swap_session_manager calls set_session
# ---------------------------------------------------------------------------


def _make_mock_session(tmp_path, monkeypatch):
    """Build a minimal Session object with mocked dependencies."""
    from nooa_cli.tui import session_manager as session_manager_module
    from nooa_cli.tui.session import Session
    from nooa_cli.tui.session_manager import SessionManager

    sid = str(uuid.uuid4())
    monkeypatch.setattr(session_manager_module, "SESSIONS_DIR", tmp_path)
    sm = SessionManager.create(session_id=sid, model="m", agent_cls="A")
    storage = sm._storage

    agent = MagicMock()
    agent._storage = storage
    agent.event_manager = MagicMock()
    agent.event_manager.on = MagicMock(return_value=lambda: None)
    agent.event_manager.keys = MagicMock(return_value=[])
    agent.handle = AsyncMock()
    # queue_manager mock with async shutdown and empty channels
    agent.queue_manager = MagicMock()
    agent.queue_manager.shutdown = AsyncMock()
    agent.queue_manager._channels = {}

    frontend = AsyncMock()
    frontend.render = AsyncMock()
    frontend.get_input = AsyncMock(return_value="")
    frontend.show_python = True
    frontend.clear_streaming_state = MagicMock()

    config = MagicMock()
    config.tui.default_model = "test-model"
    config.tui.vi_mode = False

    registry = MagicMock()
    registry.session_manager = sm
    registry._commands = {}

    session = Session.__new__(Session)
    session.frontend = frontend
    session.agent = agent
    session.config = config
    session.registry = registry
    session._session_manager = sm
    session._background_tasks = set()

    return session, sm, storage


class TestSwapSessionManagerCallsSetSession:
    async def test_calls_set_session_with_correct_format(self, tmp_path, monkeypatch):
        """_swap_session_manager must call set_session with tui-YYYYMMDD-HHMMSS-<sid8>."""
        import sys

        session, old_sm, _ = _make_mock_session(tmp_path, monkeypatch)

        new_sid = str(uuid.uuid4())
        new_sm = MagicMock()
        new_sm.session_id = new_sid
        new_sm._storage = MagicMock()

        captured: list[str] = []
        mock_set_session = MagicMock(side_effect=captured.append)
        fake_tracing = MagicMock()
        fake_tracing.set_session = mock_set_session

        with patch.dict(sys.modules, {"nooa.tracing": fake_tracing}):
            await session._swap_session_manager(new_sm)

        assert len(captured) == 1, f"set_session not called; got {captured}"
        assert TRACE_NAME_RE.match(captured[0]), f"Bad format: {captured[0]!r}"
        assert captured[0].endswith(f"-{new_sid[:8]}")

        old_sm.close()

    async def test_silent_when_import_fails(self, tmp_path, monkeypatch):
        """_swap_session_manager must not raise when tracing is not installed."""
        session, old_sm, _ = _make_mock_session(tmp_path, monkeypatch)

        new_sm = MagicMock()
        new_sm.session_id = str(uuid.uuid4())
        new_sm._storage = MagicMock()

        import builtins

        real_import = builtins.__import__

        def _raise_on_tracing(name, *args, **kwargs):
            if "openinference_instrumentation" in name:
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_raise_on_tracing):
            # Should not raise
            await session._swap_session_manager(new_sm)

        old_sm.close()

    async def test_silent_when_set_session_raises(self, tmp_path, monkeypatch):
        """_swap_session_manager must not crash if set_session raises a runtime error."""
        import sys

        session, old_sm, _ = _make_mock_session(tmp_path, monkeypatch)

        new_sm = MagicMock()
        new_sm.session_id = str(uuid.uuid4())
        new_sm._storage = MagicMock()

        fake_tracing = MagicMock()
        fake_tracing.set_session = MagicMock(side_effect=RuntimeError("tracing broken"))

        with patch.dict(sys.modules, {"nooa.tracing": fake_tracing}):
            # Should not raise
            await session._swap_session_manager(new_sm)

        old_sm.close()


# ---------------------------------------------------------------------------
# _swap_session_manager flushes queues
# ---------------------------------------------------------------------------


class TestSwapSessionManagerClearsQueues:
    async def test_flushes_queue_channels(self, tmp_path, monkeypatch):
        """_swap_session_manager must flush all queue-mode channels."""
        session, old_sm, _ = _make_mock_session(tmp_path, monkeypatch)

        # Put a stale item in a queue channel
        from nooa.runtime.channels import Channel

        ch = Channel("user_messages", "queue")
        ch.put("stale message from old session")
        session.agent.queue_manager._channels = {"user_messages": ch}
        session.agent.queue_manager.names = MagicMock(return_value=["user_messages"])
        session.agent.queue_manager.get_channel = MagicMock(side_effect=lambda name: ch)

        new_sm = MagicMock()
        new_sm.session_id = str(uuid.uuid4())
        new_sm._storage = MagicMock()

        await session._swap_session_manager(new_sm)

        assert ch.is_empty(), "queue channel should be flushed after session swap"

    async def test_calls_shutdown_on_queue_manager(self, tmp_path, monkeypatch):
        """_swap_session_manager must call shutdown() to cancel spawned jobs."""
        session, old_sm, _ = _make_mock_session(tmp_path, monkeypatch)

        new_sm = MagicMock()
        new_sm.session_id = str(uuid.uuid4())
        new_sm._storage = MagicMock()

        await session._swap_session_manager(new_sm)

        session.agent.queue_manager.shutdown.assert_awaited_once()

    async def test_no_crash_without_queue_manager(self, tmp_path, monkeypatch):
        """_swap_session_manager must not crash if agent has no queue_manager."""
        session, old_sm, _ = _make_mock_session(tmp_path, monkeypatch)
        del session.agent.queue_manager

        new_sm = MagicMock()
        new_sm.session_id = str(uuid.uuid4())
        new_sm._storage = MagicMock()

        # Should not raise
        await session._swap_session_manager(new_sm)

        old_sm.close()
