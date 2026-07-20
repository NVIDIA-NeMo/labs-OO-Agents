# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test that SQLiteStorageManager commits before closing to prevent data loss."""

from __future__ import annotations

import tempfile
from pathlib import Path

from nooa import Agent
from nooa.context_blocks import Metadata
from nooa.runtime.event_manager import EventManager
from nooa.storage import SQLiteStorageManager
from nooa.unifiedllm import CompletionClient


class _TestAgent(Agent, llm=CompletionClient(model="openai/gpt-4o-mini", api_key="test")):
    """Simple agent for testing."""

    counter: int = 0


def test_close_commits_pending_events():
    """Verify that close() commits any pending transactions before closing.

    This test reproduces the bug where /clear would lose the last session's data
    because close() was called without committing pending writes.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_session.db"

        # Create storage and add some events
        storage = SQLiteStorageManager(db_path)

        # Add a metadata event through a manager bound to storage's backend
        em = EventManager(backend=storage.event_backend)
        em.add(Metadata(content="Test event before close"))

        # Close without explicit commit - the fix should handle this
        storage.close()

        # Reopen the database and verify the event was persisted
        storage2 = SQLiteStorageManager(db_path)
        em2 = EventManager(backend=storage2.event_backend)
        events = em2.values()
        storage2.close()

        assert len(events) >= 1, "Event should be persisted after close"
        assert any("Test event before close" in str(e) for e in events), (
            "The specific test event should be found"
        )


def test_close_commits_agent_snapshot():
    """Verify that close() commits agent snapshots before closing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_session.db"

        # Create storage and save a snapshot
        storage = SQLiteStorageManager(db_path)
        agent = _TestAgent(storage=storage)
        agent.counter = 42

        storage.save_snapshot(agent)

        # Close without explicit commit
        storage.close()

        # Reopen and verify snapshot is retrievable
        storage2 = SQLiteStorageManager(db_path)
        agent2 = _TestAgent(storage=storage2)

        restored = storage2.restore_latest_snapshot(agent2)
        storage2.close()

        assert restored is True, "Snapshot should be restorable after close"
        assert agent2.counter == 42, "Snapshot data should be preserved"


def test_session_swap_preserves_old_session():
    """Simulate the /clear command scenario where sessions are swapped.

    This reproduces the user-reported bug where resuming a session after
    /clear would find the session "disappeared".
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        session1_path = Path(tmpdir) / "session1.db"
        session2_path = Path(tmpdir) / "session2.db"

        # Session 1: Create and add events
        storage1 = SQLiteStorageManager(session1_path)
        em1 = EventManager(backend=storage1.event_backend)
        em1.add(Metadata(content="Session 1 message 1"))
        em1.add(Metadata(content="Session 1 message 2"))

        # Simulate /clear: close session 1 and start session 2
        storage1.close()  # This should commit pending events

        storage2 = SQLiteStorageManager(session2_path)
        em2 = EventManager(backend=storage2.event_backend)
        em2.add(Metadata(content="Session 2 message 1"))
        storage2.close()

        # Simulate /session resume: reopen session 1
        storage1_reopened = SQLiteStorageManager(session1_path)
        em1_reopened = EventManager(backend=storage1_reopened.event_backend)
        events = em1_reopened.values()
        storage1_reopened.close()

        # Verify session 1's events are still there
        assert len(events) >= 2, "Session 1 events should be preserved after swap"
        contents = [str(e) for e in events]
        assert any("Session 1 message 1" in c for c in contents), (
            "First message should be preserved"
        )
        assert any("Session 1 message 2" in c for c in contents), (
            "Second message should be preserved"
        )


def test_multiple_close_calls_safe():
    """Verify that calling close() multiple times is safe."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_session.db"
        storage = SQLiteStorageManager(db_path)
        em = EventManager(backend=storage.event_backend)
        em.add(Metadata(content="Test event"))

        # Close multiple times - should not raise
        storage.close()
        storage.close()  # Second close should be safe

        # Verify data is still intact
        storage2 = SQLiteStorageManager(db_path)
        em2 = EventManager(backend=storage2.event_backend)
        events = em2.values()
        storage2.close()

        assert len(events) >= 1, "Event should be persisted"
