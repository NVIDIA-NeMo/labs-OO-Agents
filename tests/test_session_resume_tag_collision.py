# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression test: resumed sessions must not reuse tag numbers already in the backend.

Bug: An earlier design cached ``_next_tag_num`` on the EventManager and
restored it from the snapshot's recorded value. If events were added to
the backend *after* the snapshot was taken (e.g. session-close metadata),
those events occupied higher tag numbers and the restored agent would
collide with them.

Tag allocation now lives on the EventBackend itself and is lazy-initialised
from ``max_tag_num()``, so the restored backend always picks the next
available tag regardless of what the snapshot recorded.
"""

from __future__ import annotations

import sqlite3

from nooa import Agent
from nooa.events import EventBase
from nooa.storage import SQLiteStorageManager
from nooa.storage.sqlite import SQLiteEventBackend
from nooa.unifiedllm import CompletionClient

_LLM = CompletionClient(model="openai/gpt-4o-mini", api_key="test")


class _SimpleAgent(Agent, llm=_LLM):
    value: int = 0


def test_next_tag_num_does_not_collide_after_resume(tmp_path):
    """_next_tag_num after restore must be > every tag already in the backend."""
    db = tmp_path / "session.db"

    # --- first session ---
    storage = SQLiteStorageManager(db)
    agent = _SimpleAgent(storage=storage)

    # Add some events so the tag counter advances
    agent.event_manager.add(EventBase())
    agent.event_manager.add(EventBase())
    agent.event_manager.add(EventBase())
    # tag counter is now at 4; snapshot records next_tag_num=4

    storage.save_snapshot(agent)

    # Simulate events added AFTER the snapshot (e.g. session-close metadata)
    agent.event_manager.add(EventBase())  # tag "4"
    agent.event_manager.add(EventBase())  # tag "5"
    storage.close()

    # --- resumed session ---
    storage2 = SQLiteStorageManager(db)
    agent2 = _SimpleAgent(storage=storage2)
    storage2.restore_latest_snapshot(agent2)

    # Emitting a new event must produce a tag strictly greater than 5
    # (the highest tag already in storage) and not collide with any
    # existing tag.
    existing_tags = {e.tag for e in agent2.event_manager._backend.all_events()}
    new_tag = agent2.event_manager.add(EventBase())
    assert int(new_tag) > 5, (
        f"new event got tag {new_tag!r}; will collide with existing tags up to 5"
    )
    assert new_tag not in existing_tags, (
        f"new event was assigned tag {new_tag!r} which already exists in the backend"
    )

    storage2.close()


def test_next_tag_num_correct_after_resume_with_range_tags(tmp_path):
    """Resume must not collide when the backend contains collapsed range tags."""
    db = tmp_path / "session.db"

    # --- first session: add events then collapse to create a range tag ---
    storage = SQLiteStorageManager(db)
    agent = _SimpleAgent(storage=storage)

    agent.event_manager.add(EventBase())  # tag "1"
    agent.event_manager.add(EventBase())  # tag "2"
    agent.event_manager.add(EventBase())  # tag "3"
    # collapse "1".."3" → creates range tag "1..3"; next_tag_num stays at 4
    agent.event_manager.collapse("1", "3")

    storage.save_snapshot(agent)  # snapshot records next_tag_num=4

    # Events added after snapshot — these are what the fix must account for
    agent.event_manager.add(EventBase())  # tag "4"
    storage.close()

    # --- resumed session ---
    storage2 = SQLiteStorageManager(db)
    agent2 = _SimpleAgent(storage=storage2)
    storage2.restore_latest_snapshot(agent2)

    # Backend has tags "1..3" (range end=3) and "4"; max is 4, so next must be ≥ 5
    existing_tags = {e.tag for e in agent2.event_manager._backend.all_events()}
    new_tag = agent2.event_manager.add(EventBase())
    assert new_tag not in existing_tags, (
        f"new event assigned tag {new_tag!r} which already exists in the backend"
    )

    storage2.close()


def _make_sqlite_backend() -> SQLiteEventBackend:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE events (tag TEXT PRIMARY KEY, event_type TEXT, data TEXT, "
        "status TEXT DEFAULT 'active', insertion_order INTEGER)"
    )
    conn.execute("CREATE TABLE active_tags (tag TEXT PRIMARY KEY, position INTEGER)")
    return SQLiteEventBackend(conn)


def _insert_raw_tag(backend: SQLiteEventBackend, tag: str) -> None:
    """Insert a bare tag row directly — bypasses EventManager for unit-testing max_tag_num."""
    backend._conn.execute(
        "INSERT INTO events (tag, event_type, data, insertion_order) VALUES (?, 'test', '{}', 0)",
        (tag,),
    )


def test_sqlite_max_tag_num_multi_digit_range_tags():
    """SQLiteEventBackend.max_tag_num() must correctly parse multi-digit range tags.

    The SQL uses instr(tag, '..') + 2 to skip past '..'. Verify this is correct
    for tags where both sides are multiple digits (e.g. "12..234", "233..293432").
    """
    backend = _make_sqlite_backend()
    _insert_raw_tag(backend, "12..234")
    _insert_raw_tag(backend, "233..293432")
    _insert_raw_tag(backend, "500")
    assert backend.max_tag_num() == 293432


def test_sqlite_max_tag_num_empty():
    backend = _make_sqlite_backend()
    assert backend.max_tag_num() == 0


def test_sqlite_max_tag_num_simple_tags():
    backend = _make_sqlite_backend()
    _insert_raw_tag(backend, "1")
    _insert_raw_tag(backend, "99")
    _insert_raw_tag(backend, "42")
    assert backend.max_tag_num() == 99


def test_sqlite_direct_store_keeps_tag_counter_coherent(tmp_path):
    """SQLite-side parity for ``test_direct_store_keeps_tag_counter_coherent``.

    The store-coherence fix lives in both backends; this exercises the
    SQLite branch through a real ``SQLiteStorageManager`` so a regression
    in either implementation gets caught.
    """
    storage = SQLiteStorageManager(tmp_path / "session.db")
    em = _SimpleAgent(storage=storage).event_manager
    em.add(EventBase())  # tag "1"

    pre_existing = EventBase()
    pre_existing.tag = "99"
    storage.event_backend.store("99", pre_existing)

    new_tag = em.add(EventBase())
    assert new_tag == "100"
    storage.close()


def test_sqlite_multiple_managers_share_backend_without_tag_collision(tmp_path):
    """SQLite-side parity for ``test_multiple_managers_share_backend_without_tag_collision``.

    Mirrors the InMemory test on a real SQLite backend — covers the
    TUI's runtime configuration where the agent's stable EventManager
    and the SessionManager's thin EventManager both write through the
    same storage's backend.
    """
    from nooa.runtime.event_manager import EventManager

    storage = SQLiteStorageManager(tmp_path / "session.db")
    em1 = EventManager(backend=storage.event_backend)
    em2 = EventManager(backend=storage.event_backend)

    tags = [
        em1.add(EventBase()),
        em2.add(EventBase()),
        em1.add(EventBase()),
        em2.add(EventBase()),
    ]
    assert tags == ["1", "2", "3", "4"]
    assert len(set(tags)) == len(tags)
    storage.close()
