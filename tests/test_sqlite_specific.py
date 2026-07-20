# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SQLite-specific tests: deserialization, registry, and _CONTEXT_BLOCKS_TYPES guards.

These tests cover behavior that only applies to SQLiteEventBackend:
- _CONTEXT_BLOCKS_TYPES sanity checks (auto-derived from the Event union)
- _deserialize() fallback to Metadata for unknown event types
- register_event_type() custom subclass roundtrip
- register_event_type() overwrite warning
"""

import logging
from typing import Literal

from nooa.context_blocks import EventBase, Metadata
from nooa.context_blocks.events import AssistantEvent, ToolCallEvent, UserEvent
from nooa.storage.sqlite import _CONTEXT_BLOCKS_TYPES, SQLiteEventBackend

# ---------------------------------------------------------------------------
# _CONTEXT_BLOCKS_TYPES sanity
# ---------------------------------------------------------------------------


def test_context_blocks_types_is_not_empty():
    """_CONTEXT_BLOCKS_TYPES must contain at least one type."""
    assert len(_CONTEXT_BLOCKS_TYPES) > 0, "_CONTEXT_BLOCKS_TYPES is empty"


def test_context_blocks_types_contains_expected_members():
    """_CONTEXT_BLOCKS_TYPES must contain exactly the types in the Event union."""
    assert set(_CONTEXT_BLOCKS_TYPES) == {UserEvent, AssistantEvent, ToolCallEvent}


def test_context_blocks_types_all_are_event_base_subclasses():
    """Every member of _CONTEXT_BLOCKS_TYPES must be an EventBase subclass."""
    for cls in _CONTEXT_BLOCKS_TYPES:
        assert issubclass(cls, EventBase), (
            f"{cls.__name__} is in _CONTEXT_BLOCKS_TYPES but is not an EventBase subclass"
        )


# ---------------------------------------------------------------------------
# _deserialize() fallback for unknown event types
# ---------------------------------------------------------------------------


def test_deserialize_unknown_event_type_falls_back_to_metadata(sqlite_conn):
    """An event_type not in the registry must deserialize as Metadata, not raise."""
    import json

    backend = SQLiteEventBackend(sqlite_conn)

    # Insert a row with an unrecognised event_type directly
    unknown_json = json.dumps(
        {
            "event_type": "totally_unknown_type",
            "id": "abc-123",
            "metadata": {},
            "status": "active",
            "tag": "1",
            "timestamp": "2025-01-01T00:00:00",
        }
    )
    sqlite_conn.execute(
        "INSERT INTO events (tag, event_id, event_type, status, data, insertion_order) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("1", "abc-123", "totally_unknown_type", "active", unknown_json, 0),
    )
    sqlite_conn.execute("INSERT INTO active_tags (position, tag) VALUES (?, ?)", (0, "1"))
    sqlite_conn.commit()

    result = backend.get("1")
    assert isinstance(result, Metadata), (
        f"Unknown event type must fall back to Metadata, got {type(result).__name__}"
    )


def test_deserialize_unknown_type_logs_warning(sqlite_conn, caplog):
    """_deserialize() must log a warning when falling back to Metadata."""
    import json

    backend = SQLiteEventBackend(sqlite_conn)

    unknown_json = json.dumps(
        {
            "event_type": "mystery_event",
            "id": "xyz-999",
            "metadata": {},
            "status": "active",
            "tag": "1",
            "timestamp": "2025-01-01T00:00:00",
        }
    )
    sqlite_conn.execute(
        "INSERT INTO events (tag, event_id, event_type, status, data, insertion_order) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("1", "xyz-999", "mystery_event", "active", unknown_json, 0),
    )
    sqlite_conn.commit()

    with caplog.at_level(logging.WARNING, logger="nooa.storage.sqlite"):
        backend.get("1")

    assert any("mystery_event" in r.message for r in caplog.records), (
        "Expected a warning mentioning the unknown event_type 'mystery_event'"
    )


# ---------------------------------------------------------------------------
# register_event_type() — custom subclass roundtrip
# ---------------------------------------------------------------------------


def test_register_event_type_custom_subclass_roundtrip(sqlite_conn):
    """A registered Metadata subclass must deserialize as that subclass, not plain Metadata."""

    class TUISessionStart(Metadata):
        event_type: Literal["tui_session_start"] = "tui_session_start"
        model: str = ""
        working_dir: str = ""

    backend = SQLiteEventBackend(sqlite_conn)
    backend.register_event_type(TUISessionStart)

    ev = TUISessionStart(model="gpt-4", working_dir="/tmp")
    backend.store("1", ev)

    result = backend.get("1")
    assert type(result) is TUISessionStart, (
        f"Expected TUISessionStart after register_event_type, got {type(result).__name__}"
    )
    assert result.model == "gpt-4"
    assert result.working_dir == "/tmp"


def test_register_event_type_overwrite_logs_warning(sqlite_conn, caplog):
    """Registering a different class under an existing event_type key must log a warning."""

    class V1(Metadata):
        event_type: Literal["versioned_evt"] = "versioned_evt"

    class V2(Metadata):
        event_type: Literal["versioned_evt"] = "versioned_evt"

    backend = SQLiteEventBackend(sqlite_conn)
    backend.register_event_type(V1)

    with caplog.at_level(logging.WARNING, logger="nooa.storage.sqlite"):
        backend.register_event_type(V2)

    assert any("versioned_evt" in r.message for r in caplog.records), (
        "Expected a warning when overwriting an existing event_type key in the registry"
    )
