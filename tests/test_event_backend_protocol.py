# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EventBackend protocol tests — parametrized over all registered backends.

Every EventBackend implementation must pass all tests in this file.

To add a new backend: add a branch to the ``backend`` fixture below.

These tests cover:
- Basic CRUD (store / get / remove / update)
- Active-tag ordering and archiving
- find_tag() and remove_active_tag() return values
- insert_active_tag() edge cases (start, middle, end)
- all_events() insertion-order iteration
- max_tag_num() across simple and range tags
- Event-type roundtrip: events must deserialize as their concrete types,
  not fall back to Metadata. (For in-memory this is trivial; for serializing
  backends like SQLite it requires a complete _CORE_TYPES registry.)
"""

import pytest

from nooa.context_blocks import EventStatus
from nooa.context_blocks.events import (
    AssistantEvent,
    ResultStatus,
    ToolCallEvent,
    ToolResult,
    UserEvent,
)
from nooa.context_blocks.models import Role
from nooa.events import LLMOutput, Task
from nooa.runtime.event_backend import InMemoryBackend
from nooa.storage.sqlite import SQLiteEventBackend

# ---------------------------------------------------------------------------
# Backend fixture — add new backends here
# ---------------------------------------------------------------------------


@pytest.fixture(params=["memory", "sqlite"])
def backend(request, sqlite_conn):
    """Parametrized EventBackend fixture.

    To test a new backend implementation, add a branch here:

        elif request.param == "mybackend":
            yield MyBackend()
    """
    if request.param == "memory":
        return InMemoryBackend()
    elif request.param == "sqlite":
        return SQLiteEventBackend(sqlite_conn)
    else:
        raise ValueError(f"Unknown backend param: {request.param!r}")


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------


def test_store_and_get(backend):
    """store() + get() round-trip returns the same event content."""
    ev = UserEvent(content="hello")
    backend.store("1", ev)

    result = backend.get("1")
    assert result is not None
    assert result.content == "hello"


def test_get_missing_tag_returns_none(backend):
    backend.store("1", UserEvent(content="x"))
    assert backend.get("99") is None


def test_get_by_id(backend):
    ev = UserEvent(content="find me by id")
    backend.store("1", ev)

    result = backend.get_by_id(ev.id)
    assert result is not None
    assert result.content == "find me by id"


def test_get_by_id_missing_returns_none(backend):
    assert backend.get_by_id("no-such-id") is None


def test_remove_returns_true_on_success(backend):
    backend.store("1", UserEvent(content="to remove"))
    assert backend.remove("1") is True
    assert backend.get("1") is None


def test_remove_returns_false_when_missing(backend):
    assert backend.remove("nonexistent") is False


def test_remove_also_removes_from_active_tags(backend):
    backend.store("1", UserEvent(content="a"))
    backend.store("2", UserEvent(content="b"))
    backend.remove("1")
    assert "1" not in backend.active_tags()
    assert "2" in backend.active_tags()


def test_update_modifies_field(backend):
    ev = UserEvent(content="before")
    backend.store("1", ev)

    assert backend.update("1", content="after") is True
    result = backend.get("1")
    assert result.content == "after"


def test_update_returns_false_when_missing(backend):
    assert backend.update("99", content="x") is False


def test_update_merges_metadata(backend):
    ev = UserEvent(content="x")
    ev.metadata["existing"] = "value"
    backend.store("1", ev)

    backend.update("1", metadata={"new_key": "new_value"})
    result = backend.get("1")
    assert result.metadata.get("existing") == "value"
    assert result.metadata.get("new_key") == "new_value"


def test_update_status_field_modifies_event(backend):
    """update(status=...) must change the status field on the stored event."""
    backend.store("1", UserEvent(content="a"))
    backend.update("1", status=EventStatus.ARCHIVED)

    result = backend.get("1")
    assert result.status == EventStatus.ARCHIVED


def test_update_status_does_not_remove_from_active_tags(backend):
    """update(status=...) changes the field only — it does NOT modify active_tags.

    Archiving via the EventManager workflow requires both set_status() AND
    remove_active_tag(). update() is a generic field setter; it should not
    silently alter the active-tag list.
    """
    backend.store("1", UserEvent(content="a"))
    backend.store("2", UserEvent(content="b"))
    backend.update("1", status=EventStatus.ARCHIVED)

    assert "1" in backend.active_tags()  # tag still active — only field changed
    assert "2" in backend.active_tags()


def test_len_counts_all_events(backend):
    assert len(backend) == 0
    backend.store("1", UserEvent(content="a"))
    backend.store("2", UserEvent(content="b"))
    assert len(backend) == 2


def test_len_counts_archived_events(backend):
    backend.store("1", UserEvent(content="a"))
    backend.set_status("1", EventStatus.ARCHIVED)
    assert len(backend) == 1  # archived still counts


def test_clear_removes_everything(backend):
    backend.store("1", UserEvent(content="a"))
    backend.store("2", UserEvent(content="b"))
    backend.clear()
    assert len(backend) == 0
    assert backend.active_tags() == []
    assert list(backend.all_events()) == []


# ---------------------------------------------------------------------------
# Active-tag ordering and archiving
# ---------------------------------------------------------------------------


def test_active_tags_insertion_order(backend):
    backend.store("1", UserEvent(content="a"))
    backend.store("2", AssistantEvent(content="b"))
    backend.store("3", UserEvent(content="c"))
    assert backend.active_tags() == ["1", "2", "3"]


def test_set_status_archived_does_not_remove_from_active_tags(backend):
    """set_status() only flips the status field.

    Removing from active_tags is a separate step done by collapse() via
    remove_active_tag(). set_status() alone must NOT silently alter the
    active-tag list.
    """
    backend.store("1", UserEvent(content="a"))
    backend.store("2", UserEvent(content="b"))
    backend.set_status("1", EventStatus.ARCHIVED)

    tags = backend.active_tags()
    assert "1" in tags  # tag still active — removal is the caller's responsibility
    assert "2" in tags


def test_set_status_archived_preserves_event_in_storage(backend):
    ev = UserEvent(content="archived but stored")
    backend.store("1", ev)
    backend.set_status("1", EventStatus.ARCHIVED)

    result = backend.get("1")
    assert result is not None
    assert result.status == EventStatus.ARCHIVED


def test_set_status_returns_false_when_missing(backend):
    assert backend.set_status("99", EventStatus.ARCHIVED) is False


def test_remove_active_tag_returns_true_when_present(backend):
    backend.store("1", UserEvent(content="a"))
    assert backend.remove_active_tag("1") is True


def test_remove_active_tag_returns_false_when_missing(backend):
    assert backend.remove_active_tag("nonexistent") is False


def test_remove_active_tag_leaves_event_in_storage(backend):
    ev = UserEvent(content="stays in storage")
    backend.store("1", ev)
    backend.remove_active_tag("1")

    assert "1" not in backend.active_tags()
    assert backend.get("1") is not None  # still retrievable


def test_insert_active_tag_at_middle(backend):
    """Insert at position 1 (between two existing tags)."""
    backend.store("1", UserEvent(content="a"))
    backend.store("3", UserEvent(content="c"))
    backend.store("2", UserEvent(content="b"))
    backend.remove_active_tag("2")
    backend.insert_active_tag("2", 1)
    assert backend.active_tags() == ["1", "2", "3"]


def test_insert_active_tag_at_start(backend):
    """Insert at position 0 (prepend)."""
    backend.store("2", UserEvent(content="b"))
    backend.store("3", UserEvent(content="c"))
    backend.store("1", UserEvent(content="a"))
    backend.remove_active_tag("1")
    backend.insert_active_tag("1", 0)
    assert backend.active_tags() == ["1", "2", "3"]


def test_insert_active_tag_at_end(backend):
    """Insert at position len(active_tags) (append)."""
    backend.store("1", UserEvent(content="a"))
    backend.store("2", UserEvent(content="b"))
    backend.store("3", UserEvent(content="c"))
    backend.remove_active_tag("3")
    backend.insert_active_tag("3", 2)  # index 2 = end of ["1", "2"]
    assert backend.active_tags() == ["1", "2", "3"]


# ---------------------------------------------------------------------------
# find_tag()
# ---------------------------------------------------------------------------


def test_find_tag_returns_correct_tag(backend):
    ev = UserEvent(content="findable")
    backend.store("42", ev)

    assert backend.find_tag(ev) == "42"


def test_find_tag_returns_none_for_unstored_event(backend):
    ev = UserEvent(content="never stored")
    assert backend.find_tag(ev) is None


def test_find_tag_returns_none_after_remove(backend):
    ev = UserEvent(content="stored then removed")
    backend.store("1", ev)
    backend.remove("1")
    assert backend.find_tag(ev) is None


# ---------------------------------------------------------------------------
# all_events() — insertion order, includes archived
# ---------------------------------------------------------------------------


def test_all_events_insertion_order(backend):
    backend.store("1", UserEvent(content="first"))
    backend.store("2", AssistantEvent(content="second"))
    backend.store("3", UserEvent(content="third"))

    events = list(backend.all_events())
    assert len(events) == 3
    assert events[0].content == "first"
    assert events[1].content == "second"
    assert events[2].content == "third"


def test_all_events_includes_archived(backend):
    backend.store("1", UserEvent(content="active"))
    backend.store("2", UserEvent(content="archived"))
    backend.set_status("2", EventStatus.ARCHIVED)

    events = list(backend.all_events())
    assert len(events) == 2


def test_all_events_empty(backend):
    assert list(backend.all_events()) == []


# ---------------------------------------------------------------------------
# Event-type roundtrip — concrete types, not Metadata fallback
# ---------------------------------------------------------------------------


def test_user_event_type_preserved(backend):
    """UserEvent must come back as UserEvent, not Metadata."""
    backend.store("1", UserEvent(content="hello user"))
    result = backend.get("1")
    assert type(result) is UserEvent, f"Expected UserEvent, got {type(result).__name__}"
    assert result.content == "hello user"


def test_assistant_event_type_preserved(backend):
    backend.store("1", AssistantEvent(content="hello back"))
    result = backend.get("1")
    assert type(result) is AssistantEvent, f"Expected AssistantEvent, got {type(result).__name__}"
    assert result.content == "hello back"


def test_tool_call_event_type_preserved(backend):
    ev = ToolCallEvent(tool_call_id="tc1", name="my_tool", arguments={"x": 1})
    backend.store("1", ev)
    result = backend.get("1")
    assert type(result) is ToolCallEvent, f"Expected ToolCallEvent, got {type(result).__name__}"
    assert result.name == "my_tool"
    assert result.arguments == {"x": 1}


def test_tool_call_with_result_type_preserved(backend):
    """Nested ToolResult must survive roundtrip field-for-field."""
    ev = ToolCallEvent(
        tool_call_id="tc2",
        name="run_code",
        arguments={"code": "print(1)"},
        result=ToolResult(
            tool_call_id="tc2",
            content="1\n",
            result_status=ResultStatus.COMPLETE,
        ),
    )
    backend.store("1", ev)
    result = backend.get("1")

    assert type(result) is ToolCallEvent
    assert isinstance(result.result, ToolResult)
    assert result.result.content == "1\n"
    assert result.result.result_status == ResultStatus.COMPLETE


def test_nemo_event_type_preserved(backend):
    """nooa event types (Task, LLMOutput, etc.) must also round-trip."""
    backend.store("1", Task(prompt="do the thing"))
    backend.store("2", LLMOutput(content="done"))
    events = list(backend.all_events())
    assert type(events[0]) is Task
    assert type(events[1]) is LLMOutput


def test_context_blocks_roles_correct_after_roundtrip(backend):
    """_role must reflect the concrete type, not the Metadata fallback Role.METADATA."""
    backend.store("1", UserEvent(content="u"))
    backend.store("2", AssistantEvent(content="a"))
    backend.store("3", ToolCallEvent(tool_call_id="tc", name="t", arguments={}))

    results = list(backend.all_events())
    assert results[0]._role == Role.USER
    assert results[1]._role == Role.ASSISTANT
    assert results[2]._role == Role.ASSISTANT


def test_archived_status_preserved_after_roundtrip(backend):
    """Archived events must come back with status=ARCHIVED."""
    backend.store("1", UserEvent(content="old message"))
    backend.set_status("1", EventStatus.ARCHIVED)

    result = backend.get("1")
    assert type(result) is UserEvent
    assert result.status == EventStatus.ARCHIVED
