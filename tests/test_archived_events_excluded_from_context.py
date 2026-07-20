# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests: archived events must not appear in LLM context after collapse().

When EventManager.collapse() archives a range of events, those events must
not show up in the context blocks passed to the LLM.

_phase_events uses event_manager.values() (active_tags only) for all paths —
archived events are represented by the Summary that replaced them in active_tags.
"""

import pytest

from nooa.context_blocks.events import AssistantEvent, ToolCallEvent, UserEvent
from nooa.runtime.context_builder import _phase_events
from nooa.runtime.event_manager import EventManager
from nooa.storage.sqlite import SQLiteEventBackend

# ---------------------------------------------------------------------------
# Parametrized EventManager fixture
# ---------------------------------------------------------------------------


@pytest.fixture(params=["memory", "sqlite"])
def event_manager(request, sqlite_conn):
    """EventManager backed by either InMemory or SQLite."""
    if request.param == "memory":
        return EventManager()
    elif request.param == "sqlite":
        return EventManager(backend=SQLiteEventBackend(sqlite_conn))
    else:
        raise ValueError(f"Unknown backend param: {request.param!r}")


def _phase_event_keys(em) -> set[str]:
    """Return the set of block keys produced by _phase_events (no active query)."""
    return {b.key for b in _phase_events([], em)}


# ---------------------------------------------------------------------------
# No-query path: only active events appear
# ---------------------------------------------------------------------------


def test_archived_context_blocks_events_not_in_context(event_manager):
    """context_blocks events collapsed into a Summary must not appear in context."""
    em = event_manager
    em.add(UserEvent(content="message 1"))  # tag "1"
    em.add(AssistantEvent(content="reply 1"))  # tag "2"
    em.add(UserEvent(content="message 2"))  # tag "3"

    em.collapse("1", "2", summary_text="summarized")

    assert em.keys() == ["1..2", "3"]

    keys = _phase_event_keys(em)

    assert "event_1" not in keys, "Archived UserEvent must not appear in context"
    assert "event_2" not in keys, "Archived AssistantEvent must not appear in context"
    assert "event_1..2" in keys, "Summary must appear in context"
    assert "event_3" in keys, "Active UserEvent must appear in context"


def test_archived_nemo_events_not_in_context(event_manager):
    """nooa events collapsed into a Summary must not appear in context."""
    from nooa.events import LLMOutput, Task

    em = event_manager
    em.add(Task(prompt="do the thing"))  # tag "1"
    em.add(LLMOutput(content="done"))  # tag "2"
    em.add(Task(prompt="do another thing"))  # tag "3"

    em.collapse("1", "2", summary_text="summarized first exchange")

    assert em.keys() == ["1..2", "3"]

    keys = _phase_event_keys(em)

    assert "event_1" not in keys, "Archived tag 1 must not appear in context"
    assert "event_2" not in keys, "Archived tag 2 must not appear in context"
    assert "event_1..2" in keys, "Summary must appear in context"
    assert "event_3" in keys, "Active tag 3 must appear in context"


def test_archived_tool_call_not_in_context(event_manager):
    """Archived ToolCallEvent must not appear in context."""
    em = event_manager
    em.add(ToolCallEvent(tool_call_id="tc1", name="run", arguments={}))  # "1"
    em.add(UserEvent(content="result"))  # "2"
    em.add(UserEvent(content="next step"))  # "3"

    em.collapse("1", "2", summary_text="collapsed tool exchange")

    assert em.keys() == ["1..2", "3"]

    keys = _phase_event_keys(em)

    assert "event_1" not in keys, "Archived ToolCallEvent must not appear in context"
    assert "event_2" not in keys, "Archived UserEvent must not appear in context"
    assert "event_1..2" in keys
    assert "event_3" in keys


def test_only_active_events_in_context_after_multiple_collapses(event_manager):
    """After multiple collapses, only active events appear in context."""
    em = event_manager
    for i in range(6):
        em.add(UserEvent(content=f"msg {i + 1}"))  # tags "1".."6"

    em.collapse("1", "2", summary_text="first summary")
    em.collapse("3", "4", summary_text="second summary")

    assert em.keys() == ["1..2", "3..4", "5", "6"]

    keys = _phase_event_keys(em)

    for archived_tag in ("event_1", "event_2", "event_3", "event_4"):
        assert archived_tag not in keys, f"{archived_tag} is archived and must not appear"

    for active_key in ("event_1..2", "event_3..4", "event_5", "event_6"):
        assert active_key in keys, f"{active_key} is active and must appear"


# ---------------------------------------------------------------------------
# Display order: Summary must appear in its display position, not insertion order
# ---------------------------------------------------------------------------


def test_phase_events_display_order_after_collapse(event_manager):
    """After collapse(), _phase_events returns blocks in display order.

    The Summary must appear WHERE the collapsed events were, not at the end
    of the insertion sequence.
    """
    from nooa.events import Task

    em = event_manager
    em.add(Task(prompt="first"))  # "1"
    em.add(Task(prompt="second"))  # "2"
    em.add(Task(prompt="third"))  # "3"

    em.collapse("1", "2", summary_text="summarized first two")

    assert em.keys() == ["1..2", "3"]

    blocks = _phase_events([], em)
    assert len(blocks) == 2
    assert blocks[0].key == "event_1..2", f"Summary must be first, got {blocks[0].key}"
    assert blocks[1].key == "event_3", f"Active event must be second, got {blocks[1].key}"


# ---------------------------------------------------------------------------
# Active-query path: query operates on active events only
# ---------------------------------------------------------------------------


def test_phase_events_with_type_query_on_real_event_manager(event_manager):
    """EventQuery(type=...) filters correctly — uses values() not filter()."""
    from nooa.events import Error, Task
    from nooa.runtime.event_query import EventQuery

    em = event_manager
    em.add(Task(prompt="do it"))  # "1"
    em.add(Error(content="oops"))  # "2"
    em.add(Task(prompt="retry"))  # "3"

    blocks = _phase_events([], em, agent_event_query=EventQuery(type="Task"))

    assert len(blocks) == 2
    assert all(b.event.event_type == "Task" for b in blocks)


def test_phase_events_query_does_not_include_archived_events(event_manager):
    """Active query must not surface archived events — uses values(), not filter().

    Scenario: tasks 1-2 are collapsed into a Summary; task 3 is active.
    An EventQuery(type="Task") must match only the active Task (tag "3"),
    not the archived originals (tags "1", "2") behind the Summary.
    """
    from nooa.events import Task
    from nooa.runtime.event_query import EventQuery

    em = event_manager
    em.add(Task(prompt="first"))  # "1"
    em.add(Task(prompt="second"))  # "2"
    em.add(Task(prompt="third"))  # "3"

    em.collapse("1", "2", summary_text="summarized first two tasks")

    # active_tags: ["1..2" (Summary), "3" (Task)]; "1" and "2" are archived
    assert em.keys() == ["1..2", "3"]

    blocks = _phase_events([], em, agent_event_query=EventQuery(type="Task"))

    # Only the active Task (tag "3") should match — archived tags "1" and "2" must not
    assert len(blocks) == 1, (
        f"Expected 1 active Task, got {len(blocks)}. "
        "Archived events must not appear even when a query is active."
    )
    assert blocks[0].key == "event_3"
    assert blocks[0].event.prompt == "third"
