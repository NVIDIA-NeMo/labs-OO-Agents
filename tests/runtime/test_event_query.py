# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for EventQuery - event filtering configuration."""

from nooa.events import Task
from nooa.runtime.event_query import EventQuery


class TestEventQueryApply:
    """Tests for EventQuery.apply() filtering."""

    def test_call_id_filters_by_metadata(self):
        """apply(call_id=...) filters by event.metadata['call_id'], not attribute.

        Events store call_id in metadata (set by EventManager.add()), not as
        a top-level attribute. EventQuery must use metadata for filtering.
        """
        t1 = Task(prompt="Task 1")
        t1.metadata["call_id"] = "call-1"
        t2 = Task(prompt="Task 2")
        t2.metadata["call_id"] = "call-2"
        t3 = Task(prompt="Task 3")
        t3.metadata["call_id"] = "call-1"
        events = [t1, t2, t3]

        result = EventQuery(call_id="call-1").apply(events)
        assert len(result) == 2
        assert result[0].prompt == "Task 1"
        assert result[1].prompt == "Task 3"

        result = EventQuery(call_id="call-2").apply(events)
        assert len(result) == 1
        assert result[0].prompt == "Task 2"

    def test_current_resolves_to_current_call_id(self):
        """apply(call_id='current', current_call_id=...) filters by current call."""
        t1 = Task(prompt="A")
        t1.metadata["call_id"] = "current-call-id"
        t2 = Task(prompt="B")
        t2.metadata["call_id"] = "other-call"
        events = [t1, t2]

        result = EventQuery(call_id="current").apply(events, current_call_id="current-call-id")
        assert len(result) == 1
        assert result[0].prompt == "A"

    def test_events_without_call_id_metadata_excluded(self):
        """Events with no metadata['call_id'] are excluded when filtering by call_id."""
        t1 = Task(prompt="No call_id")
        # t1.metadata has no "call_id"
        t2 = Task(prompt="With call_id")
        t2.metadata["call_id"] = "call-1"
        events = [t1, t2]

        result = EventQuery(call_id="call-1").apply(events)
        assert len(result) == 1
        assert result[0].prompt == "With call_id"

    def test_current_call_keeps_task_message(self):
        """EventQuery.current_call() must keep the task (user) message for this call.

        When filtering by current call_id, we must have at least system + task
        so the LLM receives the task prompt. This test ensures the task event
        (with matching metadata['call_id']) is kept.
        """
        current_id = "call-abc-123"
        task = Task(prompt="Classify the sentiment of: Hello world")
        task.metadata["call_id"] = current_id
        task.tag = "1"
        other = Task(prompt="Other call task")
        other.metadata["call_id"] = "other-call"
        other.tag = "2"
        events = [task, other]

        result = EventQuery(call_id="current").apply(events, current_call_id=current_id)
        assert len(result) >= 1, "must keep at least the task for this call"
        task_prompts = [e.prompt for e in result if isinstance(e, Task)]
        assert "Classify the sentiment" in task_prompts[0]
