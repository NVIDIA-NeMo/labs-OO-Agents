# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for event filtering via filter().

Architecture note: ToolResult is now nested inside ToolCallEvent.result,
so there are no separate ToolResultEvent events. PythonOutput is
a user-role message (not a tool result) so it doesn't require pairing
validation - providers don't enforce pairing for user messages.
"""

from nooa.context_blocks import ResultStatus
from nooa.context_blocks.events import ToolCallEvent
from nooa.events import PythonOutput
from nooa.runtime.event_manager import EventManager


def test_filter_with_limit_returns_most_recent_events():
    """Test that filter(limit=N) returns the last N events from events."""
    events = EventManager()

    # Add 100 events
    for i in range(100):
        event = ToolCallEvent(
            tool_call_id=f"tool_{i}",
            name="dummy_tool",
            arguments={},
        )
        events.add(event)

    # Get recent 50 events
    recent_events = events.filter(limit=50)

    # Should get exactly 50 events
    assert len(recent_events) == 50

    # Should be the last 50 (tool_50 through tool_99)
    tool_ids = [e.tool_call_id for e in recent_events]
    assert tool_ids[0] == "tool_50"
    assert tool_ids[-1] == "tool_99"


def test_filter_includes_execute_python_events():
    """Test that filter() includes PythonOutput regardless of tool call presence.

    PythonOutput is a user-role message showing execution output.
    It doesn't need to be paired with its tool call since providers don't
    validate pairing for user messages.
    """
    events = EventManager()

    # Add 49 dummy events (indices 0-48)
    for i in range(49):
        event = ToolCallEvent(
            tool_call_id=f"dummy_{i}",
            name="dummy_tool",
            arguments={},
        )
        events.add(event)

    # Add the tool call that will be truncated (index 49)
    tool_call = ToolCallEvent(
        tool_call_id="tooluse_will_be_cut",
        name="execute_python",
        arguments={"code": "print('test')"},
    )
    events.add(tool_call)

    # Add PythonOutput (index 50)
    exec_event = PythonOutput(
        tool_call_id="tooluse_will_be_cut",
        execution_count=1,
        stdout="test output",
        execution_status=ResultStatus.COMPLETE,
    )
    events.add(exec_event)

    # Add more events after (indices 51-99 = 49 more events)
    for i in range(49):
        event = ToolCallEvent(
            tool_call_id=f"after_{i}",
            name="dummy_tool",
            arguments={},
        )
        events.add(event)

    # Total: 100 events
    # filter(limit=50) returns events[50:100], including the PythonOutput

    recent_events = events.filter(limit=50)

    # Should have exactly 50 events
    assert len(recent_events) == 50

    # The PythonOutput should be included (it's at position 0 of the slice)
    assert recent_events[0].event_type == "PythonOutput"
    assert recent_events[0].tool_call_id == "tooluse_will_be_cut"


def test_filter_returns_all_when_under_limit():
    """Test that filter() returns all events when total is under limit."""
    events = EventManager()

    # Add only 10 events
    for i in range(10):
        event = ToolCallEvent(
            tool_call_id=f"tool_{i}",
            name="dummy_tool",
            arguments={},
        )
        events.add(event)

    # Request 50 - should get all 10
    recent_events = events.filter(limit=50)
    assert len(recent_events) == 10


def test_filter_handles_empty_events():
    """Test that filter() handles empty events gracefully."""
    events = EventManager()

    recent_events = events.filter(limit=50)
    assert len(recent_events) == 0
    assert recent_events == []


def test_filter_by_call_id():
    """Test that filter(call_id=...) returns only events for that invocation."""
    from nooa.events import Task

    events = EventManager()

    t1 = Task(prompt="First")
    t1.metadata["call_id"] = "call-1"
    events.add(t1)

    t2 = Task(prompt="Second")
    t2.metadata["call_id"] = "call-2"
    events.add(t2)

    t3 = Task(prompt="Third")
    t3.metadata["call_id"] = "call-1"
    events.add(t3)

    result = events.filter(call_id="call-1")
    assert len(result) == 2
    assert result[0].prompt == "First"
    assert result[1].prompt == "Third"


def test_filter_by_call_id_and_type():
    """Test that call_id and type filters are ANDed together."""
    from nooa.events import LLMOutput, Task

    events = EventManager()

    t1 = Task(prompt="Task for call-1")
    t1.metadata["call_id"] = "call-1"
    events.add(t1)

    llm1 = LLMOutput(content="LLM for call-1")
    llm1.metadata["call_id"] = "call-1"
    events.add(llm1)

    t2 = Task(prompt="Task for call-2")
    t2.metadata["call_id"] = "call-2"
    events.add(t2)

    # Only tasks for call-1
    result = events.filter(call_id="call-1", type="Task")
    assert len(result) == 1
    assert result[0].prompt == "Task for call-1"

    # LLM output for call-1
    result = events.filter(call_id="call-1", type="LLMOutput")
    assert len(result) == 1
    assert result[0].content == "LLM for call-1"


def test_filter_by_call_id_with_limit():
    """Test that call_id filter respects limit parameter."""
    from nooa.events import Task

    events = EventManager()

    for i in range(10):
        t = Task(prompt=f"Event {i}")
        t.metadata["call_id"] = "call-1"
        events.add(t)

    result = events.filter(call_id="call-1", limit=3)
    assert len(result) == 3
    # limit takes most recent
    assert result[0].prompt == "Event 7"
    assert result[2].prompt == "Event 9"
