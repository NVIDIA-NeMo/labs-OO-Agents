# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Event types used in history pipeline."""

from nooa import Agent
from nooa.context_blocks import ResultStatus
from nooa.runtime.events import EventsApi
from nooa.unifiedllm import FakeLLMClient

_LLM = FakeLLMClient()


class _TestAgent(Agent, llm=_LLM):
    pass


class TestEventTypes:
    """Test Event type definitions and serialization."""

    def test_task_event(self):
        """Task has event_type='task' and prompt."""
        from nooa.events import Task

        event = Task(prompt="Do something")
        assert event.event_type == "Task"
        assert event.prompt == "Do something"
        # Backward compat alias
        event2 = Task(prompt="Test")
        assert event2.event_type == "Task"

    def test_message_event(self):
        """Message for user-facing messages."""
        from nooa.events import Message

        event = Message(content="Hello!")
        assert event.event_type == "Message"
        assert event.content == "Hello!"
        # Backward compat alias
        event2 = Message(content="Test")
        assert event2.event_type == "Message"

    def test_reasoning_event(self):
        """Reasoning for chain-of-thought."""
        from nooa.events import Reasoning

        event = Reasoning(content="Let me think...")
        assert event.event_type == "Reasoning"
        assert event.content == "Let me think..."
        # Backward compat alias
        event2 = Reasoning(content="Test")
        assert event2.event_type == "Reasoning"

    def test_error_event(self):
        """Error for execution errors."""
        from nooa.events import Error

        event = Error(content="SyntaxError: invalid")
        assert event.event_type == "Error"
        assert event.content == "SyntaxError: invalid"
        # Backward compat alias
        event2 = Error(content="Test")
        assert event2.event_type == "Error"

    def test_feedback_event(self):
        """Feedback for execution feedback."""
        from nooa.events import Feedback

        event = Feedback(content="Code executed. Output: 42")
        assert event.event_type == "Feedback"
        assert event.content == "Code executed. Output: 42"
        # Backward compat alias
        event2 = Feedback(content="Test")
        assert event2.event_type == "Feedback"

    def test_llm_output_event(self):
        """LLMOutput for LLM responses."""
        from nooa.events import LLMOutput

        event = LLMOutput(content="def foo(): pass")
        assert event.event_type == "LLMOutput"
        assert event.content == "def foo(): pass"
        # Verify event_type
        event2 = LLMOutput(content="Test")
        assert event2.event_type == "LLMOutput"

    def test_tag_property_returns_event_position(self):
        """tag property returns event position (set by EventManager)."""
        from nooa.events import Task

        event = Task(prompt="test")
        # Not yet in event manager
        assert event.tag is None
        # After assignment
        event.tag = "5"
        assert event.tag == "5"

    def test_event_serialization(self):
        """Events serialize to JSON with discriminator."""
        from nooa.events import Task

        event = Task(prompt="test")
        d = event.model_dump()
        assert d["event_type"] == "Task"
        assert d["prompt"] == "test"

    def test_event_has_metadata(self):
        """Events can have metadata set after construction."""
        from nooa.events import Task

        event = Task(prompt="test")
        event.metadata["call_id"] = "abc123"
        event.metadata["parent_id"] = "xyz789"
        assert event.metadata["call_id"] == "abc123"
        assert event.metadata["parent_id"] == "xyz789"

    def test_event_has_timestamp(self):
        """Events have auto-generated timestamp."""
        from datetime import datetime

        from nooa.events import Task

        event = Task(prompt="test")
        assert isinstance(event.timestamp, datetime)


class TestExecutionResult:
    """Test ExecutionResult model."""

    def test_execution_result_success(self):
        """ExecutionResult for successful execution."""
        from nooa.events import ExecutionResult

        result = ExecutionResult(
            stdout="Hello\n",
            defined_methods={"foo": lambda: 42},
        )
        assert result.stdout == "Hello\n"
        assert result.success is True
        assert result.error is None
        assert "foo" in result.defined_methods

    def test_execution_result_error(self):
        """ExecutionResult with error."""
        from nooa.events import ExecutionResult

        result = ExecutionResult(
            stdout="",
            error=SyntaxError("invalid syntax"),
        )
        assert result.success is False
        assert isinstance(result.error, SyntaxError)

    def test_has_method(self):
        """has_method() checks if method was defined."""
        from nooa.events import ExecutionResult

        result = ExecutionResult(
            stdout="",
            defined_methods={"process": lambda x: x * 2},
        )
        assert result.has_method("process") is True
        assert result.has_method("other") is False


class TestPythonOutputEvent:
    """Test PythonOutput pformat rendering (was PythonOutput)."""

    def test_event_type_excluded_from_repr(self):
        """event_type field should not appear in repr (repr=False)."""
        from nooa.events import PythonOutput

        event = PythonOutput(
            tool_call_id="call_123",
            execution_count=1,
            stdout="hello world\n",
            execution_status=ResultStatus.COMPLETE,
        )
        repr_str = repr(event)
        assert "event_type=" not in repr_str
        assert "tool_call_id=" in repr_str
        assert "stdout=" in repr_str

    def test_public_fields_in_repr(self):
        """LLM-visible fields appear in repr; infrastructure fields (repr=False) do not."""
        from nooa.events import PythonOutput

        event = PythonOutput(
            tool_call_id="call_123",
            execution_count=1,
            stdout="output",
            stderr="warning",
            error="error msg",
            value=42,
            execution_status=ResultStatus.COMPLETE,
        )
        repr_str = repr(event)
        assert "tool_call_id=" in repr_str
        assert "execution_count=" not in repr_str  # repr=False — infrastructure field
        assert "stdout=" in repr_str
        assert "stderr=" in repr_str
        assert "error=" in repr_str
        assert "value=" in repr_str
        assert "status=" in repr_str

    def test_backward_compat_alias(self):
        """PythonOutput alias still works."""
        from nooa.events import PythonOutput

        event = PythonOutput(
            tool_call_id="call_123",
            execution_count=1,
            stdout="hello world\n",
            execution_status=ResultStatus.COMPLETE,
        )
        assert isinstance(event, PythonOutput)
        assert event.event_type == "PythonOutput"


class TestSummaryEvent:
    """Test Summary event with tag and children properties."""

    def test_summary_tag_returns_summary_tag(self):
        """Summary.tag is set by EventManager (same as regular events)."""
        from nooa.events import Summary

        summary = Summary(
            summary_tag="2..40",
            replaced_range=(2, 40),
            children_tags=["2", "3", "4"],
        )
        assert summary.event_type == "Summary"
        # tag is None until EventManager sets it
        assert summary.tag is None
        # EventManager sets tag = summary_tag
        summary.tag = summary.summary_tag
        assert summary.tag == "2..40"

    def test_summary_children_tags_field(self):
        """Summary.children_tags returns the list of child tags."""
        from nooa.events import Summary

        summary = Summary(
            summary_tag="2..40",
            replaced_range=(2, 40),
            children_tags=["2", "3", "4"],
        )
        assert summary.children_tags == ["2", "3", "4"]

    def test_summary_with_summary_text(self):
        """Summary with summary_text represents summarization (not truncation)."""
        from nooa.events import Summary

        summary = Summary(
            summary_tag="5..10",
            replaced_range=(5, 10),
            children_tags=["5", "6", "7", "8", "9", "10"],
            summary_text="The agent explored the database schema and wrote queries.",
        )
        assert summary.summary_text == "The agent explored the database schema and wrote queries."
        assert summary.replaced_range == (5, 10)
        assert len(summary.children_tags) == 6

    def test_summary_without_summary_text_is_truncation(self):
        """Summary without summary_text represents truncation."""
        from nooa.events import Summary

        summary = Summary(
            summary_tag="1..5",
            replaced_range=(1, 5),
        )
        assert summary.summary_text is None
        assert summary.children_tags == []

    def test_summary_serialization(self):
        """Summary serializes to dict with all fields."""
        from nooa.events import Summary

        summary = Summary(
            summary_tag="2..4",
            replaced_range=(2, 4),
            children_tags=["2", "3", "4"],
            summary_text="A brief summary.",
            doc='Use self.events["2..4"].children_tags',
        )
        d = summary.model_dump()
        assert d["event_type"] == "Summary"
        assert d["summary_tag"] == "2..4"
        assert d["replaced_range"] == (2, 4)
        assert d["children_tags"] == ["2", "3", "4"]
        assert d["summary_text"] == "A brief summary."

    def test_summary_role_is_assistant(self):
        """Summary has assistant role (LLM's own recap)."""
        from nooa.context_blocks.models import Role
        from nooa.events import Summary

        summary = Summary(
            summary_tag="1..3",
            replaced_range=(1, 3),
        )
        assert summary._role == Role.ASSISTANT


class TestEventViewContains:
    """Test EventView.__contains__ (the 'in' operator for agent-facing API)."""

    def test_contains_by_tag(self):
        """'tag in events' returns True for existing tags."""
        from nooa.events import Task

        agent = _TestAgent()
        agent.event_manager.add(Task(prompt="First"))
        agent.event_manager.add(Task(prompt="Second"))
        events = EventsApi(agent)

        assert "1" in events
        assert "2" in events
        assert "3" not in events

    def test_contains_by_uuid(self):
        """'uuid in events' returns True for existing UUIDs."""
        from nooa.events import Task

        agent = _TestAgent()
        event = Task(prompt="Test")
        agent.event_manager.add(event)
        events = EventsApi(agent)

        assert event.id in events
        assert "nonexistent-uuid" not in events

    def test_contains_after_collapse(self):
        """'summary_tag in events' returns True after collapse."""
        from nooa.events import Task

        agent = _TestAgent()
        for i in range(4):
            agent.event_manager.add(Task(prompt=f"Event {i + 1}"))

        agent.event_manager.collapse("1", "3", summary_text="Summary")
        events = EventsApi(agent)

        assert "1..3" in events
        assert "4" in events

    def test_event_view_repr(self):
        """EventsApi.__repr__ shows active event count."""
        from nooa.events import Task

        agent = _TestAgent()
        agent.event_manager.add(Task(prompt="One"))
        agent.event_manager.add(Task(prompt="Two"))
        events = EventsApi(agent)

        assert "EventsApi(2 active)" in repr(events)
