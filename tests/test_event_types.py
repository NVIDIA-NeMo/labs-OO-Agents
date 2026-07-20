# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Event types used in event pipeline."""

from nooa.events import (
    Error,
    Feedback,
    LLMOutput,
    Message,
    Reasoning,
    Task,
)
from nooa.runtime.event_manager import EventManager


class TestEventTypes:
    """Test Event type definitions and serialization."""

    def test_task_event(self):
        """Task has event_type='task' and prompt."""
        event = Task(prompt="Do something")
        assert event.event_type == "Task"
        assert event.prompt == "Do something"

    def test_message_event(self):
        """Message for user-facing messages."""
        event = Message(content="Hello!")
        assert event.event_type == "Message"
        assert event.content == "Hello!"

    def test_reasoning_event(self):
        """Reasoning for chain-of-thought."""
        event = Reasoning(content="Let me think...")
        assert event.event_type == "Reasoning"
        assert event.content == "Let me think..."

    def test_error_event(self):
        """Error for execution errors."""
        event = Error(content="SyntaxError: invalid")
        assert event.event_type == "Error"
        assert event.content == "SyntaxError: invalid"

    def test_feedback_event(self):
        """Feedback for execution feedback."""
        event = Feedback(content="Code executed. Output: 42")
        assert event.event_type == "Feedback"
        assert event.content == "Code executed. Output: 42"

    def test_llm_output_event(self):
        """LLMOutput for LLM responses."""
        event = LLMOutput(content="def foo(): pass")
        assert event.event_type == "LLMOutput"
        assert event.content == "def foo(): pass"

    def test_event_serialization(self):
        """Events serialize to JSON with discriminator."""
        event = Task(prompt="test")
        d = event.model_dump()
        assert d["event_type"] == "Task"
        assert d["prompt"] == "test"

    def test_event_has_metadata(self):
        """Events can have metadata set after construction."""
        event = Task(prompt="test")
        event.metadata["call_id"] = "abc123"
        event.metadata["parent_id"] = "xyz789"
        assert event.metadata["call_id"] == "abc123"
        assert event.metadata["parent_id"] == "xyz789"

    def test_event_has_timestamp(self):
        """Events have auto-generated timestamp."""
        from datetime import datetime

        event = Task(prompt="test")
        assert isinstance(event.timestamp, datetime)


class TestBackwardCompatAliases:
    """Test that old event names still work."""

    def test_task_event_alias(self):
        event = Task(prompt="test")
        assert event.event_type == "Task"

    def test_message_event_alias(self):
        event = Message(content="test")
        assert event.event_type == "Message"

    def test_reasoning_event_alias(self):
        event = Reasoning(content="test")
        assert event.event_type == "Reasoning"

    def test_error_event_alias(self):
        event = Error(content="test")
        assert event.event_type == "Error"

    def test_feedback_event_alias(self):
        event = Feedback(content="test")
        assert event.event_type == "Feedback"

    def test_assistant_event_alias(self):
        event = LLMOutput(content="test")
        assert event.event_type == "LLMOutput"


class TestEventManagerEventAPI:
    """Test EventManager with Event objects."""

    def test_add_task_event(self):
        """add() accepts Task."""
        em = EventManager()
        event = Task(prompt="Do something")

        em.add(event)

        assert len(em) == 1
        assert em.values()[0].prompt == "Do something"
        assert em.values()[0].event_type == "Task"

    def test_add_llm_output_event(self):
        """add() accepts LLMOutput."""
        em = EventManager()
        event = LLMOutput(content="def foo(): pass")

        em.add(event)

        assert len(em) == 1
        assert em.values()[0].content == "def foo(): pass"
        assert em.values()[0].event_type == "LLMOutput"

    def test_add_error_event(self):
        """add() accepts Error."""
        em = EventManager()
        event = Error(content="SyntaxError: invalid")

        em.add(event)

        assert len(em) == 1
        assert "SyntaxError" in em.values()[0].content
        assert em.values()[0].event_type == "Error"

    def test_add_feedback_event(self):
        """add() accepts Feedback."""
        em = EventManager()
        event = Feedback(content="Code executed. Output: 42")

        em.add(event)

        assert len(em) == 1
        assert "Output: 42" in em.values()[0].content
        assert em.values()[0].event_type == "Feedback"

    def test_add_message_event(self):
        """add() accepts Message (user-facing message)."""
        em = EventManager()
        event = Message(content="Hello!")

        em.add(event)

        assert len(em) == 1
        assert em.values()[0].content == "Hello!"
        assert em.values()[0].event_type == "Message"

    def test_add_with_record_false(self):
        """add(event, record=False) doesn't store in event manager."""
        em = EventManager()
        event = Reasoning(content="Let me think...")

        em.add(event, record=False)

        assert len(em) == 0

    def test_add_with_metadata_call_id(self):
        """Event metadata can be set after construction."""
        em = EventManager()
        event = Task(prompt="test")
        event.metadata["call_id"] = "abc123"

        em.add(event)

        assert em.values()[0].metadata.get("call_id") == "abc123"


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
