# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for context_blocks event types.

Tests the new simplified event model where:
- Private fields (_field) are excluded from repr/pformat
- event_type and _role are class-level attributes for rendering
- tag property returns event position (set by EventManager)
- No render_spec() method
"""

from datetime import datetime


class TestEventBase:
    """Tests for EventBase model."""

    def test_timestamp_has_default(self):
        """EventBase should auto-generate timestamp."""
        from nooa.context_blocks.events import UserEvent

        event = UserEvent(content="test")
        assert event.timestamp is not None
        assert isinstance(event.timestamp, datetime)

    def test_metadata_defaults_to_empty_dict(self):
        """EventBase.metadata should default to empty dict."""
        from nooa.context_blocks.events import UserEvent

        event = UserEvent(content="test")
        assert event.metadata == {}

    def test_metadata_can_be_updated(self):
        """EventBase.metadata should be mutable after construction."""
        from nooa.context_blocks.events import UserEvent

        event = UserEvent(content="test")
        event.metadata["custom_key"] = "value"
        event.metadata["number"] = 42
        assert event.metadata["custom_key"] == "value"
        assert event.metadata["number"] == 42

    def test_tag_defaults_to_none(self):
        """EventBase tag should default to None (unassigned).

        tag is set by EventManager.add() when the event is
        added to the event manager, not at creation time.
        """
        from nooa.context_blocks.events import UserEvent

        event = UserEvent(content="test")
        assert event.tag is None  # Unassigned until added to event manager

    def test_tag_can_be_set(self):
        """EventBase tag should be settable via property."""
        from nooa.context_blocks.events import UserEvent

        event = UserEvent(content="test")
        event.tag = "42"
        assert event.tag == "42"

    def test_tag_on_typed_events(self):
        """All typed events should have tag attribute (defaulting to None)."""
        from nooa.context_blocks.events import (
            AssistantEvent,
            ToolCallEvent,
            UserEvent,
        )

        user = UserEvent(content="Hello")
        assistant = AssistantEvent(content="Hi")
        tool_call = ToolCallEvent(tool_call_id="tc1", name="test", arguments={})

        # All events should have tag (defaulting to None)
        assert user.tag is None
        assert assistant.tag is None
        assert tool_call.tag is None

    def test_event_type_and_role(self):
        """Events should have event_type field and _role class attribute."""
        from nooa.context_blocks.events import AssistantEvent, ToolCallEvent, UserEvent
        from nooa.context_blocks.models import Role

        # event_type is auto-derived from the class name
        assert UserEvent(content="test").event_type == "UserEvent"
        assert AssistantEvent(content="test").event_type == "AssistantEvent"
        assert (
            ToolCallEvent(tool_call_id="tc1", name="test", arguments={}).event_type
            == "ToolCallEvent"
        )

        # _role is a ClassVar
        assert UserEvent._role == Role.USER
        assert AssistantEvent._role == Role.ASSISTANT
        assert ToolCallEvent._role == Role.ASSISTANT

    def test_tag_property_returns_event_position(self):
        """Events should have tag property returning event position."""
        from nooa.context_blocks.events import UserEvent

        # Unassigned events have tag=None
        event = UserEvent(content="test")
        assert event.tag is None

        # After setting tag, it returns the value
        event.tag = "5"
        assert event.tag == "5"


class TestInstanceValuesEmptySuppression:
    """__instance_values__ drops None / empty str / empty list / empty dict fields."""

    def test_empty_fields_dropped(self):
        from pydantic import Field

        from nooa.context_blocks.events import EventBase

        class Sample(EventBase):
            name: str = Field(default="")
            tags: list[str] = Field(default_factory=list)
            extra: dict[str, str] = Field(default_factory=dict)
            coords: tuple[int, ...] = Field(default_factory=tuple)
            seen: set[str] = Field(default_factory=set)
            note: str | None = Field(default=None)

        event = Sample(name="ok")
        values = event.__instance_values__()
        assert "name" in values and values["name"] == "ok"
        assert "tags" not in values
        assert "extra" not in values
        assert "coords" not in values
        assert "seen" not in values
        assert "note" not in values

    def test_zero_and_false_are_kept(self):
        """0 and False are semantically meaningful and must not be suppressed."""
        from pydantic import Field

        from nooa.context_blocks.events import EventBase

        class Sample(EventBase):
            count: int = Field(default=0)
            done: bool = Field(default=False)

        values = Sample().__instance_values__()
        assert values["count"] == 0
        assert values["done"] is False

    def test_pformat_omits_empty_fields(self):
        """Suppression is observable through agentdoc.pformat (the LLM render path)."""
        from pydantic import Field

        from nooa.agentdoc import pformat
        from nooa.context_blocks.events import EventBase

        class Sample(EventBase):
            stdout: str = Field(default="")
            stderr: str = Field(default="")
            value: str | None = Field(default=None)

        rendered = pformat(Sample(stdout="hello"))
        assert "stdout=" in rendered
        assert "stderr=" not in rendered
        assert "value=" not in rendered


class TestUserEvent:
    """Tests for UserEvent."""

    def test_user_event_type(self):
        """UserEvent should have event_type='UserEvent' (auto-derived from class name)."""
        from nooa.context_blocks.events import UserEvent

        event = UserEvent(content="Hello")
        assert event.event_type == "UserEvent"

    def test_user_event_with_content(self):
        """UserEvent should hold content."""
        from nooa.context_blocks.events import UserEvent

        event = UserEvent(content="What's the weather?")
        assert event.content == "What's the weather?"


class TestAssistantEvent:
    """Tests for AssistantEvent."""

    def test_assistant_event_type(self):
        """AssistantEvent should have event_type='AssistantEvent' (auto-derived from class name)."""
        from nooa.context_blocks.events import AssistantEvent

        event = AssistantEvent(content="I can help with that.")
        assert event.event_type == "AssistantEvent"

    def test_assistant_event_with_content(self):
        """AssistantEvent should hold content."""
        from nooa.context_blocks.events import AssistantEvent

        event = AssistantEvent(content="The weather is sunny.")
        assert event.content == "The weather is sunny."


class TestToolCallEvent:
    """Tests for ToolCallEvent."""

    def test_tool_call_event_type(self):
        """ToolCallEvent should have event_type='ToolCallEvent' (auto-derived from class name)."""
        from nooa.context_blocks.events import ToolCallEvent

        event = ToolCallEvent(tool_call_id="tc_1", name="search", arguments={})
        assert event.event_type == "ToolCallEvent"

    def test_tool_call_event_with_data(self):
        """ToolCallEvent should hold tool call fields."""
        from nooa.context_blocks.events import ToolCallEvent

        event = ToolCallEvent(
            tool_call_id="call_123", name="get_weather", arguments={"location": "NYC"}
        )
        assert event.tool_call_id == "call_123"
        assert event.name == "get_weather"
        assert event.arguments["location"] == "NYC"


class TestToolResult:
    """Tests for ToolResult (nested in ToolCallEvent)."""

    def test_tool_result_fields(self):
        """ToolResult should hold result data."""
        from nooa.context_blocks.events import ToolResult

        result = ToolResult(tool_call_id="call_123", content="Sunny, 72°F")
        assert result.tool_call_id == "call_123"
        assert result.content == "Sunny, 72°F"
        assert result.result_status == "complete"  # default

    def test_tool_result_error_status(self):
        """ToolResult should support error status."""
        from nooa.context_blocks.events import ResultStatus, ToolResult

        result = ToolResult(
            tool_call_id="call_123", content="Error occurred", result_status=ResultStatus.ERROR
        )
        assert result.result_status == "error"


class TestEventRendering:
    """Tests for event rendering via pformat."""

    def test_private_fields_excluded_from_repr(self):
        """Private fields should not appear in repr/pformat."""
        from nooa.context_blocks.events import UserEvent

        event = UserEvent(content="Hello")
        repr_str = repr(event)

        # Public field should be in repr
        assert "content=" in repr_str
        assert "Hello" in repr_str

        # Private fields should NOT be in repr
        assert "_id=" not in repr_str
        assert "tag=" not in repr_str
        assert "timestamp=" not in repr_str
        assert "_metadata=" not in repr_str

    def test_pformat_shows_public_fields_only(self):
        """pformat should show class name and public fields."""
        from pprint import pformat

        from nooa.context_blocks.events import ToolCallEvent

        event = ToolCallEvent(tool_call_id="tc_1", name="search", arguments={"q": "test"})
        formatted = pformat(event)

        assert "ToolCallEvent" in formatted
        assert "tool_call_id=" in formatted
        assert "name=" in formatted
        assert "arguments=" in formatted


class TestEventSerialization:
    """Tests for Event serialization/deserialization."""

    def test_user_event_to_json_and_back(self):
        """UserEvent should serialize to JSON and deserialize correctly."""
        from nooa.context_blocks.events import UserEvent

        event = UserEvent(content="Test message")
        json_str = event.model_dump_json()
        restored = UserEvent.model_validate_json(json_str)

        assert restored.event_type == "UserEvent"
        assert restored.content == "Test message"

    def test_tool_call_event_to_json_and_back(self):
        """ToolCallEvent should serialize to JSON and deserialize correctly."""
        from nooa.context_blocks.events import ToolCallEvent

        event = ToolCallEvent(tool_call_id="call_abc", name="get_data", arguments={"key": "value"})
        json_str = event.model_dump_json()
        restored = ToolCallEvent.model_validate_json(json_str)

        assert restored.event_type == "ToolCallEvent"
        assert restored.tool_call_id == "call_abc"
        assert restored.name == "get_data"
        assert restored.arguments["key"] == "value"

    def test_event_list_serialization(self):
        """List of Event should serialize and deserialize correctly."""
        from pydantic import TypeAdapter

        from nooa.context_blocks.events import (
            AssistantEvent,
            Event,
            UserEvent,
        )

        events: list[Event] = [
            UserEvent(content="Question"),
            AssistantEvent(content="Answer"),
        ]

        adapter = TypeAdapter(list[Event])
        json_str = adapter.dump_json(events)
        restored = adapter.validate_json(json_str)

        assert len(restored) == 2
        assert restored[0].event_type == "UserEvent"
        assert restored[1].event_type == "AssistantEvent"


class TestNestedToolResult:
    """Tests for nested ToolResult pattern in ToolCallEvent."""

    def test_tool_call_with_nested_result(self):
        """ToolCallEvent should hold nested ToolResult."""
        from nooa.context_blocks.events import ToolCallEvent, ToolResult

        call_event = ToolCallEvent(
            tool_call_id="call_weather_123",
            name="get_weather",
            arguments={"location": "SF"},
            result=ToolResult(
                tool_call_id="call_weather_123",
                content="Sunny, 72°F",
            ),
        )

        assert call_event.result is not None
        assert call_event.result.tool_call_id == call_event.tool_call_id
        assert call_event.result.content == "Sunny, 72°F"

    def test_tool_call_without_result(self):
        """ToolCallEvent.result should default to None."""
        from nooa.context_blocks.events import ToolCallEvent

        call_event = ToolCallEvent(
            tool_call_id="tc_1",
            name="get_weather",
            arguments={"loc": "SF"},
        )

        assert call_event.result is None

    def test_access_result_via_event(self):
        """Should access result directly via ToolCallEvent.result."""
        from nooa.context_blocks.events import (
            AssistantEvent,
            Event,
            ToolCallEvent,
            ToolResult,
            UserEvent,
        )

        events: list[Event] = [
            UserEvent(content="What's the weather?"),
            ToolCallEvent(
                tool_call_id="tc_1",
                name="get_weather",
                arguments={"loc": "SF"},
                result=ToolResult(tool_call_id="tc_1", content="Sunny"),
            ),
            AssistantEvent(content="It's sunny in SF."),
        ]

        # Find the tool call and access its nested result
        tool_call = next(e for e in events if e.event_type == "ToolCallEvent")
        assert tool_call.result is not None
        assert tool_call.result.content == "Sunny"
