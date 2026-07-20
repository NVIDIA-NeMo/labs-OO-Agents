# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Event auto-registration via __init_subclass__.

Auto-derive event_type from class name (cls.__name__) and auto-register
EventBase subclasses in a global registry.
"""

from __future__ import annotations

import logging

import pytest

from nooa.context_blocks.events import _EVENT_REGISTRY, EventBase, Metadata


@pytest.fixture(autouse=True)
def _restore_event_registry():
    """Snapshot and restore the global event registry to prevent test pollution."""
    snapshot = dict(_EVENT_REGISTRY)
    yield
    _EVENT_REGISTRY.clear()
    _EVENT_REGISTRY.update(snapshot)


# ---------------------------------------------------------------------------
# Auto-derived event_type
# ---------------------------------------------------------------------------


class TestAutoDerivedEventType:
    """Test that subclasses without explicit event_type get cls.__name__."""

    def test_auto_derived_simple(self):
        """A subclass without explicit event_type gets its class name directly."""

        class MyCustomEvent(EventBase):
            pass

        assert MyCustomEvent().event_type == "MyCustomEvent"

    def test_auto_derived_with_acronym(self):
        """Acronyms are preserved as-is in auto-derivation."""

        class GPUMetrics(EventBase):
            pass

        assert GPUMetrics().event_type == "GPUMetrics"

    def test_auto_derived_metadata_subclass(self):
        """Metadata subclass without explicit event_type gets class name directly."""

        class SessionInfo(Metadata):
            model_name: str = ""

        assert SessionInfo().event_type == "SessionInfo"


class TestExplicitEventTypePreserved:
    """Test that explicit Literal event_type fields are not overridden."""

    def test_explicit_literal_field(self):
        """Subclass with explicit Literal event_type keeps its value."""
        from typing import Literal

        from pydantic import Field

        class MyEvent(EventBase):
            event_type: Literal["custom_name"] = Field(default="custom_name", repr=False)

        assert MyEvent().event_type == "custom_name"

    def test_explicit_non_literal_field(self):
        """Subclass with explicit plain str event_type keeps its value."""
        from pydantic import Field

        class AnotherEvent(EventBase):
            event_type: str = Field(default="my_special_type", repr=False)

        assert AnotherEvent().event_type == "my_special_type"


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------


class TestRegistryPopulated:
    """Test that auto-registered classes appear in the global registry."""

    def test_auto_registered_in_global_registry(self):
        """A new subclass appears in _EVENT_REGISTRY."""

        class RegistryTestEvent(EventBase):
            pass

        assert "RegistryTestEvent" in _EVENT_REGISTRY
        assert _EVENT_REGISTRY["RegistryTestEvent"] is RegistryTestEvent

    def test_explicit_event_type_registered_under_explicit_key(self):
        """Explicit event_type uses the explicit value as registry key."""
        from typing import Literal

        from pydantic import Field

        class ExplicitKeyEvent(EventBase):
            event_type: Literal["my_explicit_key"] = Field(default="my_explicit_key", repr=False)

        assert "my_explicit_key" in _EVENT_REGISTRY
        assert _EVENT_REGISTRY["my_explicit_key"] is ExplicitKeyEvent

    def test_metadata_subclass_registered(self):
        """Metadata subclasses are also registered."""

        class MetaRegistryTest(Metadata):
            pass

        assert "MetaRegistryTest" in _EVENT_REGISTRY
        assert _EVENT_REGISTRY["MetaRegistryTest"] is MetaRegistryTest


# ---------------------------------------------------------------------------
# Collision warning
# ---------------------------------------------------------------------------


class TestCollisionWarning:
    """Test that two classes with same derived event_type logs a warning."""

    def test_collision_logs_warning(self, caplog):
        """Second class with same event_type logs a warning."""

        class CollisionA(EventBase):
            event_type: str = "collision_test_type"

        with caplog.at_level(logging.WARNING, logger="nooa.context_blocks.events"):

            class CollisionB(EventBase):
                event_type: str = "collision_test_type"

        assert any("collision_test_type" in record.message for record in caplog.records)
        # Last-registered wins
        assert _EVENT_REGISTRY["collision_test_type"] is CollisionB

    def test_explicit_overrides_auto_derived(self, caplog):
        """Explicit event_type collides with an auto-derived class name."""

        class WidgetStatus(EventBase):
            value: int = 0

        # "WidgetStatus" is now auto-registered.
        assert _EVENT_REGISTRY["WidgetStatus"] is WidgetStatus

        with caplog.at_level(logging.WARNING, logger="nooa.context_blocks.events"):

            class UnrelatedName(EventBase):
                event_type: str = "WidgetStatus"
                value: int = 99

        assert any("WidgetStatus" in record.message for record in caplog.records)
        # Last-registered wins
        assert _EVENT_REGISTRY["WidgetStatus"] is UnrelatedName

    def test_two_explicit_same_value_collide(self, caplog):
        """Two classes with the same explicit event_type trigger a warning."""

        class AlphaEvent(EventBase):
            event_type: str = "shared_type"
            code: int = 0

        assert _EVENT_REGISTRY.get("shared_type") is AlphaEvent

        with caplog.at_level(logging.WARNING, logger="nooa.context_blocks.events"):

            class BetaEvent(EventBase):
                event_type: str = "shared_type"
                code: int = 1

        assert any("shared_type" in record.message for record in caplog.records)
        # Last-registered wins
        assert _EVENT_REGISTRY["shared_type"] is BetaEvent


# ---------------------------------------------------------------------------
# Private class skipped
# ---------------------------------------------------------------------------


class TestPrivateClassSkipped:
    """Test that classes with names starting with _ are not registered."""

    def test_private_class_not_registered(self):

        class _InternalEvent(EventBase):
            pass

        # _InternalEvent should NOT be in the registry
        assert "_InternalEvent" not in _EVENT_REGISTRY

    def test_private_still_gets_event_type(self):
        """Private classes still get auto-derived event_type, just not registered."""

        class _HelperEvent(EventBase):
            pass

        assert _HelperEvent().event_type == "_HelperEvent"


# ---------------------------------------------------------------------------
# Existing core events unchanged
# ---------------------------------------------------------------------------


class TestExistingEventsStillWork:
    """Verify that all existing core event types have correct event_type values."""

    def test_context_blocks_events(self):
        from nooa.context_blocks.events import AssistantEvent, ToolCallEvent, UserEvent

        assert UserEvent(content="hi").event_type == "UserEvent"
        assert AssistantEvent(content="hi").event_type == "AssistantEvent"
        assert ToolCallEvent(tool_call_id="t", name="n", arguments={}).event_type == "ToolCallEvent"

    def test_nemo_events(self):
        from nooa.events import (
            AfterTurn,
            BeforeTurn,
            Error,
            Feedback,
            LLMOutput,
            Message,
            PythonOutput,
            Reasoning,
            Summary,
            Task,
        )

        assert Task(prompt="x").event_type == "Task"
        assert Message(content="x").event_type == "Message"
        assert Reasoning(content="x").event_type == "Reasoning"
        assert Error(content="x").event_type == "Error"
        assert Feedback(content="x").event_type == "Feedback"
        assert LLMOutput(content="x").event_type == "LLMOutput"
        assert (
            PythonOutput(
                tool_call_id="t", execution_status="complete", execution_count=1
            ).event_type
            == "PythonOutput"
        )
        assert Summary(summary_tag="1..2", replaced_range=(1, 2)).event_type == "Summary"
        assert (
            BeforeTurn(method_name="m", strategy="s", generation_id="g", turn_number=1).event_type
            == "BeforeTurn"
        )
        assert (
            AfterTurn(
                method_name="m", strategy="s", generation_id="g", turn_number=1, is_final=False
            ).event_type
            == "AfterTurn"
        )

    def test_rich_output_event(self):
        from nooa.tools.web_publisher import RichOutput

        assert RichOutput().event_type == "RichOutput"

    def test_core_events_in_registry(self):
        """All core event types should be in the global registry."""
        from nooa.events import (
            AfterTurn,
            BeforeTurn,
            Error,
            Feedback,
            LLMOutput,
            Message,
            PythonOutput,
            Reasoning,
            Summary,
            Task,
        )

        assert _EVENT_REGISTRY.get("Task") is Task
        assert _EVENT_REGISTRY.get("Message") is Message
        assert _EVENT_REGISTRY.get("Reasoning") is Reasoning
        assert _EVENT_REGISTRY.get("Error") is Error
        assert _EVENT_REGISTRY.get("Feedback") is Feedback
        assert _EVENT_REGISTRY.get("LLMOutput") is LLMOutput
        assert _EVENT_REGISTRY.get("PythonOutput") is PythonOutput
        assert _EVENT_REGISTRY.get("Summary") is Summary
        assert _EVENT_REGISTRY.get("BeforeTurn") is BeforeTurn
        assert _EVENT_REGISTRY.get("AfterTurn") is AfterTurn


# ---------------------------------------------------------------------------
# SQLite backend uses global registry
# ---------------------------------------------------------------------------


class TestSQLiteBackendUsesRegistry:
    """Test that auto-registered types survive SQLite round-trip without manual register."""

    def test_auto_registered_type_sqlite_roundtrip(self, sqlite_conn):
        """Auto-registered type survives SQLite round-trip without manual register_event_type()."""
        from typing import Annotated

        from pydantic import Field

        from nooa.storage.sqlite import SQLiteEventBackend

        class CustomPayload(EventBase):
            """Auto-registered — no manual register_event_type needed."""

            payload: Annotated[str, Field(description="test payload")] = "hello"

        backend = SQLiteEventBackend(sqlite_conn)
        # We do NOT call backend.register_event_type(CustomPayload)

        event = CustomPayload(payload="test_data")
        backend.store("1", event)

        retrieved = backend.get("1")
        assert retrieved is not None
        assert isinstance(retrieved, CustomPayload)
        assert retrieved.payload == "test_data"
        assert retrieved.event_type == "CustomPayload"

    def test_auto_registered_metadata_subclass_sqlite_roundtrip(self, sqlite_conn):
        """Auto-registered Metadata subclass survives SQLite round-trip."""
        from nooa.storage.sqlite import SQLiteEventBackend

        class AutoMetaEvent(Metadata):
            label: str = ""

        backend = SQLiteEventBackend(sqlite_conn)
        event = AutoMetaEvent(label="important")
        backend.store("1", event)

        retrieved = backend.get("1")
        assert retrieved is not None
        assert isinstance(retrieved, AutoMetaEvent)
        assert retrieved.label == "important"

    def test_explicit_event_type_sqlite_roundtrip(self, sqlite_conn):
        """Explicit event_type also uses registry for SQLite round-trip."""
        from typing import Literal

        from pydantic import Field

        from nooa.storage.sqlite import SQLiteEventBackend

        class ExplicitSQLiteEvent(EventBase):
            event_type: Literal["my_explicit_sqlite"] = Field(
                default="my_explicit_sqlite", repr=False
            )
            data: str = ""

        backend = SQLiteEventBackend(sqlite_conn)
        event = ExplicitSQLiteEvent(data="test")
        backend.store("1", event)

        retrieved = backend.get("1")
        assert retrieved is not None
        assert isinstance(retrieved, ExplicitSQLiteEvent)
        assert retrieved.data == "test"


# ---------------------------------------------------------------------------
# register_event_type() still works
# ---------------------------------------------------------------------------


class TestRegisterEventTypeStillWorks:
    """Verify register_event_type() on EventManager is still functional."""

    def test_register_event_type_accepts_auto_registered_class(self):
        """register_event_type() should not error on auto-registered classes."""
        from nooa.runtime.event_manager import EventManager

        class ManualRegEvent(EventBase):
            pass

        em = EventManager()
        # Should not raise — even though it's already auto-registered
        em.register_event_type(ManualRegEvent)

    def test_register_event_type_rejects_non_eventbase(self):
        """register_event_type() still rejects non-EventBase classes."""
        from nooa.runtime.event_manager import EventManager

        em = EventManager()
        with pytest.raises(TypeError, match="Expected an EventBase subclass"):
            em.register_event_type(str)  # type: ignore[arg-type]
