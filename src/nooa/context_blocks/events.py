# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed event models with discriminated unions.

Events represent conversation items: user messages, assistant responses,
tool calls, and tool results.

Rendering: Events are rendered using pformat(event). Fields with repr=False
are excluded from LLM context.

Each event class has:
- event_type: Auto-derived from class name (repr=False, excluded from display)
- id: Unique identifier (repr=False, excluded from display)
- metadata: Arbitrary metadata dict (repr=False, excluded from display)
- _role: ClassVar for provider role (USER/ASSISTANT/TOOL)
- tag: Event position (e.g., '5' or '2..40'), set by EventManager
- timestamp: Creation time
"""

import inspect
import logging
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, ClassVar
from uuid import uuid4

from pydantic import BaseModel, Field

from nooa.context_blocks.roles import Role

_logger = logging.getLogger(__name__)

# === Global Event Registry ===

# Mapping of event_type string -> EventBase subclass.
# Populated automatically by EventBase.__init_subclass__.
# Backends (e.g. SQLiteEventBackend) use this as a fallback for deserialization.
_EVENT_REGISTRY: dict[str, type["EventBase"]] = {}


def _is_empty(value: Any) -> bool:
    """True if ``value`` should be suppressed from LLM-facing serialization.

    ``None`` and any empty string/container are empty. Numeric zero and
    ``False`` are NOT empty — they carry meaning.
    """
    if value is None:
        return True
    if isinstance(value, (bool, int, float)):
        return False
    if isinstance(value, (str, list, dict, tuple, set, frozenset)):
        return len(value) == 0
    return False


# === Enums ===


class EventStatus(StrEnum):
    """Status of an event in the event manager."""

    ACTIVE = "active"
    ARCHIVED = "archived"


class ResultStatus(StrEnum):
    """Status of a tool result."""

    COMPLETE = "complete"
    ERROR = "error"


# === Base Event ===


class EventBase(BaseModel):
    """Base class for all events.

    Subclasses define:
    - event_type: Auto-derived from class name (repr=False), or explicit override
    - _role: ClassVar for provider role
    - Public fields which are rendered via pformat()

    Auto-registration: When a subclass is defined, ``__init_subclass__``
    automatically derives ``event_type`` from the class name (e.g.
    ``MyCustomEvent`` -> ``"MyCustomEvent"``) if the subclass does not
    explicitly define one.  The class is also added to the global
    ``_EVENT_REGISTRY`` (unless the class name starts with ``_``).
    """

    _role: ClassVar[Role] = Role.USER

    # Discriminator field - excluded from repr.
    # Default is "" (empty); model_post_init fills it with cls.__name__ if unset.
    event_type: str = Field(default="", repr=False, description="Event type discriminator")

    def model_post_init(self, __context: Any) -> None:
        """Auto-derive event_type from class name when not explicitly set."""
        super().model_post_init(__context)
        if not self.event_type:
            self.event_type = type(self).__name__

    def __instance_values__(self) -> dict[str, Any]:
        """Return non-empty field values for agentdoc serialization.

        Implements the ``SupportsInstanceValues`` protocol so ``agentdoc.pformat``
        skips fields whose value is ``None`` or an empty container/string.
        Trims noise like ``stderr=''``, ``error=''``, ``value=None`` from
        LLM-visible event serializations. ``0`` and ``False`` are intentionally
        preserved since they are semantically meaningful
        (e.g. ``execution_count=0``, ``success=False``).
        """
        return {
            name: value
            for name in type(self).model_fields
            if not _is_empty(value := getattr(self, name, None))
        }

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Register subclass in global registry.

        Uses ``__pydantic_init_subclass__`` (not ``__init_subclass__``) because
        Pydantic has fully built the model by this point — each subclass has its
        own ``model_fields`` dict and compiled schema.

        Registration only — ``model_post_init`` handles auto-deriving
        ``event_type`` at instance creation time.
        """
        super().__pydantic_init_subclass__(**kwargs)

        # Determine the registry key.
        # If the subclass explicitly declares event_type with a non-empty
        # default, use that value. Otherwise use the class name.
        own_annotations = inspect.get_annotations(cls)
        has_explicit = "event_type" in own_annotations

        if has_explicit:
            fi = cls.model_fields.get("event_type")
            default = fi.default if fi is not None and isinstance(fi.default, str) else ""
            event_type_str = default if default else cls.__name__
        else:
            event_type_str = cls.__name__

        # Skip registration for private/internal classes (names starting with _)
        if cls.__name__.startswith("_"):
            return

        # Register in global registry with collision warning
        existing = _EVENT_REGISTRY.get(event_type_str)
        if existing is not None and existing is not cls:
            _logger.warning(
                "Event type %r collision: %s overwrites %s in global registry",
                event_type_str,
                cls.__name__,
                existing.__name__,
            )
        _EVENT_REGISTRY[event_type_str] = cls

    # Fields excluded from repr (not shown to LLM)
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        repr=False,
        description="Unique UUID for this event",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        repr=False,
        description="Arbitrary metadata (call_id, source, etc.)",
    )
    status: EventStatus = Field(
        default=EventStatus.ACTIVE, repr=False, description="Active or archived"
    )
    tag: str | None = Field(
        default=None,
        repr=False,
        description="Positional tag assigned by EventManager ('1', '2', '2..40')",
    )
    timestamp: datetime = Field(
        default_factory=datetime.now, repr=False, description="Creation time"
    )


# === Typed Events ===


class UserEvent(EventBase):
    """User message event."""

    _role: ClassVar[Role] = Role.USER

    content: Annotated[str, Field(description="Message content")]


class AssistantEvent(EventBase):
    """Assistant response event."""

    _role: ClassVar[Role] = Role.ASSISTANT

    content: Annotated[str, Field(description="Response content from the assistant")]


class ToolResult(BaseModel):
    """Result of a tool call, stored as child of ToolCallEvent.

    This is a nested model (not an event) used for the unified tool call pattern.
    One ToolCallEvent with a nested ToolResult provides both call and result in a
    single event.
    """

    tool_call_id: Annotated[str, Field(description="ID of the tool call this is a result for")]
    content: Annotated[str, Field(description="Result content from the tool")]
    result_status: ResultStatus = Field(
        default=ResultStatus.COMPLETE, description="Execution status"
    )


class ToolCallEvent(EventBase):
    """Tool invocation event with optional nested result.

    The nested `result` field enables a unified tool call pattern where one
    event represents both the call and its result.

    Benefits:
    - Structurally impossible to orphan call or result
    - Archive one event = archive the whole interaction
    - Access result via self.events[n].result
    """

    _role: ClassVar[Role] = Role.ASSISTANT

    tool_call_id: Annotated[str, Field(description="Unique identifier for this tool call")]
    name: Annotated[str, Field(description="Name of the tool being called")]
    arguments: Annotated[dict[str, Any], Field(description="Arguments passed to the tool")]

    # Nested result (filled after execution via EventManager.update())
    result: ToolResult | None = Field(
        default=None, description="Result of the tool call, added after execution"
    )


# === Metadata Extension Point ===


class Metadata(EventBase):
    """Base class for stored metadata events.

    Subclass to attach arbitrary structured data to a session that
    persists to storage but is never included in LLM context.

    The core ``Event`` union does not include ``Metadata`` subtypes.
    Subtypes are **auto-registered** via ``__pydantic_init_subclass__``
    in the global ``_EVENT_REGISTRY``, so backends (e.g. SQLite) can
    deserialize them without manual ``register_event_type()`` calls.

    The ``event_type`` is auto-derived from the class name directly
    (e.g. ``MyMetadata`` -> ``"MyMetadata"``).  Override explicitly if
    you need a custom name.

    Example::

        class TUISessionStart(Metadata):
            model: str = ""
            working_dir: str = ""
            # event_type auto-derived as "TUISessionStart"

        # Or with explicit event_type:
        class TUISessionStart(Metadata):
            event_type: Literal["tui_session_start"] = "tui_session_start"
            model: str = ""
            working_dir: str = ""
    """

    model_config = {"extra": "allow"}

    event_type: str = Field(default="", repr=False)
    _role: ClassVar[Role] = Role.METADATA


# === Union of context-blocks event types ===

Event = UserEvent | AssistantEvent | ToolCallEvent
