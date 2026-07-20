# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Event storage backend protocol.

Defines the interface for pluggable event storage backends.
Implementations can use in-memory, SQLite, files, or other storage mechanisms.

The EventManager uses a backend for storage while handling:
- Event emission to subscribers
- Tag assignment and validation
- Query convenience methods (search, since, for_call, etc.)
"""

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

from nooa.context_blocks import EventBase, EventStatus


def _tag_max_num(tag: str) -> int:
    """Return the highest numeric value encoded in a tag.

    For simple tags like ``"5"``, returns 5.
    For range tags like ``"2..40"``, returns 40 (the end of the range).
    Returns 0 for tags that cannot be parsed as a number.
    """
    try:
        if ".." in tag:
            return int(tag.split("..")[1])
        return int(tag)
    except ValueError:
        return 0


@runtime_checkable
class EventBackend(Protocol):
    """Protocol for event storage backends.

    Backends are responsible for:
    - Storing events with their assigned tags
    - Maintaining tag -> event mappings
    - Tracking active vs archived status
    - Preserving insertion order for active tags

    Thread safety is the responsibility of each implementation.

    Example implementations:
    - InMemoryBackend: Dict + list (current default)
    - SQLiteBackend: SQLite database with indexes
    - FileBackend: JSON files with manifest
    """

    def store(self, tag: str, event: EventBase) -> None:
        """Store an event with its assigned tag.

        The event's status field indicates active/archived state.
        New events are always stored as active.

        Args:
            tag: The string tag (e.g., "1", "2..40").
            event: The event to store.
        """
        ...

    def get(self, tag: str) -> EventBase | None:
        """Retrieve an event by tag.

        Works for both active and archived events.

        Args:
            tag: The string tag to look up.

        Returns:
            The event if found, None otherwise.
        """
        ...

    def get_by_id(self, event_id: str) -> EventBase | None:
        """Retrieve an event by UUID.

        Fallback lookup when tag is unknown.

        Args:
            event_id: The event's UUID.

        Returns:
            The event if found, None otherwise.
        """
        ...

    def update(self, tag: str, **fields: Any) -> bool:
        """Update fields on an event.

        Args:
            tag: The tag of the event to update.
            **fields: Fields to update. Special handling for 'metadata' (merge).

        Returns:
            True if found and updated, False if not found.
        """
        ...

    def remove(self, tag: str) -> bool:
        """Remove an event from storage.

        Args:
            tag: The tag of the event to remove.

        Returns:
            True if found and removed, False if not found.
        """
        ...

    def set_status(self, tag: str, status: EventStatus) -> bool:
        """Update the status of an event.

        Used by collapse() to archive events.

        Args:
            tag: The tag of the event.
            status: New status (EventStatus.ACTIVE or EventStatus.ARCHIVED).

        Returns:
            True if found and updated, False if not found.
        """
        ...

    def active_tags(self) -> list[str]:
        """Return ordered list of active (non-archived) tags.

        Order must be preserved from insertion for correct event rendering.

        Returns:
            List of active tags in insertion order.
        """
        ...

    def insert_active_tag(self, tag: str, index: int) -> None:
        """Insert a tag into the active list at a specific position.

        Used when creating Summary events that replace a range.

        Args:
            tag: The tag to insert.
            index: Position in the active list.
        """
        ...

    def remove_active_tag(self, tag: str) -> bool:
        """Remove a tag from the active list (without removing the event).

        Used during collapse to remove tags before archiving.

        Args:
            tag: The tag to remove from active list.

        Returns:
            True if found and removed, False if not found.
        """
        ...

    def all_events(self) -> Iterator[EventBase]:
        """Iterate over all events in insertion order.

        Includes both active and archived events.

        Yields:
            Events in the order they were stored.
        """
        ...

    def find_tag(self, event: EventBase) -> str | None:
        """Find the tag for a given event.

        Args:
            event: The event to look up (by identity or ID).

        Returns:
            The tag if found, None otherwise.
        """
        ...

    def register_event_type(self, cls: type[EventBase]) -> None:
        """Register a custom EventBase subclass for deserialization.

        Backends that deserialize from storage (e.g. SQLite) use the
        registered types to reconstruct the correct class.  Backends that
        keep live Python objects (e.g. InMemoryBackend) can ignore this.

        Args:
            cls: An EventBase subclass whose ``event_type`` Literal value
                 will be used as the deserialization key.
        """
        ...

    def clear(self) -> None:
        """Remove all events and reset state."""
        ...

    def allocate_next_tag(self) -> str:
        """Allocate and return the next sequential tag.

        Tag allocation lives on the backend so multiple EventManagers
        writing through the same backend (e.g. the agent's stable
        manager and a SessionManager recording TUI metadata) never hand
        out the same tag. ``store(tag, event)`` is expected to follow.
        """
        ...

    def __len__(self) -> int:
        """Return total number of stored events (active + archived)."""
        ...


class InMemoryBackend:
    """In-memory implementation of EventBackend.

    Uses:
    - dict for tag -> event mapping (O(1) lookup)
    - list for ordered active tags (O(1) append, O(n) remove)
    - list for all events in order (for iteration)

    This is the default backend, suitable for most use cases.
    """

    def __init__(self) -> None:
        self._events: list[EventBase] = []
        self._tag_to_event: dict[str, EventBase] = {}
        self._active_tags: list[str] = []
        # Monotonic tag counter. Eager-init from storage so a fresh
        # backend (or one populated externally) starts at the right
        # number. ``store()`` keeps the counter coherent if a caller
        # writes a tag higher than the current value, so direct stores
        # and ``allocate_next_tag()`` never collide.
        self._next_tag_num: int = self.max_tag_num() + 1

    def store(self, tag: str, event: EventBase) -> None:
        self._events.append(event)
        self._tag_to_event[tag] = event
        self._active_tags.append(tag)
        self._next_tag_num = max(self._next_tag_num, _tag_max_num(tag) + 1)

    def get(self, tag: str) -> EventBase | None:
        return self._tag_to_event.get(tag)

    def get_by_id(self, event_id: str) -> EventBase | None:
        for event in self._events:
            if event.id == event_id:
                return event
        return None

    def update(self, tag: str, **fields: Any) -> bool:
        event = self.get(tag)
        if event is None:
            return False

        for key, value in fields.items():
            if key == "metadata":
                event.metadata.update(value)
            elif hasattr(event, key):
                setattr(event, key, value)

        return True

    def remove(self, tag: str) -> bool:
        event = self._tag_to_event.get(tag)
        if event is None:
            return False

        # Remove from all structures
        del self._tag_to_event[tag]
        if tag in self._active_tags:
            self._active_tags.remove(tag)
        # Remove from events list
        for i, e in enumerate(self._events):
            if e.id == event.id:
                self._events.pop(i)
                break

        return True

    def set_status(self, tag: str, status: EventStatus) -> bool:
        event = self.get(tag)
        if event is None:
            return False
        event.status = status
        return True

    def active_tags(self) -> list[str]:
        return self._active_tags.copy()

    def insert_active_tag(self, tag: str, index: int) -> None:
        self._active_tags.insert(index, tag)

    def remove_active_tag(self, tag: str) -> bool:
        if tag in self._active_tags:
            self._active_tags.remove(tag)
            return True
        return False

    def all_events(self) -> Iterator[EventBase]:
        return iter(self._events)

    def find_tag(self, event: EventBase) -> str | None:
        for tag, e in self._tag_to_event.items():
            if e is event or e.id == event.id:
                return tag
        return None

    def register_event_type(self, cls: type[EventBase]) -> None:
        """No-op: in-memory backend stores live objects, no deserialization needed."""

    def clear(self) -> None:
        self._events.clear()
        self._tag_to_event.clear()
        self._active_tags.clear()
        self._next_tag_num = 1

    def max_tag_num(self) -> int:
        if not self._tag_to_event:
            return 0
        return max(_tag_max_num(tag) for tag in self._tag_to_event)

    def allocate_next_tag(self) -> str:
        tag = str(self._next_tag_num)
        self._next_tag_num += 1
        return tag

    def __len__(self) -> int:
        return len(self._events)
