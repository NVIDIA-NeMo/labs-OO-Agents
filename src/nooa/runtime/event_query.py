# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EventQuery - type-safe event filtering configuration.

EventQuery provides a declarative way to specify event filtering that mirrors
the event_manager.filter() API. It can be used at agent, decorator, and runtime levels.
"""

from dataclasses import dataclass
from typing import Self

from nooa.context_blocks import EventBase as EventBase


@dataclass(frozen=True)
class EventQuery:
    """Type-safe event filtering configuration.

    Mirrors the parameters of event_manager.filter() for consistency.
    Can be used at multiple levels:
    - Agent level: Default query for all methods
    - Decorator level: Override for specific method via @strategy
    - Runtime level: Temporary override via event_manager.set_event_query()

    Args:
        type: Filter by event type (e.g., "Task", "Error")
        call_id: Filter by call ID (use "current" for current call)
        query: Text query for filtering event content
        regex: Whether to treat query as regex pattern
        limit: Maximum number of events to return

    Examples:
        # Filter to current method's events
        EventQuery(call_id="current")

        # Last 10 events only
        EventQuery(limit=10)

        # Error events only
        EventQuery(type="Error")

        # Combine filters
        EventQuery(type="Task", call_id="current", limit=5)
    """

    type: str | None = None
    call_id: str | None = None
    query: str | None = None
    regex: bool = False
    limit: int | None = None

    @classmethod
    def current_call(cls, limit: int | None = None) -> Self:
        """Filter to events from the current method call only.

        This is the primary use case for method-scoped event filtering.

        Args:
            limit: Optional limit on number of events

        Example:
            @strategy(context=ScopedContext(events=EventQuery.current_call()))
            async def solve_with_reflection(self, problem: str):
                '''LLM sees only events from this method.'''
                ...
        """
        return cls(call_id="current", limit=limit)

    @classmethod
    def by_type(cls, event_type: str, limit: int | None = None) -> Self:
        """Filter to specific event type(s).

        Args:
            event_type: Event type to filter (e.g., "Task", "Error")
            limit: Optional limit on number of events
        """
        return cls(type=event_type, limit=limit)

    @classmethod
    def last_n(cls, n: int) -> Self:
        """Get the last N events only.

        Args:
            n: Number of recent events to include
        """
        return cls(limit=n)

    def apply(
        self, events: list[EventBase], *, current_call_id: str | None = None
    ) -> list[EventBase]:
        """Apply this query to filter a list of events.

        Args:
            events: List of events to filter
            current_call_id: Current call ID (for resolving "current")

        Returns:
            Filtered list of events
        """
        filtered = events

        # Filter by call_id (events store call_id in metadata, not as attribute)
        if self.call_id is not None:
            target_call_id = current_call_id if self.call_id == "current" else self.call_id
            filtered = [
                e for e in filtered if getattr(e, "metadata", {}).get("call_id") == target_call_id
            ]

        # Filter by type
        if self.type is not None:
            filtered = [e for e in filtered if e.__class__.__name__ == self.type]

        # Filter by query
        if self.query is not None:
            import re as re_module

            if self.regex:
                pattern = re_module.compile(self.query)
                filtered = [e for e in filtered if pattern.search(str(e))]
            else:
                filtered = [e for e in filtered if self.query.lower() in str(e).lower()]

        # Apply limit
        if self.limit is not None:
            filtered = filtered[-self.limit :]

        return filtered
