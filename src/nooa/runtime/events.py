# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-facing EventsApi for querying past events.

EventsApi provides a minimal, read-only interface for agents to query
past events. It wraps EventManager and exposes only 3 methods + 2 dunders.

The design follows "Type Names are Prompts" - the LLM sees `EventsApi` as
a database-like interface for querying past interactions.
"""

from typing import TYPE_CHECKING

from nooa.events import EventBase
from nooa.skill import Skill

if TYPE_CHECKING:
    from nooa.agent import Agent


class EventsApi(Skill):
    """Query past events by type, tag, text, or call ID. Compact context by summarizing or collapsing events.

    Use self.events to search conversation history: tool calls, outputs, tasks, errors.
    All queries return events in chronological order. Tags are stable string labels
    ("1", "2", ...) assigned on insert. Summaries use range tags ("1..22").

    Query:
        events.query(limit=50)                      # recent 50
        events.query(type="Task")                   # all task events
        events.query(type="PythonOutput")           # execution outputs
        events.query(call_id="abc123")              # events for one call
        events.query(query="error")                 # text search
        events.query(query="error.*db", regex=True) # regex search
        events.query(type="Task", call_id="abc")    # combined (AND)

    Access by tag:
        events.get("5")                             # by tag, None if missing
        events["5"]                                 # by tag, KeyError if missing
        events[["2", "3", "4"]]                     # multiple tags
        "5" in events                               # existence check

    Access summary children:
        summary = events["1..22"]
        children = events[summary.children_tags]

    Examples:
        # Find the last error
        errors = events.query(type="Error", limit=1)

        # Get all outputs from current call
        outputs = events.query(type="PythonOutput", call_id=call_id)

        # Check if a specific task was done
        tasks = events.query(type="Task", query="summarise")

    Context compaction:
        When context grows too large, use events to summarize or collapse
        old history. The summarizer does this automatically at token budget
        thresholds, but you can also compact manually:

        events.query(limit=20)       # inspect what's taking space
        # Summarization happens at the runtime level — old events get
        # replaced by summary events with range tags like "1..22".
        # After compaction, access originals via summary.children_tags.

    Load this library:
        doc(self.events)
    """

    def __init__(self, agent: "Agent"):
        self._manager = agent.event_manager

    def query(
        self,
        *,
        type: str | None = None,
        call_id: str | None = None,
        query: str | None = None,
        regex: bool = False,
        limit: int | None = None,
    ) -> list[EventBase]:
        """Query events with AND semantics.

        Multiple filters are ANDed together. Results are returned in
        chronological order, with limit taking the most recent.

        Args:
            type: Event type filter (e.g., "Task", "PythonOutput", "ToolCallEvent")
            call_id: Call ID filter (matches metadata.call_id)
            query: Text search (case-insensitive substring, or regex if regex=True)
            regex: If True, treat query as regex pattern
            limit: Maximum results (most recent first when limit < total)

        Returns:
            List of matching events.

        Examples:
            events.query(limit=50)                      # Recent 50
            events.query(type="Task")                   # All task events
            events.query(type="PythonOutput")           # Execution outputs
            events.query(call_id="abc123")              # Events for call
            events.query(query="error")                 # Text search
            events.query(query="error.*db", regex=True) # Regex search
            events.query(type="Task", call_id="abc")    # Combined (ANDed)
        """
        return self._manager.filter(
            type=type,
            call_id=call_id,
            query=query,
            regex=regex,
            limit=limit,
        )

    def get(self, key: str | list[str]) -> EventBase | list[EventBase] | None:
        """Get event(s) by tag or uuid.

        Safe access that returns None if not found (for single key)
        or filters out missing events (for list of keys).

        Args:
            key: Single tag/uuid or list of tags/uuids.
                - "5" → single event by tag
                - "abc123-..." → single event by uuid
                - "1..22" → Summary by range tag
                - ["2", "3", "4"] → list of events
                - summary.children_tags → list of child tags

        Returns:
            - Single key: Event | None
            - List of keys: list[Event] (missing events filtered out)

        Examples:
            events.get("5")                    # By tag, None if missing
            events.get("abc123-uuid...")        # By uuid, None if missing
            events.get("1..22")                # Summary event
            events.get(["2", "3", "4"])        # Multiple events
            events.get(summary.children_tags)  # Child events of a summary
        """
        if isinstance(key, list):
            result = []
            for k in key:
                event = self._manager.get(k)
                if event is not None:
                    result.append(event)
            return result
        return self._manager.get(key)

    def __getitem__(self, key: str | list[str]) -> EventBase | list[EventBase]:
        """Get event(s) by tag or uuid. Raises KeyError if not found.

        Args:
            key: Single tag/uuid or list of tags/uuids.

        Returns:
            Event or list of events.

        Raises:
            KeyError: If single key not found, or if any key in list not found.

        Examples:
            events["5"]                        # By tag
            events["1..22"]                    # Summary
            events[["2", "3", "4"]]            # Multiple tags
            events[summary.children_tags]      # Child events
        """
        if isinstance(key, list):
            result = []
            for k in key:
                event = self._manager.get(k)
                if event is None:
                    raise KeyError(f"No event with tag or uuid '{k}'")
                result.append(event)
            return result
        event = self._manager.get(key)
        if event is None:
            raise KeyError(f"No event with tag or uuid '{key}'")
        return event

    def __contains__(self, key: str) -> bool:
        """Check if tag or uuid exists.

        Args:
            key: Tag or uuid to check.

        Returns:
            True if event exists, False otherwise.

        Example:
            if "5" in events:
                print("Event 5 exists")
        """
        return self._manager.get(key) is not None

    def collapse(self, start_tag: str, end_tag: str, summary_text: str | None = None) -> str:
        """Archive a range of events into a single Summary marker.

        Replaces tags start_tag..end_tag with one summary tag.
        Original events remain accessible by their individual tags.

        Args:
            start_tag: First tag to collapse (inclusive), e.g. "2".
            end_tag: Last tag to collapse (inclusive), e.g. "40".
            summary_text: Optional summary. None = truncation (no text).

        Returns:
            The summary tag, e.g. "2..40".

        Examples:
            events.collapse("2", "40")                       # truncate
            events.collapse("2", "40", "User discussed X")   # summarize
            summary = events[events.collapse("2", "40")]     # get the Summary event
        """
        return self._manager.collapse(start_tag, end_tag, summary_text)

    def keys(self) -> list[str]:
        """Return the list of active event tags in chronological order.

        After a collapse, the range is replaced by a single "start..end" tag.

        Returns:
            List of tag strings, e.g. ["1", "2..40", "41", "42"].

        Examples:
            tags = events.keys()
            first, last = tags[0], tags[-1]
            events.collapse(first, last, "session summary")
        """
        return list(self._manager.keys())

    def __repr__(self) -> str:
        """String representation showing count of active events."""
        count = len(self._manager.keys())
        return f"<EventsApi({count} active)>"
