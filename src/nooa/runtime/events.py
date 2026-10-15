# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-facing EventsApi for querying past events.

EventsApi provides a minimal, read-only interface for agents to query
past events. It wraps EventManager and exposes only 3 methods + 2 dunders.

The design follows "Type Names are Prompts" - the LLM sees `EventsApi` as
a database-like interface for querying past interactions.
"""

import inspect
import re
from typing import TYPE_CHECKING, Any

from nooa.events import EventBase
from nooa.skill import Skill

if TYPE_CHECKING:
    from nooa.agent import Agent


_TAG_RE = re.compile(r"\d+(?:\.\.\d+)?")
# The transcript marker, copied with or without its angle brackets: sys tag="22".
_SYS_TAG_RE = re.compile(r'<?\s*(?:sys\s+)?tag\s*=\s*"(\d+(?:\.\.\d+)?)"\s*>?')
_TAG_HELP = (
    'A tag is the number N from <sys tag="N"> in the transcript, e.g. "22", or a '
    'summary range such as "2..40" (span and call ids are not tags). '
    "List active tags with self.events.keys()."
)


def _normalize_tag(name: str, tag: str) -> str:
    """Return a numeric tag, unwrapping an exact copied ``sys tag="N"`` marker."""
    text = tag.strip()
    if _TAG_RE.fullmatch(text):
        return text
    marker = _SYS_TAG_RE.fullmatch(text)
    if marker:
        return marker.group(1)
    raise ValueError(f"Invalid {name} {tag!r}. {_TAG_HELP}")


def _signature_without_var_keyword(func: Any) -> inspect.Signature:
    signature = inspect.signature(func)
    return signature.replace(
        parameters=[
            p for p in signature.parameters.values() if p.kind is not inspect.Parameter.VAR_KEYWORD
        ]
    )


class EventsApi(Skill):
    """Query past events by type, tag, text, or call ID. Compact context by summarizing or collapsing events.

    Use self.events to search conversation history: tool calls, outputs, tasks, errors.
    All queries return events in chronological order. Tags are stable string labels
    ("1", "2", ...) assigned on insert. Summaries use range tags ("1..22").

    Query:
        events.query(limit=50)                      # recent 50
        events.query(type="Task")                   # all task events
        events.query(type="PythonOutput")           # execution outputs
        events.query(type="PythonOutput", execution_status="error", limit=1)
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
        # Find the most recent strategy/retry Error event
        errors = events.query(type="Error", limit=1)

        # Cell failures are PythonOutput events, not standalone Error events
        failed_outputs = events.query(
            type="PythonOutput", execution_status="error", limit=1
        )

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
        execution_status: str | None = None,
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
            execution_status: PythonOutput status filter (e.g. "error" or "complete")
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
            execution_status=execution_status,
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

    def collapse(
        self,
        start_tag: str | int,
        end_tag: str | int,
        summary_text: str | None = None,
        **unsupported: Any,
    ) -> str:
        """Archive a range of events into one Summary marker. Synchronous: do not await.

        Replaces tags start_tag..end_tag with one summary tag.
        Original events remain accessible by their individual tags.
        A tag is the number N shown as <sys tag="N"> in the transcript (e.g. "22"),
        or a summary range such as "2..10"; span or call ids are not tags. The
        literal form 'sys tag="22"' is also accepted. Integers are accepted if
        they identify existing events (including archived ones). List the active
        tags with events.keys().

        Args:
            start_tag: First tag to collapse (inclusive), e.g. "2".
            end_tag: Last tag to collapse (inclusive), e.g. "40".
            summary_text: Optional summary. None = truncation (no text).
                These three are the only parameters.

        Returns:
            The summary tag, e.g. "2..40".

        Examples:
            events.collapse("2", "40")                       # truncate
            events.collapse("2", "40", "User discussed X")   # summarize
            summary = events[events.collapse("2", "40")]     # get the Summary event
        """
        if unsupported:
            names = ", ".join(f"{name}=" for name in sorted(unsupported))
            raise TypeError(
                f"collapse() does not accept {names}; nothing was collapsed. "
                "Its only parameters are start_tag, end_tag and summary_text, e.g. "
                'self.events.collapse("2", "40", "summary of events 2-40"). '
                "It is synchronous: call it without await."
            )
        tags: list[str] = []
        for name, tag in (("start_tag", start_tag), ("end_tag", end_tag)):
            if type(tag) is int:
                tag = str(tag)
                if self._manager.get(tag) is None:
                    raise ValueError(
                        f"{name} does not identify an existing event: {tag!r}. {_TAG_HELP}"
                    )
            elif isinstance(tag, str):
                tag = _normalize_tag(name, tag)
            else:
                raise TypeError(
                    f"{name} must be a string tag or an integer event tag, got "
                    f"{type(tag).__name__}. {_TAG_HELP}"
                )
            tags.append(tag)

        return self._manager.collapse(tags[0], tags[1], summary_text)

    # ``**unsupported`` exists only to turn misuse such as ``summary=`` into the
    # friendly error above; keep it out of the model-facing signature (agentdoc,
    # inspect.signature) so the doc does not advertise arbitrary keywords.
    collapse.__signature__ = _signature_without_var_keyword(collapse)  # type: ignore[attr-defined]

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
