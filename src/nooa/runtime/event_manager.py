# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Event manager - unified event pipeline.

EventManager is a unified event pipeline:
- Recording events (if record=True)
- Notifying subscribers via on()
- Query methods for filtering

Design: phase-2-strategy-middleware.md
"""

import itertools
import logging
import re
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from nooa.context_blocks import EventStatus
from nooa.context_blocks.models import Role
from nooa.events import (
    EventBase,
    Summary,
)
from nooa.runtime.context_vars import _current_event_format_var, _get_agent_call_stack
from nooa.runtime.event_backend import EventBackend, InMemoryBackend

if TYPE_CHECKING:
    from nooa.runtime.event_query import EventQuery
    from nooa.runtime.middleware import (
        AgentCallContext,
        AgentCallMiddleware,
        AgentCallNext,
        ExecutePythonContext,
        ExecutePythonMiddleware,
        ExecutePythonNext,
        LLMCallContext,
        LLMCallMiddleware,
        LLMCallNext,
    )

logger = logging.getLogger(__name__)

# Type alias for event handlers
EventHandler = Callable[[EventBase], None]

# Monotonic counter for stable EventManager identity (middleware re-entry guard).
_em_id_counter = itertools.count(1)


def _make_next(
    mw_fn: Callable[..., Awaitable[Any]],
    next_fn: Callable[..., Awaitable[Any]],
) -> Callable[..., Awaitable[Any]]:
    """Build one layer of the middleware chain.

    Standalone function so each call gets its own closure scope.
    """

    async def _next(ctx: Any) -> Any:
        result = await mw_fn(ctx, next_fn)
        if result is None:
            raise RuntimeError(
                "Middleware returned None. Middleware must return the context "
                "object (return await nxt(ctx), or set ctx.result and return ctx)."
            )
        return result

    return _next


class EventManager:
    """Unified event pipeline with string-tagged access.

    All events flow through add() which:
    1. Notifies subscribers via on()
    2. Records event (if record=True)
    3. Assigns a string tag ("1", "2", etc.)

    String tags provide stable access:
    - manager["5"] always returns event 5, even after summarization
    - manager["2..40"] returns the Summary for that range
    - keys() returns only tags visible to the LLM

    Ordering: Events are stored in insert order (``_events`` list in the
    backend). Active tags (``_active_tags``) track display order — which
    may differ after summarization collapses a range into a single tag.

    Events are automatically tagged with call_id from the agent call stack.
    """

    def __init__(
        self,
        backend: EventBackend | None = None,
    ):
        """Initialize EventManager.

        Args:
            backend: Storage backend. Defaults to InMemoryBackend.
        """
        self._backend: EventBackend = backend if backend is not None else InMemoryBackend()
        self._handlers: dict[str, list[EventHandler]] = defaultdict(list)

        # Runtime event query override (set via set_event_query())
        self._event_query: EventQuery | None = None

        # Middleware engine — initialized here, methods defined below.
        # Uses lazy import of middleware types to avoid circular dependency
        # (middleware.py imports Agent which imports runtime/__init__ which imports us).
        self._middleware: dict[str, list[Any]] = {
            "agent_call": [],
            "llm_call": [],
            "execute_python": [],
        }
        self._middleware_id: int = next(_em_id_counter)

    # === Core Methods ===

    def add(self, event: EventBase, *, record: bool = True) -> str:
        """Add event to the pipeline.

        Args:
            event: The event to add
            record: If True, store event for LLM context.
                    If False, just notify callbacks.
                    Runtime events (Role.RUNTIME_EVENT) are never recorded regardless.

        Returns:
            The assigned string tag (e.g., "1", "2", or "2..40" for summaries).
        """
        # Auto-inject call_id from the agent call stack context variable.
        # This tags every event with the current method invocation ID.
        stack = _get_agent_call_stack()
        if stack and "call_id" not in event.metadata:
            event.metadata["call_id"] = stack[-1]
        event_format = _current_event_format_var.get()
        if event_format is not None and "truncation_event_format" not in event.metadata:
            event.metadata["truncation_event_format"] = event_format

        # Emit to handlers
        self._emit(event)

        # Runtime events are never recorded, regardless of record parameter
        if event._role == Role.RUNTIME_EVENT:
            record = False

        # Optionally store with assigned tag
        if record:
            # Does the event already have a tag? Use it. Otherwise let the
            # backend allocate one — backend-driven allocation guarantees
            # that multiple managers writing through the same backend
            # never hand out the same tag.
            if isinstance(event, Summary) and event.summary_tag:
                tag = event.summary_tag
                self._validate_tag(tag)
            else:
                tag = self._backend.allocate_next_tag()

            # Set tag on event and store via backend
            event.tag = tag
            self._backend.store(tag, event)

            return tag

        return str(event.id)

    def register_event_type(self, cls: type[EventBase]) -> None:
        """Register a custom EventBase subclass for deserialization.

        Delegates to the backend.  Backends that persist to storage (e.g.
        SQLite) use the registered types to reconstruct the correct class on
        read.  Call this once at startup for each custom event type before
        reading from storage.

        Args:
            cls: An EventBase subclass. The registry key is derived from the
                class name or an explicit ``event_type`` field default.

        Raises:
            TypeError: If *cls* is not an EventBase subclass.
        """
        if not (isinstance(cls, type) and issubclass(cls, EventBase)):
            raise TypeError(f"Expected an EventBase subclass, got {cls!r}")
        self._backend.register_event_type(cls)

    def on(self, event_type: str, handler: EventHandler) -> Callable[[], None]:
        """Subscribe to events of a specific type.

        Args:
            event_type: Event type (e.g., "Task", "LLMOutput", "Error")
                       or "*" for all events.
            handler: Callback function receiving Event.

        Returns:
            Unsubscribe function - call to remove handler.
        """
        self._handlers[event_type].append(handler)

        def unsubscribe() -> None:
            try:
                self._handlers[event_type].remove(handler)
            except ValueError:
                pass  # already removed — idempotent

        return unsubscribe

    def set_backend(self, backend: EventBackend) -> None:
        """Swap the persistence backend; handlers and middleware are preserved."""
        self._backend = backend

    def set_event_query(self, query: "EventQuery | None") -> None:
        """Set runtime event query for filtering context events.

        This overrides the agent-level and decorator-level event queries
        at runtime. Useful for temporarily changing event visibility.

        Args:
            query: EventQuery to use for filtering, or None to clear.

        Example:
            from nooa import EventQuery

            # Show only errors temporarily
            agent.event_manager.set_event_query(EventQuery.by_type("Error"))
            result = await agent.process()

            # Restore to default
            agent.event_manager.set_event_query(None)
        """
        self._event_query = query

    def get_event_query(self) -> "EventQuery | None":
        """Get current runtime event query.

        Returns:
            Current EventQuery or None if not set.
        """
        return self._event_query

    def _emit(self, event: EventBase) -> None:
        """Emit event to registered handlers.

        Snapshots handler lists before iterating so that handlers can safely
        subscribe or unsubscribe during emission without affecting the
        current cycle.  Exceptions in individual handlers are logged and
        swallowed so one misbehaving handler cannot silence the rest.
        """
        event_type = event.event_type

        for handler in list(self._handlers[event_type]):
            try:
                handler(event)
            except Exception:
                logger.warning("Event handler %r raised", handler, exc_info=True)

        for handler in list(self._handlers["*"]):
            try:
                handler(event)
            except Exception:
                logger.warning("Wildcard handler %r raised", handler, exc_info=True)

    # === Middleware (intercept) ===

    def intercept(
        self,
        kind: str,
        fn: "AgentCallMiddleware | LLMCallMiddleware | ExecutePythonMiddleware",
    ) -> Callable[[], None]:
        """Register middleware that wraps a lifecycle operation.

        ``intercept()`` vs ``on()`` — two complementary APIs:

        - ``intercept(kind, fn)`` — **wraps** a live operation.
          Middleware can transform inputs, modify outputs, or block (guardrail).
        - ``on(event_type, fn)`` — **observes** a recorded event, fire-and-forget.

        Execution order::

            agent_call middleware        ← auth, rate limiting
              → llm_call middleware      ← per-call guardrails
                → acall()
              → execute_python middleware ← per-exec guardrails
                → execute_code()
              → result recorded as event
                → on() handlers fire     ← observe only

        Registration order = execution order.  First registered = outermost.

        Args:
            kind: ``"agent_call"``, ``"llm_call"``, or ``"execute_python"``.
            fn: Async middleware ``(ctx, nxt) -> ctx``.

        Returns:
            Unsubscribe callable.

        Raises:
            ValueError: If *kind* is not recognised.
        """
        if kind not in self._middleware:
            raise ValueError(f"Unknown middleware kind {kind!r}, expected {set(self._middleware)}")
        self._middleware[kind].append(fn)

        def _unsubscribe() -> None:
            try:
                self._middleware[kind].remove(fn)
            except ValueError:
                pass

        return _unsubscribe

    async def run_middleware(
        self,
        kind: str,
        ctx: "AgentCallContext | LLMCallContext | ExecutePythonContext",
        core: "AgentCallNext | LLMCallNext | ExecutePythonNext",
    ) -> "AgentCallContext | LLMCallContext | ExecutePythonContext":
        """Execute middleware chain for *kind*, finishing with *core*.

        When no middleware is registered, calls *core* directly (zero overhead).
        """
        entries = self._middleware.get(kind)
        if not entries:
            return await core(ctx)  # type: ignore[arg-type]

        current = core
        for fn in reversed(list(entries)):
            current = _make_next(fn, current)

        return await current(ctx)  # type: ignore[arg-type]

    # === Query Methods ===

    def get(self, key: str) -> EventBase | None:
        """Get event by tag or ID.

        Args:
            key: Event tag (e.g., "1", "2..40") or UUID to look up.

        Returns:
            Event if found, None otherwise.
        """
        # First, check if it's a tag
        event = self._backend.get(key)
        if event is not None:
            return event

        # Fallback: check by UUID
        return self._backend.get_by_id(key)

    def filter(
        self,
        *,
        type: str | None = None,
        call_id: str | None = None,
        query: str | None = None,
        regex: bool = False,
        limit: int | None = None,
    ) -> list[EventBase]:
        """Filter events with AND semantics.

        Args:
            type: Event type filter (e.g., "Task", "PythonOutput")
            call_id: Call ID filter (matches metadata.call_id)
            query: Text search (case-insensitive substring, or regex if regex=True)
            regex: If True, treat query as regex pattern
            limit: Maximum results (most recent first when limit < total)

        Returns:
            List of matching events.
        """
        events = list(self._backend.all_events())

        # Apply type filter
        if type is not None:
            events = [e for e in events if e.event_type == type]

        # Apply call_id filter
        if call_id is not None:
            events = [e for e in events if e.metadata.get("call_id") == call_id]

        # Apply text/regex filter
        if query is not None:
            if regex:
                pattern = re.compile(query, re.IGNORECASE)
                events = [e for e in events if pattern.search(self._get_searchable_text(e))]
            else:
                query_lower = query.lower()
                events = [e for e in events if query_lower in self._get_searchable_text(e).lower()]

        # Apply limit (from end = most recent)
        if limit is not None and len(events) > limit:
            events = events[-limit:]

        return events

    def _get_searchable_text(self, event: EventBase) -> str:
        """Extract searchable text from an event's public fields."""
        parts: list[str] = []

        # Get all public fields from model_dump (excludes private fields)
        for _field_name, value in event.model_dump().items():
            if value is not None:
                if isinstance(value, list):
                    parts.append(" ".join(str(item) for item in value))
                else:
                    parts.append(str(value))

        return " ".join(parts) if parts else event.event_type

    # === Dict-like Methods (active events) ===

    def items(self) -> list[tuple[str, EventBase]]:
        """Return (tag, event) pairs for active events.

        Uses the keys list to maintain correct ordering, including
        Summaries at their conceptual position.

        Returns:
            List of (tag, event) tuples in view order.
        """
        return [
            (tag, evt)
            for tag in self._backend.active_tags()
            if (evt := self._backend.get(tag)) is not None
        ]

    def keys(self) -> list[str]:
        """Return ordered list of active (non-archived) tags.

        This is what the LLM sees. After summarization:
        ["1", "2..40", "41", "42", ...]

        Returns:
            List of active tags in order.
        """
        return self._backend.active_tags()

    def values(self) -> list[EventBase]:
        """Return only non-archived events in view order (without tags).

        Returns:
            List of active events ordered by view position.
        """
        return [
            evt
            for tag in self._backend.active_tags()
            if (evt := self._backend.get(tag)) is not None
        ]

    # === Mutation Methods ===

    def update(self, key: str, **fields: Any) -> bool:
        """Update event fields. Does NOT re-emit to handlers.

        Args:
            key: Event tag or UUID to update.
            **fields: Fields to update. Use metadata={...} to merge metadata.

        Returns:
            True if found and updated, False if not found.
        """
        # Try as tag first
        if self._backend.update(key, **fields):
            return True

        # Fallback: find tag by UUID, then update
        event = self._backend.get_by_id(key)
        if event is not None:
            tag = self._backend.find_tag(event)
            if tag:
                return self._backend.update(tag, **fields)

        return False

    def remove(self, key: str) -> bool:
        """Remove an event by tag or ID.

        Args:
            key: Event tag or UUID to remove.

        Returns:
            True if found and removed, False if not found.
        """
        # Try as tag first
        if self._backend.remove(key):
            return True

        # Fallback: find tag by UUID, then remove
        event = self._backend.get_by_id(key)
        if event is not None:
            tag = self._backend.find_tag(event)
            if tag:
                return self._backend.remove(tag)

        return False

    def clear(self) -> None:
        """Clear all events and reset counters."""
        # Backend's clear() resets its tag counter; nothing else for the
        # manager to reset since handlers/middleware/event_query are
        # subscriber state, not event state.
        self._backend.clear()

    # === Archive Operations ===

    def collapse(self, start_tag: str, end_tag: str, summary_text: str | None = None) -> str:
        """Archive a range of events and create a Summary marker.

        This is the primary method for both truncation and summarization:
        - collapse("2", "40") → truncation (summary_text=None)
        - collapse("2", "40", summary_text="...") → summarization

        After collapse, the range is replaced with a single tag "start..end":
        - keys() changes from ["1", "2", ..., "40", "41"] to ["1", "2..40", "41"]
        - manager["5"] still returns the original event (stable access)
        - manager["2..40"] returns the new Summary

        Args:
            start_tag: First tag to collapse (inclusive), e.g., "2"
            end_tag: Last tag to collapse (inclusive), e.g., "40"
            summary_text: Optional LLM-generated summary. None = truncation.

        Returns:
            The tag of the newly created Summary (e.g., "2..40").
        """
        # Parse tag numbers for range calculation
        start_num = self._parse_tag_start(start_tag)
        end_num = self._parse_tag_end(end_tag)

        # Find events/tags to collapse
        tags_to_collapse: list[str] = []
        for tag in self._backend.active_tags():
            tag_start = self._parse_tag_start(tag)
            tag_end = self._parse_tag_end(tag)
            # Include if ranges overlap
            if tag_start <= end_num and tag_end >= start_num:
                tags_to_collapse.append(tag)

        if not tags_to_collapse:
            raise ValueError(f"No active events in range '{start_tag}'..'{end_tag}'")

        # Archive the events and remove from active list
        for tag in tags_to_collapse:
            self._backend.set_status(tag, EventStatus.ARCHIVED)
            self._backend.remove_active_tag(tag)

        # Calculate the full range for the new summary (flatten nested summaries)
        actual_start = min(self._parse_tag_start(t) for t in tags_to_collapse)
        actual_end = max(self._parse_tag_end(t) for t in tags_to_collapse)

        # Create the summary tag (flattened range)
        summary_tag = f"{actual_start}..{actual_end}"

        # Find insertion point (where first collapsed tag was)
        first_collapsed_start = self._parse_tag_start(tags_to_collapse[0])
        insert_idx = 0
        active_tags = self._backend.active_tags()
        for i, tag in enumerate(active_tags):
            if self._parse_tag_start(tag) > first_collapsed_start:
                insert_idx = i
                break
        else:
            insert_idx = len(active_tags)

        # Create the Summary
        summary = Summary(
            summary_tag=summary_tag,
            replaced_range=(actual_start, actual_end),
            children_tags=tags_to_collapse,
            summary_text=summary_text,
            doc=f'To access collapsed events, call self.events["{summary_tag}"].children_tags',
        )
        summary.tag = summary_tag

        # Store via backend (this adds to events list and tag mapping)
        # But we need to insert at specific position, so use lower-level methods
        # First store the event (adds to end of events list)
        self._backend.store(summary_tag, summary)
        # Remove from end of active list (store adds it there)
        self._backend.remove_active_tag(summary_tag)
        # Insert at correct position
        self._backend.insert_active_tag(summary_tag, insert_idx)

        # Emit symmetrically with add() so "Summary" subscribers actually
        # see collapses. Historically collapse() was silent.
        self._emit(summary)

        return summary_tag

    # === Introspection (internal) ===

    def __getitem__(self, key: str) -> EventBase:
        """Access events by string tag or UUID.

        Supports both:
        - String tags: manager["5"], manager["2..40"]
        - UUIDs: manager["abc123-..."] (the event.id)

        Tags are checked first, then UUID lookup as fallback.

        Args:
            key: String tag (e.g., "5", "2..40") or UUID.

        Returns:
            The event with that tag or UUID.

        Raises:
            KeyError: If no event matches.
        """
        # Try tag first via backend
        event = self._backend.get(key)
        if event is not None:
            return event

        # Fallback to UUID lookup
        event = self._backend.get_by_id(key)
        if event is not None:
            return event

        raise KeyError(f"No event with tag or id '{key}'")

    def __contains__(self, key: str) -> bool:
        """Check if tag or ID exists."""
        return self.get(key) is not None

    def __len__(self) -> int:
        """Number of recorded events."""
        return len(self._backend)

    # === Private Helpers ===

    def _all_originals(self, tag: str) -> list[EventBase]:
        """Recursively expand a Summary to get all original leaf events.

        This fully expands the tree, returning only non-summary events.

        Args:
            tag: The tag of the Summary (e.g., "2..40").

        Returns:
            List of all original (non-summary) events in the tree.
        """
        event = self._backend.get(tag)
        if event is None:
            return []
        if not isinstance(event, Summary):
            return [event]

        result: list[EventBase] = []
        for child_tag in event.children_tags:
            child = self._backend.get(child_tag)
            if child is None:
                continue
            if isinstance(child, Summary):
                result.extend(self._all_originals(child.summary_tag))
            else:
                result.append(child)
        return result

    def _parse_tag_start(self, tag: str) -> int:
        """Get the start number from a tag ("5" -> 5, "2..40" -> 2).

        Raises:
            ValueError: If tag is not a valid numeric tag format.
        """
        try:
            if ".." in tag:
                return int(tag.split("..")[0])
            return int(tag)
        except ValueError as e:
            raise ValueError(
                f"Invalid tag format '{tag}': tags must be numeric (e.g., '5') "
                f"or ranges (e.g., '2..40')"
            ) from e

    def _parse_tag_end(self, tag: str) -> int:
        """Get the end number from a tag ("5" -> 5, "2..40" -> 40).

        Raises:
            ValueError: If tag is not a valid numeric tag format.
        """
        try:
            if ".." in tag:
                return int(tag.split("..")[1])
            return int(tag)
        except ValueError as e:
            raise ValueError(
                f"Invalid tag format '{tag}': tags must be numeric (e.g., '5') "
                f"or ranges (e.g., '2..40')"
            ) from e

    def _validate_tag(self, tag: str) -> None:
        """Validate that a tag is in correct format.

        Tags must be either:
        - Simple numeric: "1", "42", "100"
        - Range format: "2..40", "1..5"

        Raises:
            ValueError: If tag format is invalid.
        """
        if ".." in tag:
            parts = tag.split("..")
            if len(parts) != 2:
                raise ValueError(f"Invalid range tag '{tag}': must have exactly one '..' separator")
            try:
                start = int(parts[0])
                end = int(parts[1])
            except ValueError as e:
                raise ValueError(
                    f"Invalid range tag '{tag}': start and end must be integers"
                ) from e
            if start > end:
                raise ValueError(
                    f"Invalid range tag '{tag}': start ({start}) must be <= end ({end})"
                )
        else:
            try:
                int(tag)
            except ValueError as e:
                raise ValueError(
                    f"Invalid tag '{tag}': must be a numeric string (e.g., '5')"
                ) from e
