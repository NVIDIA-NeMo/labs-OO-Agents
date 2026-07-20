# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scoped context block overrides.

Provides a context manager for temporary block overrides that apply
during a `with` block and are automatically cleaned up. Can also be used
as a parameter to the @strategy decorator for methods with ellipsis bodies.

Usage:
    # As a context manager:
    with ScopedContext(context={"focus": "Analyze security only"}):
        result = await agent.analyze(data)  # sees the "focus" block

    # As a decorator parameter (for ellipsis methods):
    from nooa.runtime import EventQuery
    @strategy(CodeActStrategy(), ScopedContext(events=EventQuery.current_call()))
    async def my_method(self):
        ...  # Cannot use 'with' statement in ellipsis body
"""

import contextvars
from typing import TYPE_CHECKING, Any

from nooa.context_blocks.models import DynamicContext

if TYPE_CHECKING:
    from nooa.runtime.event_query import EventQuery

# Context variable for scoped block overrides (inherits to nested calls).
# Structure: {key: str | DynamicContext | None, ...} or None
# Read by _prepare_context() in the runtime to apply temporary overrides.
_scoped_blocks_var: contextvars.ContextVar[dict[str, str | DynamicContext | None] | None] = (
    contextvars.ContextVar("scoped_blocks", default=None)
)

# Context variable for scoped event query (inherits to nested calls).
# Structure: EventQuery or None
# Read by _prepare_context() to filter events shown in context.
_scoped_events_var: contextvars.ContextVar["EventQuery | None"] = contextvars.ContextVar(
    "scoped_events", default=None
)


class ScopedContext:
    """Temporarily override context blocks and event filtering within a scope.

    Overrides apply to all LLM calls within the scope. Supports nesting:
    inner scopes inherit parent blocks and can override them.

    Can be used in two ways:
    1. As a context manager with `with` statement (for methods with bodies)
    2. As a parameter to @strategy decorator (for ellipsis methods that cannot use `with`)

    Args:
        context: Dict of context block overrides (system prompt).
            - str: Static content override
            - DynamicContext("expr"): DynamicContext expression override
            - None: Remove block within this scope
        events: EventQuery for filtering which events appear in context.
            Replaces the default event list with a filtered subset.
            See EventQuery for filtering options (by call_id, type, limit, etc.)

    Example:
        from nooa.runtime import EventQuery

        # As context manager - show only current method's events:
        with ScopedContext(events=EventQuery.current_call()):
            result = await agent.analyze(data)

        # As decorator parameter - filter to last 10 events:
        @strategy(
            CodeActStrategy(),
            ScopedContext(
                context={"focus": "Write clean code"},
                events=EventQuery.last_n(10)
            )
        )
        async def implement_feature(self, spec: str):
            ...  # Cannot use 'with' in ellipsis body

        # Nested scopes inherit and override:
        with ScopedContext(context={"a": "1"}):
            with ScopedContext(context={"b": "2"}, events=EventQuery.current_call()):
                # Both "a" and "b" are visible, plus filtered events
                pass
    """

    def __init__(
        self,
        context: dict[str, str | DynamicContext | None] | None = None,
        events: "EventQuery | None" = None,
    ):
        """Initialize scoped block and event filtering overrides.

        Args:
            context: Context block overrides to apply
            events: EventQuery for filtering events (replaces default event list)
        """
        self.context = context
        self.events = events
        self._ctx_token: contextvars.Token[Any] | None = None
        self._evt_token: contextvars.Token[Any] | None = None

    def __enter__(self) -> "ScopedContext":
        """Enter the context manager scope.

        Merges context with parent scope; overrides event query.
        """
        parent_ctx = _scoped_blocks_var.get()
        parent_evt = _scoped_events_var.get()

        # Merge context blocks with parent scope (inheritance)
        merged_ctx: dict[str, str | DynamicContext | None] = {}
        if parent_ctx:
            merged_ctx.update(parent_ctx)
        if self.context:
            merged_ctx.update(self.context)

        # Event query: use most specific (child overrides parent)
        active_evt = self.events if self.events is not None else parent_evt

        self._ctx_token = _scoped_blocks_var.set(merged_ctx)
        self._evt_token = _scoped_events_var.set(active_evt)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the context manager scope.

        Resets context variables to previous values.
        """
        if self._ctx_token is not None:
            _scoped_blocks_var.reset(self._ctx_token)
        if self._evt_token is not None:
            _scoped_events_var.reset(self._evt_token)
