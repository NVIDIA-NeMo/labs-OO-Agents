# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve explicitly selected context and events for decision-model state."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from nooa.context_blocks import Context, DynamicContext, Role

if TYPE_CHECKING:
    from nooa.context_blocks import EventBase
    from nooa.runtime.event_query import EventQuery

type DynamicResolver = Callable[[str, DynamicContext], Awaitable[Any]]


async def resolve_decision_context(
    *,
    decorator_context: Mapping[str, Any] | None,
    scoped_context: Mapping[str, Any] | None,
    disabled_keys: set[str],
    resolve_dynamic: DynamicResolver,
) -> dict[str, Any]:
    """Merge and resolve context blocks explicitly selected for a decision."""
    selected = {**(decorator_context or {}), **(scoped_context or {})}
    resolved: dict[str, Any] = {}
    for key, raw_value in selected.items():
        if key in disabled_keys or raw_value is None:
            continue
        value = raw_value
        if isinstance(value, Context):
            value = value.to_dynamic_context() if value.is_dynamic else value.value
        if isinstance(value, DynamicContext):
            value = await resolve_dynamic(key, value)
        resolved[key] = value
    return resolved


def select_decision_events(
    events: Sequence[EventBase],
    *,
    runtime_query: EventQuery | None,
    scoped_query: EventQuery | None,
    decorator_query: EventQuery | None,
    agent_query: EventQuery | None,
    current_call_id: str | None,
) -> list[dict[str, Any]] | None:
    """Select and serialize events when an effective event query opts in."""
    query = runtime_query or scoped_query or decorator_query or agent_query
    if query is None:
        return None

    selected = query.apply(list(events), current_call_id=current_call_id)
    return [
        payload for event in selected if (payload := _decision_event_payload(event)) is not None
    ]


def _decision_event_payload(event: EventBase) -> dict[str, Any] | None:
    """Return stable model-visible event data without persistence metadata."""
    role = getattr(event, "_role", Role.USER)
    if role in {Role.RUNTIME_EVENT, Role.METADATA} or event.is_empty:
        return None

    public_fields = {name for name, field in type(event).model_fields.items() if field.repr}
    data = event.model_dump(
        mode="python",
        include=public_fields,
        exclude_none=True,
    )
    replay_content = getattr(event, "replay_content", None)
    if not data and isinstance(replay_content, str):
        data = {"content": replay_content}
    return {
        "type": event.event_type,
        "role": role.value,
        "data": data,
    }


__all__ = ["resolve_decision_context", "select_decision_events"]
