# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gather the context blocks and events that go into a decision request's state.

A decision model only sees what the decision method opts in to. These helpers
resolve the context blocks selected for a call and serialize the events chosen
by an event query, so ``render_decision_state`` can add them to the JSON state.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from nooa.context_blocks import Context, DynamicContext, Role
from nooa.decisions.media import reject_nested_media

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
    """Merge and resolve the context blocks selected for a decision call.

    Blocks declared on the method's decorator are merged with blocks from an
    active ``ScopedContext``; a scoped block replaces a decorator block with the same
    key. Keys in ``disabled_keys`` and ``None`` values are skipped. Dynamic
    blocks are evaluated through ``resolve_dynamic``.

    Returns:
        A mapping from block key to its resolved value.
    """
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
    """Select and serialize the events to include in a decision's state.

    The first query that is set wins, in this order: the runtime query, the
    scoped query, the decorator query, then the agent's default query.

    Returns:
        ``None`` when no query is set (events are left out of the state),
        otherwise the serialized events selected by the query.
    """
    query = runtime_query or scoped_query or decorator_query or agent_query
    if query is None:
        return None

    selected = query.apply(list(events), current_call_id=current_call_id)
    return [
        payload for event in selected if (payload := _decision_event_payload(event)) is not None
    ]


def _decision_event_payload(event: EventBase) -> dict[str, Any] | None:
    """Serialize one event's model-visible fields, or return ``None`` to skip it.

    Runtime and metadata events, and empty events, are skipped. Only fields
    shown in ``repr`` are included; fields with ``repr=False`` hold internal
    or storage data and are left out.

    Raises:
        TypeError: If a model-visible field contains a Media object.
    """
    role = getattr(event, "_role", Role.USER)
    if role in {Role.RUNTIME_EVENT, Role.METADATA} or event.is_empty:
        return None

    public_fields = {name for name, field in type(event).model_fields.items() if field.repr}
    # Check for Media before calling model_dump: a custom serializer could
    # turn a nested Image into text. Fields with repr=False are not checked
    # because they are not sent to the model.
    for name in public_fields:
        try:
            reject_nested_media(getattr(event, name))
        except TypeError as exc:
            raise TypeError(f"Decision event {event.event_type!r} field {name!r}: {exc}") from exc
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
