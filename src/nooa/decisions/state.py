# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Encode selected context items as decision-model supporting state."""

from collections.abc import Sequence
from typing import Any

from nooa.context_blocks import EventBase, Role
from nooa.context_blocks.exceptions import UnsupportedContextLayout
from nooa.context_view import Block, CacheBoundary, ContextItem


def decision_context_from_items(
    items: Sequence[ContextItem],
) -> tuple[dict[str, Any], list[dict[str, Any]] | None]:
    """Encode unique supporting blocks followed by events without selecting content.

    Decision state has no conversation roles for supporting blocks. Only SYSTEM
    blocks map to its context dictionary; event roles remain part of event data.
    Unrepresentable layouts fail rather than reorder or overwrite view output.
    """
    context: dict[str, Any] = {}
    events: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, CacheBoundary):
            continue  # Decision requests do not support explicit cache boundaries.
        if isinstance(item, Block):
            if item.role != Role.SYSTEM:
                raise UnsupportedContextLayout(
                    f"Decision context block {item.key!r} requires role SYSTEM, got {item.role.value}"
                )
            if events:
                raise UnsupportedContextLayout(
                    f"Decision context block {item.key!r} follows an event; blocks must precede events"
                )
            if item.key in context:
                raise UnsupportedContextLayout(f"Duplicate decision context block key {item.key!r}")
            context[item.key] = item.content
        elif isinstance(item, EventBase):
            events.append(_decision_event_payload(item))
        else:
            raise TypeError(f"Unsupported decision context item: {type(item).__name__}")
    return context, events or None


def _decision_event_payload(event: EventBase) -> dict[str, Any]:
    """Serialize one view-selected event, without applying another visibility filter."""
    public_fields = {name for name, field in type(event).model_fields.items() if field.repr}
    data = event.model_dump(mode="python", include=public_fields, exclude_none=True)
    replay_content = getattr(event, "replay_content", None)
    if not data and isinstance(replay_content, str):
        data = {"content": replay_content}
    return {"type": event.event_type, "role": event._role.value, "data": data}


__all__ = ["decision_context_from_items"]
