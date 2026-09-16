# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Composable context assembly contracts and reusable mechanisms."""

from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict

from nooa.context_blocks.events import EventBase
from nooa.context_blocks.models import BlockMetadata
from nooa.context_blocks.roles import Role
from nooa.llm_types import CacheBoundary

if TYPE_CHECKING:
    from nooa.strategies.current_call import CurrentCall


class Block(BaseModel):
    """One materialized context block."""

    model_config = ConfigDict(frozen=True)

    key: str
    content: str
    role: Role = Role.SYSTEM
    metadata: BlockMetadata | None = None


type ContextItem = Block | EventBase | CacheBoundary


@runtime_checkable
class ContextView[Owner](Protocol):
    """Produce an ordered context stream for an owner."""

    def assemble(
        self,
        owner: Owner,
        call: "CurrentCall",
    ) -> AsyncIterator[ContextItem]: ...


def resolve_context_view(owner: Any, *, default: ContextView[Any]) -> ContextView[Any]:
    """Resolve through ``owner.__context_view__()``, otherwise use ``default``."""
    resolver = getattr(owner, "__context_view__", None)
    if resolver is None:
        return default
    view = resolver()
    return cast(ContextView[Any], view if view is not None else default)


def select_context_events(events: Any, *, call: "CurrentCall") -> tuple[EventBase, ...]:
    """Select active, model-visible events using the call's resolved query."""
    active = tuple(event for key in events.keys() if (event := events.get(key)) is not None)
    selected = active
    if call.event_query is not None:
        selected = tuple(call.event_query.apply(list(active), current_call_id=call.invocation_id))
    return tuple(
        event
        for event in selected
        if getattr(event, "_role", Role.USER) not in (Role.RUNTIME_EVENT, Role.METADATA)
        and not event.is_empty
    )


async def collect_context_items[Owner](
    view: ContextView[Owner], owner: Owner, call: "CurrentCall"
) -> tuple[ContextItem, ...]:
    """Collect and validate a view into one immutable snapshot."""
    items: list[ContextItem] = []
    async for item in view.assemble(owner, call):
        if not isinstance(item, (Block, EventBase, CacheBoundary)):
            raise TypeError(
                f"{type(view).__name__}.assemble() yielded {type(item).__name__}; "
                "expected Block, EventBase, or CacheBoundary"
            )
        items.append(item)
    return tuple(items)


def context_text(value: Any, *, call: "CurrentCall") -> str:
    """Render a value with the call's bounded context-block format."""
    if isinstance(value, str):
        return value

    from nooa.agentdoc import pformat

    kwargs = call.context_format.model_dump() if call.context_format is not None else {}
    return pformat(value, unquote_strings=True, **kwargs)


async def evaluate_context_expression(
    expression: str,
    *,
    owner: Any,
    call: "CurrentCall",
) -> Any:
    """Evaluate a declarative expression with ``self`` bound to ``owner``."""
    agent = call.agent
    runtime = getattr(agent, "runtime", None)
    if runtime is None:
        raise RuntimeError("Expression evaluation requires an agent-bound CurrentCall")

    return await runtime.evaluate_expression(
        expression,
        extra_context={
            "self": owner,
            "call": call,
            "method": call.method,
            "call_args": call.args,
            "call_kwargs": call.kwargs,
            "strategy": call.strategy,
            "datetime": datetime,
            "runtime": runtime,
        },
        error_mode="raise",
    )


def apply_context_budget(
    items: Sequence[ContextItem],
    *,
    call: "CurrentCall",
    evictable: Sequence[Block],
) -> tuple[ContextItem, ...]:
    """Apply the call budget by replacing candidates in the supplied priority."""
    limit = call.context_budget
    if limit is None:
        return tuple(items)
    counter = call.context_token_counter
    if counter is None:
        from nooa.token_counter import char_approximate_token_counter

        counter = char_approximate_token_counter

    result = list(items)
    total = sum(counter(item.content) for item in result if isinstance(item, Block))
    if total <= limit:
        return tuple(result)

    by_identity = {id(item): index for index, item in enumerate(result)}
    for candidate in evictable:
        if total <= limit:
            break
        index = by_identity.get(id(candidate))
        if index is None:
            continue
        block = result[index]
        if not isinstance(block, Block):
            continue
        size = counter(block.content)
        replacement = f"EVICTED: over context budget (block_tokens={size:,})"
        update: dict[str, Any] = {"content": replacement}
        if block.metadata is not None:
            update["metadata"] = block.metadata.model_copy(update={"truncated": True})
        result[index] = block.model_copy(update=update)
        total += counter(replacement) - size
    return tuple(result)


__all__ = [
    "Block",
    "CacheBoundary",
    "ContextItem",
    "ContextView",
    "apply_context_budget",
    "collect_context_items",
    "context_text",
    "evaluate_context_expression",
    "resolve_context_view",
    "select_context_events",
]
