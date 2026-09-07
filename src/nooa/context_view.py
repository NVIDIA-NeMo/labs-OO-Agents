# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Composable context assembly contracts, helpers, and default policy."""

import logging
from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from nooa.context_blocks.events import EventBase
from nooa.context_blocks.models import BlockMetadata, DynamicContext, Role

if TYPE_CHECKING:
    from nooa.agent import Agent
    from nooa.skill import Skill
    from nooa.strategies.current_call import CurrentCall

logger = logging.getLogger(__name__)


class Block(BaseModel):
    """One materialized context block."""

    model_config = ConfigDict(frozen=True)

    key: str
    content: str
    role: Role = Role.SYSTEM


class _MaterializedBlock(Block):
    """Manager-backed block metadata used for compatibility metrics."""

    metadata: BlockMetadata = BlockMetadata()


type ContextItem = Block | EventBase


@runtime_checkable
class ContextView[Owner](Protocol):
    """Produce an ordered context stream for an owner."""

    def assemble(
        self,
        owner: Owner,
        call: "CurrentCall",
    ) -> AsyncIterator[ContextItem]: ...


def resolve_context_view(owner: Any, *, default: ContextView[Any]) -> ContextView[Any]:
    """Resolve instance, then class, then default view."""
    instance_view = vars(owner).get("_context_view")
    if instance_view is not None:
        return instance_view
    class_view = getattr(type(owner), "_context_view", None)
    return class_view if class_view is not None else default


async def collect_context[Owner](
    view: ContextView[Owner], owner: Owner, call: "CurrentCall"
) -> tuple[ContextItem, ...]:
    """Collect and validate a view into one immutable snapshot."""
    items: list[ContextItem] = []
    async for item in view.assemble(owner, call):
        if not isinstance(item, (Block, EventBase)):
            raise TypeError(
                f"{type(view).__name__}.assemble() yielded {type(item).__name__}; "
                "expected Block or EventBase"
            )
        items.append(item)
    return tuple(items)


def context_text(value: Any, *, call: "CurrentCall") -> str:
    """Render a value with the call's bounded context-block format."""
    if isinstance(value, str):
        return value

    from nooa.agentdoc import pformat

    kwargs = call._context_format.model_dump() if call._context_format is not None else {}
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
            "method": call._method,
            "call_args": call.args,
            "call_kwargs": call.kwargs,
            "strategy": call.strategy,
            "datetime": datetime,
            "runtime": runtime,
        },
        error_mode="raise",
    )


async def materialize_managed_context(
    agent: "Agent", call: "CurrentCall"
) -> tuple[tuple[Block, ...], tuple[Block, ...], tuple[Block, ...]]:
    """Translate ContextManager state into prefix, trailing, and eviction order."""
    from nooa.runtime.context_builder import build_managed_context

    async def resolve_value(key: str, value: str | DynamicContext) -> str:
        if not isinstance(value, DynamicContext):
            return value
        try:
            resolved = await evaluate_context_expression(value.expr, owner=agent, call=call)
        except Exception as exc:
            logger.warning(
                "Dynamic context block %r failed to resolve: %s: %s (expr: %s)",
                key,
                type(exc).__name__,
                exc,
                value.expr,
            )
            return f"{type(exc).__name__}: {exc}"
        return context_text(resolved, call=call)

    result = await build_managed_context(
        context_manager=agent.context_manager,
        strategy=call.strategy,
        resolve_fn=resolve_value,
        decorator_context=call._decorator_context,
        scoped_context=call._scoped_context,
        context_block_format=call._context_format,
    )
    agent.context_manager._update_resolved(result.resolved_cache)

    blocks = tuple(
        _MaterializedBlock(
            key=block.key,
            content=block.content,
            role=block.role if block.metadata.static else Role.USER,
            metadata=block.metadata,
        )
        for block in result.blocks
    )
    prefix = tuple(block for block in blocks if block.metadata.static)
    trailing = tuple(block for block in blocks if not block.metadata.static)
    evictable = tuple(
        block
        for user_only in (True, False)
        for block in reversed(trailing)
        if block.metadata.user_block is user_only
    )
    return prefix, trailing, evictable


def events_from_manager(agent: "Agent", call: "CurrentCall") -> tuple[EventBase, ...]:
    """Select visible events from an agent's EventManager for this call."""
    events = agent.event_manager.values()
    if call.event_query is not None:
        events = call.event_query.apply(events, current_call_id=call._context_call_id or call.id)
    return tuple(
        event
        for event in events
        if getattr(event, "_role", Role.USER) not in (Role.RUNTIME_EVENT, Role.METADATA)
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
    counter = call._context_token_counter
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
        update: dict[str, Any] = {
            "content": f"EVICTED: over context budget (block_tokens={size:,})"
        }
        if isinstance(block, _MaterializedBlock):
            update["metadata"] = block.metadata.model_copy(update={"truncated": True})
        result[index] = block.model_copy(update=update)
        total -= size
    return tuple(result)


class DefaultAgentView(ContextView["Agent"]):
    """Default assembly policy over optional manager and skill sources."""

    async def assemble(self, owner: "Agent", call: "CurrentCall") -> AsyncIterator[ContextItem]:
        prefix, trailing, evictable = await materialize_managed_context(owner, call)
        items: list[ContextItem] = [*prefix]

        for skill in owner.active_skills():
            view = resolve_context_view(skill, default=DefaultSkillView())
            items.extend(await collect_context(view, skill, call))

        items.extend(events_from_manager(owner, call))
        items.extend(trailing)
        for item in apply_context_budget(items, call=call, evictable=evictable):
            yield item


class DefaultSkillView(ContextView["Skill"]):
    """Materialize the legacy ``Skill.context_block`` shorthand."""

    async def assemble(self, owner: "Skill", call: "CurrentCall") -> AsyncIterator[ContextItem]:
        declaration = owner.context_block
        if declaration is None:
            return
        agent = call.agent
        if agent is None:
            raise RuntimeError("DefaultSkillView requires an agent-bound CurrentCall")
        key, expression = declaration
        try:
            value = await evaluate_context_expression(expression, owner=agent, call=call)
        except Exception as exc:
            logger.warning(
                "Skill context block %r failed to resolve: %s: %s",
                key,
                type(exc).__name__,
                exc,
            )
            value = f"{type(exc).__name__}: {exc}"
        yield _MaterializedBlock(key=key, content=context_text(value, call=call))


__all__ = [
    "Block",
    "ContextItem",
    "ContextView",
    "DefaultAgentView",
    "DefaultSkillView",
    "apply_context_budget",
    "collect_context",
    "context_text",
    "evaluate_context_expression",
    "events_from_manager",
    "materialize_managed_context",
    "resolve_context_view",
]
