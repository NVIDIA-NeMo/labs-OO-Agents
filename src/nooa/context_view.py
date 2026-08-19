# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context assembly contracts and default views."""

from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from nooa.context_blocks.events import EventBase
from nooa.context_blocks.models import BlockMetadata, Role

if TYPE_CHECKING:
    from nooa.agent import Agent
    from nooa.skill import Skill
    from nooa.strategies.current_call import CurrentCall


class Block(BaseModel):
    """A materialized context block."""

    model_config = ConfigDict(frozen=True)

    key: str
    content: str
    role: Role = Role.SYSTEM


class _MaterializedBlock(Block):
    """Internal compatibility data used while legacy state APIs migrate."""

    metadata: BlockMetadata = BlockMetadata()


type ContextItem = Block | EventBase
type AssembledContext = tuple[ContextItem, ...]


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


async def assemble_context[Owner](
    view: ContextView[Owner], owner: Owner, call: "CurrentCall"
) -> AssembledContext:
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


class DefaultAgentView(ContextView["Agent"]):
    """Compatibility view over the current agent state APIs."""

    async def assemble(self, owner: "Agent", call: "CurrentCall") -> AsyncIterator[ContextItem]:
        async for item in owner.runtime._assemble_default_context(call):
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
        yield await agent.runtime._materialize_skill_context_block(key, expression, call)


def _apply_default_budget(
    blocks: list[Any], total_limit: int, count_fn: Callable[[str], int]
) -> tuple[list[Any], int]:
    """DefaultAgentView policy: evict volatile manager blocks newest first."""
    total = sum(count_fn(block.content) for block in blocks)
    if total <= total_limit:
        return blocks, 0

    to_evict: set[int] = set()
    sizes: dict[int, int] = {}
    for user_only in (True, False):
        for index in range(len(blocks) - 1, -1, -1):
            if total <= total_limit:
                break
            block = blocks[index]
            if index in to_evict or block.metadata.static:
                continue
            if user_only and not block.metadata.user_block:
                continue
            size = count_fn(block.content)
            total -= size
            to_evict.add(index)
            sizes[index] = size

    result = []
    for index, block in enumerate(blocks):
        if index not in to_evict:
            result.append(block)
            continue
        result.append(
            block.model_copy(
                update={
                    "content": (f"EVICTED: over context budget (block_tokens={sizes[index]:,})"),
                    "metadata": block.metadata.model_copy(update={"truncated": True}),
                }
            )
        )
    return result, len(to_evict)


__all__ = [
    "AssembledContext",
    "Block",
    "ContextItem",
    "ContextView",
    "DefaultAgentView",
    "DefaultSkillView",
    "assemble_context",
    "resolve_context_view",
]
