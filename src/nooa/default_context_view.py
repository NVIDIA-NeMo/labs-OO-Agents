# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default context policy implemented against public framework interfaces."""

import logging
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from nooa.context_blocks import BlockMetadata, Context, DynamicContext, Role
from nooa.context_blocks.events import EventBase
from nooa.context_view import (
    Block,
    CacheBoundary,
    ContextItem,
    ContextView,
    apply_context_budget,
    collect_context,
    context_text,
    evaluate_context_expression,
    resolve_context_view,
    select_context_events,
)

if TYPE_CHECKING:
    from nooa.agent import Agent
    from nooa.skill import Skill
    from nooa.strategies.current_call import CurrentCall

logger = logging.getLogger(__name__)

_FRAMEWORK_KEYS = frozenset({"system_prompt", "self", "state"})


def _block(
    key: str,
    content: str,
    metadata: BlockMetadata,
    *,
    role: Role | None = None,
) -> Block:
    """Create a materialized block with placement represented by its role."""
    return Block(
        key=key,
        content=content,
        role=role or (Role.SYSTEM if metadata.static else Role.USER),
        metadata=metadata,
    )


async def _resolve_value(
    key: str,
    value: Any,
    *,
    agent: "Agent",
    call: "CurrentCall",
) -> str:
    """Resolve one declaration using the framework expression environment."""
    if not isinstance(value, DynamicContext):
        return context_text(value, call=call)
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


async def stored_context_blocks(
    manager: Any,
    agent: "Agent",
    call: "CurrentCall",
    *,
    include: frozenset[str] | None = None,
    exclude: frozenset[str] = frozenset(),
) -> tuple[Block, ...]:
    """Materialize selected declarations from one ContextManager snapshot."""
    blocks: list[Block] = []
    resolved_cache: dict[str, Any] = {}
    for key, value in manager.declarations():
        if (include is not None and key not in include) or key in exclude:
            continue
        if manager.is_disabled(key):
            continue

        is_dynamic = isinstance(value, DynamicContext)
        content = await _resolve_value(key, value, agent=agent, call=call)
        if is_dynamic:
            resolved_cache[key] = content
        protected = manager.is_protected(key)
        metadata = BlockMetadata(
            expr=value.expr if is_dynamic else f'self.context["{key}"]',
            user_block=not protected,
            static=manager.is_static(key),
            source_dynamic=is_dynamic,
        )
        blocks.append(_block(key, content, metadata))

    manager.update_resolved(resolved_cache)
    return tuple(blocks)


def resolve_agent_system_prompt(agent: "Agent") -> str:
    """Resolve the nearest agent class docstring as a prompt template."""
    import string

    prompt = next((cls.__doc__ for cls in type(agent).__mro__ if cls.__doc__), "")
    if not prompt or "{" not in prompt:
        return prompt

    parts: list[str] = []
    for literal, field, format_spec, conversion in string.Formatter().parse(prompt):
        parts.append(literal)
        if field is None:
            continue
        try:
            value = eval(field, {"self": agent, "type": type})  # noqa: S307
            if conversion == "r":
                value = repr(value)
            elif conversion == "s":
                value = str(value)
            if format_spec:
                value = format(value, format_spec)
            parts.append(str(value))
        except Exception:
            logger.debug("Prompt template eval failed for %r", field, exc_info=True)
            placeholder = field
            if conversion:
                placeholder = f"{field}!{conversion}"
            if format_spec:
                placeholder = f"{placeholder}:{format_spec}"
            parts.append("{" + placeholder + "}")
    return "".join(parts)


def _source_error(key: str, exc: Exception) -> str:
    logger.warning("Context source %r failed to resolve: %s: %s", key, type(exc).__name__, exc)
    return f"{type(exc).__name__}: {exc}"


async def system_prompt_block(agent: "Agent", call: "CurrentCall") -> Block | None:
    """Build the default system prompt directly from the agent."""
    try:
        content = resolve_agent_system_prompt(agent)
    except Exception as exc:
        content = _source_error("system_prompt", exc)
    return _block(
        "system_prompt",
        content,
        BlockMetadata(
            expr="self._resolve_system_prompt()",
            static=True,
            source_dynamic=True,
        ),
    )


async def agent_interface_block(agent: "Agent", call: "CurrentCall") -> Block | None:
    """Build the default documented agent interface directly from the agent."""
    from nooa.agentdoc import doc

    try:
        content = doc(type(agent))
    except Exception as exc:
        content = _source_error("self", exc)
    return _block(
        "self",
        content,
        BlockMetadata(expr="doc(type(self))", static=True, source_dynamic=True),
    )


async def agent_state_block(agent: "Agent", call: "CurrentCall") -> Block | None:
    """Build the default visible agent state directly from the agent."""
    from nooa.agentdoc import pformat

    try:
        content = pformat(agent, max_length=50, max_string=500, max_depth=4)
    except Exception as exc:
        content = _source_error("state", exc)
    return _block(
        "state",
        content,
        BlockMetadata(
            expr="pformat(self, max_length=50, max_string=500, max_depth=4)",
            user_block=False,
            static=False,
            source_dynamic=True,
        ),
    )


async def apply_context_overrides(
    blocks: Sequence[Block],
    overrides: Mapping[str, Any] | None,
    *,
    agent: "Agent",
    call: "CurrentCall",
    static_expr: Callable[[str], str],
    static_keys: set[str] | None = None,
    disabled: set[str] | None = None,
) -> tuple[Block, ...]:
    """Replace, append, or remove named blocks using one override source."""
    if not overrides:
        return tuple(blocks)

    disabled = disabled or set()
    result: list[Block | None] = list(blocks)
    index = {block.key: position for position, block in enumerate(blocks)}

    for key, declaration in overrides.items():
        position = index.get(key)
        if key in disabled or declaration is None:
            if position is not None:
                result[position] = None
            continue

        explicit_static: bool | None = None
        value = declaration
        if isinstance(value, Context):
            explicit_static = value.prefix
            if value.is_dynamic:
                assert value.expr is not None
                value = DynamicContext(value.expr)
            else:
                value = value.value

        existing = result[position] if position is not None else None
        existing_static = existing.metadata.static if existing and existing.metadata else None
        if explicit_static is not None:
            is_static = explicit_static
        elif existing_static is not None:
            is_static = existing_static
        elif static_keys and key in static_keys:
            is_static = True
        else:
            is_static = not isinstance(value, DynamicContext)

        content = await _resolve_value(key, value, agent=agent, call=call)
        metadata = BlockMetadata(
            expr=value.expr if isinstance(value, DynamicContext) else static_expr(key),
            static=is_static,
        )
        replacement = _block(key, content, metadata)
        if position is None:
            index[key] = len(result)
            result.append(replacement)
        else:
            result[position] = replacement

    return tuple(block for block in result if block is not None)


def order_blocks(blocks: Sequence[Block], preferred: Sequence[str] | None) -> tuple[Block, ...]:
    """Place preferred keys first and preserve relative order for all others."""
    if preferred is None:
        return tuple(blocks)
    rank = {key: position for position, key in enumerate(preferred)}
    ordered = sorted(
        enumerate(blocks),
        key=lambda pair: (rank.get(pair[1].key, len(rank)), pair[0]),
    )
    return tuple(block for _, block in ordered)


def replace_blocks_by_key(
    blocks: Sequence[Block], replacements: Sequence[Block]
) -> tuple[Block, ...]:
    """Replace matching blocks in place and append new keys in source order."""
    result = list(blocks)
    index = {block.key: position for position, block in enumerate(result)}
    for replacement in replacements:
        position = index.get(replacement.key)
        if position is None:
            index[replacement.key] = len(result)
            result.append(replacement)
        else:
            result[position] = replacement
    return tuple(result)


def partition_blocks(blocks: Sequence[Block]) -> tuple[tuple[Block, ...], tuple[Block, ...]]:
    """Partition cacheable prefix blocks from volatile trailing blocks."""
    prefix = tuple(block for block in blocks if block.metadata and block.metadata.static)
    trailing = tuple(block for block in blocks if not block.metadata or not block.metadata.static)
    return prefix, trailing


def visible_events(agent: "Agent", call: "CurrentCall") -> tuple[EventBase, ...]:
    """Select model-visible events using the invocation's resolved query."""
    return select_context_events(agent.events, call=call)


class DefaultAgentView(ContextView["Agent"]):
    """Readable reference implementation of NOOA's default context policy."""

    async def assemble(self, owner: "Agent", call: "CurrentCall") -> AsyncIterator[ContextItem]:
        manager = owner.context_manager
        disabled = manager.disabled()
        declaration_keys = {key for key, _ in manager.declarations()}
        default_builders = (
            ("system_prompt", system_prompt_block),
            ("self", agent_interface_block),
            ("state", agent_state_block),
        )
        blocks_list: list[Block] = []
        for key, build in default_builders:
            if key in disabled:
                continue
            if key in declaration_keys:
                overrides = await stored_context_blocks(
                    manager, owner, call, include=frozenset({key})
                )
                block = overrides[0] if overrides else None
            elif not manager.is_protected(key):
                continue
            else:
                block = await build(owner, call)
                if block is not None:
                    manager.update_resolved({key: block.content})
            if block is not None:
                blocks_list.append(block)

        blocks: tuple[Block, ...] = tuple(blocks_list)
        blocks += await stored_context_blocks(manager, owner, call, exclude=_FRAMEWORK_KEYS)

        custom_skill_items: list[ContextItem] = []
        for skill in owner.active_skills():
            default_skill_view = DefaultSkillView()
            view = resolve_context_view(skill, default=default_skill_view)
            contribution = await collect_context(view, skill, call)
            if view is default_skill_view:
                default_blocks = tuple(
                    item
                    for item in contribution
                    if isinstance(item, Block)
                    and item.key not in disabled
                    and not manager.is_protected(item.key)
                )
                blocks = replace_blocks_by_key(blocks, default_blocks)
            else:
                custom_skill_items.extend(contribution)

        strategy = call.strategy
        if strategy is not None:
            get_overrides = getattr(strategy, "get_block_overrides", None)
            blocks = await apply_context_overrides(
                blocks,
                get_overrides() if get_overrides is not None else None,
                agent=owner,
                call=call,
                static_expr=lambda key: f"strategy.{key}",
                static_keys=getattr(strategy, "get_static_block_keys", lambda: set())(),
                disabled=disabled,
            )

        blocks = await apply_context_overrides(
            blocks,
            call.decorator_context,
            agent=owner,
            call=call,
            static_expr=lambda key: f'@strategy.context["{key}"]',
            disabled=disabled,
        )
        blocks = await apply_context_overrides(
            blocks,
            call.scoped_context,
            agent=owner,
            call=call,
            static_expr=lambda key: f'self.context["{key}"]',
            disabled=disabled,
        )
        get_order = getattr(strategy, "get_block_order", None) if strategy is not None else None
        blocks = order_blocks(blocks, get_order() if get_order is not None else None)
        prefix, trailing = partition_blocks(blocks)

        items: list[ContextItem] = [*prefix, *custom_skill_items]
        items.extend(visible_events(owner, call))
        if items:
            items.append(CacheBoundary())
        items.extend(trailing)

        evictable = tuple(
            block
            for user_only in (True, False)
            for block in reversed(trailing)
            if bool(block.metadata and block.metadata.user_block) is user_only
        )
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
        value = await _resolve_value(key, DynamicContext(expression), agent=agent, call=call)
        yield _block(
            key,
            value,
            BlockMetadata(expr=expression, user_block=True, source_dynamic=True),
        )


__all__ = [
    "DefaultAgentView",
    "DefaultSkillView",
    "agent_interface_block",
    "agent_state_block",
    "apply_context_overrides",
    "order_blocks",
    "partition_blocks",
    "replace_blocks_by_key",
    "stored_context_blocks",
    "system_prompt_block",
    "visible_events",
]
