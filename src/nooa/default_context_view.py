# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default context policy implemented against public framework interfaces."""

import inspect
import logging
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from nooa.context_blocks import BlockMetadata, Context, DynamicContext, Role
from nooa.context_blocks.events import EventBase, ResultStatus, ToolCallEvent, ToolResult
from nooa.context_view import (
    Block,
    CacheBoundary,
    ContextItem,
    ContextView,
    apply_context_budget,
    context_text,
    evaluate_context_expression,
    is_model_visible,
)

if TYPE_CHECKING:
    from nooa.agent import Agent
    from nooa.skill import Skill
    from nooa.strategies.current_call import CurrentCall

logger = logging.getLogger(__name__)

_FRAMEWORK_KEYS = frozenset({"system_prompt", "self", "state", "events_api"})


def active_skills(agent: "Agent") -> tuple["Skill", ...]:
    """Return visible attached skills in stable attribute order."""
    from nooa.skill import Skill

    values = tuple(agent.__instance_values__().values())
    visible_ids = {id(value) for value in values}
    managed_ids = {
        id(skill)
        for value in values
        if isinstance(value, Skill)
        for skill in getattr(value, "_managed_skills", lambda: ())()
    }
    skills: list[Skill] = []
    seen: set[int] = set()
    for value in values:
        if not isinstance(value, Skill) or id(value) in seen:
            continue
        if id(value) in managed_ids:
            continue
        seen.add(id(value))
        skills.append(value)
        active = getattr(value, "active_skills", None)
        if callable(active):
            for nested in cast("tuple[Skill, ...]", active()):
                if id(nested) in visible_ids and id(nested) not in seen:
                    seen.add(id(nested))
                    skills.append(nested)
    return tuple(skills)


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


async def materialize_manager_blocks(
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
        value = agent._resolve_system_prompt()
        if inspect.isawaitable(value):
            value = await value
        content = context_text(value, call=call)
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


async def events_api_block(agent: "Agent", call: "CurrentCall") -> Block | None:
    """Describe the events API only when it is exposed to the model."""
    content = agent._events_api_prompt()
    if not content:
        return None
    return _block(
        "events_api",
        content,
        BlockMetadata(expr="self._events_api_prompt()", static=True, source_dynamic=True),
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


async def apply_block_overrides(
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


def visible_events(agent: "Agent") -> tuple[EventBase, ...]:
    """Select active history, projecting unavailable tool results without changing state."""
    events = [event for event in agent.event_manager.values() if is_model_visible(event)]

    def unavailable(call_id: str) -> ToolResult:
        return ToolResult(
            tool_call_id=call_id,
            content="Tool result unavailable in the active history; execution outcome is unknown.",
            result_status=ResultStatus.ERROR,
        )

    projected: list[EventBase] = []
    for event in events:
        if isinstance(event, ToolCallEvent) and event.result is None:
            event = event.model_copy(update={"result": unavailable(event.tool_call_id)})
        projected.append(event)
    # Do not reconstruct absent execution records: they may be deliberately
    # archived. Incomplete canonical batches remain omitted by atomic replay.
    return tuple(projected)


async def collect_skill_blocks(
    agent: "Agent",
    call: "CurrentCall",
    *,
    disabled: set[str],
    declaration_keys: frozenset[str],
) -> tuple[Block, ...]:
    """Evaluate active skill declarations without selecting their global placement."""
    blocks: tuple[Block, ...] = ()
    for skill in active_skills(agent):
        if skill.context_block is None:
            continue
        key, expression = skill.context_block
        if key in disabled or key in declaration_keys or agent.context_manager.is_protected(key):
            continue
        value = await _resolve_value(key, DynamicContext(expression), agent=agent, call=call)
        block = _block(
            key, value, BlockMetadata(expr=expression, user_block=True, source_dynamic=True)
        )
        blocks = replace_blocks_by_key(blocks, [block])
    return blocks


class ManagedContextView(ContextView["Agent"]):
    """Reusable assembly for the built-in manager-backed context policies."""

    block_order: tuple[str, ...] | None = None
    static_keys: frozenset[str] = frozenset()

    def instruction_blocks(self, call: "CurrentCall") -> Mapping[str, Any]:
        """Return this view's instruction declarations."""
        return {}

    async def _materialize_agent_blocks(
        self,
        owner: "Agent",
        call: "CurrentCall",
        *,
        disabled: set[str],
        declaration_keys: frozenset[str],
    ) -> tuple[Block, ...]:
        manager = owner.context_manager
        default_builders = (
            ("system_prompt", system_prompt_block),
            ("self", agent_interface_block),
            ("events_api", events_api_block),
            ("state", agent_state_block),
        )
        blocks_list: list[Block] = []
        for key, build in default_builders:
            if key in disabled:
                continue
            if key in declaration_keys:
                overrides = await materialize_manager_blocks(
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
        blocks += await materialize_manager_blocks(manager, owner, call, exclude=_FRAMEWORK_KEYS)
        return blocks

    async def _apply_override_layers(
        self,
        owner: "Agent",
        call: "CurrentCall",
        blocks: tuple[Block, ...],
        *,
        disabled: set[str],
    ) -> tuple[Block, ...]:
        blocks = await apply_block_overrides(
            blocks,
            self.instruction_blocks(call),
            agent=owner,
            call=call,
            static_expr=lambda key: f"strategy.{key}",
            static_keys=set(self.static_keys),
            disabled=disabled,
        )

        blocks = await apply_block_overrides(
            blocks,
            call.decorator_context,
            agent=owner,
            call=call,
            static_expr=lambda key: f'@strategy.context["{key}"]',
            disabled=disabled,
        )
        blocks = await apply_block_overrides(
            blocks,
            call.scoped_context,
            agent=owner,
            call=call,
            static_expr=lambda key: f'self.context["{key}"]',
            disabled=disabled,
        )
        return blocks

    async def assemble(self, owner: "Agent", call: "CurrentCall") -> AsyncIterator[ContextItem]:
        manager = owner.context_manager
        disabled = manager.disabled()
        from nooa.agentdoc.visibility import is_hidden_field

        if is_hidden_field(owner, "events"):
            disabled = disabled | {"events_api"}
        declaration_keys = frozenset(key for key, _ in manager.declarations())

        blocks = await self._materialize_agent_blocks(
            owner,
            call,
            disabled=disabled,
            declaration_keys=declaration_keys,
        )
        default_prompt = next(
            (
                block
                for block in blocks
                if block.key == "system_prompt" and "system_prompt" not in declaration_keys
            ),
            None,
        )
        skill_blocks = await collect_skill_blocks(
            owner,
            call,
            disabled=disabled,
            declaration_keys=declaration_keys,
        )
        blocks = replace_blocks_by_key(blocks, skill_blocks)
        blocks = await self._apply_override_layers(
            owner,
            call,
            blocks,
            disabled=disabled,
        )

        # Only a surviving built-in prompt receives guidance. Wait until all
        # overrides have run: even evaluating discarded guidance can fail or
        # have side effects. Identity also detects a same-text replacement.
        if default_prompt is not None and any(block is default_prompt for block in blocks):
            from nooa.agent import Agent

            prompt_class = next((cls for cls in type(owner).__mro__ if cls.__doc__), None)
            if (
                prompt_class is Agent
                and type(owner)._resolve_system_prompt is Agent._resolve_system_prompt
                and "_resolve_system_prompt" not in vars(owner)
            ):
                guidance = owner.render_config.block_formatter.format_description().strip()
                if guidance:
                    guided_prompt = default_prompt.model_copy(
                        update={
                            "content": context_text(default_prompt.content, call=call)
                            + "\n\n## Context format\n"
                            + guidance
                        }
                    )
                    blocks = tuple(
                        guided_prompt if block is default_prompt else block for block in blocks
                    )
                    manager.update_resolved({"system_prompt": guided_prompt.content})

        blocks = order_blocks(blocks, self.block_order)
        prefix, trailing = partition_blocks(blocks)

        items: list[ContextItem] = [*prefix]
        items.extend(visible_events(owner))
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


class CodeActView(ManagedContextView):
    """Execution protocol, environment, agent context, and active history."""

    block_order = ("system_prompt", "strategy_prompt", "sandbox", "execution_context", "self")
    static_keys = frozenset({"execution_context", "strategy_prompt", "sandbox"})

    def instruction_blocks(self, call: "CurrentCall") -> Mapping[str, Any]:
        assert call.strategy is not None
        blocks: dict[str, Any] = {
            "strategy_prompt": DynamicContext("strategy.strategy_instructions(runtime)"),
            "execution_context": DynamicContext("strategy.execution_context(runtime)"),
        }
        config = call.strategy.config
        if config.execution_backend == "sandbox" and config.sandbox.context_block:
            blocks["sandbox"] = DynamicContext("strategy.sandbox_context(runtime)")
        return blocks


class PredictView(ManagedContextView):
    """Structured-answer instructions, agent context, and active history."""

    def instruction_blocks(self, call: "CurrentCall") -> Mapping[str, Any]:
        return {"strategy_prompt": DynamicContext("strategy.strategy_instructions(runtime)")}


class PurePythonView(ManagedContextView):
    """Python-only execution instructions, agent context, and active history."""

    def instruction_blocks(self, call: "CurrentCall") -> Mapping[str, Any]:
        return {"strategy_prompt": DynamicContext("strategy.strategy_instructions(runtime)")}


class CodeActV2View(CodeActView):
    """Single-tool execution context; protocol instructions live on the tool."""

    block_order = ("system_prompt", "sandbox", "python_cell_context", "python_cell_state", "self")
    static_keys = frozenset({"sandbox", "python_cell_context"})

    def instruction_blocks(self, call: "CurrentCall") -> Mapping[str, Any]:
        blocks = dict(super().instruction_blocks(call))
        blocks["strategy_prompt"] = None
        blocks["execution_context"] = None
        blocks["python_cell_context"] = DynamicContext("strategy.python_cell_context(runtime)")
        blocks["python_cell_state"] = DynamicContext("strategy.python_cell_state_context(runtime)")
        return blocks


class ReflexionView(ContextView["Agent"]):
    """Delegate context assembly to a wrapped strategy using its call environment."""

    def __init__(self, base_strategy: Any):
        self.base_strategy = base_strategy

    async def assemble(self, owner: "Agent", call: "CurrentCall") -> AsyncIterator[ContextItem]:
        from dataclasses import replace

        base_call = replace(call, strategy=self.base_strategy)
        async for item in self.base_strategy.default_context_view.assemble(owner, base_call):
            yield item


__all__ = [
    "ManagedContextView",
    "CodeActView",
    "PredictView",
    "PurePythonView",
    "CodeActV2View",
    "ReflexionView",
    "collect_skill_blocks",
    "agent_interface_block",
    "agent_state_block",
    "events_api_block",
    "apply_block_overrides",
    "order_blocks",
    "partition_blocks",
    "replace_blocks_by_key",
    "materialize_manager_blocks",
    "system_prompt_block",
    "visible_events",
]
