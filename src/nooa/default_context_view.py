# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default context policy implemented against public framework interfaces."""

import ast
import logging
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from nooa.context_blocks import BlockMetadata, Context, DynamicContext, Role
from nooa.context_blocks.events import EventBase
from nooa.context_view import (
    Block,
    CacheBoundary,
    ContextItem,
    ContextView,
    apply_context_budget,
    context_text,
    evaluate_context_expression,
    is_model_visible,
    replayable,
)

if TYPE_CHECKING:
    from nooa.agent import Agent
    from nooa.skill import Skill
    from nooa.strategies.current_call import CurrentCall

logger = logging.getLogger(__name__)


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
    owner: Any,
    call: "CurrentCall",
) -> str:
    """Resolve one declaration with ``self`` bound to its declaring owner."""
    if isinstance(value, Context):
        value = value.value if value.expr is None else DynamicContext(value.expr)
    if not isinstance(value, DynamicContext):
        return context_text(value, call=call)
    try:
        resolved = await evaluate_context_expression(value.expr, owner=owner, call=call)
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
) -> tuple[Block, ...]:
    """Materialize the enabled user blocks of one ContextManager snapshot."""
    blocks: list[Block] = []
    for key, value in manager.declarations():
        if manager.is_disabled(key):
            continue

        is_dynamic = isinstance(value, DynamicContext)
        content = await _resolve_value(key, value, owner=agent, call=call)
        metadata = BlockMetadata(
            expr=value.expr if is_dynamic else None,
            user_block=True,
            static=manager.is_static(key),
        )
        blocks.append(_block(key, content, metadata))
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


_EVENTS_API_PROMPT = """self.events: Query past events by type, tag, text, or call ID. Like a database, not an array.
BEFORE starting any task, check if this skill applies.
You MUST call doc(self.events) before using it — do not assume you know the API.
Inspect in REPL: print(doc(self.events))
"""


async def system_prompt_block(agent: "Agent", call: "CurrentCall") -> Block | None:
    """Build the default system prompt directly from the agent."""
    try:
        content = context_text(resolve_agent_system_prompt(agent), call=call)
    except Exception as exc:
        content = _source_error("system_prompt", exc)
    return _block("system_prompt", content, BlockMetadata(static=True))


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
        BlockMetadata(expr="doc(type(self))", static=True),
    )


async def events_api_block(agent: "Agent", call: "CurrentCall") -> Block | None:
    """Describe the events API only when it is exposed to the model."""
    from nooa.agentdoc.visibility import is_hidden_field

    if is_hidden_field(agent, "events"):
        return None
    return _block("events_api", _EVENTS_API_PROMPT, BlockMetadata(static=True))


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
        ),
    )


async def apply_block_overrides(
    blocks: Sequence[Block],
    overrides: Mapping[str, Any] | None,
    *,
    agent: "Agent",
    call: "CurrentCall",
    disabled: set[str] | None = None,
) -> tuple[Block, ...]:
    """Replace, append, or remove named blocks using one override source.

    Only expression overrides carry an ``expr`` label.
    """
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
        else:
            # A new bare value lands in the volatile suffix, as with self.context.
            is_static = False

        content = await _resolve_value(key, value, owner=agent, call=call)
        metadata = BlockMetadata(
            expr=value.expr if isinstance(value, DynamicContext) else None,
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
    """Select active model-visible history. Pass the selection through ``replayable``."""
    return tuple(event for event in agent.event_manager.values() if is_model_visible(event))


class _SelfToAttribute(ast.NodeTransformer):
    def __init__(self, attr: str):
        self.attr = attr

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id != "self":
            return node
        attribute = ast.Attribute(value=ast.Name("self", ast.Load()), attr=self.attr, ctx=node.ctx)
        return ast.copy_location(attribute, node)


def _rebind_self(expr: str, attr: str) -> str:
    """Rewrite a skill's ``self.status()`` as the agent's ``self.<attr>.status()``."""
    return ast.unparse(_SelfToAttribute(attr).visit(ast.parse(expr, mode="eval")))


async def collect_skill_blocks(
    agent: "Agent",
    call: "CurrentCall",
    *,
    disabled: set[str],
    declaration_keys: frozenset[str],
) -> tuple[Block, ...]:
    """Evaluate active skill declarations without selecting their global placement."""
    attrs: dict[int, str] = {}
    for attr, value in agent.__instance_values__().items():
        attrs.setdefault(id(value), attr)
    blocks: tuple[Block, ...] = ()
    for skill in active_skills(agent):
        if skill.context_block is None:
            continue
        key, declaration = skill.context_block
        if key in disabled or key in declaration_keys:
            continue
        content = await _resolve_value(key, declaration, owner=skill, call=call)
        expr = declaration.expr
        metadata = BlockMetadata(
            expr=_rebind_self(expr, attrs[id(skill)]) if expr is not None else None,
            static=declaration.prefix,
            user_block=True,
        )
        block = _block(key, content, metadata)
        blocks = replace_blocks_by_key(blocks, [block])
    return blocks


class ManagedContextView(ContextView["Agent"]):
    """Reusable assembly for the built-in manager-backed context policies.

    The view builds its own blocks: the agent's system prompt, its documented
    interface (``self``), the events API when exposed, the agent state, and the
    strategy instruction blocks. User blocks from ``self.context``, skill blocks,
    and decorator/scoped context are added next to them; a block with the same
    name is an extra block and never replaces or hides one the view builds.

    Args:
        state: Include the ``state`` block, the agent's current field values.
    """

    block_order: tuple[str, ...] | None = None
    state: bool = True

    def __init__(self, *, state: bool = True) -> None:
        self.state = state

    async def instruction_blocks(self, owner: "Agent", call: "CurrentCall") -> Mapping[str, Any]:
        """Return this view's strategy instruction blocks."""
        return {}

    def history(self, owner: "Agent", call: "CurrentCall") -> Iterable[EventBase]:
        """Select the events this view replays; blocks and placement are unchanged.

        ``assemble`` passes the selection through ``replayable``.
        """
        return visible_events(owner)

    async def view_blocks(self, owner: "Agent", call: "CurrentCall") -> tuple[Block, ...]:
        """Build the blocks this view owns from the agent and the strategy.

        Owners that are not an ``Agent``, such as standalone functions, get only
        the strategy blocks.
        """
        from nooa.agent import Agent

        builders = []
        if isinstance(owner, Agent):
            builders = [system_prompt_block, agent_interface_block, events_api_block]
            if self.state:
                builders.append(agent_state_block)
        blocks = [block for build in builders if (block := await build(owner, call)) is not None]
        return await apply_block_overrides(
            blocks, await self.instruction_blocks(owner, call), agent=owner, call=call
        )

    async def context_blocks(self, owner: "Agent", call: "CurrentCall") -> tuple[Block, ...]:
        """Materialize user, skill, and decorator/scoped blocks."""
        manager = owner.context_manager
        disabled = manager.disabled()
        blocks = await materialize_manager_blocks(manager, owner, call)
        skill_blocks = await collect_skill_blocks(
            owner,
            call,
            disabled=disabled,
            declaration_keys=frozenset(key for key, _ in manager.declarations()),
        )
        blocks = replace_blocks_by_key(blocks, skill_blocks)
        for overrides in (call.decorator_context, call.scoped_context):
            blocks = await apply_block_overrides(
                blocks, overrides, agent=owner, call=call, disabled=disabled
            )
        return blocks

    async def assemble(self, owner: "Agent", call: "CurrentCall") -> AsyncIterator[ContextItem]:
        blocks = (*await self.view_blocks(owner, call), *await self.context_blocks(owner, call))
        blocks = order_blocks(blocks, self.block_order)
        prefix, trailing = partition_blocks(blocks)

        items: list[ContextItem] = [*prefix]
        items.extend(replayable(self.history(owner, call)))
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

    async def instruction_blocks(self, owner: "Agent", call: "CurrentCall") -> Mapping[str, Any]:
        strategy, runtime = call.strategy, owner.runtime
        assert strategy is not None
        return {
            "strategy_prompt": Context(await strategy.strategy_instructions(runtime), prefix=True),
            "execution_context": Context(await strategy.execution_context(runtime), prefix=True),
            **await _sandbox_block(strategy, runtime),
        }


async def _sandbox_block(strategy: Any, runtime: Any) -> dict[str, Context]:
    config = strategy.config
    if config.execution_backend == "sandbox" and config.sandbox.context_block:
        return {"sandbox": Context(await strategy.sandbox_context(runtime), prefix=True)}
    return {}


class PredictView(ManagedContextView):
    """Structured-answer instructions, agent context, and active history."""

    async def instruction_blocks(self, owner: "Agent", call: "CurrentCall") -> Mapping[str, Any]:
        assert call.strategy is not None
        instructions = await call.strategy.strategy_instructions(owner.runtime)
        return {"strategy_prompt": Context(instructions, prefix=True)}


class PurePythonView(PredictView):
    """Python-only execution instructions, agent context, and active history."""


class CodeActV2View(CodeActView):
    """Single-tool execution context; protocol instructions live on the tool."""

    block_order = ("system_prompt", "sandbox", "python_cell_context", "python_cell_state", "self")

    async def instruction_blocks(self, owner: "Agent", call: "CurrentCall") -> Mapping[str, Any]:
        strategy, runtime = call.strategy, owner.runtime
        assert strategy is not None
        return {
            **await _sandbox_block(strategy, runtime),
            "python_cell_context": Context(
                await strategy.python_cell_context(runtime), prefix=True
            ),
            "python_cell_state": Context(await strategy.python_cell_state_context(runtime)),
        }


__all__ = [
    "ManagedContextView",
    "CodeActView",
    "PredictView",
    "PurePythonView",
    "CodeActV2View",
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
