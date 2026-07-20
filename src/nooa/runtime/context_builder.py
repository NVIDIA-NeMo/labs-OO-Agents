# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context builder: the pipeline that gathers, resolves, and orders context blocks.

Extracted from actor.py so _prepare_context() is thin.

Pipeline phases (must run in order — later phases override earlier ones):
    1. Persistent blocks from context_manager (protected framework blocks +
       user blocks — all registered via the same ContextManager API)
    2. Strategy overrides (strategy.get_block_overrides())
    3. Decorator context (@strategy(ScopedContext(context={...})))
    4. Scoped context (with ScopedContext(context={...}))
    5. Events -> ResolvedBlocks with roles
       Event filtering priority: runtime > scoped > decorator > agent > all events

All phase functions are pure — they return new lists, never mutate inputs.
The only side effect (updating ContextApi's resolved cache) is performed
by the caller in _prepare_context(), not by build_context() itself.
"""

import logging
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any, NamedTuple

from nooa.agentdoc import pformat as _pformat_value
from nooa.context_blocks import (
    BlockMetadata,
    Context,
    DynamicContext,
    ResolvedBlock,
    Role,
)

if TYPE_CHECKING:
    from nooa.config.truncation_config import FormatConfig
    from nooa.runtime.context_manager import ContextManager
    from nooa.runtime.event_query import EventQuery
    from nooa.strategies.base import GenerationStrategy

logger = logging.getLogger(__name__)

# Type alias for the async resolve function
ResolveFunc = Callable[[str, "str | DynamicContext"], Coroutine[Any, Any, "str | None"]]


# ---------------------------------------------------------------------------
# Pure list helpers — always return new lists, never mutate inputs
# ---------------------------------------------------------------------------
async def _apply_overrides(
    blocks: list[ResolvedBlock],
    overrides: dict[str, str | DynamicContext | None],
    resolve_fn: ResolveFunc,
    static_expr: Callable[[str], str],
    static_keys: set[str] | None = None,
    disabled_keys: set[str] | None = None,
) -> list[ResolvedBlock]:
    """Apply a dict of overrides (replace/append/remove) to blocks.

    Shared logic for strategy overrides, decorator context, and scoped blocks.
    Uses a single pass with an index for O(n + k) instead of O(n * k).

    Args:
        blocks: Current block list (not mutated).
        overrides: Dict of key -> value. None removes, str/DynamicContext replaces or appends.
        resolve_fn: Async function to resolve DynamicContext values.
        static_expr: Function(key) -> metadata expr string for static values.
        static_keys: Keys that should be placed in the static partition when new.
        disabled_keys: Block keys to suppress even if an override would add them.
    """
    if not overrides:
        return blocks

    disabled_keys = disabled_keys or set()

    # Build index: key -> position in blocks list (first occurrence)
    index: dict[str, int] = {}
    for i, b in enumerate(blocks):
        if b.key not in index:
            index[b.key] = i

    # Resolve all overrides and build a replacement map + append list
    replacements: dict[int, ResolvedBlock | None] = {}  # position -> new block (None = remove)
    appends: list[ResolvedBlock] = []

    for key, value in overrides.items():
        if key in disabled_keys:
            if key in index:
                replacements[index[key]] = None
            continue

        if value is None:
            if key in index:
                replacements[index[key]] = None
            continue

        # Normalize Context to DynamicContext/str for the pipeline
        context_prefix = None
        if isinstance(value, Context):
            context_prefix = value.prefix
            if value.is_dynamic:
                assert value.expr is not None  # guaranteed by is_dynamic
                value = DynamicContext(value.expr)
            else:
                assert value.value is not None  # guaranteed by not is_dynamic
                value = value.value

        content = await resolve_fn(key, value)
        if content is None:
            content = "None"

        # Determine static partition: inherit from existing block if present,
        # otherwise check static_keys or default based on value type.
        existing_static = None
        if key in index:
            existing_static = blocks[index[key]].metadata.static
        if context_prefix is not None:
            is_block_static = context_prefix
        elif existing_static is not None:
            is_block_static = existing_static
        elif static_keys and key in static_keys:
            is_block_static = True
        else:
            # New block: DynamicContext is dynamic, plain values are static
            is_block_static = not isinstance(value, DynamicContext)

        if isinstance(value, DynamicContext):
            meta = BlockMetadata(expr=value.expr, static=is_block_static)
        else:
            meta = BlockMetadata(expr=static_expr(key), static=is_block_static)

        new_block = ResolvedBlock(key=key, content=content, role=Role.SYSTEM, metadata=meta)
        if key in index:
            replacements[index[key]] = new_block
        else:
            appends.append(new_block)

    # Single pass: rebuild list applying replacements and filtering removals
    result = []
    for i, block in enumerate(blocks):
        if i in replacements:
            replacement = replacements[i]
            if replacement is not None:
                result.append(replacement)
            # else: removed (skip)
        else:
            result.append(block)

    result.extend(appends)
    return result


# ---------------------------------------------------------------------------
# Block reordering — applied after all content phases, before events
# ---------------------------------------------------------------------------
def _reorder_blocks(
    blocks: list[ResolvedBlock],
    strategy: "GenerationStrategy | None",
) -> list[ResolvedBlock]:
    """Reorder system blocks according to strategy.get_block_order().

    Listed keys appear first (in given order), unlisted system blocks follow
    in their original relative order. Returns a new list.
    """
    if strategy is None:
        return blocks

    order = strategy.get_block_order()
    if order is None:
        return blocks

    key_rank = {key: i for i, key in enumerate(order)}

    # Partition into ordered (has a rank) and remainder (no rank)
    ordered: list[tuple[int, ResolvedBlock]] = []
    remainder: list[ResolvedBlock] = []
    for block in blocks:
        if block.key in key_rank:
            ordered.append((key_rank[block.key], block))
        else:
            remainder.append(block)

    ordered.sort(key=lambda pair: pair[0])
    return [block for _, block in ordered] + remainder


# ---------------------------------------------------------------------------
# Result type for build_context — separates blocks from resolved cache
# ---------------------------------------------------------------------------
class BuildResult(NamedTuple):
    """Result of build_context(): resolved blocks + cache for ContextApi."""

    blocks: list[ResolvedBlock]
    resolved_cache: dict[str, Any]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
async def build_context(
    *,
    context_manager: "ContextManager",
    event_manager: Any,
    strategy: "GenerationStrategy | None",
    resolve_fn: ResolveFunc,
    decorator_context: dict[str, Any] | None = None,
    scoped_context: dict[str, Any] | None = None,
    runtime_event_query: "EventQuery | None" = None,
    decorator_event_query: "EventQuery | None" = None,
    scoped_event_query: "EventQuery | None" = None,
    agent_event_query: "EventQuery | None" = None,
    current_call_id: str | None = None,
    context_block_format: "FormatConfig | None" = None,
) -> BuildResult:
    """Build the complete context block list from all sources.

    This is a pure pipeline — it does NOT mutate agent or context_manager state.
    The caller is responsible for applying resolved_cache via
    ``context_manager._update_resolved(result.resolved_cache)``.

    All blocks (including protected framework blocks like system_prompt, self,
    state) live in ``context_manager`` and are emitted by
    ``_phase_persistent_blocks``.  Protected blocks get ``user_block=False``
    so they survive truncation.

    Args:
        context_manager: The agent's ContextManager (read-only here).
        event_manager: The agent's EventManager (for events phase).
        strategy: Current generation strategy (or None).
        resolve_fn: Async function(key, value) -> str that resolves DynamicContext values.
        decorator_context: Merged @strategy(context={...}) overrides from the
            current method and its parents. Passed explicitly by the actor
            instead of being read from a context variable.
        scoped_context: Scoped block overrides from with ScopedContext(...).
            Passed explicitly by the actor instead of reading from contextvars.
        runtime_event_query: EventQuery from event_manager.set_event_query().
        decorator_event_query: EventQuery from @strategy(ScopedContext(events=...)).
        scoped_event_query: EventQuery from with ScopedContext(events=...).
        agent_event_query: Default EventQuery from agent-level configuration.
        current_call_id: Current call ID for resolving EventQuery(call_id="current").

    Returns:
        BuildResult with blocks and resolved_cache.
    """
    blocks: list[ResolvedBlock] = []

    # --- All persistent blocks (protected framework blocks + user blocks) ---
    blocks, resolved_cache = await _phase_persistent_blocks(
        blocks, context_manager, resolve_fn, context_block_format=context_block_format
    )

    disabled_keys = context_manager.disabled()

    # --- Strategy block overrides ---
    blocks = await _phase_strategy_overrides(blocks, strategy, resolve_fn, disabled_keys)

    # --- @strategy(context=...) decorator overrides ---
    blocks = await _phase_decorator_context(blocks, decorator_context, resolve_fn, disabled_keys)

    # --- Scoped context blocks ---
    blocks = await _phase_scoped_blocks(blocks, scoped_context, resolve_fn, disabled_keys)

    # --- Strategy block reordering (between content phases and events) ---
    blocks = _reorder_blocks(blocks, strategy)

    # --- Events (with optional filtering via EventQuery) ---
    blocks = _phase_events(
        blocks,
        event_manager,
        runtime_event_query=runtime_event_query,
        scoped_event_query=scoped_event_query,
        decorator_event_query=decorator_event_query,
        agent_event_query=agent_event_query,
        current_call_id=current_call_id,
    )

    return BuildResult(blocks=blocks, resolved_cache=resolved_cache)


async def _phase_persistent_blocks(
    blocks: list[ResolvedBlock],
    context_manager: "ContextManager",
    resolve_fn: ResolveFunc,
    context_block_format: "FormatConfig | None" = None,
) -> tuple[list[ResolvedBlock], dict[str, Any]]:
    """Add persistent blocks from context_manager.

    Protected blocks (framework blocks like system_prompt, self, state) get
    ``user_block=False`` so they survive truncation.  User-set blocks get
    ``user_block=True``.  All DynamicContext blocks get ``source_dynamic=True``
    regardless of origin.

    Returns:
        Tuple of (updated blocks, resolved_cache dict).
        The caller should pass resolved_cache to context_manager._update_resolved().
    """
    resolved_cache: dict[str, Any] = {}
    protected_keys = context_manager.protected_keys

    for key, value in context_manager._raw_items():
        if context_manager.is_disabled(key):
            continue

        is_user = key not in protected_keys
        is_static = context_manager.is_static(key)
        if isinstance(value, DynamicContext):
            # DynamicContext path: resolve via async eval
            # resolve_fn returns pre-formatted string (already pprinted if non-string)
            resolved = await resolve_fn(key, value)
            content = resolved if resolved is not None else "None"
            meta = BlockMetadata(
                expr=value.expr,
                user_block=is_user,
                static=is_static,
                source_dynamic=True,
            )
            # Cache the resolved value for __getitem__ access
            resolved_cache[key] = resolved
        else:
            # Static path: render value through cfg.context_block_format bounds.
            # Strings pass verbatim; non-strings go through pformat with the
            # supplied structural knobs (max_string / max_length / max_depth).
            if value is None:
                content = "None"
            else:
                kwargs = context_block_format.model_dump() if context_block_format else {}
                content = _pformat_value(value, unquote_strings=True, **kwargs)
            meta = BlockMetadata(
                expr=f'self.context["{key}"]', user_block=is_user, static=is_static
            )

        blocks = [
            *blocks,
            ResolvedBlock(key=key, content=content, role=Role.SYSTEM, metadata=meta),
        ]

    return blocks, resolved_cache


async def _phase_strategy_overrides(
    blocks: list[ResolvedBlock],
    strategy: "GenerationStrategy | None",
    resolve_fn: ResolveFunc,
    disabled_keys: set[str] | None = None,
) -> list[ResolvedBlock]:
    """Apply strategy.get_block_overrides()."""
    if not strategy or not hasattr(strategy, "get_block_overrides"):
        return blocks

    static_keys = None
    if hasattr(strategy, "get_static_block_keys"):
        static_keys = strategy.get_static_block_keys()  # type: ignore[attr-defined]

    return await _apply_overrides(
        blocks,
        strategy.get_block_overrides(),
        resolve_fn,
        static_expr=lambda key: f"strategy.{key}",
        static_keys=static_keys,
        disabled_keys=disabled_keys,
    )


async def _phase_decorator_context(
    blocks: list[ResolvedBlock],
    decorator_context: dict[str, Any] | None,
    resolve_fn: ResolveFunc,
    disabled_keys: set[str] | None = None,
) -> list[ResolvedBlock]:
    """Apply @strategy(context=...) decorator overrides.

    The decorator_context dict is the merged context from the current method's
    @strategy(context={...}) and its parent's inherited context. Passed
    explicitly by the actor.
    """
    if not decorator_context:
        return blocks

    return await _apply_overrides(
        blocks,
        decorator_context,
        resolve_fn,
        static_expr=lambda key: f'@strategy.context["{key}"]',
        disabled_keys=disabled_keys,
    )


async def _phase_scoped_blocks(
    blocks: list[ResolvedBlock],
    scoped_context: dict[str, Any] | None,
    resolve_fn: ResolveFunc,
    disabled_keys: set[str] | None = None,
) -> list[ResolvedBlock]:
    """Apply scoped block overrides.

    Scoped context is passed explicitly by the caller instead of reading
    from _scoped_blocks_var, keeping this function pure.
    """
    if not scoped_context:
        return blocks

    return await _apply_overrides(
        blocks,
        scoped_context,
        resolve_fn,
        static_expr=lambda key: f'self.context["{key}"]',
        disabled_keys=disabled_keys,
    )


def _phase_events(
    blocks: list[ResolvedBlock],
    event_manager: Any,
    *,
    runtime_event_query: "EventQuery | None" = None,
    scoped_event_query: "EventQuery | None" = None,
    decorator_event_query: "EventQuery | None" = None,
    agent_event_query: "EventQuery | None" = None,
    current_call_id: str | None = None,
) -> list[ResolvedBlock]:
    """Convert events to ResolvedBlocks with appropriate roles.

    Events are filtered using EventQuery with 4-level priority:
    1. Runtime query (set via event_manager.set_event_query()) - highest priority
    2. Scoped query (with ScopedContext(events=...)) - high priority
    3. Decorator query (@strategy(ScopedContext(events=...))) - medium priority
    4. Agent query (agent-level default) - low priority
    5. No filter (show all events) - default

    The original event is carried on ``block.event`` so that provider
    formatters can read structured data (e.g., ToolCallEvent fields)
    directly without intermediate types.

    Args:
        blocks: Current block list (not mutated).
        event_manager: EventManager for accessing events.
        runtime_event_query: EventQuery from event_manager.set_event_query().
        scoped_event_query: EventQuery from with ScopedContext(events=...).
        decorator_event_query: EventQuery from @strategy(ScopedContext(events=...)).
        agent_event_query: EventQuery from agent-level configuration.
        current_call_id: Current call ID for resolving EventQuery(call_id="current").
    """
    # Determine active query (runtime > scoped > decorator > agent > none)
    active_query = (
        runtime_event_query or scoped_event_query or decorator_event_query or agent_event_query
    )

    # Use only active (non-archived) events in their display order.
    # values() uses active_tags() — excludes events archived by collapse().
    if active_query:
        events = active_query.apply(event_manager.values(), current_call_id=current_call_id)
    else:
        events = event_manager.values()

    new_blocks: list[ResolvedBlock] = []

    for event in events:
        tag = event.tag if event.tag is not None else event.id
        event_role = getattr(event, "_role", Role.USER)
        meta = BlockMetadata(expr=f'self.events["{tag}"]', tag=tag)

        # All events carry their raw object on block.event with content="".
        # ToolCallEvents are handled by ProviderFormatter; other events are
        # serialized at render time via block_formatter.format_event().
        new_blocks.append(
            ResolvedBlock(
                key=f"event_{tag}",
                content="",
                role=event_role,
                metadata=meta,
                event=event,
            )
        )

    return [*blocks, *new_blocks]


# Phase 7 and 8 (decorator events and scoped events) have been removed.
# Events are now filtered directly in _phase_events() using EventQuery.
