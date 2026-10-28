# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render pre-resolved context blocks into provider-specific output.

Pipeline:

1. Partition blocks by role for truncation.
2. Pre-serialize non-tool event content so it can be measured and trimmed.
3. Apply block-level and total-budget truncation.
4. Hand the full ordered list of blocks to ``block_formatter`` — it produces a
   neutral ``list[RenderedMessage]`` covering system + events + any extra
   trailing messages the formatter chooses to emit.
5. Hand the neutral list to ``provider_formatter`` to reshape into the
   provider-specific wire format.

``render_context()`` never mutates its input blocks — truncation and
serialization produce new :class:`ResolvedBlock` instances via ``model_copy()``.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

from pydantic import BaseModel

from nooa._immutable_json import utf8_safe_text

if TYPE_CHECKING:
    from nooa.config.truncation_config import FormatConfig

from nooa.context_blocks.events import ToolCallEvent
from nooa.context_blocks.formatter import (
    FORMAT_PLAIN,
    FORMAT_XML,
    BlockFormatter,
    ProviderFormatter,
    _markdown_message_content,
    _xml_message_content,
)
from nooa.context_blocks.models import (
    ContextWindowStats,
    RenderedMessage,
    ResolvedBlock,
    Role,
    ToolCallInfo,
)
from nooa.context_blocks.utils import camel_to_snake


class RenderResult(NamedTuple):
    """Result of :func:`render_context`: provider output + utilization stats.

    ``messages`` is the neutral ``list[RenderedMessage]`` produced by the
    BlockFormatter *after* truncation and *before* provider formatting.
    Block-aware formatters populate ``RenderedMessage.parts`` on this list,
    which the journal publisher walks to build a content-addressed skeleton.
    """

    output: Any
    stats: ContextWindowStats
    messages: list[RenderedMessage]


def format_message_content(block: ResolvedBlock, format_type: str) -> str:
    """Wrap an event block's content with a role tag and metadata.

    Utility for callers that want to reuse the stock XML / Markdown / plain
    message wrapping outside the renderer pipeline (e.g. summarization agents
    rendering events into an LLM prompt).
    """
    if format_type == FORMAT_XML:
        return _xml_message_content(block)
    if format_type == FORMAT_PLAIN:
        if block.event is not None:
            event_tag = camel_to_snake(type(block.event).__name__)
            tag_attr = f' tag="{block.metadata.tag}"' if block.metadata.tag else ""
            return f"<{event_tag}{tag_attr}>\n{block.content}\n</{event_tag}>"
        if block.metadata.tag:
            role_label = f"{block.role.value}_message"
            return f'<{role_label} tag="{block.metadata.tag}">\n{block.content}\n</{role_label}>'
        return block.content
    # Markdown and any other value
    return _markdown_message_content(block)


def _apply_context_total_limit(
    blocks: list[ResolvedBlock],
    total_limit: int,
    count_fn: Callable[[str], int],
) -> tuple[list[ResolvedBlock], int]:
    """Mark over-budget context blocks as EVICTED in-place.

    Two-pass strategy: select user blocks (``self.context``) from the end first,
    then remaining blocks from the end until the total fits. Selected blocks
    keep their original key/position and render an EVICTED label with per-block
    size stats.
    """
    total = sum(count_fn(b.content) for b in blocks)
    if total <= total_limit:
        return blocks, 0

    to_evict: set[int] = set()
    evicted_sizes: dict[int, int] = {}

    for i in range(len(blocks) - 1, -1, -1):
        if total <= total_limit:
            break
        if blocks[i].metadata.user_block and not blocks[i].metadata.static:
            size = count_fn(blocks[i].content)
            total -= size
            to_evict.add(i)
            evicted_sizes[i] = size

    if total > total_limit:
        for i in range(len(blocks) - 1, -1, -1):
            if total <= total_limit:
                break
            if i not in to_evict and not blocks[i].metadata.static:
                size = count_fn(blocks[i].content)
                total -= size
                to_evict.add(i)
                evicted_sizes[i] = size

    rendered: list[ResolvedBlock] = []
    for i, block in enumerate(blocks):
        if i not in to_evict:
            rendered.append(block)
            continue
        size = evicted_sizes.get(i, count_fn(block.content))
        msg = f"EVICTED: over context budget (block_tokens={size:,})"
        rendered.append(
            block.model_copy(
                update={
                    "content": msg,
                    "metadata": block.metadata.model_copy(update={"truncated": True}),
                }
            )
        )

    return rendered, len(to_evict)


def _utf8_safe(value: Any) -> Any:
    """Return ``value`` with lone UTF-16 surrogates escaped as ``\\udXXX`` text.

    Python strings can hold lone surrogates (an agent writing ``"\\ud800"`` in
    code, ``os.listdir`` on non-UTF-8 names), but UTF-8 cannot encode them, so a
    single one anywhere in the context makes every request fail to serialize.
    Walks strings, dicts, lists, tuples and pydantic models; anything that needs
    no change is returned as the same object, so callers' data is never mutated.
    """
    if isinstance(value, str):
        return utf8_safe_text(value)
    if isinstance(value, dict):
        items = [(_utf8_safe(k), _utf8_safe(v)) for k, v in value.items()]
        if all(k is k0 and v is v0 for (k, v), (k0, v0) in zip(items, value.items(), strict=True)):
            return value
        return dict(items)
    if isinstance(value, (list, tuple)):
        safe = [_utf8_safe(v) for v in value]
        if all(v is v0 for v, v0 in zip(safe, value, strict=True)):
            return value
        return safe if isinstance(value, list) else tuple(safe)
    if isinstance(value, BaseModel):
        update = {}
        for name in type(value).model_fields:
            # Native replay turns are resent verbatim; a turn whose content
            # changed here is rendered from that content instead.
            if name == "replay_message":
                continue
            field = getattr(value, name)
            # Dict arguments reach the request and the journal only through
            # json.dumps, whose ASCII escapes keep them exact and encodable.
            if isinstance(value, ToolCallInfo) and name == "arguments" and isinstance(field, dict):
                continue
            safe = _utf8_safe(field)
            if safe is not field:
                update[name] = safe
        return value.model_copy(update=update) if update else value
    return value


def _with_utf8_safe_content(block: ResolvedBlock) -> ResolvedBlock:
    """Escape lone surrogates in ``block.content`` so budgets measure the sent text."""
    content = _utf8_safe(block.content)
    return block if content is block.content else block.model_copy(update={"content": content})


def render_context(
    blocks: list[ResolvedBlock],
    *,
    block_formatter: BlockFormatter,
    provider_formatter: ProviderFormatter,
    context_limit: int | None = None,
    count_tokens: Callable[[str], int] | None = None,
    event_format: "FormatConfig | None" = None,
    event_format_resolver: Callable[[Any], "FormatConfig | None"] | None = None,
    model_context_window: int | None = None,
    reserved_output_tokens: int | None = None,
) -> RenderResult:
    """Render resolved blocks into provider-specific output with utilization stats.

    Never mutates input blocks. Per-block head/tail truncation has been removed
    — content passes through verbatim. Context blocks over budget are marked
    EVICTED in place.

    ``event_format`` carries the default structural bounds (max_string /
    max_length / max_depth) for event-field rendering at trajectory build time.
    ``event_format_resolver`` can override those bounds for a single event,
    which lets method-level ``@strategy(truncation=...)`` affect events from
    that method without re-rendering the rest of the context under that config.
    """
    if count_tokens is None and context_limit is not None:
        raise ValueError(
            "max_context_tokens requires a token counter. "
            "Pass count_tokens=llm.count_tokens to render_context()."
        )

    count_fn: Callable[[str], int] = count_tokens if count_tokens is not None else len

    # Partition for truncation only.
    system_blocks = [_with_utf8_safe_content(b) for b in blocks if b.role == Role.SYSTEM]
    message_blocks = [b for b in blocks if b.role != Role.SYSTEM]

    # Pre-serialize non-tool events so their content can be measured.
    # ToolCallEvents stay at content="" — the BlockFormatter handles them structurally.
    serialized_messages: list[ResolvedBlock] = []
    for block in message_blocks:
        if block.event is not None and not isinstance(block.event, ToolCallEvent):
            resolved_event_format = (
                event_format_resolver(block.event)
                if event_format_resolver is not None
                else event_format
            )
            content = block_formatter.format_event(block.event, event_format=resolved_event_format)
            block = block.model_copy(update={"content": content})
        serialized_messages.append(_with_utf8_safe_content(block))
    message_blocks = serialized_messages

    # Total-context eviction: mark over-budget blocks EVICTED in place.
    # The eviction count is reported to the caller via
    # ``ContextWindowStats.context_blocks_dropped`` (below). The runtime owns
    # its ``HarnessMetrics`` singleton and increments
    # ``context_limits_blocks_evicted`` from that value — this leaf library
    # never reaches into the runtime's metrics (see issue #330).
    context_blocks_dropped = 0
    if context_limit is not None:
        system_blocks, context_blocks_dropped = _apply_context_total_limit(
            system_blocks, context_limit, count_fn
        )

    # Stats — structural only. Token figures are NOT estimated here; the
    # runtime writes the provider-reported prompt_tokens back after the call.
    # We record raw character sizes (post-eviction) so the provider total can
    # later be attributed across context blocks vs. events by character share.
    stats = ContextWindowStats(
        context_blocks_count=len(system_blocks),
        events_count=len(message_blocks),
        prompt_tokens=None,
        context_blocks_chars=sum(len(b.content) for b in system_blocks),
        events_chars=sum(len(b.content) for b in message_blocks),
        max_context_tokens=context_limit,
        model_context_window=model_context_window,
        context_blocks_dropped=context_blocks_dropped,
        events_dropped=0,
        reserved_output_tokens=reserved_output_tokens,
    )

    # Neutral message list → provider wire format. Block content is already
    # escaped (so the budget and stats above measure the sent text); this pass
    # covers the remaining fields (tool calls, images, parts). Both the request
    # and the tracing journal encode these messages.
    messages = _utf8_safe(block_formatter.format([*system_blocks, *message_blocks]))
    output = provider_formatter.format(messages)
    return RenderResult(
        output=output,
        stats=stats,
        messages=messages,
    )
