# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render assembled context items into provider-specific output.

Pipeline:

1. Expand each item in place into an internal resolved block.
2. Serialize non-tool events in place.
3. Hand the ordered list of blocks to ``block_formatter`` — it produces a
   neutral ``list[RenderedMessage]`` covering system + events + any extra
   trailing messages the formatter chooses to emit.
5. Hand the neutral list to ``provider_formatter`` to reshape into the
   provider-specific wire format.

``render_context()`` never mutates its input blocks — truncation and
serialization produce new :class:`ResolvedBlock` instances via ``model_copy()``.
"""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from nooa.config.truncation_config import FormatConfig

from nooa.context_blocks.events import EventBase, ToolCallEvent
from nooa.context_blocks.formatter import (
    FORMAT_PLAIN,
    FORMAT_XML,
    BlockFormatter,
    ProviderFormatter,
    _markdown_message_content,
    _xml_message_content,
)
from nooa.context_blocks.models import (
    BlockMetadata,
    ContextWindowStats,
    RenderedMessage,
    ResolvedBlock,
    Role,
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


def render_context(
    blocks: Sequence[Any],
    *,
    block_formatter: BlockFormatter,
    provider_formatter: ProviderFormatter,
    context_limit: int | None = None,
    count_tokens: Callable[[str], int] | None = None,
    event_format: "FormatConfig | None" = None,
    event_format_resolver: Callable[[Any], "FormatConfig | None"] | None = None,
    model_context_window: int | None = None,
    reserved_output_tokens: int | None = None,
    context_blocks_dropped: int = 0,
) -> RenderResult:
    """Render resolved blocks into provider-specific output with utilization stats.

    Never mutates or reorders input items. Budget policy belongs to the view.

    ``event_format`` carries the default structural bounds (max_string /
    max_length / max_depth) for event-field rendering at trajectory build time.
    ``event_format_resolver`` can override those bounds for a single event,
    which lets method-level ``@strategy(truncation=...)`` affect events from
    that method without re-rendering the rest of the context under that config.
    """
    from nooa.context_view import Block, _MaterializedBlock

    resolved: list[ResolvedBlock] = []
    for item in blocks:
        if isinstance(item, ResolvedBlock):
            block = item
        elif isinstance(item, Block):
            metadata = item.metadata if isinstance(item, _MaterializedBlock) else None
            block = ResolvedBlock(
                key=item.key,
                content=item.content,
                role=item.role,
                metadata=metadata or BlockMetadata(),
            )
        elif isinstance(item, EventBase):
            tag = item.tag if item.tag is not None else item.id
            block = ResolvedBlock(
                key=f"event_{tag}",
                content="",
                role=getattr(item, "_role", Role.USER),
                metadata=BlockMetadata(expr=f'self.events["{tag}"]', tag=tag),
                event=item,
            )
        else:
            raise TypeError(f"Expected Block or EventBase, got {type(item).__name__}")

        if block.event is not None and not isinstance(block.event, ToolCallEvent):
            resolved_event_format = (
                event_format_resolver(block.event)
                if event_format_resolver is not None
                else event_format
            )
            content = block_formatter.format_event(block.event, event_format=resolved_event_format)
            block = block.model_copy(update={"content": content})
        resolved.append(block)

    context_blocks = [block for block in resolved if block.event is None]
    event_blocks = [block for block in resolved if block.event is not None]

    # Stats — structural only. Token figures are NOT estimated here; the
    # runtime writes the provider-reported prompt_tokens back after the call.
    # We record raw character sizes (post-eviction) so the provider total can
    # later be attributed across context blocks vs. events by character share.
    stats = ContextWindowStats(
        context_blocks_count=len(context_blocks),
        events_count=len(event_blocks),
        prompt_tokens=None,
        context_blocks_chars=sum(len(b.content) for b in context_blocks),
        events_chars=sum(len(b.content) for b in event_blocks),
        max_context_tokens=context_limit,
        model_context_window=model_context_window,
        context_blocks_dropped=context_blocks_dropped,
        events_dropped=0,
        reserved_output_tokens=reserved_output_tokens,
    )

    # Neutral message list → provider wire format.
    messages = block_formatter.format(resolved)
    output = provider_formatter.format(messages)
    return RenderResult(
        output=output,
        stats=stats,
        messages=messages,
    )
