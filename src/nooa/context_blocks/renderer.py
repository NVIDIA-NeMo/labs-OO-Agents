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
    blocks: list[ResolvedBlock],
    *,
    block_formatter: BlockFormatter,
    provider_formatter: ProviderFormatter,
    event_format: "FormatConfig | None" = None,
    event_format_resolver: Callable[[Any], "FormatConfig | None"] | None = None,
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
    # Partition for truncation only.
    system_blocks = [b for b in blocks if b.role == Role.SYSTEM]
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
        serialized_messages.append(block)
    message_blocks = serialized_messages

    # Stats — structural only. Token figures are NOT estimated here; the
    # runtime writes the provider-reported prompt_tokens back after the call.
    # We record raw character sizes (post-eviction) so the provider total can
    # later be attributed across context blocks vs. events by character share.
    stats = ContextWindowStats(
        context_blocks_count=len(system_blocks),
        events_count=len(message_blocks),
        context_blocks_chars=sum(len(b.content) for b in system_blocks),
        events_chars=sum(len(b.content) for b in message_blocks),
    )

    # Neutral message list → provider wire format.
    messages = block_formatter.format([*system_blocks, *message_blocks])
    output = provider_formatter.format(messages)
    return RenderResult(
        output=output,
        stats=stats,
        messages=messages,
    )
