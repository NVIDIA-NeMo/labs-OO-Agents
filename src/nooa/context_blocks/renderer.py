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
from nooa.context_blocks.exceptions import UnsupportedContextLayout
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
    from nooa.context_view import Block, CacheBoundary
    from nooa.llm_types import LLMResponse

    resolved: list[ResolvedBlock] = []
    segments: list[tuple[list[ResolvedBlock], bool]] = []
    segment: list[ResolvedBlock] = []
    segment_index = 0
    response_segments: dict[str, int] = {}
    linked_execution_segments: list[tuple[str, str, int]] = []
    for item in blocks:
        if isinstance(item, CacheBoundary):
            if segment:
                segments.append((segment, True))
                segment = []
            segment_index += 1
            continue
        if isinstance(item, ResolvedBlock):
            block = item
        elif isinstance(item, Block):
            block = ResolvedBlock(
                key=item.key,
                content=item.content,
                role=item.role,
                metadata=item.metadata or BlockMetadata(),
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
            raise TypeError(
                f"Expected Block, EventBase, or CacheBoundary, got {type(item).__name__}"
            )

        if isinstance(block.event, LLMResponse):
            response_segments[block.event.id] = segment_index
        elif isinstance(block.event, ToolCallEvent) and block.event.llm_response_id is not None:
            linked_execution_segments.append(
                (block.event.tool_call_id, block.event.llm_response_id, segment_index)
            )

        if isinstance(block.event, ToolCallEvent) and block.event.result is None:
            raise UnsupportedContextLayout(
                f"ToolCallEvent {block.event.tool_call_id!r} has no result"
            )

        if block.event is not None and not isinstance(block.event, ToolCallEvent):
            resolved_event_format = (
                event_format_resolver(block.event)
                if event_format_resolver is not None
                else event_format
            )
            content = block_formatter.format_event(block.event, event_format=resolved_event_format)
            block = block.model_copy(update={"content": content})
        resolved.append(block)
        segment.append(block)

    if segment:
        segments.append((segment, False))

    for tool_call_id, response_id, execution_segment in linked_execution_segments:
        response_segment = response_segments.get(response_id)
        if response_segment is not None and response_segment != execution_segment:
            raise UnsupportedContextLayout(
                f"CacheBoundary splits tool execution {tool_call_id!r} from its "
                "canonical LLMResponse"
            )

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
    messages: list[RenderedMessage] = []
    for segment, boundary_after in segments:
        rendered = block_formatter.format(segment)
        if boundary_after:
            if not rendered:
                raise UnsupportedContextLayout(
                    f"{type(block_formatter).__name__} emitted no message before CacheBoundary"
                )
        messages.extend(rendered)
        if boundary_after:
            messages.append(RenderedMessage(role=Role.METADATA, replay_message=CacheBoundary()))
    output = provider_formatter.format(messages)
    return RenderResult(
        output=output,
        stats=stats,
        messages=messages,
    )
