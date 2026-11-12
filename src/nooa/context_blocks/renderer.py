# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render assembled context items into provider-specific output.

Pipeline:

1. Expand each item in place into an internal resolved block.
2. Serialize non-tool events in place.
3. Hand the ordered list of blocks to ``block_formatter`` — it produces a
   neutral ``list[RenderedMessage]`` covering system + events + any extra
   trailing messages the formatter chooses to emit.
4. Hand the neutral list to ``provider_formatter`` to reshape into the
   provider-specific wire format.

``render_context()`` never mutates, reorders, or evicts its input items —
serialization produces new :class:`ResolvedBlock` instances via ``model_copy()``.
"""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

from pydantic import BaseModel

if TYPE_CHECKING:
    from nooa.config.truncation_config import FormatConfig
    from nooa.context_view import ContextItem

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
    ToolCallInfo,
)
from nooa.context_blocks.utils import camel_to_snake


class RenderResult(NamedTuple):
    """Result of :func:`render_context`: provider output + utilization stats.

    ``messages`` is the neutral ``list[RenderedMessage]`` produced by the
    BlockFormatter *before* provider formatting.
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


def _utf8_safe_text(text: str) -> str:
    """Return ``text`` with lone UTF-16 surrogates escaped as ``\\udXXX`` text."""
    if text.isascii():
        return text
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return text.encode("utf-8", "backslashreplace").decode("utf-8")
    return text


def _utf8_safe(value: Any) -> Any:
    """Return ``value`` with lone UTF-16 surrogates escaped as ``\\udXXX`` text.

    Python strings can hold lone surrogates (an agent writing ``"\\ud800"`` in
    code, ``os.listdir`` on non-UTF-8 names), but UTF-8 cannot encode them, so a
    single one anywhere in the context makes every request fail to serialize.
    Walks strings, dicts, lists, tuples and pydantic models; anything that needs
    no change is returned as the same object, so callers' data is never mutated.
    """
    if isinstance(value, str):
        return _utf8_safe_text(value)
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
    blocks: Sequence["ContextItem"],
    *,
    block_formatter: BlockFormatter,
    provider_formatter: ProviderFormatter,
    context_format: "FormatConfig | None" = None,
    event_format: "FormatConfig | None" = None,
    event_format_resolver: Callable[[Any], "FormatConfig | None"] | None = None,
    model_context_window: int | None = None,
    reserved_output_tokens: int | None = None,
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
            if segment_index:
                raise UnsupportedContextLayout("A view may place at most one CacheBoundary")
            segments.append((segment, True))
            segment = []
            segment_index += 1
            continue
        if isinstance(item, Block):
            from nooa.agentdoc import pformat

            block = ResolvedBlock(
                key=item.key,
                content=item.content
                if isinstance(item.content, str)
                else pformat(
                    item.content,
                    unquote_strings=True,
                    **(context_format.model_dump() if context_format is not None else {}),
                ),
                role=item.role,
                metadata=item.metadata or BlockMetadata(),
            )
        elif isinstance(item, EventBase):
            tag = item.tag if item.tag is not None else item.id
            block = ResolvedBlock(
                key=f"event_{tag}",
                content="",
                role=getattr(item, "_role", Role.USER),
                metadata=BlockMetadata(tag=tag),
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
        # Escape lone surrogates so the request and the journal can encode it.
        block = _with_utf8_safe_content(block)
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
    # We record raw character sizes (after the view's eviction) so the provider total can
    # later be attributed across context blocks vs. events by character share.
    stats = ContextWindowStats(
        context_blocks_count=len(context_blocks),
        events_count=len(event_blocks),
        prompt_tokens=None,
        context_blocks_chars=sum(len(b.content) for b in context_blocks),
        events_chars=sum(len(b.content) for b in event_blocks),
        model_context_window=model_context_window,
        context_blocks_dropped=sum(block.metadata.evicted for block in context_blocks),
        events_dropped=0,
        reserved_output_tokens=reserved_output_tokens,
    )

    # Neutral message list → provider wire format. Block content is already
    # escaped; this pass covers the remaining fields (tool calls, images,
    # parts). Both the request and the tracing journal encode these messages.
    messages: list[RenderedMessage] = []
    for segment, boundary_after in segments:
        rendered = block_formatter.format(segment) if segment else []
        messages.extend(rendered)
        if boundary_after:
            messages.append(RenderedMessage(role=Role.METADATA, replay_message=CacheBoundary()))
    messages = _utf8_safe(messages)
    output = provider_formatter.format(messages)
    return RenderResult(
        output=output,
        stats=stats,
        messages=messages,
    )
