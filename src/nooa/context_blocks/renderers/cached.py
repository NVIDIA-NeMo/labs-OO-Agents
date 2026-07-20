# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cached renderer: partitions blocks by ``metadata.static`` for prefix caching.

Structure produced:

    (SYSTEM)    static blocks, stable across turns — cacheable prefix
    (events)    the full event history, append-only
    (USER)      trailing message wrapping dynamic blocks in a ``<context>``
                envelope (always emitted as its own message — never merged
                into a historical event — so the bytes of every prior
                message stay stable across consecutive renders, which is
                what enables provider-side prompt caching to hit on the
                event tail)

Implemented as a single :class:`CachedBlockFormatter`. Pair with any stock
provider formatter (``OpenAIProviderFormatter``, ``AnthropicProviderFormatter``);
no paired provider formatter is needed.

Decoration is minimal by design: no "you are an agent" prose. The format
description mirrors XMLBlockFormatter since the wire format is the same XML
envelope.
"""

from nooa.context_blocks.formatter import (
    FORMAT_XML,
    BlockFormatter,
    FormatType,
    _event_block_to_messages,
    _xml_system_block,
)
from nooa.context_blocks.models import (
    BlockPart,
    MessagePart,
    RenderedMessage,
    ResolvedBlock,
    Role,
    TextPart,
)


def _partition(
    system_blocks: list[ResolvedBlock],
) -> tuple[list[ResolvedBlock], list[ResolvedBlock]]:
    """Split SYSTEM-role blocks into (static, dynamic) by metadata flag."""
    static: list[ResolvedBlock] = []
    dynamic: list[ResolvedBlock] = []
    for b in system_blocks:
        (static if b.metadata.static else dynamic).append(b)
    return static, dynamic


def _xml_concat(blocks: list[ResolvedBlock], separator: str = "\n\n") -> str:
    return separator.join(_xml_system_block(b) for b in blocks)


def _concat_parts(
    blocks: list[ResolvedBlock], separator: str = "\n\n"
) -> tuple[str, list[MessagePart]]:
    """Render ``blocks`` as XML and return (joined_content, parts) for journaling."""
    pieces = [_xml_system_block(b) for b in blocks]
    parts: list[MessagePart] = []
    for i, (block, rendered) in enumerate(zip(blocks, pieces, strict=True)):
        if i > 0:
            parts.append(TextPart(text=separator))
        parts.append(BlockPart(key=block.key, content=rendered))
    return separator.join(pieces), parts


class CachedBlockFormatter(BlockFormatter):
    """Partitions system blocks into a static prefix and a dynamic suffix.

    The static half becomes a SYSTEM message at the head of the output. The
    dynamic half, if any, is wrapped in a ``<context>`` envelope and emitted
    as its own trailing USER message — always appended, never merged into a
    historical event. Merging would mutate the bytes of a prior event message
    when a later turn becomes the new trailing event, which breaks
    provider-side prompt caching for the entire event tail (see issue #208).

    Both the SYSTEM message and the trailing ``<context>`` USER message carry
    ``parts`` with per-block references so the journal publisher can
    content-address each block individually.
    """

    @property
    def format_type(self) -> FormatType:
        return FORMAT_XML

    def format_description(self) -> str:
        return (
            "Your prompt is organized in XML context blocks: `<name>CONTENT</name>`.\n"
            "Blocks produced by `self.context.set_dynamic()` carry an "
            '`expr="..."` attribute whose value is the Python expression '
            "re-evaluated each turn.\n"
            'Event history: system entries in `<sys tag="N">`; '
            'reference via `self.events["N"]`.'
        )

    def format(self, blocks: list[ResolvedBlock]) -> list[RenderedMessage]:
        system_blocks = [b for b in blocks if b.role == Role.SYSTEM]
        message_blocks = [b for b in blocks if b.role != Role.SYSTEM]

        static_blocks, dynamic_blocks = _partition(system_blocks)

        # When no blocks are marked static, fall back to XMLBlockFormatter-style:
        # put everything in SYSTEM. The caching partition only activates when at
        # least one block is explicitly marked static.
        if not static_blocks:
            static_blocks = dynamic_blocks
            dynamic_blocks = []

        messages: list[RenderedMessage] = []
        if static_blocks:
            content, parts = _concat_parts(static_blocks)
            messages.append(RenderedMessage(role=Role.SYSTEM, content=content, parts=parts))

        # Event messages (wrap like XMLBlockFormatter does, except ToolCallEvents
        # still fan out into tool_call + tool_result messages).
        for block in message_blocks:
            messages.extend(
                _event_block_to_messages(
                    block,
                    wrap_content=_xml_message_content_shim,
                )
            )

        if dynamic_blocks:
            dynamic_rendered = [_xml_system_block(b) for b in dynamic_blocks]
            suffix_inner = "\n".join(dynamic_rendered)
            suffix = f"<context>\n{suffix_inner}\n</context>"
            # Build parts for the <context>...</context> envelope.
            envelope_parts: list[MessagePart] = [TextPart(text="<context>\n")]
            for i, (block, rendered) in enumerate(
                zip(dynamic_blocks, dynamic_rendered, strict=True)
            ):
                if i > 0:
                    envelope_parts.append(TextPart(text="\n"))
                envelope_parts.append(BlockPart(key=block.key, content=rendered))
            envelope_parts.append(TextPart(text="\n</context>"))

            # Always append; never merge into a trailing event. Merging would
            # mutate the bytes of a historical event message whenever a later
            # turn becomes the new trailing event, breaking provider prompt
            # caching for the entire event tail (issue #208).
            messages.append(RenderedMessage(role=Role.USER, content=suffix, parts=envelope_parts))

        return messages


def _xml_message_content_shim(block: ResolvedBlock) -> str:
    """Small indirection so ToolCallEvent fan-out still works via the shared helper."""
    from nooa.context_blocks.formatter import _xml_message_content

    return _xml_message_content(block)
