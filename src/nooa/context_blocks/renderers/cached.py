# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cached renderer: partitions blocks by ``metadata.static`` for prefix caching.

Structure produced:

    (SYSTEM)    static blocks, stable across turns — cacheable prefix
    (events)    the full event history, append-only
    (METADATA)  standalone cache boundary, translated by the provider formatter
    (USER)      trailing message wrapping dynamic blocks in a ``<context>``
                envelope (always emitted as its own message — never merged
                into a historical event — so the bytes of every prior
                message stay stable across consecutive renders, which is
                what enables provider-side prompt caching to hit on the
                event tail)

Implemented as a single :class:`CachedBlockFormatter`. Pair with the stock
``OpenAIProviderFormatter``; no paired provider formatter is needed.

Decoration is minimal by design: no "you are an agent" prose. The format
description mirrors XMLBlockFormatter since the wire format is the same XML
envelope.
"""

from nooa.context_blocks.events import LiveContextSnapshot
from nooa.context_blocks.formatter import (
    FORMAT_XML,
    BlockFormatter,
    FormatType,
    _event_blocks_to_messages,
    _xml_system_block,
)
from nooa.context_blocks.live_context import (
    AppendedLiveContext,
    DynamicLiveContext,
    LiveContextPolicy,
    NoLiveContext,
    coerce_live_context,
    snapshot_preface,
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

    A standalone CacheBoundary separates history from live context. Its position
    is decided here; the provider formatter passes the object through unchanged.
    Only UnifiedLLM interprets it when preparing the provider request.
    """

    def __init__(self, *, live_context: LiveContextPolicy | None = None) -> None:
        self.live_context = (
            coerce_live_context(live_context) if live_context is not None else AppendedLiveContext()
        )

    @property
    def format_type(self) -> FormatType:
        return FORMAT_XML

    def format_description(self) -> str:
        expr_note = (
            "Blocks produced by `self.context.set_dynamic()` carry an "
            '`expr="..."` attribute whose value is the Python expression '
            "re-evaluated each turn.\n"
        )
        if isinstance(self.live_context, NoLiveContext):
            expr_note = ""
        return (
            "Your prompt is organized in XML context blocks: `<name>CONTENT</name>`.\n"
            f"{expr_note}"
            'Event history: system entries in `<sys tag="N">`; '
            'reference via `self.events["N"]`.'
        )

    def format(self, blocks: list[ResolvedBlock]) -> list[RenderedMessage]:
        system_blocks = [b for b in blocks if b.role == Role.SYSTEM]
        message_blocks = [b for b in blocks if b.role != Role.SYSTEM]

        static_blocks, dynamic_blocks = _partition(system_blocks)

        # When no block is marked static, fall back to XMLBlockFormatter-style:
        # put the system blocks in SYSTEM. The caching partition only activates
        # when at least one block is explicitly marked static. Under the dynamic
        # policy everything goes to SYSTEM, as before; the other policies keep
        # only fixed text there, so expression-backed blocks still follow the
        # policy (omitted under off, snapshotted on cadence under append).
        if not static_blocks:
            if isinstance(self.live_context, DynamicLiveContext):
                static_blocks, dynamic_blocks = dynamic_blocks, []
            else:
                static_blocks = [b for b in dynamic_blocks if not _is_expression_block(b)]
                dynamic_blocks = [b for b in dynamic_blocks if _is_expression_block(b)]

        messages: list[RenderedMessage] = []
        if static_blocks:
            content, parts = _concat_parts(static_blocks)
            messages.append(RenderedMessage(role=Role.SYSTEM, content=content, parts=parts))

        # The standalone CacheBoundary separates history from live context. Its
        # position is decided here; the provider formatter passes it through and
        # UnifiedLLM maps it to the provider's cache marker.
        from nooa.llm_types import CacheBoundary

        boundary = RenderedMessage(role=Role.METADATA, replay_message=CacheBoundary())

        match self.live_context:
            case DynamicLiveContext():
                messages.extend(_render_events(message_blocks))
                if dynamic_blocks:
                    # Always append; never merge into a trailing event. Merging
                    # would mutate the bytes of a historical event message
                    # whenever a later turn becomes the new trailing event,
                    # breaking provider prompt caching for the entire event
                    # tail (issue #208).
                    messages.append(boundary)
                    messages.append(_snapshot_message(dynamic_blocks, preface=None))
            case NoLiveContext() | AppendedLiveContext():
                # The whole history is the cacheable prefix; nothing follows it.
                # Under append the live context is already in the history: the
                # runtime records a LiveContextSnapshot event before the request
                # (see ``snapshot_event``), rendered raw like what the model saw.
                messages.extend(_render_events(message_blocks))
                messages.append(boundary)
        return messages


def _snapshot_message(
    dynamic_blocks: list[ResolvedBlock], *, preface: str | None
) -> RenderedMessage:
    """The trailing USER message wrapping the dynamic blocks in a ``<context>`` envelope.

    ``preface`` (``append`` mode) is the one-line note that opens the message;
    ``dynamic`` sends the bare envelope. ``parts`` reference each block so the
    journal publisher can content-address them individually.
    """
    dynamic_rendered = [_xml_system_block(b) for b in dynamic_blocks]
    head = f"{preface}\n<context>\n" if preface is not None else "<context>\n"
    content = head + "\n".join(dynamic_rendered) + "\n</context>"
    envelope_parts: list[MessagePart] = [TextPart(text=head)]
    for i, (block, rendered) in enumerate(zip(dynamic_blocks, dynamic_rendered, strict=True)):
        if i > 0:
            envelope_parts.append(TextPart(text="\n"))
        envelope_parts.append(BlockPart(key=block.key, content=rendered))
    envelope_parts.append(TextPart(text="\n</context>"))
    return RenderedMessage(role=Role.USER, content=content, parts=envelope_parts)


def _is_expression_block(block: ResolvedBlock) -> bool:
    """An expression-backed block (``Context(expr=...)`` / ``set_dynamic``), re-evaluated every turn.

    ``source_dynamic`` is the flag the context builder sets only for those;
    fixed values carry an ``expr`` too (``self.context["key"]``, how to read them
    back), so ``expr`` alone does not mean live.
    """
    return block.metadata.source_dynamic


def _render_events(blocks: list[ResolvedBlock]) -> list[RenderedMessage]:
    return _event_blocks_to_messages(blocks, wrap_content=_xml_message_content_shim)


def live_blocks(blocks: list[ResolvedBlock]) -> list[ResolvedBlock]:
    """The live context of a prepared request under ``append``: the blocks the
    formatter would put after the boundary, with the same static-less fallback."""
    system_blocks = [b for b in blocks if b.role == Role.SYSTEM]
    static_blocks, dynamic_blocks = _partition(system_blocks)
    if not static_blocks:
        return [b for b in dynamic_blocks if _is_expression_block(b)]
    return dynamic_blocks


def snapshot_event(live: list[ResolvedBlock], every_n: int) -> LiveContextSnapshot:
    """The :class:`LiveContextSnapshot` for these live blocks: the prefaced envelope the
    model will see, plus each rendered block by key for change detection."""
    message = _snapshot_message(live, preface=snapshot_preface(every_n))
    return LiveContextSnapshot(
        content=message.content or "",
        comparable={b.key: _xml_system_block(b) for b in live},
    )


def _xml_message_content_shim(block: ResolvedBlock) -> str:
    """Small indirection so ToolCallEvent fan-out still works via the shared helper."""
    from nooa.context_blocks.formatter import _xml_message_content

    return _xml_message_content(block)
