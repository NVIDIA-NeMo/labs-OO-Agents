# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Formatters for context blocks and provider output.

Two orthogonal formatters:

1. :class:`BlockFormatter` — turns a list of resolved blocks (both system-role
   context blocks and event blocks) into a neutral ``list[RenderedMessage]``.
   Preserves view-selected order and membership while wrapping blocks
   (XML / Markdown / plain), serializing events, and coalescing adjacent
   materialized blocks with the same role. Canonical tool replay is atomic.

2. :class:`ProviderFormatter` — a thin syntactic adapter that reshapes the
   neutral message list into client input: ordinary message dictionaries,
   retained ``LLMResponse`` objects, and ``CacheBoundary`` markers.
   Provider-specific wire translation happens downstream
   (LiteLLM, or the client's own projection), never here.

This split keeps the "format" axis (XML / Markdown / Plain) orthogonal to the
message-shape axis. Neither knows about the other.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypeGuard

if TYPE_CHECKING:
    from nooa.config.truncation_config import FormatConfig
    from nooa.llm_types import CacheBoundary, LLMResponse

from nooa.agentdoc import pformat
from nooa.context_blocks.events import CODEACT_INLINE_RETURN, EventBase, ToolCallEvent
from nooa.context_blocks.exceptions import UnsupportedContextLayout
from nooa.context_blocks.models import (
    RenderedMessage,
    ResolvedBlock,
    Role,
    ToolCallInfo,
)
from nooa.llm_types import assistant_message

logger = logging.getLogger(__name__)


def _is_llm_response(event: EventBase | None) -> TypeGuard["LLMResponse"]:
    """Recognize the public assistant IR without exposing its provider internals.

    The formatter owns event-to-message semantics. Keeping this dispatch here
    avoids assistant-specific hooks on every EventBase subclass. The lazy import
    avoids the cycle through LLMResponse's EventBase definition.
    """
    from nooa.llm_types import LLMResponse

    return isinstance(event, LLMResponse)


class FormatType(StrEnum):
    """Block/message wrapping style, independent of selection and budgeting."""

    XML = "xml"
    MARKDOWN = "markdown"
    PLAIN = "plain"


FORMAT_XML = FormatType.XML
FORMAT_MARKDOWN = FormatType.MARKDOWN
FORMAT_PLAIN = FormatType.PLAIN


# ---------------------------------------------------------------------------
# BlockFormatter — blocks → neutral message list
# ---------------------------------------------------------------------------


class BlockFormatter(ABC):
    """Turn resolved blocks into a neutral list of :class:`RenderedMessage`.

    The view owns selection, placement, eviction, and cache boundaries.
    Subclasses serialize its ordered inputs without repartitioning by role.
    Stock implementations coalesce adjacent materialized blocks of the same
    role and expand canonical tool-call responses with their linked results
    atomically at the response position. Incomplete replay groups may be
    omitted; missing results are not invented. Wrapper customizations should
    reuse the stock formatting path to retain these replay rules.
    """

    @property
    @abstractmethod
    def format_type(self) -> FormatType:
        """Return format type (FORMAT_XML / FORMAT_MARKDOWN / FORMAT_PLAIN).

        Identifies wrapper style for message-formatting helpers; it does not
        select content or control truncation, eviction, or caching.
        """
        ...

    def format(self, blocks: list[ResolvedBlock]) -> list[RenderedMessage]:
        """Convert all resolved blocks into a neutral message list.

        Args:
            blocks: One ordered segment selected by the view, with non-tool
                event content serialized by :func:`render_context`. Roles may
                be interleaved. Cache markers delimit segments outside this
                method; no renderer-owned eviction or truncation occurs.

        Returns:
            A list of :class:`RenderedMessage`, in final display order.
        """
        return _build_messages(
            blocks,
            wrap_system=self.wrap_system_block,
            wrap_message=self.wrap_message_block,
        )

    def wrap_system_block(self, block: ResolvedBlock) -> str:
        """Wrap one materialized block, regardless of its selected message role.

        Override this hook to customize presentation while retaining shared
        ordering and canonical tool-replay handling in :meth:`format`.
        """
        return block.content

    def wrap_message_block(self, block: ResolvedBlock) -> str:
        """Wrap an ordinary event; canonical assistant replay bypasses this hook."""
        return block.content

    def format_event(
        self,
        event: EventBase,
        event_format: "FormatConfig | None" = None,
    ) -> str:
        """Serialize a non-tool-call event into string content.

        Called by :func:`render_context` to materialize event content as a
        string. ToolCallEvents are handled structurally by :meth:`format` —
        they are not routed through ``format_event``.

        ASSISTANT-role events (``Message``, ``Reasoning``, ``LLMResponse``) with
        string content replay verbatim — no ``EventName(...)`` repr — so the
        model's past turns look exactly like what it naturally emits.
        Non-string content (e.g. structured Pydantic models from Predict)
        falls through to ``pformat`` like other events.

        Everything else uses :func:`pformat` with structural bounds
        (``max_string``, ``max_length``, ``max_depth``) from ``event_format``.
        No OOM-safety cap is applied here — that belongs to L2 (stdout/stderr
        capture).
        """
        if _is_llm_response(event):
            return event.replay_content
        if event.event_role is Role.ASSISTANT:
            content = getattr(event, "content", None)
            if isinstance(content, str):
                return content
        kwargs: dict[str, Any] = {}
        if event_format is not None:
            kwargs.update(event_format.model_dump())
        return pformat(event, **kwargs)

    def format_description(self) -> str:
        """Return a human-readable description of the block format.

        Built-in manager-backed context views explicitly include this in the
        built-in Agent prompt. Custom prompts and views choose their own guidance;
        rendering never injects instructions. Return an empty string for no guidance.
        """
        return ""


# ---------------------------------------------------------------------------
# Helpers shared by the stock BlockFormatters
# ---------------------------------------------------------------------------


def _xml_system_block(block: ResolvedBlock) -> str:
    """Wrap a system block's content in an XML tag.

    ``expr=`` is emitted only when metadata marks a dynamic source, including
    expression context and default framework sources. Literal declarations
    and value overrides omit the attribute.
    """
    show_expr = block.metadata.source_dynamic and block.metadata.expr
    attrs = f' expr="{block.metadata.expr}"' if show_expr else ""
    return f"<{block.key}{attrs}>\n{block.content}\n</{block.key}>"


def _xml_message_content(block: ResolvedBlock) -> str:
    """Wrap an event block's content with a minimal role tag carrying metadata.

    * ``<sys>`` — USER-role events are framework-generated feedback to
      the LLM (``Task`` prompts, ``PythonOutput`` cells, ``Error`` /
      ``Feedback``, etc.). They share one wrapper because the event's
      class name is already inside the rendered content
      (``PythonOutput(...)``, ``Task(...)``) — nothing is lost, and the
      tag stays uniform across event types.
    * ASSISTANT-role events (``Message``, ``Reasoning``, ``LLMResponse``)
      are the LLM's own outputs: they pass through **unwrapped** so the
      model's past turns replay exactly as it produced them. Wrapping
      them (the old ``<agent>`` tag) taught models to emit
      ``<agent>...</agent>`` literally in new outputs.

    Other roles fall back to ``{role}_message`` (rare — events mostly
    carry USER or ASSISTANT).
    """
    if block.role == Role.ASSISTANT:
        return block.content
    role_label = "sys" if block.role == Role.USER else f"{block.role.value}_message"
    attrs: list[str] = []
    if block.metadata.source_dynamic and block.metadata.expr:
        attrs.append(f'expr="{block.metadata.expr}"')
    if block.metadata.tag:
        attrs.append(f'tag="{block.metadata.tag}"')
    attr_str = (" " + " ".join(attrs)) if attrs else ""
    return f"<{role_label}{attr_str}>\n{block.content}\n</{role_label}>"


def _markdown_system_block(block: ResolvedBlock) -> str:
    header = block.key.replace("_", " ").title()
    inline_meta = ""
    if block.metadata.source_dynamic and block.metadata.expr:
        inline_meta = ' `{"expr": "' + block.metadata.expr + '"}`'
    return f"# {header}{inline_meta}\n\n{block.content}"


def _markdown_message_content(block: ResolvedBlock) -> str:
    role_label = block.role.value.replace("_", " ").title() + " Message"
    meta_parts: list[str] = []
    if block.metadata.source_dynamic and block.metadata.expr:
        meta_parts.append(f'"expr": "{block.metadata.expr}"')
    if block.metadata.tag:
        meta_parts.append(f'"tag": "{block.metadata.tag}"')
    inline_meta = (" `{" + ", ".join(meta_parts) + "}`") if meta_parts else ""
    return f"### {role_label}{inline_meta}\n\n{block.content}"


def _tool_result_message(event: ToolCallEvent) -> RenderedMessage:
    """Project one execution event into a provider-neutral tool result."""
    if event.result is not None:
        content = event.result.content
    else:
        logger.warning(
            "ToolCallEvent %s has result=None — emitting placeholder tool_result "
            "to prevent context corruption.",
            event.tool_call_id,
        )
        content = "(no result recorded)"
    return RenderedMessage(
        role=Role.TOOL,
        content=content,
        tool_call_id=event.tool_call_id,
    )


def _event_block_to_messages(
    block: ResolvedBlock,
    *,
    wrap_content: "Callable[[ResolvedBlock], str] | None",
) -> list[RenderedMessage]:
    """Convert a single event block into one or two :class:`RenderedMessage`s.

    ToolCallEvents fan out into an assistant tool-call message and a tool
    result message. Other events produce a single role-tagged message with
    wrapped content.

    Non-tool messages carry a one-element ``parts=[BlockPart(...)]`` so
    the journal publisher can content-address each event independently —
    repeated events across turns hash identically and don't retransmit.
    """
    from nooa.context_blocks.models import BlockPart

    if _is_llm_response(block.event):
        event = block.event
        if event.is_empty:
            return []
        # The event hook emits a portable replacement when rendering omits calls
        # or edits text, so incomplete turns cannot retain native authority.
        return [
            RenderedMessage(
                role=Role.ASSISTANT,
                content=block.content,
                reasoning=event.reasoning,
                replay_message=event,
            )
        ]

    if isinstance(block.event, ToolCallEvent):
        event = block.event
        if event.metadata.get("synthetic_type") == CODEACT_INLINE_RETURN:
            # Inline return_result() ran inside a Python cell. CodeAct records its
            # value for traces, but the provider never issued a separate return_result
            # tool call. Replaying this marker would invent an assistant turn/tool
            # pair (and duplicate the cell's completion) in subsequent prompts.
            return []
        return [
            RenderedMessage(
                role=Role.ASSISTANT,
                tool_calls=(
                    ToolCallInfo(
                        id=event.tool_call_id,
                        name=event.name,
                        arguments=event.arguments,
                    ),
                ),
            ),
            _tool_result_message(event),
        ]

    # Non-tool event
    content = (
        wrap_content(block)
        if wrap_content is not None
        and (block.event is None or block.metadata.expr or block.metadata.tag)
        else (block.content or "")
    )
    images = getattr(block.event, "images", None) if block.event is not None else None
    return [
        RenderedMessage(
            role=block.role,
            content=content,
            parts=[BlockPart(key=block.key, content=content)],
            images=images,
        )
    ]


def _event_blocks_to_messages(
    blocks: list[ResolvedBlock],
    *,
    wrap_content: "Callable[[ResolvedBlock], str] | None",
) -> list[RenderedMessage]:
    """Project the public event-list IR into neutral conversation turns.

    ``LLMResponse`` is the canonical assistant turn. Linked ``ToolCallEvent``
    objects describe executions of calls from that turn; they are not separate
    assistant turns. Grouping here preserves the provider's call batch and
    prevents call/result interleaving from changing its meaning.

    Legacy and synthetic ``ToolCallEvent`` objects without a response link
    remain independently renderable. Linked executions are only projected as
    part of a complete canonical response/result batch.
    """
    return [
        message
        for projection in _event_block_projections(blocks, wrap_content=wrap_content)
        for message in projection
    ]


def _event_block_projections(
    blocks: list[ResolvedBlock],
    *,
    wrap_content: "Callable[[ResolvedBlock], str] | None",
) -> list[list[RenderedMessage]]:
    """Project each input position, keeping canonical tool batches atomic.

    A replayable ``LLMResponse`` expands at its own position to the assistant
    call batch followed by all linked results in call order, even when their
    stored positions are interleaved with other items. Linked execution
    sidecars then project to an empty list at their stored positions. Missing
    or incomplete canonical groups are omitted without inventing results.
    This replay exception does not authorize arbitrary history reordering.
    Wrapper customizations can reuse this helper through _build_messages().
    """
    replayable_turn_ids = {
        block.event.id
        for block in blocks
        if _is_llm_response(block.event) and block.event.replay_tool_calls
    }
    executions: dict[str, dict[str, ToolCallEvent]] = {}
    for block in blocks:
        event = block.event
        if (
            isinstance(event, ToolCallEvent)
            and event.llm_response_id is not None
            and event.llm_response_id in replayable_turn_ids
        ):
            executions.setdefault(event.llm_response_id, {})[event.tool_call_id] = event

    projections: list[list[RenderedMessage]] = []
    for block in blocks:
        event = block.event
        if _is_llm_response(event) and event.replay_tool_calls:
            by_call_id = executions.get(event.id, {})
            # Old archives/direct constructions can contain duplicate ids.
            # They cannot be paired with executions, even when one id matches.
            calls = event.replay_tool_calls
            if len({call.id for call in calls}) != len(calls) or any(
                call.id not in by_call_id or by_call_id[call.id].result is None
                for call in event.replay_tool_calls
            ):
                logger.warning(
                    "LLMResponse %s has an incomplete visible execution batch — "
                    "omitting the assistant tool-call turn from replay.",
                    event.id,
                )
                projections.append([])
                continue
            projection = [
                RenderedMessage(
                    role=Role.ASSISTANT,
                    content=block.content,
                    reasoning=event.reasoning,
                    tool_calls=tuple(
                        ToolCallInfo(id=call.id, name=call.name, arguments=call.arguments)
                        for call in event.replay_tool_calls
                    ),
                    replay_message=event,
                )
            ]
            for call in event.replay_tool_calls:
                projection.append(_tool_result_message(by_call_id[call.id]))
            projections.append(projection)
            continue

        # A linked execution is not an independent assistant turn. If its
        # source response was filtered out (or its batch is incomplete), fail
        # closed instead of fabricating provider history from the sidecar.
        if isinstance(event, ToolCallEvent) and event.llm_response_id is not None:
            projections.append([])
            continue

        projections.append(_event_block_to_messages(block, wrap_content=wrap_content))
    return projections


def _build_messages(
    blocks: list[ResolvedBlock],
    *,
    wrap_system: "Callable[[ResolvedBlock], str]",
    wrap_message: "Callable[[ResolvedBlock], str] | None",
    system_separator: str = "\n\n",
) -> list[RenderedMessage]:
    """Common BlockFormatter.format() body used by XML and Markdown variants.

    Preserves input positions. Adjacent materialized blocks with the same role
    coalesce into one block-aware message. Canonical assistant tool-call turns
    expand atomically at the source response position.
    """
    if not blocks:
        return []

    from nooa.context_blocks.models import BlockPart, TextPart

    projections = _event_block_projections(blocks, wrap_content=wrap_message)
    messages: list[RenderedMessage] = []
    pending_blocks: list[ResolvedBlock] = []

    def flush_blocks() -> None:
        if not pending_blocks:
            return
        rendered = [wrap_system(block) for block in pending_blocks]
        parts: list[TextPart | BlockPart] = []
        for i, (block, content) in enumerate(zip(pending_blocks, rendered, strict=True)):
            if i:
                parts.append(TextPart(text=system_separator))
            parts.append(BlockPart(key=block.key, content=content))
        messages.append(
            RenderedMessage(
                role=pending_blocks[0].role,
                content=system_separator.join(rendered),
                parts=parts,
            )
        )
        pending_blocks.clear()

    for block, projection in zip(blocks, projections, strict=True):
        if block.event is None:
            if pending_blocks and pending_blocks[0].role != block.role:
                flush_blocks()
            pending_blocks.append(block)
            continue
        if not projection:
            continue
        flush_blocks()
        messages.extend(projection)
    flush_blocks()
    return messages


# ---------------------------------------------------------------------------
# Stock BlockFormatters
# ---------------------------------------------------------------------------


class XMLBlockFormatter(BlockFormatter):
    """Wrap materialized blocks in XML; annotate event role/tag metadata.

    Canonical assistant responses replay unwrapped.

    Example system output::

        <notes expr="self.context['notes']">
        Here are my notes...
        </notes>

    Example event output::

        <sys tag="1">
        Hello world
        </sys>
    """

    @property
    def format_type(self) -> FormatType:
        return FORMAT_XML

    def format_description(self) -> str:
        return (
            "Your prompt is organized in XML context blocks: `<name>CONTENT</name>`.\n"
            "Expression context blocks carry an "
            '`expr="..."` attribute whose value is the Python expression '
            "re-evaluated each turn.\n"
            'Event history: system entries in `<sys tag="N">`; '
            'reference via `self.events["N"]`. Your own messages appear '
            "as plain assistant turns."
        )

    def wrap_system_block(self, block: ResolvedBlock) -> str:
        return _xml_system_block(block)

    def wrap_message_block(self, block: ResolvedBlock) -> str:
        return _xml_message_content(block)


class MarkdownBlockFormatter(BlockFormatter):
    """Render materialized blocks as Markdown sections and annotate event metadata.

    Canonical assistant responses retain their original text and replay data.

    Example system output::

        # Notes `{"expr": "self.context['notes']"}`

        Here are my notes...

    Example event output::

        ### User Message `{"tag": "1"}`

        Hello world
    """

    @property
    def format_type(self) -> FormatType:
        return FORMAT_MARKDOWN

    def format_description(self) -> str:
        return (
            "Your prompt is organized in context blocks with markdown headers: "
            "`# Block Name`.\n"
            "Expression context blocks carry an inline "
            '`{"expr": "..."}` annotation whose value is the Python expression '
            "re-evaluated each turn.\n"
            'Event history carries `{"tag": "N"}` metadata; '
            'reference entries via `self.events["N"]`. Your own messages '
            "retain their original assistant text and replay data."
        )

    def wrap_system_block(self, block: ResolvedBlock) -> str:
        return _markdown_system_block(block)

    def wrap_message_block(self, block: ResolvedBlock) -> str:
        return _markdown_message_content(block)


# ---------------------------------------------------------------------------
# ProviderFormatter — neutral messages → client input
# ---------------------------------------------------------------------------


class ProviderFormatter(ABC):
    """Reshape neutral messages into client input, preserving replay carriers.

    Thin syntactic adapter: does not wrap content, does not re-order, does not
    decide what goes where. Just translates message shape and tool-call
    conventions to the target API.
    """

    @abstractmethod
    def format(self, messages: list[RenderedMessage]) -> Any:
        """Return client input; final provider wire encoding happens downstream."""
        ...


def _append_openai_image_message(
    out: "list[dict[str, Any] | LLMResponse | CacheBoundary]", msg: RenderedMessage
) -> None:
    content_parts: list[dict] = [{"type": "text", "text": msg.content or ""}]
    content_parts.extend(msg.images or [])
    out.append({"role": msg.role.value, "content": content_parts})


def _with_reasoning(message: dict[str, Any], reasoning: str | None) -> dict[str, Any]:
    """Portable reasoning on synthetic messages needs no native-state carrier."""
    if reasoning:
        message["reasoning_content"] = reasoning
    return message


class OpenAIProviderFormatter(ProviderFormatter):
    """Emit UnifiedLLM input while retaining native replay and cache markers.

    This is not final wire encoding. The client owns provider-specific replay
    projection, including retained reasoning and exact tool argument strings.
    """

    def format(
        self, messages: list[RenderedMessage]
    ) -> "list[dict[str, Any] | LLMResponse | CacheBoundary]":
        out: list[dict[str, Any] | LLMResponse | CacheBoundary] = []
        for msg in messages:
            if msg.replay_message is not None:
                out.append(
                    msg.replay_message.render_message(
                        msg.content, msg.tool_calls, reasoning=msg.reasoning
                    )
                )
            elif msg.role in (Role.RUNTIME_EVENT, Role.METADATA):
                raise UnsupportedContextLayout(
                    f"OpenAI cannot represent context role {msg.role.value!r}"
                )
            elif msg.tool_calls:
                out.append(
                    assistant_message(
                        msg.content, tool_calls=msg.tool_calls, reasoning=msg.reasoning
                    )
                )
            elif msg.tool_call_id is not None:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.tool_call_id,
                        "content": msg.content or "",
                    }
                )
            elif msg.images:
                _append_openai_image_message(out, msg)
            elif msg.role is Role.ASSISTANT:
                out.append(assistant_message(msg.content, reasoning=msg.reasoning))
            else:
                out.append(
                    _with_reasoning(
                        {"role": msg.role.value, "content": msg.content or ""},
                        msg.reasoning,
                    )
                )
        return out
