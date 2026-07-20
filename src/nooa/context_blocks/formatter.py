# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Formatters for context blocks and provider output.

Two orthogonal formatters:

1. :class:`BlockFormatter` — turns a list of resolved blocks (both system-role
   context blocks and event blocks) into a neutral ``list[RenderedMessage]``.
   Owns semantic choices: how each block is wrapped (XML / Markdown / plain),
   how events are serialized, how system blocks are concatenated into a single
   system message.

2. :class:`ProviderFormatter` — a thin syntactic adapter that reshapes the
   neutral message list into provider-specific wire format (OpenAI ``list[dict]``,
   Anthropic ``{"system": ..., "messages": [...]}``).

This split keeps the "format" axis (XML / Markdown / Plain) orthogonal to the
"provider" axis (OpenAI / Anthropic / ...). Neither knows about the other.
"""

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nooa.config.truncation_config import FormatConfig

from nooa.agentdoc import pformat
from nooa.context_blocks.events import EventBase, ToolCallEvent
from nooa.context_blocks.models import (
    RenderedMessage,
    ResolvedBlock,
    Role,
    ToolCallInfo,
)

logger = logging.getLogger(__name__)


class FormatType(StrEnum):
    """Format type identifier used for block-level truncation."""

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

    Subclasses implement :meth:`format` to decide both the ordering of messages
    and the wrapping of each block's content. The stock implementations
    partition by ``block.role``: SYSTEM blocks concatenate into a single system
    message, event blocks emit per-event messages.
    """

    @property
    @abstractmethod
    def format_type(self) -> FormatType:
        """Return format type (FORMAT_XML / FORMAT_MARKDOWN / FORMAT_PLAIN).

        Used by :func:`render_context` to pick the right block-level truncation
        strategy for oversized content.
        """
        ...

    @abstractmethod
    def format(self, blocks: list[ResolvedBlock]) -> list[RenderedMessage]:
        """Convert all resolved blocks into a neutral message list.

        Args:
            blocks: All resolved blocks, already truncated and with event
                content pre-serialized by :func:`render_context`. System
                blocks appear before event blocks.

        Returns:
            A list of :class:`RenderedMessage`, in final display order.
        """
        ...

    def format_event(
        self,
        event: EventBase,
        event_format: "FormatConfig | None" = None,
    ) -> str:
        """Serialize a non-tool-call event into string content.

        Called by :func:`render_context` to materialize event content as a
        string. ToolCallEvents are handled structurally by :meth:`format` —
        they are not routed through ``format_event``.

        ASSISTANT-role events (``Message``, ``Reasoning``, ``LLMOutput``) with
        string content replay verbatim — no ``EventName(...)`` repr — so the
        model's past turns look exactly like what it naturally emits.
        Non-string content (e.g. structured Pydantic models from Predict)
        falls through to ``pformat`` like other events.

        Everything else uses :func:`pformat` with structural bounds
        (``max_string``, ``max_length``, ``max_depth``) from ``event_format``.
        No OOM-safety cap is applied here — that belongs to L2 (stdout/stderr
        capture).
        """
        if getattr(event, "_role", None) is Role.ASSISTANT:
            content = getattr(event, "content", None)
            if isinstance(content, str):
                return content
        kwargs: dict[str, Any] = {}
        if event_format is not None:
            kwargs.update(event_format.model_dump())
        return pformat(event, **kwargs)

    def format_description(self) -> str:
        """Return a human-readable description of the block format.

        Embedded in the default agent system prompt so the LLM understands
        how context blocks are structured. Subclasses should override.
        """
        return ""


# ---------------------------------------------------------------------------
# Helpers shared by the stock BlockFormatters
# ---------------------------------------------------------------------------


def _xml_system_block(block: ResolvedBlock) -> str:
    """Wrap a system block's content in an XML tag.

    ``expr=`` is only emitted for blocks produced by
    :meth:`ContextApi.set_dynamic` — static blocks, framework blocks, and
    strategy overrides render without the attribute so the prompt doesn't
    leak internal accessor syntax the LLM can't use.
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
    * ASSISTANT-role events (``Message``, ``Reasoning``, ``LLMOutput``)
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

    if isinstance(block.event, ToolCallEvent):
        event = block.event
        messages: list[RenderedMessage] = [
            RenderedMessage(
                role=Role.ASSISTANT,
                tool_call=ToolCallInfo(
                    id=event.tool_call_id,
                    name=event.name,
                    arguments=event.arguments,
                ),
            )
        ]
        if event.result is not None:
            messages.append(
                RenderedMessage(
                    role=Role.TOOL,
                    content=event.result.content,
                    tool_call_id=event.tool_call_id,
                )
            )
        else:
            # Defensive: ToolCallEvent with result=None should not happen in
            # normal flow, but if it does (e.g. a GenerationError was raised
            # before the caller could update the event), emit a minimal
            # tool_result to avoid producing tool_use without a matching
            # tool_result — which corrupts the conversation for the next
            # session.
            logger.warning(
                "ToolCallEvent %s has result=None — emitting placeholder tool_result "
                "to prevent context corruption.",
                event.tool_call_id,
            )
            messages.append(
                RenderedMessage(
                    role=Role.TOOL,
                    content="(no result recorded)",
                    tool_call_id=event.tool_call_id,
                )
            )
        return messages

    # Non-tool event
    content = (
        wrap_content(block)
        if wrap_content is not None and (block.metadata.expr or block.metadata.tag)
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


def _build_messages(
    blocks: list[ResolvedBlock],
    *,
    wrap_system: "Callable[[ResolvedBlock], str]",
    wrap_message: "Callable[[ResolvedBlock], str] | None",
    system_separator: str = "\n\n",
) -> list[RenderedMessage]:
    """Common BlockFormatter.format() body used by XML and Markdown variants.

    Always emits a SYSTEM message (empty content if no SYSTEM blocks) unless the
    input is completely empty. This matches the old renderer's behavior of
    always including a system slot for providers that expect one.

    The SYSTEM message is emitted with block-aware ``parts`` so the journal
    publisher can separate each block's content for dedup/hashing.
    """
    if not blocks:
        return []

    from nooa.context_blocks.models import BlockPart, TextPart

    system_blocks = [b for b in blocks if b.role == Role.SYSTEM]
    event_blocks = [b for b in blocks if b.role != Role.SYSTEM]

    system_rendered = [wrap_system(b) for b in system_blocks]
    system_content = system_separator.join(system_rendered)

    system_parts: list[TextPart | BlockPart] = []
    for i, (block, rendered) in enumerate(zip(system_blocks, system_rendered, strict=True)):
        if i > 0:
            system_parts.append(TextPart(text=system_separator))
        system_parts.append(BlockPart(key=block.key, content=rendered))

    messages: list[RenderedMessage] = [
        RenderedMessage(
            role=Role.SYSTEM,
            content=system_content,
            parts=system_parts if system_parts else None,
        )
    ]
    for block in event_blocks:
        messages.extend(_event_block_to_messages(block, wrap_content=wrap_message))
    return messages


# ---------------------------------------------------------------------------
# Stock BlockFormatters
# ---------------------------------------------------------------------------


class XMLBlockFormatter(BlockFormatter):
    """Wrap each system block in an XML tag; wrap each event in a role tag.

    Example system output::

        <notes expr="self.context['notes']">
        Here are my notes...
        </notes>

    Example event output::

        <user_message expr='self.events["1"]' tag="1">
        Hello world
        </user_message>
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
            'reference via `self.events["N"]`. Your own messages appear '
            "as plain assistant turns."
        )

    def format(self, blocks: list[ResolvedBlock]) -> list[RenderedMessage]:
        return _build_messages(
            blocks,
            wrap_system=_xml_system_block,
            wrap_message=_xml_message_content,
        )


class MarkdownBlockFormatter(BlockFormatter):
    """Render each system block as a Markdown section; wrap events with Markdown headers.

    Example system output::

        # Notes `{"expr": "self.context['notes']"}`

        Here are my notes...
    """

    @property
    def format_type(self) -> FormatType:
        return FORMAT_MARKDOWN

    def format_description(self) -> str:
        return (
            "Your prompt is organized in context blocks with markdown headers: "
            "`# Block Name`.\n"
            "Blocks produced by `self.context.set_dynamic()` carry an inline "
            '`{"expr": "..."}` annotation whose value is the Python expression '
            "re-evaluated each turn."
        )

    def format(self, blocks: list[ResolvedBlock]) -> list[RenderedMessage]:
        return _build_messages(
            blocks,
            wrap_system=_markdown_system_block,
            wrap_message=_markdown_message_content,
        )


# ---------------------------------------------------------------------------
# ProviderFormatter — neutral messages → provider wire format
# ---------------------------------------------------------------------------


class ProviderFormatter(ABC):
    """Reshape a neutral :class:`RenderedMessage` list into provider wire format.

    Thin syntactic adapter: does not wrap content, does not re-order, does not
    decide what goes where. Just translates message shape and tool-call
    conventions to the target API.
    """

    @abstractmethod
    def format(self, messages: list[RenderedMessage]) -> Any:
        """Return provider-specific output for this message list."""
        ...


def _append_openai_image_message(out: list[dict], msg: RenderedMessage) -> None:
    content_parts: list[dict] = [{"type": "text", "text": msg.content or ""}]
    content_parts.extend(msg.images or [])
    out.append({"role": msg.role.value, "content": content_parts})


class OpenAIProviderFormatter(ProviderFormatter):
    """Emit OpenAI-compatible messages (``list[dict]``)."""

    def format(self, messages: list[RenderedMessage]) -> list[dict]:
        out: list[dict] = []
        for msg in messages:
            if msg.tool_call is not None:
                out.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": msg.tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": msg.tool_call.name,
                                    "arguments": json.dumps(msg.tool_call.arguments),
                                },
                            }
                        ],
                    }
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
            else:
                if msg.role in (Role.RUNTIME_EVENT, Role.METADATA):
                    continue
                out.append({"role": msg.role.value, "content": msg.content or ""})
        return out


class AnthropicProviderFormatter(ProviderFormatter):
    """Emit Anthropic-native messages (``{"system": str, "messages": list[dict]}``)."""

    def format(self, messages: list[RenderedMessage]) -> dict:
        system_parts: list[str] = []
        out: list[dict] = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                if msg.content:
                    system_parts.append(msg.content)
                continue

            if msg.tool_call is not None:
                out.append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": msg.tool_call.id,
                                "name": msg.tool_call.name,
                                "input": msg.tool_call.arguments,
                            }
                        ],
                    }
                )
            elif msg.tool_call_id is not None:
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id,
                                "content": msg.content or "",
                            }
                        ],
                    }
                )
            elif msg.images:
                # Keep the universal LiteLLM image_url shape — LiteLLM translates for Anthropic.
                role = msg.role if msg.role in (Role.USER, Role.ASSISTANT) else Role.USER
                content_parts: list[dict] = [{"type": "text", "text": msg.content or ""}]
                content_parts.extend(msg.images)
                out.append({"role": role.value, "content": content_parts})
            else:
                if msg.role in (Role.RUNTIME_EVENT, Role.METADATA):
                    continue
                role = msg.role if msg.role in (Role.USER, Role.ASSISTANT) else Role.USER
                out.append({"role": role.value, "content": msg.content or ""})

        return {"system": "\n\n".join(system_parts), "messages": out}


class ResponsesProviderFormatter(ProviderFormatter):
    """Emit OpenAI Responses API native format (``list[dict]``).

    The Responses API uses a different wire format from Chat Completions:
    - Tool calls → ``{"type": "function_call", "call_id": ..., "name": ..., "arguments": ...}``
    - Tool results → ``{"type": "function_call_output", "call_id": ..., "output": ...}``
    - User/Assistant messages → ``{"role": "user"|"assistant", "content": ...}``
    - System messages → kept as ``{"role": "system", ...}`` in the list so that
      downstream budget-clamping can identify and protect them. The LLM client
      extracts them into the ``instructions`` API parameter at call time.
    """

    def format(self, messages: list[RenderedMessage]) -> list[dict]:
        out: list[dict] = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                out.append({"role": "system", "content": msg.content or ""})
                continue

            if msg.tool_call is not None:
                # Preserve assistant text that precedes the tool call
                if msg.content and msg.role == Role.ASSISTANT:
                    out.append({"role": "assistant", "content": msg.content})
                out.append(
                    {
                        "type": "function_call",
                        "call_id": msg.tool_call.id,
                        "name": msg.tool_call.name,
                        "arguments": json.dumps(msg.tool_call.arguments)
                        if isinstance(msg.tool_call.arguments, dict)
                        else msg.tool_call.arguments,
                    }
                )
            elif msg.tool_call_id is not None:
                out.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.tool_call_id,
                        "output": msg.content or "",
                    }
                )
            elif msg.images:
                role = msg.role if msg.role in (Role.USER, Role.ASSISTANT) else Role.USER
                content_parts: list[dict] = []
                if msg.content:
                    content_parts.append({"type": "input_text", "text": msg.content})
                for img in msg.images:
                    if img.get("type") == "image_url":
                        # Responses API requires input_image, and its image_url must
                        # be the URL STRING (not the Chat-Completions {"url": ...}
                        # object) — otherwise the API rejects it with
                        # "expected an image URL, but got an object instead".
                        iu = img["image_url"]
                        if isinstance(iu, dict):
                            url = iu.get("url")
                            if not url:
                                # Fail fast: an empty image_url only yields an opaque
                                # "invalid URL" from the API, hiding the real problem.
                                raise ValueError(
                                    "image_url dict has no 'url'; cannot build a "
                                    f"Responses input_image block: {iu!r}"
                                )
                            part = {"type": "input_image", "image_url": url}
                            if iu.get("detail"):
                                part["detail"] = iu["detail"]
                        else:
                            part = {"type": "input_image", "image_url": iu}
                        content_parts.append(part)
                    else:
                        content_parts.append(img)
                out.append({"role": role.value, "content": content_parts})
            else:
                if msg.role in (Role.RUNTIME_EVENT, Role.METADATA):
                    continue
                role = msg.role if msg.role in (Role.USER, Role.ASSISTANT) else Role.USER
                out.append({"role": role.value, "content": msg.content or ""})
        return out
