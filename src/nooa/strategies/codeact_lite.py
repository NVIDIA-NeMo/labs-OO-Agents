# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CodeActLite strategy - simplified CodeAct with clean message rendering.

A variant of CodeActStrategy that removes three sources of LLM confusion:

1. **Scoped events** — Only shows events from the current method call
2. **No XML tags on messages** — Messages render as plain text, not wrapped in
   `<user_message expr="..." tag="...">...</user_message>`
3. **No Python type wrappers** — Events render as plain text, not
   `Task(prompt='...')` but just the prompt content
4. **Inline tool results** — PythonOutput is merged into the tool response
   instead of appearing as a separate user message

System prompt blocks still use XML formatting (via XMLBlockFormatter).
Only conversation messages are simplified.
"""

import logging
from typing import TYPE_CHECKING, Any

from nooa.context_blocks import (
    RenderedMessage,
    ResolvedBlock,
    ToolCallEvent,
    ToolCallInfo,
)
from nooa.context_blocks.formatter import XMLBlockFormatter
from nooa.context_blocks.models import Role
from nooa.context_blocks.scoped import ScopedContext
from nooa.context_blocks.utils import truncating_pformat
from nooa.events import (
    Error,
    Feedback,
    LLMOutput,
    Message,
    PythonOutput,
    Reasoning,
    Task,
)
from nooa.runtime.event_query import EventQuery
from nooa.strategies.codeact import CodeActStrategy

if TYPE_CHECKING:
    from nooa.config.truncation_config import FormatConfig
    from nooa.strategies.base import RuntimeServices
    from nooa.strategies.current_call import CurrentCall

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plain-text event rendering
# ---------------------------------------------------------------------------


def plain_event_content(
    event: Any,
    event_format: "FormatConfig | None" = None,
) -> str:
    """Render an event as plain text — no type wrappers, no metadata.

    Extracts human-readable content from event objects instead of using
    `pformat(event)` which produces `Task(prompt='...')` style output.

    Args:
        event:     An event object (Task, Error, PythonOutput, etc.)
        event_format: Structural bounds (max_string / max_length / max_depth)
                   for nested values within event fields. Without these,
                   pformat's structured-instance fallback caps nested strings
                   at 150 chars and the LLM sees `str(len=N, [:75]=..., [-75:]=...)`
                   instead of the actual content.

    Returns:
        Plain text content suitable for LLM consumption.
    """
    # Task — use the prompt directly
    if isinstance(event, Task):
        return event.prompt

    # PythonOutput — format stdout/value/error cleanly
    if isinstance(event, PythonOutput):
        parts: list[str] = []
        if event.stdout:
            parts.append(event.stdout)
        if event.error:
            parts.append(f"Error: {event.error}")
        if event.stderr and not event.error:
            parts.append(f"Stderr: {event.stderr}")
        if event.value is not None:
            # truncating_pformat handles non-strings; strings pass verbatim.
            # event_format provides max_string/max_length/max_depth so nested
            # strings within structured values are bounded by cfg.event_format
            # rather than pformat's hidden 150-char fallback.
            fc_kwargs = event_format.model_dump() if event_format is not None else {}
            value_str = truncating_pformat(event.value, **fc_kwargs)
            parts.append(f"Out[{event.execution_count}]: {value_str}")
        return "\n".join(parts) if parts else "(no output)"

    # Error, Message, Reasoning, LLMOutput, Feedback — use content directly
    if isinstance(event, (Error, Message, Reasoning, LLMOutput, Feedback)):
        return event.content

    # Fallback
    return str(event)


# ---------------------------------------------------------------------------
# PlainCodeActBlockFormatter — handles all three rendering changes
# ---------------------------------------------------------------------------


class PlainCodeActBlockFormatter(XMLBlockFormatter):  # type: ignore[misc]  # untyped base class from nooa.context_blocks
    """BlockFormatter that emits clean messages without XML wrappers or type wrappers.

    Handles three changes compared to XMLBlockFormatter + OpenAIProviderFormatter:

    1. Strips XML wrapping from message content (uses ``plain_event_content``)
    2. Removes Python type wrappers (``Task(...)`` → just the prompt text)
    3. Merges PythonOutput into tool response (inline tool results)

    System prompt blocks are not affected — they still use XML formatting
    (inherited from XMLBlockFormatter).

    Args:
        event_format: Structural bounds for nested values inside event fields
            (max_string / max_length / max_depth). Set from
            ``TruncationConfig.event_format`` by CodeActLiteStrategy.
    """

    def __init__(
        self,
        event_format: "FormatConfig | None" = None,
    ):
        self._event_format = event_format  # set from tc.event_format by CodeActLiteStrategy

    def format_event(
        self,
        event: Any,
        event_format: "FormatConfig | None" = None,
    ) -> str:
        return plain_event_content(event, event_format=event_format or self._event_format)

    def _content_for_block(self, block: ResolvedBlock) -> str:
        if block.content:
            return block.content
        if block.event is not None:
            return plain_event_content(block.event, event_format=self._event_format)
        return ""

    def format(self, blocks: list[ResolvedBlock]) -> list[RenderedMessage]:
        # System blocks: reuse XML wrapping from the base class by calling it on
        # the SYSTEM-role blocks only. That gives us a list with a single
        # RenderedMessage(SYSTEM, xml_content) — we keep that and replace the
        # per-event messages with our custom merging logic below.
        system_blocks = [b for b in blocks if b.role == Role.SYSTEM]
        message_blocks = [b for b in blocks if b.role != Role.SYSTEM]

        messages: list[RenderedMessage] = []
        if system_blocks:
            # super().format() returns a list; the first entry is the SYSTEM message.
            base = super().format(system_blocks)
            # Take the SYSTEM entry only (there are no event blocks in ``system_blocks``).
            messages.extend(m for m in base if m.role == Role.SYSTEM)

        # Index PythonOutput blocks by tool_call_id for merging.
        python_outputs: dict[str, ResolvedBlock] = {}
        for block in message_blocks:
            if isinstance(block.event, PythonOutput):
                python_outputs[block.event.tool_call_id] = block

        for block in message_blocks:
            if block.role == Role.RUNTIME_EVENT:
                continue

            if isinstance(block.event, ToolCallEvent):
                event = block.event
                messages.append(
                    RenderedMessage(
                        role=Role.ASSISTANT,
                        tool_call=ToolCallInfo(
                            id=event.tool_call_id,
                            name=event.name,
                            arguments=event.arguments,
                        ),
                    )
                )
                # Tool result: merge PythonOutput content if available.
                py_out_block = python_outputs.get(event.tool_call_id)
                if py_out_block and py_out_block.event:
                    content = self._content_for_block(py_out_block)
                elif event.result is not None:
                    content = event.result.content
                else:
                    content = ""
                messages.append(
                    RenderedMessage(
                        role=Role.TOOL,
                        content=content,
                        tool_call_id=event.tool_call_id,
                    )
                )

            elif isinstance(block.event, PythonOutput):
                # Already merged into tool result above.
                assert block.event.tool_call_id in python_outputs, (
                    "PythonOutput not in python_outputs index — indexing loop above should capture all"
                )
                continue

            else:
                content = self._content_for_block(block)
                messages.append(RenderedMessage(role=block.role, content=content))

        return messages


# Back-compat alias — code outside this file may still import the old name.
PlainProviderFormatter = PlainCodeActBlockFormatter


# ---------------------------------------------------------------------------
# CodeActLiteStrategy
# ---------------------------------------------------------------------------


class CodeActLiteStrategy(CodeActStrategy):
    """Simplified CodeAct strategy with clean message rendering.

    Extends CodeActStrategy with three changes:
    1. Events scoped to current call only (ScopedContext + EventQuery.current_call())
    2. Messages rendered as plain text (no XML tags, no Python type wrappers)
    3. Tool results inlined (PythonOutput merged into tool response)

    Usage:
        from nooa.strategies.codeact_lite import CodeActLiteStrategy

        @strategy(CodeActLiteStrategy())
        async def my_method(self, x: str) -> str:
            '''Classify x.'''
            ...
    """

    @property
    def name(self) -> str:
        """Strategy name."""
        return "CODEACT_LITE"

    async def execute(self, runtime: "RuntimeServices", call: "CurrentCall") -> Any:
        """Execute with scoped events and plain message formatting.

        Wraps the parent CodeActStrategy.execute() with:
        1. ScopedContext to limit events to current call
        2. Temporary formatter swap to PlainProviderFormatter

        Note: We use the explicit call_id from the agent call stack rather than
        EventQuery.current_call() because CodeActStrategy.execute() mutates
        call.id to a tag number, which breaks the "current" resolution.
        """
        from nooa.runtime.context_vars import _get_agent_call_stack

        # Capture the call_id from the agent call stack BEFORE super().execute()
        # mutates call.id. Events are tagged with this value via the call stack,
        # so the EventQuery must match it explicitly.
        stack = _get_agent_call_stack()
        call_id = stack[-1] if stack else call.id

        # Swap the agent's render_config to use our plain formatter, threading
        # the block char limit from TruncationConfig. In the new pipeline this
        # is a BlockFormatter swap (semantic decisions live there); the stock
        # OpenAIProviderFormatter handles provider-shape.
        original_render_config = runtime.agent.render_config
        tc = runtime.agent._truncation
        runtime.agent.render_config = original_render_config.model_copy(
            update={
                "block_formatter": PlainCodeActBlockFormatter(
                    event_format=tc.event_format,
                )
            }
        )

        try:
            # Scope events to current call using the explicit call_id
            with ScopedContext(events=EventQuery(call_id=call_id)):
                return await super().execute(runtime, call)
        finally:
            # Restore original render config
            runtime.agent.render_config = original_render_config
