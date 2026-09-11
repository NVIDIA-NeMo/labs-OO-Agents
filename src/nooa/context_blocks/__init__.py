# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context blocks: Pre-resolved context block rendering for LLM applications.

Features:
- Dynamic: Marker for blocks that are re-evaluated each LLM turn
- ResolvedBlock: Pre-resolved block ready for rendering (no eval needed)
- render_context: Dead simple render function (partition by role, format, truncate)
- Pluggable formatters: XML, Markdown for blocks; OpenAI, Anthropic for providers
- Typed event models for conversation history
"""

from nooa.context_blocks.events import (
    AssistantEvent,
    Event,
    EventBase,
    EventStatus,
    Metadata,
    ResultStatus,
    ToolCallEvent,
    ToolResult,
    UserEvent,
)
from nooa.context_blocks.exceptions import (
    BlockError,
    BlockSyntaxError,
    DynamicNotResolvedError,
    ProtectedBlockError,
)
from nooa.context_blocks.formatter import (
    FORMAT_MARKDOWN,
    FORMAT_XML,
    AnthropicProviderFormatter,
    BlockFormatter,
    FormatType,
    MarkdownBlockFormatter,
    OpenAIProviderFormatter,
    ProviderFormatter,
    ResponsesProviderFormatter,
    XMLBlockFormatter,
)
from nooa.context_blocks.models import (
    BlockMetadata,
    Context,
    ContextWindowStats,
    DynamicContext,
    RenderedMessage,
    ResolvedBlock,
    Role,
    ToolCallInfo,
)
from nooa.context_blocks.render_config import RenderConfig
from nooa.context_blocks.renderer import (
    RenderResult,
    format_message_content,
    render_context,
)
from nooa.context_blocks.scoped import ScopedContext

__all__ = [
    # Core types
    "Context",
    "DynamicContext",
    "ResolvedBlock",
    "Role",
    "BlockMetadata",
    "ContextWindowStats",
    "RenderedMessage",
    "ToolCallInfo",
    # Renderer
    "render_context",
    "RenderResult",
    "format_message_content",
    # Format type enum and constants
    "FormatType",
    "FORMAT_XML",
    "FORMAT_MARKDOWN",
    # BlockFormatter (system prompt blocks -> string)
    "BlockFormatter",
    "XMLBlockFormatter",
    "MarkdownBlockFormatter",
    # ProviderFormatter (system prompt + messages -> provider output)
    "ProviderFormatter",
    "OpenAIProviderFormatter",
    "AnthropicProviderFormatter",
    "ResponsesProviderFormatter",
    # Event base, metadata extension point, and union
    "EventBase",
    "Metadata",
    "Event",
    # Status enums
    "EventStatus",
    "ResultStatus",
    # Typed events
    "UserEvent",
    "AssistantEvent",
    "ToolCallEvent",
    "ToolResult",
    # Render config
    "RenderConfig",
    # Scoped overrides
    "ScopedContext",
    # Exceptions
    "BlockError",
    "BlockSyntaxError",
    "DynamicNotResolvedError",
    "ProtectedBlockError",
]
