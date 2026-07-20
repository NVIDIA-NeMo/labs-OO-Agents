# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for context_blocks formatters.

BlockFormatter.format() takes ``list[ResolvedBlock]`` (both SYSTEM and event
blocks) and returns ``list[RenderedMessage]``. ProviderFormatter.format() takes
``list[RenderedMessage]`` and returns provider-specific wire format.
"""

import pytest

from nooa.context_blocks.events import ToolCallEvent, ToolResult
from nooa.context_blocks.formatter import (
    AnthropicProviderFormatter,
    MarkdownBlockFormatter,
    OpenAIProviderFormatter,
    ResponsesProviderFormatter,
    XMLBlockFormatter,
)
from nooa.context_blocks.models import (
    BlockMetadata,
    RenderedMessage,
    ResolvedBlock,
    Role,
    ToolCallInfo,
)


def _tool_call_block(
    *,
    key: str = "tc",
    tool_call_id: str,
    name: str,
    arguments: dict,
    result_content: str | None = None,
) -> ResolvedBlock:
    """Helper: ResolvedBlock carrying a ToolCallEvent."""
    result = (
        ToolResult(tool_call_id=tool_call_id, content=result_content)
        if result_content is not None
        else None
    )
    event = ToolCallEvent(tool_call_id=tool_call_id, name=name, arguments=arguments, result=result)
    return ResolvedBlock(key=key, content="", role=Role.ASSISTANT, event=event)


def _system_content(messages: list[RenderedMessage]) -> str:
    """Return the concatenated SYSTEM message content (empty string if none)."""
    return "\n\n".join(m.content or "" for m in messages if m.role == Role.SYSTEM)


class TestBlockFormatterABC:
    def test_is_abstract(self):
        from nooa.context_blocks.formatter import BlockFormatter

        with pytest.raises(TypeError):
            BlockFormatter()  # type: ignore[abstract]

    def test_requires_format_method(self):
        from nooa.context_blocks.formatter import BlockFormatter

        class IncompleteFormatter(BlockFormatter):
            pass

        with pytest.raises(TypeError):
            IncompleteFormatter()  # type: ignore[abstract]


class TestXMLBlockFormatter:
    def test_single_block(self):
        messages = XMLBlockFormatter().format(
            [ResolvedBlock(key="persona", content="You are helpful.")]
        )
        assert len(messages) == 1
        assert messages[0].role == Role.SYSTEM
        assert "<persona>" in messages[0].content
        assert "</persona>" in messages[0].content
        assert "You are helpful." in messages[0].content

    def test_multiple_blocks(self):
        messages = XMLBlockFormatter().format(
            [
                ResolvedBlock(key="persona", content="You are helpful."),
                ResolvedBlock(key="tools", content="Available tools: search, calculate"),
            ]
        )
        content = _system_content(messages)
        assert "<persona>" in content and "<tools>" in content
        assert "You are helpful." in content and "Available tools:" in content

    def test_empty_blocks(self):
        messages = XMLBlockFormatter().format([])
        assert messages == []

    def test_preserves_content_newlines(self):
        messages = XMLBlockFormatter().format(
            [ResolvedBlock(key="content", content="Line 1\nLine 2\nLine 3")]
        )
        assert "Line 1\nLine 2\nLine 3" in messages[0].content

    def test_with_metadata_expr_renders_only_for_dynamic(self):
        # Dynamic-source block: expr is shown.
        dynamic = XMLBlockFormatter().format(
            [
                ResolvedBlock(
                    key="notes",
                    content="My notes",
                    metadata=BlockMetadata(expr="self.context['notes']", source_dynamic=True),
                )
            ]
        )
        assert "expr=\"self.context['notes']\"" in dynamic[0].content
        assert "My notes" in dynamic[0].content

        # Non-dynamic block with the same expr: expr is suppressed.
        static = XMLBlockFormatter().format(
            [
                ResolvedBlock(
                    key="notes",
                    content="My notes",
                    metadata=BlockMetadata(expr="self.context['notes']"),
                )
            ]
        )
        assert "expr=" not in static[0].content
        assert "<notes>" in static[0].content
        assert "My notes" in static[0].content

    def test_format_type(self):
        assert XMLBlockFormatter().format_type == "xml"


class TestMarkdownBlockFormatter:
    def test_single_block(self):
        messages = MarkdownBlockFormatter().format(
            [ResolvedBlock(key="persona", content="You are helpful.")]
        )
        content = _system_content(messages)
        assert "# Persona" in content
        assert "You are helpful." in content

    def test_multiple_blocks(self):
        messages = MarkdownBlockFormatter().format(
            [
                ResolvedBlock(key="persona", content="You are helpful."),
                ResolvedBlock(key="tools", content="Available tools"),
            ]
        )
        content = _system_content(messages)
        assert "# Persona" in content and "# Tools" in content

    def test_key_with_underscores(self):
        messages = MarkdownBlockFormatter().format(
            [ResolvedBlock(key="python_tools", content="Tool list")]
        )
        assert "# Python Tools" in _system_content(messages)

    def test_empty_blocks(self):
        assert MarkdownBlockFormatter().format([]) == []

    def test_with_metadata_expr_renders_only_for_dynamic(self):
        dynamic = MarkdownBlockFormatter().format(
            [
                ResolvedBlock(
                    key="notes",
                    content="My notes",
                    metadata=BlockMetadata(expr="self.context['notes']", source_dynamic=True),
                )
            ]
        )
        dyn_content = _system_content(dynamic)
        assert "# Notes" in dyn_content and '"expr"' in dyn_content and "My notes" in dyn_content

        static = MarkdownBlockFormatter().format(
            [
                ResolvedBlock(
                    key="notes",
                    content="My notes",
                    metadata=BlockMetadata(expr="self.context['notes']"),
                )
            ]
        )
        stat_content = _system_content(static)
        assert (
            "# Notes" in stat_content
            and '"expr"' not in stat_content
            and "My notes" in stat_content
        )

    def test_format_type(self):
        assert MarkdownBlockFormatter().format_type == "markdown"


class TestProviderFormatterABC:
    def test_is_abstract(self):
        from nooa.context_blocks.formatter import ProviderFormatter

        with pytest.raises(TypeError):
            ProviderFormatter()  # type: ignore[abstract]


class TestOpenAIProviderFormatter:
    def test_system_message_only(self):
        messages = [RenderedMessage(role=Role.SYSTEM, content="You are helpful.")]
        result = OpenAIProviderFormatter().format(messages)
        assert result == [{"role": "system", "content": "You are helpful."}]

    def test_user_message(self):
        messages = [
            RenderedMessage(role=Role.SYSTEM, content="System"),
            RenderedMessage(role=Role.USER, content="Hello"),
        ]
        result = OpenAIProviderFormatter().format(messages)
        assert len(result) == 2
        assert result[1] == {"role": "user", "content": "Hello"}

    def test_assistant_message(self):
        messages = [
            RenderedMessage(role=Role.SYSTEM, content="System"),
            RenderedMessage(role=Role.ASSISTANT, content="Hi there!"),
        ]
        result = OpenAIProviderFormatter().format(messages)
        assert result[1]["role"] == "assistant" and result[1]["content"] == "Hi there!"

    def test_tool_call_message(self):
        messages = [
            RenderedMessage(role=Role.SYSTEM, content="System"),
            RenderedMessage(
                role=Role.ASSISTANT,
                tool_call=ToolCallInfo(
                    id="call_abc", name="get_weather", arguments={"location": "SF"}
                ),
            ),
        ]
        result = OpenAIProviderFormatter().format(messages)
        assert len(result) == 2
        msg = result[1]
        assert msg["role"] == "assistant" and msg["content"] is None
        assert msg["tool_calls"][0]["id"] == "call_abc"
        assert msg["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_tool_call_with_result(self):
        messages = [
            RenderedMessage(role=Role.SYSTEM, content="System"),
            RenderedMessage(
                role=Role.ASSISTANT,
                tool_call=ToolCallInfo(
                    id="call_abc", name="get_weather", arguments={"location": "SF"}
                ),
            ),
            RenderedMessage(role=Role.TOOL, content="Sunny", tool_call_id="call_abc"),
        ]
        result = OpenAIProviderFormatter().format(messages)
        assert len(result) == 3
        assert result[1]["role"] == "assistant" and "tool_calls" in result[1]
        assert result[2] == {"role": "tool", "tool_call_id": "call_abc", "content": "Sunny"}

    def test_runtime_event_skipped(self):
        messages = [
            RenderedMessage(role=Role.USER, content="Hello"),
            RenderedMessage(role=Role.RUNTIME_EVENT, content="internal"),
        ]
        result = OpenAIProviderFormatter().format(messages)
        roles = [m["role"] for m in result]
        assert "runtime_event" not in roles and roles == ["user"]

    def test_metadata_skipped(self):
        messages = [
            RenderedMessage(role=Role.USER, content="Hello"),
            RenderedMessage(role=Role.METADATA, content="session-start"),
        ]
        result = OpenAIProviderFormatter().format(messages)
        assert [m["role"] for m in result] == ["user"]


class TestAnthropicProviderFormatter:
    def test_returns_dict_with_system_and_messages(self):
        messages = [RenderedMessage(role=Role.SYSTEM, content="You are helpful.")]
        result = AnthropicProviderFormatter().format(messages)
        assert result == {"system": "You are helpful.", "messages": []}

    def test_user_message(self):
        messages = [
            RenderedMessage(role=Role.SYSTEM, content="System"),
            RenderedMessage(role=Role.USER, content="Hello"),
        ]
        result = AnthropicProviderFormatter().format(messages)
        assert result["system"] == "System"
        assert result["messages"] == [{"role": "user", "content": "Hello"}]

    def test_assistant_message(self):
        messages = [
            RenderedMessage(role=Role.SYSTEM, content="System"),
            RenderedMessage(role=Role.ASSISTANT, content="Hi!"),
        ]
        result = AnthropicProviderFormatter().format(messages)
        assert result["messages"] == [{"role": "assistant", "content": "Hi!"}]

    def test_tool_call_message(self):
        messages = [
            RenderedMessage(role=Role.SYSTEM, content="System"),
            RenderedMessage(
                role=Role.ASSISTANT,
                tool_call=ToolCallInfo(id="tc_1", name="search", arguments={"q": "test"}),
            ),
        ]
        result = AnthropicProviderFormatter().format(messages)
        msg = result["messages"][0]
        assert msg["role"] == "assistant"
        assert msg["content"][0]["type"] == "tool_use" and msg["content"][0]["id"] == "tc_1"

    def test_tool_call_with_result(self):
        messages = [
            RenderedMessage(role=Role.SYSTEM, content="System"),
            RenderedMessage(
                role=Role.ASSISTANT,
                tool_call=ToolCallInfo(id="tc_1", name="search", arguments={"q": "test"}),
            ),
            RenderedMessage(role=Role.TOOL, content="Result", tool_call_id="tc_1"),
        ]
        result = AnthropicProviderFormatter().format(messages)
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "assistant"
        assert result["messages"][1]["role"] == "user"
        assert result["messages"][1]["content"][0]["type"] == "tool_result"

    def test_tool_role_mapped_to_user(self):
        """TOOL role without tool_call_id falls back to user (matches old behavior)."""
        messages = [
            RenderedMessage(role=Role.SYSTEM, content="System"),
            RenderedMessage(role=Role.TOOL, content="Tool output"),
        ]
        result = AnthropicProviderFormatter().format(messages)
        assert result["messages"][0]["role"] == "user"

    def test_metadata_skipped(self):
        messages = [
            RenderedMessage(role=Role.USER, content="Hello"),
            RenderedMessage(role=Role.METADATA, content="session-start"),
        ]
        result = AnthropicProviderFormatter().format(messages)
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"


class TestEndToEndPipelines:
    """Compose BlockFormatter + ProviderFormatter through the neutral type."""

    def test_xml_with_openai(self):
        blocks = [
            ResolvedBlock(key="persona", content="You are helpful."),
            ResolvedBlock(key="tools", content="search, calculate"),
            ResolvedBlock(key="msg", content="Hello", role=Role.USER),
        ]
        messages = XMLBlockFormatter().format(blocks)
        result = OpenAIProviderFormatter().format(messages)
        assert len(result) == 2
        assert "<persona>" in result[0]["content"]
        assert "Hello" in result[1]["content"]

    def test_markdown_with_anthropic(self):
        blocks = [
            ResolvedBlock(key="persona", content="You are helpful."),
            ResolvedBlock(key="msg", content="Hello", role=Role.USER),
        ]
        messages = MarkdownBlockFormatter().format(blocks)
        result = AnthropicProviderFormatter().format(messages)
        assert "# Persona" in result["system"]
        assert result["messages"][0]["content"] == "Hello"


class TestBlockFormatterFormatEvent:
    """format_event() still serializes raw events to content strings."""

    def test_xml_format_event_user_event(self):
        from nooa.context_blocks.events import UserEvent

        event = UserEvent(content="Hello world", tag="1")
        assert "Hello world" in XMLBlockFormatter().format_event(event)

    def test_markdown_format_event_user_event(self):
        from nooa.context_blocks.events import UserEvent

        event = UserEvent(content="Hello world", tag="1")
        assert "Hello world" in MarkdownBlockFormatter().format_event(event)

    def test_format_event_has_default(self):
        """BlockFormatter.format_event() is concrete by default (agentdoc pformat)."""
        from nooa.context_blocks.events import UserEvent
        from nooa.context_blocks.formatter import BlockFormatter, FormatType

        class MinimalFormatter(BlockFormatter):
            @property
            def format_type(self):
                return FormatType.XML

            def format(self, blocks):
                return []

        event = UserEvent(content="Hello world", tag="1")
        assert "Hello world" in MinimalFormatter().format_event(event)


class TestResponsesProviderFormatterImages:
    """Responses API image blocks: image_url must be a URL STRING, not the
    Chat-Completions {"url": ...} object (regression — the object shape makes the
    API reject the request with 'expected an image URL, but got an object')."""

    def _image_message(self, image_url):
        img = {"type": "image_url", "image_url": image_url}
        return [RenderedMessage(role=Role.USER, content="the grid", images=[img])]

    def test_image_url_object_becomes_input_image_string(self):
        msgs = self._image_message({"url": "data:image/png;base64,AAAA", "detail": "high"})
        out = ResponsesProviderFormatter().format(msgs)
        parts = out[-1]["content"]
        assert parts[0] == {"type": "input_text", "text": "the grid"}
        img = parts[1]
        assert img["type"] == "input_image"
        assert isinstance(img["image_url"], str), img
        assert img["image_url"] == "data:image/png;base64,AAAA"
        assert img.get("detail") == "high"

    def test_image_url_already_string_passthrough(self):
        out = ResponsesProviderFormatter().format(self._image_message("data:image/png;base64,BBBB"))
        img = out[-1]["content"][1]
        assert img == {"type": "input_image", "image_url": "data:image/png;base64,BBBB"}

    def test_image_url_dict_without_url_raises(self):
        # Fail fast instead of emitting an empty image_url the API rejects opaquely.
        with pytest.raises(ValueError, match="no 'url'"):
            ResponsesProviderFormatter().format(self._image_message({"detail": "high"}))
