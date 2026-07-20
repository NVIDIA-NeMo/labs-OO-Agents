# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for render_context().

render_context() takes pre-resolved blocks (list[ResolvedBlock]) and renders
them via BlockFormatter + ProviderFormatter. No eval function, no expression
evaluation, no class — just a pure function.
"""

import pytest

from nooa.context_blocks.formatter import (
    AnthropicProviderFormatter,
    MarkdownBlockFormatter,
    OpenAIProviderFormatter,
    XMLBlockFormatter,
)
from nooa.context_blocks.models import BlockMetadata, ResolvedBlock, Role
from nooa.context_blocks.renderer import render_context


class TestRenderContextBasic:
    """Basic rendering tests."""

    def test_render_empty_blocks(self):
        """render_context() with empty list returns no messages.

        The old renderer always emitted an empty system message; the new
        pipeline emits nothing when there are no blocks. Downstream code that
        wants a guaranteed system slot should pass at least one block.
        """
        result = render_context(
            [],
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        ).output

        assert result == []

    def test_render_single_system_block(self):
        """render_context() formats a single system block."""
        blocks = [ResolvedBlock(key="persona", content="You are helpful.")]

        result = render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        ).output

        assert len(result) == 1
        system_content = result[0]["content"]
        assert "<persona" in system_content
        assert "You are helpful." in system_content
        assert "</persona>" in system_content

    def test_render_system_block_with_metadata(self):
        """render_context() includes ``expr=`` only for set_dynamic-sourced blocks."""
        blocks = [
            ResolvedBlock(
                key="notes",
                content="My notes",
                metadata=BlockMetadata(expr="self.context['notes']", source_dynamic=True),
            )
        ]

        result = render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        ).output

        system_content = result[0]["content"]
        assert "expr=\"self.context['notes']\"" in system_content
        assert "My notes" in system_content

    def test_render_multiple_system_blocks(self):
        """render_context() formats multiple system blocks in order."""
        blocks = [
            ResolvedBlock(key="persona", content="Be helpful."),
            ResolvedBlock(key="tools", content="search, calculate"),
        ]

        result = render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        ).output

        system_content = result[0]["content"]
        assert "<persona" in system_content
        assert "Be helpful." in system_content
        assert "<tools" in system_content
        assert "search, calculate" in system_content

    def test_render_user_message_block(self):
        """render_context() routes USER role blocks to messages."""
        blocks = [
            ResolvedBlock(key="persona", content="System prompt"),
            ResolvedBlock(key="msg1", content="Hello", role=Role.USER),
        ]

        result = render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        ).output

        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert "Hello" in result[1]["content"]

    def test_render_assistant_message_block(self):
        """render_context() routes ASSISTANT role blocks to messages."""
        blocks = [
            ResolvedBlock(key="msg1", content="Hi", role=Role.USER),
            ResolvedBlock(key="msg2", content="Hello!", role=Role.ASSISTANT),
        ]

        result = render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        ).output

        assert len(result) == 3  # system + user + assistant
        assert result[2]["role"] == "assistant"
        assert "Hello!" in result[2]["content"]


class TestRenderContextTruncation:
    """Truncation tests."""

    def test_block_limit_no_longer_truncates(self):
        """Block-level head/tail truncation has been removed; ``block_limit``
        is preserved in the API but no longer head/tail-squashes content.
        Per-value bounds come from cfg.context_blocks at render time;
        whole-block eviction (L4) handles overflow at assembly."""
        long_content = "A" * 1000
        blocks = [ResolvedBlock(key="data", content=long_content)]

        result = render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        ).output

        system_content = result[0]["content"]
        # Long content survives intact
        assert long_content in system_content
        # No legacy wrapper
        assert "output too large" not in system_content.lower()
        assert "<truncated-output>" not in system_content

    def test_no_truncation_without_limit(self):
        """render_context() does not truncate when no limit is set."""
        long_content = "A" * 1000
        blocks = [ResolvedBlock(key="data", content=long_content)]

        result = render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        ).output

        system_content = result[0]["content"]
        assert "A" * 1000 in system_content


class TestRenderContextMarkdown:
    """Markdown formatter tests."""

    def test_markdown_formatting(self):
        """render_context() works with MarkdownBlockFormatter."""
        blocks = [
            ResolvedBlock(
                key="instructions",
                content="Follow these rules.",
                metadata=BlockMetadata(expr="self.instructions"),
            )
        ]

        result = render_context(
            blocks,
            block_formatter=MarkdownBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        ).output

        system_content = result[0]["content"]
        assert "# Instructions" in system_content
        assert "Follow these rules." in system_content


class TestRenderContextAnthropic:
    """Anthropic provider formatter tests."""

    def test_anthropic_format(self):
        """render_context() produces Anthropic-style output."""
        blocks = [
            ResolvedBlock(key="persona", content="Be helpful."),
            ResolvedBlock(key="msg", content="Hello", role=Role.USER),
        ]

        result = render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=AnthropicProviderFormatter(),
        ).output

        assert isinstance(result, dict)
        assert "system" in result
        assert "messages" in result
        assert "Be helpful." in result["system"]
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"


class TestRenderContextToolCalls:
    """Tool call message rendering tests."""

    def test_tool_call_block(self):
        """render_context() handles ToolCallEvent on block.event correctly."""
        from nooa.context_blocks.events import ToolCallEvent, ToolResult

        event = ToolCallEvent(
            tool_call_id="call_123",
            name="search",
            arguments={"query": "test"},
            result=ToolResult(tool_call_id="call_123", content="Found it"),
        )
        blocks = [
            ResolvedBlock(
                key="tool1",
                content="",
                role=Role.ASSISTANT,
                event=event,
            ),
        ]

        result = render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        ).output

        # System + assistant (tool_calls) + tool (result)
        assert len(result) == 3
        assert result[1]["role"] == "assistant"
        assert "tool_calls" in result[1]
        assert result[1]["tool_calls"][0]["function"]["name"] == "search"
        assert result[2]["role"] == "tool"
        assert result[2]["content"] == "Found it"


class TestRenderContextNoMutation:
    """render_context() must never mutate input blocks."""

    def test_system_block_truncation_preserves_original(self):
        """Truncating system blocks must not modify the original ResolvedBlock."""
        original_content = "A" * 1000
        block = ResolvedBlock(
            key="data", content=original_content, metadata=BlockMetadata(expr="test")
        )

        render_context(
            [block],
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        )

        # Original block must be unchanged
        assert block.content == original_content
        assert block.metadata.truncated is False

    def test_event_block_truncation_preserves_original(self):
        """Truncating message blocks must not modify the original ResolvedBlock."""
        original_content = "B" * 1000
        block = ResolvedBlock(
            key="event_1",
            content=original_content,
            role=Role.USER,
            metadata=BlockMetadata(expr="self.events['1']", tag="1"),
        )

        render_context(
            [block],
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        )

        # Original block must be unchanged
        assert block.content == original_content
        assert block.metadata.truncated is False

    def test_render_twice_produces_same_result(self):
        """Calling render_context() twice must produce identical results (no state leakage)."""
        long_content = "C" * 1000
        blocks = [
            ResolvedBlock(key="data", content=long_content),
            ResolvedBlock(key="msg", content=long_content, role=Role.USER),
        ]

        result1 = render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        ).output

        result2 = render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        ).output

        assert result1 == result2


class TestFormatMessageContent:
    """Tests for format_message_content() — wraps message blocks with metadata."""

    def test_xml_with_expr(self):
        """``expr=`` is only rendered for source_dynamic blocks (set_dynamic).

        USER-role events wrap in ``<sys>``; assistant events pass through bare.
        """
        from nooa.context_blocks.renderer import format_message_content

        dynamic = ResolvedBlock(
            key="msg",
            content="Hello",
            role=Role.USER,
            metadata=BlockMetadata(expr="self.events['1']", source_dynamic=True),
        )
        result = format_message_content(dynamic, "xml")
        assert "<sys expr=\"self.events['1']\">" in result
        assert "Hello" in result
        assert "</sys>" in result

        # Non-dynamic block: expr is suppressed.
        static = ResolvedBlock(
            key="msg",
            content="Hello",
            role=Role.USER,
            metadata=BlockMetadata(expr="self.events['1']"),
        )
        assert "expr=" not in format_message_content(static, "xml")

    def test_assistant_events_render_bare(self):
        """Assistant events pass through unwrapped — no ``<agent>`` tag."""
        from nooa.context_blocks.renderer import format_message_content

        block = ResolvedBlock(
            key="msg",
            content="Hi",
            role=Role.ASSISTANT,
            metadata=BlockMetadata(tag="42"),
        )
        result = format_message_content(block, "xml")
        assert result == "Hi"
        assert "<agent" not in result

    def test_xml_with_expr_and_tag(self):
        from nooa.context_blocks.renderer import format_message_content

        block = ResolvedBlock(
            key="msg",
            content="content",
            role=Role.USER,
            metadata=BlockMetadata(expr="self.events['1']", tag="1", source_dynamic=True),
        )
        result = format_message_content(block, "xml")
        assert "expr=\"self.events['1']\"" in result
        assert 'tag="1"' in result

    def test_markdown_with_expr(self):
        from nooa.context_blocks.renderer import format_message_content

        block = ResolvedBlock(
            key="msg",
            content="Hello",
            role=Role.USER,
            metadata=BlockMetadata(expr="self.events['1']", source_dynamic=True),
        )
        result = format_message_content(block, "markdown")
        assert "### User Message" in result
        assert '"expr": "self.events[\'1\']"' in result
        assert "Hello" in result

    def test_markdown_with_tag(self):
        from nooa.context_blocks.renderer import format_message_content

        block = ResolvedBlock(
            key="msg",
            content="Hi",
            role=Role.ASSISTANT,
            metadata=BlockMetadata(tag="42"),
        )
        result = format_message_content(block, "markdown")
        assert "### Assistant Message" in result
        assert '"tag": "42"' in result

    def test_no_metadata_produces_no_wrapping(self):
        from nooa.context_blocks.renderer import format_message_content

        block = ResolvedBlock(
            key="msg",
            content="Hello",
            role=Role.USER,
            metadata=BlockMetadata(),
        )
        result = format_message_content(block, "xml")
        # No expr or tag, so no attributes — but still wrapped.
        # USER-role events wrap in <sys>.
        assert "<sys>" in result
        assert "Hello" in result


class TestFormatMessageContentPlain:
    """Tests for plain format_type in format_message_content()."""

    def test_plain_regular_message_with_tag_uses_role_tag(self):
        """Plain format wraps regular messages in <role_message tag="N">."""
        from nooa.context_blocks.renderer import format_message_content

        block = ResolvedBlock(
            key="msg",
            content="Hello world",
            role=Role.USER,
            metadata=BlockMetadata(tag="1"),
        )
        result = format_message_content(block, "plain")

        assert '<user_message tag="1">' in result
        assert "Hello world" in result
        assert "</user_message>" in result
        assert "expr" not in result

    def test_plain_assistant_message_with_tag(self):
        """Plain format wraps assistant messages in <assistant_message tag="N">."""
        from nooa.context_blocks.renderer import format_message_content

        block = ResolvedBlock(
            key="msg",
            content="result = 42",
            role=Role.ASSISTANT,
            metadata=BlockMetadata(tag="3"),
        )
        result = format_message_content(block, "plain")

        assert '<assistant_message tag="3">' in result
        assert "result = 42" in result
        assert "</assistant_message>" in result

    def test_plain_without_tag_or_event_returns_content_unchanged(self):
        """Plain format returns content as-is when no tag and no event."""
        from nooa.context_blocks.renderer import format_message_content

        block = ResolvedBlock(
            key="msg",
            content="Just a message",
            role=Role.USER,
            metadata=BlockMetadata(expr="self.events['1']"),
        )
        result = format_message_content(block, "plain")

        assert result == "Just a message"

    def test_plain_event_block_uses_event_type_as_outer_tag(self):
        """Plain format uses snake_case event class name as outer tag for event blocks."""
        from nooa.context_blocks.events import UserEvent
        from nooa.context_blocks.renderer import format_message_content

        event = UserEvent(content="Hello there", tag="2")
        block = ResolvedBlock(
            key="event_1",
            content="Hello there",  # already serialized by format_event
            role=Role.USER,
            metadata=BlockMetadata(tag="2"),
            event=event,
        )
        result = format_message_content(block, "plain")

        assert '<user_event tag="2">' in result
        assert "Hello there" in result
        assert "</user_event>" in result
        assert "expr" not in result

    def test_plain_event_block_no_tag(self):
        """Plain format event block without tag: outer tag has no tag attr."""
        from nooa.context_blocks.events import UserEvent
        from nooa.context_blocks.renderer import format_message_content

        event = UserEvent(content="Hello", tag="x")
        block = ResolvedBlock(
            key="event_1",
            content="Hello",
            role=Role.USER,
            metadata=BlockMetadata(expr="self.events['1']"),
            event=event,
        )
        result = format_message_content(block, "plain")

        assert "<user_event>" in result
        assert "tag=" not in result
        assert "Hello" in result

    def test_plain_ignores_expr_attribute(self):
        """Plain format never includes expr= in output even if block has expr metadata."""
        from nooa.context_blocks.renderer import format_message_content

        block = ResolvedBlock(
            key="msg",
            content="Hi",
            role=Role.ASSISTANT,
            metadata=BlockMetadata(expr="self.messages[-1]", tag="5"),
        )
        result = format_message_content(block, "plain")

        assert "expr" not in result
        assert 'tag="5"' in result


class TestRenderContextEventSerialization:
    """Tests for the event serialization step in render_context()."""

    def test_non_tool_event_content_serialized_before_output(self):
        """render_context() must serialize event content via block_formatter.format_event()."""
        from nooa.context_blocks.events import UserEvent

        event = UserEvent(content="Hello world", tag="1")
        block = ResolvedBlock(
            key="event_1",
            content="",  # deferred — as produced by _phase_events now
            role=Role.USER,
            metadata=BlockMetadata(expr="self.events['1']", tag="1"),
            event=event,
        )
        result = render_context(
            [block],
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        ).output

        # Content must appear in the user message
        user_msg = result[1]
        assert user_msg["role"] == "user"
        assert "Hello world" in user_msg["content"]

    def test_event_serialization_happens_at_render_time(self):
        """Events are serialized at render time so block-level passes see real content.
        After block-level truncation removal, ``block_limit`` is no longer a hard
        cap — content passes through; bounds come from per-value annotations and
        cfg.events.* knobs (here the raw string in UserEvent.content gets the
        marker-family treatment via pformat with default max_string=10K, so a
        200-char string remains untouched)."""
        from nooa.context_blocks.events import UserEvent

        long_content = "A" * 200
        event = UserEvent(content=long_content, tag="1")
        block = ResolvedBlock(
            key="event_1",
            content="",
            role=Role.USER,
            metadata=BlockMetadata(tag="1"),
            event=event,
        )

        result = render_context(
            [block],
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        ).output

        user_msg = result[1]
        # Content was serialized (not empty)
        assert "UserEvent" in user_msg["content"] or long_content in user_msg["content"]
        # No legacy wrapper
        assert "output too large" not in user_msg["content"].lower()

    def test_tool_call_event_not_passed_to_format_event(self):
        """ToolCallEvent blocks must not be passed through format_event — ProviderFormatter handles them."""
        from nooa.context_blocks.events import ToolCallEvent, ToolResult

        event = ToolCallEvent(
            tool_call_id="tc_1",
            name="execute_python",
            arguments={"code": "print(1)"},
            result=ToolResult(tool_call_id="tc_1", content="status: complete"),
        )
        block = ResolvedBlock(
            key="event_1",
            content="",
            role=Role.ASSISTANT,
            event=event,
        )
        result = render_context(
            [block],
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        ).output

        # Tool call should appear as assistant tool_calls message, not as a user message
        msgs = result[1:]
        assert any(m["role"] == "assistant" and "tool_calls" in m for m in msgs)

    def test_block_with_no_event_preserves_existing_content(self):
        """Blocks with event=None must pass through the serialization step unchanged."""
        block = ResolvedBlock(
            key="msg",
            content="pre-existing content",
            role=Role.USER,
            event=None,
        )
        result = render_context(
            [block],
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        ).output

        user_msg = result[1]
        assert user_msg["role"] == "user"
        assert "pre-existing content" in user_msg["content"]


class TestCountTokens:
    """Tests for count_tokens parameter in render_context()."""

    def test_raises_if_context_limit_set_without_counter(self):
        """ValueError when context_limit set but no count_tokens."""
        with pytest.raises(ValueError, match="max_context_tokens requires a token counter"):
            render_context(
                [],
                block_formatter=XMLBlockFormatter(),
                provider_formatter=OpenAIProviderFormatter(),
                context_limit=10_000,
                count_tokens=None,
            )

    def test_accepts_none_counter_when_no_token_limits(self):
        """No error when context_limit is not set."""
        result = render_context(
            [],
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
            count_tokens=None,
        ).output
        assert result is not None

    def test_uses_count_tokens_for_context_limit(self):
        """count_tokens is called when context_limit is set."""
        call_count = []

        def counter(s: str) -> int:
            call_count.append(1)
            return len(s) // 4

        blocks = [ResolvedBlock(key="sys", content="Hello world")]
        render_context(
            blocks,
            block_formatter=XMLBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
            context_limit=10_000,
            count_tokens=counter,
        )
        assert len(call_count) > 0
