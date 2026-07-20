# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the cached renderer (static-prefix / events / dynamic-suffix)."""

from nooa.context_blocks.events import (
    AssistantEvent,
    ToolCallEvent,
    ToolResult,
    UserEvent,
)
from nooa.context_blocks.formatter import (
    AnthropicProviderFormatter,
    OpenAIProviderFormatter,
)
from nooa.context_blocks.models import BlockMetadata, DynamicContext, ResolvedBlock, Role
from nooa.context_blocks.renderer import render_context
from nooa.context_blocks.renderers.cached import CachedBlockFormatter


def _static_block(key: str, content: str, expr: str | None = None) -> ResolvedBlock:
    return ResolvedBlock(
        key=key,
        content=content,
        role=Role.SYSTEM,
        metadata=BlockMetadata(expr=expr, static=True),
    )


def _dynamic_block(key: str, content: str, expr: str | None = None) -> ResolvedBlock:
    return ResolvedBlock(
        key=key,
        content=content,
        role=Role.SYSTEM,
        metadata=BlockMetadata(expr=expr, static=False, user_block=True),
    )


class TestImmutableMetadata:
    def test_block_metadata_has_static_field(self):
        meta = BlockMetadata(static=True)
        assert meta.static is True

    def test_block_metadata_static_defaults_false(self):
        assert BlockMetadata().static is False

    def test_dynamic_context_has_no_static_field(self):
        """DynamicContext no longer carries a static flag — partition is determined by ContextManager."""
        dc = DynamicContext("doc(self)")
        assert dc.expr == "doc(self)"
        assert not hasattr(dc, "static")

    def test_dynamic_context_repr(self):
        assert repr(DynamicContext("x")) == "DynamicContext('x')"


class TestCachedBlockFormatterPartition:
    def test_all_static_single_system_message(self):
        fmt = CachedBlockFormatter()
        messages = fmt.format([_static_block("a", "A"), _static_block("b", "B")])
        assert len(messages) == 1
        assert messages[0].role == Role.SYSTEM
        assert "<a>" in messages[0].content and "<b>" in messages[0].content

    def test_all_volatile_falls_back_to_system(self):
        """When no blocks are static, all go to SYSTEM (XMLBlockFormatter-compatible)."""
        fmt = CachedBlockFormatter()
        messages = fmt.format([_dynamic_block("plan", "do stuff")])
        assert len(messages) == 1
        assert messages[0].role == Role.SYSTEM
        assert "<plan>" in messages[0].content
        assert "</plan>" in messages[0].content

    def test_mixed_preserves_order_within_halves(self):
        fmt = CachedBlockFormatter()
        messages = fmt.format(
            [
                _static_block("sys", "S"),
                _dynamic_block("plan", "P"),
                _static_block("self_doc", "D"),
                _dynamic_block("state", "T"),
            ]
        )
        assert len(messages) == 2
        sys_msg = messages[0]
        assert sys_msg.index("<sys>") < sys_msg.index("<self_doc>") if False else True  # sanity
        assert sys_msg.content.index("<sys>") < sys_msg.content.index("<self_doc>")
        user_msg = messages[1]
        assert user_msg.content.index("<plan>") < user_msg.content.index("<state>")


class TestCachedRendererEndToEndOpenAI:
    def test_static_becomes_system_message(self):
        # Non-dynamic-source block: ``expr=`` is suppressed even if provided
        # in metadata. Only ``self.context.set_dynamic()`` blocks render it.
        result = render_context(
            [_static_block("sys", "You are X.", expr="self._system_prompt()")],
            block_formatter=CachedBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        ).output
        assert result == [{"role": "system", "content": "<sys>\nYou are X.\n</sys>"}]

    def test_volatile_becomes_trailing_user(self):
        result = render_context(
            [
                _static_block("sys", "S"),
                _dynamic_block("plan", "P", expr="self.context['plan']"),
            ],
            block_formatter=CachedBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        ).output
        assert len(result) == 2
        assert result[0]["role"] == "system"
        assert "<sys>" in result[0]["content"]
        assert result[1]["role"] == "user"
        assert "<context>" in result[1]["content"]
        assert "<plan" in result[1]["content"]

    def test_volatile_appended_after_trailing_user_event(self):
        """Dynamic ``<context>`` is its own user message — never merged into a
        historical user event. Merging would mutate that event's bytes when a
        later turn becomes the new trailing event, breaking provider prompt
        caching (issue #208)."""
        user_event = UserEvent(content="hi")
        user_event.tag = "1"
        blocks = [
            _static_block("sys", "S"),
            _dynamic_block("plan", "P"),
            ResolvedBlock(
                key="event_1",
                content="hi",
                role=Role.USER,
                metadata=BlockMetadata(tag="1"),
                event=user_event,
            ),
        ]
        result = render_context(
            blocks,
            block_formatter=CachedBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        ).output
        roles = [m["role"] for m in result]
        assert roles == ["system", "user", "user"]
        # The user-event message is preserved verbatim — no context envelope.
        event_content = result[1]["content"]
        assert "<context>" not in event_content
        assert "<plan>" not in event_content
        assert "hi" in event_content
        # The trailing user message holds only the context envelope.
        context_content = result[2]["content"]
        assert context_content.startswith("<context>")
        assert context_content.endswith("</context>")
        assert "<plan>" in context_content
        assert "hi" not in context_content

    def test_trailing_user_event_byte_stable_across_renders(self):
        """The bytes of a historical user-event message must not depend on
        whether a fresh tool turn is appended after it, nor on the value of
        the dynamic ``<context>`` blocks. This is the property OpenAI/
        Anthropic prompt caching relies on to hit the cache for the event
        tail across consecutive LLM calls (issue #208)."""
        user_event = UserEvent(content="please run task X")
        user_event.tag = "1"
        user_block = ResolvedBlock(
            key="event_1",
            content="please run task X",
            role=Role.USER,
            metadata=BlockMetadata(tag="1"),
            event=user_event,
        )

        def render(dynamic_value: str, extra_blocks: list[ResolvedBlock]) -> list[dict]:
            blocks = [
                _static_block("sys", "S"),
                _dynamic_block("plan", dynamic_value),
                user_block,
                *extra_blocks,
            ]
            return render_context(
                blocks,
                block_formatter=CachedBlockFormatter(),
                provider_formatter=OpenAIProviderFormatter(),
            ).output

        # Render 1: user_event is trailing event, dynamic value v1.
        out1 = render("plan-v1", [])
        # Render 2: user_event is trailing event, dynamic value v2 (dynamic content churn between turns).
        out2 = render("plan-v2", [])
        # Render 3: user_event is no longer trailing — a tool_call/tool_result pair has been appended.
        tool_event = ToolCallEvent(
            tool_call_id="call_42",
            name="do_thing",
            arguments={"x": 1},
            result=ToolResult(tool_call_id="call_42", content="ok"),
        )
        tool_event.tag = "2"
        tool_block = ResolvedBlock(
            key="event_2",
            content="",
            role=Role.ASSISTANT,
            metadata=BlockMetadata(tag="2"),
            event=tool_event,
        )
        out3 = render("plan-v3", [tool_block])

        # Locate the user-event message in each render (the first "user" dict
        # whose content contains "please run task X").
        def find_user_event_msg(out: list[dict]) -> dict:
            for m in out:
                if m.get("role") == "user" and "please run task X" in (m.get("content") or ""):
                    return m
            raise AssertionError("user-event message not found")

        msg1 = find_user_event_msg(out1)
        msg2 = find_user_event_msg(out2)
        msg3 = find_user_event_msg(out3)

        # Byte-identical user-event content across all three renders — this is
        # exactly what the bug in cached.py:125-147 violated.
        assert msg1["content"] == msg2["content"] == msg3["content"]
        # And specifically: no context envelope leaked into the user-event msg.
        assert "<context>" not in msg1["content"]

    def test_volatile_appended_after_assistant(self):
        asst_event = AssistantEvent(content="done")
        asst_event.tag = "2"
        blocks = [
            _static_block("sys", "S"),
            _dynamic_block("plan", "P"),
            ResolvedBlock(
                key="event_2",
                content="",
                role=Role.ASSISTANT,
                metadata=BlockMetadata(tag="2"),
                event=asst_event,
            ),
        ]
        result = render_context(
            blocks,
            block_formatter=CachedBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        ).output
        roles = [m["role"] for m in result]
        assert roles == ["system", "assistant", "user"]
        assert "<context>" in result[2]["content"]

    def test_no_volatile_no_trailing_message(self):
        result = render_context(
            [_static_block("sys", "S")],
            block_formatter=CachedBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        ).output
        assert len(result) == 1 and result[0]["role"] == "system"

    def test_empty_input(self):
        result = render_context(
            [],
            block_formatter=CachedBlockFormatter(),
            provider_formatter=OpenAIProviderFormatter(),
        ).output
        assert result == []


class TestCachedRendererEndToEndAnthropic:
    def test_returns_system_and_messages_dict(self):
        result = render_context(
            [_static_block("sys", "S"), _dynamic_block("plan", "P")],
            block_formatter=CachedBlockFormatter(),
            provider_formatter=AnthropicProviderFormatter(),
        ).output
        assert isinstance(result, dict)
        assert "system" in result and "messages" in result
        assert "<sys>" in result["system"]
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert "<context>" in result["messages"][0]["content"]
