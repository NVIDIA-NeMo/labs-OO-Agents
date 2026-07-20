# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for provider formatter image rendering.

Verifies that PythonOutput events with images produce multi-part content
in both OpenAI and Anthropic message formats. Both use LiteLLM's universal
image_url format — no provider-specific conversion needed.
"""

from nooa.context_blocks.events import ResultStatus
from nooa.context_blocks.formatter import (
    AnthropicProviderFormatter,
    OpenAIProviderFormatter,
    XMLBlockFormatter,
)
from nooa.context_blocks.models import BlockMetadata, ResolvedBlock, Role
from nooa.context_blocks.renderer import render_context
from nooa.events import PythonOutput


def _render_openai(blocks: list[ResolvedBlock]) -> list[dict]:
    return render_context(
        blocks,
        block_formatter=XMLBlockFormatter(),
        provider_formatter=OpenAIProviderFormatter(),
    ).output


def _render_anthropic(blocks: list[ResolvedBlock]) -> dict:
    return render_context(
        blocks,
        block_formatter=XMLBlockFormatter(),
        provider_formatter=AnthropicProviderFormatter(),
    ).output


def _make_python_output_block(
    stdout: str = "hello",
    images: list[dict] | None = None,
) -> ResolvedBlock:
    """Create a ResolvedBlock carrying a PythonOutput event with optional images."""
    event = PythonOutput(
        tool_call_id="tc_1",
        execution_count=1,
        execution_status=ResultStatus.COMPLETE,
        stdout=stdout,
        images=images or [],
    )
    return ResolvedBlock(
        key="python_output",
        content=f"<stdout>{stdout}</stdout>",
        role=Role.USER,
        metadata=BlockMetadata(tag="1"),
        event=event,
    )


class TestOpenAIProviderFormatterImages:
    def test_no_images_plain_content(self):
        block = _make_python_output_block(stdout="hello", images=[])
        result = _render_openai([block])
        # System + 1 user message
        assert len(result) == 2
        msg = result[1]
        assert msg["role"] == "user"
        assert isinstance(msg["content"], str)

    def test_with_images_multipart_content(self):
        image_block = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        block = _make_python_output_block(stdout="analyzed", images=[image_block])
        result = _render_openai([block])
        msg = result[1]
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["type"] == "text"
        assert "analyzed" in msg["content"][0]["text"]
        assert msg["content"][1] == image_block

    def test_multiple_images(self):
        images = [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,BBBB"}},
        ]
        block = _make_python_output_block(images=images)
        result = _render_openai([block])
        msg = result[1]
        assert isinstance(msg["content"], list)
        assert len(msg["content"]) == 3  # text + 2 images


class TestAnthropicProviderFormatterImages:
    def test_no_images_plain_content(self):
        block = _make_python_output_block(stdout="hello", images=[])
        result = _render_anthropic([block])
        msg = result["messages"][0]
        assert isinstance(msg["content"], str)

    def test_with_images_uses_universal_format(self):
        """Anthropic formatter uses LiteLLM's universal image_url format — LiteLLM converts."""
        image_block = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,AAAA", "format": "image/png"},
        }
        block = _make_python_output_block(stdout="analyzed", images=[image_block])
        result = _render_anthropic([block])
        msg = result["messages"][0]
        assert isinstance(msg["content"], list)
        assert msg["content"][0]["type"] == "text"
        assert "analyzed" in msg["content"][0]["text"]
        assert msg["content"][1] == image_block


class TestPlainFormatterExcludesImages:
    """Verify that repr=False on PythonOutput.images excludes them from text rendering."""

    def test_images_excluded_from_plain_format(self):
        from nooa.plain_formatter import PlainBlockFormatter

        event = PythonOutput(
            tool_call_id="tc_1",
            execution_count=1,
            execution_status=ResultStatus.COMPLETE,
            stdout="hello",
            images=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,HUGE"}}],
        )
        text = PlainBlockFormatter().format_event(event)
        # Images should NOT appear in text rendering
        assert "HUGE" not in text
        assert "image_url" not in text
        # But stdout should
        assert "hello" in text
