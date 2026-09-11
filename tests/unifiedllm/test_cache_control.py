# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for explicit cache-boundary transport in CompletionClient."""

from unittest.mock import AsyncMock, patch

import litellm
import pytest

from nooa.context_blocks.models import CACHE_BOUNDARY_MESSAGE_KEY
from nooa.unifiedllm import CompletionClient, FakeLLMClient


def make_mock_response(content: str = "ok") -> litellm.ModelResponse:
    msg = litellm.Message(content=content, role="assistant")
    choice = litellm.Choices(message=msg, index=0, finish_reason="stop")
    return litellm.ModelResponse(choices=[choice], model="test-model")


class TestApplyCacheBoundaries:
    @pytest.fixture
    def client(self):
        return CompletionClient(model="test-model")

    def test_no_boundary_returns_original_messages(self, client):
        messages = [{"role": "system", "content": "System"}]

        assert client._apply_cache_boundaries(messages) is messages
        assert "cache_control" not in messages[0]

    def test_maps_only_explicit_boundary_without_mutating_input(self, client):
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "First", CACHE_BOUNDARY_MESSAGE_KEY: True},
            {"role": "user", "content": "Second"},
        ]

        result = client._apply_cache_boundaries(messages)

        assert "cache_control" not in result[0]
        assert result[1]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in result[2]
        assert all(CACHE_BOUNDARY_MESSAGE_KEY not in message for message in result)
        assert CACHE_BOUNDARY_MESSAGE_KEY in messages[1]

    def test_false_boundary_marker_is_removed(self, client):
        messages = [{"role": "system", "content": "System", CACHE_BOUNDARY_MESSAGE_KEY: False}]

        assert client._apply_cache_boundaries(messages) == [{"role": "system", "content": "System"}]

    def test_unsupported_boundary_is_removed(self, client):
        messages = [{"role": "user", "content": "Hi", CACHE_BOUNDARY_MESSAGE_KEY: True}]

        assert client._apply_cache_boundaries(messages, supported=False) == [
            {"role": "user", "content": "Hi"}
        ]

    def test_provider_native_cache_control_is_preserved(self, client):
        native = {"type": "provider-specific", "ttl": "1h"}
        messages = [
            {
                "role": "user",
                "content": "Hi",
                "cache_control": native,
                CACHE_BOUNDARY_MESSAGE_KEY: True,
            }
        ]

        result = client._apply_cache_boundaries(messages)

        assert result == [{"role": "user", "content": "Hi", "cache_control": native}]

    @pytest.mark.asyncio
    async def test_fake_strips_boundary_without_changing_content(self):
        client = FakeLLMClient()
        messages = [{"role": "user", "content": "Hi", CACHE_BOUNDARY_MESSAGE_KEY: True}]

        await client.acall(messages)

        assert client.last_messages == [{"role": "user", "content": "Hi"}]
        assert CACHE_BOUNDARY_MESSAGE_KEY in messages[0]

    def test_anthropic_boundary_marks_last_content_block(self):
        client = CompletionClient(model="anthropic/claude-haiku-4-5")
        messages = [{"role": "user", "content": "Hi", CACHE_BOUNDARY_MESSAGE_KEY: True}]

        result = client._apply_cache_boundaries(messages)

        assert result[0]["content"] == [
            {"type": "text", "text": "Hi", "cache_control": {"type": "ephemeral"}}
        ]

    @pytest.mark.parametrize("content", ["working", None])
    def test_anthropic_tool_call_boundary_follows_tool_use(self, content):
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        client = CompletionClient(model="anthropic/claude-haiku-4-5")
        messages = [
            {"role": "user", "content": "run it"},
            {
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {
                        "id": "tc1",
                        "type": "function",
                        "function": {"name": "run", "arguments": "{}"},
                    }
                ],
                CACHE_BOUNDARY_MESSAGE_KEY: True,
            },
            {"role": "tool", "tool_call_id": "tc1", "content": "done"},
        ]

        result = client._apply_cache_boundaries(messages)
        transformed = AnthropicConfig().transform_request(
            model="claude-haiku-4-5",
            messages=result,
            optional_params={},
            litellm_params={},
            headers={},
        )

        assistant_blocks = transformed["messages"][1]["content"]
        assert assistant_blocks[-1]["type"] == "tool_use"
        assert assistant_blocks[-1]["cache_control"] == {"type": "ephemeral"}


class TestRemovedImplicitCacheRules:
    def test_constructor_rejects_legacy_option(self):
        with pytest.raises(TypeError, match="cache_control_injection_points was removed"):
            CompletionClient(model="test-model", cache_control_injection_points=[])

    def test_call_rejects_legacy_option(self):
        client = CompletionClient(model="test-model")
        with pytest.raises(TypeError, match="cache_control_injection_points was removed"):
            client.call([], cache_control_injection_points=[])

    @pytest.mark.asyncio
    async def test_acall_rejects_legacy_option(self):
        client = CompletionClient(model="test-model")
        with pytest.raises(TypeError, match="cache_control_injection_points was removed"):
            await client.acall([], cache_control_injection_points=[])


class TestCacheControlPreservePatch:
    def test_patch_preserves_for_anthropic(self):
        from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

        messages = [{"role": "system", "content": "Hi", "cache_control": {"type": "ephemeral"}}]
        result, _ = OpenAIGPTConfig().remove_cache_control_flag_from_messages_and_tools(
            model="openai/aws/anthropic/bedrock-claude-sonnet-4-5-v1",
            messages=messages,
        )

        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_patch_strips_for_non_anthropic(self):
        from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

        messages = [{"role": "system", "content": "Hi", "cache_control": {"type": "ephemeral"}}]
        result, _ = OpenAIGPTConfig().remove_cache_control_flag_from_messages_and_tools(
            model="openai/gpt-4o",
            messages=messages,
        )

        assert "cache_control" not in result[0]


class TestCacheBoundaryEndToEnd:
    @pytest.mark.asyncio
    async def test_acall_without_boundary_adds_no_cache_control(self):
        client = CompletionClient(model="test-model")

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = make_mock_response()
            await client.acall([{"role": "system", "content": "System"}])

        sent = mock_acompletion.call_args.kwargs["messages"]
        assert sent == [{"role": "system", "content": "System"}]

    @pytest.mark.asyncio
    async def test_acall_maps_boundary_before_litellm(self):
        client = CompletionClient(model="test-model")
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Boundary", CACHE_BOUNDARY_MESSAGE_KEY: True},
        ]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = make_mock_response()
            await client.acall(messages)

        sent = mock_acompletion.call_args.kwargs["messages"]
        assert "cache_control" not in sent[0]
        assert sent[1]["cache_control"] == {"type": "ephemeral"}

    def test_call_maps_boundary_before_litellm(self):
        client = CompletionClient(model="test-model")
        messages = [{"role": "system", "content": "System", CACHE_BOUNDARY_MESSAGE_KEY: True}]

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = make_mock_response()
            client.call(messages)

        sent = mock_completion.call_args.kwargs["messages"]
        assert sent[0]["cache_control"] == {"type": "ephemeral"}
