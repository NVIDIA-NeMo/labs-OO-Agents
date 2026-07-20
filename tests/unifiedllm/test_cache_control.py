# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Anthropic prompt caching support in CompletionClient.

Tests the cache_control injection and the litellm patch that prevents
cache_control from being stripped for Anthropic models.
"""

from unittest.mock import AsyncMock, patch

import litellm
import pytest

from nooa.unifiedllm import CompletionClient


def make_mock_response(content: str = "ok") -> litellm.ModelResponse:
    """Create a minimal litellm.ModelResponse for testing."""
    msg = litellm.Message(content=content, role="assistant")
    choice = litellm.Choices(message=msg, index=0, finish_reason="stop")
    return litellm.ModelResponse(choices=[choice], model="test-model")


# ---------------------------------------------------------------------------
# _inject_cache_control
# ---------------------------------------------------------------------------


class TestInjectCacheControl:
    """Tests for CompletionClient._inject_cache_control."""

    @pytest.fixture
    def client(self):
        return CompletionClient(model="test-model")

    def test_adds_cache_control_to_system_messages(self, client):
        """cache_control is added to messages matching the target role."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        injection_points = [{"location": "message", "role": "system"}]

        result = client._inject_cache_control(messages, injection_points)

        assert result[0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in result[1]

    def test_does_not_mutate_original(self, client):
        """Original message list must not be modified."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        injection_points = [{"location": "message", "role": "system"}]

        result = client._inject_cache_control(messages, injection_points)

        assert "cache_control" not in messages[0]
        assert "cache_control" in result[0]

    def test_multiple_roles(self, client):
        """Can target multiple roles at once."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Bye"},
        ]
        injection_points = [
            {"location": "message", "role": "system"},
            {"location": "message", "role": "user"},
        ]

        result = client._inject_cache_control(messages, injection_points)

        assert "cache_control" in result[0]  # system
        assert "cache_control" in result[1]  # user
        assert "cache_control" not in result[2]  # assistant
        assert "cache_control" in result[3]  # user

    def test_empty_injection_points(self, client):
        """Empty injection_points returns messages unchanged."""
        messages = [{"role": "system", "content": "Hi"}]
        result = client._inject_cache_control(messages, [])
        assert result is messages  # same object, no copy needed

    def test_no_matching_role(self, client):
        """No-op when no messages match the target role."""
        messages = [{"role": "user", "content": "Hi"}]
        injection_points = [{"location": "message", "role": "system"}]

        result = client._inject_cache_control(messages, injection_points)

        assert "cache_control" not in result[0]


# ---------------------------------------------------------------------------
# Default cache_control_injection_points
# ---------------------------------------------------------------------------


class TestDefaultCacheControlInjectionPoints:
    """Tests for default cache control configuration."""

    def test_default_targets_system_and_last_tool(self):
        """Default injection points target system role + last tool."""
        client = CompletionClient(model="test-model")
        assert client.cache_control_injection_points == [
            {"role": "system"},
            {"role": "tool", "position": "last"},
        ]

    def test_custom_injection_points(self):
        """Custom injection points override the default."""
        custom = [{"location": "message", "role": "user"}]
        client = CompletionClient(model="test-model", cache_control_injection_points=custom)
        assert client.cache_control_injection_points == custom

    def test_empty_list_disables(self):
        """Passing an empty list disables cache control injection."""
        client = CompletionClient(model="test-model", cache_control_injection_points=[])
        assert client.cache_control_injection_points == []


# ---------------------------------------------------------------------------
# litellm patch preserves cache_control for Anthropic models
# ---------------------------------------------------------------------------


class TestCacheControlPreservePatch:
    """Tests for the monkey-patch that prevents litellm from stripping cache_control."""

    def test_patch_preserves_for_anthropic(self):
        """cache_control survives for Anthropic model names."""
        from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

        config = OpenAIGPTConfig()
        messages = [
            {"role": "system", "content": "Hi", "cache_control": {"type": "ephemeral"}},
            {"role": "user", "content": "Hello"},
        ]

        result_messages, _ = config.remove_cache_control_flag_from_messages_and_tools(
            model="openai/aws/anthropic/bedrock-claude-sonnet-4-5-v1",
            messages=messages,
        )

        # cache_control should be preserved
        assert result_messages[0].get("cache_control") == {"type": "ephemeral"}

    def test_patch_strips_for_non_anthropic(self):
        """cache_control is still stripped for non-Anthropic models."""
        from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig

        config = OpenAIGPTConfig()
        messages = [
            {"role": "system", "content": "Hi", "cache_control": {"type": "ephemeral"}},
            {"role": "user", "content": "Hello"},
        ]

        result_messages, _ = config.remove_cache_control_flag_from_messages_and_tools(
            model="openai/gpt-4o",
            messages=messages,
        )

        # cache_control should be stripped for non-Anthropic
        assert "cache_control" not in result_messages[0]


# ---------------------------------------------------------------------------
# End-to-end: cache_control reaches litellm.completion
# ---------------------------------------------------------------------------


class TestCacheControlEndToEnd:
    """Verify cache_control is present in the messages passed to litellm."""

    @pytest.mark.asyncio
    async def test_acall_passes_cache_control(self):
        """acall() should inject cache_control before calling litellm."""
        client = CompletionClient(model="openai/aws/anthropic/bedrock-claude-sonnet-4-5-v1")
        mock_response = make_mock_response()

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            await client.acall(
                [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hi"},
                ]
            )

            call_kwargs = mock_acompletion.call_args[1]
            sent_messages = call_kwargs["messages"]

            # System message should have cache_control
            assert sent_messages[0].get("cache_control") == {"type": "ephemeral"}
            # User message should not
            assert "cache_control" not in sent_messages[1]

    @pytest.mark.asyncio
    async def test_acall_no_injection_points_skips(self):
        """No cache_control when injection_points is empty."""
        client = CompletionClient(
            model="openai/aws/anthropic/bedrock-claude-sonnet-4-5-v1",
            cache_control_injection_points=[],
        )
        mock_response = make_mock_response()

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            await client.acall(
                [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hi"},
                ]
            )

            call_kwargs = mock_acompletion.call_args[1]
            sent_messages = call_kwargs["messages"]

            assert "cache_control" not in sent_messages[0]

    def test_sync_call_passes_cache_control(self):
        """Sync call() should also inject cache_control."""
        client = CompletionClient(model="openai/aws/anthropic/bedrock-claude-sonnet-4-5-v1")
        mock_response = make_mock_response()

        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = mock_response

            client.call(
                [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hi"},
                ]
            )

            call_kwargs = mock_completion.call_args[1]
            sent_messages = call_kwargs["messages"]

            assert sent_messages[0].get("cache_control") == {"type": "ephemeral"}
            assert "cache_control" not in sent_messages[1]


# ---------------------------------------------------------------------------
# Position-based cache_control injection
# ---------------------------------------------------------------------------


class TestPositionBasedInjection:
    """Tests for position-based cache_control injection (position='last')."""

    @pytest.fixture
    def client(self):
        return CompletionClient(model="test-model", cache_control_injection_points=[])

    def test_last_assistant_marked(self, client):
        """position='last' marks only the last message of the specified role (content-block level)."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Turn 2"},
            {"role": "assistant", "content": "Response 2"},
            {"role": "user", "content": "Turn 3"},
        ]
        injection_points = [{"role": "assistant", "position": "last"}]

        result = client._inject_cache_control(messages, injection_points)

        # Only the last assistant message should have cache_control on content block
        assert "cache_control" not in result[0]  # system
        assert "cache_control" not in result[1]  # user
        assert "cache_control" not in result[2]  # assistant (not last) - unchanged
        assert "cache_control" not in result[3]  # user
        # Last assistant: content converted to array with cache_control on the block
        assert result[4]["content"] == [
            {"type": "text", "text": "Response 2", "cache_control": {"type": "ephemeral"}}
        ]
        assert "cache_control" not in result[5]  # user (current turn)

    def test_combined_role_and_position(self, client):
        """Role-based and position-based injection work together."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Turn 2"},
            {"role": "assistant", "content": "Response 2"},
            {"role": "user", "content": "Turn 3"},
        ]
        injection_points = [
            {"role": "system"},
            {"role": "assistant", "position": "last"},
        ]

        result = client._inject_cache_control(messages, injection_points)

        assert result[0]["cache_control"] == {
            "type": "ephemeral"
        }  # system (role-based, message-level)
        assert "cache_control" not in result[1]  # user
        assert "cache_control" not in result[2]  # assistant (not last)
        assert "cache_control" not in result[3]  # user
        # Last assistant: content-block-level injection
        assert result[4]["content"] == [
            {"type": "text", "text": "Response 2", "cache_control": {"type": "ephemeral"}}
        ]
        assert "cache_control" not in result[5]  # user

    def test_no_message_of_role_is_noop(self, client):
        """position='last' is a no-op if no messages of that role exist."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "First message"},
        ]
        injection_points = [{"role": "assistant", "position": "last"}]

        result = client._inject_cache_control(messages, injection_points)

        assert "cache_control" not in result[0]
        assert "cache_control" not in result[1]

    def test_single_assistant_message(self, client):
        """Works with only one assistant message."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Bye"},
        ]
        injection_points = [{"role": "assistant", "position": "last"}]

        result = client._inject_cache_control(messages, injection_points)

        # Content-block-level injection on the assistant message
        assert result[2]["content"] == [
            {"type": "text", "text": "Hello", "cache_control": {"type": "ephemeral"}}
        ]
        assert "cache_control" not in result[0]
        assert "cache_control" not in result[1]
        assert "cache_control" not in result[3]

    def test_does_not_mutate_original(self, client):
        """Position-based injection does not mutate the original messages."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        injection_points = [{"role": "assistant", "position": "last"}]

        client._inject_cache_control(messages, injection_points)

        # Original message should be unchanged (content still a string, no cache_control)
        assert messages[2]["content"] == "Hello"
        assert "cache_control" not in messages[2]

    def test_does_not_mutate_original_list_content(self, client):
        """Position-based injection does not mutate existing content blocks."""
        messages = [
            {"role": "tool", "content": [{"type": "text", "text": "tool output"}]},
        ]
        injection_points = [{"role": "tool", "position": "last"}]

        result = client._inject_cache_control(messages, injection_points)

        assert messages[0]["content"] == [{"type": "text", "text": "tool output"}]
        assert result[0]["content"] == [
            {"type": "text", "text": "tool output", "cache_control": {"type": "ephemeral"}}
        ]

    def test_last_user_position(self, client):
        """position='last' works for user role too."""
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": "Turn 2"},
        ]
        injection_points = [{"role": "user", "position": "last"}]

        result = client._inject_cache_control(messages, injection_points)

        assert "cache_control" not in result[1]  # first user - unchanged
        # Last user: content-block-level injection
        assert result[3]["content"] == [
            {"type": "text", "text": "Turn 2", "cache_control": {"type": "ephemeral"}}
        ]


class TestDefaultIncludesPositionBased:
    """Verify the updated default includes position-based assistant caching."""

    def test_default_has_system_and_last_tool(self):
        """Default injection points mark system AND last tool."""
        client = CompletionClient(model="test-model")
        assert {"role": "system"} in client.cache_control_injection_points
        assert {"role": "tool", "position": "last"} in client.cache_control_injection_points

    @pytest.mark.asyncio
    async def test_default_caches_system_and_last_tool(self):
        """End-to-end: default config marks system + last tool (CodeAct-style)."""
        client = CompletionClient(model="openai/aws/anthropic/bedrock-claude-sonnet-4-5-v1")
        mock_response = make_mock_response()

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            await client.acall(
                [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Do something"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tc1",
                                "type": "function",
                                "function": {"name": "run", "arguments": "{}"},
                            }
                        ],
                    },
                    {"role": "tool", "content": "first result", "tool_call_id": "tc1"},
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "tc2",
                                "type": "function",
                                "function": {"name": "run", "arguments": "{}"},
                            }
                        ],
                    },
                    {"role": "tool", "content": "final result", "tool_call_id": "tc2"},
                    {"role": "user", "content": "Current turn"},
                ],
                tools=[],
            )

            sent_messages = mock_acompletion.call_args[1]["messages"]

            # System should be marked at message level
            assert sent_messages[0].get("cache_control") == {"type": "ephemeral"}
            # Last tool (index 5) should have content-block-level cache_control
            assert sent_messages[5]["content"] == [
                {"type": "text", "text": "final result", "cache_control": {"type": "ephemeral"}}
            ]
            # Earlier tool (index 3) should NOT be marked
            assert sent_messages[3]["content"] == "first result"
            assert "cache_control" not in sent_messages[3]
            # Assistant messages should NOT be marked
            assert "cache_control" not in sent_messages[2]
            assert "cache_control" not in sent_messages[4]
