# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transport preservation of native cache markers."""

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
