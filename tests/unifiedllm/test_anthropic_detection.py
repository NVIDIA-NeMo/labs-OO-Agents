# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `_is_anthropic_model`."""

from __future__ import annotations

import pytest

from nooa.unifiedllm.unifiedllm import (
    _is_anthropic_model,
)


@pytest.mark.parametrize(
    "model,expected",
    [
        # Direct Anthropic API
        ("anthropic/claude-sonnet-4-5", True),
        ("anthropic.claude-3-5-haiku-20241022", True),
        # NVIDIA gateway → direct Anthropic
        ("aws/anthropic/claude-haiku-4-5-v1", True),
        # Bedrock-routed Anthropic
        ("bedrock/anthropic.claude-3-5-sonnet", True),
        ("openai/aws/anthropic/bedrock-claude-sonnet-4-5-v1", True),
        # Bedrock-routed Claude (no "anthropic" in id, just "claude")
        ("bedrock/claude-3-5-sonnet", True),
        # Short claude- aliases (some clients use these)
        ("claude-3-5-sonnet", True),
        ("claude/foo", True),
        # OpenAI (direct and Azure)
        ("openai/gpt-5.5", False),
        ("openai/openai/gpt-5.5", False),
        ("azure/openai/gpt-5.5", False),
        ("openai/azure/openai/gpt-5-mini", False),
        # NVIDIA NIM
        ("nvidia/nvidia/nemotron-3-super-v3", False),
        # Hugging Face / vLLM passthrough
        ("huggingface/meta-llama/Llama-3.1-70B-Instruct", False),
        # Non-Anthropic Bedrock providers — must NOT match, otherwise
        # cache_control gets attached to Titan/Cohere/Llama which don't
        # use it.
        ("bedrock/amazon.titan-text-express-v1", False),
        ("aws/cohere.command-r-plus-v1", False),
        ("bedrock/meta.llama3-70b-instruct-v1", False),
        # Edge cases
        ("", False),
    ],
)
def test_is_anthropic_model(model: str, expected: bool) -> None:
    assert _is_anthropic_model(model) is expected
