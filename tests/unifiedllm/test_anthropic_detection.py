# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for `_is_anthropic_model` and `_inject_cache_control` behavior."""

from __future__ import annotations

from copy import deepcopy

import pytest

from nooa.unifiedllm.unifiedllm import (
    CompletionClient,
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


def _make_messages_with_tool_result() -> list[dict]:
    return [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Compute 2*5."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "execute_python", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "status: complete"},
        {"role": "user", "content": "follow-up"},
    ]


def test_inject_cache_control_active_on_anthropic_models() -> None:
    """On Anthropic models, the last tool message should be flipped to parts
    format with cache_control attached, and system gets cache_control on the
    message envelope."""
    client = CompletionClient(model="anthropic/claude-sonnet-4-5")
    original = _make_messages_with_tool_result()
    result = client._inject_cache_control(
        original,
        [{"role": "system"}, {"role": "tool", "position": "last"}],
    )
    # system message — gains cache_control at message level
    assert result[0]["cache_control"] == {"type": "ephemeral"}
    # last tool message — content converted to parts with cache_control on last block
    tool_content = result[3]["content"]
    assert isinstance(tool_content, list), "tool content should be parts array"
    assert tool_content[-1].get("cache_control") == {"type": "ephemeral"}


def test_inject_cache_control_noop_on_empty_injection_points() -> None:
    """Empty injection_points short-circuits without mutation."""
    client = CompletionClient(model="anthropic/claude-sonnet")
    original = _make_messages_with_tool_result()
    baseline = deepcopy(original)
    result = client._inject_cache_control(original, [])
    assert result == baseline
    assert original == baseline
