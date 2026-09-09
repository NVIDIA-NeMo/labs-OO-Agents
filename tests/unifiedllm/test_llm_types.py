# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical UnifiedLLM response and usage contracts."""

import pytest

from nooa.unifiedllm import FakeLLMClient, LLMResponse, LLMUsage


@pytest.mark.parametrize(
    ("provider_usage", "expected"),
    [
        (
            {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 75},
                "completion_tokens_details": {"reasoning_tokens": 8},
                "total_tokens": 120,
            },
            LLMUsage(
                input_tokens=100,
                output_tokens=20,
                cached_input_tokens=75,
                reasoning_tokens=8,
                total_tokens=120,
            ),
        ),
        (
            {
                "input_tokens": 90,
                "output_tokens": 10,
                "cache_read_input_tokens": 60,
                "cache_creation_input_tokens": 15,
            },
            LLMUsage(
                input_tokens=90,
                output_tokens=10,
                cached_input_tokens=60,
                cache_write_input_tokens=15,
                total_tokens=100,
            ),
        ),
    ],
)
def test_usage_normalizes_provider_cache_and_reasoning_fields(provider_usage, expected):
    assert LLMUsage.from_provider(provider_usage) == expected


@pytest.mark.asyncio
async def test_fake_materializes_repeated_script_aliases_as_distinct_events():
    response = LLMResponse(content="same response")
    fake = FakeLLMClient(scripted_responses=[response, response, response])

    first = await fake.acall([])
    second = await fake.acall([])
    third = await fake.acall([])

    assert first is response
    assert second is not response
    assert third is not response
    assert len({first.id, second.id, third.id}) == 3
    assert [first.content, second.content, third.content] == ["same response"] * 3
