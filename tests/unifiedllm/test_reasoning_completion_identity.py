# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reasoning cleanup must not replace the canonical response object."""

from unittest.mock import AsyncMock, patch

import pytest

from nooa.unifiedllm import CompletionClient, LLMResponse, ReasoningCompletionClient


def _response() -> LLMResponse:
    return LLMResponse(
        content="<think>new thought</think>public answer",
        parsed={"answer": 42},
        reasoning="provider thought",
        llm_state={"opaque": "state"},
        usage={"input_tokens": 10, "output_tokens": 5},
    )


def _assert_cleaned_in_place(response: LLMResponse, returned: LLMResponse) -> None:
    assert returned is response
    assert returned.content == "public answer"
    assert returned.reasoning == "provider thought\n\nnew thought"
    assert returned.parsed == {"answer": 42}
    assert returned.llm_state == {"opaque": "state"}
    assert returned.usage == response.usage


def test_sync_cleanup_mutates_only_reasoning_views() -> None:
    response = _response()
    identity = (response.id, response.timestamp)
    client = ReasoningCompletionClient(model="test-model")

    with patch.object(CompletionClient, "call", return_value=response):
        returned = client.call([{"role": "user", "content": "hi"}])

    _assert_cleaned_in_place(response, returned)
    assert (returned.id, returned.timestamp) == identity
    client.close()


@pytest.mark.asyncio
async def test_async_cleanup_mutates_only_reasoning_views() -> None:
    response = _response()
    identity = (response.id, response.timestamp)
    client = ReasoningCompletionClient(model="test-model")

    with patch.object(CompletionClient, "acall", new=AsyncMock(return_value=response)):
        returned = await client.acall([{"role": "user", "content": "hi"}])

    _assert_cleaned_in_place(response, returned)
    assert (returned.id, returned.timestamp) == identity
    await client.aclose()
