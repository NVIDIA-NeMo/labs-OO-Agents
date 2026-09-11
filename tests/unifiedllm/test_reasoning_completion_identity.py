# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reasoning cleanup replaces public parts without mutating the captured turn."""

from unittest.mock import AsyncMock, patch

import pytest

from nooa.unifiedllm import CompletionClient, LLMResponse, ReasoningCompletionClient


def _response() -> LLMResponse:
    return LLMResponse(
        content="<think>new thought</think>public answer",
        parsed={"answer": 42},
        reasoning="provider thought",
        usage={"input_tokens": 10, "output_tokens": 5},
    )


def _assert_cleaned_in_place(response: LLMResponse, returned: LLMResponse) -> None:
    assert returned is not response
    assert response.content == "<think>new thought</think>public answer"
    assert returned.content == "public answer"
    assert returned.reasoning == "provider thought\n\nnew thought"
    assert returned.parsed is None
    assert returned.replay_scope is None
    assert all(part.native is None for part in returned.parts)
    assert returned.usage == response.usage


def test_sync_cleanup_mutates_only_reasoning_views() -> None:
    response = _response()
    identity = (response.id, response.timestamp)
    client = ReasoningCompletionClient(model="test-model")

    turns = {response.id: response}
    with patch.object(CompletionClient, "call", return_value=response) as parent:
        returned = client.call([{"role": "user", "content": "hi"}], turns=turns)
    assert parent.call_args.kwargs["turns"] is turns

    _assert_cleaned_in_place(response, returned)
    assert (returned.id, returned.timestamp) == identity
    client.close()


@pytest.mark.asyncio
async def test_async_cleanup_mutates_only_reasoning_views() -> None:
    response = _response()
    identity = (response.id, response.timestamp)
    client = ReasoningCompletionClient(model="test-model")

    turns = {response.id: response}
    with patch.object(CompletionClient, "acall", new=AsyncMock(return_value=response)) as parent:
        returned = await client.acall([{"role": "user", "content": "hi"}], turns=turns)
    assert parent.call_args.kwargs["turns"] is turns

    _assert_cleaned_in_place(response, returned)
    assert (returned.id, returned.timestamp) == identity
    await client.aclose()
