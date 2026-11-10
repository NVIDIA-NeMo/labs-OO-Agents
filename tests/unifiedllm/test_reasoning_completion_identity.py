# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reasoning cleanup replaces public parts without mutating the captured turn."""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from nooa.unifiedllm import CompletionClient, LLMResponse, ReasoningCompletionClient


class Answer(BaseModel):
    answer: int


def _response() -> LLMResponse:
    return LLMResponse(
        content='<think>new thought</think>{"answer":42}',
        parsed=Answer(answer=42),
        reasoning="provider thought",
        usage={"input_tokens": 10, "output_tokens": 5},
    )


def _assert_cleaned_in_place(response: LLMResponse, returned: LLMResponse) -> None:
    assert returned is not response
    assert response.content == '<think>new thought</think>{"answer":42}'
    assert returned.content == '{"answer":42}'
    assert returned.reasoning == "provider thought\n\nnew thought"
    assert returned.parsed is response.parsed
    assert returned.replay_scope is None
    assert all(part.native is None for part in returned.parts)
    assert returned.usage == response.usage


def test_sync_cleanup_mutates_only_reasoning_views() -> None:
    response = _response()
    identity = (response.id, response.timestamp)
    client = ReasoningCompletionClient(model="test-model")

    with patch.object(CompletionClient, "call", return_value=response) as parent:
        returned = client.call([{"role": "user", "content": "hi"}], output_model=Answer)
    assert "turns" not in parent.call_args.kwargs

    _assert_cleaned_in_place(response, returned)
    assert (returned.id, returned.timestamp) == identity
    client.close()


@pytest.mark.asyncio
async def test_async_cleanup_mutates_only_reasoning_views() -> None:
    response = _response()
    identity = (response.id, response.timestamp)
    client = ReasoningCompletionClient(model="test-model")

    with patch.object(CompletionClient, "acall", new=AsyncMock(return_value=response)) as parent:
        returned = await client.acall([{"role": "user", "content": "hi"}], output_model=Answer)
    assert "turns" not in parent.call_args.kwargs

    _assert_cleaned_in_place(response, returned)
    assert (returned.id, returned.timestamp) == identity
    await client.aclose()
