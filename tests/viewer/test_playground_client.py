# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import AsyncMock

import pytest

from nooa.llm_types import LLMResponse, LLMUsage


@pytest.mark.asyncio
async def test_playground_uses_shared_client_and_public_response(monkeypatch):
    from nooa import unifiedllm
    from nooa.viewer import trace_routes

    client = AsyncMock()
    client.acall.return_value = LLMResponse(
        content="answer",
        reasoning="readable",
        usage=LLMUsage(input_tokens=3, output_tokens=2, total_tokens=5),
    )
    monkeypatch.setattr(unifiedllm, "CompletionClient", lambda **kwargs: client)
    monkeypatch.setattr(trace_routes, "get_model_config", lambda model: {})
    result = await trace_routes.run_inference(
        trace_routes.InferenceRequest(
            model="test", messages=[{"role": "user", "content": "question"}]
        )
    )
    assert result["response"]["content"] == "answer"
    assert result["response"]["reasoning_content"] == "readable"
    assert result["usage"]["prompt_tokens"] == 3
    assert client.acall.await_count == client.aclose.await_count == 1
