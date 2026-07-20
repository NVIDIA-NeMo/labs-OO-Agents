# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for ActorRuntime.generate() tool-list forwarding.

Bedrock via litellm rejects ``tools=[]`` with UnsupportedParamsError ("Bedrock
doesn't support tool calling without tools= param specified"). Callers like
``PredictStrategy`` pass ``tools=[]`` to signal "tool-free generation"; the
runtime must translate that intent to "omit tools from the request" (i.e.
forward ``tools=None`` to the underlying LLM client), not forward the empty
list verbatim.

These tests pin that behaviour both at the generate() seam and end-to-end
through PredictStrategy.
"""

from typing import Any

import pytest
from pydantic import BaseModel

from nooa import Agent, PredictStrategy, strategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse


def _resp(content: Any) -> LLMResponse:
    """Build a minimal LLMResponse for a FakeLLMClient."""
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": str(content)},
    )


class _Label(BaseModel):
    label: str


@pytest.mark.asyncio
async def test_generate_empty_tools_list_is_normalized_to_none():
    """generate(tools=[]) must not forward an empty list to the LLM client.

    The FakeLLMClient records last_tools=None when no tools are passed and
    last_tools=<list> when a non-empty list is passed; we rely on that to
    assert normalization.
    """
    fake = FakeLLMClient(scripted_responses=[_resp(_Label(label="x"))])

    class A(Agent, llm=fake):
        @strategy(PredictStrategy())
        async def classify(self, text: str) -> _Label:
            """Classify {text}."""
            ...

    agent = A()
    await agent.classify("hello")

    assert fake.last_tools is None, (
        f"Expected tools=None to be forwarded, got {fake.last_tools!r}. "
        "ActorRuntime.generate() must normalize empty tool lists to None so "
        "providers that reject tools=[] (Bedrock) work."
    )


# Non-empty tools paths are exercised end-to-end by the existing CodeAct and
# PurePython strategy tests — those strategies always pass a real tools list
# to generate() and would break if the normalization over-applied.
