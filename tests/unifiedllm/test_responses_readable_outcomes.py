# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Successful answers remain usable when their full wire shape cannot be replayed."""

from types import SimpleNamespace

import pytest

from nooa.llm_types import LLMResponse, LLMUsage
from nooa.unifiedllm import ResponsesClient
from nooa.unifiedllm.response_parts import project_turn


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("shape", ["builtin", "refusal", "both"])
@pytest.mark.asyncio
async def test_builtin_output_and_refusal_keep_readable_outcome(
    monkeypatch, caplog, is_async, shape
):
    refusal = shape in {"refusal", "both"}
    text = "I cannot help with that." if refusal else "The search found an answer."
    block = (
        {"type": "refusal", "refusal": text} if refusal else {"type": "output_text", "text": text}
    )
    raw = SimpleNamespace(
        output=[
            {"type": "reasoning", "encrypted_content": "secret", "summary": []},
            *(
                [{"type": "web_search_call", "id": "ws_1", "status": "completed"}]
                if shape != "refusal"
                else []
            ),
            {"type": "message", "role": "assistant", "content": [block]},
        ],
        model="gpt-5.6",
        status="completed",
    )
    monkeypatch.setattr("litellm.responses", lambda **_: raw)

    async def respond(**_):
        return raw

    monkeypatch.setattr("litellm.aresponses", respond)
    monkeypatch.setattr("nooa.unifiedllm.unifiedllm._extract_usage", lambda _: LLMUsage())
    async with ResponsesClient("openai/gpt-5.6", api_key="test") as client:
        result = await client.acall([]) if is_async else client.call([])
    assert result.content == text
    assert all(part.native is None for part in result.parts)
    assert result.replay_scope is None
    assert "Unsupported Responses output" in caplog.text
    assert "secret" not in caplog.text
    restored = LLMResponse.model_validate_json(result.model_dump_json())
    assert project_turn(restored, "responses:openai:test") == [
        {"role": "assistant", "content": text}
    ]


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_responses_calibration_counts_instructions(monkeypatch, is_async):
    from nooa.unifiedllm import unifiedllm as implementation

    calibration = implementation.TokenCalibration()
    monkeypatch.setattr(implementation, "_token_calibration", calibration)
    raw = SimpleNamespace(output=[], model="gpt-5.6", status="completed")
    monkeypatch.setattr("litellm.responses", lambda **_: raw)

    async def respond(**_):
        return raw

    monkeypatch.setattr("litellm.aresponses", respond)
    monkeypatch.setattr(implementation, "_extract_usage", lambda _: LLMUsage(input_tokens=1001))
    estimates = []

    def count(**kwargs):
        estimates.append(kwargs["messages"])
        return sum(len(m["content"]) for m in kwargs["messages"])

    monkeypatch.setattr("litellm.token_counter", count)
    messages = [{"role": "system", "content": "x" * 1000}, {"role": "user", "content": "y"}]
    async with ResponsesClient("openai/gpt-5.6", api_key="test") as client:
        await client.acall(messages) if is_async else client.call(messages)
    assert estimates == [messages]
    assert calibration.ratio("openai/gpt-5.6") == 1.0


@pytest.mark.parametrize("bad", [None, 42, {}])
def test_malformed_refusal_still_raises(bad):
    from nooa.unifiedllm.errors import ReasoningReplayError
    from nooa.unifiedllm.response_parts import capture_parts

    with pytest.raises(ReasoningReplayError, match="text must be a string"):
        capture_parts(
            [{"type": "message", "content": [{"type": "refusal", "refusal": bad}]}],
            "responses:openai:test",
        )
