# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The live smoke test must fail if summarization stops working."""

import json

import httpx
import pytest

from nooa.agents import TokenBudgetSummarizer
from nooa.unifiedllm import CompletionClient, ResponsesClient
from tests.integration.test_summarizer_live import exercise_summarization


@pytest.mark.parametrize("family", ["openai", "anthropic"])
@pytest.mark.parametrize("broken", [None, "fork", "collapse", "facts"])
async def test_release_scenario_detects_missing_summarization(family, broken, monkeypatch):
    """Exercise the same scenario through mocked HTTP, including negative controls."""
    calls = 0

    async def send(http_client, request, **kwargs):
        nonlocal calls
        calls += 1
        text = "READY" if calls == 1 else "Launch ticket-4242 on 2026-11-04.\nOwner: alex.moreau."
        if calls == 2 and broken == "facts":
            text = "A launch was planned."
        if family == "openai":
            data = {
                "id": f"r{calls}",
                "created_at": 0,
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "call_id": f"c{calls}",
                        "name": "return_result",
                        "arguments": json.dumps({"result": text}),
                    }
                ],
                "usage": {"input_tokens": 8000, "output_tokens": 20, "total_tokens": 8020},
            }
        else:
            data = {
                "id": f"r{calls}",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [
                    {
                        "type": "tool_use",
                        "id": f"c{calls}",
                        "name": "return_result",
                        "input": {"result": text},
                    }
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 8000, "output_tokens": 20},
            }
        return httpx.Response(200, json=data, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    if broken == "fork":

        async def no_fork(self, ctx, nxt):
            return await nxt(ctx)

        monkeypatch.setattr(TokenBudgetSummarizer, "_fork_after_call", no_fork)
    elif broken == "collapse":
        monkeypatch.setattr(TokenBudgetSummarizer, "_handle_before_turn", lambda *_: None)
    cls = ResponsesClient if family == "openai" else CompletionClient
    model = "openai/test" if family == "openai" else "anthropic/claude-sonnet-4-5"
    limit = {"max_output_tokens": 2048} if family == "openai" else {"max_tokens": 2048}
    async with cls(
        model, api_key="test", api_base="https://provider.test", cache_breakpoint=family, **limit
    ) as client:
        if broken:
            message = {
                "fork": "did not fork",
                "collapse": "did not apply",
                "facts": "lost a key fact",
            }[broken]
            with pytest.raises(AssertionError, match=message):
                await exercise_summarization(client, family, monkeypatch)
        else:
            assert (await exercise_summarization(client, family, monkeypatch))["applied"] is True
