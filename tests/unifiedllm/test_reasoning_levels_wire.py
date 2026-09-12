# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the installed SDK, not just parameters passed to LiteLLM."""

import json
from pathlib import Path

import httpx
import pytest
import yaml

from nooa.unifiedllm import RetryConfig, get_llm_client

CONFIG_PATH = Path(__file__).resolve().parents[2] / "examples/reasoning_levels/llm_config.yaml"
MODELS = yaml.safe_load(CONFIG_PATH.read_text())["models"]


def _reply(alias):
    if alias == "gpt-5.6-sol":
        return {
            "id": "resp_test",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "gpt-5.6-sol",
            "parallel_tool_calls": False,
            "store": False,
            "tools": [],
            "output": [
                {
                    "id": "msg_test",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
        }
    if alias == "claude-sonnet-5":
        return {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": alias,
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 1},
        }
    return {
        "id": "chat_test",
        "object": "chat.completion",
        "created": 0,
        "model": alias,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    }


@pytest.mark.parametrize(
    "alias,level",
    [(alias, level) for alias, model in MODELS.items() for level in model["reasoning_levels"]],
)
async def test_declared_settings_survive_the_sdk(alias, level, monkeypatch):
    from nooa.unifiedllm import registry

    monkeypatch.setattr(registry, "ensure_loaded", lambda: None)
    monkeypatch.setattr(registry, "MODELS", MODELS)
    bodies = []

    async def send(http_client, request, **kwargs):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_reply(alias), request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    async with get_llm_client(
        alias,
        api_key="test",
        drop_params=False,
        num_retries=0,
        retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
    ) as client:
        result = await client.acall([{"role": "user", "content": "hello"}], reasoning_level=level)
    assert result.content == "ok"
    assert len(bodies) == 1
    body = bodies[0]
    for key, value in MODELS[alias]["reasoning_levels"][level].items():
        assert body[key] == value
    assert not {"reasoning_levels", "reasoning_default", "reasoning_level"} & body.keys()
