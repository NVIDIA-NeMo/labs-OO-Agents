# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in Hub acceptance probes: three requests, capped at 256 output tokens each."""

import json
import os
from pathlib import Path

import httpx
import pytest
import yaml

from nooa.unifiedllm import RetryConfig, get_llm_client

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("NOOA_RUN_REASONING_LEVELS_LIVE") != "1", reason="opt-in paid Hub test"
    ),
]
MODELS = yaml.safe_load(
    (Path(__file__).resolve().parents[2] / "examples/reasoning_levels/llm_config.yaml").read_text()
)["models"]


@pytest.mark.parametrize("alias", MODELS)
async def test_low_effort_on_hub(alias, monkeypatch):
    from nooa.secrets import load_secrets_into_env
    from nooa.unifiedllm import registry

    load_secrets_into_env()
    key = os.getenv("NVIDIA_INFERENCE_API_KEY") or os.getenv("NVIDIA_INTERNAL_API_KEY")
    if not key:
        pytest.fail("Set NVIDIA_INFERENCE_API_KEY to run Hub probes")
    monkeypatch.setattr(registry, "ensure_loaded", lambda: None)
    monkeypatch.setattr(registry, "MODELS", MODELS)
    settings = MODELS[alias]["reasoning_levels"]["low"]
    sent = []
    original_send = httpx.AsyncClient.send

    async def send(client, request, **kwargs):
        if request.method == "POST" and request.url.host == "inference-api.nvidia.com":
            body = json.loads(request.content)
            sent.append({field: body.get(field) for field in settings})
        return await original_send(client, request, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    cap = (
        {"max_output_tokens": 256}
        if MODELS[alias].get("client_type") == "responses"
        else {"max_tokens": 256}
    )
    async with get_llm_client(
        alias,
        api_key=key,
        drop_params=False,
        num_retries=0,
        retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
        **cap,
    ) as client:
        response = await client.acall(
            [{"role": "user", "content": "Compute 17 times 19. Reply with only the number."}],
            reasoning_level="low",
        )
    assert sent == [settings]  # Exactly one attempt, with the declared wire fields.
    assert response.usage.input_tokens > 0
    assert response.finish_reason != "error"
    print(
        json.dumps(
            {
                "model": alias,
                "level": "low",
                "usage": response.usage.model_dump(),
                "finish_reason": response.finish_reason,
            }
        )
    )
