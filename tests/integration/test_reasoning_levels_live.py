# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in registry acceptance probes, capped at 256 output tokens per alias."""

import json
import os

import httpx
import litellm
import pytest

from nooa.unifiedllm import RetryConfig, get_llm_client

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("NOOA_RUN_REASONING_LEVELS_LIVE") != "1", reason="opt-in paid provider test"
    ),
]
ALIASES = [
    alias.strip()
    for alias in os.getenv("NOOA_REASONING_TEST_MODELS", "").split(",")
    if alias.strip()
]


@pytest.mark.parametrize("alias", ALIASES)
async def test_low_effort_on_configured_route(alias, monkeypatch):
    from nooa.secrets import load_secrets_into_env
    from nooa.unifiedllm import registry

    load_secrets_into_env()
    config = registry.get_registry_config(alias)
    if not config:
        pytest.skip(f"Registry alias {alias!r} is not configured")
    monkeypatch.setattr(litellm, "drop_params", False)
    settings = config["reasoning_levels"]["low"]
    sent = []
    original_send = httpx.AsyncClient.send

    async def send(client, request, **kwargs):
        if request.method == "POST":
            body = json.loads(request.content)
            sent.append({field: body.get(field) for field in settings})
        return await original_send(client, request, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    cap = (
        {"max_output_tokens": 256}
        if config.get("client_type") == "responses"
        else {"max_tokens": 256}
    )
    async with get_llm_client(
        alias,
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
