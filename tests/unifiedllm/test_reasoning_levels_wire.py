# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the installed SDK, not just parameters passed to LiteLLM."""

import json
import runpy
from pathlib import Path

import httpx
import litellm
import pytest
import yaml

from nooa.unifiedllm import RetryConfig, get_llm_client

CONFIG_PATH = Path(__file__).resolve().parents[2] / "examples/reasoning_levels/llm_config.yaml"
MODELS = yaml.safe_load(CONFIG_PATH.read_text())["models"]


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
@pytest.mark.parametrize("selection", ["default", "level", "override", "alias", "extra"])
@pytest.mark.parametrize("asynchronous", [True, False])
async def test_context_reserve_matches_serialized_reply_cap(
    style, selection, asynchronous, monkeypatch
):
    from nooa.unifiedllm import CompletionClient, ResponsesClient

    bodies = []

    def send_sync(http_client, request, **kwargs):
        bodies.append(json.loads(request.content))
        alias = {"chat": "example", "responses": "gpt-5.6-sol", "anthropic": "claude-sonnet-5"}[
            style
        ]
        return httpx.Response(200, json=_reply(alias), request=request)

    async def send(http_client, request, **kwargs):
        return send_sync(http_client, request, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    monkeypatch.setattr(httpx.Client, "send", send_sync)
    monkeypatch.setattr(litellm, "drop_params", False)
    cls = ResponsesClient if style == "responses" else CompletionClient
    model = "anthropic/claude-sonnet-4-5" if style == "anthropic" else "openai/example"
    cap_key = "max_output_tokens" if style == "responses" else "max_tokens"
    overrides = {
        "default": {},
        "level": {"reasoning_level": "high"},
        "override": {"max_tokens": 16_000},
        "alias": {cap_key: 24_000},
        "extra": {"extra_body": {cap_key: 20_000}},
    }[selection]
    async with cls(
        model,
        api_base="https://gateway.example.com/v1",
        api_key="test",
        context_window=128_000,
        max_tokens=64_000,
        reasoning_levels={"high": {cap_key: 48_000}},
        retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
    ) as llm:
        limits = llm.get_context_limits(overrides)
        messages = [{"role": "user", "content": "hello"}]
        if asynchronous:
            await llm.acall(messages, **overrides)
        else:
            llm.call(messages, **overrides)
    assert len(bodies) == 1
    caps = [
        v
        for k, v in bodies[0].items()
        if k in {"max_tokens", "max_completion_tokens", "max_output_tokens"}
    ]
    assert caps == [limits.reserved_output_tokens]
    assert limits.usable_input_tokens == 128_000 - caps[0]


@pytest.mark.parametrize("alias", MODELS)
async def test_live_probe_uses_registry_configuration_without_route_assumptions(alias, monkeypatch):
    from nooa import secrets
    from nooa.unifiedllm import registry

    monkeypatch.setattr(secrets, "load_secrets_into_env", lambda: None)
    monkeypatch.setenv("MODEL_API_KEY", "test")
    monkeypatch.setattr(registry, "ensure_loaded", lambda: None)
    monkeypatch.setattr(registry, "MODELS", MODELS)
    requests = []

    async def send(http_client, request, **kwargs):
        requests.append(request)
        return httpx.Response(200, json=_reply(alias), request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    probe = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "integration/test_reasoning_levels_live.py")
    )
    await probe["test_low_effort_on_configured_route"](alias, monkeypatch)
    assert len(requests) == 1
    assert requests[0].url.host == "gateway.example.com"


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
    # Other suites enable LiteLLM's process-global parameter dropping. It
    # overrides even per-call False; this test verifies an unsuppressed request.
    monkeypatch.setattr(litellm, "drop_params", False)
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


async def test_selected_level_replaces_extra_body_default_on_wire(monkeypatch):
    from nooa.unifiedllm import registry

    monkeypatch.setattr(registry, "ensure_loaded", lambda: None)
    monkeypatch.setattr(registry, "MODELS", MODELS)
    monkeypatch.setattr(litellm, "drop_params", False)
    bodies = []

    async def send(http_client, request, **kwargs):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_reply("gpt-5.6-sol"), request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    inherited = {"reasoning": {"effort": "high"}, "metadata": {"test": "kept"}}
    async with get_llm_client(
        "gpt-5.6-sol",
        api_key="test",
        extra_body=inherited,
        retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
    ) as client:
        await client.acall([{"role": "user", "content": "hello"}], reasoning_level="low")
    assert len(bodies) == 1
    assert bodies[0]["reasoning"] == {"effort": "low"}
    assert bodies[0]["metadata"] == {"test": "kept"}
    assert inherited["reasoning"] == {"effort": "high"}
