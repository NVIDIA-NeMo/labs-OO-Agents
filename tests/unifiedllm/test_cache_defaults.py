# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Default registry clients must honor the default renderer's cache boundary."""

import copy
import json

import httpx
import pytest

from nooa.context_blocks.formatter import OpenAIProviderFormatter, ResponsesProviderFormatter
from nooa.context_blocks.models import BlockMetadata, ResolvedBlock, Role
from nooa.context_blocks.renderer import render_context
from nooa.context_blocks.renderers.cached import CachedBlockFormatter
from nooa.unifiedllm import CacheBoundary, ResponsesClient, RetryConfig, registry


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("style", ["responses", "anthropic", "chat"])
@pytest.mark.parametrize("mode", ["default", "no-boundary", "disabled"])
async def test_registry_default_cache_on_wire(monkeypatch, style, mode, asynchronous):
    """Exercise renderer -> registry client -> SDK -> HTTP, with no cache opt-in."""
    entry = {
        "model_name": "anthropic/claude-sonnet-4-5" if style == "anthropic" else "openai/gpt-4o",
        "client_type": "responses" if style == "responses" else "completion",
        "api_base": "https://models.example"
        if style == "anthropic"
        else "https://models.example/v1",
        "max_tokens": 100,
    }
    monkeypatch.setattr(registry, "MODELS", {"cache-test": entry})
    monkeypatch.setattr(registry, "ensure_loaded", lambda: None)
    bodies = []

    def respond(request):
        assert request.url.host == "models.example"
        bodies.append(json.loads(request.content))
        if style == "responses":
            assert request.url.path == "/v1/responses"
            data = {
                "id": "r",
                "object": "response",
                "created_at": 1,
                "model": "gpt-4o",
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": 10, "output_tokens": 0, "total_tokens": 10},
            }
        elif style == "anthropic":
            assert request.url.path == "/v1/messages"
            data = {
                "id": "m",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [{"type": "text", "text": "OK"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 1},
            }
        else:
            assert request.url.path == "/v1/chat/completions"
            data = {
                "id": "c",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            }
        return httpx.Response(200, json=data, request=request)

    async def send_async(self, request, **kwargs):
        return respond(request)

    monkeypatch.setattr(httpx.Client, "send", lambda self, request, **kwargs: respond(request))
    monkeypatch.setattr(httpx.AsyncClient, "send", send_async)
    messages = render_context(
        [
            ResolvedBlock(
                key="instructions",
                content="Stable instructions",
                role=Role.SYSTEM,
                metadata=BlockMetadata(static=True),
            ),
            ResolvedBlock(
                key="live",
                content="Volatile suffix",
                role=Role.SYSTEM,
                metadata=BlockMetadata(static=False, user_block=True),
            ),
        ],
        block_formatter=CachedBlockFormatter(),
        provider_formatter=ResponsesProviderFormatter()
        if style == "responses"
        else OpenAIProviderFormatter(),
    ).output
    assert sum(isinstance(m, CacheBoundary) for m in messages) == 1
    if mode == "no-boundary":
        messages = [m for m in messages if not isinstance(m, CacheBoundary)]
    before = copy.deepcopy(messages)
    overrides = {"cache_breakpoint": None} if mode == "disabled" else {}
    async with registry.get_llm_client(
        "cache-test",
        api_key="test",
        retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
        **overrides,
    ) as client:
        if asynchronous:
            await client.acall(messages)
        else:
            client.call(messages)
    assert messages == before
    assert "cache_breakpoint" not in entry
    assert len(bodies) == 1
    body = bodies[0]
    encoded = json.dumps(body)
    assert "nooa_cache_boundary" not in encoded
    explicit = style == "responses" and mode == "default"
    assert ("prompt_cache_options" in body) is explicit
    assert encoded.count('"prompt_cache_breakpoint"') == int(explicit)
    if explicit:
        assert body["prompt_cache_options"]["mode"] == "explicit"
        marked = body["input"][0]["content"][-1]
        assert "Stable instructions" in marked["text"]
        assert marked["prompt_cache_breakpoint"] == {"mode": "explicit"}
    # Completion's existing no-boundary behavior marks only leading instructions.
    assert ('"cache_control"' in encoded) is (style == "anthropic" and mode != "disabled")
    assert "Volatile suffix" in encoded


@pytest.mark.parametrize(
    "messages",
    [
        [CacheBoundary(), {"role": "user", "content": "live"}],
        [
            {"role": "assistant", "content": "answer"},
            CacheBoundary(),
            {"role": "user", "content": "live"},
        ],
    ],
)
def test_auto_without_eligible_prefix_stays_implicit(messages, caplog):
    with ResponsesClient("openai/gpt-4o", api_key="test") as client:
        wire, instructions, explicit = client._prepare_cache_boundary(messages, responses=True)
    assert wire == [m for m in messages if not isinstance(m, CacheBoundary)]
    assert instructions is None
    assert not explicit
    assert not caplog.text
