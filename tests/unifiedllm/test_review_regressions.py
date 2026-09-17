# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression evidence for the integrated direct-transport review."""

import asyncio
import json
from copy import deepcopy
from types import SimpleNamespace

import httpx
import pytest

from nooa.llm_types import CacheBoundary
from nooa.unifiedllm import CompletionClient, replay_state
from nooa.unifiedllm._legacy import preserve_readable_reasoning
from nooa.unifiedllm.direct import DirectTransport, anthropic_request
from nooa.unifiedllm.http_config import HttpConfig
from nooa.unifiedllm.registry import client_from_config


@pytest.mark.parametrize("transport", ["litellm", "direct"])
async def test_anthropic_cache_uses_actual_route(transport):
    async with CompletionClient(
        "anthropic/claude-test",
        transport=transport,
        api_key="test",
        cache_breakpoint="auto",
    ) as client:
        result = client._prepare_cache_boundary(
            [{"role": "user", "content": "cached"}, CacheBoundary()], responses=False
        )
        assert "cache_control" in str(result)


async def test_alias_replacement_clears_inherited_wire_style_and_window():
    entry = {
        "model_name": "anthropic/old",
        "transport": "direct",
        "api_style": "anthropic",
        "replay_vendor": "anthropic",
        "context_window": 99999,
        "api_base": "https://old.example/v1",
    }
    async with client_from_config(
        "old", entry, model="openai/new", api_base="https://new.example/v1", api_key="test"
    ) as client:
        assert client._direct.api_style == "chat"
        assert client.replay_vendor is None
        assert client.context_window != 99999


@pytest.mark.parametrize(
    "model", ["bedrock/test", "vertex_ai/test", "groq/test", "mistral/test", "company/test"]
)
def test_unresolved_slash_route_never_defaults_to_openai(model):
    with pytest.raises(ValueError, match="api_base"):
        DirectTransport(model, "chat", None, {}, HttpConfig())


def test_native_azure_is_not_a_generic_openai_route():
    with pytest.raises(ValueError, match="Azure"):
        DirectTransport(
            "azure/deployment",
            "chat",
            None,
            {"api_base": "https://resource.openai.azure.com"},
            HttpConfig(),
        )


def test_unresolvable_readable_reasoning_does_not_raise(monkeypatch):
    from nooa.unifiedllm import _legacy

    def fail(**kwargs):
        raise ValueError("cannot resolve")

    monkeypatch.setattr(_legacy.litellm, "get_llm_provider", fail)
    params = {
        "model": "unknown",
        "messages": [{"role": "assistant", "content": "answer", "reasoning_content": "reason"}],
    }
    original = deepcopy(params)
    assert preserve_readable_reasoning(params) == params
    assert params == original


@pytest.mark.parametrize("provider", ["vertex_ai", "bedrock"])
def test_unverified_native_routes_have_no_replay_scope(monkeypatch, provider):
    monkeypatch.setattr(
        replay_state.litellm, "get_llm_provider", lambda **kw: ("test", provider, None, None)
    )
    assert replay_state.replay_scope(f"{provider}/test", "chat", {}) is None


@pytest.mark.parametrize("role", ["user", "assistant"])
def test_anthropic_empty_turn_is_rejected_not_removed(role):
    with pytest.raises(ValueError, match="empty"):
        anthropic_request({"messages": [{"role": role, "content": ""}], "max_tokens": 10})


async def test_none_retries_is_unset():
    transport = DirectTransport("openai/test", "chat", None, {}, HttpConfig())
    try:
        _, body = transport._request(
            {"model": "openai/test", "api_key": "test", "num_retries": None}, asynchronous=True
        )
        assert "num_retries" not in body
    finally:
        await transport.aclose()


async def test_empty_transport_environment_is_unset(monkeypatch):
    monkeypatch.setenv("NOOA_LLM_TRANSPORT", "")
    async with CompletionClient("openai/test", transport="direct", api_key="test") as client:
        assert client.transport == "direct"


async def test_anthropic_bare_model_tool_history_gets_dummy_tool(monkeypatch):
    sent = []

    async def respond(self, request):
        import json

        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "model": "test",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        )

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", respond)
    async with CompletionClient(
        "anthropic/claude-test", transport="direct", api_key="test", max_tokens=100
    ) as client:
        await client.acall(
            [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call", "content": "result"},
            ]
        )
    assert sent[0].get("tools")


async def test_direct_context_metadata_keeps_known_window(monkeypatch):
    from nooa.unifiedllm import unifiedllm

    monkeypatch.setattr(
        unifiedllm.litellm, "get_model_info", lambda model: {"max_input_tokens": 128000}
    )
    async with CompletionClient(
        "openai/test", transport="direct", api_key="test", max_tokens=32000
    ) as client:
        assert client.get_context_limits().usable_input_tokens == 96000


def test_anthropic_thinking_is_visible_to_empty_reply_guard():
    from anthropic.types import Message

    from nooa.unifiedllm.direct import anthropic_response
    from nooa.unifiedllm.unifiedllm import _extract_reasoning_and_usage

    response = anthropic_response(
        Message.model_validate(
            {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "model": "test",
                "content": [{"type": "thinking", "thinking": "reason", "signature": "synthetic"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 1},
            }
        )
    )
    assert _extract_reasoning_and_usage(response)[0] == "reason"


async def test_sync_override_does_not_allocate_unused_async_pool(monkeypatch):
    from nooa.unifiedllm import unifiedllm

    async with CompletionClient("openai/test", api_key="test") as client:
        monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: pytest.fail("unused async pool"))
        reply = unifiedllm.litellm.ModelResponse()
        monkeypatch.setattr(unifiedllm.litellm, "completion", lambda **kw: reply)
        assert client._send({"model": "openai/other", "api_key": "test"}) is reply


async def test_cancelled_override_keeps_pool_until_provider_finishes(monkeypatch):
    from nooa.unifiedllm import unifiedllm

    entered, release, closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    pool = SimpleNamespace(async_client=object())

    async def close():
        closed.set()

    pool.aclose = close

    async def completion(**kw):
        entered.set()
        await release.wait()
        assert not closed.is_set()
        return "reply"

    async with CompletionClient("openai/test", api_key="test") as client:
        monkeypatch.setattr(unifiedllm._ClientHttp, "for_completion", lambda *a, **kw: pool)
        monkeypatch.setattr(unifiedllm.litellm, "acompletion", completion)
        task = asyncio.create_task(client._asend({"model": "openai/other"}))
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        try:
            assert not closed.is_set()
        finally:
            release.set()
            await asyncio.wait_for(closed.wait(), 2)


def test_gemini_invocation_metadata_does_not_duplicate_history():
    from nooa.tracing import _llm_hooks

    attrs = {}
    span = SimpleNamespace(set_attribute=lambda k, v: attrs.__setitem__(k, v))
    token = _llm_hooks._active.set((span, {"model": "test", "input_recorded": True}))
    try:
        _llm_hooks.capture_request(
            httpx.Request(
                "POST",
                "https://example.com",
                json={
                    "contents": [{"role": "user", "parts": [{"text": "history"}]}],
                    "system_instruction": {"parts": [{"text": "system"}]},
                    "systemInstruction": {"parts": [{"text": "system"}]},
                    "temperature": 0.5,
                },
            )
        )
    finally:
        _llm_hooks._active.reset(token)
    assert json.loads(attrs["llm.invocation_parameters"]) == {"temperature": 0.5}


@pytest.mark.parametrize("cost", [None, 0.0, 0.2])
def test_unknown_cost_is_not_reported_as_free(monkeypatch, cost, caplog):
    from contextlib import contextmanager

    from nooa.llm_types import LLMResponse, LLMUsage
    from nooa.tracing import _llm_hooks

    attrs = {}
    span = SimpleNamespace(set_attribute=lambda k, v: attrs.__setitem__(k, v))

    @contextmanager
    def start(*a, **kw):
        yield span

    monkeypatch.setattr(_llm_hooks, "_tracer", lambda: SimpleNamespace(start_as_current_span=start))
    monkeypatch.setattr(_llm_hooks, "callbacks", [])
    usage = LLMUsage.from_provider(
        {"input_tokens": 10, **({"cost": cost} if cost is not None else {})}
    )
    with _llm_hooks._call("test") as finish:
        finish(LLMResponse(content="ok", usage=usage))
    assert attrs.get("llm.cost.total") == cost
    assert attrs["llm.input.capture"] == "unavailable"
    assert "wire input was not captured" in caplog.text


@pytest.mark.parametrize(
    "original,replacement", [("anthropic/old", "openai/new"), ("openai/old", "anthropic/new")]
)
async def test_cross_format_call_override_is_rejected_before_http(
    monkeypatch, original, replacement
):
    def forbidden(*a, **kw):
        pytest.fail("must reject before SDK dispatch")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)
    async with CompletionClient(
        original, transport="direct", api_key="test", max_tokens=100
    ) as client:
        with pytest.raises(ValueError, match="new client"):
            await client.acall([{"role": "user", "content": "test"}], model=replacement)
