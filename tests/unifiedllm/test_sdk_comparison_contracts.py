# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native parameter semantics identified by the cross-library comparison."""

import json

import httpx
import pytest

from nooa.unifiedllm import CompletionClient, RetryConfig
from nooa.unifiedllm.errors import UnsupportedStopReasonError


@pytest.fixture
def wire(monkeypatch):
    bodies = []
    state = {"stop": "end_turn"}

    def send(request):
        bodies.append(json.loads(request.content))
        if request.url.path.endswith("/messages"):
            data = {
                "id": "t",
                "type": "message",
                "role": "assistant",
                "model": "claude-test",
                "content": [{"type": "text", "text": "synthetic answer"}],
                "stop_reason": state["stop"],
                "usage": {"input_tokens": 3, "output_tokens": 2},
            }
        else:
            data = {
                "id": "t",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "synthetic answer"},
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
            }
        return httpx.Response(200, json=data, request=request)

    async def asend(self, request):
        return send(request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, request: send(request))
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", asend)
    monkeypatch.delenv("NOOA_LLM_TRANSPORT", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    return bodies, state


def client(transport, model="anthropic/claude-sonnet-4-5", **params):
    return CompletionClient(
        model,
        transport=transport,
        api_key="test",
        max_tokens=32,
        retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
        **params,
    )


async def call(llm, asynchronous, **params):
    messages = [{"role": "user", "content": "synthetic"}]
    return await llm.acall(messages, **params) if asynchronous else llm.call(messages, **params)


@pytest.mark.parametrize("transport", ["litellm", "direct"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stop", ["END", ["END", "STOP"], None])
async def test_stop_sequences_on_wire(wire, transport, asynchronous, stop):
    bodies, _ = wire
    async with client(transport, api_base="https://models.example/v1") as llm:
        await call(llm, asynchronous, stop=stop)
    assert len(bodies) == 1
    assert "stop" not in bodies[0]
    expected = [stop] if isinstance(stop, str) else stop
    assert bodies[0].get("stop_sequences") == expected


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "patch,pattern",
    [
        ({"stop": "END", "stop_sequences": ["OTHER"]}, "either stop or stop_sequences"),
        ({"stop": 4}, "string or list"),
        ({"stop": [4]}, "string or list"),
    ],
)
async def test_invalid_stop_makes_no_request(wire, asynchronous, patch, pattern):
    bodies, _ = wire
    async with client("direct", api_base="https://models.example/v1") as llm:
        with pytest.raises(ValueError, match=pattern):
            await call(llm, asynchronous, **patch)
    assert bodies == []


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "reason,expected",
    [
        ("end_turn", "stop"),
        ("stop_sequence", "stop"),
        ("max_tokens", "length"),
        ("model_context_window_exceeded", "length"),
        ("refusal", "error"),
    ],
)
async def test_native_stop_semantics(wire, asynchronous, reason, expected):
    _, state = wire
    state["stop"] = reason
    async with client("direct", api_base="https://models.example/v1") as llm:
        result = await call(llm, asynchronous)
    assert result.finish_reason == expected
    assert result.raw_response.provider_specific_fields["stop_reason"] == reason


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("reason", ["pause_turn", "unrecognized", None])
async def test_unsupported_continuation_is_not_a_safety_error(wire, asynchronous, reason):
    bodies, state = wire
    state["stop"] = reason
    async with client("direct", api_base="https://models.example/v1") as llm:
        with pytest.raises(UnsupportedStopReasonError, match="stop_reason") as exc:
            await call(llm, asynchronous)
    assert exc.value.stop_reason == reason
    assert len(bodies) == 1


@pytest.mark.parametrize("transport", ["litellm", "direct"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "endpoint,expected",
    [
        (None, "max_completion_tokens"),
        ("https://models.example/v1", "max_tokens"),
    ],
)
async def test_chat_cap_on_wire(wire, transport, asynchronous, endpoint, expected):
    bodies, _ = wire
    model = "openai/o3" if endpoint is None else "openai/deployment"
    async with client(transport, model, api_base=endpoint) as llm:
        await call(llm, asynchronous)
        assert llm.config["max_tokens"] == 32
    assert len(bodies) == 1
    assert bodies[0][expected] == 32
    assert len({"max_tokens", "max_completion_tokens"} & bodies[0].keys()) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_auto_cap_uses_effective_endpoint_per_call(wire, asynchronous, monkeypatch):
    bodies, _ = wire
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env-models.example/v1")
    async with client("direct", "openai/deployment") as llm:
        await call(llm, asynchronous)
        await call(llm, asynchronous, base_url="https://api.openai.com/v1")
    assert bodies[0]["max_tokens"] == 32
    assert bodies[1]["max_completion_tokens"] == 32


@pytest.mark.parametrize(
    "patch",
    [
        {"max_tokens": 20, "max_completion_tokens": 10},
        {"max_tokens": 20, "extra_body": {"max_completion_tokens": 10}},
    ],
)
async def test_conflicting_cap_never_reaches_http(wire, patch):
    bodies, _ = wire
    async with client("direct", "openai/deployment") as llm:
        with pytest.raises(ValueError, match="only one reply token limit field"):
            await call(llm, True, **patch)
    assert bodies == []


@pytest.mark.parametrize(
    "patch", [{"max_completion_tokens": 10}, {"extra_body": {"max_completion_tokens": 10}}]
)
async def test_call_cap_replaces_inherited_alias_on_direct_wire(wire, patch):
    bodies, _ = wire
    async with client("direct", "openai/deployment") as llm:
        await call(llm, True, **patch)
    assert len(bodies) == 1
    assert bodies[0]["max_completion_tokens"] == 10
    assert "max_tokens" not in bodies[0]


@pytest.mark.parametrize("transport", ["direct", "litellm"])
@pytest.mark.parametrize("native", [False, True])
async def test_existing_registry_entry_supplies_reply_budget(wire, monkeypatch, transport, native):
    from nooa.unifiedllm import registry

    monkeypatch.setattr(registry, "ensure_loaded", lambda: None)
    monkeypatch.setenv("TEST_CAP_API_KEY", "test")
    entry = {
        "model_name": "openai/o3" if native else "openai/deployment",
        "api_key_env": "TEST_CAP_API_KEY",
        "transport": transport,
        "max_tokens": 32,
        "retry_config": False,
    }
    if not native:
        entry["api_base"] = "https://models.example/v1"
    monkeypatch.setattr(registry, "MODELS", {"test-cap": entry})
    async with registry.get_llm_client("test-cap") as llm:
        assert llm.config["max_tokens"] == 32
        await call(llm, True)
    bodies, _ = wire
    field = "max_completion_tokens" if native else "max_tokens"
    assert bodies[0][field] == 32
    assert len({"max_tokens", "max_completion_tokens"} & bodies[0].keys()) == 1
