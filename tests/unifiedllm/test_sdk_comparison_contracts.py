# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native parameter semantics identified by the cross-library comparison."""

import json

import httpx
import pytest

from nooa.unifiedllm import CompletionClient, ResponsesClient, RetryConfig
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
    "endpoint,field,expected",
    [
        (None, "auto", "max_completion_tokens"),
        ("https://models.example/v1", "auto", "max_tokens"),
        ("https://models.example/v1", "max_completion_tokens", "max_completion_tokens"),
        ("https://models.example/v1", "max_tokens", "max_tokens"),
    ],
)
async def test_chat_cap_on_wire(wire, transport, asynchronous, endpoint, field, expected):
    bodies, _ = wire
    model = "openai/o3" if endpoint is None else "openai/deployment"
    async with client(transport, model, api_base=endpoint, chat_max_tokens_field=field) as llm:
        await call(llm, asynchronous)
        assert llm.config["max_tokens"] == 32
    assert len(bodies) == 1
    assert bodies[0][expected] == 32
    assert len({"max_tokens", "max_completion_tokens"} & bodies[0].keys()) == 1
    assert "chat_max_tokens_field" not in bodies[0]


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
    "patch,pattern",
    [
        ({"max_completion_tokens": 10}, "either max_tokens or max_completion_tokens"),
        ({"extra_body": {"max_completion_tokens": 10}}, "not in extra_body"),
        ({"chat_max_tokens_field": "max_tokens"}, "client settings"),
    ],
)
async def test_conflicting_cap_never_reaches_http(wire, patch, pattern):
    bodies, _ = wire
    async with client("direct", "openai/deployment") as llm:
        with pytest.raises(ValueError, match=pattern):
            await call(llm, True, **patch)
    assert bodies == []


@pytest.mark.parametrize("transport", ["direct", "litellm"])
@pytest.mark.parametrize("invalid", ["bad", None, []])
def test_cap_setting_is_validated(transport, invalid):
    with pytest.raises(ValueError, match="chat_max_tokens_field"):
        client(transport, "openai/deployment", chat_max_tokens_field=invalid)
    with pytest.raises(ValueError, match="only to Chat"):
        client(transport, chat_max_tokens_field="max_completion_tokens")
    with pytest.raises(ValueError, match="only to Chat"):
        ResponsesClient(
            "openai/test", transport=transport, api_key="test", chat_max_tokens_field="max_tokens"
        )


def test_registry_preserves_cap_mapping(monkeypatch):
    from nooa.unifiedllm import registry

    monkeypatch.setattr(registry, "ensure_loaded", lambda: None)
    entry = {
        "model_name": "openai/deployment",
        "api_base": "https://models.example/v1",
        "api_key": "test",
        "transport": "direct",
        "max_tokens": 32,
        "chat_max_tokens_field": "max_completion_tokens",
    }
    monkeypatch.setattr(registry, "MODELS", {"test-cap": entry})
    with registry.get_llm_client("test-cap") as llm:
        assert llm.chat_max_tokens_field == "max_completion_tokens"
        assert llm.config["max_tokens"] == 32
