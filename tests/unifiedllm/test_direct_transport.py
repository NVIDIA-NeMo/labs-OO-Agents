# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The direct path uses real SDK serialization against a mocked HTTP server."""

import json
import subprocess
import sys

import httpx
import pytest

from nooa.llm_types import LLMResponse
from nooa.unifiedllm import CompletionClient, ResponsesClient, RetryConfig

NO_RETRY = RetryConfig(max_retries=0, rate_limit_extra_retries=0)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Test escaped its mocked HTTP transport")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", forbidden)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", forbidden)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
@pytest.mark.asyncio
async def test_sdk_round_trip(style, asynchronous, monkeypatch):
    bodies = []

    def respond(request):
        body = json.loads(request.content)
        bodies.append(body)
        assert body["model"] == "test-model"
        assert "transport" not in body
        assert "api_style" not in body
        if style == "responses":
            assert request.url.path == "/v1/responses"
            result = {
                "id": "r1",
                "object": "response",
                "created_at": 1,
                "model": "test-model",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "id": "m1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "42", "annotations": []}],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            }
        elif style == "anthropic":
            assert request.url.path == "/v1/messages"
            result = {
                "id": "m1",
                "type": "message",
                "role": "assistant",
                "model": "test-model",
                "content": [{"type": "text", "text": "42"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 2},
            }
        else:
            assert request.url.path == "/v1/chat/completions"
            result = {
                "id": "c1",
                "object": "chat.completion",
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "42",
                            "reasoning_content": "calculation",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            }
        return httpx.Response(200, json=result)

    # Both SDKs must use NOOA's owned HTTP clients, never a default network pool.
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, req: respond(req))

    async def handle_async(self, req):
        return respond(req)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle_async)
    cls = ResponsesClient if style == "responses" else CompletionClient
    client = cls(
        "test-model",
        transport="direct",
        api_style=style,
        api_base="https://models.example/v1",
        api_key="test-key",
        max_tokens=100,
        retry_config=NO_RETRY,
    )
    try:
        history = [{"role": "user", "content": "Question"}]
        response = await client.acall(history) if asynchronous else client.call(history)
        assert isinstance(response, LLMResponse)
        assert response.content == "42"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 2
        assert len(bodies) == 1
    finally:
        await client.aclose()


def test_direct_import_does_not_import_litellm():
    script = """
import sys
from nooa.unifiedllm import CompletionClient
c = CompletionClient('test', transport='direct', api_style='chat', api_key='test')
assert 'litellm' not in sys.modules
c.close()
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_invalid_transport_rejected():
    with pytest.raises(ValueError, match="transport"):
        CompletionClient("test", transport="typo")


@pytest.mark.parametrize("key", ["transport", "api_style", "replay_vendor"])
def test_transport_settings_are_constructor_only(key):
    with CompletionClient("test", transport="direct", api_key="test") as client:
        with pytest.raises(ValueError, match="client setting"):
            client.call([], **{key: "direct"})
        with pytest.raises(ValueError, match="client setting"):
            client.call([], extra_body={key: "direct"})


def test_soak_override_infers_anthropic_from_existing_route_prefix(monkeypatch):
    monkeypatch.setenv("NOOA_LLM_TRANSPORT", "direct")
    with CompletionClient("anthropic/test", api_key="test") as client:
        assert client._direct.api_style == "anthropic"


@pytest.mark.parametrize(
    "style,model",
    [
        ("chat", "openai/vendor/model"),
        ("responses", "openai/vendor/model"),
        ("anthropic", "anthropic/vendor/model"),
        ("chat", "deepseek/deepseek-reasoner"),
        ("chat", "nvidia_nim/vendor/model"),
        ("chat", "openrouter/vendor/model"),
        ("chat", "hosted_vllm/vendor/model"),
    ],
)
def test_scope_matches_legacy_and_readable_reasoning_is_retained(style, model):
    from nooa.unifiedllm.chat_parts import capture_chat_parts, project_chat_turn
    from nooa.unifiedllm.replay_state import replay_scope

    params = {"api_base": "https://models.example/v1", "api_key": "test"}
    cls = ResponsesClient if style == "responses" else CompletionClient
    with cls(model, transport="direct", api_style=style, **params) as client:
        api = "responses" if style == "responses" else "chat"
        scope = client._replay_scope(model, api, params)
        assert scope == replay_scope(model, api, params)
        assert scope is not None
        if api == "chat":
            parts = capture_chat_parts({"content": "answer", "reasoning_content": "thought"}, scope)
            turn = LLMResponse(parts=parts, replay_scope=scope)
            restored = LLMResponse.model_validate_json(turn.model_dump_json())
            assert project_chat_turn(restored, scope)[0]["reasoning_content"] == "thought"
            assert "reasoning_content" not in project_chat_turn(restored, scope + "-other")[0]


def test_shipped_model_scopes_match():
    from pathlib import Path

    import yaml

    from nooa.unifiedllm.direct import DirectTransport
    from nooa.unifiedllm.http_config import HttpConfig
    from nooa.unifiedllm.replay_state import replay_scope, scope_for_route

    root = Path(__file__).resolve().parents[2]
    checked = 0
    for path in (root / "examples").rglob("llm_config.yaml"):
        for entry in yaml.safe_load(path.read_text())["models"].values():
            model = entry["model_name"]
            style = "responses" if entry.get("client_type") == "responses" else "chat"
            params = {"api_base": "https://models.example/v1", "api_key": "test"}
            expected = replay_scope(model, style, params)
            if expected is None:
                continue
            with_transport = DirectTransport(model, style, None, params, HttpConfig())
            try:
                wire, vendor = with_transport.route(model, params)
                assert scope_for_route(wire, vendor, style) == expected
                checked += 1
            finally:
                with_transport.close()
    assert checked >= 3


def test_direct_error_never_falls_back_and_sdk_retries_are_disabled(monkeypatch):
    import litellm
    from openai import RateLimitError

    sent = []

    def fail(self, request):
        sent.append(request)
        return httpx.Response(429, json={"error": {"message": "busy", "type": "rate_limit"}})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", fail)
    monkeypatch.setattr(litellm, "completion", lambda **_: pytest.fail("legacy fallback"))
    with CompletionClient("test", transport="direct", api_key="test", retry_config=NO_RETRY) as c:
        with pytest.raises(RateLimitError):
            c.call([{"role": "user", "content": "Hello"}])
    assert len(sent) == 1


def test_registry_transport_and_soak_override(monkeypatch):
    from nooa.unifiedllm import registry

    monkeypatch.setattr(registry, "ensure_loaded", lambda: None)
    monkeypatch.setitem(
        registry.MODELS,
        "direct-test",
        {
            "model_name": "anthropic/test",
            "transport": "direct",
            "api_style": "anthropic",
            "replay_vendor": "anthropic",
            "context_window": 12345,
        },
    )
    with registry.get_llm_client("direct-test", api_key="test") as client:
        assert client.transport == "direct"
        assert client._direct.api_style == "anthropic"
        assert client.context_window == 12345
        assert "api_style" not in client.config
    monkeypatch.setenv("NOOA_LLM_TRANSPORT", "litellm")
    with registry.get_llm_client("direct-test", api_key="test") as client:
        assert client.transport == "litellm"


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
def test_output_schema_reaches_sdk(style, monkeypatch):
    from pydantic import BaseModel

    from nooa.unifiedllm.direct import DirectTransport
    from nooa.unifiedllm.http_config import HttpConfig

    class Answer(BaseModel):
        answer: int

    direct = DirectTransport("test", style, None, {}, HttpConfig())
    try:
        params = {"model": "test", "api_key": "test", "max_tokens": 200}
        if style == "responses":
            params.update(input=[], text_format=Answer)
        else:
            params.update(messages=[], response_format=Answer)
        _, body = direct._request(params, asynchronous=False)
        if style == "anthropic":
            schema = body["output_config"]["format"]["schema"]
        elif style == "responses":
            schema = body["text"]["format"]["schema"]
        else:
            schema = body["response_format"]["json_schema"]["schema"]
        assert schema["properties"]["answer"]["type"] == "integer"
    finally:
        direct.close()
