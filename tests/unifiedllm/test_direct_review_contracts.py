# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Review regressions exercised below the real SDK serializers, without network."""

import json

import httpx
import pytest
from pydantic import BaseModel

from nooa.unifiedllm import CompletionClient, ResponsesClient, RetryConfig, Tool
from nooa.unifiedllm.direct import DirectTransport, anthropic_request
from nooa.unifiedllm.http_config import HttpConfig


@pytest.mark.parametrize(
    "style,config,pattern",
    [
        ("invalid", {}, "api_style"),
        ("chat", {"client": object()}, "client/custom_llm_provider"),
        ("chat", {"custom_llm_provider": "openai"}, "client/custom_llm_provider"),
    ],
)
def test_direct_constructor_guards_before_http(style, config, pattern):
    with pytest.raises(ValueError, match=pattern):
        DirectTransport("openai/test", style, None, config, HttpConfig())


def test_direct_alias_requires_endpoint_and_style_matches_client():
    with pytest.raises(ValueError, match="requires api_base"):
        DirectTransport("deepseek/test", "chat", None, {}, HttpConfig())
    with pytest.raises(TypeError, match="derived"):
        ResponsesClient("openai/test", transport="direct", api_style="chat", api_key="test-key")


@pytest.mark.parametrize(
    "style,patch,pattern",
    [
        ("chat", {"additional_drop_params": ["temperature"]}, "additional_drop_params"),
        ("chat", {"num_retries": 2}, "retry_config"),
        ("chat", {"stream": True}, "stream=False"),
        ("responses", {"max_output_tokens": 10}, "either max_tokens or max_output_tokens"),
        (
            "anthropic",
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "system", "content": "late"},
                ]
            },
            "leading system",
        ),
    ],
)
async def test_direct_request_guards_make_no_http(wire, style, patch, pattern):
    requests, _ = wire
    route = "anthropic/test" if style == "anthropic" else "openai/test"
    transport = DirectTransport(route, style, None, {}, HttpConfig())
    try:
        with pytest.raises(ValueError, match=pattern):
            await transport.acall(
                {
                    "model": route,
                    "api_key": "test-key",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 20,
                    **patch,
                },
            )
        assert requests == []
    finally:
        await transport.aclose()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("transport", ["litellm", "direct"])
@pytest.mark.parametrize(
    "route",
    [
        "mistral/mistral-large-latest",
        "openai/test",
        "nvidia_nim/test",
        "openrouter/test",
        "hosted_vllm/test",
    ],
)
async def test_readable_reasoning_survives_adapter_on_wire(
    monkeypatch, asynchronous, transport, route
):
    bodies = []

    def send(request):
        bodies.append(json.loads(request.content))
        data = reply("chat", "The answer is 4.")
        data["choices"][0]["message"]["reasoning_content"] = "Two plus two is four."
        return httpx.Response(200, json=data)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, request: send(request))

    async def async_send(self, request):
        return send(request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", async_send)
    async with CompletionClient(
        route,
        transport=transport,
        api_base="https://models.example/v1",
        api_key="test-key",
        max_tokens=100,
        retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
    ) as llm:
        messages = [{"role": "user", "content": "2+2?"}]
        first = await llm.acall(messages=messages) if asynchronous else llm.call(messages=messages)
        history = [*messages, first, {"role": "user", "content": "Continue"}]
        if asynchronous:
            await llm.acall(messages=history)
        else:
            llm.call(messages=history)
    assert len(bodies) == 2
    sent = bodies[1]["messages"][1]
    if transport == "litellm" and route.startswith("mistral/"):
        assert sent["content"] == "Two plus two is four.\n\nThe answer is 4."
        assert "reasoning_content" not in sent
    else:
        assert sent["reasoning_content"] == "Two plus two is four."
        assert sent["content"] == "The answer is 4."
    assert first.content == "The answer is 4."
    assert first.reasoning == "Two plus two is four."


def reply(style, text="42"):
    if style == "anthropic":
        return {
            "id": "m",
            "type": "message",
            "role": "assistant",
            "model": "test",
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 2},
        }
    if style == "responses":
        return {
            "id": "r",
            "object": "response",
            "created_at": 1,
            "model": "test",
            "status": "completed",
            "output": [
                {
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}],
                }
            ],
            "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        }
    return {
        "id": "c",
        "object": "chat.completion",
        "created": 1,
        "model": "test",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


@pytest.fixture
def wire(monkeypatch):
    requests = []
    behavior = {"style": "chat", "text": "42"}

    def send(request):
        requests.append(request)
        if len(requests) == 1 and (failure := behavior.get("failure")):
            if failure == "timeout":
                raise httpx.ReadTimeout("test timeout", request=request)
            return httpx.Response(failure, json={"error": {"type": "test", "message": "test"}})
        return httpx.Response(200, json=reply(behavior["style"], behavior["text"]))

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, request: send(request))

    async def async_send(self, request):
        return send(request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", async_send)
    return requests, behavior


def client(style, **kwargs):
    cls = ResponsesClient if style == "responses" else CompletionClient
    return cls(
        f"{'anthropic' if style == 'anthropic' else 'openai'}/test",
        transport="direct",
        api_base="https://models.example/v1",
        api_key="test-key",
        max_tokens=100,
        **kwargs,
    )


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure", [401, 429, 503, "timeout"])
async def test_real_sdk_errors_and_nooa_retries(wire, style, asynchronous, failure):
    requests, behavior = wire
    behavior.update(style=style, failure=failure)
    policy = RetryConfig(
        max_retries=1,
        rate_limit_extra_retries=0,
        base_delay=0,
        rate_limit_base_delay=0,
        jitter_factor=0,
    )
    async with client(style, retry_config=policy) as llm:

        async def invoke():
            messages = [{"role": "user", "content": "test"}]
            return await llm.acall(messages) if asynchronous else llm.call(messages)

        if failure == 401:
            from anthropic import AuthenticationError as AnthropicAuth
            from openai import AuthenticationError as OpenAIAuth

            with pytest.raises(AnthropicAuth if style == "anthropic" else OpenAIAuth):
                await invoke()
            assert len(requests) == 1
        else:
            assert (await invoke()).content == "42"
            assert len(requests) == 2


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
async def test_extensions_reach_wire_at_top_level(wire, style):
    requests, behavior = wire
    behavior["style"] = style
    async with client(style, top_k=7, extra_body={"provider_option": "test"}) as llm:
        await llm.acall([{"role": "user", "content": "test"}])
    body = json.loads(requests[0].content)
    assert body["top_k"] == 7
    assert body["provider_option"] == "test"
    assert "extra_body" not in body


@pytest.mark.parametrize(
    "block,expected",
    [
        (
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,YQ=="}},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "YQ=="},
            },
        ),
        (
            {"type": "image_url", "image_url": {"url": "https://images.example/test.png"}},
            {"type": "image", "source": {"type": "url", "url": "https://images.example/test.png"}},
        ),
        (
            {"type": "file", "file": {"file_data": "data:application/pdf;base64,YQ=="}},
            {
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": "YQ=="},
            },
        ),
        ({"type": "input_text", "text": "test"}, {"type": "text", "text": "test"}),
        ({"type": "output_text", "text": "test"}, {"type": "text", "text": "test"}),
    ],
)
async def test_anthropic_multimodal_success_on_wire(wire, block, expected):
    requests, behavior = wire
    behavior["style"] = "anthropic"
    async with client("anthropic") as llm:
        await llm.acall([{"role": "user", "content": [block]}])
    body = json.loads(requests[0].content)
    assert requests[0].url.path == "/v1/messages"
    assert body["messages"][0]["content"] == [expected]


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
async def test_structured_output_round_trip(wire, style):
    class Answer(BaseModel):
        value: int

    requests, behavior = wire
    behavior.update(style=style, text='{"value":42}')
    async with client(style) as llm:
        result = await llm.acall([{"role": "user", "content": "test"}], output_model=Answer)
    assert isinstance(result.parsed, Answer)
    assert result.parsed.value == 42
    body = json.loads(requests[0].content)
    schema = (
        body["output_config"]["format"]["schema"]
        if style == "anthropic"
        else body["text"]["format"]["schema"]
        if style == "responses"
        else body["response_format"]["json_schema"]["schema"]
    )
    assert "value" in schema["properties"]


async def test_named_tool_choice_on_wire(wire):
    requests, behavior = wire
    behavior["style"] = "anthropic"

    def f(value: int):
        return value

    async with client("anthropic") as llm:
        await llm.acall(
            [{"role": "user", "content": "test"}],
            tools=[Tool(name="f", description="Return a value", callable=f)],
            tool_choice={"type": "function", "function": {"name": "f"}},
            parallel_tool_calls=False,
        )
    assert json.loads(requests[0].content)["tool_choice"] == {
        "type": "tool",
        "name": "f",
        "disable_parallel_tool_use": True,
    }


def test_raw_readable_reasoning_is_not_spoken_anthropic_output():
    body = anthropic_request(
        {
            "max_tokens": 10,
            "messages": [
                {"role": "assistant", "content": "answer", "reasoning_content": "private thought"}
            ],
        }
    )
    assert body["messages"][0]["content"] == [{"type": "text", "text": "answer"}]


def test_direct_pool_keeps_certificate_and_redirect_settings(monkeypatch):
    monkeypatch.setenv("SSL_CERTIFICATE", "/certs/client.pem")
    monkeypatch.setenv("SSL_VERIFY", "/certs/ca.pem")
    settings = DirectTransport._http_settings(HttpConfig())
    assert settings["cert"] == "/certs/client.pem"
    assert settings["verify"] == "/certs/ca.pem"
    assert settings["follow_redirects"] is True


@pytest.mark.parametrize("transport", ["direct", "litellm"])
@pytest.mark.parametrize("vendor", ["", "OpenAI", "bad-name", 3])
def test_replay_vendor_validated_before_pool_construction(transport, vendor):
    with pytest.raises(ValueError, match="replay_vendor"):
        CompletionClient("openai/test", transport=transport, replay_vendor=vendor)
