# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Review regressions exercised below the real SDK serializers, without network."""

import copy
import json
import socket

import httpx
import pytest
from pydantic import BaseModel

from nooa.unifiedllm import CompletionClient, ResponsesClient, RetryConfig, Tool
from nooa.unifiedllm.direct import DirectTransport, anthropic_request
from nooa.unifiedllm.http_config import HttpConfig


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


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail closed if an SDK or mock regression tries to open a real connection."""

    def forbidden(*args, **kwargs):
        raise AssertionError("Direct SDK tests must never connect to the network")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)


@pytest.fixture
def wire(monkeypatch):
    requests = []
    behavior = {"style": "chat", "text": "42"}

    def send(request):
        requests.append(request)
        if (len(requests) == 1 or behavior.get("always_fail")) and (
            failure := behavior.get("failure")
        ):
            if failure == "timeout":
                raise httpx.ReadTimeout("test timeout", request=request)
            return httpx.Response(failure, json={"error": {"type": "test", "message": "test"}})
        return httpx.Response(
            200, json=behavior.get("response") or reply(behavior["style"], behavior["text"])
        )

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, request: send(request))

    async def async_send(self, request):
        return send(request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", async_send)
    return requests, behavior


def client(style, **kwargs):
    cls = ResponsesClient if style == "responses" else CompletionClient
    return cls(
        f"{'anthropic' if style == 'anthropic' else 'openai'}/test",
        direct=True,
        api_base="https://models.example/v1",
        api_key="test-key",
        max_tokens=100,
        **kwargs,
    )


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("failure", [401, 408, 429, 503, "timeout"])
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
    assert settings["follow_redirects"] is False


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
async def test_round_trip_bypasses_litellm_and_collectors(wire, monkeypatch, style, asynchronous):
    import nooa.unifiedllm.unifiedllm as module

    def forbidden(*args, **kwargs):
        raise AssertionError("Direct SDK objects must not enter LiteLLM or stream collectors")

    for name in ("completion", "acompletion", "responses", "aresponses", "get_llm_provider"):
        monkeypatch.setattr(module.litellm, name, forbidden)
    monkeypatch.setattr(module, "_collect_sync", forbidden)
    monkeypatch.setattr(module, "_collect_async", forbidden)
    requests, behavior = wire
    behavior["style"] = style
    async with client(style) as llm:
        for _ in range(3):
            history = [
                {"role": "system", "content": "stable"},
                {"role": "user", "content": "question"},
            ]
            result = await llm.acall(history) if asynchronous else llm.call(history)
            assert result.content == "42"
            assert result.usage.input_tokens == 10
            assert result.usage.cost_usd == 0.0  # Existing public unknown-cost sentinel
        assert not llm._http.httpx_sync.is_closed
        assert not llm._http.httpx_async.is_closed
    assert llm._http.httpx_sync.is_closed
    assert llm._http.httpx_async.is_closed
    assert len(requests) == 3
    for request in requests:
        body = json.loads(request.content)
        assert body["model"] == "test"
        assert "direct" not in body
        assert "extra_body" not in body


@pytest.mark.parametrize("cls", [CompletionClient, ResponsesClient])
@pytest.mark.parametrize("direct", [None, 0, 1, "true", "false", [], {}])
def test_strict_boolean_direct(cls, direct):
    with pytest.raises(ValueError, match="direct must be a boolean"):
        cls("openai/test", direct=direct)


def test_registry_explicit_opt_in_override_and_metadata_ignored(monkeypatch):
    import nooa.unifiedllm.unifiedllm as module
    from nooa.unifiedllm.registry import client_from_config

    monkeypatch.setenv("NOOA_LLM_TRANSPORT", "direct")
    config = {
        "model_name": "openai/test",
        "api_key_env": "TEST_DIRECT_KEY",
        "api_base": "https://models.example/v1",
        "transport": "direct",
    }
    monkeypatch.setenv("TEST_DIRECT_KEY", "test-key")
    # Existing connect transport/api_style metadata must not select a transport.
    config["api_style"] = "anthropic"
    with client_from_config("alias", config) as llm:
        assert llm.direct is False
        assert isinstance(llm._http, module._ClientHttp)
        assert "direct" not in llm.config
    config["direct"] = True
    with client_from_config("alias", config) as llm:
        assert llm.direct is True
        assert isinstance(llm._http, DirectTransport)
        assert llm._http.api_style == "chat"
    with client_from_config("alias", config, direct=False) as llm:
        assert llm.direct is False
    with client_from_config("alias", {**config, "client_type": "responses"}) as llm:
        assert llm._http.api_style == "responses"
    with client_from_config("alias", {**config, "direct": False}, direct=True) as llm:
        assert llm.direct is True
    with pytest.raises(ValueError, match="boolean"):
        client_from_config("alias", {**config, "direct": "true"})


@pytest.mark.parametrize("direct", [True])
@pytest.mark.parametrize("extra", [3, [1], "bad"])
async def test_malformed_extra_body(direct, extra):
    with pytest.raises(ValueError, match="extra_body must be a mapping"):
        CompletionClient("openai/test", direct=direct, extra_body=extra)
    async with CompletionClient("openai/test", direct=direct, api_key="test") as llm:
        for invoke in (llm.call, llm.acall):
            with pytest.raises(ValueError, match="extra_body must be a mapping"):
                result = invoke([], extra_body=extra)
                if hasattr(result, "__await__"):
                    await result


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "patch",
    [
        {"direct": False},
        {"stream": True},
        {"stream": 0},
        {"client": object()},
        {"transport": "litellm"},
        {"api_style": "chat"},
        {"replay_vendor": "openai"},
        {"custom_llm_provider": "openai"},
        {"num_retries": 2},
        {"extra_body": {"direct": True}},
        {"extra_body": {"model": "other"}},
        {"extra_body": {"messages": []}},
        {"extra_body": {"input": []}},
        {"extra_body": {"instructions": "other"}},
        {"extra_body": {"stream": True}},
        {"extra_body": {"api_base": "https://other.example"}},
        {"extra_body": {"http_client": "bad"}},
    ],
)
async def test_preflight_no_requests(wire, asynchronous, patch):
    requests, _ = wire
    async with client("chat") as llm:
        with pytest.raises(ValueError):
            history = [{"role": "user", "content": "question"}]
            if asynchronous:
                await llm.acall(history, **patch)
            else:
                llm.call(history, **patch)
    assert requests == []


@pytest.mark.parametrize("cls", [CompletionClient, ResponsesClient])
@pytest.mark.parametrize("source", ["explicit", "environment"])
def test_native_azure_endpoint_rejected(cls, source, monkeypatch):
    endpoint = "https://test.openai.azure.com/openai/deployments/test"
    params = {"api_base": endpoint} if source == "explicit" else {}
    if source == "environment":
        monkeypatch.setenv("OPENAI_BASE_URL", endpoint)
    with pytest.raises(ValueError, match="Azure"):
        cls("openai/test", direct=True, **params)


@pytest.mark.parametrize(
    "model", ["bedrock/test", "vertex_ai/test", "sagemaker/test", "azure/test"]
)
def test_native_provider_prefix_rejected(model):
    with pytest.raises(ValueError, match="unsupported"):
        CompletionClient(model, direct=True, api_base="https://models.example/v1")


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_translated_fields_cannot_be_overwritten(wire, asynchronous):
    class Answer(BaseModel):
        value: int

    requests, _ = wire
    async with client("anthropic") as llm:
        for patch, output in [
            ({"stop": "INTENDED", "extra_body": {"stop_sequences": ["OVERRIDE"]}}, None),
            (
                {
                    "extra_body": {
                        "output_config": {
                            "format": {"type": "json_schema", "schema": {"type": "string"}}
                        }
                    }
                },
                Answer,
            ),
        ]:
            with pytest.raises(ValueError, match="overwrite"):
                history = [{"role": "user", "content": "test"}]
                if asynchronous:
                    await llm.acall(history, output_model=output, **patch)
                else:
                    llm.call(history, output_model=output, **patch)
    assert requests == []


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("status", [302, 307, 308])
async def test_anthropic_redirects_never_leak_key_or_body(monkeypatch, asynchronous, status):
    requests = []

    def send(request):
        requests.append(request)
        return httpx.Response(status, headers={"location": "https://other.example/stolen"})

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, req: send(req))

    async def asend(self, request):
        return send(request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", asend)
    async with client("anthropic", retry_config=RetryConfig(max_retries=0)) as llm:
        from anthropic import APIStatusError

        with pytest.raises(APIStatusError):
            history = [{"role": "user", "content": "secret"}]
            if asynchronous:
                await llm.acall(history)
            else:
                llm.call(history)
    assert len(requests) == 1
    assert requests[0].url.host == "models.example"


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
async def test_cancellation_propagates_and_pool_can_be_reused(wire, monkeypatch, style):
    import asyncio

    requests, behavior = wire
    behavior["style"] = style
    original = httpx.AsyncHTTPTransport.handle_async_request
    entered = asyncio.Event()
    release = asyncio.Event()

    async def blocked(self, request):
        entered.set()
        await release.wait()
        return await original(self, request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", blocked)
    async with client(style) as llm:
        task = asyncio.create_task(llm.acall([{"role": "user", "content": "test"}]))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not llm._http.httpx_async.is_closed
        assert requests == []
        release.set()
        assert (await llm.acall([{"role": "user", "content": "test"}])).content == "42"
    assert len(requests) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
async def test_per_call_route_and_auth_overrides(wire, monkeypatch, style, asynchronous):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://test.openai.azure.com")
    requests, behavior = wire
    behavior["style"] = style
    async with client(style) as llm:
        history = [{"role": "user", "content": "question"}]
        params = {"base_url": "https://override.example/v1", "api_key": "replacement"}
        if asynchronous:
            await llm.acall(history, **params)
        else:
            llm.call(history, **params)
        prefix = "anthropic" if style == "anthropic" else "openai"
        assert llm._replay_scope(
            f"{prefix}/test", "responses" if style == "responses" else "chat", params
        ) == llm._replay_scope(
            llm.model, "responses" if style == "responses" else "chat", llm.config
        )
    assert requests[0].url.host == "override.example"
    assert (
        requests[0].headers["x-api-key"]
        if style == "anthropic"
        else requests[0].headers["authorization"]
    ) in ("replacement", "Bearer replacement")


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_latest80_cache_checkpoints_and_growth_use_prepared_wire(wire, asynchronous):
    from nooa.llm_types import CacheBoundary

    requests, behavior = wire
    behavior["style"] = "responses"
    stable = [{"role": "user", "content": f"stable-{i}"} for i in range(85)]
    async with client("responses") as llm:

        async def send(history):
            return await llm.acall(history) if asynchronous else llm.call(history)

        await send([*stable, CacheBoundary(), {"role": "system", "content": "live-a"}])
        await send([*stable, CacheBoundary(), {"role": "system", "content": "live-b"}])
        await send(
            [
                *stable,
                {"role": "user", "content": "new-stable"},
                CacheBoundary(),
                {"role": "system", "content": "live-c"},
            ]
        )
    bodies = [json.loads(request.content) for request in requests]
    assert bodies[0]["input"][:-1] == bodies[1]["input"][:-1]
    assert bodies[0]["prompt_cache_options"] == {"mode": "explicit"}
    for index, body in enumerate(bodies):
        inputs = body["input"]
        eligible = inputs[:-1]
        assert sum("prompt_cache_breakpoint" in item["content"][-1] for item in eligible) == 80
        cutoff = 5 if index < 2 else 6
        for i, item in enumerate(eligible):
            block = item["content"][-1]
            assert block["type"] == "input_text"
            assert ("prompt_cache_breakpoint" in block) == (i >= cutoff)
        assert inputs[-1]["content"] == [{"type": "input_text", "text": f"live-{chr(97 + index)}"}]
        assert "nooa_cache_boundary" not in json.dumps(body)
    assert bodies[0]["input"][6:85] == bodies[2]["input"][6:85]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_anthropic_nested_tool_result_markers_and_dynamic_system_suffix(wire, asynchronous):
    from nooa.llm_types import CacheBoundary

    requests, behavior = wire
    behavior["style"] = "anthropic"
    history = [
        {"role": "system", "content": "stable instructions"},
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "call", "type": "function", "function": {"name": "f", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "call", "content": "stable result"},
        CacheBoundary(),
    ]
    async with client("anthropic") as llm:
        for state in ("a", "b"):
            messages = [*history, {"role": "system", "content": f"dynamic-{state}"}]
            if asynchronous:
                await llm.acall(messages)
            else:
                llm.call(messages)
    bodies = [json.loads(request.content) for request in requests]
    assert bodies[0]["system"] == [{"type": "text", "text": "stable instructions"}]
    assert bodies[0]["system"] == bodies[1]["system"]
    first, second = bodies[0]["messages"][-1], bodies[1]["messages"][-1]
    assert first["role"] == second["role"] == "user"
    assert (
        first["content"][0]
        == second["content"][0]
        == {
            "type": "tool_result",
            "tool_use_id": "call",
            "content": [
                {"type": "text", "text": "stable result", "cache_control": {"type": "ephemeral"}}
            ],
        }
    )
    assert first["content"][-1] == {"type": "text", "text": "dynamic-a"}
    assert second["content"][-1] == {"type": "text", "text": "dynamic-b"}
    assert "dynamic" not in json.dumps(bodies[0]["system"])
    assert "cache_control" not in history[3]
    assert history[3]["content"] == "stable result"


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_anthropic_native_nested_tool_result_marker_is_not_moved(wire, asynchronous):
    import copy

    from nooa.llm_types import CacheBoundary

    requests, behavior = wire
    behavior["style"] = "anthropic"
    nested = {
        "type": "tool_result",
        "tool_use_id": "call",
        "content": [
            {"type": "text", "text": "stable result", "cache_control": {"type": "ephemeral"}}
        ],
    }
    before = copy.deepcopy(nested)
    async with client("anthropic") as llm:
        messages = [
            {"role": "user", "content": [nested]},
            CacheBoundary(),
            {"role": "system", "content": "dynamic"},
        ]
        if asynchronous:
            await llm.acall(messages)
        else:
            llm.call(messages)
    content = json.loads(requests[0].content)["messages"][0]["content"]
    assert content[0]["content"] == before["content"]
    assert content[0]["cache_control"] == {"type": "ephemeral"}
    assert content[-1] == {"type": "text", "text": "dynamic"}
    assert nested == before


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_gemini_compatible_signature_roundtrip(wire, asynchronous):
    requests, behavior = wire
    native_reply = reply("chat", text="")
    message = native_reply["choices"][0]["message"]
    message["tool_calls"] = [
        {
            "id": "call",
            "type": "function",
            "function": {"name": "f", "arguments": "{}"},
            "extra_content": {"google": {"thought_signature": "SECRET_SIGNATURE"}},
        }
    ]
    native_reply["choices"][0]["finish_reason"] = "tool_calls"
    behavior["response"] = native_reply
    async with CompletionClient(
        "gemini/test",
        direct=True,
        api_base="https://models.example/v1",
        api_key="test",
        max_tokens=100,
    ) as llm:

        async def send(history, **patch):
            return await llm.acall(history, **patch) if asynchronous else llm.call(history, **patch)

        turn = await send([{"role": "user", "content": "test"}])
        assert turn.tool_calls[0].id == "call"
        assert "SECRET_SIGNATURE" not in str(dict(turn))
        assert (
            turn.tool_calls[0].native["provider_specific_fields"]["thought_signature"]
            == "SECRET_SIGNATURE"
        )
        archive = turn.model_dump_json()
        history = [turn, {"role": "tool", "tool_call_id": "call", "content": "result"}]
        await send(history)
        await send(history, model="gemini/replacement")
        assert turn.model_dump_json() == archive
    body = json.loads(requests[1].content)
    assert body["messages"][0]["tool_calls"][0]["extra_content"] == {
        "google": {"thought_signature": "SECRET_SIGNATURE"}
    }
    assert "provider_specific_fields" not in body["messages"][0]["tool_calls"][0]
    assert body["messages"][1]["tool_call_id"] == "call"
    assert "SECRET_SIGNATURE" not in requests[2].content.decode()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "reason,expected",
    [
        ("end_turn", "stop"),
        ("tool_use", "tool_calls"),
        ("max_tokens", "length"),
        ("model_context_window_exceeded", "length"),
        ("refusal", "error"),
    ],
)
async def test_native_stop_reason_contract(wire, asynchronous, reason, expected):
    requests, behavior = wire
    behavior["response"] = {**reply("anthropic"), "stop_reason": reason}
    async with client("anthropic") as llm:
        history = [{"role": "user", "content": "test"}]
        result = await llm.acall(history) if asynchronous else llm.call(history)
        assert result.finish_reason == expected
    assert len(requests) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("reason", ["pause_turn", "unexpected"])
async def test_unsupported_continuation_raises_without_retry(wire, asynchronous, reason):
    from nooa.unifiedllm.errors import UnsupportedStopReasonError

    requests, behavior = wire
    behavior["response"] = {**reply("anthropic"), "stop_reason": reason}
    async with client("anthropic") as llm:
        history = [{"role": "user", "content": "test"}]
        with pytest.raises(UnsupportedStopReasonError):
            if asynchronous:
                await llm.acall(history)
            else:
                llm.call(history)
    assert len(requests) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_native_encrypted_replay_and_gateway_demotion(wire, monkeypatch, asynchronous):
    import nooa.unifiedllm.unifiedllm as module

    requests, behavior = wire
    response = reply("responses")
    response["output"] = [
        {
            "id": "rs",
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "thought"}],
            "encrypted_content": "SECRET",
        },
        {
            "type": "function_call",
            "call_id": "call",
            "id": "fc",
            "name": "f",
            "arguments": "{}",
            "status": "completed",
        },
    ]
    behavior["response"] = response
    # LiteLLM global routing must not alter the SDK's native route decision.
    monkeypatch.setattr(module.litellm, "api_base", "https://gateway.example/v1")
    monkeypatch.setenv("OPENAI_API_BASE", "https://gateway.example/v1")
    async with ResponsesClient("openai/test", direct=True, api_key="test", max_tokens=100) as llm:

        async def send(history, **patch):
            return await llm.acall(history, **patch) if asynchronous else llm.call(history, **patch)

        turn = await send([{"role": "user", "content": "test"}])
        archive = turn.model_dump_json()
        await send([turn, {"role": "tool", "tool_call_id": "call", "content": "result"}])
        assert turn.model_dump_json() == archive
        assert requests[0].url.host == "api.openai.com"
        assert json.loads(requests[0].content)["include"] == ["reasoning.encrypted_content"]
        reasoning = json.loads(requests[1].content)["input"][0]
        assert reasoning["encrypted_content"] == "SECRET"
        assert reasoning["summary"] == [{"type": "summary_text", "text": "thought"}]
        response["output"][0].pop("encrypted_content")
        gateway_turn = await send(
            [{"role": "user", "content": "test"}], base_url="https://gateway.example/v1"
        )
        assert "include" not in json.loads(requests[2].content)
        assert gateway_turn.parts[0].native is not None
        await send([gateway_turn, {"role": "tool", "tool_call_id": "call", "content": "result"}])
        native_body = json.loads(requests[3].content)
        assert "thought" in json.dumps(native_body["input"])
        assert not any(item.get("type") == "reasoning" for item in native_body["input"])
        await send([{"role": "user", "content": "test"}], include=[])
        assert "include" not in json.loads(requests[4].content)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_responses_raw_transport_translated_alias_collision_rejected(wire, asynchronous):
    requests, _ = wire
    transport = DirectTransport("openai/test", "responses", {}, HttpConfig())
    params = {
        "model": "openai/test",
        "api_key": "test",
        "api_base": "https://models.example/v1",
        "input": [{"role": "user", "content": "x"}],
        "max_tokens": 100,
        "extra_body": {"max_output_tokens": 100000},
    }
    try:
        with pytest.raises(ValueError, match="overwrite translated"):
            if asynchronous:
                await transport.acall(params)
            else:
                transport.call(params)
    finally:
        await transport.aclose()
    assert requests == []


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_extra_body_reply_cap_uses_current_public_override_semantics(wire, asynchronous):
    requests, behavior = wire
    behavior["style"] = "responses"
    async with client("responses") as llm:
        history = [{"role": "user", "content": "test"}]
        patch = {"extra_body": {"max_output_tokens": 1234}}
        # Current apply_reasoning_level promotes a per-call cap, replacing the
        # inherited cap before either transport. Do not regress that API.
        assert llm.get_context_limits(patch).reserved_output_tokens == 1234
        if asynchronous:
            await llm.acall(history, **patch)
        else:
            llm.call(history, **patch)
    assert json.loads(requests[0].content)["max_output_tokens"] == 1234


def test_direct_is_not_a_reasoning_level_setting():
    with pytest.raises(ValueError, match="reserved fields"):
        CompletionClient("openai/test", reasoning_levels={"bad": {"direct": True}})


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "endpoint,expected",
    [
        ("https://api.openai.com/v1", "max_completion_tokens"),
        ("https://gateway.example/v1", "max_tokens"),
    ],
)
async def test_chat_reply_cap_follows_actual_endpoint(wire, asynchronous, endpoint, expected):
    requests, _ = wire
    async with client("chat") as llm:
        history = [{"role": "user", "content": "test"}]
        if asynchronous:
            await llm.acall(history, base_url=endpoint)
        else:
            llm.call(history, base_url=endpoint)
    body = json.loads(requests[0].content)
    assert body[expected] == 100
    assert len({"max_tokens", "max_completion_tokens"} & body.keys()) == 1


@pytest.mark.parametrize(
    "model,wire_model",
    [
        ("openai/my-org/my-model", "my-org/my-model"),
        ("my-org/my-model", "my-org/my-model"),
        ("openai/anthropic/model", "anthropic/model"),
        ("deepseek/test", "test"),
    ],
)
async def test_explicit_model_routing_not_name_guessing(wire, model, wire_model):
    requests, _ = wire
    async with CompletionClient(
        model, direct=True, api_key="test", max_tokens=100, api_base="https://models.example/v1"
    ) as llm:
        assert llm._http.api_style == "chat"
        await llm.acall([{"role": "user", "content": "test"}])
    body = json.loads(requests[0].content)
    assert body["model"] == wire_model
    assert requests[0].url.path == "/v1/chat/completions"
    assert "cache_control" not in json.dumps(body)


async def test_http_config_is_owned_and_close_is_idempotent():
    config = HttpConfig(max_connections=3, max_keepalive_connections=2, read_timeout=123)
    async with CompletionClient(
        "openai/test", direct=True, api_key="test", http_config=config
    ) as llm:
        transport = llm._http
        assert transport.httpx_sync.timeout.read == 123
        assert transport.httpx_async.timeout.read == 123
        assert transport.httpx_sync._transport._pool._max_connections == 3
        assert transport.httpx_async._transport._pool._max_keepalive_connections == 2
        llm.close()
        llm.close()
        assert transport.httpx_sync.is_closed
        assert not transport.httpx_async.is_closed
    await llm.aclose()
    assert transport.httpx_async.is_closed


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("redacted", [False, True])
async def test_native_thinking_only_configured_empty_retry(wire, asynchronous, enabled, redacted):
    from nooa.llm_types import AssistantReasoning, LLMResponse
    from nooa.unifiedllm.errors import EmptyContentError

    requests, behavior = wire
    block = (
        {"type": "redacted_thinking", "data": "SECRET"}
        if redacted
        else {"type": "thinking", "thinking": "thought", "signature": "SIGNED"}
    )
    behavior["response"] = {**reply("anthropic"), "content": [block], "stop_reason": "max_tokens"}
    policy = RetryConfig(
        max_retries=2, base_delay=0, jitter_factor=0, retry_on_empty_content=enabled
    )
    async with client("anthropic", retry_config=policy) as llm:
        history = [{"role": "user", "content": "test"}]
        if enabled:
            with pytest.raises(EmptyContentError):
                await llm.acall(history) if asynchronous else llm.call(history)
            assert len(requests) == 3  # configured attempts, not SDK retries
        else:
            turn = await llm.acall(history) if asynchronous else llm.call(history)
            assert len(requests) == 1
            assert turn.content == ""
            assert len([p for p in turn.parts if isinstance(p, AssistantReasoning)]) == 1
            assert turn.reasoning == (None if redacted else "thought")
            assert "reasoning_content" not in turn.raw_response.choices[0].message.model_dump()
            restored = LLMResponse.model_validate_json(turn.model_dump_json())
            await llm.acall([restored]) if asynchronous else llm.call([restored])
            assert json.loads(requests[-1].content)["messages"][0]["content"] == [block]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_native_thinking_only_retry_then_success(wire, monkeypatch, asynchronous):
    requests, behavior = wire
    behavior["response"] = {
        **reply("anthropic"),
        "stop_reason": "max_tokens",
        "content": [{"type": "thinking", "thinking": "thought", "signature": "SIGNED"}],
    }
    # Change the mocked response when NOOA schedules its retry.
    policy = RetryConfig(
        max_retries=1,
        base_delay=0,
        jitter_factor=0,
        retry_on_empty_content=True,
        on_retry=lambda *args: behavior.update(response=reply("anthropic")),
    )
    async with client("anthropic", retry_config=policy) as llm:
        history = [{"role": "user", "content": "test"}]
        turn = await llm.acall(history) if asynchronous else llm.call(history)
    assert turn.content == "42"
    assert len(requests) == 2


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("conflict", [False, True])
async def test_responses_output_model_preserves_text_options(wire, asynchronous, conflict):
    class Answer(BaseModel):
        value: int

    requests, behavior = wire
    behavior.update(style="responses", text='{"value":42}')
    text = {"verbosity": "low"}
    if conflict:
        text["format"] = {"type": "text"}
    async with client("responses", text=text) as llm:
        history = [{"role": "user", "content": "test"}]
        if conflict:
            with pytest.raises(ValueError, match="text.format conflicts"):
                await llm.acall(history, output_model=Answer) if asynchronous else llm.call(
                    history, output_model=Answer
                )
            assert not requests
        else:
            turn = (
                await llm.acall(history, output_model=Answer)
                if asynchronous
                else llm.call(history, output_model=Answer)
            )
            assert turn.parsed == Answer(value=42)
    if not conflict:
        body = json.loads(requests[0].content)
        assert body["text"]["verbosity"] == "low"
        assert body["text"]["format"]["type"] == "json_schema"
        assert body["text"]["format"]["strict"] is True
        assert body["text"]["format"]["schema"]["properties"]["value"]["type"] == "integer"
    assert text == (
        {"verbosity": "low", "format": {"type": "text"}} if conflict else {"verbosity": "low"}
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("ttl", ["5m", "1h"])
async def test_native_function_tool_metadata_on_sdk_wire(wire, asynchronous, strict, ttl):
    requests, behavior = wire
    behavior["style"] = "anthropic"
    tools = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "description": "test",
                "parameters": {"type": "object"},
                "strict": strict,
            },
            "cache_control": {"type": "ephemeral", "ttl": ttl},
        }
    ]
    async with client(
        "anthropic", tools=tools, tool_choice={"type": "function", "function": {"name": "f"}}
    ) as llm:
        history = [{"role": "user", "content": "test"}]
        await llm.acall(history) if asynchronous else llm.call(history)
    tool = json.loads(requests[0].content)["tools"][0]
    assert tool == {
        "type": "custom",
        "name": "f",
        "description": "test",
        "input_schema": {"type": "object"},
        "strict": strict,
        "cache_control": {"type": "ephemeral", "ttl": ttl},
    }
    assert json.loads(requests[0].content)["tool_choice"] == {"type": "tool", "name": "f"}
    assert tools[0]["function"]["strict"] is strict


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("explicit_exclusion", [False, True])
async def test_native_overload_default_policy_and_status_optout(
    wire, asynchronous, explicit_exclusion
):
    from anthropic import OverloadedError

    requests, behavior = wire
    behavior.update(style="anthropic", failure=529)
    settings = {"max_retries": 1, "base_delay": 0, "jitter_factor": 0}
    if explicit_exclusion:
        settings["retryable_status_codes"] = RetryConfig().retryable_status_codes
    policy = RetryConfig(**settings)
    async with client("anthropic", retry_config=policy) as llm:
        assert (529 in llm.retry_config.retryable_status_codes) is not explicit_exclusion
        history = [{"role": "user", "content": "test"}]
        if explicit_exclusion:
            with pytest.raises(OverloadedError) as error:
                await llm.acall(history) if asynchronous else llm.call(history)
            assert error.value.status_code == 529
            assert len(requests) == 1
        else:
            turn = await llm.acall(history) if asynchronous else llm.call(history)
            assert turn.content == "42"
            assert len(requests) == 2
    assert 529 not in policy.retryable_status_codes  # no mutation/global change
    assert 529 not in RetryConfig().retryable_status_codes


def test_overload_is_native_direct_default_only():
    for style in ("chat", "responses", "anthropic"):
        with client(style) as llm:
            assert (529 in llm.retry_config.retryable_status_codes) is (style == "anthropic")
    with CompletionClient("anthropic/test", direct=False, max_tokens=100) as llm:
        assert 529 not in llm.retry_config.retryable_status_codes


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_native_response_block_fidelity_limit_is_durable(wire, asynchronous):
    from nooa.llm_types import AssistantReasoning, LLMResponse

    requests, behavior = wire
    # The citation is a supported native text field. cache_control below is a
    # synthetic response extension characterizing loss, not a live API promise.
    citation = {
        "type": "char_location",
        "cited_text": "a",
        "document_index": 0,
        "document_title": "doc",
        "start_char_index": 0,
        "end_char_index": 1,
    }
    behavior["response"] = {
        **reply("anthropic"),
        "content": [
            {
                "type": "text",
                "text": "A",
                "citations": [citation],
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "tool_use",
                "id": "call",
                "name": "f",
                "input": {},
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": "B"},
            {"type": "thinking", "thinking": "thought", "signature": "SIGNED"},
            {"type": "text", "text": "C"},
        ],
        "stop_reason": "tool_use",
    }
    async with client("anthropic", cache_breakpoint=None) as llm:
        history = [{"role": "user", "content": "test"}]
        turn = await llm.acall(history) if asynchronous else llm.call(history)
        assert turn.content == "ABC"
        assert turn.reasoning == "thought"
        assert len([p for p in turn.parts if isinstance(p, AssistantReasoning)]) == 1
        restored = LLMResponse.model_validate_json(turn.model_dump_json())
        history = [restored, {"role": "tool", "tool_call_id": "call", "content": "result"}]
        await llm.acall(history) if asynchronous else llm.call(history)
    assert json.loads(requests[1].content)["messages"][0]["content"] == [
        {"type": "thinking", "thinking": "thought", "signature": "SIGNED"},
        {"type": "text", "text": "ABC"},
        {"type": "tool_use", "id": "call", "name": "f", "input": {}},
    ]
    assert "citations" not in restored.model_dump_json()
    assert "cache_control" not in restored.model_dump_json()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("prefix", ["openai", "gemini"])
async def test_google_extension_requires_explicit_gemini_protocol(wire, asynchronous, prefix):
    from nooa.llm_types import LLMResponse

    requests, behavior = wire
    response = reply("chat", text="")
    response["choices"][0]["finish_reason"] = "tool_calls"
    response["choices"][0]["message"]["tool_calls"] = [
        {
            "id": "call",
            "type": "function",
            "function": {"name": "f", "arguments": "{}"},
            "extra_content": {"google": {"thought_signature": "SECRET"}},
        }
    ]
    behavior["response"] = response
    async with CompletionClient(
        f"{prefix}/gateway-model",
        direct=True,
        base_url="https://first.example/v1",
        api_key="test",
        max_tokens=100,
    ) as llm:
        history = [{"role": "user", "content": "test"}]
        turn = await llm.acall(history) if asynchronous else llm.call(history)
        restored = LLMResponse.model_validate_json(turn.model_dump_json())
        assert bool(restored.tool_calls[0].native) is (prefix == "gemini")
        history = [restored, {"role": "tool", "tool_call_id": "call", "content": "result"}]
        # URL rotation intentionally keeps replay scope. gemini/ declares the
        # caller's chosen destination implements and is trusted with this state.
        await llm.acall(
            history, base_url="https://second.example/v1"
        ) if asynchronous else llm.call(history, base_url="https://second.example/v1")
        await llm.acall(history, model="openai/gateway-model") if asynchronous else llm.call(
            history, model="openai/gateway-model"
        )
    assert ("SECRET" in requests[1].content.decode()) is (prefix == "gemini")
    assert requests[1].url.host == "second.example"
    assert "SECRET" not in requests[2].content.decode()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "extension",
    [1, {"google": 1}, {"google": {"thought_signature": ""}}, {"google": {"thought_signature": 1}}],
)
async def test_malformed_google_signature_is_terminal(wire, asynchronous, extension):
    from nooa.unifiedllm.errors import ReasoningReplayError

    requests, behavior = wire
    response = reply("chat", text="")
    response["choices"][0]["message"]["tool_calls"] = [
        {
            "id": "call",
            "type": "function",
            "function": {"name": "f", "arguments": "{}"},
            "extra_content": extension,
        }
    ]
    behavior["response"] = response
    async with CompletionClient(
        "gemini/test",
        direct=True,
        api_key="test",
        api_base="https://models.example/v1",
        max_tokens=100,
    ) as llm:
        with pytest.raises(ReasoningReplayError, match="Gemini"):
            await llm.acall([{"role": "user", "content": "test"}]) if asynchronous else llm.call(
                [{"role": "user", "content": "test"}]
            )
    assert len(requests) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("direct_setting", [None, False])
@pytest.mark.parametrize("extra", [3, [1], "bad"])
async def test_default_extra_body_retains_existing_call_validation(
    monkeypatch, asynchronous, direct_setting, extra
):
    from litellm import ModelResponse

    import nooa.unifiedllm.unifiedllm as module

    calls = []

    def send(**kwargs):
        calls.append(kwargs)
        return ModelResponse(**reply("chat"))

    async def asend(**kwargs):
        return send(**kwargs)

    monkeypatch.setattr(module.litellm, "completion", send)
    monkeypatch.setattr(module.litellm, "acompletion", asend)
    kwargs = {} if direct_setting is None else {"direct": direct_setting}
    async with CompletionClient("openai/test", extra_body=extra, **kwargs) as llm:
        history = [{"role": "user", "content": "test"}]
        with pytest.raises(ValueError, match="extra_body must be a mapping"):
            await llm.acall(history) if asynchronous else llm.call(history)
        with pytest.raises(ValueError, match="extra_body must be a mapping"):
            await llm.acall(history, extra_body=extra) if asynchronous else llm.call(
                history, extra_body=extra
            )
    assert not calls


@pytest.mark.parametrize("registered", [None, False, True])
@pytest.mark.parametrize("override", [None, False, True])
@pytest.mark.parametrize("client_type", ["completion", "responses"])
def test_registry_base_url_only_forwarded_for_effective_direct(registered, override, client_type):
    from nooa.unifiedllm.registry import client_from_config

    config = {
        "model_name": "openai/test",
        "base_url": "https://models.example/v1",
        "api_key_env": "TEST_KEY",
        "client_type": client_type,
    }
    if registered is not None:
        config["direct"] = registered
    kwargs = {} if override is None else {"direct": override}
    effective = override if override is not None else bool(registered)
    with client_from_config("alias", config, **kwargs) as llm:
        assert llm.direct is effective
        assert ("base_url" in llm.config) is effective
    with client_from_config("alias", config, base_url="https://caller.example/v1", **kwargs) as llm:
        assert llm.config["base_url"] == "https://caller.example/v1"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("direct_setting", [None, False])
@pytest.mark.parametrize("style", ["chat", "responses"])
async def test_existing_alias_default_litellm_real_wire(
    wire, monkeypatch, asynchronous, direct_setting, style
):
    import nooa.unifiedllm.unifiedllm as module
    from nooa.unifiedllm.registry import client_from_config

    requests, behavior = wire
    behavior["style"] = style
    monkeypatch.setenv("TEST_DEFAULT_KEY", "test-key")
    config = {
        "model_name": "openai/test",
        "api_base": "https://models.example/v1",
        "base_url": "https://must-not-forward.example/v1",
        "api_key_env": "TEST_DEFAULT_KEY",
        "client_type": "responses" if style == "responses" else "completion",
        "max_tokens": 100,
    }
    if direct_setting is not None:
        config["direct"] = direct_setting
    async with client_from_config("existing-alias", config) as llm:
        assert isinstance(llm._http, module._ClientHttp)
        history = [{"role": "user", "content": "test"}]
        result = await llm.acall(history) if asynchronous else llm.call(history)
        assert result.content == "42"
    assert len(requests) == 1
    assert requests[0].url.host == "models.example"
    body = json.loads(requests[0].content)
    assert body["model"] == "test"
    assert "direct" not in body
    assert "base_url" not in body
    assert requests[0].url.path == (
        "/v1/responses" if style == "responses" else "/v1/chat/completions"
    )


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("prefix", ["openai", "gemini"])
@pytest.mark.parametrize("spelling", ["field", "inline", "both"])
async def test_direct_tool_signatures_all_spellings_require_declared_protocol(
    wire, asynchronous, prefix, spelling
):
    from nooa.llm_types import LLMResponse

    requests, behavior = wire
    response = reply("chat", text="")
    call = {"id": "call", "type": "function", "function": {"name": "f", "arguments": "{}"}}
    if spelling in {"field", "both"}:
        call["provider_specific_fields"] = {"thought_signature": "SECRET"}
    if spelling in {"inline", "both"}:
        call["id"] += "__thought__SECRET"
    response["choices"][0]["message"]["tool_calls"] = [call]
    response["choices"][0]["finish_reason"] = "tool_calls"
    behavior["response"] = response
    async with CompletionClient(
        f"{prefix}/test",
        direct=True,
        api_key="test",
        api_base="https://models.example/v1",
        max_tokens=100,
    ) as llm:
        history = [{"role": "user", "content": "test"}]
        turn = await llm.acall(history) if asynchronous else llm.call(history)
        assert turn.tool_calls[0].id == "call"
        restored = LLMResponse.model_validate_json(turn.model_dump_json())
        assert bool(restored.tool_calls[0].native) is (prefix == "gemini")
        assert "SECRET" not in str(dict(restored))
        history = [restored, {"role": "tool", "tool_call_id": "call", "content": "result"}]
        await llm.acall(history) if asynchronous else llm.call(history)
    assert ("SECRET" in requests[1].content.decode()) is (prefix == "gemini")


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_native_output_config_options_and_schema_reach_sdk_wire(wire, asynchronous):
    requests, behavior = wire
    behavior["style"] = "anthropic"
    output = {"effort": "low", "format": {"type": "json_schema", "schema": {"type": "object"}}}
    async with client("anthropic", output_config=output) as llm:
        history = [{"role": "user", "content": "test"}]
        await llm.acall(history) if asynchronous else llm.call(history)
    assert json.loads(requests[0].content)["output_config"] == output


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_conflicting_google_signature_is_terminal(wire, asynchronous):
    from nooa.unifiedllm.errors import ReasoningReplayError

    requests, behavior = wire
    response = reply("chat", text="")
    response["choices"][0]["message"]["tool_calls"] = [
        {
            "id": "call",
            "type": "function",
            "function": {"name": "f", "arguments": "{}"},
            "extra_content": {"google": {"thought_signature": "SECRET"}},
            "provider_specific_fields": {"thought_signature": "OTHER"},
        }
    ]
    behavior["response"] = response
    async with CompletionClient(
        "gemini/test",
        direct=True,
        api_key="test",
        api_base="https://models.example/v1",
        max_tokens=100,
    ) as llm:
        with pytest.raises(ReasoningReplayError, match="Conflicting"):
            history = [{"role": "user", "content": "test"}]
            await llm.acall(history) if asynchronous else llm.call(history)
    assert len(requests) == 1


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("source", ["constructor", "call"])
@pytest.mark.parametrize(
    "key_params,explicit_header,with_other_header",
    [
        ({}, None, True),
        ({"prompt_cache_key": None}, None, True),
        ({"prompt_cache_key": ""}, None, True),
        ({"prompt_cache_key": 17}, None, True),
        ({"prompt_cache_key": "session-key"}, None, True),
        ({"prompt_cache_key": "session-key"}, None, False),
        ({"prompt_cache_key": "session-key"}, "x-session-affinity", True),
        ({"prompt_cache_key": "session-key"}, "X-Session-Affinity", True),
        ({"prompt_cache_key": "session-key"}, "X-SESSION-AFFINITY", True),
    ],
    ids=[
        "absent",
        "null",
        "empty",
        "non-string",
        "generated",
        "no-headers",
        "explicit-lower",
        "explicit-mixed",
        "explicit-upper",
    ],
)
async def test_real_sdk_session_affinity_headers(
    wire, style, asynchronous, source, key_params, explicit_header, with_other_header
):
    """Check the actual SDK wire, including Messages' stripped JSON cache key."""
    requests, behavior = wire
    behavior["style"] = style
    params = copy.deepcopy(key_params)
    headers = {"x-other": "preserved"} if with_other_header else {}
    if explicit_header:
        headers[explicit_header] = "caller-pinned"
    if headers:
        params["extra_headers"] = headers
    before = copy.deepcopy(params)
    history = [{"role": "user", "content": "test"}]
    history_before = copy.deepcopy(history)
    async with client(style, **(params if source == "constructor" else {})) as llm:
        config_before = copy.deepcopy(llm.config)
        for _ in range(2):
            patch = params if source == "call" else {}
            turn = await llm.acall(history, **patch) if asynchronous else llm.call(history, **patch)
            assert turn.content == "42"
        assert llm.config == config_before
    assert params == before
    assert history == history_before
    assert len(requests) == 2
    key = params.get("prompt_cache_key")
    expected = "caller-pinned" if explicit_header else key if isinstance(key, str) and key else None
    for request in requests:
        assert request.headers.get("x-session-affinity") == expected
        assert request.headers.get_list("x-session-affinity") == (
            [] if expected is None else [expected]
        )
        if with_other_header:
            assert request.headers["x-other"] == "preserved"
        body = json.loads(request.content)
        assert "extra_headers" not in body
        if style == "anthropic":
            assert request.url.path == "/v1/messages"
            assert "prompt_cache_key" not in body
        elif "prompt_cache_key" in params:
            assert body["prompt_cache_key"] == key


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_real_sdk_affinity_uses_effective_call_key(wire, style, asynchronous):
    requests, behavior = wire
    behavior["style"] = style
    headers = {"x-other": "preserved"}
    async with client(style, prompt_cache_key="constructor-key", extra_headers=headers) as llm:
        for patch, expected in [
            ({}, "constructor-key"),
            ({"prompt_cache_key": "call-key"}, "call-key"),
            ({"prompt_cache_key": None}, None),
            ({}, "constructor-key"),
        ]:
            history = [{"role": "user", "content": "test"}]
            await llm.acall(history, **patch) if asynchronous else llm.call(history, **patch)
            assert requests[-1].headers.get("x-session-affinity") == expected
            assert requests[-1].headers["x-other"] == "preserved"
        assert llm.config["prompt_cache_key"] == "constructor-key"
        assert llm.config["extra_headers"] == headers
    assert headers == {"x-other": "preserved"}
    assert len(requests) == 4


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("outcome", ["success", "excluded", "exhausted"])
async def test_real_sdk_408_explicit_retry_policy(wire, monkeypatch, style, asynchronous, outcome):
    """408 uses only NOOA's ordinary budget/backoff, never SDK or 429 retries."""
    from anthropic import APIStatusError as AnthropicStatus
    from openai import APIStatusError as OpenAIStatus

    import nooa.unifiedllm.retry as retry_module

    requests, behavior = wire
    behavior.update(style=style, failure=408, always_fail=outcome == "exhausted")
    sleeps = []
    retries = []
    monkeypatch.setattr(retry_module.time, "sleep", sleeps.append)

    async def asleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(retry_module.asyncio, "sleep", asleep)
    policy = RetryConfig(
        max_retries=1,
        rate_limit_extra_retries=4,
        base_delay=0.25,
        rate_limit_base_delay=9,
        jitter_factor=0,
        retryable_status_codes=frozenset() if outcome == "excluded" else frozenset({408}),
        on_retry=lambda attempt, error, delay: retries.append((attempt, error.status_code, delay)),
    )
    policy_before = policy.model_dump()
    params = {"prompt_cache_key": "retry-session", "extra_headers": {"x-other": "preserved"}}
    params_before = copy.deepcopy(params)
    async with client(style, retry_config=policy) as llm:
        history = [{"role": "user", "content": "test"}]
        if outcome == "success":
            turn = (
                await llm.acall(history, **params) if asynchronous else llm.call(history, **params)
            )
            assert turn.content == "42"
        else:
            with pytest.raises(AnthropicStatus if style == "anthropic" else OpenAIStatus) as error:
                await llm.acall(history, **params) if asynchronous else llm.call(history, **params)
            assert error.value.status_code == 408
    assert len(requests) == (1 if outcome == "excluded" else 2)
    assert sleeps == ([] if outcome == "excluded" else [0.25])
    assert retries == ([] if outcome == "excluded" else [(1, 408, 0.25)])
    assert params == params_before
    assert policy.model_dump() == policy_before
    for request in requests:
        assert request.headers["x-session-affinity"] == "retry-session"
        assert request.headers["x-other"] == "preserved"
        assert "extra_headers" not in json.loads(request.content)
    if len(requests) == 2:
        assert requests[0].content == requests[1].content
