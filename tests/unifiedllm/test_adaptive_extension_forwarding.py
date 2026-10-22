# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded provider extensions through real LiteLLM/SDKs and mock HTTP."""

from copy import deepcopy

import pytest
from pydantic import BaseModel, ConfigDict

from nooa.unifiedllm import CompletionClient, RetryConfig
from nooa.unifiedllm.unifiedllm import Tool
from tests.unifiedllm.test_anthropic_native_profile import (
    OriginalResult,
    TypedAgent,
    assert_native,
    http_fixture,
)
from tests.unifiedllm.test_anthropic_native_profile import (
    client as profile_client,
)


def client(style="chat", **kwargs):
    # Explicit route permission is part of the diagnosed caller configuration;
    # production normalization must never force an allowlist globally.
    if style == "anthropic":
        return CompletionClient(
            "anthropic/claude-opus-4-6",
            api_key="fake",
            api_base="https://api.test",
            max_tokens=8192,
            cache_breakpoint=None,
            retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
            allowed_openai_params=["thinking", "output_config"],
            **kwargs,
        )
    return profile_client(style, allowed_openai_params=["thinking", "output_config"], **kwargs)


class OpenToolParameters(BaseModel):
    model_config = ConfigDict(extra="allow")
    value: int = 7
    payload: dict[str, object]


def answer(**kwargs):
    return kwargs


MESSAGES = [{"role": "user", "content": "hello"}]
LEVELS = {
    level: {"thinking": {"type": "adaptive"}, "output_config": {"effort": level}}
    for level in ("low", "high")
}


async def invoke(c, is_async, **kwargs):
    return await c.acall(MESSAGES, **kwargs) if is_async else c.call(MESSAGES, **kwargs)


def assert_raw(body, level):
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": level}
    assert (
        not {"extra_body", "response_format", "output_format", "text_format", "text"} & body.keys()
    )
    assert body["max_tokens"] == 8192


@pytest.mark.parametrize("style", ["chat", "anthropic"])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("mode", [None, "anthropic"])
@pytest.mark.parametrize("level", ["low", "high"])
async def test_raw_adaptive_levels_exact_wire_and_unmutated_inputs(
    monkeypatch, style, is_async, mode, level
):
    bodies, paths = http_fixture(monkeypatch, style)
    levels = deepcopy(LEVELS)
    c = client(style, structured_output_mode=mode, reasoning_levels=levels)
    original = deepcopy(c.config)
    overrides = {"reasoning_level": level}
    try:
        result = await invoke(c, is_async, **overrides)
        assert result.content == '{"value":323}' and result.parsed is None
        assert len(bodies) == 1
        assert_raw(bodies[0], level)
        assert paths == ["/v1/chat/completions" if style == "chat" else "/v1/messages"]
        assert levels == LEVELS and c.config == original
        assert overrides == {"reasoning_level": level}
        assert c._reasoning_config.levels == levels
    finally:
        await c.aclose()


@pytest.mark.parametrize("style", ["chat", "anthropic"])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("level", ["low", "high"])
async def test_typed_then_raw_and_original_nonstrict_raw_tools(monkeypatch, style, is_async, level):
    bodies, _ = http_fixture(monkeypatch, style)
    c = client(style, structured_output_mode="anthropic", **LEVELS[level])
    # This raw caller schema is intentionally open, including an open nested
    # object and optional/defaulted property. Never close it for a raw tool call.
    tool = Tool(
        name="answer",
        description="An open tool",
        callable=answer,
        parameters_model=OpenToolParameters,
    )
    schema = deepcopy(tool.get_parameter_schema())
    assert schema.get("additionalProperties") is not False
    assert schema["properties"]["payload"].get("additionalProperties") is not False
    assert "value" not in schema["required"]
    settings = {"tool_choice": "required"}
    original_settings = deepcopy(settings)
    original_config = deepcopy(c.config)
    try:
        result = await invoke(c, is_async, output_model=OriginalResult)
        assert type(result.parsed) is OriginalResult
        assert_native(bodies[0])
        assert bodies[0]["thinking"] == {"type": "adaptive"}
        assert bodies[0]["output_config"]["effort"] == level
        await invoke(c, is_async)
        assert_raw(bodies[1], level)
        await invoke(c, is_async, tools=[tool], **settings)
        assert_raw(bodies[2], level)
        actual = (
            bodies[2]["tools"][0]["function"]["parameters"]
            if style == "chat"
            else bodies[2]["tools"][0]["input_schema"]
        )
        assert actual == schema
        assert settings == original_settings and c.config == original_config
        assert len(bodies) == 3
    finally:
        await c.aclose()


@pytest.mark.parametrize("level", ["low", "high"])
@pytest.mark.parametrize("style", ["chat", "anthropic"])
async def test_actual_predict_adaptive_typed_then_raw(monkeypatch, level, style):
    bodies, _ = http_fixture(monkeypatch, style)
    c = client(style, structured_output_mode="anthropic", **LEVELS[level])
    config = deepcopy(c.config)
    try:
        result = await TypedAgent(llm=c).compute()
        assert type(result) is OriginalResult and result.value == 323
        assert_native(bodies[0])
        assert bodies[0]["thinking"] == {"type": "adaptive"}
        assert bodies[0]["output_config"]["effort"] == level
        await c.acall(MESSAGES)
        assert_raw(bodies[1], level)
        assert c.config == config
    finally:
        await c.aclose()


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("field", ["thinking", "output_config"])
@pytest.mark.parametrize("equal", [False, True])
async def test_gateway_existing_extra_body_conflicts_explicit(
    monkeypatch, is_async, typed, field, equal
):
    bodies, _ = http_fixture(monkeypatch)
    extension = deepcopy(LEVELS["low"][field])
    extra_value = deepcopy(extension) if equal else {"different": True}
    settings = {
        field: extension,
        "extra_body": {field: extra_value, "custom_gateway": {"keep": [1, 2]}},
    }
    original = deepcopy(settings)
    c = client(structured_output_mode="anthropic")
    try:
        if not equal:
            with pytest.raises(ValueError, match=f"Conflicting {field}") as error:
                await invoke(
                    c, is_async, output_model=OriginalResult if typed else None, **settings
                )
            assert not bodies
            assert not hasattr(error.value, "status_code")
        else:
            await invoke(c, is_async, output_model=OriginalResult if typed else None, **settings)
            assert len(bodies) == 1 and "extra_body" not in bodies[0]
            assert bodies[0]["custom_gateway"] == {"keep": [1, 2]}
            if typed and field == "output_config":
                assert bodies[0][field]["effort"] == "low"
                assert_native(bodies[0])
            else:
                assert bodies[0][field] == extension
        assert settings == original
    finally:
        await c.aclose()


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("extra", [None, [], "invalid"])
async def test_gateway_extra_body_validation_is_local(monkeypatch, is_async, extra):
    bodies, _ = http_fixture(monkeypatch)
    c = client()
    try:
        if extra is None:
            await invoke(c, is_async, extra_body=extra, **LEVELS["high"])
            assert_raw(bodies[0], "high")
        else:
            with pytest.raises(ValueError, match="extra_body must be a mapping"):
                await invoke(c, is_async, extra_body=extra, **LEVELS["high"])
            assert not bodies
    finally:
        await c.aclose()


@pytest.mark.parametrize("is_async", [False, True])
async def test_resolved_provider_not_model_name_and_override_unmutated(monkeypatch, is_async):
    bodies, _ = http_fixture(monkeypatch)
    c = CompletionClient(
        "opaque-model",
        custom_llm_provider="openai",
        api_key="fake",
        base_url="https://api.test/v1",
        cache_breakpoint=None,
        retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
        max_tokens=8192,
        thinking={"type": "disabled"},
        output_config={"effort": "low"},
    )
    original = deepcopy(c.config)
    overrides = {**deepcopy(LEVELS["high"]), "temperature": 0.25, "top_p": 0.9}
    snapshot = deepcopy(overrides)
    try:
        await invoke(c, is_async, **overrides)
        assert_raw(bodies[0], "high")
        assert bodies[0]["temperature"] == 0.25 and bodies[0]["top_p"] == 0.9
        assert c.config == original and overrides == snapshot
        await invoke(c, is_async)
        assert bodies[1]["thinking"] == {"type": "disabled"}
        assert bodies[1]["output_config"] == {"effort": "low"}
    finally:
        await c.aclose()


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("field", ["thinking", "output_config"])
async def test_gateway_explicit_null_extension_is_preserved(monkeypatch, is_async, field):
    bodies, _ = http_fixture(monkeypatch)
    c = client()
    try:
        await invoke(c, is_async, **{field: None})
        assert field in bodies[0] and bodies[0][field] is None
        assert "extra_body" not in bodies[0]
    finally:
        await c.aclose()


@pytest.mark.parametrize("is_async", [False, True])
async def test_adaptive_real_http_error_is_not_swallowed_or_retried(monkeypatch, is_async):
    import httpx

    from nooa.unifiedllm.unifiedllm import _ClientHttp

    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            422, json={"error": {"message": "invalid", "type": "invalid_request_error"}}
        )

    monkeypatch.setattr(
        _ClientHttp,
        "_httpx_hardening",
        staticmethod(lambda: {"transport": httpx.MockTransport(respond), "trust_env": False}),
    )
    c = client(**LEVELS["high"])
    try:
        with pytest.raises(Exception) as error:
            await invoke(c, is_async)
        assert error.value.status_code == 422
        assert len(requests) == 1
    finally:
        await c.aclose()


@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("text", ['{"value":323', '{"value":322}'])
async def test_adaptive_native_original_validation_still_fails(monkeypatch, is_async, text):
    bodies, _ = http_fixture(monkeypatch, text=text)
    c = client(structured_output_mode="anthropic", **LEVELS["low"])
    try:
        with pytest.raises(ValueError):
            await invoke(c, is_async, output_model=OriginalResult)
        assert len(bodies) == 1
        assert bodies[0]["thinking"] == {"type": "adaptive"}
        assert bodies[0]["output_config"]["effort"] == "low"
        assert_native(bodies[0])
    finally:
        await c.aclose()


@pytest.mark.parametrize(
    "provider",
    [
        "anthropic",
        "bedrock",
        "azure",
        "deepseek",
        "azure_ai",
        "fireworks_ai",
        "ragflow",
        "xai",
        "groq",
        "cometapi",
        "hosted_vllm",
    ],
)
def test_routes_outside_bounded_chat_sdk_dispatch_are_not_normalized(monkeypatch, provider):
    import litellm

    # Test the transport decision without constructing unrelated provider
    # clients or changing their own native/HTTP serialization contracts.
    monkeypatch.setattr(litellm, "get_llm_provider", lambda **kw: ("opaque", provider, None, None))
    params = {**deepcopy(LEVELS["high"]), "extra_body": {"keep": [1, 2]}}
    original = deepcopy(params)
    CompletionClient._normalize_chat_extensions(params, "opaque")
    assert params == original


@pytest.mark.parametrize("http_adapter", [False, True])
def test_openai_resolved_http_adapter_switch_not_sdk_client_guess(monkeypatch, http_adapter):
    monkeypatch.setenv("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", str(http_adapter).lower())
    params = {**deepcopy(LEVELS["low"]), "model": "openai/opaque"}
    original = deepcopy(params)
    CompletionClient._normalize_chat_extensions(params, "openai/opaque")
    if http_adapter:
        assert params == original
    else:
        assert params == {"model": "openai/opaque", "extra_body": LEVELS["low"]}
