# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit native carrier through Predict, LiteLLM, SDKs and mock HTTP only."""

import json
from copy import deepcopy

import httpx
import litellm
import pytest
from pydantic import BaseModel, ConfigDict, Field, RootModel, ValidationError, field_validator

from nooa import Agent, no_trace
from nooa.config import ModelConfig
from nooa.config.strategy_config import PredictConfig
from nooa.decorators import strategy
from nooa.strategies.predict import PredictStrategy
from nooa.unifiedllm import CompletionClient, ResponsesClient, RetryConfig, connect, registry
from nooa.unifiedllm.connect._baseline import typed_baseline
from nooa.unifiedllm.connect._records import public_record, text_baseline_passed
from nooa.unifiedllm.structured_output import anthropic_output_format
from nooa.unifiedllm.unifiedllm import _ClientHttp
from tests.unifiedllm.connect.connect_http import response_body


class OriginalResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int = Field(ge=300)

    @field_validator("value")
    @classmethod
    def exact_answer(cls, value):
        if value != 323:
            raise ValueError("original validator requires 323")
        return value


class TypedAgent(Agent):
    @no_trace
    @strategy(PredictStrategy(config=PredictConfig(max_retries=1)))
    async def compute(self) -> OriginalResult:
        """Return 17 * 19 in value."""
        ...


def http_fixture(monkeypatch, style="chat", text='{"value":323}', finish="stop", tool=False):
    bodies = []
    paths = []
    monkeypatch.setenv("LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS", "True")

    def respond(request):
        bodies.append(json.loads(request.content))
        paths.append(request.url.path)
        body = response_body(style, text)
        if style == "chat":
            body["choices"][0]["finish_reason"] = finish
            if tool:
                body["choices"][0]["message"]["tool_calls"] = [
                    {
                        "id": "call",
                        "type": "function",
                        "function": {"name": "answer", "arguments": '{"value":323}'},
                    }
                ]
        else:
            body["stop_reason"] = {
                "stop": "end_turn",
                "length": "max_tokens",
                "tool_calls": "tool_use",
            }.get(finish, "refusal")
            if tool:
                body["content"] = [
                    {"type": "tool_use", "id": "call", "name": "answer", "input": {"value": 323}}
                ]
        return httpx.Response(200, json=body)

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        _ClientHttp,
        "_httpx_hardening",
        staticmethod(lambda: {"transport": transport, "trust_env": False}),
    )
    return bodies, paths


def client(style="chat", **kwargs):
    return CompletionClient(
        "openai/azure/anthropic/claude-opus-5-5" if style == "chat" else "anthropic/model",
        api_key="fake",
        api_base="https://api.test/v1" if style == "chat" else "https://api.test",
        max_tokens=8192,
        cache_breakpoint=None,
        retry_config=RetryConfig(max_retries=0, rate_limit_extra_retries=0),
        **kwargs,
    )


def assert_native(body, value_type="integer"):
    assert "structured_output_mode" not in json.dumps(body)
    assert (
        not {"response_format", "text", "text_format", "output_format", "tools", "tool_choice"}
        & body.keys()
    )
    assert "extra_body" not in body
    assert body["max_tokens"] == 8192
    fmt = body["output_config"]["format"]
    assert set(fmt) == {"type", "schema"}  # No invented name/strict wrapper.
    assert fmt["type"] == "json_schema"
    schema = fmt["schema"]
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["value"]
    assert set(schema["properties"]) == {"value"}
    assert schema["properties"]["value"]["type"] == value_type


@pytest.mark.parametrize("style", ["chat", "anthropic"])
async def test_actual_predict_original_model_and_native_http(monkeypatch, style):
    bodies, paths = http_fixture(monkeypatch, style)
    c = client(style, structured_output_mode="anthropic")
    try:
        result = await TypedAgent(llm=c).compute()
        assert type(result) is OriginalResult and result.value == 323
        assert len(bodies) == 1
        assert_native(bodies[0])
        assert paths == ["/v1/chat/completions" if style == "chat" else "/v1/messages"]
        assert (
            bodies[0]["output_config"]["format"]["schema"]["properties"]["value"]["minimum"] == 300
        )
        assert c.get_context_limits().reserved_output_tokens == 8192
        assert "structured_output_mode" not in c.config
    finally:
        await c.aclose()


@pytest.mark.parametrize("style", ["chat", "anthropic"])
@pytest.mark.parametrize("is_async", [False, True])
async def test_sync_async_compilation_and_no_format_without_output_model(
    monkeypatch, style, is_async
):
    bodies, _ = http_fixture(monkeypatch, style)
    c = client(style, structured_output_mode="anthropic", output_config={"effort": "high"})
    try:
        messages = [{"role": "user", "content": "hello"}]
        call = c.acall if is_async else c.call
        result = (
            await call(messages, output_model=OriginalResult)
            if is_async
            else call(messages, output_model=OriginalResult)
        )
        assert type(result.parsed) is OriginalResult
        assert_native(bodies[0])
        assert bodies[0]["output_config"]["effort"] == "high"
        # None output_model never injects a format; profile and cache are independent.
        await call(messages) if is_async else call(messages)
        assert "format" not in bodies[1].get("output_config", {})
        assert "response_format" not in bodies[1]
        assert "cache_control" not in json.dumps(bodies)
        assert c.structured_output_mode == "anthropic" and c.cache_breakpoint is None
    finally:
        await c.aclose()


@pytest.mark.parametrize("style", ["chat", "anthropic"])
@pytest.mark.parametrize(
    "text,finish,tool",
    [
        ('{"value":322}', "stop", False),
        ('{"value":"wrong"}', "stop", False),
        ('{"value":323,"extra":1}', "stop", False),
        ('{"value":323', "stop", False),
        ("323", "stop", False),
        ('```json\n{"value":323}\n```', "stop", False),
        ('{"value":323}', "length", False),
        ("", "tool_calls", True),
    ],
)
async def test_original_validation_and_complete_final_stop_required(
    monkeypatch, style, text, finish, tool
):
    bodies, _ = http_fixture(monkeypatch, style, text, finish, tool)
    c = client(style, structured_output_mode="anthropic")
    try:
        with pytest.raises(ValueError):
            await c.acall([{"role": "user", "content": "hello"}], output_model=OriginalResult)
        assert len(bodies) == 1
    finally:
        await c.aclose()


@pytest.mark.parametrize("mode", [None, "auto"])
async def test_default_gpt_generic_carrier_is_unchanged(monkeypatch, mode):
    bodies, _ = http_fixture(monkeypatch)
    c = CompletionClient(
        "openai/gpt-4o-mini",
        api_key="fake",
        api_base="https://api.test/v1",
        max_tokens=8192,
        structured_output_mode=mode,
    )
    try:
        response, result = await typed_baseline(c)
        assert result == 323 and response.finish_reason == "stop"
        assert bodies[0]["response_format"]["json_schema"]["strict"] is True
        assert "output_config" not in bodies[0]
        assert not {"tools", "tool_choice", "structured_output_mode"} & bodies[0].keys()
    finally:
        await c.aclose()


@pytest.mark.parametrize(
    "settings",
    [
        {"output_config": {"format": {"type": "json_schema", "schema": {}}}},
        {"extra_body": {"output_config": {"format": None}}},
        {"response_format": {"type": "json_object"}},
        {"extra_body": {"response_format": {"type": "json_schema"}}},
        {"text": {"format": {"type": "json_schema"}}},
        {"extra_body": {"output_format": {}}},
    ],
)
async def test_explicit_format_conflicts_fail_before_http(monkeypatch, settings):
    original = deepcopy(settings)
    bodies, _ = http_fixture(monkeypatch)
    c = client(structured_output_mode="anthropic", **settings)
    try:
        with pytest.raises(ValueError, match="conflict"):
            await c.acall([{"role": "user", "content": "hello"}], output_model=OriginalResult)
        assert not bodies and settings == original
    finally:
        await c.aclose()


async def test_no_output_model_preserves_tool_only_replies(monkeypatch):
    bodies, _ = http_fixture(monkeypatch, text="", finish="tool_calls", tool=True)
    c = client(structured_output_mode="anthropic")
    try:
        response = await c.acall([{"role": "user", "content": "hello"}])
        assert len(response.tool_calls) == 1 and response.finish_reason == "tool_calls"
        assert "output_config" not in bodies[0]
    finally:
        await c.aclose()


def test_schema_compiler_preserves_original_constraints_and_model():
    class Nested(BaseModel):
        answer: OriginalResult

    original = deepcopy(Nested.model_json_schema())
    schema = anthropic_output_format(Nested)["schema"]
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["OriginalResult"]["additionalProperties"] is False
    assert schema["$defs"]["OriginalResult"]["properties"]["value"]["minimum"] == 300
    assert Nested.model_json_schema() == original
    with pytest.raises(ValueError, match="object-root"):
        anthropic_output_format(RootModel[int])


@pytest.mark.parametrize("style", ["responses", "systemone", "openai"])
def test_native_plan_rejects_other_carriers_without_io(style):
    with pytest.raises(ValueError, match="Completion-only"):
        connect.plan(
            "local", "model", style, "https://api.test/v1", "", structured_output_mode="anthropic"
        )
    with pytest.raises(ValidationError):
        ModelConfig(api_style=style, structured_output_mode="anthropic")


def test_constructors_reject_invalid_profile_and_responses_native():
    with pytest.raises(ValueError, match="structured_output_mode"):
        CompletionClient("openai/model", structured_output_mode="bogus")
    with pytest.raises(ValueError, match="CompletionClient"):
        ResponsesClient("openai/model", structured_output_mode="anthropic")
    with pytest.raises(ValueError, match="Decision"):
        registry.client_from_config(
            "local",
            {
                "client_type": "decision",
                "endpoint": "https://api.test/decisions",
                "structured_output_mode": "anthropic",
            },
        )


@pytest.mark.parametrize(
    "replacement",
    [
        {"model": "openai/other"},
        {"api_base": "https://other.test/v1"},
        {"client_type": "responses"},
    ],
)
def test_registry_route_override_does_not_inherit_native(replacement):
    entry = connect.plan(
        "local",
        "model",
        "chat",
        "https://api.test/v1",
        "",
        reply_tokens=8192,
        structured_output_mode="anthropic",
    ).entry
    c = registry.client_from_config("local", entry, api_key="fake", **replacement)
    try:
        assert c.structured_output_mode is None
    finally:
        c.close()


async def test_call_override_requires_explicit_replacement_profile(monkeypatch):
    bodies, _ = http_fixture(monkeypatch)
    c = client(structured_output_mode="anthropic")
    try:
        with pytest.raises(ValueError, match="cannot inherit"):
            await c.acall(
                [{"role": "user", "content": "hello"}],
                output_model=OriginalResult,
                model="openai/other",
            )
        assert not bodies
        # Explicit auto opts out, consumed by NOOA rather than sent to SDK.
        await c.acall(
            [{"role": "user", "content": "hello"}],
            output_model=OriginalResult,
            structured_output_mode="auto",
        )
        assert "response_format" in bodies[0] and "structured_output_mode" not in bodies[0]
    finally:
        await c.aclose()


async def test_runtime_final_wire_guard_rejects_dropped_schema_before_send(monkeypatch):
    bodies, _ = http_fixture(monkeypatch, "anthropic")
    # Existing LiteLLM's global drop mode can drop output_config for an unknown
    # model. Native profile must error before HTTP, not validate a fake reply.
    monkeypatch.setattr(litellm, "drop_params", True)
    c = client("anthropic", structured_output_mode="anthropic")
    try:
        with pytest.raises(Exception, match="changed or dropped"):
            await c.acall([{"role": "user", "content": "hello"}], output_model=OriginalResult)
        assert not bodies
    finally:
        await c.aclose()


@pytest.mark.parametrize("style", ["chat", "anthropic"])
async def test_connect_native_wire_settings_evidence_and_media(monkeypatch, style):
    bodies = []
    monkeypatch.setenv("LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS", "True")

    def respond(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=response_body(style, '{"value":323}' if len(bodies) == 1 else '{"value":true}'),
        )

    monkeypatch.setattr(
        _ClientHttp,
        "_httpx_hardening",
        staticmethod(lambda: {"transport": httpx.MockTransport(respond), "trust_env": False}),
    )
    proposal = connect.plan(
        "local",
        "model",
        style,
        "https://api.test/v1",
        "",
        reply_tokens=8192,
        cache_enabled=False,
        structured_output_mode="anthropic",
        test_media_inputs=["image"],
    )
    result = await connect.check_stage(proposal, "media", api_key="fake")
    routing = result.entry["provenance"]["probes"]["routing"]
    assert text_baseline_passed(routing), routing
    assert routing["settings_sent"] is True
    assert routing["structured_output_carrier"] == "output_config.format"
    assert public_record(routing)["structured_output_schema_observed"] is True
    media = result.entry["provenance"]["probes"]["media:image"]
    assert media["typed_result_validated"] and media["output_settings_observed"]
    assert len(bodies) == 2
    assert_native(bodies[0])
    assert_native(bodies[1], "boolean")
    assert "schema" not in public_record(routing)


@pytest.mark.parametrize(
    "default,mode", [(None, "anthropic"), ("anthropic", None), ("anthropic", "auto")]
)
async def test_reply_cap_recovery_preserves_explicit_carrier_and_opt_out(
    monkeypatch, default, mode
):
    from nooa.unifiedllm.limits import reduced_reply_params

    bodies, _ = http_fixture(monkeypatch)
    c = client(structured_output_mode=default)
    original = {"structured_output_mode": mode, "max_tokens": 8192}
    try:
        recovered = reduced_reply_params(c, original, 4096)
        assert recovered["structured_output_mode"] == mode
        assert recovered["max_tokens"] == 4096
        assert original == {"structured_output_mode": mode, "max_tokens": 8192}
        for params in (original, recovered):
            await c.acall(
                [{"role": "user", "content": "hello"}], output_model=OriginalResult, **params
            )
        assert [b["max_tokens"] for b in bodies] == [8192, 4096]
        for body in bodies:
            assert ("output_config" in body) == (mode == "anthropic")
            assert ("response_format" in body) == (mode != "anthropic")
            assert "structured_output_mode" not in body
    finally:
        await c.aclose()


@pytest.mark.parametrize("in_extra", [False, True])
@pytest.mark.parametrize("is_async", [False, True])
async def test_native_typed_surrogate_tools_rejected_before_send(monkeypatch, in_extra, is_async):
    tools = [
        {
            "type": "function",
            "function": {
                "name": "json_surrogate",
                "parameters": OriginalResult.model_json_schema(),
            },
        }
    ]
    settings = {"tools": tools, "tool_choice": "required"}
    bodies, _ = http_fixture(monkeypatch)
    c = client(
        structured_output_mode="anthropic", **({"extra_body": settings} if in_extra else settings)
    )
    try:
        with pytest.raises(ValueError, match="conflict"):
            if is_async:
                await c.acall([{"role": "user", "content": "hello"}], output_model=OriginalResult)
            else:
                c.call([{"role": "user", "content": "hello"}], output_model=OriginalResult)
        assert not bodies
    finally:
        await c.aclose()


@pytest.mark.parametrize("style", ["chat", "anthropic"])
@pytest.mark.parametrize(
    "text,finish,tool",
    [
        ('{"value":322}', "stop", False),
        ('{"value":"wrong"}', "stop", False),
        ('{"value":323', "stop", False),
        ('{"value":323}', "length", False),
        ("", "tool_calls", True),
    ],
)
async def test_actual_native_predict_baseline_failures_never_pass(
    monkeypatch, style, text, finish, tool
):
    bodies, _ = http_fixture(monkeypatch, style, text, finish, tool)
    proposal = connect.plan(
        "local",
        "model",
        style,
        "https://api.test/v1",
        "",
        reply_tokens=8192,
        structured_output_mode="anthropic",
        cache_enabled=False,
    )
    result = await connect.check_stage(proposal, "routing", api_key="fake")
    record = result.entry["provenance"]["probes"]["routing"]
    assert not text_baseline_passed(record)
    assert len(bodies) == 1
    assert_native(bodies[0])
    assert record["structured_output_schema_observed"] is True
    if text == '{"value":322}':
        assert record["typed_result_validated"] and not record["answer_correct"]
    else:
        assert not record["typed_result_validated"]


@pytest.mark.parametrize("mutation", ["type", "required", "additionalProperties"])
async def test_connect_native_schema_settings_not_bypassed_by_typed_result(monkeypatch, mutation):
    from contextlib import nullcontext

    # Isolate Connect's independent schema-settings observation from the
    # runtime's pre-send guard. Even a validated reply must not bypass it.
    monkeypatch.setattr(
        CompletionClient, "_native_output_request_scope", lambda *a, **k: nullcontext()
    )
    original = CompletionClient._set_output_format

    def altered(params, output_model, model, native):
        original(params, output_model, model, native)
        if native and output_model is not None:
            schema = params["extra_body"]["output_config"]["format"]["schema"]
            if mutation == "type":
                schema["properties"]["value"]["type"] = "string"
            elif mutation == "required":
                schema["required"] = []
            else:
                schema["additionalProperties"] = True

    monkeypatch.setattr(CompletionClient, "_set_output_format", staticmethod(altered))
    bodies, _ = http_fixture(monkeypatch)
    result = await connect.check_stage(
        connect.plan(
            "local",
            "model",
            "chat",
            "https://api.test/v1",
            "",
            reply_tokens=8192,
            structured_output_mode="anthropic",
        ),
        "routing",
        api_key="fake",
    )
    record = result.entry["provenance"]["probes"]["routing"]
    # The fake server returns a valid integer, but the wrong requested schema
    # must not become settings-sent evidence for the original Predict contract.
    assert record["typed_result_validated"] and record["answer_correct"]
    assert not record["settings_sent"] and not record["structured_output_schema_observed"]
    assert not text_baseline_passed(record) and len(bodies) == 1


async def test_registry_native_output_config_effort_and_conflict_are_preserved(monkeypatch):
    bodies, _ = http_fixture(monkeypatch)
    entry = connect.plan(
        "local",
        "model",
        "chat",
        "https://api.test/v1",
        "",
        reply_tokens=8192,
        structured_output_mode="anthropic",
    ).entry
    entry["output_config"] = {"effort": "high"}
    c = registry.client_from_config("local", entry, api_key="fake")
    try:
        await c.acall([{"role": "user", "content": "hello"}], output_model=OriginalResult)
        assert bodies[0]["output_config"]["effort"] == "high"
    finally:
        await c.aclose()
    entry["output_config"]["format"] = {"type": "json_schema", "schema": {}}
    c = registry.client_from_config("local", entry, api_key="fake")
    try:
        with pytest.raises(ValueError, match="conflict"):
            await c.acall([{"role": "user", "content": "hello"}], output_model=OriginalResult)
        assert len(bodies) == 1
    finally:
        await c.aclose()
