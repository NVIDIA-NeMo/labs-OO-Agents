# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual core SDK HTTP serialization/parser tests for native projections."""

import json
from pathlib import Path

import anthropic
import httpx
import openai
import pytest
from pydantic import BaseModel

from nooa.llm_types import AssistantReasoning, AssistantText, CacheBoundary, LLMResponse, ToolCall
from nooa.unifiedllm.adapters._native_projection import _project_request
from nooa.unifiedllm.adapters._native_response import _normalize_response, _parse_typed
from nooa.unifiedllm.adapters._native_settings import _resolve_native
from nooa.unifiedllm.adapters._native_usage import _numeric_usage
from nooa.unifiedllm.errors import FailureKind, LLMFailure
from nooa.unifiedllm.request import LLMRequest
from nooa.unifiedllm.tools import create_tool_from_callable

FAMILIES = ["openai-chat", "openai-responses", "anthropic-messages"]


class Answer(BaseModel):
    value: int


def lookup(query: str):
    """Look up a query."""
    return query


def settings(family, **options):
    return _resolve_native(
        family,
        "alias",
        dict(
            model_name="model",
            api_key="fixture-key",
            api_base="https://fixture.invalid/v1",
            max_tokens=100,
            media_inputs=["image"] if family == "openai-chat" else ["image", "pdf"],
            **options,
        ),
        {},
    )[0]


def payload(family, content='{"value":42}', finish=None):
    if family == "openai-chat":
        return {
            "id": "c1",
            "object": "chat.completion",
            "created": 1,
            "model": "model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish or "stop",
                }
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
        }
    if family == "openai-responses":
        return {
            "id": "r1",
            "object": "response",
            "created_at": 1,
            "model": "model",
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "status": "incomplete" if finish else "completed",
            "incomplete_details": {"reason": "max_output_tokens"} if finish == "length" else None,
            "output": [
                {
                    "id": "m1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": content, "annotations": []}],
                }
            ],
            "usage": {
                "input_tokens": 7,
                "output_tokens": 2,
                "total_tokens": 9,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
    return {
        "id": "m1",
        "type": "message",
        "role": "assistant",
        "model": "model",
        "content": [{"type": "text", "text": content}],
        "stop_reason": "max_tokens" if finish == "length" else finish or "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 7,
            "output_tokens": 2,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 4,
            "cache_creation": {"ephemeral_5m_input_tokens": 4, "ephemeral_1h_input_tokens": 0},
        },
    }


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("choice", ["auto", "required", "none", "lookup"])
async def test_actual_sdk_projected_body_and_typed_reply(family, mode, choice):
    seen = []

    def handler(request):
        path = {
            "openai-chat": "/v1/chat/completions",
            "openai-responses": "/v1/responses",
            "anthropic-messages": "/v1/messages",
        }[family]
        assert request.method == "POST" and str(request.url) == "https://fixture.invalid" + path
        assert request.headers[
            "x-api-key" if family == "anthropic-messages" else "authorization"
        ] == ("fixture-key" if family == "anthropic-messages" else "Bearer fixture-key")
        assert request.headers["x-private-host"] == "fixture"
        assert b"fixture-key" not in request.content
        body = json.loads(request.content)
        seen.append(body)
        expected = json.loads(
            (Path(__file__).parent / "native_wire_goldens" / (family + ".json")).read_text()
        )
        if choice != "auto":
            expected["tool_choice"] = (
                (
                    {"type": "any" if choice == "required" else "none"}
                    if choice != "lookup"
                    else {"type": "tool", "name": "lookup"}
                )
                if family == "anthropic-messages"
                else choice
                if choice != "lookup"
                else {"type": "function", "function": {"name": "lookup"}}
                if family == "openai-chat"
                else {"type": "function", "name": "lookup"}
            )
        assert body == expected
        return httpx.Response(200, json=payload(family), headers={"x-response-cost": "0.03"})

    config = settings(family, host_profile={"cost_header": "x-response-cost"})
    request = LLMRequest(
        messages=[{"role": "system", "content": "stable"}, {"role": "user", "content": "question"}],
        tools=[create_tool_from_callable(lookup)],
        tool_choice=choice,
        output_type=Answer,
    )
    kwargs = _project_request(request, config)
    cls = (
        (anthropic.Anthropic if mode == "sync" else anthropic.AsyncAnthropic)
        if family == "anthropic-messages"
        else (openai.OpenAI if mode == "sync" else openai.AsyncOpenAI)
    )
    http = (
        httpx.Client(transport=httpx.MockTransport(handler))
        if mode == "sync"
        else httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    sdk = cls(
        api_key="fixture-key",
        base_url="https://fixture.invalid"
        if family == "anthropic-messages"
        else "https://fixture.invalid/v1",
        http_client=http,
        max_retries=0,
        default_headers={"x-private-host": "fixture"},
    )
    resource = (
        sdk.messages
        if family == "anthropic-messages"
        else sdk.chat.completions
        if family == "openai-chat"
        else sdk.responses
    )
    try:
        raw = (
            resource.with_raw_response.create(**kwargs)
            if mode == "sync"
            else await resource.with_raw_response.create(**kwargs)
        )
        reply = raw.parse()
        numeric = _numeric_usage(family, reply, raw.headers, cost_header="x-response-cost")
        response = _parse_typed(_normalize_response(family, "model", reply, numeric), Answer)
        assert response.parsed == Answer(value=42)
        assert response.raw_response is reply
        assert response.usage.input_tokens == (14 if family == "anthropic-messages" else 7)
        assert response.usage.cost_usd == 0.03
        assert numeric.cost_source == "gateway_header" and len(seen) == 1
    finally:
        if mode == "sync":
            sdk.close()
        else:
            await sdk.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_paid_parse_failure_limit_and_unknown_cost(family):
    cls = (
        openai.types.chat.ChatCompletion
        if family == "openai-chat"
        else openai.types.responses.Response
        if family == "openai-responses"
        else anthropic.types.Message
    )
    for finish in (None, "length"):
        reply = cls.model_validate(payload(family, content="not json", finish=finish))
        numeric = _numeric_usage(family, reply)
        assert numeric.cost_usd is None and numeric.cost_source == "unknown"
        response = _normalize_response(family, "model", reply, numeric)
        with pytest.raises(LLMFailure) as caught:
            _parse_typed(response, Answer)
        assert caught.value.kind == (
            FailureKind.OUTPUT_LIMIT if finish else FailureKind.BAD_REQUEST
        )
        assert caught.value.__cause__ is not None
    reply = cls.model_validate(payload(family, content=" "))
    assert _normalize_response(family, "model", reply, _numeric_usage(family, reply)).is_empty


@pytest.mark.parametrize("family", FAMILIES)
def test_portable_tool_args_and_result_linkage(family):
    arguments = '{ "query" : "x" }'
    history = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_bad-id",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": arguments},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_bad-id", "content": "result"},
    ]
    wire = _project_request(
        LLMRequest(messages=history),
        settings(family, host_profile={"tool_id_pattern": "alphanumeric"}),
    )
    items = wire.get("input", wire.get("messages"))
    if family == "openai-chat":
        call = items[0]["tool_calls"][0]
        assert call["function"]["arguments"] == arguments
        assert call["id"] == items[1]["tool_call_id"]
    elif family == "openai-responses":
        call = next(item for item in items if item.get("type") == "function_call")
        assert call["arguments"] == arguments
        assert call["call_id"] == items[-1]["call_id"]
    else:
        call = items[0]["content"][-1]
        assert call["input"] == {"query": "x"}
        assert call["id"] == items[1]["content"][0]["tool_use_id"]
    assert history[0]["tool_calls"][0]["id"] == "call_bad-id"


def test_anthropic_signed_replay_scope_cross_family_and_malformed():
    blocks = [
        {"type": "thinking", "thinking": "private thought", "signature": "private-signed"},
        {"type": "redacted_thinking", "data": "private-encrypted"},
        {"type": "text", "text": "answer"},
    ]
    data = payload("anthropic-messages")
    data["content"] = blocks
    reply = anthropic.types.Message.model_validate(data)
    response = _normalize_response(
        "anthropic-messages",
        "model",
        reply,
        _numeric_usage("anthropic-messages", reply),
        replay_scope=settings("anthropic-messages").replay_scope,
    )
    restored = LLMResponse.model_validate_json(response.model_dump_json())
    before = restored.model_dump_json()
    wire = _project_request(LLMRequest(messages=[restored]), settings("anthropic-messages"))
    assert wire["messages"] == [{"role": "assistant", "content": blocks}]
    cross = _project_request(LLMRequest(messages=[restored]), settings("openai-chat"))
    assert "private-signed" not in json.dumps(cross) and "private-encrypted" not in json.dumps(
        cross
    )
    assert restored.model_dump_json() == before
    assert (
        _project_request(LLMRequest(messages=[restored]), settings("anthropic-messages"))[
            "messages"
        ]
        == wire["messages"]
    )
    broken = LLMResponse(
        parts=(
            AssistantReasoning(
                native={"family": "anthropic-messages", "version": 1, "wire": "bad"}
            ),
        ),
        replay_scope=settings("anthropic-messages").replay_scope,
    )
    with pytest.raises(LLMFailure):
        _project_request(LLMRequest(messages=[broken]), settings("openai-chat"))


def test_mid_history_instructions_and_stable_cache():
    request = LLMRequest(
        messages=[{"role": "user", "content": "first"}, {"role": "system", "content": "middle"}]
    )
    with pytest.raises(LLMFailure):
        _project_request(request, settings("anthropic-messages"))
    wire = _project_request(request, settings("openai-responses"))
    assert [item["role"] for item in wire["input"]] == ["user", "system"]
    request = LLMRequest(
        messages=[
            {"role": "system", "content": "stable"},
            CacheBoundary(),
            {"role": "user", "content": "dynamic"},
        ]
    )
    wire = _project_request(request, settings("anthropic-messages", cache_breakpoint="anthropic"))
    assert wire["system"] == [
        {"type": "text", "text": "stable", "cache_control": {"type": "ephemeral"}}
    ]
    assert "cache_control" not in wire["messages"][0]["content"][0]


def test_native_replay_rejects_mixed_foreign_and_uncovered_chat_parts():
    first = AssistantText(
        text="first",
        native={
            "family": "openai-responses",
            "version": 1,
            "wire": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "first"}],
            },
        },
    )
    foreign = AssistantText(
        text="must not drop",
        native={
            "family": "anthropic-messages",
            "version": 1,
            "wire": {"type": "text", "text": "must not drop"},
        },
    )
    mixed = LLMResponse(
        parts=(first, foreign), replay_scope=settings("openai-responses").replay_scope
    )
    with pytest.raises(LLMFailure, match="Incomplete native replay"):
        _project_request(LLMRequest(messages=[mixed]), settings("openai-responses"))
    carrier = AssistantText(
        text="first",
        native={
            "family": "openai-chat",
            "version": 1,
            "wire": {"role": "assistant", "content": "first", "opaque": "state"},
        },
    )
    appended = LLMResponse(
        parts=(carrier, ToolCall(id="newcall", name="lookup", arguments="{}")),
        replay_scope=settings("openai-chat").replay_scope,
    )
    with pytest.raises(LLMFailure, match="Incomplete native Chat replay"):
        _project_request(LLMRequest(messages=[appended]), settings("openai-chat"))


def test_normalized_chat_complete_opaque_carrier_covers_reasoning_text_and_tools():
    data = payload("openai-chat", content="answer")
    message = data["choices"][0]["message"]
    message.update(
        reasoning_content="thought",
        opaque="state",
        tool_calls=[
            {
                "id": "call1",
                "type": "function",
                "function": {"name": "lookup", "arguments": '{ "query": "x" }'},
            }
        ],
    )
    reply = openai.types.chat.ChatCompletion.model_validate(data)
    response = _normalize_response(
        "openai-chat",
        "model",
        reply,
        _numeric_usage("openai-chat", reply),
        replay_scope=settings("openai-chat").replay_scope,
    )
    wire = _project_request(LLMRequest(messages=[response]), settings("openai-chat"))
    assert wire["messages"] == [message]


@pytest.mark.parametrize("read,creation", [(None, None), ("bad", 4), (3, -1), (True, 0), (0, None)])
def test_anthropic_unknown_cache_components_do_not_prove_total_input(read, creation):
    usage = {"input_tokens": 7, "output_tokens": 2}
    if read is not None:
        usage["cache_read_input_tokens"] = read
    if creation is not None:
        usage["cache_creation_input_tokens"] = creation
    numeric = _numeric_usage("anthropic-messages", {"usage": usage})
    assert numeric.input_tokens is None and numeric.total_tokens is None
    assert numeric.output_tokens == 2


def test_anthropic_explicit_zero_cache_components_prove_uncached_only_total():
    numeric = _numeric_usage(
        "anthropic-messages",
        {
            "usage": {
                "input_tokens": 7,
                "output_tokens": 2,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            }
        },
    )
    assert numeric.input_tokens == 7 and numeric.total_tokens == 9
