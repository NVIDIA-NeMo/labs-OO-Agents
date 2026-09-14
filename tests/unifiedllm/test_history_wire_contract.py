# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""History transformations must preserve the request, not merely get HTTP 200.

Synthetic replies exercise real SDK serialization without recording secrets or
spending tokens. The transport captures requests built by today's code, not a
previous recording. JSON container whitespace is irrelevant; argument strings,
part order, reasoning fields, tools and instructions must survive unchanged.
"""

import copy
import json

import httpx
import pytest

from nooa.context_blocks.events import ToolCallEvent, ToolResult, UserEvent
from nooa.context_blocks.formatter import OpenAIProviderFormatter
from nooa.context_blocks.models import BlockMetadata, ResolvedBlock, Role
from nooa.context_blocks.renderer import render_context
from nooa.context_blocks.renderers.cached import CachedBlockFormatter
from nooa.nemo_relay_middleware import _reconcile_messages
from nooa.storage import SQLiteStorageManager
from nooa.unifiedllm import CompletionClient, ResponsesClient, Tool
from nooa.unifiedllm.retry_config import RetryConfig
from nooa.unifiedllm.unifiedllm import _ClientHttp

MODELS = {
    "responses": "openai/gpt-5.6",
    "anthropic": "anthropic/claude-sonnet-4-5",
    "gemini-hub": "openai/gateway-gemini",
    "reasoning-content": "openai/gateway-deepseek",
}
THOUGHT = "Check the original inputs."
SECRET = "synthetic-signature-not-a-real-provider-secret"
ARGUMENTS = '{ "code" : "print(1)" }'


def execute_python(code: str) -> str:
    raise AssertionError("The wire tests must never execute a tool")


TOOL = Tool(name="execute_python", description="Evaluate Python", callable=execute_python)


def _reply(family):
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": TOOL.name, "arguments": ARGUMENTS},
    }
    if family == "responses":
        return {
            "id": "resp_test",
            "object": "response",
            "created_at": 0,
            "status": "completed",
            "model": "gpt-5.6",
            "output": [
                {
                    "id": "rs_1",
                    "type": "reasoning",
                    "encrypted_content": SECRET,
                    "summary": [{"type": "summary_text", "text": THOUGHT}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "id": "fc_1",
                    "name": TOOL.name,
                    "arguments": ARGUMENTS,
                    "status": "completed",
                },
            ],
            "parallel_tool_calls": False,
            "store": False,
            "tools": [],
        }
    if family == "anthropic":
        return {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [
                {"type": "thinking", "thinking": THOUGHT, "signature": SECRET},
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": TOOL.name,
                    "input": json.loads(ARGUMENTS),
                },
            ],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [call],
        "reasoning_content": THOUGHT,
    }
    if family == "gemini-hub":
        call["id"] += "__thought__" + SECRET
        call["provider_specific_fields"] = {"thought_signature": SECRET}
    return {
        "id": "chat_test",
        "object": "chat.completion",
        "created": 0,
        "model": MODELS[family].removeprefix("openai/"),
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls"}],
    }


@pytest.fixture
def wire(monkeypatch):
    """Capture at HTTP, with a hard failure if a client bypasses our transport."""
    bodies = []
    reply = {}

    def respond(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=copy.deepcopy(reply))

    def no_network(*args, **kwargs):
        raise AssertionError("Test escaped its mock HTTP transport")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_network)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", no_network)
    monkeypatch.setattr(
        _ClientHttp,
        "_httpx_hardening",
        staticmethod(lambda: {"transport": httpx.MockTransport(respond)}),
    )
    return bodies, reply


def _client(family, *, different_model=False):
    config = {
        "model": MODELS[family] + ("-other" if different_model else ""),
        "api_key": "test",
        "api_base": "https://provider.test/v1",
        "num_retries": 0,
        "retry_config": RetryConfig(max_retries=0, rate_limit_extra_retries=0),
    }
    if family == "responses":
        return ResponsesClient(**config, cache_breakpoint="openai")
    return CompletionClient(**config)


def _render(turn, state):
    user = UserEvent(content="Check using Python", tag="1")
    call = turn.tool_calls[0]
    result = ToolCallEvent(
        tag="3",
        tool_call_id=call.id,
        name=call.name,
        arguments=json.loads(call.arguments),
        llm_response_id=turn.id,
        result=ToolResult(tool_call_id=call.id, content="1"),
    )
    blocks = [
        ResolvedBlock(
            key="instructions",
            content="Stable instructions",
            role=Role.SYSTEM,
            metadata=BlockMetadata(static=True),
        )
    ]
    blocks += [
        ResolvedBlock(
            key=f"event_{event.tag}",
            content="",
            event=event,
            role=Role.USER if event is user else Role.ASSISTANT,
            metadata=BlockMetadata(tag=event.tag),
        )
        for event in (user, turn, result)
    ]
    blocks.append(
        ResolvedBlock(
            key="live",
            content=f"live-state={state}",
            role=Role.SYSTEM,
            metadata=BlockMetadata(static=False, user_block=True),
        )
    )
    return render_context(
        blocks, block_formatter=CachedBlockFormatter(), provider_formatter=OpenAIProviderFormatter()
    ).output


def _resume(turn, path):
    with SQLiteStorageManager(path) as storage:
        storage.event_backend.store("2", turn)
    with SQLiteStorageManager(path) as storage:
        restored = next(storage.event_backend.all_events())
    assert restored is not turn
    assert restored.parts == turn.parts
    assert restored.replay_scope == turn.replay_scope
    return restored


def _relay(messages):
    # Exercise the production reconcile seam; the real Rust relay round trip
    # itself is already covered by test_nemo_relay_middleware.py.
    public = json.loads(json.dumps([dict(message) for message in messages]))
    assert SECRET not in json.dumps(public)
    return _reconcile_messages(messages, public)


async def _send(client, messages, asynchronous=True):
    if asynchronous:
        return await client.acall(messages, tools=[TOOL])
    return client.call(messages, tools=[TOOL])


def _stable_request(body, family):
    """Remove exactly the known live suffix, retaining every other request field.

    Anthropic can merge the suffix into the user message containing tool results.
    Remove that last text block, not the whole message and its stable results.
    This checks request equality, not a simulation of a provider's cache key.
    """
    stable = copy.deepcopy(body)
    messages = stable["input" if family == "responses" else "messages"]
    last = messages[-1]
    if isinstance(last.get("content"), list):
        suffix = last["content"].pop()
        if not last["content"]:
            messages.pop()
    else:
        suffix = messages.pop()
    assert "live-state=" in json.dumps(suffix), "Expected a trailing live-context block"
    return stable, suffix


def _assert_same_prefix(first, second, family):
    first_stable, _ = _stable_request(first, family)
    second_stable, _ = _stable_request(second, family)
    assert first_stable == second_stable, "History transformation changed the stable HTTP request"


def _assert_replay(body, family):
    if family == "responses":
        reasoning = [item for item in body["input"] if item.get("type") == "reasoning"]
        assert len(reasoning) == 1
        assert reasoning[0]["encrypted_content"] == SECRET
        assert reasoning[0]["summary"] == [{"type": "summary_text", "text": THOUGHT}]
        calls = [item for item in body["input"] if item.get("type") == "function_call"]
        assert calls[0]["arguments"] == ARGUMENTS
        results = [item for item in body["input"] if item.get("type") == "function_call_output"]
        assert len(results) == 1 and results[0]["call_id"] == "call_1"
        assert "prompt_cache_breakpoint" in json.dumps(results[0])
    elif family == "anthropic":
        assistant = next(item for item in body["messages"] if item["role"] == "assistant")
        assert assistant["content"][0] == {
            "type": "thinking",
            "thinking": THOUGHT,
            "signature": SECRET,
        }
        assert assistant["content"][1]["input"] == json.loads(ARGUMENTS)
        results = [
            block
            for item in body["messages"]
            for block in item["content"]
            if block["type"] == "tool_result"
        ]
        assert len(results) == 1 and results[0]["tool_use_id"] == "call_1"
        assert results[0]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    else:
        assistant = next(item for item in body["messages"] if item["role"] == "assistant")
        assert assistant["reasoning_content"] == THOUGHT
        call = assistant["tool_calls"][0]
        assert call["function"]["arguments"] == ARGUMENTS
        expected_id = "call_1" + ("__thought__" + SECRET if family == "gemini-hub" else "")
        assert call["id"] == expected_id
        result = next(item for item in body["messages"] if item["role"] == "tool")
        assert result["tool_call_id"] == expected_id
    assert "nooa_cache_boundary" not in json.dumps(body)


@pytest.mark.parametrize("family", MODELS)
@pytest.mark.parametrize("roundtrip", ["render", "sqlite", "relay", "sqlite-relay"])
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_history_roundtrip_preserves_actual_http_prefix(
    family, roundtrip, asynchronous, wire, tmp_path
):
    bodies, reply = wire
    reply.update(_reply(family))
    async with _client(family) as client:
        turn = await _send(client, [{"role": "user", "content": "Check"}], asynchronous)
        turn.tag = "2"
        archive = turn.model_dump_json()
        await _send(client, _render(turn, "before"), asynchronous)
    restored = _resume(turn, tmp_path / "session.db") if "sqlite" in roundtrip else turn
    messages = _render(restored, "after")
    if "relay" in roundtrip:
        messages = _relay(messages)
    assert any(message is restored for message in messages)
    # A fresh client also guards against accidentally relying on transient SDK state.
    async with _client(family) as client:
        await _send(client, messages, asynchronous)
    assert len(bodies) == 3
    _assert_same_prefix(bodies[1], bodies[2], family)
    assert _stable_request(bodies[1], family)[1] != _stable_request(bodies[2], family)[1]
    _assert_replay(bodies[2], family)
    assert turn.model_dump_json() == archive


@pytest.mark.parametrize("family", MODELS)
@pytest.mark.parametrize("change", ["edit", "switch", "cross-provider"])
async def test_intentional_changes_discard_native_but_keep_readable_reasoning(
    family, change, wire, tmp_path
):
    bodies, reply = wire
    reply.update(_reply(family))
    async with _client(family) as client:
        turn = await _send(client, [{"role": "user", "content": "Check"}])
    turn.tag = "2"
    turn = _resume(turn, tmp_path / "session.db")
    if change == "edit":
        turn = turn.replace_text("Edited answer")
    target = (
        ("anthropic" if family == "responses" else "responses")
        if change == "cross-provider"
        else family
    )
    reply.clear()
    reply.update(_reply(target))
    async with _client(target, different_model=change == "switch") as client:
        await _send(client, _render(turn, "after"))
    assert len(bodies) == 2
    sent = json.dumps(bodies[-1])
    assert SECRET not in sent
    assert THOUGHT in sent
    assert "reasoning_content" not in sent
    assert "encrypted_content" not in sent
    if change == "edit":
        assert "Edited answer" in sent


@pytest.mark.parametrize("family", MODELS)
@pytest.mark.parametrize("corruption", ["native", "arguments", "instructions", "tools"])
async def test_wire_assertions_detect_corruption_even_when_http_succeeds(family, corruption, wire):
    bodies, reply = wire
    reply.update(_reply(family))
    async with _client(family) as client:
        turn = await _send(client, [{"role": "user", "content": "Check"}])
        turn.tag = "2"
        await _send(client, _render(turn, "before"))
    original = bodies[-1]
    _assert_replay(original, family)
    # Mutate only the captured outgoing request, not the accepted canned reply.
    # This negative control would stay green if we inspected response fixtures.
    needle = {
        "native": THOUGHT if family == "reasoning-content" else SECRET,
        "arguments": "print(1)",
        "instructions": "Stable instructions",
        "tools": "Evaluate Python",
    }[corruption]
    encoded = json.dumps(original)
    assert needle in encoded
    damaged = json.loads(encoded.replace(needle, "CORRUPTED"))
    with pytest.raises(AssertionError):
        _assert_same_prefix(original, damaged, family)
    if corruption == "native":
        with pytest.raises(AssertionError):
            _assert_replay(damaged, family)
