# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Responses pass directly through history; replacing a message discards native state."""

import copy
import inspect
import json
from types import SimpleNamespace

import pytest
from litellm.types.utils import Choices, Message, ModelResponse

from nooa.llm_types import LLMResponse
from nooa.unifiedllm import CompletionClient, ReasoningCompletionClient, ResponsesClient, UnifiedLLM
from nooa.unifiedllm.chat_parts import capture_chat_parts
from nooa.unifiedllm.fake import FakeLLMClient
from nooa.unifiedllm.replay_state import replay_scope
from nooa.unifiedllm.response_parts import capture_parts


@pytest.mark.parametrize(
    "client_type",
    [UnifiedLLM, CompletionClient, ResponsesClient, ReasoningCompletionClient, FakeLLMClient],
)
@pytest.mark.parametrize("method", ["call", "acall"])
def test_clients_need_no_private_turn_lookup(client_type, method):
    assert "turns" not in inspect.signature(getattr(client_type, method)).parameters


@pytest.mark.asyncio
@pytest.mark.parametrize("async_call", [False, True])
async def test_fake_projects_response_objects_as_portable_messages(async_call):
    scope = replay_scope("anthropic/claude-sonnet-4", "chat", {})
    turn = LLMResponse(
        parts=capture_chat_parts(
            {
                "role": "assistant",
                "content": "answer",
                "thinking_blocks": [
                    {
                        "type": "thinking",
                        "thinking": "portable thought",
                        "signature": "private-signature",
                    }
                ],
            },
            scope,
        ),
        replay_scope=scope,
    )
    messages = [turn]
    async with FakeLLMClient() as client:
        if async_call:
            await client.acall(messages)
        else:
            client.call(messages)
        assert client.last_messages == [
            {"role": "assistant", "content": "portable thought\n\nanswer"}
        ]
        assert messages[0] is turn
        assert turn.parts[0].native is not None


def test_hand_built_null_content_call_uses_the_same_public_builder():
    from nooa.context_blocks.formatter import OpenAIProviderFormatter
    from nooa.context_blocks.models import RenderedMessage, Role, ToolCallInfo
    from nooa.llm_types import ToolCall

    arguments = ' { "x": 1 } '
    turn = LLMResponse(parts=(ToolCall(id="c", name="run", arguments=arguments),))
    message = RenderedMessage(
        role=Role.ASSISTANT,
        tool_calls=(ToolCallInfo(id="c", name="run", arguments=arguments),),
        replay_message=turn,
    )
    message = message.model_copy(update={"content": ""})
    public = OpenAIProviderFormatter().format([message])
    assert public[0] is turn
    assert public[0]["tool_calls"][0]["function"]["arguments"] is arguments


@pytest.mark.parametrize(
    "shape",
    [
        "text",
        "calls",
        "calls_null",
        "text_calls",
        "reasoning_calls",
        "truncated",
    ],
)
@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("responses", [False, True])
def test_real_renderer_preserves_every_unedited_assistant_shape(shape, cached, responses):
    from nooa.context_blocks.events import ToolCallEvent, ToolResult
    from nooa.context_blocks.formatter import (
        OpenAIProviderFormatter,
        ResponsesProviderFormatter,
        XMLBlockFormatter,
    )
    from nooa.context_blocks.models import ResolvedBlock, Role
    from nooa.context_blocks.renderer import render_context
    from nooa.context_blocks.renderers.cached import CachedBlockFormatter

    content = (
        None if shape == "calls_null" else "" if shape in {"calls", "reasoning_calls"} else "answer"
    )
    message = {"role": "assistant", "content": content}
    if "calls" in shape:
        message["tool_calls"] = [
            {"id": "c", "type": "function", "function": {"name": "run", "arguments": "{}"}}
        ]
    if shape == "reasoning_calls":
        message["thinking_blocks"] = [
            {"type": "thinking", "thinking": "why", "signature": "native-secret"}
        ]
    scope = replay_scope("anthropic/claude-sonnet-4", "chat", {})
    turn = LLMResponse(
        parts=capture_chat_parts(message, scope),
        replay_scope=scope,
        finish_reason="tool_calls" if "calls" in shape else "stop",
    )
    blocks = [ResolvedBlock(key="turn", content="", role=Role.ASSISTANT, event=turn)]
    for call in turn.tool_calls:
        execution = ToolCallEvent(
            tool_call_id=call.id,
            name=call.name,
            arguments={},
            llm_response_id=turn.id,
            result=ToolResult(tool_call_id=call.id, content="done"),
        )
        blocks.append(
            ResolvedBlock(key="execution", content="", role=Role.ASSISTANT, event=execution)
        )
    base = CachedBlockFormatter if cached else XMLBlockFormatter

    class Formatter(base):
        def format_event(self, event, event_format=None):
            text = super().format_event(event, event_format)
            return text[:3] if shape == "truncated" else text

    result = render_context(
        blocks,
        block_formatter=Formatter(),
        provider_formatter=ResponsesProviderFormatter() if responses else OpenAIProviderFormatter(),
    )
    resolved = next(message for message in result.output if message.get("role") == "assistant")
    if shape == "truncated":
        assert type(resolved) is dict
        assert resolved["content"] == "ans"
    else:
        assert resolved is turn


@pytest.mark.parametrize("middleware", [False, True])
@pytest.mark.parametrize("recover", [False, True])
@pytest.mark.asyncio
async def test_runtime_preserves_response_objects_and_rebuilds_on_recovery(
    monkeypatch, middleware, recover
):
    from nooa import Agent
    from nooa.runtime.actor import _current_llm_var, _current_method_var
    from nooa.unifiedllm import FakeLLMClient

    client = FakeLLMClient()

    class TestAgent(Agent, llm=client):
        async def respond(self) -> str:
            """Continue the conversation."""
            ...

    agent = TestAgent()
    _, previous = source_turn("responses")
    agent.event_manager.add(previous)
    seen = []
    intercepted = []

    class ContextWindowExceededError(Exception):
        pass

    async def dispatch(messages, **kwargs):
        seen.append(messages)
        assert "native-secret" not in json.dumps([dict(m) for m in messages])
        assert "turns" not in kwargs
        if recover and len(seen) == 1:
            raise ContextWindowExceededError("context window exceeded")
        return LLMResponse(content="done")

    async def intercept(ctx, nxt):
        assert "turns" not in ctx.params
        assert "native-secret" not in json.dumps([dict(m) for m in ctx.messages])
        intercepted.append(ctx)
        return await nxt(ctx)

    if middleware:
        agent.event_manager.intercept("llm_call", intercept)
    monkeypatch.setattr(client, "acall", dispatch)
    monkeypatch.setattr("nooa.runtime.actor._compute_reduced_max_tokens", lambda *args: 1024)
    monkeypatch.setattr(
        agent.runtime,
        "_archive_on_context_error",
        lambda *args, **kwargs: agent.event_manager.remove(previous.id),
    )
    token = _current_llm_var.set(client)
    method_token = _current_method_var.set(type(agent).respond)
    try:
        await agent.runtime.generate(max_tokens=2048)
    finally:
        _current_llm_var.reset(token)
        _current_method_var.reset(method_token)

    assert any(message is previous for message in seen[0])
    assert len(seen) == (2 if recover else 1)
    if recover:
        assert not any(message is previous for message in seen[1])
    assert len(intercepted) == (len(seen) if middleware else 0)


def source_turn(api):
    model = "openai/gpt-5.6" if api == "responses" else "anthropic/claude-sonnet-4"
    scope = replay_scope(model, api, {})
    if api == "responses":
        parts = capture_parts(
            [
                {
                    "type": "reasoning",
                    "encrypted_content": "native-secret",
                    "summary": [{"type": "summary_text", "text": "why"}],
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "answer"}],
                },
            ],
            scope,
        )
    else:
        parts = capture_chat_parts(
            {
                "role": "assistant",
                "content": "answer",
                "thinking_blocks": [
                    {"type": "thinking", "thinking": "why", "signature": "native-secret"}
                ],
            },
            scope,
        )
    return model, LLMResponse(parts=parts, replay_scope=scope)


@pytest.mark.asyncio
async def test_render_borrows_existing_response_without_reading_storage_again(monkeypatch):
    from unittest.mock import AsyncMock, Mock

    from nooa import Agent
    from nooa.context_blocks.models import ResolvedBlock, Role
    from nooa.unifiedllm import FakeLLMClient

    class TestAgent(Agent, llm=FakeLLMClient()):
        async def respond(self) -> str: ...

    agent = TestAgent()
    _, original = source_turn("responses")
    blocks = [ResolvedBlock(key="turn", content="", role=Role.ASSISTANT, event=original)]
    monkeypatch.setattr(agent.runtime, "_prepare_context", AsyncMock(return_value=blocks))
    read_again = Mock(side_effect=AssertionError("History must not be loaded a second time"))
    monkeypatch.setattr(agent.event_manager, "values", read_again)
    messages = await agent.runtime._build_messages(type(agent).respond)
    assert any(message is original for message in messages)
    read_again.assert_not_called()


@pytest.mark.parametrize("api", ["responses", "chat"])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("edit", [False, True])
@pytest.mark.asyncio
async def test_response_replay_and_dict_replacement_edit_contract(api, is_async, edit, monkeypatch):
    model, turn = source_turn(api)
    messages = [turn, {"role": "user", "content": "live state 1"}]
    if edit:
        messages[0] = {**dict(turn), "content": "edited answer"}
    before = copy.deepcopy(messages)
    captured = []

    def respond(**kwargs):
        captured.append(kwargs)
        if api == "responses":
            return SimpleNamespace(output=[], usage=None, status="completed")
        return ModelResponse(choices=[Choices(message=Message(role="assistant", content="ok"))])

    async def arespond(**kwargs):
        return respond(**kwargs)

    method = "responses" if api == "responses" else "completion"
    monkeypatch.setattr("litellm." + method, respond)
    monkeypatch.setattr("litellm.a" + method, arespond)
    client_type = ResponsesClient if api == "responses" else CompletionClient
    async with client_type(model=model, api_key="test") as client:
        if is_async:
            await client.acall(messages)
        else:
            client.call(messages)
    assert messages == before
    request = captured[0]
    assert "turns" not in request
    wire = request["input" if api == "responses" else "messages"]
    encoded = json.dumps(wire)
    assert "nooa_turn" not in encoded
    assert ("native-secret" in encoded) is not edit
    assert "why" in encoded
    assert ("edited answer" in encoded) is edit


@pytest.mark.parametrize("api", ["responses", "chat"])
def test_scope_uses_the_per_call_model_override(api, monkeypatch):
    model, turn = source_turn(api)
    captured = []

    def respond(**kwargs):
        captured.append(kwargs)
        return (
            SimpleNamespace(output=[], usage=None, status="completed")
            if api == "responses"
            else ModelResponse(choices=[Choices(message=Message(role="assistant", content="ok"))])
        )

    monkeypatch.setattr("litellm." + ("responses" if api == "responses" else "completion"), respond)
    client_type = ResponsesClient if api == "responses" else CompletionClient
    with client_type(model=model, api_key="test") as client:
        client.call(
            [turn],
            model="openai/gpt-4o",
        )
    assert captured[0]["model"] == "openai/gpt-4o"
    assert "native-secret" not in json.dumps(captured[0].get("input", captured[0].get("messages")))
    assert "turns" not in captured[0]


def test_response_is_a_read_only_public_mapping():
    from collections.abc import Mapping

    _, response = source_turn("responses")
    assert isinstance(response, Mapping)
    public = dict(response)
    assert public == response.public_message()
    assert "native-secret" not in json.dumps(public)
    assert "native-secret" in response.model_dump_json()
    assert response.get("role") == "assistant"
    assert response["content"] == "answer"
    assert set(response) == set(public)
    assert len(response) == len(public)
    assert "parts" not in response
    with pytest.raises(TypeError, match="Replace the history element"):
        response["content"] = "edited"
    with pytest.raises(TypeError, match="Replace the history element"):
        del response["content"]
    # Nested public containers are projections, never aliases into the turn.
    public["content"] = "edited"
    assert response.content == "answer"
