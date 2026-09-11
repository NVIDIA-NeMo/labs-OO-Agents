# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collapsing history must not discard active native state or replay archived state."""

import json
from types import SimpleNamespace

import pytest
from litellm.types.utils import Choices, Message, ModelResponse

from nooa import Agent, Context
from nooa.context_blocks.events import ToolCallEvent, ToolResult
from nooa.llm_types import LLMResponse
from nooa.runtime.actor import _current_llm_var, _current_method_var
from nooa.storage.sqlite import SQLiteStorageManager
from nooa.unifiedllm import CompletionClient, ResponsesClient
from nooa.unifiedllm.chat_parts import capture_chat_parts
from nooa.unifiedllm.replay_state import replay_scope
from nooa.unifiedllm.response_parts import capture_parts


def _native_turn(model, number):
    responses = model.startswith("openai/")
    scope = replay_scope(model, "responses" if responses else "chat", {})
    secret = f"native-secret-{number}"
    call_id = f"call_{number}"
    if responses:
        parts = capture_parts(
            [
                {"type": "reasoning", "encrypted_content": secret, "summary": []},
                {"type": "function_call", "call_id": call_id, "name": "run", "arguments": "{}"},
            ],
            scope,
        )
    else:
        call = {"id": call_id, "type": "function", "function": {"name": "run", "arguments": "{}"}}
        message = {"role": "assistant", "content": None, "tool_calls": [call]}
        if model.startswith("anthropic/"):
            message["thinking_blocks"] = [
                {"type": "thinking", "thinking": f"thought-{number}", "signature": secret}
            ]
        else:
            call["id"] = f"{call_id}__thought__{secret}"
            call["provider_specific_fields"] = {"thought_signature": secret}
        parts = capture_chat_parts(message, scope)
    return LLMResponse(parts=parts, replay_scope=scope, finish_reason="tool_calls")


def _result(turn):
    call = turn.tool_calls[0]
    return ToolCallEvent(
        tool_call_id=call.id,
        name=call.name,
        arguments={},
        llm_response_id=turn.id,
        result=ToolResult(tool_call_id=call.id, content=f"result-{call.id}"),
    )


@pytest.mark.parametrize(
    "model", ["openai/gpt-5.6", "anthropic/claude-sonnet-4", "gemini/gemini-2.5-pro"]
)
@pytest.mark.parametrize("backend", ["memory", "sqlite_resume"])
@pytest.mark.parametrize("collapse", ["whole_turn", "source_only", "result_only"])
@pytest.mark.asyncio
async def test_collapse_keeps_only_complete_active_native_turns(
    model, backend, collapse, tmp_path, monkeypatch, caplog
):
    responses = model.startswith("openai/")
    client_type = ResponsesClient if responses else CompletionClient
    client = client_type(model=model, api_key="test")

    class TestAgent(Agent, llm=client):
        async def respond(self) -> str:
            """Continue after the earlier work."""
            ...

    agent = TestAgent()
    agent.context["live"] = Context(expr="'trailing live state'")
    storage = (
        SQLiteStorageManager(tmp_path / "collapsed.db") if backend == "sqlite_resume" else None
    )
    if storage is not None:
        agent.event_manager.set_backend(storage.event_backend)
    first, second = _native_turn(model, 1), _native_turn(model, 2)
    first_tag = agent.event_manager.add(first)
    result_tag = agent.event_manager.add(_result(first))
    agent.event_manager.add(second)
    agent.event_manager.add(_result(second))
    start = result_tag if collapse == "result_only" else first_tag
    end = first_tag if collapse == "source_only" else result_tag
    summary_tag = agent.events.collapse(start, end, "Earlier work summarized.")
    if storage is not None:
        storage.close()
        storage = SQLiteStorageManager(tmp_path / "collapsed.db")
        agent.event_manager.set_backend(storage.event_backend)

    # Observe the actual objects rendered, including objects loaded from SQLite.
    rendered_events = {}
    prepare = agent.runtime._prepare_context

    async def observe_context(*args, **kwargs):
        blocks = await prepare(*args, **kwargs)
        rendered_events.update((b.event.id, b.event) for b in blocks if b.event is not None)
        return blocks

    monkeypatch.setattr(agent.runtime, "_prepare_context", observe_context)
    dispatch = client.acall
    captured = []

    async def observe_dispatch(messages, *, turns, **kwargs):
        assert set(turns) == {second.id}
        assert turns[second.id] is rendered_events[second.id]
        resolved = client._resolve_turns(messages, turns)
        assert any(item is turns[second.id] for item in resolved)
        summary = next(m for m in messages if "Earlier work summarized." in str(m.get("content")))
        assert summary["role"] == "assistant"
        assert set(summary) == {"role", "content"}
        assert "native-secret" not in json.dumps(messages)
        return await dispatch(messages, turns=turns, **kwargs)

    async def provider(**kwargs):
        captured.append(kwargs)
        if responses:
            return SimpleNamespace(output=[], status="completed", usage=None)
        return ModelResponse(choices=[Choices(message=Message(role="assistant", content="ok"))])

    monkeypatch.setattr(client, "acall", observe_dispatch)
    monkeypatch.setattr("litellm.aresponses" if responses else "litellm.acompletion", provider)
    token = _current_llm_var.set(client)
    method_token = _current_method_var.set(type(agent).respond)
    try:
        await agent.runtime.generate()
        assert agent.events[summary_tag].render_reference() is None
        # Collapse archives, rather than destroys, the original record.
        assert "native-secret-1" in agent.events[first_tag].model_dump_json()
    finally:
        _current_llm_var.reset(token)
        _current_method_var.reset(method_token)
        await client.aclose()
        if storage is not None:
            storage.close()

    assert len(captured) == 1
    api_params = captured[0]
    assert "turns" not in api_params
    # The local HTTP transport object is not part of the provider JSON body.
    encoded = json.dumps({key: value for key, value in api_params.items() if key != "client"})
    assert "native-secret-1" not in encoded
    assert "native-secret-2" in encoded
    assert "result-call_1" not in encoded
    assert "result-call_2" in encoded
    assert "Earlier work summarized." in encoded
    assert "trailing live state" in encoded
    assert "nooa_turn" not in encoded
    assert "nooa_cache_boundary" not in encoded
    assert ("incomplete visible execution batch" in caplog.text) is (collapse == "result_only")
