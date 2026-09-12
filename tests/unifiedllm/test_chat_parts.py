# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chat-native replay through the canonical turn, with no live inference."""

import copy
import json

import pytest
from litellm.types.utils import Choices, Message, ModelResponse

from nooa.llm_types import LLMResponse
from nooa.runtime.middleware import LLMCallContext
from nooa.storage.sqlite import SQLiteStorageManager
from nooa.unifiedllm import CompletionClient
from nooa.unifiedllm.chat_parts import capture_chat_parts, project_chat_turn
from nooa.unifiedllm.replay_state import ReasoningReplayError, prepare_chat_messages, replay_scope
from nooa.unifiedllm.response_parts import project_turn

MODELS = ["anthropic/claude-sonnet-4", "gemini/gemini-2.5-pro", "openai/gateway-gemini"]


@pytest.mark.parametrize("scope", [None, "chat:gemini:test", "chat:openai:test"])
def test_unsigned_thinking_is_portable_without_dropping_neighbor_signatures(scope):
    source = message("gemini/test")
    source["thinking_blocks"] = [{"type": "thinking", "thinking": "Check inputs."}]
    source["reasoning_content"] = "Check inputs."
    turn = LLMResponse(parts=capture_chat_parts(source, scope), replay_scope=scope)
    assert turn.reasoning == "Check inputs."
    assert turn.parts[0].native is None
    restored = LLMResponse.model_validate_json(turn.model_dump_json())
    projected, _ = project_chat_turn(restored, scope)
    assert projected["content"] == "Check inputs."
    assert "thinking_blocks" not in projected
    if scope is not None:
        assert projected["tool_calls"] == source["tool_calls"]
        assert projected["provider_specific_fields"] == source["provider_specific_fields"]
    else:
        assert "opaque-signature" not in json.dumps(projected)


@pytest.mark.parametrize("signature", [None, "", 42])
def test_present_but_malformed_thinking_signature_still_raises(signature):
    source = {"thinking_blocks": [{"type": "thinking", "thinking": "x", "signature": signature}]}
    with pytest.raises(ReasoningReplayError, match="Malformed signed thinking"):
        capture_chat_parts(source, "chat:openai:test")


def message(model):
    calls = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "execute_python", "arguments": '{ "code": "print(1)" }'},
        }
    ]
    result = {"role": "assistant", "content": None, "tool_calls": calls}
    if model.startswith("anthropic"):
        result["thinking_blocks"] = [
            {"type": "thinking", "thinking": "Check inputs.", "signature": "signed-thinking"},
            {"type": "redacted_thinking", "data": "encrypted-redaction"},
        ]
        result["reasoning_content"] = "Check inputs."
    else:
        calls[0]["id"] = "call_1__thought__opaque-signature"
        calls[0]["provider_specific_fields"] = {"thought_signature": "opaque-signature"}
        result["provider_specific_fields"] = {"thought_signatures": ["opaque-signature"]}
    return result


def capture(model):
    scope = replay_scope(model, "chat", {})
    return LLMResponse(
        parts=capture_chat_parts(Message(**message(model)), scope), replay_scope=scope
    )


@pytest.mark.parametrize("model", MODELS)
def test_capture_project_exact_native_fields_without_public_duplicates(model):
    turn = capture(model)
    projected, ids = project_chat_turn(turn, turn.replay_scope)
    expected = message(model)
    expected.pop("reasoning_content", None)  # LiteLLM's duplicate readable view.
    assert projected == expected
    assert turn.tool_calls[0].id == "call_1"
    assert turn.tool_calls[0].arguments == '{ "code": "print(1)" }'
    native_json = json.dumps([p.model_dump(mode="json")["native"] for p in turn.parts])
    assert "Check inputs." not in native_json
    assert "print(1)" not in native_json
    assert ids == (
        {} if model.startswith("anthropic") else {"call_1": "call_1__thought__opaque-signature"}
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("target", [None, "chat:anthropic:other", "responses:openai:other"])
def test_incompatible_scope_removes_every_native_slot(model, target):
    turn = capture(model)
    projected, ids = project_chat_turn(turn, target)
    wire = json.dumps(projected)
    for secret in ("signed-thinking", "encrypted-redaction", "opaque-signature"):
        assert secret not in wire
    if model.startswith("anthropic"):
        assert "Check inputs." in wire
    assert ids == {}
    assert projected["tool_calls"][0]["id"] == "call_1"
    assert all(
        secret not in json.dumps(project_turn(turn, target))
        for secret in ("signed-thinking", "encrypted-redaction", "opaque-signature")
    )


@pytest.mark.parametrize("model", MODELS)
def test_edited_turn_drops_native_and_keeps_portable_reasoning(model):
    turn = capture(model)
    replacement = turn.replace_text("Edited response")
    assert replacement.replay_scope is None
    assert all(part.native is None for part in replacement.parts)
    assert replacement.tool_calls[0].id == "call_1"
    assert replacement.reasoning == turn.reasoning
    assert any(part.native for part in turn.parts)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_mocked_dispatch_after_sqlite_resume_with_dynamic_suffix(
    model, is_async, tmp_path, monkeypatch
):
    captured = []

    def respond(**kwargs):
        captured.append(copy.deepcopy(kwargs["messages"]))
        return ModelResponse(
            model=model,
            choices=[Choices(message=Message(**message(model)), finish_reason="tool_calls")],
        )

    async def arespond(**kwargs):
        return respond(**kwargs)

    monkeypatch.setattr("litellm.completion", respond)
    monkeypatch.setattr("litellm.acompletion", arespond)
    client = CompletionClient(
        model=model,
        api_key="test",
    )

    async def invoke(messages):
        return await client.acall(messages) if is_async else client.call(messages)

    try:
        turn = await invoke([{"role": "user", "content": "Start"}])
        storage = SQLiteStorageManager(tmp_path / "chat.db")
        storage.event_backend.store("turn", turn)
        storage.close()
        storage = SQLiteStorageManager(tmp_path / "chat.db")
        restored = storage.event_backend.get("turn")
        storage.close()
        for suffix in ("live state one", "live state two"):
            context = LLMCallContext(
                messages=[
                    {"role": "system", "content": "Stable instructions"},
                    restored,
                    {"role": "tool", "tool_call_id": "call_1", "content": "Completed"},
                    {"role": "user", "content": suffix},
                ]
            )
            await invoke(context.messages)
        assert captured[-1][:-1] == captured[-2][:-1]
        assert captured[-1][-1] != captured[-2][-1]
        expected = message(model)
        expected.pop("reasoning_content", None)
        assert captured[-1][1] == expected
        assert captured[-1][2]["tool_call_id"] == expected["tool_calls"][0]["id"]
        assert "opaque-signature" not in json.dumps(restored.public_message())
    finally:
        await client.aclose()


@pytest.mark.parametrize(
    "field,value",
    [
        ("thinking_blocks", {}),
        ("tool_calls", {}),
        ("provider_specific_fields", []),
        ("reasoning_content", 4),
    ],
)
def test_malformed_provider_containers_fail_loudly(field, value):
    with pytest.raises(ReasoningReplayError):
        capture_chat_parts({"role": "assistant", "content": "", field: value}, "chat:gemini:model")


def test_unknown_route_keeps_readable_thinking_without_signed_state(caplog):
    parts = capture_chat_parts(message(MODELS[0]), None)
    assert all(part.native is None for part in parts)
    assert LLMResponse(parts=parts).reasoning
    assert "Unknown provider route" in caplog.text


def test_reasoning_only_turn_replays_and_plain_reasoning_crosses_models():
    turn = LLMResponse(
        parts=capture_chat_parts(
            {"role": "assistant", "content": None, "reasoning_content": "Portable thinking"}, None
        )
    )
    for model in MODELS:
        assert prepare_chat_messages([turn], replay_scope(model, "chat", {})) == [
            {"role": "assistant", "content": "Portable thinking"}
        ]


def test_signed_reasoning_only_turn_survives():
    source = message(MODELS[0])
    source.pop("tool_calls")
    scope = replay_scope(MODELS[0], "chat", {})
    turn = LLMResponse(parts=capture_chat_parts(source, scope), replay_scope=scope)
    projected = prepare_chat_messages([turn], scope)
    assert projected[0]["thinking_blocks"] == source["thinking_blocks"]
    assert projected[0]["content"] == ""
