# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Part invariants replace fingerprint/ledger checks; guarantees still get tested."""

import json

import pytest

from nooa.llm_types import AssistantReasoning, LLMResponse, ToolCall
from nooa.storage.sqlite import SQLiteStorageManager
from nooa.unifiedllm.chat_parts import capture_chat_parts, project_chat_turn
from nooa.unifiedllm.replay_state import ReasoningReplayError
from nooa.unifiedllm.response_parts import capture_parts, project_turn

REASONING = {"type": "reasoning", "id": "rs_test", "encrypted_content": "ciphertext", "summary": []}
INVALID_ITEMS = [
    123,
    {},
    {**REASONING, "type": "message"},
    {**REASONING, "encrypted_content": 123},
    {**REASONING, "encrypted_content": ""},
    {"type": "reasoning", "summary": []},
]


@pytest.mark.parametrize("invalid", INVALID_ITEMS)
def test_chat_capture_rejects_malformed_reasoning_items(invalid, caplog):
    with pytest.raises(ReasoningReplayError) as error:
        capture_chat_parts(
            {"role": "assistant", "content": "answer", "reasoning_items": [REASONING, invalid]},
            "chat:openai:test",
        )
    assert "ciphertext" not in str(error.value) + caplog.text


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("invalid", INVALID_ITEMS)
def test_corrupt_archived_reasoning_never_reaches_wire(api, invalid, caplog):
    scope = f"{api}:openai:test"
    native = {"reasoning_items": REASONING} if api == "chat" else REASONING
    turn = LLMResponse(parts=(AssistantReasoning(native=native),), replay_scope=scope)
    archive = turn.model_dump(mode="json")
    # Simulate corruption in the archive, not a sanctioned public edit.
    archive["parts"][0]["native"] = {"reasoning_items": invalid} if api == "chat" else invalid
    with pytest.raises((ReasoningReplayError, ValueError, TypeError)):
        loaded = LLMResponse.model_validate(archive)
        if api == "chat":
            project_chat_turn(loaded, scope)
        else:
            project_turn(loaded, scope)
    assert "ciphertext" not in caplog.text


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("persistence", ["json", "sqlite"])
def test_large_tool_arguments_are_stored_once_and_replay_exactly(api, persistence, tmp_path):
    arguments = json.dumps({"code": "x" * 100_000 + "é"})
    scope = f"{api}:openai:test"
    wire_call = {
        "type": "function_call",
        "call_id": "call_test",
        "name": "execute_python",
        "arguments": arguments,
    }
    chat_call = {
        "id": "call_test",
        "type": "function",
        "function": {"name": "execute_python", "arguments": arguments},
    }
    chat = {
        "role": "assistant",
        "content": None,
        "tool_calls": [chat_call],
        "reasoning_items": [REASONING],
    }
    parts = (
        capture_chat_parts(chat, scope)
        if api == "chat"
        else capture_parts([REASONING, wire_call], scope)
    )
    response = LLMResponse(parts=parts, replay_scope=scope)
    encoded = response.model_dump_json()
    assert encoded.count("x" * 100_000) == 1
    assert "x" * 100 not in json.dumps([p.model_dump(mode="json")["native"] for p in parts])
    if persistence == "json":
        resumed = LLMResponse.model_validate_json(encoded)
    else:
        storage = SQLiteStorageManager(tmp_path / "parts.db")
        storage.event_backend.store("turn", response)
        storage.close()
        storage = SQLiteStorageManager(tmp_path / "parts.db")
        resumed = storage.event_backend.get("turn")
        storage.close()
    if api == "chat":
        assert project_chat_turn(resumed, scope)[0] == chat
    else:
        assert project_turn(resumed, scope) == [REASONING, wire_call]


@pytest.mark.parametrize("mutation", ["arguments", "name", "id", "reorder", "drop", "append"])
def test_public_call_edit_removes_all_native_authority(mutation):
    first = ToolCall(
        id="a",
        name="first",
        arguments="{}",
        native={"provider_specific_fields": {"thought_signature": "secret"}},
    )
    second = ToolCall(id="b", name="second", arguments="{}")
    original = LLMResponse(
        parts=(AssistantReasoning(text="think"), first, second), replay_scope="chat:gemini:test"
    )
    calls = original.tool_calls
    if mutation == "reorder":
        calls.reverse()
    elif mutation == "drop":
        calls.pop()
    elif mutation == "append":
        calls.append(ToolCall(id="c", name="third", arguments="{}"))
    else:
        calls[0] = calls[0].model_copy(
            update={mutation: '{"changed":true}' if mutation == "arguments" else "changed"}
        )
    replacement = original.replace_parts((original.parts[0], *calls))
    assert all(part.native is None for part in replacement.parts)
    assert replacement.replay_scope is None
    assert replacement.reasoning == "think"
    assert original.tool_calls[0] is first
    assert first.native is not None
    assert "secret" not in json.dumps(project_chat_turn(replacement, original.replay_scope)[0])


def test_legacy_flat_archive_migrates_portable_only(tmp_path):
    archive = {
        "event_type": "LLMResponse",
        "content": "answer",
        "reasoning": "why",
        "llm_state": {"version": 2, "scope": "chat:openai:test", "payload": {"secret": "opaque"}},
    }
    storage = SQLiteStorageManager(tmp_path / "old.db")
    try:
        loaded = storage.event_backend._deserialize(json.dumps(archive))
    finally:
        storage.close()
    assert loaded.content == "answer"
    assert loaded.reasoning == "why"
    assert loaded.replay_scope is None
    assert all(part.native is None for part in loaded.parts)
