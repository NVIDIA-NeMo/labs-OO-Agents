# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compact replay bindings survive storage, reject edits, and validate wire items."""

import copy
import json
import sqlite3

import pytest

from nooa._llm_state import ReplayCarryingMessage
from nooa.storage.sqlite import SQLiteEventBackend, _ensure_schema
from nooa.unifiedllm import LLMResponse, ToolCall
from nooa.unifiedllm.replay_state import (
    ReasoningReplayError,
    capture_chat_state,
    capture_responses_state,
    prepare_chat_messages,
    prepare_responses_batch,
)

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
def test_chat_capture_rejects_malformed_reasoning_items(invalid, caplog) -> None:
    with pytest.raises(ReasoningReplayError, match="malformed") as error:
        capture_chat_state(
            {"role": "assistant", "content": "answer", "reasoning_items": [REASONING, invalid]},
            "chat:openai:sha256:test",
        )
    assert "ciphertext" not in str(error.value) + caplog.text


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("invalid", INVALID_ITEMS)
def test_corrupt_stored_items_never_reach_replay(api, invalid, caplog) -> None:
    scope = f"{api}:openai:sha256:test"
    public = {"role": "assistant", "content": "answer"}
    if api == "chat":
        state = capture_chat_state({**public, "reasoning_items": [REASONING]}, scope)
        assert state is not None
        state["payload"]["reasoning_items"] = [REASONING, invalid]
        with pytest.raises(ReasoningReplayError, match="malformed") as error:
            prepare_chat_messages([ReplayCarryingMessage(public, state, "why")], scope)
    else:
        state = capture_responses_state(
            [
                REASONING,
                {"type": "message", "content": [{"type": "output_text", "text": "answer"}]},
            ],
            scope,
        )
        assert state is not None
        state["payload"]["items"][0] = invalid
        with pytest.raises(ReasoningReplayError, match="Malformed") as error:
            prepare_responses_batch([public], state, scope, "why")
    assert "ciphertext" not in str(error.value) + caplog.text


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("persistence", ["json", "sqlite"])
def test_large_tool_arguments_are_stored_once_and_replay_exactly(api, persistence) -> None:
    arguments = json.dumps({"code": "x" * 100_000 + "é"})
    scope = f"{api}:openai:sha256:test"
    call = {
        "type": "function_call",
        "call_id": "call_test",
        "name": "execute_python",
        "arguments": arguments,
    }
    chat = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_test",
                "type": "function",
                "function": {"name": "execute_python", "arguments": arguments},
            }
        ],
    }
    state = (
        capture_chat_state({**chat, "reasoning_items": [REASONING]}, scope)
        if api == "chat"
        else capture_responses_state([REASONING, call], scope)
    )
    assert state is not None
    response = LLMResponse(
        tool_calls=[ToolCall(id="call_test", name="execute_python", arguments=arguments)],
        llm_state=state,
    )
    encoded = response.model_dump_json()
    # The envelope adds constant-size bindings, not another 100 KB argument.
    assert len(encoded) - len(response.model_dump_json(exclude={"llm_state"})) < 1000
    assert "x" * 100 not in json.dumps(state)
    if persistence == "json":
        resumed = LLMResponse.model_validate_json(encoded)
    else:
        connection = sqlite3.connect(":memory:")
        try:
            _ensure_schema(connection)
            backend = SQLiteEventBackend(connection)
            backend.store("response", response)
            resumed = backend.get("response")
            assert isinstance(resumed, LLMResponse)
        finally:
            connection.close()
    assert resumed.llm_state == state
    if api == "chat":
        result = prepare_chat_messages([ReplayCarryingMessage(chat, resumed.llm_state)], scope)
        assert result == [{**chat, "reasoning_items": [REASONING]}]
    else:
        assert prepare_responses_batch([call], resumed.llm_state, scope) == [REASONING, call]


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_prior_draft_envelope_warns_and_demotes_text(api, caplog) -> None:
    scope = f"{api}:openai:sha256:test"
    public = {"role": "assistant", "content": "answer"}
    state = {
        "version": 1,
        "scope": scope,
        "format": "litellm-chat" if api == "chat" else "openai-responses",
        "payload": {},
    }
    result = (
        prepare_chat_messages([ReplayCarryingMessage(public, state, "why")], scope)
        if api == "chat"
        else prepare_responses_batch([public], state, scope, "why")
    )
    assert result == [{"role": "assistant", "content": "why\n\nanswer"}]
    assert "unsupported or legacy version" in caplog.text


@pytest.mark.parametrize("mutation", ["id", "name", "arguments", "drop", "reorder"])
def test_responses_fingerprints_reject_edited_calls(mutation, caplog) -> None:
    scope = "responses:openai:sha256:test"
    calls = [
        {
            "type": "function_call",
            "call_id": f"call_{i}",
            "name": "execute_python",
            "arguments": '{"code":"print(1)"}',
        }
        for i in range(2)
    ]
    state = capture_responses_state([REASONING, *calls], scope)
    assert state is not None
    edited = copy.deepcopy(calls)
    if mutation == "drop":
        edited.pop()
    elif mutation == "reorder":
        edited.reverse()
    else:
        edited[0]["call_id" if mutation == "id" else mutation] = "changed"
    result = prepare_responses_batch(edited, state, scope)
    assert result == edited
    assert "carriers changed" in caplog.text


def test_chat_fingerprint_is_key_order_independent_and_does_not_store_text() -> None:
    scope = "chat:openai:sha256:test"
    message = {
        "role": "assistant",
        "content": "large public text" * 1000,
        "reasoning_items": [REASONING],
    }
    state = capture_chat_state(message, scope)
    assert state is not None
    reordered = dict(reversed(list(message.items())))
    assert capture_chat_state(reordered, scope) == state
    assert "large public text" not in json.dumps(state)
    public = {"content": message["content"], "role": "assistant"}
    assert prepare_chat_messages([ReplayCarryingMessage(public, state)], scope) == [
        {**public, "reasoning_items": [REASONING]}
    ]
