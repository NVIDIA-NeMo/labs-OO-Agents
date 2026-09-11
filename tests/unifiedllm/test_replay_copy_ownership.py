# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay detaches public input once and never copies borrowed/rejected state."""

import copy
from typing import Any

import pytest

from nooa._llm_state import LLM_STATE_KEY, ReplayCarryingMessage, carry_replay_batch
from nooa.unifiedllm import ResponsesClient
from nooa.unifiedllm.replay_state import prepare_chat_messages, prepare_responses_batch

SCOPE = "responses:openai:sha256:test"


class _NoDeepCopy(dict[str, Any]):
    def __deepcopy__(self, memo: dict[int, Any]) -> Any:
        raise AssertionError("Opaque state must not be copied during request preparation")


class _CountCopies(dict[str, Any]):
    def __init__(self, value: dict[str, Any], copies: list[None]):
        super().__init__(value)
        self.copies = copies

    def __deepcopy__(self, memo: dict[int, Any]) -> "_CountCopies":
        self.copies.append(None)
        return _CountCopies(copy.deepcopy(dict(self), memo), self.copies)


@pytest.mark.parametrize("shape", ["batch", "chat", "native"])
@pytest.mark.parametrize("compatible", [True, False])
def test_responses_detaches_public_content_once_without_copying_state(
    shape: str, compatible: bool
) -> None:
    opaque = _NoDeepCopy(
        type="reasoning", id="rs_test", encrypted_content="test-ciphertext", summary=[]
    )
    state = {
        "version": 2,
        "scope": SCOPE,
        "format": "openai-responses",
        "payload": {
            "items": [opaque],
            "order": [
                {"type": "reasoning", "index": 0},
                {"type": "message", "content": "hello"},
            ],
        },
    }
    copies: list[None] = []
    marker = _CountCopies({"type": "ephemeral"}, copies)
    message: dict[str, Any] = {"role": "assistant", "content": "hello", "cache_control": marker}
    if shape == "native":
        message["type"] = "message"
    messages: list[dict[str, Any]] = (
        carry_replay_batch([message], state, "why")
        if shape == "batch"
        else [ReplayCarryingMessage(message, state, "why")]
    )
    target = SCOPE if compatible else "responses:openai:sha256:other"
    with ResponsesClient(model="openai/gpt-5.6", api_key="test") as client:
        prepared, _ = client._transform_messages(messages, target)

    assert len(copies) == 1
    if compatible:
        assert prepared[0] is opaque
    else:
        assert all(item.get("type") != "reasoning" for item in prepared)
    assistant = next(item for item in prepared if item.get("role") == "assistant")
    assert assistant["content"] == ("hello" if compatible else "why\n\nhello")
    assistant["cache_control"]["type"] = "changed"
    assert marker == {"type": "ephemeral"}
    assert message["content"] == "hello"


@pytest.mark.parametrize("path", ["chat", "responses", "batch"])
@pytest.mark.parametrize("key", [LLM_STATE_KEY, "reasoning_items"])
def test_rejected_raw_state_is_removed_before_copying(path: str, key: str) -> None:
    message = {"role": "assistant", "content": "hello", key: _NoDeepCopy(secret="test")}
    if path == "chat":
        prepared = prepare_chat_messages([message], None)
    elif path == "batch":
        prepared = prepare_responses_batch([message], None, None)
    else:
        with ResponsesClient(model="openai/gpt-5.6", api_key="test") as client:
            prepared, _ = client._transform_messages([message])

    assert prepared == [{"role": "assistant", "content": "hello"}]
    assert key in message


@pytest.mark.parametrize("role", ["user", "assistant", "tool"])
def test_responses_passthrough_still_detaches_nested_public_content(role: str) -> None:
    marker = {"type": "ephemeral"}
    message = {"role": role, "content": "hello", "cache_control": marker}
    if role == "tool":
        message["tool_call_id"] = "call_test"
    with ResponsesClient(model="openai/gpt-5.6", api_key="test") as client:
        prepared, _ = client._transform_messages([message])

    prepared[0]["cache_control"]["type"] = "changed"
    assert marker == {"type": "ephemeral"}
