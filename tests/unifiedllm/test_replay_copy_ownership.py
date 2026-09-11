# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request projection allocates containers, not copies of immutable turn data."""

import copy
from typing import Any

import pytest

from nooa._llm_state import LLM_STATE_KEY
from nooa.llm_types import AssistantReasoning, AssistantText, LLMResponse, ToolCall
from nooa.unifiedllm import ResponsesClient
from nooa.unifiedllm.chat_parts import project_chat_turn
from nooa.unifiedllm.replay_state import ReasoningReplayError, prepare_chat_messages
from nooa.unifiedllm.response_parts import project_turn

SCOPE = "responses:openai:sha256:test"


class _NoDeepCopy(dict[str, Any]):
    def __deepcopy__(self, memo):
        raise AssertionError("Rejected opaque state must not be copied")


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("compatible", [True, False])
def test_projection_borrows_immutable_leaves_without_copying_history(api, compatible):
    scope = f"{api}:openai:sha256:test"
    encrypted = "private" * 100_000
    arguments = '{"code":"' + "x" * 100_000 + '"}'
    native = {"type": "reasoning", "encrypted_content": encrypted, "summary": []}
    turn = LLMResponse(
        parts=(
            AssistantReasoning(native={"reasoning_items": native} if api == "chat" else native),
            AssistantText(text="hello"),
            ToolCall(id="call_test", name="run", arguments=arguments),
        ),
        replay_scope=scope,
    )
    target = scope if compatible else None
    assert copy.deepcopy(turn).parts[0].native is turn.parts[0].native
    if api == "chat":
        wire, _ = project_chat_turn(turn, target)
        assert wire["tool_calls"][0]["function"]["arguments"] is arguments
        if compatible:
            assert wire["reasoning_items"][0]["encrypted_content"] is encrypted
            wire["reasoning_items"][0]["summary"].append("request-owned")
        else:
            assert "reasoning_items" not in wire
    else:
        wire = project_turn(turn, target)
        assert wire[-1]["arguments"] is arguments
        if compatible:
            assert wire[0]["encrypted_content"] is encrypted
            wire[0]["summary"].append("request-owned")
        else:
            assert all(item.get("type") != "reasoning" for item in wire)
    stored = turn.parts[0].native
    assert (stored["reasoning_items"]["summary"] if api == "chat" else stored["summary"]) == ()


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("key", [LLM_STATE_KEY, "reasoning_items"])
def test_rejected_raw_state_is_not_copied(api, key):
    message = {"role": "assistant", "content": "hello", key: _NoDeepCopy(secret="test")}
    with ResponsesClient(model="openai/gpt-5.6", api_key="test") as client:

        def prepare():
            return (
                prepare_chat_messages([message], None)
                if api == "chat"
                else client._transform_messages([message])[0]
            )

        if (api == "chat" and key == LLM_STATE_KEY) or (
            api == "responses" and key == "reasoning_items"
        ):
            with pytest.raises(ReasoningReplayError):
                prepare()
        else:
            assert prepare() == [{"role": "assistant", "content": "hello"}]
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
