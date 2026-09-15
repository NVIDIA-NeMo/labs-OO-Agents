# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider variation must preserve portable output without granting replay authority."""

import json
from types import SimpleNamespace

import pytest

from nooa.llm_types import LLMResponse
from nooa.unifiedllm.chat_parts import capture_chat_parts, project_chat_turn
from nooa.unifiedllm.replay_state import ReasoningReplayError
from nooa.unifiedllm.response_parts import capture_parts, project_turn
from nooa.unifiedllm.unifiedllm import _extract_usage


@pytest.mark.parametrize("api,native", [("chat", True), ("responses", False), ("responses", True)])
def test_string_reasoning_summary_roundtrips_without_public_duplication(api, native):
    item = {"type": "reasoning", "summary": "portable summary"}
    if native:
        item["encrypted_content"] = "test-cipher"
    scope = f"{api}:openai:model" if native else None
    if api == "chat":
        parts = capture_chat_parts({"content": "", "reasoning_items": [item]}, scope)
    else:
        parts = capture_parts([item], scope)
    turn = LLMResponse(parts=parts, replay_scope=scope)
    archive = turn.model_dump_json()
    assert archive.count("portable summary") == 1
    restored = LLMResponse.model_validate_json(archive)
    assert restored.reasoning == "portable summary"
    if native:
        projected = (
            project_turn(restored, scope)
            if api == "responses"
            else project_chat_turn(restored, scope)[0]["reasoning_items"]
        )
        assert projected == [item]


@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("call_id", ["", "same"])
def test_nonempty_ids_must_be_unique_but_portable_empty_ids_are_accepted(api, call_id):
    calls = [{"type": "function_call", "call_id": call_id, "name": "run", "arguments": "{}"}] * 2

    def capture():
        if api == "responses":
            return capture_parts(calls, None)
        return capture_chat_parts(
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": c["call_id"],
                        "function": {"name": c["name"], "arguments": c["arguments"]},
                    }
                    for c in calls
                ],
            },
            None,
        )

    if call_id:
        with pytest.raises(ReasoningReplayError, match="unique"):
            capture()
    else:
        turn = LLMResponse(parts=capture())
        assert [call.id for call in turn.tool_calls] == ["", ""]


def test_empty_id_cannot_bind_native_tool_signature():
    message = {
        "content": "",
        "tool_calls": [
            {
                "id": "",
                "function": {"name": "run", "arguments": "{}"},
                "provider_specific_fields": {"thought_signature": "secret"},
            }
        ],
    }
    with pytest.raises(ReasoningReplayError, match="nonempty"):
        capture_chat_parts(message, "chat:gemini:model")


def test_unknown_route_strips_message_and_inline_tool_signatures(caplog):
    message = {
        "content": "ok",
        "reasoning_content": "readable",
        "provider_specific_fields": {"thought_signatures": ["secret"]},
        "tool_calls": [
            {
                "id": "call__thought__secret",
                "function": {"name": "run", "arguments": "{}"},
                "provider_specific_fields": {"thought_signature": "secret"},
            }
        ],
    }
    turn = LLMResponse(parts=capture_chat_parts(message, None))
    assert turn.reasoning == "readable"
    assert turn.tool_calls[0].id == "call"
    assert turn.replay_scope is None
    assert "secret" not in json.dumps(turn.model_dump(mode="json"))
    assert "Unknown provider route" in caplog.text


@pytest.mark.parametrize("bad_cost", [True, "unknown", float("nan"), -1.0])
def test_malformed_cost_warns_without_losing_usage(bad_cost, caplog):
    raw = SimpleNamespace(
        usage={"input_tokens": 10, "output_tokens": 2}, _hidden_params={"response_cost": bad_cost}
    )
    usage = _extract_usage(raw)
    assert usage.input_tokens == 10
    assert usage.cost_usd == 0.0
    assert "Ignoring malformed LiteLLM response_cost" in caplog.text
