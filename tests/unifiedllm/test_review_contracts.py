# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider variation and public projection contracts from the full core review."""

import json
from unittest.mock import patch

import pytest
from litellm import ModelResponse

from nooa.context_blocks.events import EventBase
from nooa.context_blocks.formatter import AnthropicProviderFormatter
from nooa.context_blocks.models import RenderedMessage, Role, ToolCallInfo
from nooa.llm_types import AssistantText, LLMResponse, ToolCall, assistant_message
from nooa.unifiedllm import CompletionClient, ResponsesClient
from nooa.unifiedllm.chat_parts import capture_chat_parts, project_chat_turn


@pytest.mark.parametrize("scope", ["chat:openai:model", None])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("summary", ["readable", [{"type": "summary_text", "text": "readable"}]])
def test_chat_summary_only_reasoning_is_portable(scope, mixed, summary, caplog):
    items = [{"type": "reasoning", "summary": summary}]
    if mixed:
        items.insert(0, {"type": "reasoning", "encrypted_content": "cipher", "summary": "first"})
    turn = LLMResponse(
        parts=capture_chat_parts({"content": "answer", "reasoning_items": items}, scope),
        replay_scope=scope,
    )
    assert all(part.native is None for part in turn.parts)
    assert turn.reasoning == ("first\nreadable" if mixed else "readable")
    wire, _ = project_chat_turn(turn, scope)
    assert wire["content"].endswith("readable\n\nanswer")
    assert "cipher" not in json.dumps(wire)
    if mixed:
        assert "Incomplete native reasoning sequence" in caplog.text


@pytest.mark.parametrize("model", ["openai/gpt-5.6", "openrouter/anthropic/claude-sonnet-4"])
def test_successful_chat_call_can_return_summary_without_encrypted_content(model):
    response = ModelResponse(
        choices=[
            {
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_items": [{"type": "reasoning", "summary": "readable"}],
                }
            }
        ]
    )
    with CompletionClient(model, api_key="test") as client:
        with patch("litellm.completion", return_value=response) as request:
            result = client.call([{"role": "user", "content": "go"}])
    assert result.content == "answer"
    assert result.reasoning == "readable"
    assert all(part.native is None for part in result.parts)
    assert request.call_count == 1


def test_public_projection_is_built_once_and_edits_invalidate_only_the_copy():
    turn = LLMResponse(
        parts=(
            AssistantText(text="answer"),
            ToolCall(id="c", name="run", arguments="{}"),
        )
    )
    with patch("nooa.llm_types.assistant_message", wraps=assistant_message) as build:
        assert turn["role"] == "assistant"
        assert turn.get("content") == "answer"
        assert "tool_calls" in turn
        assert len(turn) == 3
        assert list(turn.keys()) == list(turn)
        public = turn.public_message()
        public["tool_calls"][0]["function"]["name"] = "changed"
        assert dict(turn.items())["tool_calls"][0]["function"]["name"] == "run"
        assert build.call_count == 1
        stamped = turn.model_copy(update={"tag": "2"})
        assert stamped["content"] == "answer"
        assert build.call_count == 1
        edited = turn.replace_text("new")
        assert edited["content"] == "new"
        assert turn["content"] == "answer"
        assert build.call_count == 2
    assert "_public_projection" not in turn.model_dump_json()
    assert LLMResponse.model_validate_json(turn.model_dump_json()) == turn


def test_anthropic_export_is_native_public_shape_without_framework_markers():
    turn = LLMResponse(
        parts=(
            AssistantText(text="answer"),
            ToolCall(id="c", name="run", arguments='{"x":1}'),
        )
    )
    result = AnthropicProviderFormatter().format(
        [
            RenderedMessage(
                role=Role.ASSISTANT,
                content="answer",
                reasoning="why",
                tool_calls=(ToolCallInfo(id="c", name="run", arguments='{"x":1}'),),
                replay_message=turn,
                cache_boundary_before=True,
            )
        ]
    )
    assert result == {
        "system": "",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "why\n\nanswer"},
                    {"type": "tool_use", "id": "c", "name": "run", "input": {"x": 1}},
                ],
            }
        ],
    }


def test_extra_function_metadata_does_not_break_public_responses_input():
    message = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "c",
                "function": {
                    "name": "run",
                    "arguments": "{}",
                    "parsed_arguments": {"ignored": True},
                },
            }
        ],
    }
    with ResponsesClient("openai/gpt-5.6") as client:
        wire, _ = client._transform_messages([message])
    assert wire == [{"type": "function_call", "call_id": "c", "name": "run", "arguments": "{}"}]


def test_replay_turn_probe_is_explicit():
    assert not EventBase().is_replay_turn
    assert LLMResponse().is_replay_turn
