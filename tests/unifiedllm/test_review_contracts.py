# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider variation and public projection contracts from the full core review."""

import json
from typing import Any
from unittest.mock import patch

import pytest
from litellm import ModelResponse
from pydantic import BaseModel

from nooa.context_blocks.events import EventBase
from nooa.context_blocks.formatter import AnthropicProviderFormatter
from nooa.context_blocks.models import RenderedMessage, Role, ToolCallInfo
from nooa.llm_types import AssistantText, LLMResponse, ToolCall, assistant_message
from nooa.unifiedllm import CompletionClient, ResponsesClient
from nooa.unifiedllm.chat_parts import capture_chat_parts, project_chat_turn
from nooa.unifiedllm.errors import ReasoningReplayError
from nooa.unifiedllm.replay_state import prepare_chat_messages
from nooa.unifiedllm.response_parts import capture_parts


@pytest.mark.parametrize("api_style", ["chat", "responses"])
@pytest.mark.parametrize(
    ("location", "empty"),
    [
        ("message", None),
        ("call", None),
        ("thinking", None),
        ("message", {}),
        ("call", {}),
        ("thinking", []),
    ],
)
def test_empty_sdk_provider_fields_are_portable(api_style, location, empty):
    message = {
        "role": "assistant",
        "content": "answer",
        "tool_calls": [
            {"id": "c", "type": "function", "function": {"name": "run", "arguments": "{}"}}
        ],
    }
    if location == "thinking":
        message["thinking_blocks"] = empty
    elif location == "call":
        message["tool_calls"][0]["provider_specific_fields"] = empty
    else:
        message["provider_specific_fields"] = empty
    if api_style == "chat":
        assert prepare_chat_messages([message], None)[0] == message
    else:
        with ResponsesClient("openai/gpt-5.6", api_key="test") as client:
            wire, _ = client._transform_messages([message])
        assert wire[0]["content"] == "answer"
        assert wire[1]["call_id"] == "c"


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
                    "reasoning_items": [
                        {
                            "type": "reasoning",
                            "summary": [{"type": "summary_text", "text": "readable"}],
                        }
                    ],
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


def test_generic_events_do_not_expose_assistant_replay_hooks():
    for name in ("is_replay_turn", "replay_tool_calls", "render_message", "replay_content"):
        assert not hasattr(EventBase(), name)


@pytest.mark.parametrize("calls", [[None], ["bad"], [42], {}, "bad", [{"function": None}], [{}]])
@pytest.mark.parametrize("client_type", [CompletionClient, ResponsesClient])
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.asyncio
async def test_malformed_raw_calls_fail_before_transport(calls, client_type, is_async):
    with client_type("openai/gpt-5.6", api_key="test") as client:
        with patch("litellm.completion") as chat, patch("litellm.responses") as responses:
            with patch("litellm.acompletion") as achat, patch("litellm.aresponses") as aresponses:
                with pytest.raises(ReasoningReplayError, match="Malformed tool"):
                    messages = [{"role": "assistant", "tool_calls": calls}]
                    if is_async:
                        await client.acall(messages)
                    else:
                        client.call(messages)
                for request in (chat, responses, achat, aresponses):
                    request.assert_not_called()


@pytest.mark.parametrize(
    "item",
    [
        {"type": "message"},
        {"type": "message", "content": None},
        {"type": "message", "content": 42},
        {"type": "message", "content": "bad"},
        {"type": "message", "content": [{"type": "output_text"}]},
        {"type": "message", "content": [{"type": "output_text", "text": None}]},
        {"type": "message", "content": [{"type": "output_text", "text": 42}]},
        {"type": "function_call", "call_id": "c", "arguments": "{}"},
        {"type": "function_call", "call_id": "c", "name": "run"},
        {"type": "function_call", "call_id": "c", "name": None, "arguments": "{}"},
        {"type": "function_call", "call_id": "c", "name": "run", "arguments": {}},
    ],
)
@pytest.mark.parametrize("sdk_dump", [False, True])
def test_malformed_responses_fields_raise_contract_error(item, sdk_dump):
    if sdk_dump:

        class SDKItem(BaseModel):
            type: str
            content: Any = None
            call_id: Any = None
            name: Any = None
            arguments: Any = None

        item = SDKItem(**item)
    with pytest.raises(ReasoningReplayError):
        capture_parts([item], "responses:openai:model")


def test_explicit_raw_reasoning_field_is_not_rewritten():
    message = {"role": "assistant", "content": "answer", "reasoning_content": "caller reasoning"}
    prepared = prepare_chat_messages([message], "chat:openai:model")
    assert prepared == [message]
    assert prepared[0] is not message
    # Captured responses retain the separate automatic portable-replay policy.
    turn = LLMResponse(parts=capture_chat_parts(message, None))
    assert prepare_chat_messages([turn], None) == [
        {"role": "assistant", "content": "caller reasoning\n\nanswer"}
    ]
