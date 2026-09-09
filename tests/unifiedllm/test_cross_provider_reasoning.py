# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Closed-provider opaque state and provider-independent text reasoning replay."""

import json
from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import AsyncMock, patch

import pytest
from litellm.types.utils import Choices, Message, ModelResponse

from nooa._llm_state import ReplayCarryingMessage
from nooa.context_blocks.events import ToolCallEvent, ToolResult
from nooa.context_blocks.formatter import (
    OpenAIProviderFormatter,
    ResponsesProviderFormatter,
    XMLBlockFormatter,
)
from nooa.context_blocks.models import ResolvedBlock, Role
from nooa.unifiedllm import CompletionClient, LLMResponse, ResponsesClient, Tool
from nooa.unifiedllm.replay_state import (
    capture_responses_state,
    prepare_chat_messages,
    prepare_responses_batch,
    replay_scope,
)

ANTHROPIC_THINKING = [
    {"type": "thinking", "thinking": "Check the inputs.", "signature": "anthropic-sig"},
    {"type": "redacted_thinking", "data": "anthropic-redacted"},
]
GEMINI_SIGNATURE = "Z2VtaW5pLXNpZ25hdHVyZQ=="
GEMINI_SIGNATURE_2 = "c2Vjb25kLXNpZ25hdHVyZQ=="


def _execute_python(code: str) -> str:
    return code


TOOL = Tool(name="execute_python", description="Run code", callable=_execute_python)


def _tool_call(call_id: str = "call_1", provider_specific_fields: dict | None = None) -> dict:
    call = {
        "id": call_id,
        "type": "function",
        "function": {"name": "execute_python", "arguments": '{"code":"print(1)"}'},
    }
    if provider_specific_fields:
        call["provider_specific_fields"] = provider_specific_fields
    return call


def _chat_response(message: Message, finish_reason: str = "tool_calls") -> ModelResponse:
    return ModelResponse(
        model="test-model",
        choices=[Choices(message=message, finish_reason=finish_reason)],
    )


def _anthropic_response() -> ModelResponse:
    return _chat_response(
        Message(
            role="assistant",
            content=None,
            tool_calls=[_tool_call()],
            thinking_blocks=cast(Any, ANTHROPIC_THINKING),
            reasoning_content="Check the inputs.",
        )
    )


def _gemini_response() -> ModelResponse:
    return _chat_response(
        Message(
            role="assistant",
            content=None,
            tool_calls=[
                _tool_call(
                    f"call_1__thought__{GEMINI_SIGNATURE}",
                    {"thought_signature": GEMINI_SIGNATURE, "private": "discard-me"},
                ),
                _tool_call(
                    f"call_2__thought__{GEMINI_SIGNATURE_2}",
                    {"thought_signature": GEMINI_SIGNATURE_2},
                ),
            ],
            provider_specific_fields={
                "thought_signatures": [GEMINI_SIGNATURE, GEMINI_SIGNATURE_2],
                "private": "discard-me",
            },
            reasoning_content="Inspect the value.",
        )
    )


def _render(response: LLMResponse, *, responses: bool = False) -> list[dict]:
    blocks = [ResolvedBlock(key="turn", content="", role=Role.ASSISTANT, event=response)]
    blocks.extend(
        ResolvedBlock(
            key=f"execution-{call.id}",
            content="",
            role=Role.ASSISTANT,
            event=ToolCallEvent(
                tool_call_id=call.id,
                name=call.name,
                arguments=(
                    json.loads(call.arguments)
                    if isinstance(call.arguments, str)
                    else call.arguments
                ),
                llm_response_id=response.id,
                result=ToolResult(tool_call_id=call.id, content="complete"),
            ),
        )
        for call in response.tool_calls
    )
    neutral = XMLBlockFormatter().format(blocks)
    formatter = ResponsesProviderFormatter() if responses else OpenAIProviderFormatter()
    return formatter.format(neutral)


def test_anthropic_thinking_blocks_round_trip_exactly() -> None:
    client = CompletionClient(model="anthropic/claude-sonnet-4", api_key="account-a")
    try:
        with patch(
            "litellm.completion", side_effect=[_anthropic_response(), _anthropic_response()]
        ) as completion:
            first = client.call([{"role": "user", "content": "run"}], tools=[TOOL])
            client.call(_render(first), tools=[TOOL])

        assert first.reasoning == "Check the inputs."
        assert first.llm_state is not None
        assert first.llm_state["payload"]["thinking_blocks"] == ANTHROPIC_THINKING
        assistant = next(
            message
            for message in completion.call_args_list[1].kwargs["messages"]
            if message.get("role") == "assistant"
        )
        assert assistant["thinking_blocks"] == ANTHROPIC_THINKING
        assert assistant["content"] is None
    finally:
        client.close()


@pytest.mark.asyncio
async def test_async_anthropic_capture_matches_sync() -> None:
    client = CompletionClient(model="anthropic/claude-sonnet-4", api_key="account-a")
    try:
        with patch("litellm.acompletion", AsyncMock(return_value=_anthropic_response())):
            response = await client.acall([{"role": "user", "content": "run"}], tools=[TOOL])
        assert response.llm_state is not None
        assert response.llm_state["payload"]["thinking_blocks"] == ANTHROPIC_THINKING
    finally:
        await client.aclose()


def test_gemini_signatures_round_trip_without_becoming_public_call_ids() -> None:
    client = CompletionClient(model="gemini/gemini-2.5-pro", api_key="account-a")
    try:
        with patch(
            "litellm.completion", side_effect=[_gemini_response(), _gemini_response()]
        ) as completion:
            first = client.call([{"role": "user", "content": "run"}], tools=[TOOL])
            client.call(_render(first), tools=[TOOL])

        assert [call.id for call in first.tool_calls] == ["call_1", "call_2"]
        assert "discard-me" not in json.dumps(first.llm_state)
        assistant = next(
            message
            for message in completion.call_args_list[1].kwargs["messages"]
            if message.get("role") == "assistant"
        )
        assert assistant["provider_specific_fields"] == {
            "thought_signatures": [GEMINI_SIGNATURE, GEMINI_SIGNATURE_2]
        }
        assert [call["id"] for call in assistant["tool_calls"]] == ["call_1", "call_2"]
        assert [
            call["provider_specific_fields"]["thought_signature"]
            for call in assistant["tool_calls"]
        ] == [GEMINI_SIGNATURE, GEMINI_SIGNATURE_2]
    finally:
        client.close()


def test_non_gemini_tool_call_id_with_thought_substring_is_unchanged() -> None:
    call_id = "call_business__thought__phase"
    response = _chat_response(
        Message(role="assistant", content=None, tool_calls=[_tool_call(call_id)])
    )
    client = CompletionClient(model="openai/gpt-4o", api_key="account-a")
    try:
        with patch("litellm.completion", return_value=response):
            first = client.call([{"role": "user", "content": "run"}], tools=[TOOL])
        assert first.tool_calls[0].id == call_id

        rendered = _render(first)
        assistant = next(message for message in rendered if message.get("tool_calls"))
        assert assistant["tool_calls"][0]["id"] == call_id
        prepared = prepare_chat_messages(rendered, None)
        assistant = next(message for message in prepared if message.get("tool_calls"))
        assert assistant["tool_calls"][0]["id"] == call_id
    finally:
        client.close()


def test_cross_provider_replay_hides_opaque_state_but_keeps_reasoning_text() -> None:
    source = CompletionClient(model="gemini/gemini-2.5-pro", api_key="account-a")
    target = CompletionClient(model="anthropic/claude-sonnet-4", api_key="account-a")
    try:
        with patch("litellm.completion", return_value=_gemini_response()):
            first = source.call([{"role": "user", "content": "run"}], tools=[TOOL])
        with patch(
            "litellm.completion",
            return_value=_chat_response(Message(role="assistant", content="done"), "stop"),
        ) as completion:
            target.call(_render(first), tools=[TOOL])

        replayed = completion.call_args.kwargs["messages"]
        assistant = next(message for message in replayed if message.get("role") == "assistant")
        assert assistant["content"] == "Inspect the value."
        assert [call["id"] for call in assistant["tool_calls"]] == ["call_1", "call_2"]
        assert GEMINI_SIGNATURE not in json.dumps(replayed)
        assert GEMINI_SIGNATURE_2 not in json.dumps(replayed)
    finally:
        source.close()
        target.close()


def test_plain_reasoning_replays_as_ordinary_text_for_every_model() -> None:
    source = CompletionClient(model="deepseek/deepseek-reasoner", api_key="account-a")
    target = CompletionClient(model="openai/gpt-4o", api_key="account-a")
    response = _chat_response(
        Message(
            role="assistant",
            content="Visible answer.",
            reasoning_content="Plain reasoning.",
        ),
        "stop",
    )
    try:
        with patch("litellm.completion", return_value=response):
            first = source.call([{"role": "user", "content": "think"}])
        with patch("litellm.completion", return_value=response) as completion:
            target.call(_render(first))

        assert first.llm_state is None
        assistant = next(
            message
            for message in completion.call_args.kwargs["messages"]
            if message.get("role") == "assistant"
        )
        assert assistant == {
            "role": "assistant",
            "content": "Plain reasoning.\n\nVisible answer.",
        }
    finally:
        source.close()
        target.close()


def _responses_output(*items: dict) -> SimpleNamespace:
    return SimpleNamespace(output=list(items), output_text="", status="completed", usage=None)


RESPONSES_REASONING = {
    "id": "rs_1",
    "type": "reasoning",
    "encrypted_content": "openai-secret",
    "summary": [{"type": "summary_text", "text": "Check the evidence."}],
}
RESPONSES_MESSAGE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "Answer.", "annotations": []}],
}


def test_responses_summary_stays_exact_on_match_and_demotes_on_model_change() -> None:
    source = ResponsesClient(model="openai/gpt-5.6", api_key="account-a")
    target = ResponsesClient(model="openai/gpt-5.7", api_key="account-a")
    try:
        with patch(
            "litellm.responses",
            return_value=_responses_output(RESPONSES_REASONING, RESPONSES_MESSAGE),
        ):
            first = source.call([{"role": "user", "content": "think"}])

        assert first.reasoning == "Check the evidence."
        rendered = _render(first, responses=True)
        with patch(
            "litellm.responses", return_value=_responses_output(RESPONSES_MESSAGE)
        ) as matching:
            source.call(rendered)
        assert matching.call_args.kwargs["input"][:2] == [
            RESPONSES_REASONING,
            {"role": "assistant", "content": "Answer."},
        ]

        with patch(
            "litellm.responses", return_value=_responses_output(RESPONSES_MESSAGE)
        ) as changed:
            target.call(rendered)
        assert changed.call_args.kwargs["input"] == [
            {"role": "assistant", "content": "Check the evidence.\n\nAnswer."}
        ]
        assert "openai-secret" not in json.dumps(changed.call_args.kwargs["input"])
    finally:
        source.close()
        target.close()


def test_reasoning_only_responses_turn_demotes_without_an_empty_message() -> None:
    source = ResponsesClient(model="openai/gpt-5.6", api_key="account-a")
    target = ResponsesClient(model="openai/gpt-5.7", api_key="account-a")
    try:
        with patch("litellm.responses", return_value=_responses_output(RESPONSES_REASONING)):
            first = source.call([{"role": "user", "content": "think"}])
        with patch(
            "litellm.responses", return_value=_responses_output(RESPONSES_MESSAGE)
        ) as changed:
            target.call(_render(first, responses=True))

        assert changed.call_args.kwargs["input"] == [
            {"role": "assistant", "content": "Check the evidence."}
        ]
    finally:
        source.close()
        target.close()


def test_state_only_turn_drops_empty_carrier_across_api_styles() -> None:
    responses_state = {
        "version": 1,
        "scope": "responses:openai:sha256:source",
        "format": "openai-responses",
        "payload": {"items": [RESPONSES_REASONING], "order": [], "state_only": True},
    }
    chat_carrier = ReplayCarryingMessage(
        {"role": "assistant", "content": ""}, responses_state
    )
    assert prepare_chat_messages([chat_carrier], "chat:openai:sha256:target") == []

    chat_state = {
        "version": 1,
        "scope": "chat:openai:sha256:source",
        "format": "litellm-chat",
        "payload": {"reasoning_items": [{"type": "reasoning"}], "state_only": True},
    }
    assert prepare_responses_batch(
        [{"role": "assistant", "content": ""}],
        chat_state,
        "responses:openai:sha256:target",
    ) == []


def test_responses_demotion_keeps_native_output_content_valid() -> None:
    native_message = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "Answer."}],
    }

    assert prepare_responses_batch([native_message], None, None, "Check the evidence.") == [
        {"role": "assistant", "content": "Check the evidence."},
        native_message,
    ]


@pytest.mark.parametrize(
    ("model", "api_style"),
    [
        ("anthropic/claude-sonnet-4", "responses"),
        ("gemini/gemini-2.5-pro", "responses"),
        ("vertex_ai/gemini-2.5-pro", "responses"),
        ("vertex_ai/gemini-2.5-pro", "chat"),
    ],
)
def test_unverified_closed_provider_routes_have_no_opaque_replay_scope(
    model: str, api_style: Literal["chat", "responses"]
) -> None:
    assert replay_scope(model, api_style, {"api_key": "account-a"}) is None


def test_non_openai_responses_scope_cannot_capture_or_restore_opaque_items() -> None:
    fabricated_scope = "responses:anthropic:sha256:untrusted"
    assert capture_responses_state([RESPONSES_REASONING], fabricated_scope) is None

    state = {
        "version": 1,
        "scope": fabricated_scope,
        "format": "openai-responses",
        "payload": {"items": [RESPONSES_REASONING], "order": []},
    }
    assert prepare_responses_batch(
        [RESPONSES_MESSAGE], state, fabricated_scope, "Check the evidence."
    ) == [
        {"role": "assistant", "content": "Check the evidence."},
        RESPONSES_MESSAGE,
    ]


@pytest.mark.parametrize(
    ("model", "environment"),
    [
        ("anthropic/claude-sonnet-4", "ANTHROPIC_API_BASE"),
        ("gemini/gemini-2.5-pro", "GEMINI_API_BASE"),
    ],
)
def test_closed_provider_environment_endpoint_partitions_scope(
    monkeypatch, model: str, environment: str
) -> None:
    monkeypatch.setenv(environment, "https://issuer-a.example/v1")
    first = replay_scope(model, "chat", {"api_key": "account-a"})
    monkeypatch.setenv(environment, "https://issuer-b.example/v1")
    second = replay_scope(model, "chat", {"api_key": "account-a"})

    assert first is not None
    assert second is not None
    assert first != second
