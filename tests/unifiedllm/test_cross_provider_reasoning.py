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
    ReasoningReplayError,
    capture_chat_state,
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
    source = CompletionClient(
        model="anthropic/claude-sonnet-4",
        api_key="account-a",
        api_base="https://gateway-a.example/v1",
    )
    target = CompletionClient(
        model="anthropic/claude-sonnet-4",
        api_key="account-b",
        api_base="https://gateway-b.example/v1",
    )
    try:
        with patch(
            "litellm.completion", side_effect=[_anthropic_response(), _anthropic_response()]
        ) as completion:
            first = source.call([{"role": "user", "content": "run"}], tools=[TOOL])
            target.call(_render(first), tools=[TOOL])

        assert first.reasoning == "Check the inputs."
        assert first.llm_state is not None
        assert first.llm_state["payload"]["thinking_blocks"] == ANTHROPIC_THINKING
        assistant = next(
            message
            for message in completion.call_args_list[1].kwargs["messages"]
            if message.get("role") == "assistant"
        )
        assert assistant["thinking_blocks"] == ANTHROPIC_THINKING
        assert assistant["thinking_blocks"] is first.llm_state["payload"]["thinking_blocks"]
        assert assistant["content"] is None
    finally:
        source.close()
        target.close()


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
    source = CompletionClient(
        model="gemini/gemini-2.5-pro",
        api_key="account-a",
        api_base="https://gateway-a.example/v1",
    )
    target = CompletionClient(
        model="gemini/gemini-2.5-pro",
        api_key="account-b",
        api_base="https://gateway-b.example/v1",
    )
    try:
        with patch(
            "litellm.completion", side_effect=[_gemini_response(), _gemini_response()]
        ) as completion:
            first = source.call([{"role": "user", "content": "run"}], tools=[TOOL])
            target.call(_render(first), tools=[TOOL])

        assert [call.id for call in first.tool_calls] == ["call_1", "call_2"]
        assert first.llm_state is not None
        assert "discard-me" not in json.dumps(first.llm_state)
        assert [call["id"] for call in first.llm_state["payload"]["carrier"]["tool_calls"]] == [
            "call_1",
            "call_2",
        ]
        assistant = next(
            message
            for message in completion.call_args_list[1].kwargs["messages"]
            if message.get("role") == "assistant"
        )
        assert assistant["provider_specific_fields"] == {
            "thought_signatures": [GEMINI_SIGNATURE, GEMINI_SIGNATURE_2]
        }
        assert (
            assistant["provider_specific_fields"]["thought_signatures"]
            is first.llm_state["payload"]["provider_specific_fields"]["thought_signatures"]
        )
        assert [call["id"] for call in assistant["tool_calls"]] == [
            f"call_1__thought__{GEMINI_SIGNATURE}",
            f"call_2__thought__{GEMINI_SIGNATURE_2}",
        ]
        assert [
            call["provider_specific_fields"]["thought_signature"]
            for call in assistant["tool_calls"]
        ] == [GEMINI_SIGNATURE, GEMINI_SIGNATURE_2]
    finally:
        source.close()
        target.close()


def test_gateway_routed_gemini_keeps_inline_signatures_private() -> None:
    client = CompletionClient(
        model="openai/gcp/google/gemini-3.1-pro-preview",
        api_base="https://inference-api.example/v1",
    )
    try:
        with patch(
            "litellm.completion", side_effect=[_gemini_response(), _gemini_response()]
        ) as completion:
            first = client.call([{"role": "user", "content": "run"}], tools=[TOOL])
            client.call(_render(first), tools=[TOOL])

        assert [call.id for call in first.tool_calls] == ["call_1", "call_2"]
        assert first.llm_state is not None
        tool_state = first.llm_state["payload"]["tool_calls"]
        assert [call["inline_thought_signature"] for call in tool_state] == [
            GEMINI_SIGNATURE,
            GEMINI_SIGNATURE_2,
        ]
        assistant = next(
            message
            for message in completion.call_args_list[1].kwargs["messages"]
            if message.get("role") == "assistant"
        )
        assert [call["id"] for call in assistant["tool_calls"]] == [
            f"call_1__thought__{GEMINI_SIGNATURE}",
            f"call_2__thought__{GEMINI_SIGNATURE_2}",
        ]
        tool_results = [
            message
            for message in completion.call_args_list[1].kwargs["messages"]
            if message.get("role") == "tool"
        ]
        assert [message["tool_call_id"] for message in tool_results] == [
            f"call_1__thought__{GEMINI_SIGNATURE}",
            f"call_2__thought__{GEMINI_SIGNATURE_2}",
        ]
    finally:
        client.close()


def test_incompatible_gemini_state_keeps_tool_result_ids_public() -> None:
    source = CompletionClient(
        model="openai/gcp/google/gemini-3.1-pro-preview",
        api_base="https://inference-api.example/v1",
    )
    try:
        with patch("litellm.completion", return_value=_gemini_response()):
            first = source.call([{"role": "user", "content": "run"}], tools=[TOOL])

        prepared = prepare_chat_messages(_render(first), replay_scope("openai/gpt-5.6", "chat", {}))
        assistant = next(message for message in prepared if message.get("tool_calls"))
        tool_results = [message for message in prepared if message.get("role") == "tool"]
        assert [call["id"] for call in assistant["tool_calls"]] == ["call_1", "call_2"]
        assert [message["tool_call_id"] for message in tool_results] == [
            "call_1",
            "call_2",
        ]
        assert "__thought__" not in json.dumps(prepared)
    finally:
        source.close()


@pytest.mark.parametrize("mutation", ["drop", "reorder", "duplicate", "text", "name", "arguments"])
def test_gemini_tool_state_warns_and_demotes_when_public_calls_change(
    mutation: str, caplog: pytest.LogCaptureFixture
) -> None:
    client = CompletionClient(model="gemini/gemini-2.5-pro", api_key="account-a")
    try:
        with patch("litellm.completion", return_value=_gemini_response()):
            first = client.call([{"role": "user", "content": "run"}], tools=[TOOL])

        assert first.llm_state is not None
        rendered = _render(first)
        assistant = next(message for message in rendered if message.get("tool_calls"))
        if mutation == "drop":
            assistant["tool_calls"].pop()
        elif mutation == "reorder":
            assistant["tool_calls"].reverse()
        elif mutation == "duplicate":
            assistant["tool_calls"][1]["id"] = assistant["tool_calls"][0]["id"]
        elif mutation == "text":
            assistant["content"] = "replacement"
        elif mutation == "name":
            assistant["tool_calls"][0]["function"]["name"] = "replacement"
        else:
            assistant["tool_calls"][0]["function"]["arguments"] = '{"code":"changed"}'

        prepared = prepare_chat_messages(rendered, first.llm_state["scope"])
        replayed = next(message for message in prepared if message.get("tool_calls"))
        assert "provider_specific_fields" not in replayed
        assert all("provider_specific_fields" not in call for call in replayed["tool_calls"])
        assert replayed["content"].startswith("Inspect the value.")
        if mutation == "text":
            assert replayed["content"].endswith("replacement")
        assert GEMINI_SIGNATURE not in json.dumps(prepared)
        assert GEMINI_SIGNATURE_2 not in json.dumps(prepared)
        assert "public assistant carrier changed" in caplog.text
    finally:
        client.close()


def test_public_thinking_content_blocks_are_stripped(caplog: pytest.LogCaptureFixture) -> None:
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "private", "signature": "secret-a"},
                {"type": "text", "text": "public"},
                {"type": "redacted_thinking", "data": "secret-b"},
            ],
        }
    ]

    prepared = prepare_chat_messages(messages, None)
    assert prepared == [{"role": "assistant", "content": [{"type": "text", "text": "public"}]}]
    assert "secret-a" not in repr(prepared)
    assert "secret-b" not in repr(prepared)
    assert "Removed untrusted provider reasoning fields" in caplog.text


@pytest.mark.parametrize("target_model", [None, "openai/gpt-4o"])
def test_public_inline_signature_is_stripped_when_private_field_confirms_it(
    target_model: str | None,
) -> None:
    raw_id = f"call_1__thought__{GEMINI_SIGNATURE}"
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call(raw_id, {"thought_signature": GEMINI_SIGNATURE})],
        },
        {"role": "tool", "tool_call_id": raw_id, "content": "complete"},
    ]

    scope = replay_scope(target_model, "chat", {}) if target_model else None
    prepared = prepare_chat_messages(messages, scope)
    assert prepared[0]["tool_calls"][0]["id"] == "call_1"
    assert "provider_specific_fields" not in prepared[0]["tool_calls"][0]
    assert prepared[1]["tool_call_id"] == "call_1"
    assert GEMINI_SIGNATURE not in json.dumps(prepared)
    assert messages[0]["tool_calls"][0]["id"] == raw_id
    assert messages[1]["tool_call_id"] == raw_id


def test_direct_gemini_inline_signatures_cannot_bypass_the_envelope() -> None:
    raw_id = f"call_1__thought__{GEMINI_SIGNATURE}"
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_tool_call(raw_id)],
        },
        {"role": "tool", "tool_call_id": raw_id, "content": "complete"},
    ]

    prepared = prepare_chat_messages(messages, replay_scope("gemini/gemini-2.5-pro", "chat", {}))
    assert prepared[0]["tool_calls"][0]["id"] == "call_1"
    assert prepared[1]["tool_call_id"] == "call_1"
    assert GEMINI_SIGNATURE not in json.dumps(prepared)


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
        assert first.llm_state is None

        rendered = _render(first)
        assistant = next(message for message in rendered if message.get("tool_calls"))
        assert assistant["tool_calls"][0]["id"] == call_id
        prepared = prepare_chat_messages(rendered, None)
        assistant = next(message for message in prepared if message.get("tool_calls"))
        assert assistant["tool_calls"][0]["id"] == call_id
        paired = prepare_chat_messages(
            [assistant, {"role": "tool", "tool_call_id": call_id, "content": "complete"}],
            None,
        )
        assert paired[0]["tool_calls"][0]["id"] == call_id
        assert paired[1]["tool_call_id"] == call_id
    finally:
        client.close()


def test_missing_thought_signature_is_normal_but_malformed_signature_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    scope = replay_scope("gemini/gemini-2.5-pro", "chat", {})
    assert (
        capture_chat_state(
            Message(role="assistant", content=None, tool_calls=[_tool_call()]), scope
        )
        is None
    )
    assert not caplog.text

    malformed = Message(
        role="assistant",
        content=None,
        tool_calls=[_tool_call("call_1", {"thought_signature": 42})],
    )
    with pytest.raises(ReasoningReplayError, match="thought_signature"):
        capture_chat_state(malformed, scope)

    with pytest.raises(ReasoningReplayError, match="expected a mapping"):
        capture_chat_state({"tool_calls": ["not-a-tool-call"]}, scope)


def test_unknown_provider_state_warns_but_unknown_envelope_state_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    scope = replay_scope("gemini/gemini-2.5-pro", "chat", {})
    assert (
        capture_chat_state(
            {"provider_specific_fields": {"future_reasoning_state": "opaque"}}, scope
        )
        is None
    )
    assert "retention may need updating" in caplog.text

    state = {
        "version": 1,
        "scope": scope,
        "format": "litellm-chat",
        "payload": {
            "thinking_blocks": [{"type": "thinking", "signature": "opaque"}],
            "future_reasoning_state": "opaque",
        },
    }
    carrier = ReplayCarryingMessage({"role": "assistant", "content": "answer"}, state)
    with pytest.raises(ReasoningReplayError, match="malformed or unsupported"):
        prepare_chat_messages([carrier], scope)


def test_cross_provider_replay_warns_hides_opaque_state_and_keeps_reasoning_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
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
        assert "is incompatible with" in caplog.text
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
    chat_carrier = ReplayCarryingMessage({"role": "assistant", "content": ""}, responses_state)
    assert prepare_chat_messages([chat_carrier], "chat:openai:sha256:target") == []

    chat_state = {
        "version": 1,
        "scope": "chat:openai:sha256:source",
        "format": "litellm-chat",
        "payload": {"reasoning_items": [{"type": "reasoning"}], "state_only": True},
    }
    assert (
        prepare_responses_batch(
            [{"role": "assistant", "content": ""}],
            chat_state,
            "responses:openai:sha256:target",
        )
        == []
    )


def test_state_only_carrier_mutation_warns_and_preserves_public_content(
    caplog: pytest.LogCaptureFixture,
) -> None:
    chat_scope = "chat:openai:sha256:model"
    chat_state = {
        "version": 1,
        "scope": chat_scope,
        "format": "litellm-chat",
        "payload": {
            "reasoning_items": [{"opaque": "chat-secret"}],
            "carrier": {"content": "", "tool_calls": []},
            "state_only": True,
        },
    }
    chat_carrier = ReplayCarryingMessage(
        {"role": "assistant", "content": "middleware text"},
        chat_state,
        "portable reasoning",
    )
    assert prepare_chat_messages([chat_carrier], chat_scope) == [
        {"role": "assistant", "content": "portable reasoning\n\nmiddleware text"}
    ]

    responses_scope = "responses:openai:sha256:model"
    responses_state = {
        "version": 1,
        "scope": responses_scope,
        "format": "openai-responses",
        "payload": {
            "items": [RESPONSES_REASONING],
            "order": [{"type": "reasoning", "index": 0}],
            "state_only": True,
        },
    }
    assert prepare_responses_batch(
        [{"role": "assistant", "content": "middleware text"}],
        responses_state,
        responses_scope,
        "portable reasoning",
    ) == [{"role": "assistant", "content": "portable reasoning\n\nmiddleware text"}]
    assert "public assistant carrier changed" in caplog.text
    assert "empty public carrier changed" in caplog.text


def test_legacy_state_warns_while_malformed_current_state_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    legacy = ReplayCarryingMessage(
        {"role": "assistant", "content": "answer"},
        {"reasoning_items": [{"opaque": "legacy"}]},
        "portable reasoning",
    )
    assert prepare_chat_messages([legacy], None)[0]["content"] == ("portable reasoning\n\nanswer")
    assert "unsupported or legacy version" in caplog.text

    scope = "chat:openai:sha256:model"
    malformed_states = [
        {"version": 1, "scope": scope, "format": "litellm-chat", "payload": []},
        {"version": 1, "scope": scope, "format": "typo", "payload": {}},
        {"version": 1, "scope": scope, "format": "litellm-chat", "payload": {}, "extra": 1},
    ]
    for state in malformed_states:
        malformed = ReplayCarryingMessage({"role": "assistant", "content": "answer"}, state)
        with pytest.raises(ReasoningReplayError, match="Malformed version-1"):
            prepare_chat_messages([malformed], scope)


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


def test_non_openai_responses_scope_rejects_capture_and_restore() -> None:
    fabricated_scope = "responses:anthropic:sha256:untrusted"
    with pytest.raises(ReasoningReplayError, match="only supports.*OpenAI and Azure"):
        capture_responses_state([RESPONSES_REASONING], fabricated_scope)

    state = {
        "version": 1,
        "scope": fabricated_scope,
        "format": "openai-responses",
        "payload": {"items": [RESPONSES_REASONING], "order": []},
    }
    with pytest.raises(ReasoningReplayError, match="only supported for OpenAI and Azure"):
        prepare_responses_batch([RESPONSES_MESSAGE], state, fabricated_scope, "Check the evidence.")


@pytest.mark.parametrize(
    ("model", "environment"),
    [
        ("anthropic/claude-sonnet-4", "ANTHROPIC_API_BASE"),
        ("gemini/gemini-2.5-pro", "GEMINI_API_BASE"),
    ],
)
def test_closed_provider_scope_is_stable_across_environment_endpoints(
    monkeypatch, model: str, environment: str
) -> None:
    monkeypatch.setenv(environment, "https://issuer-a.example/v1")
    first = replay_scope(model, "chat", {"api_key": "account-a"})
    monkeypatch.setenv(environment, "https://issuer-b.example/v1")
    second = replay_scope(model, "chat", {"api_key": "account-a"})

    assert first is not None
    assert second is not None
    assert first == second
