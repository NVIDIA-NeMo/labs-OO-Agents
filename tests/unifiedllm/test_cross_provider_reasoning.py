# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Closed-provider opaque state and provider-independent text reasoning replay."""

import json
from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import patch

import pytest
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import Choices, Message, ModelResponse

from nooa.context_blocks.events import ToolCallEvent, ToolResult
from nooa.context_blocks.formatter import (
    OpenAIProviderFormatter,
    ResponsesProviderFormatter,
    XMLBlockFormatter,
)
from nooa.context_blocks.models import ResolvedBlock, Role
from nooa.unifiedllm import CompletionClient, LLMResponse, ResponsesClient, Tool
from nooa.unifiedllm.chat_parts import capture_chat_parts
from nooa.unifiedllm.replay_state import (
    ReasoningReplayError,
    prepare_chat_messages,
    replay_scope,
)
from nooa.unifiedllm.response_parts import capture_parts

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
    blocks = [
        ResolvedBlock(key="turn", content=response.content, role=Role.ASSISTANT, event=response)
    ]
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


def test_public_thinking_content_blocks_require_a_response(
    caplog: pytest.LogCaptureFixture,
) -> None:
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

    with pytest.raises(ReasoningReplayError, match="LLMResponse") as error:
        prepare_chat_messages(messages, None)
    assert "secret-a" not in str(error.value) + caplog.text


@pytest.mark.parametrize("target_model", [None, "openai/gpt-4o"])
def test_public_inline_signature_requires_a_response(
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
    with pytest.raises(ReasoningReplayError, match="LLMResponse"):
        prepare_chat_messages(messages, scope)
    assert messages[0]["tool_calls"][0]["id"] == raw_id


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

    with pytest.raises(ReasoningReplayError, match="LLMResponse"):
        prepare_chat_messages(messages, replay_scope("gemini/gemini-2.5-pro", "chat", {}))


def test_missing_thought_signature_is_normal_but_malformed_signature_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    scope = replay_scope("gemini/gemini-2.5-pro", "chat", {})
    parts = capture_chat_parts(
        Message(role="assistant", content=None, tool_calls=[_tool_call()]), scope
    )
    assert parts[-1].native is None
    assert not caplog.text

    malformed = Message(
        role="assistant",
        content=None,
        tool_calls=[_tool_call("call_1", {"thought_signature": 42})],
    )
    with pytest.raises(ReasoningReplayError, match="thought_signature"):
        capture_chat_parts(malformed, scope)

    with pytest.raises(ReasoningReplayError, match="expected a mapping"):
        capture_chat_parts({"tool_calls": ["not-a-tool-call"]}, scope)


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
        assert "Incompatible assistant turn" in caplog.text
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

        assert all(part.native is None for part in first.parts)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("has_answer", [False, True])
async def test_summary_without_encrypted_content_is_portable_text(is_async, has_answer) -> None:
    summary = {
        key: value for key, value in RESPONSES_REASONING.items() if key != "encrypted_content"
    }
    raw = ResponsesAPIResponse(
        id="resp",
        created_at=0,
        model="gpt-5.6",
        status="completed",
        output=[summary, RESPONSES_MESSAGE] if has_answer else [summary],
    )
    async with ResponsesClient(
        model="openai/gpt-5.6", api_key="test", api_base="https://gateway.example/v1"
    ) as client:
        target = "litellm.aresponses" if is_async else "litellm.responses"
        with patch(target, return_value=raw) as request:
            messages = [{"role": "user", "content": "think"}]
            first = await client.acall(messages) if is_async else client.call(messages)
        assert request.call_args.kwargs.get("include") is None
        assert first.reasoning == "Check the evidence."
        assert all(part.native is None for part in first.parts)
        restored = LLMResponse.model_validate_json(first.model_dump_json())
        with patch(target, return_value=raw) as replay:
            rendered = _render(restored, responses=True)
            if is_async:
                await client.acall(rendered)
            else:
                client.call(rendered)
        expected = [{"role": "assistant", "content": "Check the evidence."}]
        if has_answer:
            expected.append({"role": "assistant", "content": "Answer."})
        assert replay.call_args.kwargs["input"] == expected


@pytest.mark.parametrize("encrypted", ["", 42, False])
def test_malformed_ciphertext_is_not_hidden_by_a_summary_only_item(encrypted) -> None:
    scope = replay_scope("openai/gpt-5.6", "responses", {})
    with pytest.raises(ReasoningReplayError, match="encrypted reasoning"):
        capture_parts(
            [
                {"type": "reasoning", "summary": []},
                {**RESPONSES_REASONING, "encrypted_content": encrypted},
            ],
            scope,
        )


@pytest.mark.parametrize(
    "unsupported",
    [
        {**RESPONSES_MESSAGE, "content": [{"type": "refusal", "refusal": "Cannot comply."}]},
        {
            "type": "web_search_call",
            "id": "ws_1",
            "status": "completed",
            "action": {"type": "search", "query": "reference"},
        },
    ],
)
def test_opaque_reasoning_cannot_be_retained_beside_unprojectable_output(unsupported) -> None:
    raw = ResponsesAPIResponse.model_validate(
        {
            "id": "resp",
            "created_at": 0,
            "model": "gpt-5.6",
            "status": "completed",
            "output": [RESPONSES_REASONING, unsupported],
        }
    )
    with ResponsesClient(model="openai/gpt-5.6", api_key="test") as client:
        with patch("litellm.responses", return_value=raw):
            with pytest.raises(ReasoningReplayError, match="Unsupported Responses output"):
                client.call([{"role": "user", "content": "request"}])


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


@pytest.mark.parametrize(
    "model", ["gemini/gemini-2.5-pro", "openai/gcp/google/gemini-3.1-pro-preview"]
)
def test_multiple_gemini_signatures_remain_attached_to_owning_calls(model):
    scope = replay_scope(model, "chat", {})
    source = _gemini_response().choices[0].message
    turn = LLMResponse(parts=capture_chat_parts(source, scope), replay_scope=scope)
    resumed = LLMResponse.model_validate_json(turn.model_dump_json())
    wire = prepare_chat_messages(
        [
            resumed,
            {"role": "tool", "tool_call_id": "call_1", "content": "one"},
            {"role": "tool", "tool_call_id": "call_2", "content": "two"},
        ],
        scope,
    )
    assert [
        call["provider_specific_fields"]["thought_signature"] for call in wire[0]["tool_calls"]
    ] == [GEMINI_SIGNATURE, GEMINI_SIGNATURE_2]
    assert [item["tool_call_id"] for item in wire[1:]] == [
        f"call_1__thought__{GEMINI_SIGNATURE}",
        f"call_2__thought__{GEMINI_SIGNATURE_2}",
    ]
    assert "discard-me" not in json.dumps(wire)
    assert GEMINI_SIGNATURE not in json.dumps(resumed.public_message())


@pytest.mark.parametrize("call_id", ["business__thought__phase", "business__thought__"])
def test_non_gemini_literal_call_ids_stay_unchanged(call_id):
    scope = replay_scope("openai/gpt-4o", "chat", {})
    turn = LLMResponse(
        parts=capture_chat_parts(
            Message(role="assistant", content=None, tool_calls=[_tool_call(call_id)]), scope
        ),
        replay_scope=scope,
    )
    assert turn.tool_calls[0].id == call_id
    wire = prepare_chat_messages(
        [turn, {"role": "tool", "tool_call_id": call_id, "content": "done"}], None
    )
    assert wire[0]["tool_calls"][0]["id"] == wire[1]["tool_call_id"] == call_id


def test_unknown_provider_fields_warn_without_leaking_values(caplog):
    capture_chat_parts(
        {
            "role": "assistant",
            "content": "hello",
            "provider_specific_fields": {"future_reasoning_state": "secret"},
        },
        "chat:gemini:test",
    )
    assert "retention may need updating" in caplog.text
    assert "secret" not in caplog.text
