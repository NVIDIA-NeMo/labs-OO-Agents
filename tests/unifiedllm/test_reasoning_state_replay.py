# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Issuer-gated capture and replay of opaque OpenAI reasoning state."""

import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import ModelResponse

from nooa.context_blocks.events import ToolCallEvent, ToolResult
from nooa.context_blocks.formatter import (
    OpenAIProviderFormatter,
    ResponsesProviderFormatter,
    XMLBlockFormatter,
)
from nooa.context_blocks.models import ResolvedBlock, Role
from nooa.llm_types import AssistantText, ToolCall
from nooa.storage.sqlite import SQLiteEventBackend, _ensure_schema
from nooa.unifiedllm import CompletionClient, LLMResponse, ResponsesClient, Tool
from nooa.unifiedllm.replay_state import (
    ReasoningReplayError,
    replay_scope,
)
from nooa.unifiedllm.unifiedllm import _ClientHttp

REASONING = {
    "id": "rs_1",
    "type": "reasoning",
    "encrypted_content": "provider-secret",
    "summary": [],
}
REASONING_2 = {
    "id": "rs_2",
    "type": "reasoning",
    "encrypted_content": "provider-secret-2",
    "summary": [],
}
MESSAGE = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "done", "annotations": []}],
}
CALL = {
    "id": "fc_1",
    "type": "function_call",
    "call_id": "call_1",
    "name": "execute_python",
    "arguments": '{"code":"print(1)"}',
    "status": "completed",
}
CALL_2 = {
    "id": "fc_2",
    "type": "function_call",
    "call_id": "call_2",
    "name": "execute_python",
    "arguments": '{"code":"print(2)"}',
    "status": "completed",
}


def _responses(*items: dict) -> SimpleNamespace:
    return SimpleNamespace(output=list(items), output_text="", status="completed", usage=None)


def _chat_tool_call(call_id: str = "call_1", code: str = "print(1)") -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": "execute_python",
            "arguments": json.dumps({"code": code}, separators=(",", ":")),
        },
    }


def _chat_response(
    *,
    reasoning_items: list[dict] | None = None,
    tool_calls: list[dict] | None = None,
) -> ModelResponse:
    return ModelResponse(
        model="gpt-5.6",
        choices=[
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [_chat_tool_call()] if tool_calls is None else tool_calls,
                    "reasoning_items": reasoning_items,
                },
            }
        ],
    )


def _tool(code: str) -> str:
    return code


TOOL = Tool(name="execute_python", description="Run code", callable=_tool)


def _response_blocks(response: LLMResponse) -> list[ResolvedBlock]:
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
    return blocks


def _render_responses(response: LLMResponse) -> list[dict]:
    neutral = XMLBlockFormatter().format(_response_blocks(response))
    return ResponsesProviderFormatter().format(neutral)


def _render_chat(response: LLMResponse) -> list[dict]:
    neutral = XMLBlockFormatter().format(_response_blocks(response))
    return OpenAIProviderFormatter().format(neutral)


@pytest.mark.parametrize("call_id", ["same", ""])
@pytest.mark.parametrize("archive", ["current", "legacy"])
def test_duplicate_ids_cannot_reuse_one_execution(call_id, archive, caplog):
    calls = [ToolCall(id=call_id, name="run", arguments="{}") for _ in range(2)]
    if archive == "legacy":
        turn = LLMResponse.model_validate(
            {
                "content": "answer",
                "tool_calls": [c.model_dump() for c in calls],
                "finish_reason": "tool_calls",
            },
            context={"archive": True},
        )
    else:
        turn = LLMResponse(parts=(AssistantText(text="answer"), *calls), finish_reason="tool_calls")
        turn = LLMResponse.model_validate_json(turn.model_dump_json())
    # Only one execution exists for the two calls, including after archive load.
    neutral = XMLBlockFormatter().format(_response_blocks(turn)[:2])
    wire = OpenAIProviderFormatter().format(neutral)
    assert len(turn.tool_calls) == 2  # Preserve the original for investigation.
    assert not any(m.get("tool_calls") or m.get("role") == "tool" for m in wire)
    assert turn.content == "answer"  # The record survives, the invalid batch does not replay.
    assert not any(m.get("content") == "answer" for m in wire)
    assert "omitting the assistant tool-call turn" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("shape", ["text_blocks", "phases", "phase_only", "trailing_message"])
@pytest.mark.parametrize("persistence", ["json", "sqlite"])
async def test_real_responses_message_structure_survives_json_resume(
    monkeypatch, is_async: bool, shape: str, persistence: str
) -> None:
    """Use real LiteLLM/SDK parsing and serialization, including its output_text property."""
    phased_message = {**MESSAGE, "id": "msg_final", "phase": "final_answer"}
    if shape == "text_blocks":
        output = [
            REASONING,
            {
                **MESSAGE,
                "content": [
                    {"type": "output_text", "text": "first", "annotations": []},
                    {"type": "output_text", "text": "second", "annotations": []},
                ],
            },
        ]
    elif shape == "phases":
        output = [
            REASONING,
            {
                **MESSAGE,
                "id": "msg_commentary",
                "phase": "commentary",
                "content": [{"type": "output_text", "text": "Checking.", "annotations": []}],
            },
            REASONING_2,
            phased_message,
        ]
    elif shape == "phase_only":
        output = [phased_message]
    else:
        output = [CALL, MESSAGE]

    raw = ResponsesAPIResponse.model_validate(
        {
            "id": "resp_test",
            "created_at": 0,
            "model": "gpt-5.6",
            "status": "completed",
            "output": output,
        }
    )
    bodies: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=raw.model_dump(mode="json"))

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        _ClientHttp, "_httpx_hardening", staticmethod(lambda: {"transport": transport})
    )
    client = ResponsesClient(
        model="openai/gpt-5.6", api_key="test", api_base="https://gateway.example/v1"
    )
    try:
        prompt = [{"role": "user", "content": "think"}]
        first = await client.acall(prompt) if is_async else client.call(prompt)
        assert first.content == raw.output_text
        # The fallback for providers without output_text must use identical text.
        assert client._extract_text_from_output(_responses(*output)) == raw.output_text
        if persistence == "json":
            resumed = LLMResponse.model_validate_json(first.model_dump_json())
        else:
            connection = sqlite3.connect(":memory:")
            try:
                _ensure_schema(connection)
                backend = SQLiteEventBackend(connection)
                backend.store("response", first)
                resumed = backend.get("response")
                assert isinstance(resumed, LLMResponse)
            finally:
                connection.close()
        assert resumed.raw_response is None
        assert resumed.replay_scope is not None
        assert any(part.native is not None for part in resumed.parts)
        rendered = _render_responses(resumed)
        if is_async:
            await client.acall(rendered)
        else:
            client.call(rendered)
        expected = output + (
            [{"type": "function_call_output", "call_id": "call_1", "output": "complete"}]
            if shape == "trailing_message"
            else []
        )
        assert bodies[1]["input"] == expected

        # A public text edit must still invalidate the saved message structure.
        resumed = resumed.replace_text("edited answer")
        if is_async:
            await client.acall(_render_responses(resumed))
        else:
            client.call(_render_responses(resumed))
        assert bodies[2]["input"][0] == {"role": "assistant", "content": "edited answer"}
        assert "encrypted_content" not in json.dumps(bodies[2]["input"])
        assert "phase" not in json.dumps(bodies[2]["input"])
    finally:
        await client.aclose()


def test_responses_state_is_hidden_from_a_different_model() -> None:
    source = ResponsesClient(model="openai/gpt-5.6", api_key="account-a")
    target = ResponsesClient(model="openai/gpt-5.7", api_key="account-a")
    try:
        with patch("litellm.responses", return_value=_responses(REASONING, MESSAGE)):
            first = source.call([{"role": "user", "content": "think"}])
        with patch("litellm.responses", return_value=_responses(MESSAGE)) as call:
            target.call(_render_responses(first))

        replay = call.call_args.kwargs["input"]
        assert REASONING not in replay
        assert {"role": "assistant", "content": "done"} in replay
        assert "provider-secret" not in repr(replay)
    finally:
        source.close()
        target.close()


def test_responses_model_override_uses_effective_replay_scope() -> None:
    client = ResponsesClient(model="openai/gpt-5.6", api_key="account-a")
    try:
        with patch("litellm.responses", return_value=_responses(REASONING, MESSAGE)):
            first = client.call([{"role": "user", "content": "think"}])

        with patch("litellm.responses", return_value=_responses(MESSAGE)) as call:
            client.call(
                _render_responses(first),
                model="anthropic/claude-sonnet-4-5",
                cache_control_injection_points=[],
            )

        assert call.call_args.kwargs["model"] == "anthropic/claude-sonnet-4-5"
        assert REASONING not in call.call_args.kwargs["input"]
        assert "include" not in call.call_args.kwargs
    finally:
        client.close()


def test_completion_model_override_uses_effective_replay_scope() -> None:
    client = CompletionClient(
        model="openai/gpt-5.6",
        api_key="account-a",
        cache_control_injection_points=[],
    )
    try:
        with patch("litellm.completion", return_value=_chat_response(reasoning_items=[REASONING])):
            first = client.call([{"role": "user", "content": "run"}], tools=[TOOL])

        with patch("litellm.completion", return_value=_chat_response()) as call:
            client.call(
                _render_chat(first),
                tools=[TOOL],
                model="anthropic/claude-sonnet-4-5",
            )

        assert call.call_args.kwargs["model"] == "anthropic/claude-sonnet-4-5"
        assert "provider-secret" not in repr(call.call_args.kwargs["messages"])
    finally:
        client.close()


@pytest.mark.asyncio
async def test_async_clients_use_effective_model_for_replay_scope() -> None:
    responses = ResponsesClient(model="openai/gpt-5.6", api_key="account-a")
    completion = CompletionClient(
        model="openai/gpt-5.6",
        api_key="account-a",
        cache_control_injection_points=[],
    )
    try:
        with patch("litellm.responses", return_value=_responses(REASONING, MESSAGE)):
            responses_first = responses.call([{"role": "user", "content": "think"}])
        with patch("litellm.completion", return_value=_chat_response(reasoning_items=[REASONING])):
            completion_first = completion.call([{"role": "user", "content": "run"}], tools=[TOOL])

        with (
            patch(
                "litellm.aresponses", AsyncMock(return_value=_responses(MESSAGE))
            ) as response_call,
            patch("litellm.acompletion", AsyncMock(return_value=_chat_response())) as chat_call,
        ):
            await responses.acall(
                _render_responses(responses_first),
                model="anthropic/claude-sonnet-4-5",
                cache_control_injection_points=[],
            )
            await completion.acall(
                _render_chat(completion_first),
                tools=[TOOL],
                model="anthropic/claude-sonnet-4-5",
            )

        assert REASONING not in response_call.call_args.kwargs["input"]
        assert "provider-secret" not in repr(chat_call.call_args.kwargs["messages"])
    finally:
        await responses.aclose()
        await completion.aclose()


def test_azure_responses_state_is_captured_replayed_and_requested() -> None:
    client = ResponsesClient(
        model="azure/gpt-5.6",
        api_key="account-a",
        api_base="https://account-a.openai.azure.com",
    )
    try:
        with patch(
            "litellm.responses",
            side_effect=[_responses(REASONING, MESSAGE), _responses(MESSAGE)],
        ) as call:
            first = client.call([{"role": "user", "content": "think"}])
            client.call(_render_responses(first))

        assert first.replay_scope is not None
        assert first.replay_scope.startswith("responses:azure:")
        assert "reasoning.encrypted_content" in call.call_args_list[0].kwargs["include"]
        assert REASONING in call.call_args_list[1].kwargs["input"]
    finally:
        client.close()


def test_responses_state_replays_across_gateways() -> None:
    source_client = ResponsesClient(
        model="openai/gpt-5.6",
        api_key="account-a",
        api_base="https://gateway-a.example/v1",
    )
    target_client = ResponsesClient(
        model="openai/gpt-5.6",
        api_key="account-b",
        api_base="https://gateway-b.example/v1",
    )
    try:
        with patch("litellm.responses", return_value=_responses(REASONING, MESSAGE)):
            first = source_client.call([{"role": "user", "content": "think"}])
        with patch("litellm.responses", return_value=_responses(MESSAGE)) as target_call:
            target_client.call(_render_responses(first))

        assert REASONING in target_call.call_args.kwargs["input"]
    finally:
        source_client.close()
        target_client.close()


def test_scope_is_stable_across_explicit_endpoints() -> None:
    source = replay_scope(
        "openai/gpt-5.6",
        "responses",
        {"api_key": "account-a", "api_base": "https://gateway-a.example/v1"},
    )
    target = replay_scope(
        "openai/gpt-5.6",
        "responses",
        {"api_key": "account-a", "api_base": "https://gateway-b.example/v1"},
    )

    assert source is not None
    assert target is not None
    assert source == target


def test_scope_is_stable_across_environment_selected_endpoints(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway-a.example/v1")
    first = replay_scope("openai/gpt-5.6", "responses", {"api_key": "account-a"})
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gateway-b.example/v1")
    second = replay_scope("openai/gpt-5.6", "responses", {"api_key": "account-a"})

    assert first == second


def test_scope_is_stable_across_auth_rotation_and_account_metadata() -> None:
    first = replay_scope(
        "openai/gpt-5.6",
        "responses",
        {"api_key": "account-a", "organization": "org-a", "project": "project-a"},
    )
    second = replay_scope(
        "openai/gpt-5.6",
        "responses",
        {"api_key": "account-b", "organization": "org-b", "project": "project-b"},
    )
    without_auth = replay_scope("openai/gpt-5.6", "responses", {})

    assert first is not None
    assert first == second == without_auth
    assert "account-a" not in first


def test_responses_input_kwarg_cannot_bypass_replay_gate() -> None:
    client = ResponsesClient(model="openai/gpt-5.6", api_key="account-a")
    try:
        with pytest.raises(ValueError, match="'input' is managed by UnifiedLLM"):
            client.call(
                [{"role": "user", "content": "continue"}],
                input=[REASONING],
            )
    finally:
        client.close()


@pytest.mark.parametrize(
    ("client_type", "payload_name"),
    [(CompletionClient, "messages"), (ResponsesClient, "input")],
)
def test_constructor_payload_config_cannot_bypass_replay_gate(client_type, payload_name) -> None:
    client = client_type(
        model="openai/gpt-5.6",
        api_key="account-a",
        **{payload_name: [REASONING]},
    )
    try:
        with pytest.raises(ValueError, match=f"'{payload_name}' is managed by UnifiedLLM"):
            client.call([{"role": "user", "content": "continue"}])
    finally:
        client.close()


@pytest.mark.parametrize(
    ("client_type", "payload_name", "nested_field"),
    [
        (CompletionClient, "messages", "messages"),
        (CompletionClient, "messages", "model"),
        (ResponsesClient, "input", "input"),
        (ResponsesClient, "input", "model"),
    ],
)
def test_extra_body_cannot_override_validated_payload_or_model(
    client_type, payload_name, nested_field
) -> None:
    client = client_type(
        model="openai/gpt-5.6",
        api_key="account-a",
        extra_body={nested_field: [REASONING]},
    )
    try:
        with pytest.raises(ValueError, match="extra_body may not override reserved field"):
            client.call([{"role": "user", "content": "continue"}])
    finally:
        client.close()


@pytest.mark.asyncio
async def test_async_responses_input_kwarg_cannot_bypass_replay_gate() -> None:
    client = ResponsesClient(model="openai/gpt-5.6", api_key="account-a")
    try:
        with pytest.raises(ValueError, match="'input' is managed by UnifiedLLM"):
            await client.acall(
                [{"role": "user", "content": "continue"}],
                input=[REASONING],
            )
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_responses_reasoning_uses_per_call_override() -> None:
    client = ResponsesClient(
        model="openai/gpt-5.6",
        api_key="account-a",
        reasoning={"effort": "high"},
    )
    try:
        with patch("litellm.responses", return_value=_responses(MESSAGE)) as sync_call:
            client.call(
                [{"role": "user", "content": "continue"}],
                reasoning={"effort": "low"},
            )
        with patch("litellm.aresponses", AsyncMock(return_value=_responses(MESSAGE))) as async_call:
            await client.acall(
                [{"role": "user", "content": "continue"}],
                reasoning=None,
            )

        assert sync_call.call_args.kwargs["reasoning"] == {"effort": "low"}
        assert async_call.call_args.kwargs["reasoning"] is None
    finally:
        await client.aclose()


def test_custom_endpoint_does_not_assume_encrypted_include_support() -> None:
    client = ResponsesClient(
        model="openai/gpt-5.6",
        api_key="account-a",
        api_base="https://gateway.example/v1",
    )
    try:
        with patch("litellm.responses", return_value=_responses(MESSAGE)) as call:
            client.call([{"role": "user", "content": "hello"}])
        assert "include" not in call.call_args.kwargs
    finally:
        client.close()


@pytest.mark.asyncio
async def test_async_responses_capture_matches_sync() -> None:
    client = ResponsesClient(model="openai/gpt-5.6", api_key="account-a")
    try:
        with patch(
            "litellm.aresponses", AsyncMock(return_value=_responses(REASONING, MESSAGE))
        ) as call:
            response = await client.acall([{"role": "user", "content": "think"}])

        from nooa.unifiedllm.response_parts import project_turn

        assert project_turn(response, response.replay_scope) == [REASONING, MESSAGE]
        assert "reasoning.encrypted_content" in call.call_args.kwargs["include"]
    finally:
        await client.aclose()


def test_mixed_summary_only_turn_drops_all_native_authority(caplog):
    from nooa.unifiedllm.response_parts import capture_parts, project_turn

    scope = replay_scope("openai/gpt-5.6", "responses", {})
    summary = {"type": "reasoning", "summary": [{"type": "summary_text", "text": "why"}]}
    turn = LLMResponse(
        parts=capture_parts([REASONING, summary, MESSAGE], scope), replay_scope=scope
    )
    assert all(part.native is None for part in turn.parts)
    assert project_turn(turn, scope) == [
        {"role": "assistant", "content": "why"},
        {"role": "assistant", "content": "done"},
    ]
    assert "Incomplete native reasoning sequence" in caplog.text


def test_direct_reasoning_dict_cannot_bypass_the_canonical_turn():
    with ResponsesClient(model="openai/gpt-5.6", api_key="test") as client:
        with (
            patch("litellm.responses") as request,
            pytest.raises(ReasoningReplayError, match="canonical LLMResponse"),
        ):
            client.call([REASONING])
        request.assert_not_called()
