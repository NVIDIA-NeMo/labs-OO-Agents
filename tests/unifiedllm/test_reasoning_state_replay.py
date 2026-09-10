# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Issuer-gated capture and replay of opaque OpenAI reasoning state."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from litellm.types.utils import ModelResponse

from nooa._llm_state import (
    LLM_STATE_KEY,
    ReplayCarryingMessage,
    carried_replay_batch,
    carried_state,
)
from nooa.context_blocks.events import ToolCallEvent, ToolResult
from nooa.context_blocks.formatter import (
    OpenAIProviderFormatter,
    ResponsesProviderFormatter,
    XMLBlockFormatter,
)
from nooa.context_blocks.models import ResolvedBlock, Role
from nooa.runtime.middleware import LLMCallContext
from nooa.unifiedllm import CompletionClient, LLMResponse, ResponsesClient, Tool
from nooa.unifiedllm.replay_state import replay_scope

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


def _chat_response(*, reasoning_items: list[dict] | None = None) -> ModelResponse:
    return ModelResponse(
        model="gpt-5.6",
        choices=[
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "execute_python",
                                "arguments": '{"code":"print(1)"}',
                            },
                        }
                    ],
                    "reasoning_items": reasoning_items,
                },
            }
        ],
    )


def _tool(code: str) -> str:
    return code


TOOL = Tool(name="execute_python", description="Run code", callable=_tool)


def _response_blocks(response: LLMResponse) -> list[ResolvedBlock]:
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
    return blocks


def _render_responses(response: LLMResponse) -> list[dict]:
    neutral = XMLBlockFormatter().format(_response_blocks(response))
    return ResponsesProviderFormatter().format(neutral)


def _render_chat(response: LLMResponse) -> list[dict]:
    neutral = XMLBlockFormatter().format(_response_blocks(response))
    return OpenAIProviderFormatter().format(neutral)


def test_responses_text_state_is_captured_and_exactly_replayed() -> None:
    client = ResponsesClient(model="openai/gpt-5.6", api_key="account-a")
    try:
        with patch(
            "litellm.responses",
            side_effect=[_responses(REASONING, MESSAGE), _responses(MESSAGE)],
        ) as call:
            first = client.call([{"role": "user", "content": "think"}])
            rendered = _render_responses(first)
            assert "provider-secret" not in json.dumps(rendered)
            middleware_context = LLMCallContext(messages=rendered)
            carrier = next(
                message
                for message in middleware_context.messages
                if carried_state(message) is not None
            )
            assert carried_state(carrier) == first.llm_state
            client.call(middleware_context.messages + [{"role": "user", "content": "continue"}])

        assert first.llm_state is not None
        assert first.llm_state["payload"]["items"] == [REASONING]
        assert "reasoning.encrypted_content" in call.call_args_list[0].kwargs["include"]
        replay = call.call_args_list[1].kwargs["input"]
        assert replay[:2] == [REASONING, {"role": "assistant", "content": "done"}]
        assert LLM_STATE_KEY not in repr(replay)
    finally:
        client.close()


def test_responses_multi_call_state_preserves_provider_order() -> None:
    first_raw = _responses(REASONING, CALL, REASONING_2, CALL_2)
    client = ResponsesClient(model="openai/gpt-5.6", api_key="account-a")
    try:
        with patch("litellm.responses", side_effect=[first_raw, _responses(MESSAGE)]) as call:
            first = client.call([{"role": "user", "content": "run"}], tools=[TOOL])
            client.call(
                _render_responses(first) + [{"role": "user", "content": "continue"}],
                tools=[TOOL],
            )

        replay = call.call_args_list[1].kwargs["input"]
        assistant_batch = [item for item in replay if item.get("type") != "function_call_output"]
        assert [item.get("type") for item in assistant_batch[:4]] == [
            "reasoning",
            "function_call",
            "reasoning",
            "function_call",
        ]
        assert [
            item.get("call_id") for item in assistant_batch if item.get("type") == "function_call"
        ] == [
            "call_1",
            "call_2",
        ]
    finally:
        client.close()


def test_split_responses_replay_batch_keeps_public_call_but_drops_state() -> None:
    """Middleware may edit public items, but cannot reattach state to new neighbors."""
    client = ResponsesClient(model="openai/gpt-5.6", api_key="account-a")
    try:
        with patch("litellm.responses", return_value=_responses(REASONING, CALL, CALL_2)):
            first = client.call([{"role": "user", "content": "run"}], tools=[TOOL])
        rendered = [item for item in _render_responses(first) if carried_replay_batch(item)]
        assert len(rendered) == 2

        with patch("litellm.responses", return_value=_responses(MESSAGE)) as call:
            client.call(rendered[:1], tools=[TOOL])

        replay = call.call_args.kwargs["input"]
        assert [item.get("call_id") for item in replay] == ["call_1"]
        assert REASONING not in replay
        assert "provider-secret" not in repr(replay)
    finally:
        client.close()


def test_reasoning_only_response_replays_without_empty_assistant_message() -> None:
    client = ResponsesClient(model="openai/gpt-5.6", api_key="account-a")
    try:
        with patch(
            "litellm.responses",
            side_effect=[_responses(REASONING), _responses(MESSAGE)],
        ) as call:
            first = client.call([{"role": "user", "content": "think"}])
            rendered = _render_responses(first)
            client.call(rendered + [{"role": "user", "content": "continue"}])

        assert first.content == ""
        assert first.llm_state is not None
        assert first.llm_state["payload"]["state_only"] is True
        replay = call.call_args_list[1].kwargs["input"]
        assert REASONING in replay
        assert {"role": "assistant", "content": ""} not in replay
    finally:
        client.close()


def test_incompatible_reasoning_only_state_drops_its_internal_carrier() -> None:
    source = ResponsesClient(model="openai/gpt-5.6", api_key="account-a")
    target = ResponsesClient(model="openai/gpt-5.7", api_key="account-a")
    try:
        with patch("litellm.responses", return_value=_responses(REASONING)):
            first = source.call([{"role": "user", "content": "think"}])
        with patch("litellm.responses", return_value=_responses(MESSAGE)) as call:
            target.call(_render_responses(first) + [{"role": "user", "content": "continue"}])

        replay = call.call_args.kwargs["input"]
        assert REASONING not in replay
        assert {"role": "assistant", "content": ""} not in replay
        assert replay == [{"role": "user", "content": "continue"}]
    finally:
        source.close()
        target.close()


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

        assert first.llm_state is not None
        assert first.llm_state["scope"].startswith("responses:azure:")
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


def test_chat_state_is_captured_replayed_and_api_style_scoped() -> None:
    client = CompletionClient(
        model="openai/gpt-5.6",
        api_key="account-a",
        cache_control_injection_points=[],
    )
    try:
        with patch(
            "litellm.completion",
            side_effect=[_chat_response(reasoning_items=[REASONING]), _chat_response()],
        ) as call:
            first = client.call([{"role": "user", "content": "run"}], tools=[TOOL])
            client.call(_render_chat(first), tools=[TOOL])

        assert first.llm_state is not None
        assert first.llm_state["format"] == "litellm-chat"
        assistant = next(
            item for item in call.call_args_list[1].kwargs["messages"] if item.get("tool_calls")
        )
        assert assistant["reasoning_items"] == [REASONING]

        responses = ResponsesClient(model="openai/gpt-5.6", api_key="account-a")
        try:
            with patch("litellm.responses", return_value=_responses(MESSAGE)) as target_call:
                responses.call(_render_chat(first))
            assert REASONING not in target_call.call_args.kwargs["input"]
        finally:
            responses.close()
    finally:
        client.close()


def test_unresolved_route_drops_state_at_capture() -> None:
    client = ResponsesClient(model="unknown-route", api_key="account-a")
    try:
        with (
            patch("litellm.get_llm_provider", side_effect=ValueError("unknown")),
            patch("litellm.responses", return_value=_responses(REASONING, MESSAGE)) as call,
        ):
            response = client.call([{"role": "user", "content": "think"}])

        assert response.llm_state is None
        assert "include" not in call.call_args.kwargs
    finally:
        client.close()


def test_direct_reasoning_items_cannot_bypass_envelope_gate() -> None:
    client = ResponsesClient(model="openai/gpt-5.6", api_key="account-a")
    try:
        with patch("litellm.responses", return_value=_responses(MESSAGE)) as call:
            client.call([REASONING, {"role": "user", "content": "continue"}])

        assert REASONING not in call.call_args.kwargs["input"]
        assert "provider-secret" not in repr(call.call_args.kwargs["input"])
    finally:
        client.close()


@pytest.mark.parametrize("model", ["anthropic/claude-sonnet-4-5", "gemini/gemini-2.5-pro"])
def test_non_openai_chat_provider_cannot_receive_reasoning_state(model: str) -> None:
    assert replay_scope(model, "chat", {"api_key": "account-a"}) is None
    client = CompletionClient(model=model, api_key="account-a")
    crafted = {
        "version": 1,
        "scope": f"chat:{model.split('/', 1)[0]}:crafted",
        "format": "litellm-chat",
        "payload": {"reasoning_items": [REASONING]},
    }
    try:
        with patch("litellm.completion", return_value=_chat_response()) as call:
            client.call(
                [ReplayCarryingMessage({"role": "assistant", "content": "public"}, crafted)]
            )

        assert call.call_args.kwargs["messages"] == [{"role": "assistant", "content": "public"}]
        assert "provider-secret" not in repr(call.call_args.kwargs)
    finally:
        client.close()


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

        assert response.llm_state is not None
        assert response.llm_state["payload"]["items"] == [REASONING]
        assert "reasoning.encrypted_content" in call.call_args.kwargs["include"]
    finally:
        await client.aclose()
