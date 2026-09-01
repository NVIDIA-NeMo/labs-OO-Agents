# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for LiteLLM's Chat Completions-to-Responses bridge."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import Choices, Message, ModelResponse

from nooa.unifiedllm import MODELS, CompletionClient, Tool, get_llm_client


def _execute_python(code: str) -> str:
    return code


TOOL = Tool(
    name="execute_python",
    description="Execute Python code.",
    callable=_execute_python,
)

REASONING_ITEM = {
    "id": "rs_123",
    "type": "reasoning",
    "encrypted_content": "encrypted-state",
    "summary": [{"type": "summary_text", "text": "Use the tool."}],
}


def _responses_tool_call() -> ResponsesAPIResponse:
    return ResponsesAPIResponse(
        id="resp_123",
        created_at=0,
        model="gpt-5.6",
        object="response",
        status="completed",
        output=[
            REASONING_ITEM,
            {
                "id": "fc_123",
                "type": "function_call",
                "call_id": "call_123",
                "name": "execute_python",
                "arguments": '{"code":"print(1)"}',
                "status": "completed",
            },
        ],
    )


def _chat_response() -> ModelResponse:
    return ModelResponse(
        model="gpt-5.6",
        choices=[
            Choices(
                finish_reason="stop",
                message=Message(content="done", role="assistant"),
            )
        ],
    )


def _client(model: str = "openai/gpt-5.6", **config: object) -> CompletionClient:
    with (
        patch("nooa.unifiedllm.registry.ensure_loaded"),
        patch.dict(MODELS, {}, clear=True),
    ):
        client = get_llm_client(model, api_key="test", **config)
    assert isinstance(client, CompletionClient)
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["gpt-5.6", "openai/gpt-5.6"])
async def test_default_reasoning_tool_call_uses_responses_bridge(model: str) -> None:
    client = _client(model=model)
    try:
        with (
            patch(
                "litellm.aresponses",
                new_callable=AsyncMock,
                return_value=_responses_tool_call(),
            ) as responses,
            patch(
                "litellm.main._complete_custom_openai",
                side_effect=AssertionError("chat endpoint should not be used"),
            ),
        ):
            result = await client.acall(
                messages=[{"role": "user", "content": "Run Python."}],
                tools=[TOOL],
            )

        responses.assert_called_once()
        assert result.finish_reason == "tool_calls"
        assert result.assistant_message["reasoning_items"] == [REASONING_ITEM]
    finally:
        await client.aclose()


def test_reasoning_none_keeps_chat_completions() -> None:
    client = _client(reasoning_effort="none")
    chat_endpoint = MagicMock(return_value=_chat_response())
    try:
        with (
            patch("litellm.responses", side_effect=AssertionError("responses should not be used")),
            patch("litellm.main._complete_custom_openai", chat_endpoint),
        ):
            result = client.call(
                messages=[{"role": "user", "content": "Run Python."}],
                tools=[TOOL],
            )

        chat_endpoint.assert_called_once()
        assert result.content == "done"
    finally:
        client.close()


def test_custom_api_base_keeps_chat_completions_by_default() -> None:
    client = _client(api_base="https://gateway.example/v1")
    chat_endpoint = MagicMock(return_value=_chat_response())
    try:
        with (
            patch("litellm.responses", side_effect=AssertionError("responses should not be used")),
            patch("litellm.main._complete_custom_openai", chat_endpoint),
        ):
            result = client.call(
                messages=[{"role": "user", "content": "Run Python."}],
                tools=[TOOL],
            )

        chat_endpoint.assert_called_once()
        assert result.content == "done"
    finally:
        client.close()


def test_reasoning_items_round_trip_into_next_responses_request() -> None:
    client = _client()
    try:
        with (
            patch(
                "litellm.responses",
                side_effect=[_responses_tool_call(), _responses_tool_call()],
            ) as responses,
            patch(
                "litellm.main._complete_custom_openai",
                side_effect=AssertionError("chat endpoint should not be used"),
            ),
        ):
            first = client.call(
                messages=[{"role": "user", "content": "Run Python."}],
                tools=[TOOL],
            )
            client.call(
                messages=[
                    {"role": "user", "content": "Run Python."},
                    first.assistant_message,
                    {
                        "role": "tool",
                        "tool_call_id": first.tool_calls[0].id,
                        "content": "1",
                    },
                ],
                tools=[TOOL],
            )

        second_input = responses.call_args_list[1].kwargs["input"]
        reasoning_index = second_input.index(REASONING_ITEM)
        function_call_index = next(
            index for index, item in enumerate(second_input) if item.get("type") == "function_call"
        )
        output_index = next(
            index
            for index, item in enumerate(second_input)
            if item.get("type") == "function_call_output"
        )
        assert reasoning_index < function_call_index < output_index
    finally:
        client.close()
