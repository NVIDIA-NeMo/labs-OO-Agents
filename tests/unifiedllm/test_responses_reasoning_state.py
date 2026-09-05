# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Responses API reasoning replay regressions."""

from unittest.mock import AsyncMock, patch

import pytest
from litellm.types.llms.openai import ResponsesAPIResponse

from nooa.unifiedllm import ResponsesClient, Tool

REASONING_ITEM = {
    "id": "rs_123",
    "type": "reasoning",
    "encrypted_content": "encrypted-state",
    "summary": [],
}
FUNCTION_CALL_ITEM = {
    "id": "fc_123",
    "type": "function_call",
    "call_id": "call_123",
    "name": "execute_python",
    "arguments": '{"code":"print(1)"}',
    "status": "completed",
}


def _execute_python(code: str) -> str:
    return code


TOOL = Tool(name="execute_python", description="Execute Python.", callable=_execute_python)


def _response(*output: dict) -> ResponsesAPIResponse:
    return ResponsesAPIResponse(
        id="resp_123",
        created_at=0,
        model="gpt-5.6",
        object="response",
        status="completed",
        output=list(output),
    )


def test_tool_response_retains_and_replays_complete_native_output_batch() -> None:
    client = ResponsesClient(model="openai/gpt-5.6", api_key="test")
    raw = _response(REASONING_ITEM, FUNCTION_CALL_ITEM)
    try:
        with patch("litellm.responses", side_effect=[raw, raw]) as responses:
            first = client.call(messages=[{"role": "user", "content": "Run Python."}], tools=[TOOL])
            client.call(
                messages=[
                    {"role": "user", "content": "Run Python."},
                    first.assistant_message,
                    {"role": "tool", "tool_call_id": "call_123", "content": "1"},
                ],
                tools=[TOOL],
            )

        assert first.assistant_message["reasoning_items"] == [REASONING_ITEM]
        assert first.assistant_message["_batch"] == [REASONING_ITEM, FUNCTION_CALL_ITEM]
        second_input = responses.call_args_list[1].kwargs["input"]
        assert second_input[1:3] == [REASONING_ITEM, FUNCTION_CALL_ITEM]
        assert second_input[3] == {
            "type": "function_call_output",
            "call_id": "call_123",
            "output": "1",
        }
    finally:
        client.close()


@pytest.mark.asyncio
async def test_async_text_response_retains_reasoning() -> None:
    message_item = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "done", "annotations": []}],
    }
    client = ResponsesClient(model="openai/gpt-5.6", api_key="test")
    try:
        mock = AsyncMock(return_value=_response(REASONING_ITEM, message_item))
        with patch("litellm.aresponses", mock):
            result = await client.acall(messages=[{"role": "user", "content": "Finish."}])

        assert result.content == "done"
        assert result.assistant_message["reasoning_items"] == [REASONING_ITEM]
        assert result.assistant_message["_batch"] == [REASONING_ITEM, message_item]
    finally:
        await client.aclose()
