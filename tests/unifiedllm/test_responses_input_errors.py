# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Malformed caller messages fail with actionable input errors before transport."""

import pytest

from nooa.unifiedllm import ResponsesClient


@pytest.mark.parametrize("message", [None, 42, "not a message", []])
def test_response_message_must_be_mapping(message):
    with ResponsesClient("openai/gpt-test", api_key="test") as client:
        with pytest.raises(TypeError, match="message.*mapping"):
            client._transform_messages([message])


@pytest.mark.parametrize("role", ["user", "assistant", "tool", "system"])
@pytest.mark.parametrize("block", [None, "not a block", 3])
def test_response_content_blocks_must_be_dicts(role, block):
    with ResponsesClient("openai/gpt-test", api_key="test") as client:
        with pytest.raises(TypeError, match="content.*dict"):
            client._transform_messages([{"role": role, "content": [block], "tool_call_id": "c"}])


@pytest.mark.parametrize("call_id", [None, 42])
def test_tool_result_requires_string_call_id(call_id):
    message = {"role": "tool", "content": "done"}
    if call_id is not None:
        message["tool_call_id"] = call_id
    with ResponsesClient("openai/gpt-test", api_key="test") as client:
        with pytest.raises(ValueError, match="tool_call_id"):
            client._transform_messages([message])
