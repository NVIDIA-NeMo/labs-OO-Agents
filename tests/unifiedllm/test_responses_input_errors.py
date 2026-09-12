# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Malformed caller messages fail with actionable input errors before transport."""

from copy import deepcopy

import pytest

from nooa.unifiedllm import ResponsesClient


@pytest.mark.parametrize("kind", ["text", "input_text"])
def test_leading_system_text_blocks_become_instructions_without_reordering(kind):
    messages = [
        {"role": "system", "content": [{"type": kind, "text": "A"}, {"type": kind, "text": "B"}]},
        {"role": "system", "content": "C"},
        {"role": "user", "content": "question"},
        {"role": "system", "content": [{"type": kind, "text": "live state"}]},
    ]
    original = deepcopy(messages)
    with ResponsesClient("openai/gpt-test", api_key="test") as client:
        wire, instructions = client._transform_messages(messages)
    assert instructions == "AB\n\nC"
    assert wire == [
        {"role": "user", "content": "question"},
        {"role": "system", "content": [{"type": "input_text", "text": "live state"}]},
    ]
    assert messages == original


@pytest.mark.parametrize(
    "block",
    [
        {"type": "input_image", "image_url": "https://example.test/image"},
        {"type": "text"},
        {"type": "text", "text": None},
        {"type": "text", "text": 42},
    ],
)
def test_leading_system_rejects_unsupported_blocks_clearly(block):
    with ResponsesClient("openai/gpt-test", api_key="test") as client:
        with pytest.raises(ValueError, match="Leading system.*text"):
            client._transform_messages([{"role": "system", "content": [block]}])


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
