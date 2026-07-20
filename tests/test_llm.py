# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for LLM clients."""

import json

import pytest

from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


@pytest.mark.asyncio
async def test_fake_llm_simple_message():
    """Test FakeLLMClient with simple message."""
    client = FakeLLMClient.simple_message("Hello world")

    response = await client.acall(messages=[{"role": "user", "content": "test"}])

    assert response.content == "Hello world"
    assert response.reasoning is None
    assert len(response.tool_calls) == 0
    assert client.call_count == 1


@pytest.mark.asyncio
async def test_fake_llm_tool_call():
    """Test FakeLLMClient with tool call."""
    client = FakeLLMClient.with_tool_call(
        tool_name="search",
        tool_args={"query": "test"},
    )

    response = await client.acall(messages=[{"role": "user", "content": "search"}])

    assert len(response.tool_calls) == 1
    tool_call = response.tool_calls[0]
    assert tool_call.name == "search"
    # Note: unifiedllm.ToolCall.arguments is a JSON string
    assert json.loads(tool_call.arguments) == {"query": "test"}


@pytest.mark.asyncio
async def test_fake_llm_with_reasoning():
    """Test FakeLLMClient with reasoning."""
    client = FakeLLMClient.with_reasoning(
        reasoning="Let me think about this...",
        message="Here is my answer",
    )

    response = await client.acall(messages=[{"role": "user", "content": "question"}])

    assert response.reasoning == "Let me think about this..."
    assert response.content == "Here is my answer"
    assert "think" in response.reasoning.lower()


@pytest.mark.asyncio
async def test_fake_llm_captures_calls():
    """Test that FakeLLMClient captures call arguments."""
    client = FakeLLMClient.simple_message("test")

    messages = [{"role": "user", "content": "hello"}]
    tools = [{"type": "function", "function": {"name": "test_tool"}}]

    await client.acall(messages=messages, tools=tools)

    assert client.call_count == 1
    assert client.last_messages == messages
    assert client.last_tools == tools


@pytest.mark.asyncio
async def test_fake_llm_reset():
    """Test FakeLLMClient reset."""
    client = FakeLLMClient.simple_message("test")

    await client.acall(messages=[{"role": "user", "content": "test"}])

    assert client.call_count == 1

    client.reset()
    assert client.call_count == 0
    assert client.last_messages == []


@pytest.mark.asyncio
async def test_fake_llm_custom_responses():
    """Test FakeLLMClient with custom responses."""
    custom_responses = [
        LLMResponse(
            raw_response=None,
            content="First response",
            tool_calls=[],
            finish_reason="stop",
            assistant_message={"role": "assistant", "content": "First response"},
        ),
        LLMResponse(
            raw_response=None,
            content="Second response",
            tool_calls=[],
            finish_reason="stop",
            assistant_message={"role": "assistant", "content": "Second response"},
        ),
    ]

    client = FakeLLMClient(scripted_responses=custom_responses)

    response1 = await client.acall(messages=[])
    assert response1.content == "First response"

    response2 = await client.acall(messages=[])
    assert response2.content == "Second response"

    # Third call should return empty response
    response3 = await client.acall(messages=[])
    assert response3.content == ""


@pytest.mark.asyncio
async def test_fake_llm_multiple_calls():
    """Test FakeLLMClient with multiple scripted responses."""
    client = FakeLLMClient(
        scripted_responses=[
            LLMResponse(
                raw_response=None,
                content="Response 1",
                tool_calls=[],
                finish_reason="stop",
                assistant_message={"role": "assistant", "content": "Response 1"},
            ),
            LLMResponse(
                raw_response=None,
                content="Response 2",
                tool_calls=[],
                finish_reason="stop",
                assistant_message={"role": "assistant", "content": "Response 2"},
            ),
            LLMResponse(
                raw_response=None,
                content="Response 3",
                tool_calls=[ToolCall(id="1", name="test_tool", arguments=json.dumps({}))],
                finish_reason="tool_calls",
                assistant_message={"role": "assistant", "content": "Response 3"},
            ),
        ]
    )

    r1 = await client.acall(messages=[])
    assert r1.content == "Response 1"

    r2 = await client.acall(messages=[])
    assert r2.content == "Response 2"

    r3 = await client.acall(messages=[])
    assert r3.content == "Response 3"
    assert len(r3.tool_calls) == 1
    assert r3.tool_calls[0].name == "test_tool"
