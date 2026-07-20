# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Simple integration tests for ResponsesClient using NVIDIA inference API endpoint.

Requires NVIDIA_INFERENCE_API_KEY environment variable to be set.

These tests are marked as integration tests and skipped by default in CI.
Run with: pytest -m integration
"""

import os

import pytest
from pydantic import BaseModel

from nooa.unifiedllm import LLMResponse, ResponsesClient, create_tool_from_callable

# Mark all tests in this module as integration tests (skipped by default in CI)
pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    """Create a ResponsesClient with NVIDIA gpt-oss-20b model."""
    api_key = os.getenv("NVIDIA_INFERENCE_API_KEY")
    if not api_key:
        pytest.skip("NVIDIA_INFERENCE_API_KEY environment variable not set")

    return ResponsesClient(
        model="openai/nvidia/openai/gpt-oss-20b",
        base_url="https://inference-api.nvidia.com/v1/",
        api_key=api_key,
    )


class SimpleResponse(BaseModel):
    """Simple structured output model for testing."""

    answer: str
    confidence: float


def add_numbers(a: int, b: int) -> int:
    """Add two numbers together and return the result."""
    return a + b


class TestResponsesClientBasic:
    """Basic tests for ResponsesClient functionality."""

    def test_simple_completion(self, client):
        """Test basic text completion without structured output or tools."""
        messages = [{"role": "user", "content": "What is 2 + 2? Answer with just the number."}]

        response = client.call(
            messages=messages,
            tools=[],
            output_model=None,
        )

        assert isinstance(response, LLMResponse)
        assert response.content is not None and isinstance(response.content, str)
        assert "4" in response.content
        assert response.finish_reason == "stop"
        assert response.tool_calls == []

    def test_structured_output(self, client):
        """Test structured output parsing with Pydantic model."""
        messages = [
            {
                "role": "user",
                "content": "What is the capital of France? "
                "Provide your answer and confidence as a number between 0 and 1.",
            }
        ]

        response = client.call(
            messages=messages,
            tools=[],
            output_model=SimpleResponse,
        )

        assert isinstance(response, LLMResponse)
        assert isinstance(response.content, SimpleResponse)
        assert "Paris" in response.content.answer or "paris" in response.content.answer.lower()
        assert 0 <= response.content.confidence <= 1

    def test_tool_calling(self, client):
        """Test that the model can call tools."""
        add_tool = create_tool_from_callable(add_numbers)

        messages = [{"role": "user", "content": "Please add 17 and 25 using the add_numbers tool."}]

        response = client.call(
            messages=messages,
            tools=[add_tool],
            output_model=None,
        )

        assert isinstance(response, LLMResponse)
        assert response.finish_reason == "tool_calls"
        assert len(response.tool_calls) > 0

        tool_call = response.tool_calls[0]
        assert tool_call.name == "add_numbers"
        # The arguments should contain 17 and 25
        assert "17" in tool_call.arguments
        assert "25" in tool_call.arguments


class TestResponsesClientAsync:
    """Async tests for ResponsesClient."""

    @pytest.mark.asyncio
    async def test_async_simple_completion(self, client):
        """Test async text completion."""
        messages = [{"role": "user", "content": "Say hello in one word."}]

        response = await client.acall(
            messages=messages,
            tools=[],
            output_model=None,
        )

        assert isinstance(response, LLMResponse)
        assert response.content is not None and isinstance(response.content, str)
        assert len(response.content) > 0
        assert response.finish_reason == "stop"

    @pytest.mark.asyncio
    @pytest.mark.flaky(reruns=2, reason="Live LLM API may return malformed structured output")
    async def test_async_structured_output(self, client):
        """Test async structured output parsing."""
        messages = [
            {"role": "user", "content": "What is 10 * 10? Provide your answer and confidence."}
        ]

        response = await client.acall(
            messages=messages,
            tools=[],
            output_model=SimpleResponse,
        )

        assert isinstance(response, LLMResponse)
        assert isinstance(response.content, SimpleResponse)
        # Check that we got a valid structured response with non-empty answer
        # (LLM responses can be flaky, so we just verify structure, not exact content)
        assert response.content.answer is not None
        assert len(response.content.answer) > 0
