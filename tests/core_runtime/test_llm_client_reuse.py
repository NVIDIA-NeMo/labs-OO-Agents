# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for LLM client reuse across method calls."""

import pytest

from nooa import Agent, strategy
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse


def _resp(content: str) -> LLMResponse:
    """Create a test LLM response with the given content."""
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": content},
    )


# Module-level test LLM (can be overridden at instantiation)
_TEST_LLM = FakeLLMClient()


class SimpleAgent(Agent, llm=_TEST_LLM):
    """Agent for testing client reuse."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @strategy(PurePythonStrategy())
    async def method1(self):
        """First method."""
        ...

    @strategy(PurePythonStrategy())
    async def method2(self):
        """Second method."""
        ...


@pytest.mark.asyncio
async def test_llm_client_created_once_in_init():
    """Test that LLM client is created once during __init__."""

    # Create agent (client should be created in __init__)
    agent_instance = SimpleAgent()

    # Get initial client reference
    client1 = agent_instance._llm
    assert client1 is not None, "Client should be created in __init__"

    # Client should be the same instance after multiple accesses
    client2 = agent_instance._llm
    assert client1 is client2, "Client reference should be stable"


@pytest.mark.asyncio
async def test_same_client_used_across_multiple_calls():
    """Test that the same client instance is used across multiple method calls."""
    fake_llm = FakeLLMClient([_resp("pass"), _resp("pass")])

    # Create agent with injected client
    agent_instance = SimpleAgent(llm=fake_llm)

    # Get initial client reference
    initial_client = agent_instance._llm

    # Call first method
    await agent_instance.method1()
    client_after_call1 = agent_instance._llm

    # Call second method
    await agent_instance.method2()
    client_after_call2 = agent_instance._llm

    # All should be the same instance
    assert initial_client is client_after_call1, "Client should not change after first call"
    assert initial_client is client_after_call2, "Client should not change after second call"
    assert initial_client is fake_llm, "Client should be the injected fake client"


@pytest.mark.asyncio
async def test_injected_client_overrides_default():
    """Test that injected client overrides default creation."""
    fake_llm = FakeLLMClient([_resp("pass")])

    # Create agent with injected client
    agent_instance = SimpleAgent(llm=fake_llm)

    # Should use injected client, not create default
    assert agent_instance._llm is fake_llm


@pytest.mark.asyncio
async def test_no_client_creation_during_execution():
    """Test that _execute_with_generation doesn't create new clients."""
    fake_llm = FakeLLMClient([_resp("pass"), _resp("pass"), _resp("pass")])

    agent_instance = SimpleAgent(llm=fake_llm)
    initial_client = agent_instance._llm

    # Track client identity through multiple calls
    await agent_instance.method1()
    assert agent_instance._llm is initial_client

    await agent_instance.method2()
    assert agent_instance._llm is initial_client

    await agent_instance.method1()  # Call again
    assert agent_instance._llm is initial_client


# Deleted: test_client_reuse_with_persistent_lifetime
# PERSISTENT lifetime caching was removed - all generation methods now regenerate every call
