# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Performance tests to validate client reuse benefits."""

import logging
import time

import pytest

from nooa import Agent, strategy
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse

logger = logging.getLogger(__name__)


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


class PerformanceAgent(Agent, llm=_TEST_LLM):
    """Agent for performance testing."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @strategy(PurePythonStrategy())
    async def task(self):
        """Execute task."""
        ...


@pytest.mark.asyncio
async def test_no_overhead_from_repeated_calls():
    """Test that repeated calls reuse the same client — no per-call recreation."""

    # Create 100 fake responses - REPL-style
    responses = [_resp("pass") for i in range(100)]
    fake_llm = FakeLLMClient(responses)

    agent_instance = PerformanceAgent(llm=fake_llm)

    # Verify client is set once at init
    initial_client = agent_instance._llm
    assert initial_client is fake_llm

    # Run 100 calls, asserting client identity is preserved on every call.
    # This is the deterministic check: if the client were recreated per call,
    # `agent._llm is initial_client` would fail.
    start = time.time()
    for _i in range(100):
        await agent_instance.task()
        assert agent_instance._llm is initial_client
    elapsed = time.time() - start

    # Log timing for visibility, but don't assert on it — wall-clock thresholds
    # are inherently flaky on shared CI runners.
    logger.debug(
        "100 calls completed in %.3fs (avg %.2fms per call)", elapsed, elapsed / 100 * 1000
    )

    # Verify all calls actually happened
    assert fake_llm.call_count == 100, "All 100 calls should have been made"


@pytest.mark.asyncio
async def test_client_creation_is_one_time_cost():
    """Test that client creation happens only once (in __init__)."""
    fake_llm = FakeLLMClient([])  # Provide empty client to avoid real client creation

    # Time agent creation (with injected client, so measures framework overhead only)
    start = time.time()
    agent_instance = PerformanceAgent(llm=fake_llm)
    init_time = time.time() - start

    logger.debug("Agent creation time (with injected client): %.2fms", init_time * 1000)

    # Client should exist and be the injected one
    assert agent_instance._llm is fake_llm

    # The init time should be reasonable (< 100ms typically)
    assert init_time < 1.0, "Agent creation should be fast"


@pytest.mark.asyncio
async def test_concurrent_calls_share_client():
    """Test that concurrent method calls share the same client instance."""
    import asyncio

    class ConcurrentAgent(Agent, llm=_TEST_LLM):
        """Agent for concurrent testing."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        @strategy(
            PurePythonStrategy(),
        )
        async def task1(self):
            """Task 1."""
            ...

        @strategy(
            PurePythonStrategy(),
        )
        async def task2(self):
            """Task 2."""
            ...

        @strategy(
            PurePythonStrategy(),
        )
        async def task3(self):
            """Task 3."""
            ...

    # Create responses for all tasks
    fake_llm = FakeLLMClient([_resp("pass"), _resp("pass"), _resp("pass")])

    agent_instance = ConcurrentAgent(llm=fake_llm)
    initial_client = agent_instance._llm

    # Launch tasks concurrently (though generation is serialized, client is shared)
    results = await asyncio.gather(
        agent_instance.task1(),
        agent_instance.task2(),
        agent_instance.task3(),
    )

    # Verify client is still the same
    assert agent_instance._llm is initial_client

    # All three tasks should have completed
    assert len(results) == 3
    assert fake_llm.call_count == 3
