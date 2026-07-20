# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test PurePythonStrategy with nested @strategy methods using PredictStrategy.

This tests the scenario where an LLM-generated method defines and calls a helper
method decorated with @strategy(PredictStrategy()) in the same turn.

"""

import pytest

from nooa import Agent, strategy
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse


@pytest.mark.asyncio
async def test_pure_python_nested_structured_output_same_turn():
    """Test that PurePython can define and call a @strategy method with PredictStrategy in same turn.

    Flow:
    1. Parent method uses PurePythonStrategy
    2. LLM generates code that defines async helper with @strategy(PredictStrategy())
    3. Same code calls the helper method
    4. Helper method generates with PredictStrategy (returns {"value": "result"})
    5. Parent method completes successfully
    """
    # Create scripted responses:
    # Response 1: Parent PurePython generates code with nested PredictStrategy helper
    # Response 2: Nested PredictStrategy returns JSON with "value" field
    responses = [
        LLMResponse(
            raw_response=None,
            content='''@strategy(PredictStrategy())


async def _summarize_doc(self, doc: str) -> str:
    """Summarize a single document."""
    ...

summaries = []
for doc in documents:
    summary = await _summarize_doc(self, doc)
    summaries.append(summary)
return summaries''',
            tool_calls=[],
            finish_reason="stop",
            assistant_message={"role": "assistant", "content": "code"},
        ),
        # PredictStrategy for basic types wraps in {"value": ...}
        LLMResponse(
            raw_response=None,
            content='{"value": "Document summary"}',
            tool_calls=[],
            finish_reason="stop",
            assistant_message={"role": "assistant", "content": '{"value": "Document summary"}'},
        ),
    ]

    llm = FakeLLMClient(scripted_responses=responses)

    class SummarizeBatchAgent(Agent, llm=llm):
        @strategy(PurePythonStrategy())
        async def summarize_batch(self, documents: list[str]) -> list[str]:
            """Summarize each document in the batch."""
            ...

    # Execute - should work without 'object has no attribute' error
    agent_instance = SummarizeBatchAgent()
    result = await agent_instance.summarize_batch(["doc1"])

    # Verify
    assert result == ["Document summary"]
    assert llm.call_count == 2  # One for parent, one for nested helper
    # helpers are plain callables, never attached to the agent.
    assert not hasattr(agent_instance, "_summarize_doc")


@pytest.mark.asyncio
async def test_pure_python_nested_structured_output_multiple_calls():
    """Test nested PredictStrategy helper called multiple times in a loop."""
    # Parent generates once, nested helper called 3 times
    responses = [
        LLMResponse(
            raw_response=None,
            content='''@strategy(PredictStrategy())


async def _process_item(self, item: str) -> str:
    """Process single item."""
    ...

results = []
for item in items:
    result = await _process_item(self, item)
    results.append(result)
return results''',
            tool_calls=[],
            finish_reason="stop",
            assistant_message={"role": "assistant", "content": "code"},
        ),
        # Three PredictStrategy responses for the three loop iterations
        LLMResponse(
            raw_response=None,
            content='{"value": "processed_a"}',
            tool_calls=[],
            finish_reason="stop",
            assistant_message={"role": "assistant", "content": '{"value": "processed_a"}'},
        ),
        LLMResponse(
            raw_response=None,
            content='{"value": "processed_b"}',
            tool_calls=[],
            finish_reason="stop",
            assistant_message={"role": "assistant", "content": '{"value": "processed_b"}'},
        ),
        LLMResponse(
            raw_response=None,
            content='{"value": "processed_c"}',
            tool_calls=[],
            finish_reason="stop",
            assistant_message={"role": "assistant", "content": '{"value": "processed_c"}'},
        ),
    ]

    llm = FakeLLMClient(scripted_responses=responses)

    class BatchProcessor(Agent, llm=llm):
        @strategy(PurePythonStrategy())
        async def process_batch(self, items: list[str]) -> list[str]:
            """Process all items."""
            ...

    agent_instance = BatchProcessor()
    result = await agent_instance.process_batch(["a", "b", "c"])

    assert result == ["processed_a", "processed_b", "processed_c"]
    assert llm.call_count == 4  # 1 parent + 3 nested calls
