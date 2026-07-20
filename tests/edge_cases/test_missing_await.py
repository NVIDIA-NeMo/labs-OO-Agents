# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for detecting missing await on async method calls.

This test suite verifies that the system can detect when LLM-generated code
calls async methods without await, which would return coroutine objects
instead of actual results.
"""

import pytest

from nooa import Agent, strategy
from nooa.errors import GenerationError
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


@pytest.mark.asyncio
async def test_missing_await_on_async_method():
    """Test that missing await on async method is detected and reported."""

    # Buggy code that will be generated (missing await)
    buggy_code = """
sentiments = []
for text in texts:
    sentiment = self.classify_single(text)  # BUG: Missing await!
    sentiments.append(sentiment)
return sentiments
"""

    # Create test LLM with scripted responses
    test_llm = FakeLLMClient(
        scripted_responses=[_resp(buggy_code)] * 4  # Queue enough to exceed max_retries
    )

    class SentimentAgent(Agent, llm=test_llm):
        """Agent for sentiment classification tasks."""

        async def classify_single(self, text: str) -> str:
            """Classify the sentiment of this single text."""
            # Implemented method (not generated)
            if "good" in text.lower() or "great" in text.lower():
                return "positive"
            elif "bad" in text.lower() or "terrible" in text.lower():
                return "negative"
            return "neutral"

        async def classify_batch(self, texts: list[str]) -> list[str]:
            """Classify the sentiment of each text in the batch."""
            ...  # LLM generates code

    agent_instance = SentimentAgent(llm=test_llm)

    # Should raise GenerationError due to validation failure
    with pytest.raises(GenerationError) as exc_info:
        await agent_instance.classify_batch(
            ["This is great!", "This is terrible!", "This is okay."]
        )

    # The error should signal that the LLM never produced a valid tool call.
    # CodeAct routes stop responses through return_result() validation and
    # aborts after max_consecutive_text_only consecutive failures (issue 185).
    error_msg = str(exc_info.value).lower()
    assert (
        "max_retries" in error_msg
        or "generation failed" in error_msg
        or "validation failed" in error_msg
        or "max_consecutive_text_only" in error_msg
        or "codeact aborted" in error_msg
    )


@pytest.mark.asyncio
async def test_missing_await_multiple_methods():
    """Test detection when multiple async methods are called without await."""

    buggy_code = """
results = []
for key in keys:
    data = self.fetch_data(key)  # BUG: Missing await!
    transformed = self.transform_data(data)  # BUG: Missing await!
    results.append(transformed)
return results
"""

    test_llm = FakeLLMClient(scripted_responses=[_resp(buggy_code)] * 4)

    class DataProcessor(Agent, llm=test_llm):
        """Agent that processes data through multiple async steps."""

        async def fetch_data(self, key: str) -> dict:
            """Fetch data for the given key."""
            return {"key": key, "value": "data"}

        async def transform_data(self, data: dict) -> dict:
            """Transform the data."""
            return {**data, "transformed": True}

        async def process_pipeline(self, keys: list[str]) -> list[dict]:
            """Process multiple keys through the full pipeline."""
            ...

    agent_instance = DataProcessor(llm=test_llm)

    with pytest.raises(GenerationError):
        await agent_instance.process_pipeline(["a", "b", "c"])


@pytest.mark.asyncio
async def test_correct_await_usage_passes():
    """Test that correct await usage passes validation."""

    correct_code = """
results = []
for num in numbers:
    result = await self.helper_method(num)  # Correct: has await!
    results.append(result)
return results
"""

    test_llm = FakeLLMClient(scripted_responses=[_resp(correct_code)])

    class CorrectAgent(Agent, llm=test_llm):
        """Agent with correct async/await usage."""

        async def helper_method(self, x: int) -> int:
            """Helper that doubles the input."""
            return x * 2

        @strategy(PurePythonStrategy())
        async def process_list(self, numbers: list[int]) -> list[int]:
            """Process a list of numbers using helper_method."""
            ...

    agent_instance = CorrectAgent(llm=test_llm)

    # This should work without errors
    result = await agent_instance.process_list([1, 2, 3])
    assert result == [2, 4, 6]


@pytest.mark.asyncio
async def test_asyncio_gather_pattern_allowed():
    """Test that list comprehension pattern (for asyncio.gather) is allowed."""

    # Simpler test - just execute coroutines in list comp and await them manually
    gather_code = """
# List comprehension with coroutines (common pattern for gather)
tasks = [self.process_item(item) for item in items]

# Manually await each (simplified version)
results = []
for task in tasks:
    result = await task
    results.append(result)

return results
"""

    test_llm = FakeLLMClient(scripted_responses=[_resp(gather_code)])

    class ParallelAgent(Agent, llm=test_llm):
        """Agent that processes items in parallel."""

        async def process_item(self, item: str) -> str:
            """Process a single item."""
            return f"processed_{item}"

        @strategy(PurePythonStrategy())
        async def process_parallel(self, items: list[str]) -> list[str]:
            """Process items in parallel using coroutines."""
            ...

    agent_instance = ParallelAgent(llm=test_llm)

    # This pattern should pass validation (list comp with coroutines is allowed)
    result = await agent_instance.process_parallel(["a", "b", "c"])
    assert result == ["processed_a", "processed_b", "processed_c"]


@pytest.mark.asyncio
async def test_validation_unit():
    """Unit test the validation method directly."""
    from nooa.strategies.generated_code import GeneratedCodeValidator

    validator = GeneratedCodeValidator()

    # Create a mock agent with async methods
    test_llm = FakeLLMClient()

    class TestAgent(Agent, llm=test_llm):
        async def async_method(self, x: int) -> int:
            """An async method."""
            return x + 1

        async def sync_method(self, x: int) -> int:
            """A sync method."""
            return x + 2

    agent_instance = TestAgent(llm=test_llm)

    # Test 1: Missing await should be invalid
    code_with_error = """
result = self.async_method(5)
return result
"""
    errors = validator.validate(code_with_error, agent_instance)
    assert len(errors) > 0
    assert "async_method" in errors[0]
    assert "await" in errors[0].lower()

    # Test 2: Correct await should be valid
    code_correct = """
result = await self.async_method(5)
return result
"""
    errors = validator.validate(code_correct, agent_instance)
    assert len(errors) == 0

    # Test 3: List comprehension should be allowed (for gather)
    code_with_listcomp = """
tasks = [self.async_method(x) for x in range(10)]
results = await asyncio.gather(*tasks)
return results
"""
    errors = validator.validate(code_with_listcomp, agent_instance)
    assert len(errors) == 0
