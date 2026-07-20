# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test that PurePythonStrategy correctly rejects and retries when LLM defines target method."""

import pytest

from nooa import Agent, strategy
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient


class SampleAgent(Agent):
    """Sample agent for method rejection tests."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = []

    @strategy(PurePythonStrategy())
    async def process_data(self, items: list[str]) -> list[str]:
        """Process the data items."""
        ...


@pytest.mark.asyncio
async def test_llm_defining_target_method_triggers_error_and_retry():
    """Test that when LLM defines the target method with other non-function code, it gets rejected.

    Scenario:
    1. LLM defines `process_data` as a function WITH other top-level code (print statement)
    2. System rejects it with error message (because of other non-function code)
    3. LLM sees error and provides correct implementation with return statement
    """
    llm = FakeLLMClient.with_code_responses(
        [
            # First attempt: LLM defines target method WITH other non-function code (will be rejected)
            """


async def _helper(self, item: str) -> str:
    return item.upper()


async def process_data(self, items: list[str]) -> list[str]:
    results = []
    for item in items:
        results.append(await _helper(self, item))
    return results

print("done")  # Other non-function code - causes rejection
""",
            # Second attempt: LLM sees error, provides correct inline implementation.
            # call the helper as a plain callable — ``helper(self, item)``.
            """
results = []
for item in items:
    results.append(await _helper(self, item))
return results
""",
        ]
    )

    my_agent = SampleAgent(llm=llm)
    result = await my_agent.process_data(["hello", "world"])

    # Should successfully process after retry
    assert result == ["HELLO", "WORLD"]

    # helpers are plain callables in session_locals — not attached to agent.
    assert not hasattr(my_agent, "_helper")

    # Verify that both responses were used (first was rejected due to other code, second succeeded)
    assert llm.call_count == 2, "Should have used both LLM responses (rejection + retry)"


@pytest.mark.asyncio
async def test_helper_methods_are_not_rejected():
    """Test that helper methods with different names are installed correctly."""
    llm = FakeLLMClient.with_code_responses(
        [
            """
# Define helpers with different names (should be accepted).
# helpers are plain callables — call as helper(self, ...), not self.helper(...).
async def _helper1(self, x: str) -> str:
    return x.upper()


async def _helper2(self, x: str) -> str:
    return x.lower()

# Return result immediately
result = []
result.append(await _helper1(self, items[0]))
result.append(await _helper2(self, items[1]))
return result
"""
        ]
    )

    my_agent = SampleAgent(llm=llm)
    result = await my_agent.process_data(["Hello", "WORLD"])

    assert result == ["HELLO", "world"]
    # helpers live as plain callables in session_locals, not on the agent.
    assert not hasattr(my_agent, "_helper1")
    assert not hasattr(my_agent, "_helper2")


@pytest.mark.asyncio
async def test_multiple_rejected_methods_in_error_message():
    """Test that when method is wrapped in function definition with other code, it gets rejected."""
    llm = FakeLLMClient.with_code_responses(
        [
            # Try to define target method WITH other top-level code (will be rejected)
            """
# Some comment
async def process_data(self, items: list[str]) -> list[str]:
    return [item.upper() for item in items]
# More code
print("done")
""",
            # Retry with correct approach
            """
return [item.upper() for item in items]
""",
        ]
    )

    my_agent = SampleAgent(llm=llm)
    result = await my_agent.process_data(["test"])

    assert result == ["TEST"]

    # Verify that both responses were used (first was rejected due to other top-level code, second succeeded)
    assert llm.call_count == 2, "Should have used both LLM responses"


@pytest.mark.asyncio
async def test_function_body_extraction_when_wrapped_in_function_definition():
    """Test that when LLM returns code wrapped in function definition matching target method,
    we extract the body and execute it, updating history to show unpacked code.

    Scenario:
    1. LLM returns code wrapped in function definition: `async def process(self): ...`
    2. Function name matches target method name (`process`)
    3. No other top-level code exists
    4. System extracts function body and executes it
    5. History is updated to show unpacked code so LLM learns from example
    """

    class TestAgent(Agent):
        @strategy(PurePythonStrategy())
        async def process(self) -> str:
            """Return a greeting message."""
            ...

    # LLM returns wrapped function definition
    wrapped_code = """async def process(self) -> str:
    return "Hello, world!"
"""

    llm = FakeLLMClient.with_code_responses([wrapped_code])

    agent_instance = TestAgent(llm=llm)
    result = await agent_instance.process()

    # Should successfully execute the extracted body
    assert result == "Hello, world!"

    # Verify history was updated with unpacked code
    history_events = agent_instance.event_manager.values()
    assistant_events = [e for e in history_events if e.event_type == "LLMOutput"]

    # Should have at least one assistant event
    assert len(assistant_events) >= 1

    # The unpacked code should be in the history (without function definition wrapper)
    last_assistant_msg = assistant_events[-1].content
    assert "async def process" not in last_assistant_msg
    assert "return" in last_assistant_msg and "Hello, world!" in last_assistant_msg


@pytest.mark.asyncio
async def test_helper_method_plus_wrapped_target_method():
    """Test that helper method + wrapped target method pattern is supported.

    Scenario:
    1. LLM returns code with helper method definition + target method wrapped in function definition
    2. System extracts helper method (installs it) + extracts target method body
    3. Both are executed successfully
    """

    class SentimentAgent(Agent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.data1 = ["I love this!", "This is great"]
            self.data2 = ["I hate this", "Terrible product"]
            self.data3 = ["Neutral statement"]

        @strategy(PurePythonStrategy())
        async def find_negative_sentiment(self) -> str:
            """Find the first sentence with negative sentiment."""
            ...

    # helpers are plain callables — call as helper(self, ...).
    wrapped_code = """async def is_negative(self, sentence: str) -> bool:
    negative_words = ["hate", "terrible", "bad", "awful"]
    return any(word in sentence.lower() for word in negative_words)


async def find_negative_sentiment(self) -> str:
    for sentence in self.data1 + self.data2 + self.data3:
        if await is_negative(self, sentence):
            return sentence
    return ""
"""

    llm = FakeLLMClient.with_code_responses([wrapped_code])

    agent_instance = SentimentAgent(llm=llm)
    result = await agent_instance.find_negative_sentiment()

    # Should successfully execute and find negative sentiment
    assert isinstance(result, str)
    assert result in ["I hate this", "Terrible product"]

    # helpers are plain callables, not attached to the agent.
    assert not hasattr(agent_instance, "is_negative")

    # Verify history was updated with unpacked code
    history_events = agent_instance.event_manager.values()
    assistant_events = [e for e in history_events if e.event_type == "LLMOutput"]

    # Should have at least one assistant event
    assert len(assistant_events) >= 1

    # The unpacked code should be in the history (target method body unwrapped)
    last_assistant_msg = assistant_events[-1].content
    assert "async def find_negative_sentiment" not in last_assistant_msg
    assert "for sentence in" in last_assistant_msg or "return" in last_assistant_msg
    # Helper method definition should still be there
    assert "async def is_negative" in last_assistant_msg


@pytest.mark.asyncio
async def test_llm_calling_current_method_is_rejected():
    """Test that code calling the method it's implementing is rejected at validation.

    This is a common failure mode where the LLM tries to "delegate" to itself.
    The validator now catches this and rejects the code BEFORE execution,
    preventing infinite recursion.

    Current behavior: The call is rejected at validation with a clear error message.
    The LLM gets feedback and can retry with correct code.
    """

    class RecursionTestAgent(Agent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.call_count = 0

        @strategy(PurePythonStrategy(max_iterations=3, max_retries=3))
        async def calculate(self, x: int) -> int:
            """Double the value x."""
            ...

    # LLM first tries to call itself (rejected), then provides correct code
    llm = FakeLLMClient.with_code_responses(
        [
            # First call - LLM wrongly tries to call itself (will be rejected)
            """
self.call_count += 1
# Wrong! This calls the same method we're implementing
result = await self.calculate(x)
return result
""",
            # Second call - LLM corrects itself after rejection feedback
            """
self.call_count += 1
return x * 2
""",
        ]
    )

    agent = RecursionTestAgent(llm=llm)
    result = await agent.calculate(5)

    # The correct code executed once
    assert agent.call_count == 1, "Should have executed once (after retry)"
    assert result == 10, f"Expected 10, got {result}"
    # LLM was called twice (first rejected, second succeeded)
    assert llm.call_count == 2


@pytest.mark.asyncio
async def test_recursive_call_rejected_with_clear_error():
    """Test that recursive self-calls are rejected with a clear error message.

    The validator catches self.method_name() calls where method_name is the
    method being generated, and rejects them with a helpful error explaining
    that this would cause infinite recursion.

    The LLM receives feedback and can retry with correct code.
    """

    class RecursionTestAgent(Agent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.generation_count = 0

        @strategy(PurePythonStrategy(max_iterations=5, max_retries=5))
        async def double(self, x: int) -> int:
            """Double the value x."""
            ...

    # First response has recursive call (rejected), second is correct
    llm = FakeLLMClient.with_code_responses(
        [
            # First call - generates recursive call (will be rejected)
            """
self.generation_count += 1
result = await self.double(x)  # Rejected - would cause infinite recursion
return result
""",
            # Second call - correct implementation after feedback
            """
self.generation_count += 1
return x * 2
""",
        ]
    )

    agent = RecursionTestAgent(llm=llm)
    result = await agent.double(5)

    # Only the second (correct) generation was executed
    assert agent.generation_count == 1, "Should have 1 generation (recursive call was rejected)"
    # Correct result
    assert result == 10, f"Expected 10, got {result}"
    # LLM was called twice (first rejected, second succeeded)
    assert llm.call_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
