# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for parameter names that previously collided with builtins.

'reasoning' was a reserved parameter name while the reasoning() builtin
existed; the builtin has been removed, so no parameter names are rejected
at class creation anymore. Legacy reasoning() calls emitted by models
trained on old traces raise NameError, which reaches the model as error
feedback so it can self-correct.

Note: 'message' was previously reserved but was removed when the message()
builtin was removed from CodeAct.
"""

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


def test_reasoning_parameter_is_no_longer_reserved():
    """'reasoning' is an ordinary parameter name now that the builtin is gone."""

    test_llm = FakeLLMClient()

    class TestAgent(Agent, llm=test_llm):
        """Test agent with reasoning parameter."""

        async def analyze_with_reasoning(self, reasoning: str) -> str:
            """Generation method - reasoning param is allowed."""
            ...

    agent = TestAgent()
    assert agent is not None


def test_message_parameter_is_not_reserved():
    """Test that 'message' is not a reserved parameter name."""

    test_llm = FakeLLMClient()

    class TestAgent(Agent, llm=test_llm):
        """Test agent - message param is allowed."""

        async def process_with_message(self, message: str) -> str:
            """Implemented method - message is not reserved."""
            return f"Got: {message}"

    agent = TestAgent()
    assert agent is not None


def test_burger_order_scenario_with_message_param():
    """Test that 'message' parameter in an implemented method works."""

    test_llm = FakeLLMClient()

    class OrderAgent(Agent, llm=test_llm):
        """Agent for processing food orders."""

        async def add_item(self, item: str) -> None:
            """Add an item to the order."""
            pass

        async def process_request(self, message: str) -> str:
            """Implemented - message param is fine."""
            return f"Handled: {message}"

    agent = OrderAgent()
    assert agent is not None


@pytest.mark.asyncio
async def test_legacy_reasoning_call_raises_and_model_recovers():
    """A legacy reasoning() call raises NameError; the model self-corrects."""

    legacy_code = """
reasoning("Processing the customer request")

result = f"Processed: {request}"

return result
"""
    corrected_code = 'return f"Processed: {request}"'

    test_llm = FakeLLMClient(scripted_responses=[_resp(legacy_code), _resp(corrected_code)])

    class TestAgent(Agent, llm=test_llm):
        """Test agent whose model emits a legacy reasoning() call."""

        @strategy(PurePythonStrategy())
        async def process_request(self, request: str) -> str:
            """Process a request (safe parameter name)."""
            ...

    agent_instance = TestAgent(llm=test_llm)

    # Turn 1 hits NameError (reasoning is not defined), the error is fed
    # back, and turn 2 succeeds.
    result = await agent_instance.process_request("hello world")
    assert result == "Processed: hello world"
    assert test_llm.call_count == 2

    error_events = [
        e
        for e in agent_instance.event_manager.values()
        if e.event_type == "Error" and "NameError" in getattr(e, "content", "")
    ]
    assert error_events, "Expected NameError feedback for the legacy reasoning() call"


def test_multiple_params_none_reserved():
    """Test that neither 'reasoning' nor 'message' triggers a reserved-name error."""

    test_llm = FakeLLMClient()

    class TestAgentOk(Agent, llm=test_llm):
        async def process(self, text: str, message: str, count: int) -> str:
            """Implemented method - message param allowed."""
            return text

    assert TestAgentOk is not None

    class TestAgentAlsoOk(Agent, llm=test_llm):
        async def process(self, text: str, reasoning: str, count: int) -> str:
            """Generation method - reasoning param allowed."""
            ...

    assert TestAgentAlsoOk is not None


@pytest.mark.asyncio
async def test_implemented_methods_accept_any_parameter_names():
    """Implemented methods accept any parameter names."""

    test_llm = FakeLLMClient()

    class TestAgent(Agent, llm=test_llm):
        async def process(self, message: str) -> str:
            """Implemented method."""
            return f"Got: {message}"

    agent = TestAgent()
    result = await agent.process("hello")
    assert result == "Got: hello"
