# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration test for event management with Event-based API."""

from nooa import Agent, strategy
from nooa.events import LLMOutput, Task
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient


def _format_events_for_test(events: list) -> list[dict]:
    """Test helper: format events to OpenAI message format."""
    result = []
    for event in events:
        # Get role from class attribute
        role = event._role.value

        # Extract content from known content fields
        content = ""
        if hasattr(event, "content"):
            content = event.content
        elif hasattr(event, "prompt"):
            content = event.prompt
        if not isinstance(content, str):
            content = str(content) if content else ""
        result.append({"role": role, "content": content})
    return result


# Module-level test LLM (can be overridden at instantiation)
_TEST_LLM = FakeLLMClient()


def test_agent_has_event_manager_api():
    """Test that agents have self.event_manager as the primary API."""

    class TestAgent(Agent, llm=_TEST_LLM):
        @strategy(PurePythonStrategy())
        async def add_numbers(self, a: int, b: int) -> int:
            """Add two numbers and return the result."""
            ...

    # Create agent
    test_agent = TestAgent()

    # Verify _event_manager exists as primary API
    assert hasattr(test_agent, "event_manager")
    assert test_agent.event_manager is not None

    # Verify events is always present (hidden but auto-created)
    assert hasattr(test_agent, "events")

    # Verify history alias does NOT exist (removed)
    assert not hasattr(test_agent, "history")

    # Verify _event_manager is empty initially
    assert len(test_agent.event_manager) == 0

    print("✅ Agent has self.event_manager as primary API")
    print("✅ self.events is always present (hidden by default)")


def test_history_operations():
    """Test that history manager operations work directly."""

    # Create agent

    class SimpleAgent(Agent, llm=_TEST_LLM):
        pass

    agent_instance = SimpleAgent()
    hm = agent_instance.event_manager

    # Test basic operations with Events
    hm.add(Task(prompt="Test task"))
    assert len(hm) == 1

    hm.add(LLMOutput(content="Test response"))
    assert len(hm) == 2

    # Convert to OpenAI format via formatter
    messages = _format_events_for_test(hm.values())
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"

    print("✅ Event manager operations work correctly")


if __name__ == "__main__":
    test_agent_has_event_manager_api()
    test_history_operations()
    print("\n✅ All integration tests passed!")
