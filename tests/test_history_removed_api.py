# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test that legacy attributes have been removed and new API exists."""

from nooa import Agent
from nooa.unifiedllm import FakeLLMClient

# Module-level test LLM (can be overridden at instantiation)
_TEST_LLM = FakeLLMClient()


def test_no_legacy_attributes():
    """Verify that agents don't have legacy attributes."""

    class TestAgent(Agent, llm=_TEST_LLM):
        pass

    test_agent = TestAgent()

    # Verify no legacy attributes
    assert not hasattr(test_agent, "render_format"), "agent.render_format should not exist"
    assert not hasattr(test_agent, "prompt_stats"), "agent.prompt_stats should not exist"
    assert not hasattr(test_agent, "prompts"), (
        "agent.prompts should not exist (PromptBuilder has own)"
    )
    assert not hasattr(test_agent, "_tasks"), (
        "agent._tasks should not exist (use runtime._current_task)"
    )
    assert not hasattr(test_agent, "_tasks_completed"), "agent._tasks_completed should not exist"

    # Verify new systems exist
    assert hasattr(test_agent, "event_manager"), "agent.event_manager should exist"
    assert hasattr(test_agent, "context_manager"), "agent.context_manager should exist"
    # events and context are always present but hidden (not opt-in anymore)
    assert hasattr(test_agent, "events"), "agent.events should exist (always present, hidden)"
    assert hasattr(test_agent, "context"), "agent.context should exist (always present, hidden)"
    assert not hasattr(test_agent, "history"), "agent.history should NOT exist (removed)"
    # context_spec and blocks have been removed in context-blocks refactor
    assert not hasattr(test_agent, "context_spec"), "agent.context_spec should NOT exist (removed)"
    assert not hasattr(test_agent, "blocks"), "agent.blocks should NOT exist (removed)"

    print("✅ No legacy attributes")
    print("✅ New systems exist")


if __name__ == "__main__":
    test_no_legacy_attributes()
    print("\n✅ All tests passed!")
