# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for PURE_PYTHON behavior when LLM returns a class definition.

Class definitions are now allowed in REPL-style code. This file verifies that
the agent can handle class definitions returned by the LLM without errors,
and can actually use the defined class to produce results.
"""

import pytest

from nooa import Agent, strategy
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient

# Code where the LLM defines a helper class and uses it to compute the result.
_CLASS_AND_USE_SNIPPET = """
class Truncator:
    def __init__(self, length):
        self.length = length

    def truncate(self, s):
        return s[:self.length]

t = Truncator(5)
return [t.truncate(doc) for doc in documents]
""".strip()


@pytest.mark.asyncio
async def test_agent_can_define_and_use_class():
    """Agent-generated code may define a class and use it to produce the result."""

    llm = FakeLLMClient.with_code_responses([_CLASS_AND_USE_SNIPPET])

    class TestAgent(Agent, llm=llm):
        @strategy(PurePythonStrategy(max_iterations=5, max_retries=3))
        async def summarize_batch(self, documents: list[str]) -> list[str]:
            """Summarize each document."""
            ...

    a = TestAgent()
    result = await a.summarize_batch(["abcdef", "ghijkl"])
    assert result == ["abcde", "ghijk"]
    assert llm.call_count == 1

    # No errors should have been recorded
    errors = [e for e in a.event_manager.values() if e.event_type == "Error"]
    assert errors == []


@pytest.mark.asyncio
async def test_pure_python_class_definition_no_errors():
    """When LLM returns code with a class definition, no class-related errors are emitted."""

    llm = FakeLLMClient.with_code_responses([_CLASS_AND_USE_SNIPPET])

    class TestAgent(Agent, llm=llm):
        @strategy(PurePythonStrategy(max_iterations=5, max_retries=3))
        async def summarize_batch(self, documents: list[str]) -> list[str]:
            """Summarize each document."""
            ...

    a = TestAgent()
    await a.summarize_batch(["abcdef", "ghijkl"])

    errors = [e for e in a.event_manager.values() if e.event_type == "Error"]
    assert not any("class" in (e.content or "").lower() for e in errors)


# ---------------------------------------------------------------------------
# Multi-turn: class defined in turn 1 survives into turn 2
# ---------------------------------------------------------------------------

_TURN1_DEFINE_CLASS = """
class TodoManager:
    def __init__(self):
        self.todos = []

    def add(self, item):
        self.todos.append(item)

    def get_all(self):
        return list(self.todos)

tm = TodoManager()
tm.add("buy milk")
print("TodoManager created")
""".strip()

_TURN2_USE_CLASS = """
tm.add("buy eggs")
return tm.get_all()
""".strip()


@pytest.mark.asyncio
async def test_class_defined_in_turn1_survives_to_turn2():
    """A class and instance created in turn 1 persist into turn 2 of the same session.

    Turn 1: defines TodoManager, creates an instance, adds one item (no return → continues).
    Turn 2: adds another item, returns the full list.
    """
    llm = FakeLLMClient.with_code_responses([_TURN1_DEFINE_CLASS, _TURN2_USE_CLASS])

    class TestAgent(Agent, llm=llm):
        @strategy(PurePythonStrategy(max_iterations=5, max_retries=3))
        async def manage_todos(self) -> list[str]:
            """Manage a todo list across two turns."""
            ...

    a = TestAgent()
    result = await a.manage_todos()
    assert result == ["buy milk", "buy eggs"]
    assert llm.call_count == 2

    errors = [e for e in a.event_manager.values() if e.event_type == "Error"]
    assert errors == []
