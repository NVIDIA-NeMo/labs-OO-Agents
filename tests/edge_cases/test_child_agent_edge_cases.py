# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Edge cases for child agents.

Focus on:
- LLM inheritance from parent agent when llm parameter is omitted (cascading)
- Child agent created during generated code execution
- Explicit LLM overrides child inheritance
- Multiple levels of nesting
"""

import pytest

from nooa import Agent, strategy
from nooa.agent import _parent_agent_var
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


# ============================================================================
# Test: LLM inheritance via context variable
# ============================================================================


class TestLLMInheritanceContextVar:
    """Test that _parent_agent_var enables LLM inheritance."""

    def test_parent_agent_var_default_is_none(self):
        """Context variable should default to None."""
        assert _parent_agent_var.get() is None

    def test_subagent_inherits_llm_when_parent_set(self):
        """Subagent with llm parameter omitted inherits LLM from parent in context."""
        fake_llm = FakeLLMClient([_resp("pass")])

        class ParentAgent(Agent, llm=fake_llm):
            pass

        class ChildAgent(Agent):
            pass

        parent = ParentAgent()

        # Manually set context var (simulating execute_code behavior)
        token = _parent_agent_var.set(parent)
        try:
            child = ChildAgent()
            assert child._llm is fake_llm, "Child should inherit parent's LLM"
        finally:
            _parent_agent_var.reset(token)

    def test_explicit_llm_overrides_inheritance(self):
        """Explicit llm= parameter should override inheritance."""
        parent_llm = FakeLLMClient([_resp("parent")])
        child_llm = FakeLLMClient([_resp("child")])

        class ParentAgent(Agent, llm=parent_llm):
            pass

        class ChildAgent(Agent):
            pass

        parent = ParentAgent()

        token = _parent_agent_var.set(parent)
        try:
            # Explicit llm= should win over inheritance
            child = ChildAgent(llm=child_llm)
            assert child._llm is child_llm, "Explicit LLM should override inheritance"
        finally:
            _parent_agent_var.reset(token)

    def test_class_level_llm_overrides_inheritance(self):
        """Class with its own LLM should use that, not inherit."""
        parent_llm = FakeLLMClient([_resp("parent")])
        child_llm = FakeLLMClient([_resp("child")])

        class ParentAgent(Agent, llm=parent_llm):
            pass

        class ChildAgent(Agent, llm=child_llm):  # Has its own LLM
            pass

        parent = ParentAgent()

        token = _parent_agent_var.set(parent)
        try:
            child = ChildAgent()
            assert child._llm is child_llm, "Class-level LLM should be used, not inherited"
        finally:
            _parent_agent_var.reset(token)

    def test_no_parent_no_llm_raises_error(self):
        """Without parent and no LLM, should raise ValueError."""

        class OrphanAgent(Agent):
            pass

        with pytest.raises(ValueError, match="No LLM available"):
            OrphanAgent()


# ============================================================================
# Test: LLM inheritance during execute_code
# ============================================================================

# Define child agent at module level for test_child_inherits_llm_in_generated_code


class _ChildAgentForInheritTest(Agent):
    """Child that will inherit LLM."""

    pass


# Define child agent at module level for test_child_agent_can_run_methods


class _ChildAgentWithMethod(Agent):
    """Child agent with LLM method."""

    @strategy(PurePythonStrategy())
    async def say_hello(self) -> str:
        """Return a greeting."""
        ...


# Define child agent at module level for test_multiple_children_share_parent_llm


class _ChildAgentForSharingTest(Agent):
    pass


class TestLLMInheritanceDuringExecution:
    """Test that execute_code sets parent context correctly."""

    @pytest.mark.asyncio
    async def test_child_inherits_llm_in_generated_code(self):
        """Child agent created in generated code should inherit parent's LLM."""
        # We need enough responses for both parent and child generations
        fake_llm = FakeLLMClient(
            [
                # Parent's method generates code that creates a child
                _resp("child = self.ChildAgent()\nreturn 'created'"),
            ]
        )

        class ParentAgent(Agent, llm=fake_llm):
            """Parent agent that creates child in generated code."""

            ChildAgent = _ChildAgentForInheritTest

            @strategy(PurePythonStrategy())
            async def create_child(self) -> str:
                """Create a child agent and return confirmation."""
                ...

        parent = ParentAgent()
        result = await parent.create_child()
        assert result == "created"

        # The child should have been created with inherited LLM
        # (we can't directly access it, but no error means it worked)

    @pytest.mark.asyncio
    async def test_child_agent_can_run_methods(self):
        """Child agent with inherited LLM can execute its own methods."""
        fake_llm = FakeLLMClient(
            [
                # Parent generates code to create and call child
                _resp("child = self.ChildAgent()\nresult = await child.say_hello()\nreturn result"),
                # Child's say_hello method response
                _resp("return 'hello from child'"),
            ]
        )

        class ParentAgent(Agent, llm=fake_llm):
            """Parent that creates and calls child."""

            ChildAgent = _ChildAgentWithMethod

            @strategy(PurePythonStrategy())
            async def orchestrate(self) -> str:
                """Create child agent and call its method."""
                ...

        parent = ParentAgent()
        result = await parent.orchestrate()
        assert result == "hello from child"

    @pytest.mark.asyncio
    async def test_multiple_children_share_parent_llm(self):
        """Multiple children created in same execution share parent's LLM."""
        fake_llm = FakeLLMClient(
            [
                _resp("c1 = self.ChildAgent()\nc2 = self.ChildAgent()\nreturn c1._llm is c2._llm"),
            ]
        )

        class ParentAgent(Agent, llm=fake_llm):
            ChildAgent = _ChildAgentForSharingTest

            @strategy(PurePythonStrategy())
            async def check_llm_sharing(self) -> bool:
                """Create two children and check if they share LLM."""
                ...

        parent = ParentAgent()
        result = await parent.check_llm_sharing()
        assert result is True


# ============================================================================
# Test: Nested hierarchy (grandchild)
# ============================================================================

# Define child/grandchild agents at module level


class _GrandchildAgent(Agent):
    """Grandchild - no LLM."""

    pass


class _MiddleAgent(Agent):
    """Middle level - no LLM."""

    ChildAgent = _GrandchildAgent

    @strategy(PurePythonStrategy())
    async def create_child(self) -> str:
        """Create a grandchild."""
        ...


class TestNestedLLMInheritance:
    """Test LLM inheritance through multiple levels."""

    @pytest.mark.asyncio
    async def test_grandchild_inherits_from_grandparent(self):
        """Three-level nesting: grandchild should inherit grandparent's LLM."""
        fake_llm = FakeLLMClient(
            [
                # Grandparent creates parent
                _resp("parent = self.ParentAgent()\nreturn await parent.create_child()"),
                # Parent creates child
                _resp("child = self.ChildAgent()\nreturn 'grandchild created'"),
            ]
        )

        class GrandparentAgent(Agent, llm=fake_llm):
            """Top level with LLM."""

            ParentAgent = _MiddleAgent

            @strategy(PurePythonStrategy())
            async def orchestrate(self) -> str:
                """Create parent which creates child."""
                ...

        grandparent = GrandparentAgent()
        result = await grandparent.orchestrate()
        assert result == "grandchild created"


# ============================================================================
# Test: Error messages
# ============================================================================


class TestLLMInheritanceErrors:
    """Test error handling for LLM inheritance."""

    def test_helpful_error_without_llm(self):
        """Error message should explain how to provide LLM."""

        class NoLLMAgent(Agent):
            pass

        with pytest.raises(ValueError) as exc_info:
            NoLLMAgent()

        error_msg = str(exc_info.value)
        assert "NoLLMAgent" in error_msg
        assert "llm=" in error_msg

    def test_missing_class_level_llm_error(self):
        """Agent without class-level LLM should raise ValueError."""

        class NoLLMConfigAgent(Agent):
            pass

        # Without class-level LLM, _agent_llm doesn't exist
        with pytest.raises(ValueError, match="No LLM available"):
            NoLLMConfigAgent()
