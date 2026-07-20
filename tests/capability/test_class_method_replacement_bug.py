# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the class-method-replacement bug (now structurally prevented).

Original bug: LLM-generated code did ``ParentAgent.process = _make_process_method()``,
which attached a factory-created function to the agent CLASS. The replacement
ran outside ``execute_code()``'s context, so ``_parent_agent_var`` was ``None``
when sub-agents were instantiated, breaking LLM inheritance.

The pattern is now disabled: ``AgentMeta.__setattr__`` raises
``DynamicMethodAdditionError`` for any callable assignment to an Agent class,
and ``ClassAssignmentValidator`` rejects the equivalent AST patterns
(``MyAgent.foo = fn``, ``setattr(MyAgent, "foo", fn)``, aliased forms) before
exec ever runs. The tests below pin both layers, plus the parent-context
plumbing the bug originally exposed.
"""

import asyncio

import pytest

from nooa import Agent
from nooa.errors import DynamicMethodAdditionError
from nooa.runtime.context_vars import _parent_agent_var
from nooa.unifiedllm import FakeLLMClient, LLMResponse


def make_fake_llm() -> FakeLLMClient:
    """Create a FakeLLMClient with a default response."""
    return FakeLLMClient(
        scripted_responses=[
            LLMResponse(
                raw_response=None,
                content="test",
                tool_calls=[],
                finish_reason="stop",
                assistant_message={"role": "assistant", "content": "test"},
            )
        ]
    )


# Simple sub-agent that needs LLM inheritance
class SimpleSubAgent(Agent):
    """Sub-agent that relies on parent LLM propagation."""

    async def do_work(self) -> str:
        """Simple method that needs LLM."""
        ...


# Parent agent with sub-agent
class ParentAgent(Agent):
    """Parent agent that uses sub-agents."""

    SimpleSubAgent = SimpleSubAgent

    async def process(self, user_input: str) -> str:
        """Process by delegating to sub-agent."""
        ...


class TestGuardPreventsClassMutation:
    """The runtime guard rejects every shape of `MyAgent.method = fn`."""

    def test_named_class_attribute_assignment_is_rejected(self):
        """`ParentAgent.process = fn` raises DynamicMethodAdditionError immediately."""

        async def replacement(self, user_input: str) -> str:
            return "hi"

        with pytest.raises(DynamicMethodAdditionError, match="Cannot attach callable"):
            ParentAgent.process = replacement

    def test_setattr_on_class_is_rejected(self):
        """`setattr(ParentAgent, "process", fn)` raises DynamicMethodAdditionError."""

        async def replacement(self, user_input: str) -> str:
            return "hi"

        with pytest.raises(DynamicMethodAdditionError):
            setattr(ParentAgent, "process", replacement)  # noqa: B010

    def test_factory_returned_function_is_rejected(self):
        """The original bug shape — factory-created function attached to class — fails fast."""

        def _make_process_method():
            async def process(self, user_input: str) -> str:
                return "factory"

            return process

        with pytest.raises(DynamicMethodAdditionError):
            ParentAgent.process = _make_process_method()

    def test_validator_rejects_class_attribute_assignment(self):
        """AST-level: ClassAssignmentValidator catches the same patterns before exec."""
        from nooa.errors import RestrictedCodeError
        from nooa.runtime.code_validator import (
            UnifiedCodeValidator,
            ValidationContext,
        )

        code = """
def _make_process_method():
    async def process(self):
        pass
    return process

ParentAgent.process = _make_process_method()
"""
        agent = ParentAgent.__new__(ParentAgent)
        context = ValidationContext(agent=agent)

        with pytest.raises(RestrictedCodeError, match="class attribute"):
            UnifiedCodeValidator().validate(code, context)


class TestParentContextPlumbing:
    """The parent-context machinery the original bug exposed still works correctly."""

    def test_subagent_creation_fails_without_parent_context(self):
        """SimpleSubAgent() outside execute_code raises 'No LLM available'."""
        assert _parent_agent_var.get() is None
        with pytest.raises(ValueError, match="No LLM available"):
            SimpleSubAgent()

    def test_subagent_inherits_llm_when_parent_context_is_set(self):
        """When _parent_agent_var is set (as execute_code does), subagent inherits LLM."""
        parent = ParentAgent(llm=make_fake_llm())
        token = _parent_agent_var.set(parent)
        try:
            sub = SimpleSubAgent()
            assert sub._llm is parent._llm
        finally:
            _parent_agent_var.reset(token)

    @pytest.mark.asyncio
    async def test_execute_code_sets_parent_context(self):
        """execute_code sets _parent_agent_var so generated code can spawn subagents."""
        parent = ParentAgent(llm=make_fake_llm())
        result = await parent.runtime.execute_code(
            "parent_check = _parent_agent_var.get()",
            builtins={"_parent_agent_var": _parent_agent_var},
        )
        assert result.error is None


class TestEndToEndCorrectPattern:
    """The recommended pattern (explicit `llm=self._llm` on subagent) works end-to-end."""

    @pytest.mark.asyncio
    async def test_subagent_with_explicit_llm_inside_execute_code(self):
        """Inside execute_code, SimpleSubAgent() inherits LLM via parent context var.

        This exercises the happy path the original bug broke.
        """
        parent = ParentAgent(llm=make_fake_llm())
        token = _parent_agent_var.set(parent)
        try:
            sub = SimpleSubAgent()
            assert sub._llm is parent._llm
        finally:
            _parent_agent_var.reset(token)

    @pytest.mark.asyncio
    async def test_subagent_with_explicit_llm_outside_execute_code(self):
        """Even outside execute_code, passing `llm=self._llm` explicitly still works.

        This was the recommended fix when the bug existed; it remains the
        defensive pattern for any subagent instantiation that happens off the
        normal execute_code path.
        """
        parent = ParentAgent(llm=make_fake_llm())
        sub = SimpleSubAgent(llm=parent._llm)
        assert sub._llm is parent._llm


# Avoid F811 on legacy import users — re-exported names that other modules historically referenced
asyncio  # noqa: B018
