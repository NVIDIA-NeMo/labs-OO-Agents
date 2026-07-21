# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test that method spans have correct parent-child relationships.

This tests the bug where nested @strategy method calls (methods with ellipsis bodies
that need LLM generation) don't properly track parent-child relationships in the
agent call stack. The issue is in actor.py: _agent_call_stack is only pushed/popped
when is_nested=True, but the first method never pushes because is_nested=False for it.
So child methods get parent_call_id=None instead of the parent's call_id.
"""

import pytest

from nooa import no_trace, strategy
from nooa.agent import Agent
from nooa.runtime.hooks import get_hooks, set_hooks
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


class SpanTrackingHooks:
    """Mock hooks that track span parent relationships."""

    def __init__(self):
        self.calls = []  # List of dicts with method_name, call_id, parent_call_id
        self.generations = []  # List of dicts with method_name, generation_id, parent_generation_id

    def before_agent_call(self, agent, method_name, args, kwargs, call_id, parent_call_id, **extra):
        self.calls.append(
            {
                "method_name": method_name,
                "call_id": call_id,
                "parent_call_id": parent_call_id,
            }
        )
        return {"call_id": call_id}

    def after_agent_call(self, agent, method_name, result, exception, context, **kwargs):
        pass

    def on_messages_built(self, agent, method_name, messages, generation_id, **kwargs):
        pass

    def before_generation(
        self, agent, method_name, strategy, generation_id, parent_generation_id, **extra
    ):
        self.generations.append(
            {
                "method_name": method_name,
                "generation_id": generation_id,
                "parent_generation_id": parent_generation_id,
            }
        )
        return {"generation_id": generation_id}

    def after_generation(
        self, agent, method_name, result, exception, context, generation_id, **kwargs
    ):
        pass

    def before_code_execution(self, agent, code, execution_id, **extra):
        return {}

    def after_code_execution(self, agent, code, result, exception, context, execution_id, **kwargs):
        pass

    def before_method_invocation(self, agent, method_name, args, kwargs, invocation_id, **extra):
        return {}

    def after_method_invocation(
        self, agent, method_name, result, exception, context, invocation_id, **kwargs
    ):
        pass

    def before_tool_execution(self, agent, tool_name, arguments, execution_id, **extra):
        return {}

    def after_tool_execution(
        self, agent, tool_name, arguments, result, exception, context, execution_id, **kwargs
    ):
        pass


@pytest.fixture
def span_hooks():
    """Fixture that installs span tracking hooks and cleans up."""
    hooks = SpanTrackingHooks()
    old_hooks = get_hooks()
    set_hooks(hooks)
    yield hooks
    set_hooks(old_hooks)


# Module-level placeholder LLM (will be overridden in tests)
_TEST_LLM = FakeLLMClient()


# Module-level subagent classes for parallel invocation tests
# These need to be at module level for Python scoping rules
class _TransformAgent(Agent, llm=_TEST_LLM):
    """Subagent that transforms data."""

    @strategy(PurePythonStrategy())
    async def transform(self, data: str) -> str:
        """Transform the data."""
        ...


class _ValidateAgent(Agent, llm=_TEST_LLM):
    """Subagent that validates data."""

    @strategy(PurePythonStrategy())
    async def validate(self, data: str) -> str:
        """Validate the data."""
        ...


class TestMethodSpanParentRelationship:
    """Test that nested method calls have correct parent-child span relationships.

    These tests use methods with ellipsis bodies (...) that trigger LLM generation.
    The hooks are only called for such methods (not for methods with implementations).
    """

    @pytest.mark.asyncio
    async def test_nested_method_has_parent_call_id(self, span_hooks):
        """When a method calls another method, the child should have parent_call_id set.

        This is the core bug: outer_method calling inner_method should result in inner_method
        having parent_call_id pointing to outer_method's call_id.
        """
        # Configure FakeLLMClient to return code that calls inner_method
        fake_llm = FakeLLMClient(
            scripted_responses=[
                # outer_method generation: calls inner_method
                _resp("result = await self.inner_method()\nreturn f'outer({result})'"),
                # inner_method generation: returns simple result
                _resp("return 'inner'"),
            ]
        )

        class NestedAgent(Agent, llm=_TEST_LLM):
            """Agent with nested method calls using ellipsis bodies."""

            @strategy(PurePythonStrategy())
            async def outer_method(self) -> str:
                """Root method that calls inner method."""
                ...

            @strategy(PurePythonStrategy())
            async def inner_method(self) -> str:
                """Child method called by outer."""
                ...

        agent = NestedAgent(llm=fake_llm)
        result = await agent.outer_method()

        assert result == "outer(inner)"

        # Verify we captured both method calls
        assert len(span_hooks.calls) == 2, (
            f"Expected 2 calls, got {len(span_hooks.calls)}: {span_hooks.calls}"
        )

        outer_call = next(c for c in span_hooks.calls if c["method_name"] == "outer_method")
        inner_call = next(c for c in span_hooks.calls if c["method_name"] == "inner_method")

        # Root method should have no parent
        assert outer_call["parent_call_id"] is None, (
            f"Root method outer_method should have no parent, got {outer_call['parent_call_id']}"
        )

        # Child method should have parent pointing to outer's call_id
        # THIS IS THE BUG: inner_method has parent_call_id=None instead of outer's call_id
        assert inner_call["parent_call_id"] == outer_call["call_id"], (
            f"inner_method should have parent_call_id={outer_call['call_id']}, "
            f"but got {inner_call['parent_call_id']}"
        )

    @pytest.mark.asyncio
    async def test_deeply_nested_methods_chain_correctly(self, span_hooks):
        """Test three levels of nesting: A -> B -> C."""
        fake_llm = FakeLLMClient(
            scripted_responses=[
                # level_a generation: calls level_b
                _resp("result = await self.level_b()\nreturn f'A({result})'"),
                # level_b generation: calls level_c
                _resp("result = await self.level_c()\nreturn f'B({result})'"),
                # level_c generation: returns "C"
                _resp("return 'C'"),
            ]
        )

        class DeeplyNestedAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy())
            async def level_a(self) -> str:
                """Level A calls level B."""
                ...

            @strategy(PurePythonStrategy())
            async def level_b(self) -> str:
                """Level B calls level C."""
                ...

            @strategy(PurePythonStrategy())
            async def level_c(self) -> str:
                """Level C is the leaf."""
                ...

        agent = DeeplyNestedAgent(llm=fake_llm)
        result = await agent.level_a()

        assert result == "A(B(C))"
        assert len(span_hooks.calls) == 3, (
            f"Expected 3 calls, got {len(span_hooks.calls)}: {span_hooks.calls}"
        )

        call_a = next(c for c in span_hooks.calls if c["method_name"] == "level_a")
        call_b = next(c for c in span_hooks.calls if c["method_name"] == "level_b")
        call_c = next(c for c in span_hooks.calls if c["method_name"] == "level_c")

        # Verify chain: A has no parent, B's parent is A, C's parent is B
        assert call_a["parent_call_id"] is None
        assert call_b["parent_call_id"] == call_a["call_id"], (
            f"level_b should have parent={call_a['call_id']}, got {call_b['parent_call_id']}"
        )
        assert call_c["parent_call_id"] == call_b["call_id"], (
            f"level_c should have parent={call_b['call_id']}, got {call_c['parent_call_id']}"
        )

    @pytest.mark.asyncio
    async def test_parallel_children_have_same_parent(self, span_hooks):
        """When parent calls gather(child_a(), child_b()), both should have parent as parent_call_id.

        This tests for the ContextVar mutable list sharing bug:
        - Parent pushes its call_id to stack
        - gather() creates tasks that inherit parent's context (same list reference!)
        - child_a starts, reads parent correctly, pushes its call_id
        - child_b starts, reads from stack... but sees child_a's call_id (WRONG!)

        The fix requires copy-on-write semantics for the stack.
        """
        fake_llm = FakeLLMClient(
            scripted_responses=[
                # parent_method: calls gather with both children
                _resp("""import asyncio
results = await asyncio.gather(
    self.child_a(),
    self.child_b(),
)
return f"parent({results[0]}, {results[1]})"
"""),
                # child_a
                _resp('return "a"'),
                # child_b
                _resp('return "b"'),
            ]
        )

        class ParallelAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy())
            async def parent_method(self) -> str:
                """Parent that calls children in parallel via gather."""
                ...

            @strategy(PurePythonStrategy())
            async def child_a(self) -> str:
                """First child."""
                ...

            @strategy(PurePythonStrategy())
            async def child_b(self) -> str:
                """Second child."""
                ...

        agent = ParallelAgent(llm=fake_llm)
        result = await agent.parent_method()

        assert result == "parent(a, b)"
        assert len(span_hooks.calls) == 3, (
            f"Expected 3 calls, got {len(span_hooks.calls)}: {span_hooks.calls}"
        )

        parent_call = next(c for c in span_hooks.calls if c["method_name"] == "parent_method")
        child_a_call = next(c for c in span_hooks.calls if c["method_name"] == "child_a")
        child_b_call = next(c for c in span_hooks.calls if c["method_name"] == "child_b")

        # Parent should have no parent
        assert parent_call["parent_call_id"] is None, (
            f"parent_method should have no parent, got {parent_call['parent_call_id']}"
        )

        # BOTH children should have parent_method as their parent
        assert child_a_call["parent_call_id"] == parent_call["call_id"], (
            f"child_a should have parent={parent_call['call_id']}, "
            f"got {child_a_call['parent_call_id']}"
        )

        # THIS IS THE BUG: child_b might see child_a as its parent due to shared mutable list!
        assert child_b_call["parent_call_id"] == parent_call["call_id"], (
            f"child_b should have parent={parent_call['call_id']}, "
            f"got {child_b_call['parent_call_id']} "
            f"(if this is child_a's call_id, we have the ContextVar mutable list sharing bug!)"
        )

    @pytest.mark.asyncio
    async def test_parallel_subagent_calls_have_same_parent(self, span_hooks):
        """When parent calls gather() on methods of different subagent instances, all should have parent as parent_call_id.

        This tests the real-world pattern from RouterTestWrapper:
        - Parent (RouterAgent.process) pushes its call_id to stack
        - Generated code creates subagent instances and calls gather()
        - SubAgentA.transform() starts, reads parent correctly, pushes its call_id
        - SubAgentB.validate() starts, reads from stack... but might see transform's call_id!

        The fix requires copy-on-write semantics for the stack.
        """
        fake_llm = FakeLLMClient(
            scripted_responses=[
                # process: creates subagent instances and calls gather
                _resp("""import asyncio

# Create subagent instances (like RouterTestWrapper does)
transformer = self._TransformAgent(llm=self._llm)
validator = self._ValidateAgent(llm=self._llm)

# Call them in parallel
results = await asyncio.gather(
    transformer.transform("input"),
    validator.validate("input"),
)
return f"processed({results[0]}, {results[1]})"
"""),
                # transform response
                _resp('return "transformed"'),
                # validate response
                _resp('return "validated"'),
            ]
        )

        class RouterAgent(Agent, llm=_TEST_LLM):
            # Subagent classes available for instantiation (use module-level classes)
            _TransformAgent = _TransformAgent
            _ValidateAgent = _ValidateAgent

            @strategy(PurePythonStrategy())
            async def process(self, data: str) -> str:
                """Process data using subagents in parallel."""
                ...

        agent = RouterAgent(llm=fake_llm)
        result = await agent.process("test")

        assert result == "processed(transformed, validated)"
        assert len(span_hooks.calls) == 3, (
            f"Expected 3 calls, got {len(span_hooks.calls)}: {span_hooks.calls}"
        )

        process_call = next(c for c in span_hooks.calls if c["method_name"] == "process")
        transform_call = next(c for c in span_hooks.calls if c["method_name"] == "transform")
        validate_call = next(c for c in span_hooks.calls if c["method_name"] == "validate")

        # Process (router) should have no parent
        assert process_call["parent_call_id"] is None, (
            f"process should have no parent, got {process_call['parent_call_id']}"
        )

        # BOTH subagent methods should have process as their parent
        assert transform_call["parent_call_id"] == process_call["call_id"], (
            f"transform should have parent={process_call['call_id']}, "
            f"got {transform_call['parent_call_id']}"
        )

        # THIS IS THE BUG: validate might see transform as its parent due to shared mutable list!
        assert validate_call["parent_call_id"] == process_call["call_id"], (
            f"validate should have parent={process_call['call_id']}, "
            f"got {validate_call['parent_call_id']} "
            f"(if this equals transform's call_id {transform_call['call_id']}, "
            f"we have the ContextVar mutable list sharing bug!)"
        )

    @pytest.mark.asyncio
    async def test_parallel_generation_ids_have_same_parent(self, span_hooks):
        """When parent calls gather() with parallel generations, all should have parent as parent_generation_id.

        This tests the same bug pattern as test_parallel_subagent_calls_have_same_parent but for
        the generation_id_stack instead of the agent_call_stack. Both stacks use ContextVars
        with mutable lists and need the same copy-on-write fix.

        Without the fix:
        - Parent (process) pushes its generation_id to stack
        - transform() starts, reads parent_generation_id correctly, pushes its generation_id
        - validate() starts, reads from stack... but might see transform's generation_id!
        """
        fake_llm = FakeLLMClient(
            scripted_responses=[
                # process: creates subagent instances and calls gather
                _resp("""import asyncio

# Create subagent instances (like RouterTestWrapper does)
transformer = self._TransformAgent(llm=self._llm)
validator = self._ValidateAgent(llm=self._llm)

# Call them in parallel
results = await asyncio.gather(
    transformer.transform("input"),
    validator.validate("input"),
)
return f"processed({results[0]}, {results[1]})"
"""),
                # transform response
                _resp('return "transformed"'),
                # validate response
                _resp('return "validated"'),
            ]
        )

        class RouterAgent(Agent, llm=_TEST_LLM):
            # Subagent classes available for instantiation (use module-level classes)
            _TransformAgent = _TransformAgent
            _ValidateAgent = _ValidateAgent

            @strategy(PurePythonStrategy())
            async def process(self, data: str) -> str:
                """Process data using subagents in parallel."""
                ...

        agent = RouterAgent(llm=fake_llm)
        result = await agent.process("test")

        assert result == "processed(transformed, validated)"

        # Filter for only the agent method generations (not internal methods like _build_task_message)
        agent_generations = [
            g
            for g in span_hooks.generations
            if g["method_name"] in ("process", "transform", "validate")
        ]
        assert len(agent_generations) == 3, (
            f"Expected 3 agent generations, got {len(agent_generations)}: {agent_generations}"
        )

        process_gen = next(g for g in agent_generations if g["method_name"] == "process")
        transform_gen = next(g for g in agent_generations if g["method_name"] == "transform")
        validate_gen = next(g for g in agent_generations if g["method_name"] == "validate")

        # Process (router) should have no parent generation
        assert process_gen["parent_generation_id"] is None, (
            f"process should have no parent_generation_id, got {process_gen['parent_generation_id']}"
        )

        # BOTH subagent generations should have process as their parent
        assert transform_gen["parent_generation_id"] == process_gen["generation_id"], (
            f"transform should have parent_generation_id={process_gen['generation_id']}, "
            f"got {transform_gen['parent_generation_id']}"
        )

        # THIS IS THE BUG: validate might see transform as its parent due to shared mutable list!
        assert validate_gen["parent_generation_id"] == process_gen["generation_id"], (
            f"validate should have parent_generation_id={process_gen['generation_id']}, "
            f"got {validate_gen['parent_generation_id']} "
            f"(if this equals transform's generation_id {transform_gen['generation_id']}, "
            f"we have the ContextVar mutable list sharing bug!)"
        )


class TestNoTraceSpanParentRelationship:
    """Test that @no_trace methods are invisible in the span tree.

    A @no_trace method must not appear as a span, and its children must
    attach to the nearest traced ancestor — not become orphaned root spans.
    """

    @pytest.mark.asyncio
    async def test_no_trace_method_invisible_children_attach_to_grandparent(self, span_hooks):
        """A (traced) → B (@no_trace) → C (traced): C's parent_call_id must equal A's call_id.

        Previously, children of @no_trace methods became root spans because the
        @no_trace method's own call_id (for which no span exists) was pushed onto
        the stack. The fix pushes the *parent's* call_id instead so that children
        skip the invisible layer and attach to the nearest traced ancestor.
        """
        fake_llm = FakeLLMClient(
            scripted_responses=[
                # A generates code that calls B
                _resp("result = await self.b()\nreturn result"),
                # B generates code that calls C
                _resp("result = await self.c()\nreturn result"),
                # C generates its result
                _resp("return 'done'"),
            ]
        )

        class ChainAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy())
            async def a(self) -> str:
                """Traced root."""
                ...

            @strategy(PurePythonStrategy())
            @no_trace
            async def b(self) -> str:
                """No-trace middle layer."""
                ...

            @strategy(PurePythonStrategy())
            async def c(self) -> str:
                """Traced leaf."""
                ...

        agent = ChainAgent(llm=fake_llm)
        result = await agent.a()
        assert result == "done"

        # Only A and C should appear — B is invisible
        assert len(span_hooks.calls) == 2, (
            f"Expected 2 spans (a, c), got {len(span_hooks.calls)}: "
            + str([s["method_name"] for s in span_hooks.calls])
        )
        a_call = next(s for s in span_hooks.calls if s["method_name"] == "a")
        c_call = next(s for s in span_hooks.calls if s["method_name"] == "c")

        assert a_call["parent_call_id"] is None, "a should be a root span"
        assert c_call["parent_call_id"] == a_call["call_id"], (
            f"c should be a child of a (call_id={a_call['call_id']}), "
            f"but got parent_call_id={c_call['parent_call_id']}"
        )

    @pytest.mark.asyncio
    async def test_top_level_no_trace_children_are_root_spans(self, span_hooks):
        """When the outermost call is @no_trace with no traced ancestor,
        its children become root spans (parent_call_id=None)."""
        fake_llm = FakeLLMClient(
            scripted_responses=[
                # B (@no_trace, top-level) generates code that calls C
                _resp("result = await self.c()\nreturn result"),
                # C generates its result
                _resp("return 'leaf'"),
            ]
        )

        class TopNoTraceAgent(Agent, llm=_TEST_LLM):
            @strategy(PurePythonStrategy())
            @no_trace
            async def b(self) -> str:
                """No-trace top-level."""
                ...

            @strategy(PurePythonStrategy())
            async def c(self) -> str:
                """Traced child with no traced ancestor."""
                ...

        agent = TopNoTraceAgent(llm=fake_llm)
        result = await agent.b()
        assert result == "leaf"

        # Only C should appear — B is invisible
        assert len(span_hooks.calls) == 1, (
            f"Expected 1 span (c), got {len(span_hooks.calls)}: "
            + str([s["method_name"] for s in span_hooks.calls])
        )
        c_call = span_hooks.calls[0]
        assert c_call["method_name"] == "c"
        assert c_call["parent_call_id"] is None, (
            "c has no traced ancestor so it should be a root span"
        )
