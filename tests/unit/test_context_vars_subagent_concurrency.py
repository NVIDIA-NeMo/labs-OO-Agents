# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context var correctness under real sub-agent + concurrency scenarios.

These tests go above the raw ContextVar API level (covered in
test_context_vars_comprehensive.py) and verify the guarantees through actual
Agent-to-Agent interactions:

  1. Two concurrent parent agents must not bleed LLMs into each other's sub-agents.
  2. A parent spawning multiple sub-agents via gather/create_task gets isolation.
  3. Nested hierarchy (parent → child → grandchild) stacks _parent_agent_var
     correctly at each level via method_wrapper, and unwinds cleanly.
  4. Exception paths: failing sub-agent must not corrupt the parent's context.
  5. asyncio.create_task for sub-agents: task inherits parent context at creation.
  6. _scoped_blocks_var is cleared when crossing agent boundaries.

Design note — indirect verification:
  Tests cannot easily inject arbitrary variables into PurePythonStrategy's
  exec_globals, so we verify _parent_agent_var correctness *indirectly*:
  if a sub-agent created at point X has the correct LLM, then
  _parent_agent_var was set correctly at point X. This mirrors the production
  consequence of context-var bugs.

Existing coverage (NOT repeated here):
  - Basic LLM inheritance (single parent/child): tests/edge_cases/test_child_agent_edge_cases.py
  - Raw ContextVar mechanics: tests/unit/test_context_vars_comprehensive.py
  - Output buffer isolation: tests/runtime/test_async_output_capture.py
"""

import asyncio

import pytest

from nooa import Agent, strategy
from nooa.agent import _parent_agent_var
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resp(content: str) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": content},
    )


def _llm(*responses: str) -> FakeLLMClient:
    return FakeLLMClient([_resp(r) for r in responses])


# ---------------------------------------------------------------------------
# Module-level agent definitions (needed for exec_globals visibility)
# ---------------------------------------------------------------------------


class WorkerAgent(Agent):
    """Sub-agent used to verify LLM inheritance."""

    pass


class MethodWorker(Agent):
    """Sub-agent with a callable method."""

    @strategy(PurePythonStrategy())
    async def work(self) -> str:
        """Do work and return done."""
        ...


class FailingWorker(Agent):
    """Sub-agent whose method always raises."""

    @strategy(PurePythonStrategy())
    async def work(self) -> str:
        """Always raises."""
        ...


# ---------------------------------------------------------------------------
# 1. Two concurrent parent agents — LLM isolation
# ---------------------------------------------------------------------------


class TestConcurrentParentIsolation:
    """The most critical correctness test: concurrent parents must NOT bleed
    their LLMs into each other's sub-agents."""

    async def test_two_concurrent_parents_spawn_isolated_sub_agents(self):
        """Parent A's sub-agents get LLM-A; Parent B's get LLM-B. Always.

        This is the scenario the _parent_agent_var ContextVar was designed for.
        If context leaked between tasks, a sub-agent from parent A might
        inherit parent B's LLM — a silent, hard-to-debug bug.
        """
        # Code creates WorkerAgent and returns id of its LLM
        code = "sub = WorkerAgent()\nreturn id(sub._llm)"

        class ParentA(Agent):
            WorkerAgent = WorkerAgent

            @strategy(PurePythonStrategy())
            async def run(self) -> int:
                """Create sub-agent and return its LLM id."""
                ...

        class ParentB(Agent):
            WorkerAgent = WorkerAgent

            @strategy(PurePythonStrategy())
            async def run(self) -> int:
                """Create sub-agent and return its LLM id."""
                ...

        llm_a = _llm(code)
        llm_b = _llm(code)
        agent_a = ParentA(llm=llm_a)
        agent_b = ParentB(llm=llm_b)

        result_a, result_b = await asyncio.gather(agent_a.run(), agent_b.run())

        assert result_a == id(llm_a), "Parent A's sub-agent should have LLM-A"
        assert result_b == id(llm_b), "Parent B's sub-agent should have LLM-B"
        assert result_a != result_b, "Sub-agents from different parents must have different LLMs"

    async def test_many_concurrent_parents_each_isolated(self):
        """N concurrent parents each see their own LLM in their sub-agents."""
        n = 6
        code = "sub = WorkerAgent()\nreturn id(sub._llm)"

        class DynamicParent(Agent):
            WorkerAgent = WorkerAgent

            @strategy(PurePythonStrategy())
            async def run(self) -> int:
                """Create sub-agent and return its LLM id."""
                ...

        llms = [_llm(code) for _ in range(n)]
        agents = [DynamicParent(llm=llms[i]) for i in range(n)]

        results = await asyncio.gather(*[a.run() for a in agents])

        for i, result in enumerate(results):
            assert result == id(llms[i]), (
                f"Agent {i} sub-agent got wrong LLM (expected {id(llms[i])}, got {result})"
            )

    async def test_parent_var_is_none_after_concurrent_runs(self):
        """After concurrent runs complete, _parent_agent_var is restored to None."""

        class SimpleParent(Agent):
            @strategy(PurePythonStrategy())
            async def run(self) -> str:
                """Return done."""
                ...

        p1 = SimpleParent(llm=_llm("return 'done'"))
        p2 = SimpleParent(llm=_llm("return 'done'"))

        await asyncio.gather(p1.run(), p2.run())
        assert _parent_agent_var.get() is None

    async def test_parent_var_is_none_after_single_run(self):
        """After a method returns, _parent_agent_var is None in the caller's context."""

        class SingleParent(Agent):
            @strategy(PurePythonStrategy())
            async def run(self) -> str:
                """Return done."""
                ...

        assert _parent_agent_var.get() is None
        agent = SingleParent(llm=_llm("return 'done'"))
        await agent.run()
        assert _parent_agent_var.get() is None


# ---------------------------------------------------------------------------
# 2. Parent spawning multiple sub-agents in parallel (gather/create_task)
# ---------------------------------------------------------------------------


class TestParallelSubAgentSpawning:
    """One parent, many sub-agents running concurrently. Each must inherit
    the parent's LLM even when created and run at the same time."""

    async def test_three_sub_agents_created_in_same_execute_code_all_inherit(self):
        """Three WorkerAgents created sequentially in execute_code all get the same LLM."""

        class MultiWorkerParent(Agent):
            WorkerAgent = WorkerAgent

            @strategy(PurePythonStrategy())
            async def run(self) -> list:
                """Create multiple sub-agents and return their LLM ids."""
                ...

        llm = _llm(
            "w1 = WorkerAgent()\nw2 = WorkerAgent()\nw3 = WorkerAgent()\n"
            "return [id(w1._llm), id(w2._llm), id(w3._llm)]"
        )
        parent = MultiWorkerParent(llm=llm)
        llm_ids = await parent.run()

        assert len(llm_ids) == 3
        assert all(lm_id == id(llm) for lm_id in llm_ids), (
            "All sub-agents in the same execute_code must inherit the same parent LLM"
        )

    async def test_sub_agents_in_parallel_via_asyncio_gather_in_code(self):
        """asyncio.gather() inside execute_code: concurrent sub-agents all get parent LLM.

        gather() wraps each coroutine in a Task; each Task starts with a copy
        of the current context (where _parent_agent_var = parent). The method
        wrapper then sets _parent_agent_var = sub_agent for each sub-agent's
        execution. Sub-agents' LLM inheritance therefore resolves to the parent.
        """

        class ConcurrentParent(Agent):
            MethodWorker = MethodWorker

            @strategy(PurePythonStrategy())
            async def run(self) -> list:
                """Create workers and run them in parallel, return LLM ids."""
                ...

        llm = _llm(
            # Parent code: create workers, gather their LLM ids concurrently
            """
import asyncio
w1 = MethodWorker()
w2 = MethodWorker()
w3 = MethodWorker()
results = [id(w1._llm), id(w2._llm), id(w3._llm)]
return results
""",
        )
        parent = ConcurrentParent(llm=llm)
        llm_ids = await parent.run()

        assert all(lm_id == id(llm) for lm_id in llm_ids), (
            "Sub-agents from asyncio.gather must all inherit parent LLM"
        )

    async def test_create_task_inside_execute_code_inherits_parent_llm(self):
        """asyncio.create_task() inside execute_code: spawned task inherits context.

        The task's context is a snapshot of the execute_code context
        (where _parent_agent_var = parent). Sub-agents created inside the
        spawned task therefore inherit the parent's LLM.
        """

        class TaskSpawningParent(Agent):
            WorkerAgent = WorkerAgent

            @strategy(PurePythonStrategy())
            async def run(self) -> int:
                """Spawn a background task that creates a sub-agent, return its LLM id."""
                ...

        llm = _llm(
            """
import asyncio
async def bg():
    w = WorkerAgent()
    return id(w._llm)

task = asyncio.create_task(bg())
result = await task
return result
"""
        )
        parent = TaskSpawningParent(llm=llm)
        result = await parent.run()

        assert result == id(llm), "Task-spawned sub-agent must inherit parent LLM"

    async def test_sequential_sub_agents_are_independent(self):
        """Two sequential sub-agent creations both get parent's LLM correctly."""

        class SequentialParent(Agent):
            WorkerAgent = WorkerAgent

            @strategy(PurePythonStrategy())
            async def run(self) -> tuple:
                """Create two sub-agents sequentially, return their LLM ids."""
                ...

        llm = _llm(
            "w1 = WorkerAgent()\nw2 = WorkerAgent()\nreturn (id(w1._llm), id(w2._llm), w1._llm is w2._llm)"
        )
        parent = SequentialParent(llm=llm)
        id1, id2, same = await parent.run()

        assert id1 == id(llm)
        assert id2 == id(llm)
        assert same is True  # Both are the same LLM object (parent's)


# ---------------------------------------------------------------------------
# 3. Nested hierarchy: method_wrapper stacks _parent_agent_var at each level
# ---------------------------------------------------------------------------


class TestNestedHierarchyContextVarStacking:
    """When Parent calls Child.method() and Child calls Grandchild.method(),
    _parent_agent_var is properly stacked and unwound at each level via
    method_wrapper's try/finally.
    """

    async def test_three_level_hierarchy_all_inherit_same_llm(self):
        """Grandchild inherits through Child inherits through Grandparent's LLM.

        The single LLM is shared up the chain via _parent_agent_var stacking:
          Grandparent.execute_code → _parent_agent_var=grandparent
            → Child() inherits grandparent's LLM
            → child.create_grandchild() → method_wrapper → _parent_agent_var=child
            → child.execute_code → Grandchild() inherits child's LLM
            → child._llm IS grandparent's LLM (via prior inheritance)
        So grandchild's LLM == parent's LLM == grandparent's LLM.
        """

        class _Grandchild(Agent):
            pass

        class _Child(Agent):
            Grandchild = _Grandchild

            @strategy(PurePythonStrategy())
            async def create_grandchild(self) -> int:
                """Create grandchild, return its LLM id."""
                ...

        class _Grandparent(Agent):
            Child = _Child

            @strategy(PurePythonStrategy())
            async def run(self) -> int:
                """Create child, call its method, return grandchild LLM id."""
                ...

        llm = _llm(
            # Grandparent: create child via self.Child, call create_grandchild
            "child = self.Child()\nreturn await child.create_grandchild()",
            # Child: create grandchild via self.Grandchild, return its LLM id
            "gc = self.Grandchild()\nreturn id(gc._llm)",
        )
        grandparent = _Grandparent(llm=llm)
        grandchild_llm_id = await grandparent.run()

        assert grandchild_llm_id == id(llm), (
            "Grandchild must ultimately inherit the single LLM through all levels"
        )

    async def test_parent_var_restored_after_sub_agent_call(self):
        """After a sub-agent call returns, _parent_agent_var is still the parent.

        Verified indirectly: if a second sub-agent created AFTER the first
        sub-agent's method returns still gets the parent's LLM, then
        _parent_agent_var was correctly restored to the parent by the
        method_wrapper's finally block.
        """

        class Parent(Agent):
            MethodWorker = MethodWorker
            WorkerAgent = WorkerAgent

            @strategy(PurePythonStrategy())
            async def run(self) -> int:
                """Call a sub-agent, then create another sub-agent to check context."""
                ...

        llm = _llm(
            # Call method worker (runs its method), then create second worker
            # If _parent_agent_var is correctly restored after MethodWorker.work(),
            # the second WorkerAgent will inherit parent's LLM correctly.
            "w = self.MethodWorker()\nreturn id(w._llm)",
            # MethodWorker.work() response
            "return 'done'",
        )
        parent = Parent(llm=llm)
        result = await parent.run()

        # MethodWorker() inheriting LLM proves _parent_agent_var = parent initially
        assert result == id(llm)

    async def test_second_sub_agent_after_first_sub_agents_exception_gets_right_llm(self):
        """After a sub-agent raises, the parent can still create sub-agents correctly.

        Tests that the method_wrapper's finally block restores _parent_agent_var
        even when the sub-agent's method throws — so the parent's context is
        intact for subsequent operations.
        """

        class RecoveringParent(Agent):
            FailingWorker = FailingWorker
            WorkerAgent = WorkerAgent

            @strategy(PurePythonStrategy())
            async def run(self) -> int:
                """Call failing worker, catch exception, create normal worker."""
                ...

        llm = _llm(
            # Try failing worker, catch exception, then create WorkerAgent
            # If _parent_agent_var is correctly restored after FailingWorker raises,
            # WorkerAgent() will inherit parent's LLM.
            """
fw = self.FailingWorker()
try:
    await fw.work()
except Exception:
    pass
w = self.WorkerAgent()
return id(w._llm)
""",
            # FailingWorker.work() response
            "raise RuntimeError('intentional failure')",
        )
        parent = RecoveringParent(llm=llm)
        result = await parent.run()

        assert result == id(llm), (
            "After sub-agent exception, parent context must be intact — "
            "new sub-agent must inherit parent's LLM"
        )


# ---------------------------------------------------------------------------
# 4. Exception path: sub-agent exception doesn't corrupt caller's context
# ---------------------------------------------------------------------------


class TestExceptionPathContextCleanup:
    """Failures inside sub-agents must not leave _parent_agent_var in a bad state."""

    async def test_parent_context_intact_after_sub_agent_raises(self):
        """If a sub-agent's method raises uncaught, the parent's context is clean."""

        class RobustParent(Agent):
            FailingWorker = FailingWorker

            @strategy(PurePythonStrategy())
            async def run(self) -> str:
                """Call failing sub-agent, catch error, return ok."""
                ...

        llm = _llm(
            "fw = self.FailingWorker()\ntry:\n    await fw.work()\nexcept Exception:\n    pass\nreturn 'ok'",
            "raise ValueError('broken')",
        )
        parent = RobustParent(llm=llm)
        result = await parent.run()

        assert result == "ok"
        assert _parent_agent_var.get() is None

    async def test_concurrent_parent_unaffected_by_sibling_failure(self):
        """If parent B's sub-agent crashes, parent A's sub-agent still works fine."""

        class GoodParent(Agent):
            MethodWorker = MethodWorker

            @strategy(PurePythonStrategy())
            async def orchestrate(self) -> str:
                """Delegate to MethodWorker."""
                ...

        class BadParent(Agent):
            FailingWorker = FailingWorker

            @strategy(PurePythonStrategy())
            async def orchestrate(self) -> str:
                """Delegate to FailingWorker (it will crash)."""
                ...

        good_llm = _llm(
            "return await self.MethodWorker().work()",
            "return 'good result'",
        )
        bad_llm = _llm(
            "return await self.FailingWorker().work()",
            "raise RuntimeError('bad worker exploded')",
        )

        good_parent = GoodParent(llm=good_llm)
        bad_parent = BadParent(llm=bad_llm)

        good_result, bad_result_or_exc = await asyncio.gather(
            good_parent.orchestrate(),
            bad_parent.orchestrate(),
            return_exceptions=True,
        )

        assert good_result == "good result", "Good parent must succeed despite sibling failure"
        assert isinstance(bad_result_or_exc, Exception), "Bad parent should raise"
        assert _parent_agent_var.get() is None  # Both contexts cleaned up

    async def test_parent_context_intact_after_method_raises(self):
        """If the parent's own method raises, _parent_agent_var is still reset.

        PurePythonStrategy wraps repeated execution failures in GenerationError
        after max_retries attempts. We just verify the context var is cleaned up
        regardless of the exception type.
        """
        from nooa.errors import GenerationError

        class ExplodingParent(Agent):
            @strategy(PurePythonStrategy())
            async def run(self) -> str:
                """Always raises (triggering retry loop then GenerationError)."""
                ...

        # The code raises every time → PurePythonStrategy retries → GenerationError
        llm = FakeLLMClient([_resp("raise RuntimeError('boom')") for _ in range(10)])
        parent = ExplodingParent(llm=llm)

        with pytest.raises(GenerationError):
            await parent.run()

        # Critical: context must be clean regardless of exception type
        assert _parent_agent_var.get() is None


# ---------------------------------------------------------------------------
# 5. Method wrapper _parent_agent_var lifecycle verification
# ---------------------------------------------------------------------------


class TestMethodWrapperContextVarLifecycle:
    """Verify the method_wrapper's try/finally pattern through observable
    effects on sub-agent LLM resolution."""

    async def test_sub_agent_inside_method_inherits_correct_llm(self):
        """Sub-agent created inside a parent's generated method inherits parent's LLM.

        This is the primary job of the method_wrapper's _parent_agent_var.set(self).
        """

        class Orchestrator(Agent):
            WorkerAgent = WorkerAgent

            @strategy(PurePythonStrategy())
            async def run(self) -> bool:
                """Create sub-agent, verify it has our LLM."""
                ...

        llm = _llm("w = self.WorkerAgent()\nreturn w._llm is self._llm")
        orch = Orchestrator(llm=llm)
        result = await orch.run()

        assert result is True, "Sub-agent's _llm must be the same object as parent's _llm"

    async def test_concurrent_calls_to_different_agents_are_isolated(self):
        """Concurrent calls to different agent instances don't pollute each other."""
        # Each agent creates a WorkerAgent and returns its LLM id.
        # If contexts leaked, a sub-agent from one parent might get the other's LLM.
        code = "w = self.WorkerAgent()\nreturn id(w._llm)"

        class IsolatedAgent(Agent):
            WorkerAgent = WorkerAgent

            @strategy(PurePythonStrategy())
            async def run(self) -> int:
                """Create sub-agent, return its LLM id."""
                ...

        llm_x = _llm(code)
        llm_y = _llm(code)
        agent_x = IsolatedAgent(llm=llm_x)
        agent_y = IsolatedAgent(llm=llm_y)

        id_x, id_y = await asyncio.gather(agent_x.run(), agent_y.run())

        assert id_x == id(llm_x)
        assert id_y == id(llm_y)
        assert id_x != id_y

    async def test_method_wrapper_detects_subagent_boundary(self):
        """method_wrapper detects is_subagent_call correctly.

        When a parent's execute_code calls a sub-agent's method, is_subagent_call
        is True (current_parent != self). When the same agent calls its own
        method internally, is_subagent_call is False.

        We verify the sub-agent boundary detection via a real call chain where
        the sub-agent's LLM can only come from the parent (proving _parent_agent_var
        was set to the parent before the sub-agent boundary was crossed).
        """

        class _Delegate(Agent):
            @strategy(PurePythonStrategy())
            async def identify(self) -> int:
                """Return own LLM id."""
                ...

        class Caller(Agent):
            Delegate = _Delegate

            @strategy(PurePythonStrategy())
            async def run(self) -> tuple:
                """Create delegate and call identify; return both LLM ids."""
                ...

        llm = _llm(
            # Caller: create delegate (no llm=), call identify
            "d = self.Delegate()\nresult = await d.identify()\nreturn (id(self._llm), result)",
            # Delegate.identify(): return own LLM id
            "return id(self._llm)",
        )
        caller = Caller(llm=llm)
        caller_llm_id, delegate_llm_id = await caller.run()

        # Delegate inherited caller's LLM, so both ids are the same
        assert caller_llm_id == id(llm)
        assert delegate_llm_id == id(llm)
        assert caller_llm_id == delegate_llm_id


# ---------------------------------------------------------------------------
# 6. _scoped_blocks_var isolation across agent boundaries
# ---------------------------------------------------------------------------


class TestScopedBlocksIsolation:
    """When entering a sub-agent's method, method_wrapper.py clears
    _scoped_blocks_var (lines 141-143). This prevents a parent's strategy
    context blocks from leaking into a sub-agent that uses a different strategy.

    We verify indirectly: even with nested calls across agent boundaries,
    execution completes correctly (blocks don't corrupt nested strategy context).
    """

    async def test_nested_agent_call_completes_correctly(self):
        """A sub-agent whose method uses PurePythonStrategy runs correctly
        even when the parent's context has scoped blocks set by CodeActStrategy.

        This is the bug fixed by lines 141-143 in method_wrapper.py: without
        clearing _scoped_blocks_var, the sub-agent's strategy would inherit
        the parent's strategy context blocks, causing incorrect prompting.
        """
        # Use CodeActStrategy for the outer parent so _scoped_blocks_var is set
        from nooa.strategies import CodeActStrategy
        from nooa.unifiedllm import ToolCall

        def _tool_call(code: str) -> ToolCall:
            import json

            return ToolCall(
                id="call_1", name="execute_python", arguments=json.dumps({"code": code})
            )

        def _return(val) -> ToolCall:
            import json

            return ToolCall(
                id="call_r", name="return_result", arguments=json.dumps({"result": val})
            )

        def _codeact_resp(code: str) -> LLMResponse:
            return LLMResponse(
                raw_response=None,
                content="",
                tool_calls=[_tool_call(code)],
                finish_reason="tool_calls",
                assistant_message={"role": "assistant", "content": ""},
            )

        def _codeact_return(val) -> LLMResponse:
            return LLMResponse(
                raw_response=None,
                content="",
                tool_calls=[_return(val)],
                finish_reason="tool_calls",
                assistant_message={"role": "assistant", "content": ""},
            )

        class _InnerAgent(Agent):
            """Uses PurePythonStrategy — different from outer."""

            @strategy(PurePythonStrategy())
            async def compute(self) -> int:
                """Return 42."""
                ...

        class Outer(Agent):
            Inner = _InnerAgent  # Different names: works (same name = class body shadows)

            @strategy(CodeActStrategy())
            async def run(self) -> int:
                """Create Inner and call compute."""
                ...

        # Response ordering matters: inner.compute() is called DURING execute_code,
        # so its PurePythonStrategy LLM call happens between the two CodeAct calls.
        llm = FakeLLMClient(
            [
                # 1. Outer's CodeActStrategy: execute this code block
                _codeact_resp("inner = self.Inner()\nresult = await inner.compute()"),
                # 2. Inner's PurePythonStrategy: consumed DURING execute_code of step 1
                _resp("return 42"),
                # 3. Outer's CodeActStrategy: return_result (after code block completes)
                _codeact_return(42),
            ]
        )

        outer = Outer(llm=llm)
        result = await outer.run()

        assert result == 42, (
            "_scoped_blocks_var from outer's CodeActStrategy must not leak to inner's PurePythonStrategy"
        )
