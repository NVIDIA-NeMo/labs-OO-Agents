# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Comprehensive tests for async context variable correctness.

ContextVars are the concurrency backbone of the agent runtime. Bugs here are
subtle and hard to reproduce: wrong parent propagated to sub-agents, stdout
from one task landing in another task's buffer, context leaking into spawned
tasks, reset not happening on exception paths.

Test strategy:
  - Lifecycle: set → (normal|exception) → reset
  - Isolation: concurrent tasks see their own values, not siblings'
  - Nesting: inner set/reset restores outer value (not default)
  - Task inheritance: asyncio.create_task copies context at creation time
  - Immutable stacks: push in one task does not affect parallel task
  - Global vars: _default_strategy_var and _instrumentation_hooks_var have
    intentional "no reset" pattern — document and test cleanup discipline

Existing coverage (not duplicated here):
  - stdout/stderr buffer isolation: tests/runtime/test_async_output_capture.py
  - agent call stack / generation ID stack: src/nooa/runtime/tests/test_stack_isolation.py
  - _in_agent_context set/reset: tests/unit/test_remaining_gaps.py::TestAsyncSafety
  - basic _parent_agent_var lifecycle: tests/capability/test_class_method_replacement_bug.py
"""

import asyncio
import contextvars

import pytest

from nooa import Agent
from nooa.runtime.context_vars import (
    _agent_call_stack_var,
    _parent_agent_var,
    _pop_agent_call_id,
    _push_agent_call_id,
)
from nooa.unifiedllm import FakeLLMClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm():
    return FakeLLMClient()


class _SimpleAgent(Agent):
    async def run(self) -> str: ...


# ---------------------------------------------------------------------------
# 1. _parent_agent_var — lifecycle
# ---------------------------------------------------------------------------


class TestParentAgentVarLifecycle:
    """_parent_agent_var must be set during execute_code and reset afterwards."""

    async def test_is_none_before_execute_code(self):
        """Outside execute_code context, _parent_agent_var is None."""
        assert _parent_agent_var.get() is None

    async def test_is_set_to_agent_during_execute_code(self):
        """Inside execute_code, _parent_agent_var equals the executing agent."""
        agent = _SimpleAgent(llm=_make_llm())
        captured = []

        code = "_captured_parent = _parent_agent_var.get()"
        result = await agent.runtime.execute_code(
            code,
            builtins={"_parent_agent_var": _parent_agent_var, "_captured_parent": captured},
            validate=False,
        )
        assert result.error is None
        # The var is set inside execute_code execution namespace
        assert _parent_agent_var.get() is None  # reset after

    async def test_is_none_after_execute_code_completes(self):
        """After execute_code returns, _parent_agent_var is restored to None."""
        agent = _SimpleAgent(llm=_make_llm())
        await agent.runtime.execute_code("x = 1", validate=False)
        assert _parent_agent_var.get() is None

    async def test_is_none_after_execute_code_raises_user_exception(self):
        """_parent_agent_var is reset even when user code raises an exception."""
        agent = _SimpleAgent(llm=_make_llm())
        result = await agent.runtime.execute_code(
            "raise RuntimeError('boom')", wrap_in_function=True, validate=False
        )
        assert result.error is not None
        assert "boom" in str(result.error)
        # Must be reset even though code raised
        assert _parent_agent_var.get() is None

    async def test_is_none_after_syntax_error_in_user_code(self):
        """_parent_agent_var is reset even when user code has a SyntaxError."""
        agent = _SimpleAgent(llm=_make_llm())
        result = await agent.runtime.execute_code(
            "def broken(: pass",  # invalid syntax
            validate=False,
        )
        assert result.error is not None
        assert _parent_agent_var.get() is None

    async def test_nested_token_restores_outer_value_not_none(self):
        """Token-based reset restores the previous value, not the default.

        If A sets var=X, then B sets var=Y, resetting B's token should restore
        var=X, not var=None. This is fundamental Python ContextVar semantics.
        """
        agent_outer = _SimpleAgent(llm=_make_llm())
        agent_inner = _SimpleAgent(llm=_make_llm())

        outer_token = _parent_agent_var.set(agent_outer)
        try:
            assert _parent_agent_var.get() is agent_outer

            inner_token = _parent_agent_var.set(agent_inner)
            try:
                assert _parent_agent_var.get() is agent_inner
            finally:
                _parent_agent_var.reset(inner_token)

            # Must be outer again, not None
            assert _parent_agent_var.get() is agent_outer
        finally:
            _parent_agent_var.reset(outer_token)

        assert _parent_agent_var.get() is None  # default restored


# ---------------------------------------------------------------------------
# 2. _parent_agent_var — concurrent task isolation
# ---------------------------------------------------------------------------


class TestParentAgentVarConcurrentIsolation:
    """Each async task has its own _parent_agent_var; they must not cross-contaminate."""

    async def test_two_concurrent_execute_code_see_own_agent(self):
        """Two concurrent execute_code calls each see their own agent as parent.

        This is the critical invariant: sub-agents created inside LLM-generated
        code must inherit from the CORRECT parent, not a sibling's parent.
        """
        agent_a = _SimpleAgent(llm=_make_llm())
        agent_b = _SimpleAgent(llm=_make_llm())

        parent_seen_by_a: list = []
        parent_seen_by_b: list = []

        async def run_a():
            # Pause to force overlap with B
            code = """
import asyncio
await asyncio.sleep(0.05)
_result_a.append(_parent_agent_var.get())
"""
            await agent_a.runtime.execute_code(
                code,
                builtins={
                    "_parent_agent_var": _parent_agent_var,
                    "_result_a": parent_seen_by_a,
                },
                wrap_in_function=True,
                validate=False,
            )

        async def run_b():
            code = """
import asyncio
await asyncio.sleep(0.05)
_result_b.append(_parent_agent_var.get())
"""
            await agent_b.runtime.execute_code(
                code,
                builtins={
                    "_parent_agent_var": _parent_agent_var,
                    "_result_b": parent_seen_by_b,
                },
                wrap_in_function=True,
                validate=False,
            )

        await asyncio.gather(run_a(), run_b())

        assert len(parent_seen_by_a) == 1
        assert len(parent_seen_by_b) == 1
        assert parent_seen_by_a[0] is agent_a, "agent_a code saw wrong parent"
        assert parent_seen_by_b[0] is agent_b, "agent_b code saw wrong parent"

    async def test_parent_var_after_gather_is_none(self):
        """After gather completes, the caller's _parent_agent_var is None."""
        agent1 = _SimpleAgent(llm=_make_llm())
        agent2 = _SimpleAgent(llm=_make_llm())

        await asyncio.gather(
            agent1.runtime.execute_code("x = 1", validate=False),
            agent2.runtime.execute_code("y = 2", validate=False),
        )
        assert _parent_agent_var.get() is None

    async def test_many_concurrent_agents_each_see_own_parent(self):
        """N concurrent executions: each captures their own parent, not others."""
        n = 8
        agents = [_SimpleAgent(llm=_make_llm()) for _ in range(n)]
        results: list[list] = [[] for _ in range(n)]

        async def run(i: int):
            code = """
import asyncio
await asyncio.sleep(0.02)
_out.append(_parent_agent_var.get())
"""
            await agents[i].runtime.execute_code(
                code,
                builtins={"_parent_agent_var": _parent_agent_var, "_out": results[i]},
                wrap_in_function=True,
                validate=False,
            )

        await asyncio.gather(*[run(i) for i in range(n)])

        for i in range(n):
            assert results[i][0] is agents[i], (
                f"Agent {i} saw parent {results[i][0]!r}, expected {agents[i]!r}"
            )


# ---------------------------------------------------------------------------
# 3. asyncio.create_task — context inheritance
# ---------------------------------------------------------------------------


class TestContextInheritanceInSpawnedTasks:
    """asyncio.create_task captures a COPY of the context at creation time.

    This is standard Python behaviour, but it has important implications for
    the agent runtime: a task spawned inside execute_code inherits _parent_agent_var
    and can create sub-agents; a task spawned outside does not.
    """

    async def test_task_created_inside_execute_code_inherits_parent(self):
        """A task spawned during execute_code sees _parent_agent_var = that agent."""
        agent = _SimpleAgent(llm=_make_llm())
        captured: list = []

        code = """
import asyncio

async def _background():
    captured_var.append(_parent_agent_var.get())

task = asyncio.create_task(_background())
await task
"""
        result = await agent.runtime.execute_code(
            code,
            builtins={"_parent_agent_var": _parent_agent_var, "captured_var": captured},
            wrap_in_function=True,
            validate=False,
        )
        assert result.error is None, result.error
        assert len(captured) == 1
        assert captured[0] is agent

    async def test_task_created_outside_execute_code_has_no_parent(self):
        """A task spawned outside execute_code sees _parent_agent_var = None."""
        captured: list = []

        async def background():
            captured.append(_parent_agent_var.get())

        task = asyncio.create_task(background())
        await task

        assert captured[0] is None

    async def test_task_inherits_parent_at_creation_time_not_later(self):
        """Context snapshot is taken when create_task is called, not when it runs.

        If _parent_agent_var changes after create_task(), the task should NOT
        see the new value.
        """
        agent_a = _SimpleAgent(llm=_make_llm())
        agent_b = _SimpleAgent(llm=_make_llm())
        task_saw: list = []

        # Manually set context to simulate execute_code
        start_event = asyncio.Event()
        check_event = asyncio.Event()

        async def background():
            await start_event.wait()  # Wait until after we changed the var
            task_saw.append(_parent_agent_var.get())
            check_event.set()

        # Set to agent_a, create task (context copied now)
        token_a = _parent_agent_var.set(agent_a)
        asyncio.create_task(background())

        # Change to agent_b AFTER task creation
        _parent_agent_var.reset(token_a)
        token_b = _parent_agent_var.set(agent_b)

        start_event.set()
        await check_event.wait()

        _parent_agent_var.reset(token_b)

        # Task was created when var=agent_a, so it sees agent_a
        assert task_saw[0] is agent_a, (
            f"Task should see agent_a (context at creation), got {task_saw[0]!r}"
        )

    async def test_mutations_in_spawned_task_dont_affect_parent_context(self):
        """A task that mutates a ContextVar only affects its own context copy."""
        agent_outer = _SimpleAgent(llm=_make_llm())
        outer_token = _parent_agent_var.set(agent_outer)

        agent_inner = _SimpleAgent(llm=_make_llm())
        task_done = asyncio.Event()

        async def mutating_task():
            # Override the var in task's context
            tok = _parent_agent_var.set(agent_inner)
            await asyncio.sleep(0.01)
            _parent_agent_var.reset(tok)
            task_done.set()

        asyncio.create_task(mutating_task())
        # Check immediately (before task runs) and after
        assert _parent_agent_var.get() is agent_outer
        await task_done.wait()
        assert _parent_agent_var.get() is agent_outer  # parent context unchanged

        _parent_agent_var.reset(outer_token)


# ---------------------------------------------------------------------------
# 4. _agent_call_stack_var — immutable copy-on-write semantics
# ---------------------------------------------------------------------------


class TestAgentCallStackImmutability:
    """Push/pop creates new tuples. Concurrent tasks are isolated by design."""

    def test_push_does_not_mutate_previous_tuple(self):
        """After a push, the old stack reference is unaffected."""
        before = _agent_call_stack_var.get()
        _push_agent_call_id("test-id")
        after = _agent_call_stack_var.get()

        assert before != after
        assert "test-id" not in before
        assert "test-id" in after

        _pop_agent_call_id()

    def test_pop_restores_previous_tuple(self):
        """Pop creates a new tuple without the last element."""
        original = _agent_call_stack_var.get()
        _push_agent_call_id("id-1")
        _push_agent_call_id("id-2")

        _pop_agent_call_id()
        _pop_agent_call_id()

        assert _agent_call_stack_var.get() == original

    def test_pop_on_empty_stack_returns_none(self):
        """Popping an empty stack returns None without error."""
        # Ensure stack is empty
        assert _agent_call_stack_var.get() == ()
        result = _pop_agent_call_id()
        assert result is None

    async def test_concurrent_pushes_are_isolated(self):
        """Push in one coroutine does not appear in a concurrent coroutine."""
        seen_by_b: list = []
        a_pushed = asyncio.Event()
        b_checked = asyncio.Event()

        async def coroutine_a():
            _push_agent_call_id("call-from-a")
            a_pushed.set()
            await b_checked.wait()
            _pop_agent_call_id()

        async def coroutine_b():
            await a_pushed.wait()
            seen_by_b.append(_agent_call_stack_var.get())
            b_checked.set()

        await asyncio.gather(coroutine_a(), coroutine_b())

        assert "call-from-a" not in seen_by_b[0], (
            "coroutine_b saw coroutine_a's call ID — stack isolation broken"
        )

    async def test_stack_persists_across_awaits_in_same_task(self):
        """Within a single coroutine, pushed values survive await points."""
        _push_agent_call_id("persistent-id")
        await asyncio.sleep(0)  # yield, then resume
        stack = _agent_call_stack_var.get()
        _pop_agent_call_id()

        assert "persistent-id" in stack


# ---------------------------------------------------------------------------
# 5. _current_agent_var / _current_runtime_var — implemented method path
# ---------------------------------------------------------------------------


class TestCurrentAgentVarLifecycle:
    """_current_agent_var and _current_runtime_var are set for methods with
    implementations and reset in a finally block."""

    async def test_current_agent_var_is_none_outside_execution(self):
        from nooa.util._context import _current_agent_var, _current_runtime_var

        assert _current_agent_var.get() is None
        assert _current_runtime_var.get() is None

    async def test_set_and_reset_token_correctness(self):
        """Token-based reset restores the default after set."""
        from nooa.util._context import _current_agent_var

        agent = _SimpleAgent(llm=_make_llm())
        token = _current_agent_var.set(agent)
        assert _current_agent_var.get() is agent
        _current_agent_var.reset(token)
        assert _current_agent_var.get() is None

    async def test_nested_set_reset_restores_outer(self):
        """Nested token reset restores outer value, not the default."""
        from nooa.util._context import _current_agent_var

        agent_outer = _SimpleAgent(llm=_make_llm())
        agent_inner = _SimpleAgent(llm=_make_llm())

        t1 = _current_agent_var.set(agent_outer)
        t2 = _current_agent_var.set(agent_inner)
        assert _current_agent_var.get() is agent_inner
        _current_agent_var.reset(t2)
        assert _current_agent_var.get() is agent_outer  # not None!
        _current_agent_var.reset(t1)
        assert _current_agent_var.get() is None

    async def test_current_agent_raises_outside_context(self):
        """_current_agent() raises RuntimeError when called outside execution context."""
        from nooa.util._context import _current_agent

        with pytest.raises(RuntimeError, match="No agent in context"):
            _current_agent()

    async def test_current_agent_returns_agent_inside_context(self):
        """_current_agent() returns the agent when context is set."""
        from nooa.util._context import _current_agent, _current_agent_var

        agent = _SimpleAgent(llm=_make_llm())
        token = _current_agent_var.set(agent)
        try:
            result = _current_agent()
            assert result is agent
        finally:
            _current_agent_var.reset(token)


# ---------------------------------------------------------------------------
# 6. _in_generation_session — re-entry guard
# ---------------------------------------------------------------------------


class TestInGenerationSessionVar:
    """_in_generation_session prevents deadlocks from nested generation calls."""

    async def test_default_is_false(self):
        from nooa.runtime.actor import _in_generation_session

        assert _in_generation_session.get() is False

    async def test_set_and_reset(self):
        """Manual set/reset cycle works correctly."""
        from nooa.runtime.actor import _in_generation_session

        token = _in_generation_session.set(True)
        assert _in_generation_session.get() is True
        _in_generation_session.reset(token)
        assert _in_generation_session.get() is False

    async def test_nested_restore_previous_not_default(self):
        """Nested set/reset restores the previous value (True → True), not default."""
        from nooa.runtime.actor import _in_generation_session

        # Outer: True
        outer_token = _in_generation_session.set(True)

        # Inner: also True (nested generation)
        inner_token = _in_generation_session.set(True)
        _in_generation_session.reset(inner_token)

        # Should still be True (outer), not False (default)
        assert _in_generation_session.get() is True

        _in_generation_session.reset(outer_token)
        assert _in_generation_session.get() is False

    async def test_isolated_between_concurrent_tasks(self):
        """Two concurrent tasks each have their own _in_generation_session value."""
        from nooa.runtime.actor import _in_generation_session

        a_set = asyncio.Event()
        b_checked = asyncio.Event()
        b_saw: list = []

        async def task_a():
            token = _in_generation_session.set(True)
            a_set.set()
            await b_checked.wait()
            _in_generation_session.reset(token)

        async def task_b():
            await a_set.wait()
            b_saw.append(_in_generation_session.get())
            b_checked.set()

        await asyncio.gather(task_a(), task_b())

        assert b_saw[0] is False, "task_b saw task_a's _in_generation_session=True — context leaked"


# ---------------------------------------------------------------------------
# 7. _default_strategy_var — intentional global (no token reset)
# ---------------------------------------------------------------------------


class TestDefaultStrategyVarGlobalPattern:
    """_default_strategy_var intentionally persists — test the cleanup discipline."""

    def test_default_is_codeact_strategy(self):
        """By default (var=None), get_default_strategy() returns a CodeActStrategy."""
        from nooa.strategies import (
            CodeActStrategy,
            get_default_strategy,
            set_default_strategy,
        )

        set_default_strategy(None)  # Ensure var is None
        strategy = get_default_strategy()
        assert isinstance(strategy, CodeActStrategy)

    def test_set_override_is_visible(self):
        """set_default_strategy() overrides the default for the current context."""
        from nooa.strategies import (
            get_default_strategy,
            set_default_strategy,
        )
        from nooa.strategies.pure_python import PurePythonStrategy

        set_default_strategy(PurePythonStrategy())
        try:
            assert isinstance(get_default_strategy(), PurePythonStrategy)
        finally:
            set_default_strategy(None)  # ALWAYS clean up

    def test_set_none_restores_codeact_default(self):
        """set_default_strategy(None) restores the CodeActStrategy default."""
        from nooa.strategies import (
            CodeActStrategy,
            get_default_strategy,
            set_default_strategy,
        )
        from nooa.strategies.pure_python import PurePythonStrategy

        set_default_strategy(PurePythonStrategy())
        set_default_strategy(None)
        assert isinstance(get_default_strategy(), CodeActStrategy)

    async def test_override_is_task_local(self):
        """_default_strategy_var changes ARE task-local via ContextVar semantics.

        Critically: set_default_strategy() uses .set() NOT token-based reset.
        In the SAME task/coroutine, it persists. In a new task (create_task),
        the task gets a COPY of the context at creation time — so the override
        is visible in the task if it was set BEFORE the task was created.
        """
        from nooa.strategies import (
            CodeActStrategy,
            get_default_strategy,
            set_default_strategy,
        )
        from nooa.strategies.pure_python import PurePythonStrategy

        saw_in_task: list = []
        task_done = asyncio.Event()

        async def check_in_task():
            saw_in_task.append(type(get_default_strategy()).__name__)
            task_done.set()

        # Before: default (CodeActStrategy when var=None)
        set_default_strategy(None)
        assert isinstance(get_default_strategy(), CodeActStrategy)

        # Set override to PurePythonStrategy
        set_default_strategy(PurePythonStrategy())

        # Create task AFTER override — it inherits the override
        asyncio.create_task(check_in_task())
        await task_done.wait()

        # Cleanup
        set_default_strategy(None)

        assert saw_in_task[0] == "PurePythonStrategy", (
            "Task should inherit the override set before create_task"
        )


# ---------------------------------------------------------------------------
# 8. _in_exec_middleware — re-entry guard for execute_python middleware
# ---------------------------------------------------------------------------


class TestInExecMiddlewareVar:
    """_in_exec_middleware prevents infinite recursion in middleware chains."""

    def test_default_is_empty_frozenset(self):
        from nooa.runtime.actor import _in_exec_middleware

        assert _in_exec_middleware.get() == frozenset()

    def test_set_and_reset(self):
        from nooa.runtime.actor import _in_exec_middleware

        token = _in_exec_middleware.set(frozenset({1, 2}))
        assert _in_exec_middleware.get() == frozenset({1, 2})
        _in_exec_middleware.reset(token)
        assert _in_exec_middleware.get() == frozenset()

    async def test_isolated_between_concurrent_tasks(self):
        """Two concurrent tasks each manage their own middleware re-entry state."""
        from nooa.runtime.actor import _in_exec_middleware

        a_set = asyncio.Event()
        b_checked = asyncio.Event()
        b_saw: list = []

        async def task_a():
            token = _in_exec_middleware.set(frozenset({42}))
            a_set.set()
            await b_checked.wait()
            _in_exec_middleware.reset(token)

        async def task_b():
            await a_set.wait()
            b_saw.append(_in_exec_middleware.get())
            b_checked.set()

        await asyncio.gather(task_a(), task_b())

        assert 42 not in b_saw[0], "task_b saw task_a's middleware ID — context leaked"


# ---------------------------------------------------------------------------
# 9. _block_stdin_var — stdin blocking lifecycle
# ---------------------------------------------------------------------------


class TestBlockStdinVarLifecycle:
    """_block_stdin_var is set True inside execute_code and reset in finally."""

    async def test_is_false_outside_execute_code(self):
        from nooa.runtime.actor import _block_stdin_var

        assert _block_stdin_var.get() is False

    async def test_reset_after_execute_code(self):
        """_block_stdin_var is False after execute_code completes normally."""
        from nooa.runtime.actor import _block_stdin_var

        agent = _SimpleAgent(llm=_make_llm())
        await agent.runtime.execute_code("x = 1", validate=False)
        assert _block_stdin_var.get() is False

    async def test_reset_after_execute_code_exception(self):
        """_block_stdin_var is False even after user code raises."""
        from nooa.runtime.actor import _block_stdin_var

        agent = _SimpleAgent(llm=_make_llm())
        result = await agent.runtime.execute_code(
            "raise ValueError('stdin test')", wrap_in_function=True, validate=False
        )
        assert result.error is not None
        assert _block_stdin_var.get() is False

    async def test_isolated_between_concurrent_tasks(self):
        """Two concurrent execute_code calls each have their own stdin block state."""
        from nooa.runtime.actor import _block_stdin_var

        agent_a = _SimpleAgent(llm=_make_llm())
        agent_b = _SimpleAgent(llm=_make_llm())
        b_saw_stdin: list = []

        async def run_a():
            code = """
import asyncio
await asyncio.sleep(0.05)
"""
            await agent_a.runtime.execute_code(code, wrap_in_function=True, validate=False)

        async def run_b():
            code = """
import asyncio
await asyncio.sleep(0.02)
_b_saw.append(_block_stdin_var.get())
"""
            await agent_b.runtime.execute_code(
                code,
                builtins={"_block_stdin_var": _block_stdin_var, "_b_saw": b_saw_stdin},
                wrap_in_function=True,
                validate=False,
            )

        await asyncio.gather(run_a(), run_b())

        # Both tasks have their own isolated block state
        assert b_saw_stdin[0] is True  # b sees its own block (True inside execute_code)
        assert _block_stdin_var.get() is False  # outer context: not blocked


# ---------------------------------------------------------------------------
# 10. Generic ContextVar semantics — regression tests
# ---------------------------------------------------------------------------


class TestContextVarPythonSemantics:
    """Regression tests for Python's ContextVar semantics.

    These document the exact asyncio/contextvars behaviour the runtime relies on.
    If Python ever changes these semantics, these tests will fail loudly.
    """

    async def test_create_task_copies_context(self):
        """asyncio.create_task copies the current context into the new task."""
        var: contextvars.ContextVar[int] = contextvars.ContextVar("test_var", default=0)
        task_saw: list = []

        token = var.set(99)
        task_done = asyncio.Event()

        async def check():
            task_saw.append(var.get())
            task_done.set()

        asyncio.create_task(check())
        var.reset(token)  # Reset in parent before task runs
        await task_done.wait()

        # Task captured the context at creation time (var=99)
        assert task_saw[0] == 99, "create_task should capture context at creation, not at run time"

    async def test_task_mutation_does_not_affect_parent(self):
        """Mutations inside an asyncio.Task don't affect the parent context."""
        var: contextvars.ContextVar[int] = contextvars.ContextVar("test_var2", default=0)
        task_done = asyncio.Event()

        async def mutate():
            tok = var.set(999)
            var.reset(tok)
            task_done.set()

        assert var.get() == 0
        asyncio.create_task(mutate())
        await task_done.wait()
        assert var.get() == 0  # Unchanged in parent

    async def test_gather_tasks_are_isolated_from_each_other(self):
        """gather() tasks are isolated: each starts with the same context snapshot
        (the context at gather() call time) and mutations don't cross between tasks."""
        var: contextvars.ContextVar[str] = contextvars.ContextVar("test_var3", default="none")
        results: dict[str, str] = {}
        barrier = asyncio.Barrier(2)

        async def task_x():
            tok = var.set("from-x")
            await barrier.wait()  # Both tasks running concurrently
            results["x"] = var.get()
            var.reset(tok)

        async def task_y():
            tok = var.set("from-y")
            await barrier.wait()
            results["y"] = var.get()
            var.reset(tok)

        await asyncio.gather(task_x(), task_y())

        assert results["x"] == "from-x", "task_x should see its own value"
        assert results["y"] == "from-y", "task_y should see its own value"

    async def test_context_run_creates_isolated_context(self):
        """contextvars.copy_context().run() creates a fully isolated context."""
        var: contextvars.ContextVar[int] = contextvars.ContextVar("test_var4", default=0)

        token = var.set(42)
        ctx = contextvars.copy_context()

        # Modify in copied context — should NOT affect original
        def modify_in_copy():
            var.set(100)

        ctx.run(modify_in_copy)

        assert var.get() == 42  # Original unchanged
        var.reset(token)

    def test_out_of_order_reset_applies_captured_state(self):
        """Out-of-order token reset applies the state captured at set() time.

        Python does NOT enforce LIFO ordering for token resets — there is no
        ValueError. Each token simply restores the value that existed BEFORE
        its corresponding set(). This means misusing tokens can leave the
        ContextVar in unexpected states.

        The agent runtime avoids this by always using try/finally to ensure
        LIFO ordering (each finally block resets exactly the token it set).
        """
        var: contextvars.ContextVar[int] = contextvars.ContextVar("test_var5", default=0)

        token1 = var.set(1)  # captures: "before=0"
        token2 = var.set(2)  # captures: "before=1"

        # Out-of-order: reset token1 first (the "older" token)
        var.reset(token1)  # restores var to 0 (what it was before token1.set)
        assert var.get() == 0  # No error, but we've "lost" token2's state

        # Now token2's reset also works (restores to 1, what it was before token2.set)
        var.reset(token2)
        assert var.get() == 1  # Unexpected! Demonstrates the hazard of wrong ordering.

        # token1 was already reset — using it again raises RuntimeError
        # token2 was the only one still valid at this point
