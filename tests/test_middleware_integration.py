# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests: middleware wired through ActorRuntime.generate / execute_code."""

import warnings

import pytest

from nooa.agent import Agent
from nooa.unifiedllm import FakeLLMClient, LLMResponse

_TEST_LLM = FakeLLMClient()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(llm=None):
    """Create a minimal agent for testing."""

    class _A(Agent, llm=llm or _TEST_LLM):
        async def noop(self) -> str:
            """Say hi."""
            ...

    return _A()


# ---------------------------------------------------------------------------
# llm_call middleware via generate()
# ---------------------------------------------------------------------------


class TestLLMCallMiddlewareViaGenerate:
    """Tests that register llm_call middleware on agent.event_manager and
    exercise it through runtime.generate()."""

    @pytest.mark.asyncio
    async def test_llm_call_middleware_invoked(self):
        """Middleware is called when generate() fires."""
        trail = []
        agent = _make_agent()
        em = agent.event_manager

        async def mw(ctx, nxt):
            trail.append("mw-before")
            ctx = await nxt(ctx)
            trail.append("mw-after")
            return ctx

        em.intercept("llm_call", mw)

        # Drive generate() by setting up the required context vars
        from nooa.runtime.actor import (
            _current_call_var,
            _current_llm_var,
            _current_method_var,
        )

        class FakeCall:
            args = ()
            kwargs = {}
            method_name = "noop"

        tok1 = _current_method_var.set(agent.noop)
        tok2 = _current_call_var.set(FakeCall())
        tok3 = _current_llm_var.set(agent._llm)
        try:
            await agent.runtime.generate()
        finally:
            _current_method_var.reset(tok1)
            _current_call_var.reset(tok2)
            _current_llm_var.reset(tok3)

        assert trail == ["mw-before", "mw-after"]

    @pytest.mark.asyncio
    async def test_llm_call_middleware_can_modify_messages(self):
        """Middleware can inject a system message before the LLM sees it."""
        agent = _make_agent()
        captured_messages = []

        async def spy(ctx, nxt):
            ctx.messages.insert(0, {"role": "system", "content": "INJECTED"})
            ctx = await nxt(ctx)
            captured_messages.extend(ctx.messages)
            return ctx

        agent.event_manager.intercept("llm_call", spy)

        from nooa.runtime.actor import (
            _current_call_var,
            _current_llm_var,
            _current_method_var,
        )

        class FakeCall:
            args = ()
            kwargs = {}
            method_name = "noop"

        tok1 = _current_method_var.set(agent.noop)
        tok2 = _current_call_var.set(FakeCall())
        tok3 = _current_llm_var.set(agent._llm)
        try:
            await agent.runtime.generate()
        finally:
            _current_method_var.reset(tok1)
            _current_call_var.reset(tok2)
            _current_llm_var.reset(tok3)

        assert captured_messages[0]["content"] == "INJECTED"

    @pytest.mark.asyncio
    async def test_llm_call_middleware_short_circuit(self):
        """Middleware can return a fake response without calling the LLM."""
        agent = _make_agent()

        async def fake_response(ctx, nxt):
            ctx.response = LLMResponse(
                raw_response=None,
                content="faked",
                tool_calls=[],
                finish_reason="stop",
                assistant_message={"role": "assistant", "content": "faked"},
                reasoning=None,
                usage=None,
            )
            return ctx  # skip nxt

        agent.event_manager.intercept("llm_call", fake_response)

        from nooa.runtime.actor import (
            _current_call_var,
            _current_llm_var,
            _current_method_var,
        )

        class FakeCall:
            args = ()
            kwargs = {}
            method_name = "noop"

        tok1 = _current_method_var.set(agent.noop)
        tok2 = _current_call_var.set(FakeCall())
        tok3 = _current_llm_var.set(agent._llm)
        try:
            resp, eid = await agent.runtime.generate()
        finally:
            _current_method_var.reset(tok1)
            _current_call_var.reset(tok2)
            _current_llm_var.reset(tok3)

        assert resp.content == "faked"

    @pytest.mark.asyncio
    async def test_no_middleware_fast_path(self):
        """Without middleware, generate() still works (fast path)."""
        agent = _make_agent()

        from nooa.runtime.actor import (
            _current_call_var,
            _current_llm_var,
            _current_method_var,
        )

        class FakeCall:
            args = ()
            kwargs = {}
            method_name = "noop"

        tok1 = _current_method_var.set(agent.noop)
        tok2 = _current_call_var.set(FakeCall())
        tok3 = _current_llm_var.set(agent._llm)
        try:
            resp, eid = await agent.runtime.generate()
        finally:
            _current_method_var.reset(tok1)
            _current_call_var.reset(tok2)
            _current_llm_var.reset(tok3)

        # FakeLLMClient returns empty string when exhausted
        assert resp.content == ""

    @pytest.mark.asyncio
    async def test_multiple_llm_call_middleware_order(self):
        """Two middlewares run in registration order (first = outermost)."""
        agent = _make_agent()
        trail = []

        async def mw_a(ctx, nxt):
            trail.append("a")
            return await nxt(ctx)

        async def mw_b(ctx, nxt):
            trail.append("b")
            return await nxt(ctx)

        agent.event_manager.intercept("llm_call", mw_a)
        agent.event_manager.intercept("llm_call", mw_b)

        from nooa.runtime.actor import (
            _current_call_var,
            _current_llm_var,
            _current_method_var,
        )

        class FakeCall:
            args = ()
            kwargs = {}
            method_name = "noop"

        tok1 = _current_method_var.set(agent.noop)
        tok2 = _current_call_var.set(FakeCall())
        tok3 = _current_llm_var.set(agent._llm)
        try:
            await agent.runtime.generate()
        finally:
            _current_method_var.reset(tok1)
            _current_call_var.reset(tok2)
            _current_llm_var.reset(tok3)

        assert trail == ["a", "b"]

    @pytest.mark.asyncio
    async def test_llm_call_middleware_sees_output_model_in_params(self):
        """output_model is forwarded via ctx.params."""
        agent = _make_agent()
        seen_params = {}

        async def spy(ctx, nxt):
            seen_params.update(ctx.params)
            return await nxt(ctx)

        agent.event_manager.intercept("llm_call", spy)

        from pydantic import BaseModel as BM

        class Dummy(BM):
            x: int = 1

        from nooa.runtime.actor import (
            _current_call_var,
            _current_llm_var,
            _current_method_var,
        )

        class FakeCall:
            args = ()
            kwargs = {}
            method_name = "noop"

        tok1 = _current_method_var.set(agent.noop)
        tok2 = _current_call_var.set(FakeCall())
        tok3 = _current_llm_var.set(agent._llm)
        try:
            await agent.runtime.generate(output_model=Dummy)
        finally:
            _current_method_var.reset(tok1)
            _current_call_var.reset(tok2)
            _current_llm_var.reset(tok3)

        assert seen_params.get("output_model") is Dummy


# ---------------------------------------------------------------------------
# execute_python middleware via execute_code()
# ---------------------------------------------------------------------------


class TestExecutePythonMiddleware:
    @pytest.mark.asyncio
    async def test_execute_python_middleware_invoked(self):
        """Middleware is called when execute_code() fires."""
        trail = []
        agent = _make_agent()

        async def mw(ctx, nxt):
            trail.append("mw-before")
            ctx = await nxt(ctx)
            trail.append("mw-after")
            return ctx

        agent.event_manager.intercept("execute_python", mw)

        result = await agent.runtime.execute_code("x = 1", validate=False)
        assert trail == ["mw-before", "mw-after"]
        assert result is not None

    @pytest.mark.asyncio
    async def test_execute_python_middleware_can_modify_code(self):
        """Middleware can rewrite the code before execution."""
        agent = _make_agent()

        async def rewrite(ctx, nxt):
            ctx.code = ctx.code.replace("PLACEHOLDER", "42")
            return await nxt(ctx)

        agent.event_manager.intercept("execute_python", rewrite)

        result = await agent.runtime.execute_code(
            "print(PLACEHOLDER)",
            validate=False,
            wrap_in_function=True,
        )
        assert "42" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_python_no_middleware_fast_path(self):
        """Without middleware, execute_code works normally."""
        agent = _make_agent()
        result = await agent.runtime.execute_code("x = 1 + 1", validate=False)
        assert result.error is None

    @pytest.mark.asyncio
    async def test_execute_python_middleware_short_circuit(self):
        """Middleware can return a fake result without executing code."""
        from nooa.events import ExecutionResult

        agent = _make_agent()

        async def fake(ctx, nxt):
            ctx.result = ExecutionResult(stdout="fake-output", error=None, defined_methods={})
            return ctx

        agent.event_manager.intercept("execute_python", fake)

        result = await agent.runtime.execute_code(
            "import os; os.system('rm -rf /')", validate=False
        )
        assert result.stdout == "fake-output"

    @pytest.mark.asyncio
    async def test_execute_python_reentry_guard(self):
        """Recursive execute_code skips middleware (re-entry guard)."""
        agent = _make_agent()
        mw_count = 0

        async def counting_mw(ctx, nxt):
            nonlocal mw_count
            mw_count += 1
            return await nxt(ctx)

        agent.event_manager.intercept("execute_python", counting_mw)

        # The core exec inside middleware will recurse into execute_code,
        # but the re-entry guard prevents middleware from firing again.
        result = await agent.runtime.execute_code("x = 1", validate=False)
        assert result.error is None
        # Middleware should fire exactly once (not recursively)
        assert mw_count == 1

    @pytest.mark.asyncio
    async def test_multiple_execute_python_middleware(self):
        """Two execute_python middlewares chain correctly."""
        agent = _make_agent()
        trail = []

        async def mw_a(ctx, nxt):
            trail.append("a")
            return await nxt(ctx)

        async def mw_b(ctx, nxt):
            trail.append("b")
            return await nxt(ctx)

        agent.event_manager.intercept("execute_python", mw_a)
        agent.event_manager.intercept("execute_python", mw_b)

        await agent.runtime.execute_code("x = 1", validate=False)
        assert trail == ["a", "b"]


# ---------------------------------------------------------------------------
# Unsubscribe
# ---------------------------------------------------------------------------


class TestMiddlewareUnsubscribe:
    @pytest.mark.asyncio
    async def test_unsub_removes_middleware(self):
        agent = _make_agent()
        trail = []

        async def mw(ctx, nxt):
            trail.append("mw")
            return await nxt(ctx)

        unsub = agent.event_manager.intercept("execute_python", mw)
        await agent.runtime.execute_code("x = 1", validate=False)
        assert trail == ["mw"]

        unsub()
        trail.clear()
        await agent.runtime.execute_code("x = 2", validate=False)
        assert trail == []  # middleware no longer fires


# ---------------------------------------------------------------------------
# Cross-kind isolation
# ---------------------------------------------------------------------------


class TestCrossKindIsolation:
    @pytest.mark.asyncio
    async def test_llm_middleware_does_not_affect_execute(self):
        agent = _make_agent()
        trail = []

        async def mw(ctx, nxt):
            trail.append("llm")
            return await nxt(ctx)

        agent.event_manager.intercept("llm_call", mw)

        await agent.runtime.execute_code("x = 1", validate=False)
        assert trail == []  # llm_call middleware not invoked for execute_code

    @pytest.mark.asyncio
    async def test_execute_middleware_does_not_affect_llm(self):
        agent = _make_agent()
        trail = []

        async def mw(ctx, nxt):
            trail.append("exec")
            return await nxt(ctx)

        agent.event_manager.intercept("execute_python", mw)

        from nooa.runtime.actor import (
            _current_call_var,
            _current_llm_var,
            _current_method_var,
        )

        class FakeCall:
            args = ()
            kwargs = {}
            method_name = "noop"

        tok1 = _current_method_var.set(agent.noop)
        tok2 = _current_call_var.set(FakeCall())
        tok3 = _current_llm_var.set(agent._llm)
        try:
            await agent.runtime.generate()
        finally:
            _current_method_var.reset(tok1)
            _current_call_var.reset(tok2)
            _current_llm_var.reset(tok3)

        assert trail == []  # execute_python middleware not invoked for generate


# ---------------------------------------------------------------------------
# Context fields
# ---------------------------------------------------------------------------


class TestContextFields:
    @pytest.mark.asyncio
    async def test_llm_ctx_has_agent_and_runtime(self):
        agent = _make_agent()
        seen = {}

        async def spy(ctx, nxt):
            seen["agent"] = ctx.agent
            seen["runtime"] = ctx.runtime
            return await nxt(ctx)

        agent.event_manager.intercept("llm_call", spy)

        from nooa.runtime.actor import (
            _current_call_var,
            _current_llm_var,
            _current_method_var,
        )

        class FakeCall:
            args = ()
            kwargs = {}
            method_name = "noop"

        tok1 = _current_method_var.set(agent.noop)
        tok2 = _current_call_var.set(FakeCall())
        tok3 = _current_llm_var.set(agent._llm)
        try:
            await agent.runtime.generate()
        finally:
            _current_method_var.reset(tok1)
            _current_call_var.reset(tok2)
            _current_llm_var.reset(tok3)

        assert seen["agent"] is agent
        assert seen["runtime"] is agent.runtime

    @pytest.mark.asyncio
    async def test_exec_ctx_has_agent_and_runtime(self):
        agent = _make_agent()
        seen = {}

        async def spy(ctx, nxt):
            seen["agent"] = ctx.agent
            seen["runtime"] = ctx.runtime
            seen["code"] = ctx.code
            return await nxt(ctx)

        agent.event_manager.intercept("execute_python", spy)

        await agent.runtime.execute_code("x = 42", validate=False)
        assert seen["agent"] is agent
        assert seen["runtime"] is agent.runtime
        assert seen["code"] == "x = 42"


# ---------------------------------------------------------------------------
# Short-circuit guard: middleware must set response/result
# ---------------------------------------------------------------------------


class TestShortCircuitGuard:
    @pytest.mark.asyncio
    async def test_llm_short_circuit_without_response_raises(self):
        """Middleware that short-circuits without setting ctx.response must raise."""
        agent = _make_agent()

        async def bad_guardrail(ctx, nxt):
            # Short-circuit but forget to set ctx.response
            return ctx

        agent.event_manager.intercept("llm_call", bad_guardrail)

        from nooa.runtime.actor import (
            _current_call_var,
            _current_llm_var,
            _current_method_var,
        )

        class FakeCall:
            args = ()
            kwargs = {}
            method_name = "noop"

        tok1 = _current_method_var.set(agent.noop)
        tok2 = _current_call_var.set(FakeCall())
        tok3 = _current_llm_var.set(agent._llm)
        try:
            with pytest.raises(RuntimeError, match="response"):
                await agent.runtime.generate()
        finally:
            _current_method_var.reset(tok1)
            _current_call_var.reset(tok2)
            _current_llm_var.reset(tok3)

    @pytest.mark.asyncio
    async def test_exec_short_circuit_without_result_raises(self):
        """Middleware that short-circuits without setting ctx.result must raise."""
        agent = _make_agent()

        async def bad_guardrail(ctx, nxt):
            # Short-circuit but forget to set ctx.result
            return ctx

        agent.event_manager.intercept("execute_python", bad_guardrail)

        with pytest.raises(RuntimeError, match="result"):
            await agent.runtime.execute_code("x = 1", validate=False)


# ---------------------------------------------------------------------------
# Nested generation: execute_python middleware fires for inner method
# ---------------------------------------------------------------------------


class TestNestedGenerationReentry:
    @pytest.mark.asyncio
    async def test_execute_python_middleware_fires_for_nested_agent_method(self):
        """When method_a's code calls self.method_b(), execute_python
        middleware fires for method_b's code execution too (not suppressed
        by the re-entry guard)."""
        import json

        from nooa.unifiedllm import ToolCall

        exec_methods = []

        async def capture_exec(ctx, nxt):
            exec_methods.append(ctx.agent.__class__.__name__)
            return await nxt(ctx)

        def _tc(code, call_id="c1"):
            return ToolCall(
                id=call_id,
                name="execute_python",
                arguments=json.dumps({"code": code}),
            )

        def _rr(val, call_id="cr"):
            return ToolCall(
                id=call_id,
                name="return_result",
                arguments=json.dumps({"result": val}),
            )

        def _resp(tc_list):
            return LLMResponse(
                raw_response=None,
                content="",
                tool_calls=tc_list,
                finish_reason="tool_calls",
                assistant_message={"role": "assistant", "content": ""},
            )

        # outer: exec code that calls inner, then return
        # inner: exec code (y = 99), then return
        llm = FakeLLMClient(
            scripted_responses=[
                _resp([_tc("r = await self.inner_method()", "c1")]),
                _resp([_tc("y = 99", "c2")]),
                _resp([_rr(99, "cr1")]),
                _resp([_rr(99, "cr2")]),
            ]
        )

        class Nested(Agent, llm=llm):
            async def outer_method(self) -> int:
                """Call inner."""
                ...

            async def inner_method(self) -> int:
                """Return 99."""
                ...

        agent = Nested()
        agent.event_manager.intercept("execute_python", capture_exec)
        await agent.outer_method()

        # Middleware must fire for BOTH outer and inner code executions
        assert len(exec_methods) >= 2


# ---------------------------------------------------------------------------
# Code rewriting and local injection
# ---------------------------------------------------------------------------


class TestCodeRewriteAndInjection:
    @pytest.mark.asyncio
    async def test_middleware_can_rewrite_code(self):
        """execute_python middleware can rewrite ctx.code before execution."""
        agent = _make_agent()

        async def rewriter(ctx, nxt):
            ctx.code = "print('rewritten')"
            return await nxt(ctx)

        agent.event_manager.intercept("execute_python", rewriter)

        result = await agent.runtime.execute_code("print('original')", validate=False)
        assert "rewritten" in result.stdout
        assert "original" not in result.stdout

    @pytest.mark.asyncio
    async def test_middleware_can_inject_builtins(self):
        """execute_python middleware can inject builtins via ctx.params."""
        agent = _make_agent()

        async def injector(ctx, nxt):
            builtins = ctx.params.get("builtins") or {}
            builtins["injected_value"] = 99
            ctx.params["builtins"] = builtins
            return await nxt(ctx)

        agent.event_manager.intercept("execute_python", injector)

        result = await agent.runtime.execute_code("print(injected_value + 1)", validate=False)
        assert "100" in result.stdout


# ---------------------------------------------------------------------------
# agent_call middleware
# ---------------------------------------------------------------------------


class TestAgentCallMiddleware:
    @pytest.mark.asyncio
    async def test_agent_call_middleware_fires_on_method_call(self):
        """agent_call middleware fires when an agent method is called."""
        import json

        from nooa.runtime.middleware import AgentCallContext
        from nooa.unifiedllm import ToolCall

        trail = []

        async def mw(ctx: AgentCallContext, nxt):
            trail.append(f"pre:{ctx.method_name}")
            ctx = await nxt(ctx)
            trail.append(f"post:{ctx.method_name}")
            return ctx

        llm = FakeLLMClient(
            scripted_responses=[
                LLMResponse(
                    raw_response=None,
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="c1", name="return_result", arguments=json.dumps({"result": "hi"})
                        )
                    ],
                    finish_reason="tool_calls",
                    assistant_message={"role": "assistant", "content": ""},
                ),
            ]
        )

        class A(Agent, llm=llm):
            async def greet(self) -> str:
                """Say hi."""
                ...

        agent = A()
        agent.event_manager.intercept("agent_call", mw)

        await agent.greet()
        assert trail == ["pre:greet", "post:greet"]

    @pytest.mark.asyncio
    async def test_agent_call_middleware_sees_args(self):
        """agent_call middleware can see method arguments."""
        import json

        from nooa.runtime.middleware import AgentCallContext
        from nooa.unifiedllm import ToolCall

        captured = {}

        async def spy(ctx: AgentCallContext, nxt):
            captured["method"] = ctx.method_name
            captured["args"] = ctx.args
            captured["kwargs"] = ctx.kwargs
            return await nxt(ctx)

        llm = FakeLLMClient(
            scripted_responses=[
                LLMResponse(
                    raw_response=None,
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="c1", name="return_result", arguments=json.dumps({"result": "ok"})
                        )
                    ],
                    finish_reason="tool_calls",
                    assistant_message={"role": "assistant", "content": ""},
                ),
            ]
        )

        class A(Agent, llm=llm):
            async def process(self, x: int, label: str = "default") -> str:
                """Process {x} with {label}."""
                ...

        agent = A()
        agent.event_manager.intercept("agent_call", spy)

        await agent.process(42, label="custom")
        assert captured["method"] == "process"
        assert captured["args"] == (42,)
        assert captured["kwargs"] == {"label": "custom"}

    @pytest.mark.asyncio
    async def test_agent_call_middleware_can_see_result(self):
        """agent_call middleware can inspect the result on the way out."""
        import json

        from nooa.runtime.middleware import AgentCallContext
        from nooa.unifiedllm import ToolCall

        seen_result = {}

        async def spy(ctx: AgentCallContext, nxt):
            ctx = await nxt(ctx)
            seen_result["value"] = ctx.result
            return ctx

        llm = FakeLLMClient(
            scripted_responses=[
                LLMResponse(
                    raw_response=None,
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            name="return_result",
                            arguments=json.dumps({"result": "hello"}),
                        )
                    ],
                    finish_reason="tool_calls",
                    assistant_message={"role": "assistant", "content": ""},
                ),
            ]
        )

        class A(Agent, llm=llm):
            async def greet(self) -> str:
                """Say hi."""
                ...

        agent = A()
        agent.event_manager.intercept("agent_call", spy)

        result = await agent.greet()
        assert result == "hello"
        assert seen_result["value"] == "hello"

    @pytest.mark.asyncio
    async def test_agent_call_short_circuit(self):
        """agent_call middleware can short-circuit the entire method."""
        from nooa.runtime.middleware import AgentCallContext

        llm = FakeLLMClient()

        async def blocker(ctx: AgentCallContext, nxt):
            ctx.result = "blocked by middleware"
            return ctx

        class A(Agent, llm=llm):
            async def greet(self) -> str:
                """Say hi."""
                ...

        agent = A()
        agent.event_manager.intercept("agent_call", blocker)

        result = await agent.greet()
        assert result == "blocked by middleware"
        assert llm.call_count == 0  # LLM never called

    @pytest.mark.asyncio
    async def test_agent_call_short_circuit_without_result_raises(self):
        """Middleware that short-circuits without setting ctx.result raises."""
        from nooa.runtime.middleware import AgentCallContext

        llm = FakeLLMClient()

        async def bad_blocker(ctx: AgentCallContext, nxt):
            return ctx  # forgot to set ctx.result

        class A(Agent, llm=llm):
            async def greet(self) -> str:
                """Say hi."""
                ...

        agent = A()
        agent.event_manager.intercept("agent_call", bad_blocker)

        with pytest.raises(RuntimeError, match="ctx.result"):
            await agent.greet()

    @pytest.mark.asyncio
    async def test_agent_call_wraps_llm_call_and_execute_python(self):
        """agent_call is outermost — llm_call and execute_python fire inside it."""
        import json

        from nooa.unifiedllm import ToolCall

        order = []

        async def agent_mw(ctx, nxt):
            order.append("agent-pre")
            ctx = await nxt(ctx)
            order.append("agent-post")
            return ctx

        async def llm_mw(ctx, nxt):
            order.append("llm")
            return await nxt(ctx)

        async def exec_mw(ctx, nxt):
            order.append("exec")
            return await nxt(ctx)

        llm = FakeLLMClient(
            scripted_responses=[
                LLMResponse(
                    raw_response=None,
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            name="execute_python",
                            arguments=json.dumps({"code": "x = 1"}),
                        )
                    ],
                    finish_reason="tool_calls",
                    assistant_message={"role": "assistant", "content": ""},
                ),
                LLMResponse(
                    raw_response=None,
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="c2",
                            name="return_result",
                            arguments=json.dumps({"result": "done"}),
                        )
                    ],
                    finish_reason="tool_calls",
                    assistant_message={"role": "assistant", "content": ""},
                ),
            ]
        )

        class A(Agent, llm=llm):
            async def work(self) -> str:
                """Do work."""
                ...

        agent = A()
        agent.event_manager.intercept("agent_call", agent_mw)
        agent.event_manager.intercept("llm_call", llm_mw)
        agent.event_manager.intercept("execute_python", exec_mw)

        await agent.work()
        # agent_call wraps everything
        assert order[0] == "agent-pre"
        assert order[-1] == "agent-post"
        # llm and exec fire inside
        assert "llm" in order
        assert "exec" in order


# ---------------------------------------------------------------------------
# agent_call middleware coverage gaps
# ---------------------------------------------------------------------------


class _PlainBase:
    """A non-Agent base — its methods are never instrumented by the metaclass."""

    def inherited_sync(self) -> str:
        """Inherited from a non-Agent base."""
        return "inherited_sync"


async def _passthrough(ctx, nxt):
    return await nxt(ctx)


class TestAgentCallMiddlewareCoverage:
    """agent_call middleware wraps only instrumented async methods.

    Sync methods, @no_trace methods (sync or async), staticmethod/classmethod,
    and methods inherited from non-Agent bases all execute outside the chain.
    These tests pin that asymmetry and assert it is no longer silent.
    """

    @pytest.mark.asyncio
    async def test_sync_method_bypasses_agent_call_middleware(self):
        """A blocking guard does not stop a sync method's side effect."""
        seen = []

        async def deny(ctx, nxt):
            seen.append(ctx.method_name)
            if ctx.method_name == "charge_card":
                raise PermissionError("charge_card blocked")
            return await nxt(ctx)

        class A(Agent, llm=_TEST_LLM):
            def __init__(self):
                super().__init__()
                self.charges = []

            def charge_card(self, amount: int) -> str:
                """Record a simulated card charge."""
                self.charges.append(amount)
                return f"receipt-{amount}"

        agent = A()
        agent.event_manager.intercept("agent_call", deny)

        with pytest.warns(RuntimeWarning, match="does not apply to the synchronous method"):
            result = agent.charge_card(100)

        # The guard neither ran nor blocked: the side effect happened.
        assert result == "receipt-100"
        assert agent.charges == [100]
        assert seen == []

    @pytest.mark.asyncio
    async def test_async_method_is_blocked_by_same_middleware(self):
        """Control: the identical capability declared async IS intercepted."""
        seen = []

        async def deny(ctx, nxt):
            seen.append(ctx.method_name)
            if ctx.method_name == "charge_card":
                raise PermissionError("charge_card blocked")
            return await nxt(ctx)

        class A(Agent, llm=_TEST_LLM):
            def __init__(self):
                super().__init__()
                self.charges = []

            async def charge_card(self, amount: int) -> str:
                """Record a simulated card charge."""
                self.charges.append(amount)
                return f"receipt-{amount}"

        agent = A()
        agent.event_manager.intercept("agent_call", deny)

        with pytest.raises(PermissionError):
            await agent.charge_card(100)

        assert agent.charges == []
        assert seen == ["charge_card"]

    @pytest.mark.asyncio
    async def test_scan_names_every_uncovered_method_kind(self):
        """The entry-point scan reports kinds the sync wrapper cannot see.

        @no_trace, staticmethod, classmethod, and inherited methods are never
        wrapped at all, so there is no per-call hook in which to warn. Only a
        class-level scan finds them.
        """
        from nooa.metaclass import no_trace

        class Probe(Agent, _PlainBase, llm=_TEST_LLM):
            def plain_sync(self) -> str:
                """Plain sync."""
                return "plain_sync"

            @no_trace
            def notrace_sync(self) -> str:
                """no_trace sync."""
                return "notrace_sync"

            @no_trace
            async def notrace_async(self) -> str:
                """no_trace async."""
                return "notrace_async"

            @staticmethod
            def static_m() -> str:
                """Static."""
                return "static_m"

            @classmethod
            def class_m(cls) -> str:
                """Classmethod."""
                return "class_m"

            async def entry(self) -> str:
                """Traced async entry point."""
                return "entry"

        agent = Probe()
        agent.event_manager.intercept("agent_call", _passthrough)

        with pytest.warns(RuntimeWarning) as caught:
            await agent.entry()

        text = " ".join(str(w.message) for w in caught)
        for name in (
            "plain_sync",
            "notrace_sync",
            "notrace_async",
            "static_m",
            "class_m",
            "inherited_sync",
        ):
            assert name in text, f"{name} missing from coverage warning"

        # The one genuinely covered method must not be reported...
        assert "entry" not in text
        # ...nor should Agent's own infrastructure be listed as the user's problem.
        assert "event_manager" not in text

    @pytest.mark.asyncio
    async def test_no_warning_when_class_is_fully_covered(self):
        """An all-async agent produces no coverage warning."""

        class A(Agent, llm=_TEST_LLM):
            async def entry(self) -> str:
                """Traced async entry point."""
                return "entry"

        agent = A()
        agent.event_manager.intercept("agent_call", _passthrough)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert await agent.entry() == "entry"

    @pytest.mark.asyncio
    async def test_no_warning_when_no_agent_call_middleware(self):
        """The warning is scoped to an actual bypass, not to sync methods."""

        class A(Agent, llm=_TEST_LLM):
            def helper(self) -> str:
                """A plain sync helper."""
                return "ok"

        agent = A()

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert agent.helper() == "ok"

    @pytest.mark.asyncio
    async def test_no_warning_for_unrelated_middleware_kinds(self):
        """llm_call / execute_python middleware do not trigger the warning."""

        class A(Agent, llm=_TEST_LLM):
            def helper(self) -> str:
                """A plain sync helper."""
                return "ok"

        agent = A()
        agent.event_manager.intercept("llm_call", _passthrough)
        agent.event_manager.intercept("execute_python", _passthrough)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            assert agent.helper() == "ok"

    @pytest.mark.asyncio
    async def test_warning_emitted_once_per_method(self):
        """A sync helper in a loop warns once, not once per call."""

        class A(Agent, llm=_TEST_LLM):
            def helper(self) -> str:
                """A plain sync helper."""
                return "ok"

        agent = A()
        agent.event_manager.intercept("agent_call", _passthrough)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            for _ in range(5):
                agent.helper()

        assert len([w for w in caught if issubclass(w.category, RuntimeWarning)]) == 1

    @pytest.mark.asyncio
    async def test_warning_names_the_offending_method(self):
        """The message identifies the class and method so it is actionable."""

        class PaymentAgent(Agent, llm=_TEST_LLM):
            def charge_card(self, amount: int) -> str:
                """Record a simulated card charge."""
                return f"receipt-{amount}"

        agent = PaymentAgent()
        agent.event_manager.intercept("agent_call", _passthrough)

        with pytest.warns(RuntimeWarning, match=r"PaymentAgent\.charge_card"):
            agent.charge_card(100)

    @pytest.mark.asyncio
    async def test_dedup_is_per_agent_not_per_class(self):
        """A second agent instance still gets told about its own bypass."""

        class A(Agent, llm=_TEST_LLM):
            def helper(self) -> str:
                """A plain sync helper."""
                return "ok"

        first = A()
        first.event_manager.intercept("agent_call", _passthrough)
        with pytest.warns(RuntimeWarning):
            first.helper()

        second = A()
        second.event_manager.intercept("agent_call", _passthrough)
        with pytest.warns(RuntimeWarning):
            second.helper()


class TestAgentCallBypassWarningDelivery:
    """The warning has to survive nooa's NullHandler and CodeAct's stderr capture."""

    @pytest.mark.asyncio
    async def test_codeact_bypass_reaches_real_stderr(self):
        """A bypass inside a cell must escape the capture buffer.

        Cells point sys.stderr at a buffer that is fed back to the model, so a
        warning written there would be invisible to the developer and would
        pollute the model's context. It must land on the real stream instead.
        """
        import io
        import sys

        class PaymentAgent(Agent, llm=_TEST_LLM):
            def __init__(self):
                super().__init__()
                self.charges = []

            def charge_card(self, amount: int) -> str:
                """Record a simulated card charge."""
                self.charges.append(amount)
                return f"receipt-{amount}"

        agent = PaymentAgent()
        agent.event_manager.intercept("agent_call", _passthrough)

        # pytest replaces warnings.showwarning to record rather than print, so
        # asserting on stderr needs the default write-through restored. This one
        # writes through the *live* sys.stderr, which inside a cell is the
        # ContextVarStream — exactly the object whose routing is under test.
        def write_through(message, category, filename, lineno, file=None, line=None):
            (file or sys.stderr).write(str(message))

        real_stderr = sys.stderr
        sink = io.StringIO()
        original_showwarning = warnings.showwarning
        sys.stderr = sink
        warnings.showwarning = write_through
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("always", RuntimeWarning)
                warnings.showwarning = write_through
                result = await agent.runtime.execute_code("self.charge_card(100)")
        finally:
            warnings.showwarning = original_showwarning
            sys.stderr = real_stderr

        assert agent.charges == [100]
        # Reached the developer's real stream...
        assert "PaymentAgent.charge_card" in sink.getvalue()
        # ...without leaking into what the model reads back.
        assert "agent_call middleware" not in (result.stderr or "")

    @pytest.mark.asyncio
    async def test_warning_promoted_to_error_propagates(self):
        """-W error must raise rather than be swallowed by the diagnostic guard."""

        class A(Agent, llm=_TEST_LLM):
            def helper(self) -> str:
                """A plain sync helper."""
                return "ok"

        agent = A()
        agent.event_manager.intercept("agent_call", _passthrough)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with pytest.raises(RuntimeWarning):
                agent.helper()

    @pytest.mark.asyncio
    async def test_swallowed_warning_is_not_marked_delivered(self):
        """A warning that raised was never seen, so it must be re-emitted."""

        class A(Agent, llm=_TEST_LLM):
            def helper(self) -> str:
                """A plain sync helper."""
                return "ok"

        agent = A()
        agent.event_manager.intercept("agent_call", _passthrough)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with pytest.raises(RuntimeWarning):
                agent.helper()
            # Second call must raise again — the first was never delivered.
            with pytest.raises(RuntimeWarning):
                agent.helper()
