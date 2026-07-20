# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the middleware engine (contexts, chain, EventManager integration)."""

import pytest

from nooa.runtime.event_manager import EventManager, _make_next
from nooa.runtime.middleware import (
    ExecutePythonContext,
    LLMCallContext,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _passthrough(ctx, nxt):
    """Middleware that simply forwards."""
    return await nxt(ctx)


# ---------------------------------------------------------------------------
# LLMCallContext / ExecutePythonContext
# ---------------------------------------------------------------------------


class TestLLMCallContext:
    def test_defaults(self):
        ctx = LLMCallContext(messages=[{"role": "user", "content": "hi"}])
        assert ctx.messages == [{"role": "user", "content": "hi"}]
        assert ctx.params == {}
        assert ctx.agent is None
        assert ctx.runtime is None
        assert ctx.response is None

    def test_mutate_messages(self):
        ctx = LLMCallContext(messages=[])
        ctx.messages.append({"role": "system", "content": "injected"})
        assert len(ctx.messages) == 1

    def test_params_round_trip(self):
        ctx = LLMCallContext(messages=[], params={"tools": [], "temperature": 0.5})
        assert ctx.params["temperature"] == 0.5

    def test_typed_agent_and_runtime(self):
        """agent and runtime accept their typed values or None."""
        ctx = LLMCallContext(messages=[])
        assert ctx.agent is None
        assert ctx.runtime is None


class TestExecutePythonContext:
    def test_defaults(self):
        ctx = ExecutePythonContext(code="print(1)")
        assert ctx.code == "print(1)"
        assert ctx.result is None

    def test_mutate_code(self):
        ctx = ExecutePythonContext(code="x = 1")
        ctx.code = "x = 2"
        assert ctx.code == "x = 2"


# ---------------------------------------------------------------------------
# _make_next
# ---------------------------------------------------------------------------


class TestMakeNext:
    @pytest.mark.asyncio
    async def test_single_layer(self):
        called = []

        async def mw(ctx, nxt):
            called.append("mw")
            return await nxt(ctx)

        async def core(ctx):
            called.append("core")
            return ctx

        nxt = _make_next(mw, core)
        ctx = LLMCallContext(messages=[])
        await nxt(ctx)
        assert called == ["mw", "core"]

    @pytest.mark.asyncio
    async def test_chain_order(self):
        order = []

        async def mw_a(ctx, nxt):
            order.append("a-before")
            ctx = await nxt(ctx)
            order.append("a-after")
            return ctx

        async def mw_b(ctx, nxt):
            order.append("b-before")
            ctx = await nxt(ctx)
            order.append("b-after")
            return ctx

        async def core(ctx):
            order.append("core")
            return ctx

        # Build chain: a wraps b wraps core (a is outermost)
        chain_b = _make_next(mw_b, core)
        chain_a = _make_next(mw_a, chain_b)
        await chain_a(LLMCallContext(messages=[]))
        assert order == ["a-before", "b-before", "core", "b-after", "a-after"]


# ---------------------------------------------------------------------------
# EventManager — use()
# ---------------------------------------------------------------------------


class TestEventManagerIntercept:
    def test_intercept_returns_unsubscribe(self):
        host = EventManager()
        unsub = host.intercept("llm_call", _passthrough)
        assert callable(unsub)
        assert len(host._middleware["llm_call"]) == 1
        unsub()
        assert len(host._middleware["llm_call"]) == 0

    def test_intercept_invalid_kind(self):
        host = EventManager()
        with pytest.raises(ValueError, match="Unknown middleware kind"):
            host.intercept("bad_kind", _passthrough)

    def test_unsubscribe_idempotent(self):
        host = EventManager()
        unsub = host.intercept("llm_call", _passthrough)
        unsub()
        unsub()  # second call should not raise

    def test_multiple_middleware_same_kind(self):
        host = EventManager()
        host.intercept("llm_call", _passthrough)
        host.intercept("llm_call", _passthrough)
        assert len(host._middleware["llm_call"]) == 2

    def test_intercept_both_kinds(self):
        host = EventManager()
        host.intercept("llm_call", _passthrough)
        host.intercept("execute_python", _passthrough)
        assert len(host._middleware["llm_call"]) == 1
        assert len(host._middleware["execute_python"]) == 1


# ---------------------------------------------------------------------------
# EventManager — run_middleware()
# ---------------------------------------------------------------------------


class TestEventManagerRun:
    @pytest.mark.asyncio
    async def test_no_middleware_calls_core(self):
        host = EventManager()
        called = []

        async def core(ctx):
            called.append(True)
            return ctx

        ctx = LLMCallContext(messages=[])
        result = await host.run_middleware("llm_call", ctx, core)
        assert called == [True]
        assert result is ctx

    @pytest.mark.asyncio
    async def test_single_middleware(self):
        host = EventManager()
        trail = []

        async def mw(ctx, nxt):
            trail.append("mw")
            return await nxt(ctx)

        host.intercept("llm_call", mw)

        async def core(ctx):
            trail.append("core")
            return ctx

        await host.run_middleware("llm_call", LLMCallContext(messages=[]), core)
        assert trail == ["mw", "core"]

    @pytest.mark.asyncio
    async def test_middleware_can_mutate_context(self):
        host = EventManager()

        async def inject(ctx, nxt):
            ctx.messages.insert(0, {"role": "system", "content": "injected"})
            return await nxt(ctx)

        host.intercept("llm_call", inject)

        async def core(ctx):
            return ctx

        ctx = LLMCallContext(messages=[{"role": "user", "content": "hi"}])
        result = await host.run_middleware("llm_call", ctx, core)
        assert result.messages[0]["content"] == "injected"

    @pytest.mark.asyncio
    async def test_middleware_can_short_circuit(self):
        host = EventManager()

        async def blocker(ctx, nxt):
            ctx.response = "blocked"
            return ctx  # never calls nxt

        host.intercept("llm_call", blocker)
        core_called = []

        async def core(ctx):
            core_called.append(True)
            return ctx

        ctx = LLMCallContext(messages=[])
        result = await host.run_middleware("llm_call", ctx, core)
        assert result.response == "blocked"
        assert core_called == []

    @pytest.mark.asyncio
    async def test_execute_python_kind(self):
        host = EventManager()
        trail = []

        async def mw(ctx, nxt):
            trail.append("mw")
            return await nxt(ctx)

        host.intercept("execute_python", mw)

        async def core(ctx):
            trail.append("core")
            return ctx

        ctx = ExecutePythonContext(code="1+1")
        await host.run_middleware("execute_python", ctx, core)
        assert trail == ["mw", "core"]

    @pytest.mark.asyncio
    async def test_unsubscribe_mid_run_safe(self):
        """Unsubscribing during run_middleware doesn't affect the current chain
        because run_middleware snapshots the list."""
        host = EventManager()
        unsub = None

        async def self_removing(ctx, nxt):
            nonlocal unsub
            unsub()  # remove self from the list
            return await nxt(ctx)

        unsub = host.intercept("llm_call", self_removing)

        async def core(ctx):
            return ctx

        # Should not raise even though middleware removed itself
        await host.run_middleware("llm_call", LLMCallContext(messages=[]), core)
        assert len(host._middleware["llm_call"]) == 0


# ---------------------------------------------------------------------------
# _init_middleware
# ---------------------------------------------------------------------------


class TestInitMiddleware:
    def test_middleware_id_unique(self):
        a = EventManager()
        b = EventManager()
        assert a._middleware_id != b._middleware_id

    def test_middleware_dicts_independent(self):
        a = EventManager()
        b = EventManager()
        a.intercept("llm_call", _passthrough)
        assert len(b._middleware["llm_call"]) == 0


# ---------------------------------------------------------------------------
# EventManager integration (basic)
# ---------------------------------------------------------------------------


class TestEventManagerMiddleware:
    def test_event_manager_has_middleware(self):
        from nooa.runtime.event_manager import EventManager

        em = EventManager()
        assert hasattr(em, "_middleware")
        assert hasattr(em, "_middleware_id")
        assert "llm_call" in em._middleware
        assert "execute_python" in em._middleware

    def test_event_manager_use(self):
        from nooa.runtime.event_manager import EventManager

        em = EventManager()
        unsub = em.intercept("llm_call", _passthrough)
        assert len(em._middleware["llm_call"]) == 1
        unsub()
        assert len(em._middleware["llm_call"]) == 0

    @pytest.mark.asyncio
    async def test_event_manager_run(self):
        from nooa.runtime.event_manager import EventManager

        em = EventManager()
        trail = []

        async def mw(ctx, nxt):
            trail.append("mw")
            return await nxt(ctx)

        em.intercept("llm_call", mw)

        async def core(ctx):
            trail.append("core")
            return ctx

        await em.run_middleware("llm_call", LLMCallContext(messages=[]), core)
        assert trail == ["mw", "core"]

    @pytest.mark.asyncio
    async def test_middleware_exception_propagates(self):
        """Middleware that raises propagates the exception to the caller."""
        from nooa.runtime.event_manager import EventManager

        em = EventManager()

        async def bad_mw(ctx, nxt):
            raise RuntimeError("middleware exploded")

        em.intercept("llm_call", bad_mw)

        async def core(ctx):
            return ctx

        with pytest.raises(RuntimeError, match="middleware exploded"):
            await em.run_middleware("llm_call", LLMCallContext(messages=[]), core)

    @pytest.mark.asyncio
    async def test_nxt_called_twice_runs_core_twice(self):
        """Middleware calling nxt() twice runs the rest of the chain twice."""
        from nooa.runtime.event_manager import EventManager

        em = EventManager()
        core_count = 0

        async def retry_mw(ctx, nxt):
            r1 = await nxt(ctx)
            r2 = await nxt(ctx)
            return (r1, r2)

        em.intercept("llm_call", retry_mw)

        async def core(ctx):
            nonlocal core_count
            core_count += 1
            return f"call-{core_count}"

        result = await em.run_middleware("llm_call", LLMCallContext(messages=[]), core)
        assert core_count == 2
        assert result == ("call-1", "call-2")

    @pytest.mark.asyncio
    async def test_registration_order_is_execution_order(self):
        """First registered = outermost (pre first, post last)."""
        from nooa.runtime.event_manager import EventManager

        em = EventManager()
        order = []

        async def make_mw(name):
            async def mw(ctx, nxt):
                order.append(f"{name}-pre")
                r = await nxt(ctx)
                order.append(f"{name}-post")
                return r

            return mw

        # Register a, b, c in that order → a is outermost
        em.intercept("llm_call", await make_mw("a"))
        em.intercept("llm_call", await make_mw("b"))
        em.intercept("llm_call", await make_mw("c"))

        async def core(ctx):
            order.append("core")
            return ctx

        await em.run_middleware("llm_call", LLMCallContext(messages=[]), core)
        assert order == [
            "a-pre",
            "b-pre",
            "c-pre",
            "core",
            "c-post",
            "b-post",
            "a-post",
        ]
