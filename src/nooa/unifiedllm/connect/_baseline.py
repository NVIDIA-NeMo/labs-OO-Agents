# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native typed baseline; transient replies never enter the agent history."""

import io
from contextlib import redirect_stderr, redirect_stdout

from opentelemetry.trace import INVALID_SPAN, use_span

from nooa import Agent, no_trace
from nooa.config.strategy_config import PredictConfig
from nooa.decorators import strategy
from nooa.llm_types import LLMResponse
from nooa.runtime.hooks import hooks_scope
from nooa.strategies.predict import PredictStrategy


class _TextBaseline(Agent):
    """Compute a small integer using the configured model."""

    @no_trace
    @strategy(PredictStrategy(config=PredictConfig(max_retries=1)))
    async def compute(self) -> int:
        """Compute 17 * 19 and return the integer result."""
        ...


class _NoHooks:
    """Do not export transient baseline replies through caller instrumentation."""

    def __bool__(self):
        return False


async def _typed_call(agent, method, *args):
    """Return the same-call response and the actual method's native typed result.

    The middleware guards runtime context-window recovery *before* acall, while
    the caller's HTTP request hook also guards SDK/transport retries before send.
    Only parsed data is passed back to Predict; no reply/reasoning/native parts
    enter the ephemeral agent's event history. The original is inspected locally.
    """
    captured = []
    called = False

    async def one_shot(ctx, nxt):
        nonlocal called
        if called:
            raise RuntimeError("Configured typed check permits only one outbound call")
        called = True
        try:
            ctx = await nxt(ctx)
        except (ValueError, TypeError):
            # UnifiedLLM validates the provider JSON before returning. Do not
            # feed raw validation errors (which can quote reply values) into
            # Predict's failure metrics, feedback events or span attributes.
            raise RuntimeError("Typed check did not return a validated result") from None
        response = ctx.response
        captured.append(response)
        if response.parsed is None:
            raise RuntimeError("Typed check did not return a validated result")
        ctx.response = LLMResponse(
            parsed=response.parsed, usage=response.usage, finish_reason=response.finish_reason
        )
        return ctx

    remove = agent.event_manager.intercept("llm_call", one_shot)
    try:
        with hooks_scope(_NoHooks(), compose=False), use_span(INVALID_SPAN):
            result = await method(*args)
        if len(captured) != 1:
            raise RuntimeError("Configured typed check response was not captured")
        return captured[0], result
    finally:
        remove()
        agent.event_manager.clear()


def _probe_agent(cls, **kwargs):
    """Keep synchronous tracing setup announcements out of headless JSON.

    Do not redirect across an await: other tasks must retain their own output.
    Probe calls themselves run without caller instrumentation in _typed_call.
    """
    with hooks_scope(_NoHooks(), compose=False), use_span(INVALID_SPAN):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return cls(**kwargs)


async def typed_baseline(client):
    agent = _probe_agent(_TextBaseline, llm=client)
    return await _typed_call(agent, agent.compute)
