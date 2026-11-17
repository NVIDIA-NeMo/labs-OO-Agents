# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed test call: check that a configured model can return a typed result.

The "typed baseline" asks the model a trivial question (17 * 19) through a real
``Agent`` method with an ``int`` return type, so the normal NOOA structured
output path is exercised. The reply is only inspected by Connect: it is never
stored in an agent's event history, traced, or passed to the caller's hooks.
"""

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
    """Hooks object that is always falsy, so no caller hooks see the test call.

    Used with ``hooks_scope(..., compose=False)`` to disable instrumentation
    (tracing/export hooks) installed by the caller while a test call runs.
    """

    def __bool__(self):
        return False


async def _typed_call(agent, method, *args, observe_output_model=None):
    """Call ``method(*args)`` once and return ``(raw_response, typed_result)``.

    An ``llm_call`` interceptor makes sure exactly one model call is sent: a
    second call (for example a context-window recovery retry inside the
    runtime) raises before it reaches the client. Retries by the SDK or
    transport are blocked separately by the caller's HTTP request hook.

    The interceptor keeps the full response for Connect to inspect, but hands
    Predict a copy holding only the parsed value, usage and finish reason, so
    the reply text, reasoning and provider-specific parts never enter the
    temporary agent's event history. ``observe_output_model``, if given, is
    called with the output model the request used.

    Raises RuntimeError if the reply did not validate or was not captured.
    """
    captured = []
    called = False

    async def one_shot(ctx, nxt):
        nonlocal called
        if called:
            raise RuntimeError("Configured typed check permits only one outbound call")
        called = True
        if observe_output_model is not None:
            observe_output_model(ctx.params.get("output_model"))
        try:
            ctx = await nxt(ctx)
        except (ValueError, TypeError):
            # UnifiedLLM validates the provider's JSON before returning. Raw
            # validation errors can quote reply values, so replace them with a
            # generic error before Predict records them in failure metrics,
            # feedback events or span attributes.
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
    """Construct the test agent ``cls(**kwargs)`` with its console output discarded.

    Agent construction may print tracing setup messages, which would corrupt
    Connect's machine-readable (headless JSON) output. Output is redirected only
    around this synchronous constructor: redirecting across an ``await`` would
    also swallow output from other running tasks. The test calls themselves
    are isolated from caller instrumentation in ``_typed_call``.
    """
    with hooks_scope(_NoHooks(), compose=False), use_span(INVALID_SPAN):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return cls(**kwargs)


async def typed_baseline(client, *, observe_output_model=None):
    """Run the typed test call against ``client``; return ``(response, int_result)``."""
    agent = _probe_agent(_TextBaseline, llm=client)
    return await _typed_call(agent, agent.compute, observe_output_model=observe_output_model)
