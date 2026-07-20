# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test fixtures for subprocess worker tests.

This module is importable by subprocess workers (unlike test files).
"""


class DummyAgent:
    """Plain Python agent for testing — no LLM, no nooa."""

    def __init__(self, llm=None):
        pass

    async def classify(self, text: str) -> str:
        words = set(text.lower().split())
        if words & {"great", "love", "excellent", "good"}:
            return "positive"
        if words & {"terrible", "hate", "bad", "awful"}:
            return "negative"
        return "neutral"


class SlowAgent:
    """Agent that sleeps longer than the timeout — for testing span flush on timeout.

    Manually creates an OTel span to simulate what nooa hooks would do.
    This lets us test that the span gets ended and exported even when the task times out.
    """

    def __init__(self, llm=None):
        pass

    async def classify(self, text: str) -> str:
        import asyncio

        from opentelemetry import trace

        from nooa.tracing._hooks_impl import _get_active_spans

        # Simulate what before_agent_call() does: start a span and track it
        tracer = trace.get_tracer("test-slow-agent")
        span = tracer.start_span("method.classify")
        span.set_attribute("openinference.span.kind", "AGENT")
        span.set_attribute("agent.name", "SlowAgent")
        _get_active_spans()["slow_agent_call"] = span

        await asyncio.sleep(60)  # Will be cancelled by timeout
        return "positive"
