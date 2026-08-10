# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end span-tree assertions for generator agent methods (issue #38).

The generator tests in ``tests/test_metaclass.py`` assert through a mocked
``InstrumentationHooks``, i.e. through the *input* the wrapper hands the hook —
not the span tree the hook actually builds. That is the right level for most of
them, but it cannot catch a fault in the step in between: ``before_agent_call``
resolves a span's parent out of a **ContextVar-scoped** registry
(``_hooks_impl._get_active_spans``), and generators are the one construct whose
before- and after-hook can run in different contexts, because the body is
resumed by whoever happens to be draining it.

So this file asserts the exported spans directly.
"""

import pytest

from nooa import Agent
from nooa.runtime.hooks import set_hooks
from nooa.unifiedllm import FakeLLMClient


@pytest.fixture
def in_memory_spans():
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from nooa.tracing import NemoOOAgentsInstrumentor

    exporter = InMemorySpanExporter()
    provider = trace.get_tracer_provider()
    if not hasattr(provider, "add_span_processor"):
        provider = TracerProvider()
        trace.set_tracer_provider(provider)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    NemoOOAgentsInstrumentor().instrument(tracer_provider=provider)
    yield exporter
    set_hooks(None)


class _GeneratorAgent(Agent):
    async def produce(self, n: int):
        """Async generator: calls a helper from inside the body, then yields."""
        for i in range(n):
            await self.in_body(i)
            yield i

    async def in_body(self, i: int) -> int:
        """Called by the generator body — must parent to the generator."""
        return i

    async def between_yields(self, v: int) -> int:
        """Called by the consumer between yields — must NOT parent to the generator."""
        return v

    async def drain(self) -> list[int]:
        """Consumer. Issues `between_yields` itself; issues no `in_body` call."""
        out = []
        async for v in self.produce(2):
            await self.between_yields(v)
            out.append(v)
        return out


def _parent_namer(spans):
    by_id = {s.context.span_id: s for s in spans}

    def parent_name(span):
        parent = span.parent
        return by_id[parent.span_id].name if (parent and parent.span_id in by_id) else None

    return parent_name


@pytest.mark.asyncio
async def test_generator_body_calls_nest_under_the_generator_span(in_memory_spans):
    """The exported tree must match who actually issued each call."""
    agent = _GeneratorAgent(llm=FakeLLMClient())
    assert await agent.drain() == [0, 1]

    spans = in_memory_spans.get_finished_spans()
    parent_name = _parent_namer(spans)

    produce = [s for s in spans if s.name == "method.produce"]
    in_body = [s for s in spans if s.name == "method.in_body"]
    between = [s for s in spans if s.name == "method.between_yields"]

    assert len(produce) == 1, f"expected 1 produce span, got {len(produce)}"
    assert len(in_body) == 2, f"expected 2 in_body spans, got {len(in_body)}"
    assert len(between) == 2, f"expected 2 between_yields spans, got {len(between)}"

    assert parent_name(produce[0]) == "method.drain"

    # The defect in #38: these were parented to `drain`, which never issued them.
    for span in in_body:
        assert parent_name(span) == "method.produce", (
            f"in_body parented to {parent_name(span)!r}, expected 'method.produce'"
        )

    # The mirror-image error: consumer work must not be captured by the generator.
    for span in between:
        assert parent_name(span) == "method.drain", (
            f"between_yields parented to {parent_name(span)!r}, expected 'method.drain'"
        )


@pytest.mark.asyncio
async def test_abandoned_generator_span_is_still_exported(in_memory_spans):
    """A generator closed by the asyncio finalizer must still export its span.

    This is the cross-context case: the before-hook ran in the consumer's
    context, the after-hook runs from the async-generator finalizer.
    """
    import asyncio

    from nooa.tracing._hooks_impl import _get_active_spans

    agent = _GeneratorAgent(llm=FakeLLMClient())

    async def consumer():
        async for _v in agent.produce(1000):
            break

    await consumer()

    for _ in range(10):
        if any(s.name == "method.produce" for s in in_memory_spans.get_finished_spans()):
            break
        await asyncio.sleep(0)

    spans = in_memory_spans.get_finished_spans()
    assert any(s.name == "method.produce" for s in spans), (
        "abandoned generator never exported its span"
    )
    assert not _get_active_spans(), f"span registry not drained: {_get_active_spans()}"
