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

import asyncio
import contextvars

import pytest
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from nooa import Agent
from nooa.runtime.hooks import get_hooks
from nooa.unifiedllm import FakeLLMClient


@pytest.fixture
def in_memory_spans(monkeypatch):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from nooa.tracing import NemoOOAgentsInstrumentor

    # Keep Agent.__init__ from replacing this fixture-local backend with the
    # automatically discovered dev-viewer backend on the first instantiation.
    monkeypatch.setattr("nooa.agent._auto_tracing_attempted", True)

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = NemoOOAgentsInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    try:
        yield exporter
    finally:
        instrumentor.uninstrument()
        provider.shutdown()


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


@pytest.mark.asyncio
async def test_generator_span_reactivates_across_tasks(in_memory_spans):
    """Each resume restores native OTel context without leaking it to consumers."""
    from nooa.tracing._hooks_impl import _get_active_spans

    agent = _GeneratorAgent(llm=FakeLLMClient())
    stream = agent.produce(2)
    first_item_ready = asyncio.Event()
    inspect_starting_registry = asyncio.Event()

    async def first_resume():
        item = await anext(stream)
        first_item_ready.set()
        await inspect_starting_registry.wait()
        return item, dict(_get_active_spans())

    starting_task = asyncio.create_task(first_resume())
    await first_item_ready.wait()

    # The generator is suspended, so neither its framework context nor its
    # native OTel context may bleed into work performed by the consumer.
    assert not trace.get_current_span().get_span_context().is_valid
    assert await agent.between_yields(99) == 99

    assert await asyncio.create_task(anext(stream)) == 1
    with pytest.raises(StopAsyncIteration):
        await asyncio.create_task(anext(stream))

    # Completion happened in another task. It must still remove the lifecycle
    # span from the registry belonging to the task that started the stream.
    inspect_starting_registry.set()
    first_item, starting_registry = await starting_task
    assert first_item == 0
    assert starting_registry == {}

    spans = in_memory_spans.get_finished_spans()
    parent_name = _parent_namer(spans)
    produce = next(s for s in spans if s.name == "method.produce")
    children = [s for s in spans if s.name == "method.in_body"]
    consumer = next(s for s in spans if s.name == "method.between_yields")

    assert len(children) == 2
    assert all(parent_name(span) == "method.produce" for span in children)
    assert consumer.parent is None
    assert produce.status.status_code is StatusCode.OK
    assert not [event for event in produce.events if event.name == "exception"]


@pytest.mark.asyncio
async def test_generator_span_reactivates_in_independently_rooted_task_context(in_memory_spans):
    """The originating hooks survive a resume in a task that inherited no hooks."""
    from nooa.tracing._hooks_impl import _get_active_spans

    agent = _GeneratorAgent(llm=FakeLLMClient())
    stream = agent.produce(2)

    assert await anext(stream) == 0
    assert len(_get_active_spans()) == 1

    async def resume_once():
        item = await anext(stream)
        return item, get_hooks()

    item, consumer_hooks = await asyncio.create_task(resume_once(), context=contextvars.Context())
    assert item == 1
    assert consumer_hooks is None

    with pytest.raises(StopAsyncIteration):
        await asyncio.create_task(anext(stream), context=contextvars.Context())

    spans = in_memory_spans.get_finished_spans()
    parent_name = _parent_namer(spans)
    [produce] = [span for span in spans if span.name == "method.produce"]
    children = [span for span in spans if span.name == "method.in_body"]

    assert len(children) == 2
    assert all(parent_name(span) == "method.produce" for span in children)
    assert produce.status.status_code is StatusCode.OK
    assert not _get_active_spans()


@pytest.mark.asyncio
async def test_cross_task_aclose_parents_traced_cleanup(in_memory_spans):
    """A cleanup method run by aclose remains inside the generator lifecycle span."""

    class CleanupAgent(Agent):
        async def cleanup(self) -> None:
            return None

        async def values(self):
            try:
                yield 1
            finally:
                await self.cleanup()

    agent = CleanupAgent(llm=FakeLLMClient())
    stream = agent.values()
    assert await asyncio.create_task(anext(stream)) == 1
    await asyncio.create_task(stream.aclose())

    spans = in_memory_spans.get_finished_spans()
    stream_span = next(s for s in spans if s.name == "method.values")
    cleanup_span = next(s for s in spans if s.name == "method.cleanup")

    assert cleanup_span.parent is not None
    assert cleanup_span.parent.span_id == stream_span.context.span_id


@pytest.mark.asyncio
async def test_cross_task_generator_cancellation_marks_span_error(in_memory_spans):
    """Cancellation during a later resume ends the lifecycle span as an error."""

    class SlowAgent(Agent):
        async def values(self):
            yield "ready"
            await asyncio.Event().wait()

    stream = SlowAgent(llm=FakeLLMClient()).values()
    assert await asyncio.create_task(anext(stream)) == "ready"

    resume = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    resume.cancel()
    with pytest.raises(asyncio.CancelledError):
        await resume

    [stream_span] = [
        span for span in in_memory_spans.get_finished_spans() if span.name == "method.values"
    ]
    assert stream_span.status.status_code is StatusCode.ERROR
