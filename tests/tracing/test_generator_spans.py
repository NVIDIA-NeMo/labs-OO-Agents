# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenTelemetry lifecycle tests for deterministic async-generator methods."""

import asyncio
from contextlib import aclosing

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from nooa import Agent
from nooa.runtime.hooks import set_hooks
from nooa.tracing._hooks_impl import OpenInferenceHooks, end_active_spans
from nooa.unifiedllm import FakeLLMClient


def _install_test_tracing() -> tuple[TracerProvider, InMemorySpanExporter]:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    set_hooks(OpenInferenceHooks(provider.get_tracer("async-generator-test")))
    return provider, exporter


@pytest.mark.asyncio
async def test_async_generator_span_parents_children_across_tasks():
    """Every resume reactivates the lifecycle span in the task doing the work."""

    class StreamAgent(Agent, llm=FakeLLMClient()):
        async def child(self, value: int) -> int:
            return value * 10

        async def values(self):
            yield await self.child(1)
            yield await self.child(2)

        async def consumer_work(self) -> None:
            return None

    agent = StreamAgent()
    provider, exporter = _install_test_tracing()
    try:
        stream = agent.values()
        first_item_ready = asyncio.Event()
        check_starting_registry = asyncio.Event()

        async def first_resume():
            item = await anext(stream)
            first_item_ready.set()
            await check_starting_registry.wait()
            return item, end_active_spans()

        # Each task inherits the consumer's empty tracing registry. The stream
        # wrapper must carry and reactivate its span explicitly across resumes.
        starting_task = asyncio.create_task(first_resume())
        await first_item_ready.wait()
        assert not trace.get_current_span().get_span_context().is_valid
        await agent.consumer_work()
        assert await asyncio.create_task(anext(stream)) == 20
        with pytest.raises(StopAsyncIteration):
            await asyncio.create_task(anext(stream))

        # Completion in another task must remove the span from the registry
        # belonging to the task that started it, rather than leaving stale state.
        check_starting_registry.set()
        first_item, remaining_spans = await starting_task
        assert first_item == 10
        assert remaining_spans == 0

        spans = exporter.get_finished_spans()
    finally:
        set_hooks(None)
        provider.shutdown()

    stream_span = next(span for span in spans if span.name == "method.values")
    child_spans = [span for span in spans if span.name == "method.child"]
    consumer_span = next(span for span in spans if span.name == "method.consumer_work")

    assert len(child_spans) == 2
    assert all(
        span.parent is not None and span.parent.span_id == stream_span.context.span_id
        for span in child_spans
    )
    assert consumer_span.parent is None
    assert stream_span.status.status_code is StatusCode.OK


@pytest.mark.asyncio
async def test_async_generator_aclosing_ends_span_and_runs_cleanup():
    """Explicit early close completes both Python cleanup and the lifecycle span."""

    class StreamAgent(Agent, llm=FakeLLMClient()):
        cleaned_up = False

        async def values(self):
            try:
                yield 1
                yield 2
            finally:
                self.cleaned_up = True

    agent = StreamAgent()
    provider, exporter = _install_test_tracing()
    try:
        async with aclosing(agent.values()) as stream:
            async for item in stream:
                assert item == 1
                break
        spans = exporter.get_finished_spans()
    finally:
        set_hooks(None)
        provider.shutdown()

    [stream_span] = [span for span in spans if span.name == "method.values"]
    assert agent.cleaned_up is True
    assert stream_span.status.status_code is StatusCode.OK


@pytest.mark.asyncio
async def test_async_generator_aclose_parents_traced_cleanup_across_tasks():
    """Cleanup executed by aclose remains inside the lifecycle span."""

    class StreamAgent(Agent, llm=FakeLLMClient()):
        async def cleanup(self) -> None:
            return None

        async def values(self):
            try:
                yield 1
            finally:
                await self.cleanup()

    agent = StreamAgent()
    provider, exporter = _install_test_tracing()
    try:
        stream = agent.values()
        assert await asyncio.create_task(anext(stream)) == 1

        async def close_stream() -> None:
            await stream.aclose()

        await asyncio.create_task(close_stream())
        spans = exporter.get_finished_spans()
    finally:
        set_hooks(None)
        provider.shutdown()

    stream_span = next(span for span in spans if span.name == "method.values")
    cleanup_span = next(span for span in spans if span.name == "method.cleanup")
    assert cleanup_span.parent is not None
    assert cleanup_span.parent.span_id == stream_span.context.span_id


@pytest.mark.asyncio
async def test_async_generator_cancellation_ends_span_as_error():
    """Cancellation in a later task closes and marks the lifecycle span."""

    class StreamAgent(Agent, llm=FakeLLMClient()):
        async def values(self):
            yield "ready"
            await asyncio.Event().wait()

    agent = StreamAgent()
    provider, exporter = _install_test_tracing()
    try:
        stream = agent.values()
        assert await asyncio.create_task(anext(stream)) == "ready"

        resume = asyncio.create_task(anext(stream))
        await asyncio.sleep(0)
        resume.cancel()
        with pytest.raises(asyncio.CancelledError):
            await resume
        spans = exporter.get_finished_spans()
    finally:
        set_hooks(None)
        provider.shutdown()

    [stream_span] = [span for span in spans if span.name == "method.values"]
    assert stream_span.status.status_code is StatusCode.ERROR
