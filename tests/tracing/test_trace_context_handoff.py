# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""capture_trace_context / attach_trace_context hand a trace to a fresh context."""

import contextvars

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from nooa.tracing import (
    attach_trace_context,
    capture_trace_context,
    get_span_attributes,
    session_scope,
    with_span_attributes,
)
from nooa.tracing._hooks_impl import _current_code_execution
from nooa.tracing._session import get_session
from nooa.tracing._session_processor import SessionSpanProcessor


def _provider():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SessionSpanProcessor())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("t"), exporter


def test_a_fresh_context_continues_the_captured_trace_under_the_running_cell():
    tracer, exporter = _provider()
    with session_scope("root-session"), tracer.start_as_current_span("turn"):
        cell = tracer.start_span("code_execution")
        token = _current_code_execution.set((cell, None))
        try:
            captured = capture_trace_context()
        finally:
            _current_code_execution.reset(token)
        cell.end()

    def child() -> None:
        attach_trace_context(captured, {"nooa.atom.session.id": "child-1"})
        assert get_session() == "root-session"
        with tracer.start_as_current_span("child.turn"):
            pass

    contextvars.Context().run(child)
    spans = {s.name: s for s in exporter.get_finished_spans()}
    child_span = spans["child.turn"]
    assert child_span.parent.span_id == spans["code_execution"].context.span_id
    assert child_span.context.trace_id == spans["turn"].context.trace_id
    assert child_span.attributes["session.id"] == "root-session"
    assert child_span.attributes["nooa.atom.session.id"] == "child-1"
    assert "nooa.atom.session.id" not in spans["turn"].attributes


def test_without_a_running_cell_the_current_span_is_captured():
    tracer, exporter = _provider()
    with tracer.start_as_current_span("method") as method:
        captured = capture_trace_context()
    assert trace.get_current_span(captured) is method


def test_span_attributes_merge_and_a_later_key_wins():
    first = with_span_attributes({"a": 1, "b": "x"})
    second = with_span_attributes({"b": "y"}, first)
    assert get_span_attributes(second) == {"a": 1, "b": "y"}
    assert get_span_attributes(first) == {"a": 1, "b": "x"}
    assert get_span_attributes() == {}
