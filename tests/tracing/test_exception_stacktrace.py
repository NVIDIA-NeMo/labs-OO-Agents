# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for traceback capture on error spans."""

from __future__ import annotations

import tempfile
from typing import Any
from unittest.mock import MagicMock

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from otlp_test_helpers import read_all_otlp_jsonl_spans

from nooa.tracing._hooks_impl import OpenInferenceHooks
from nooa.tracing._otlp_file_exporter import OtlpJsonFileExporter


def _raise_nested_error() -> None:
    def inner() -> None:
        raise ValueError("boom")

    inner()


def _captured_exception() -> Exception:
    try:
        _raise_nested_error()
    except Exception as exc:
        return exc
    raise AssertionError("unreachable")


@pytest.fixture
def hooks_and_spans():
    with tempfile.TemporaryDirectory() as tmpdir:
        exporter = OtlpJsonFileExporter(tmpdir)
        provider = TracerProvider(resource=Resource(attributes={}))
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        hooks = OpenInferenceHooks(tracer=provider.get_tracer("test"))

        def read_spans() -> list[dict[str, Any]]:
            provider.force_flush()
            return read_all_otlp_jsonl_spans(tmpdir)

        yield hooks, read_spans
        provider.shutdown()


def _assert_exception_stacktrace(span: dict[str, Any]) -> None:
    attrs = span["attributes"]
    assert attrs["error.type"] == "ValueError"
    assert attrs["error.message"] == "boom"
    assert attrs["exception.type"] == "ValueError"
    assert attrs["exception.message"] == "boom"
    assert "exception.stacktrace" in attrs
    assert "Traceback (most recent call last)" in attrs["exception.stacktrace"]
    assert "_raise_nested_error" in attrs["exception.stacktrace"]
    assert "ValueError: boom" in attrs["exception.stacktrace"]


def test_agent_error_span_records_exception_stacktrace(hooks_and_spans):
    """Verify that agent error spans record exception attributes and stacktrace."""
    hooks, read_spans = hooks_and_spans
    agent = MagicMock()
    ctx = hooks.before_agent_call(
        agent=agent,
        method_name="solve",
        args=(),
        kwargs={},
        call_id="call-001",
        parent_call_id=None,
    )

    hooks.after_agent_call(
        agent=agent,
        method_name="solve",
        result=None,
        exception=_captured_exception(),
        context=ctx,
    )

    [span] = [s for s in read_spans() if s["name"] == "method.solve"]
    _assert_exception_stacktrace(span)


@pytest.mark.parametrize(
    ("span_name", "before", "after_kwargs"),
    [
        (
            "generation",
            lambda hooks, agent: hooks.before_generation(
                agent=agent,
                method_name="solve",
                strategy="PredictStrategy",
                generation_id="gen-001",
                parent_generation_id=None,
            ),
            {"method_name": "solve", "generation_id": "gen-001"},
        ),
        (
            "code_execution",
            lambda hooks, agent: hooks.before_code_execution(
                agent=agent,
                code="raise ValueError('boom')",
                execution_id="exec-001",
            ),
            {"code": "raise ValueError('boom')", "execution_id": "exec-001"},
        ),
        (
            "method_call.helper",
            lambda hooks, agent: hooks.before_method_invocation(
                agent=agent,
                method_name="helper",
                args=(),
                kwargs={},
                invocation_id="inv-001",
            ),
            {"method_name": "helper", "invocation_id": "inv-001"},
        ),
        (
            "tool_execution.return_result",
            lambda hooks, agent: hooks.before_tool_execution(
                agent=agent,
                tool_name="return_result",
                arguments={},
                execution_id="tool-001",
            ),
            {"tool_name": "return_result", "arguments": {}, "execution_id": "tool-001"},
        ),
    ],
)
def test_non_agent_error_spans_record_exception_stacktrace(
    hooks_and_spans,
    span_name: str,
    before,
    after_kwargs: dict[str, Any],
):
    """Verify that non-agent error spans record exception attributes and stacktrace."""
    hooks, read_spans = hooks_and_spans
    agent = MagicMock()
    ctx = before(hooks, agent)
    exception = _captured_exception()

    after_method = {
        "generation": hooks.after_generation,
        "code_execution": hooks.after_code_execution,
        "method_call.helper": hooks.after_method_invocation,
        "tool_execution.return_result": hooks.after_tool_execution,
    }[span_name]
    after_method(
        agent=agent,
        result=None,
        exception=exception,
        context=ctx,
        **after_kwargs,
    )

    [span] = [s for s in read_spans() if s["name"] == span_name]
    _assert_exception_stacktrace(span)
