# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trace previews: span ``input.value`` / ``output.value`` go through ``nooa.snapshot``.

Replaces the tests of ``_safe_serialize``, ``_safe_json_value`` and
``_safe_serialize_execution_result`` (gl-74): the preview is
``render_trace(capture(value))``'s JSON, always valid and at most 50,000
characters, plus the ``nooa.{input,output}.preview.*`` attributes.
"""

import json

from openinference.semconv.trace import OpenInferenceMimeTypeValues, SpanAttributes

from nooa.events import ExecutionResult
from nooa.snapshot import capture
from nooa.tracing._hooks_impl import (
    _PREVIEW_LIMITS,
    _call_snapshot,
    _code_result_snapshot,
    _error_message,
    _set_preview,
)

MAX = _PREVIEW_LIMITS.max_chars


class _Span:
    """Records ``set_attribute`` calls."""

    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value


def _output(value):
    span = _Span()
    text = _set_preview(span, "output", capture(value, _PREVIEW_LIMITS))
    assert span.attributes[SpanAttributes.OUTPUT_VALUE] == text
    assert (
        span.attributes[SpanAttributes.OUTPUT_MIME_TYPE] == OpenInferenceMimeTypeValues.JSON.value
    )
    assert len(text) <= MAX
    return json.loads(text), span.attributes


def test_small_values_are_json_and_complete():
    for value in ("hello", 42, None, {"key": "value"}, [1, 2.5, True]):
        parsed, attributes = _output(value)
        assert parsed == value
        assert attributes["nooa.output.preview.incomplete"] is False
        assert "nooa.output.preview.paths" not in attributes


def test_large_string_is_cut_and_says_so():
    parsed, attributes = _output("x" * 200_000)
    assert set(parsed) == {"x"} and 40_000 < len(parsed) < MAX
    assert attributes["nooa.output.preview.incomplete"] is True
    assert attributes["nooa.output.preview.paths"] == ("",)
    assert attributes["nooa.output.preview.reasons"] == ("max_length",)


def test_large_nested_payload_is_bounded_and_names_the_cut_paths():
    obj = {"files": [{"name": f"f{i}", "content": "x" * 100_000} for i in range(40)]}
    parsed, attributes = _output(obj)
    assert parsed["files"][0]["name"] == "f0"
    assert attributes["nooa.output.preview.incomplete"] is True
    assert "/files/0/content" in attributes["nooa.output.preview.paths"]


def test_broken_repr_is_never_called():
    class Bomb:
        def __repr__(self):
            raise RuntimeError("boom")

    parsed, attributes = _output(Bomb())
    # The opaque label is module-qualified (PR 2c); traces never run repr.
    assert (
        parsed
        == "<opaque: tests.tracing.test_trace_previews.test_broken_repr_is_never_called.<locals>.Bomb>"
    )
    assert attributes["nooa.output.preview.reasons"] == ("opaque",)


def test_deep_nesting_is_depth_limited():
    obj: dict = {}
    current = obj
    for _ in range(500):
        current["nested"] = {}
        current = current["nested"]
    parsed, attributes = _output(obj)
    assert json.dumps(parsed).count("nested") <= _PREVIEW_LIMITS.max_depth
    assert "max_depth" in attributes["nooa.output.preview.reasons"]


def test_pydantic_model_shows_its_fields():
    parsed, _ = _output(ExecutionResult(stdout="hello", stderr="", returned_value=42))
    assert parsed["stdout"] == "hello"
    assert parsed["returned_value"] == 42


def test_call_input_keeps_args_and_kwargs_on_overflow():
    span = _Span()
    text = _set_preview(span, "input", _call_snapshot(["x" * 200_000], {"k": "y" * 200_000}))
    parsed = json.loads(text)
    assert set(parsed) == {"args", "kwargs"}
    assert len(text) <= MAX
    assert span.attributes[SpanAttributes.INPUT_VALUE] == text
    assert span.attributes["nooa.input.preview.incomplete"] is True


def test_execution_result_keeps_every_key():
    snapshot = _code_result_snapshot(
        ExecutionResult(stdout="s" * 100_000, stderr="e" * 100_000, returned_value=[1] * 100_000)
    )
    span = _Span()
    parsed = json.loads(_set_preview(span, "output", snapshot))
    assert set(parsed) == {"stdout", "stderr", "returned_value"}
    assert parsed["stdout"].startswith("s")
    assert span.attributes["nooa.output.preview.incomplete"] is True


def test_execution_result_without_return_shows_null():
    snapshot = _code_result_snapshot(ExecutionResult(stdout="hi\n", stderr=""))
    parsed = json.loads(_set_preview(_Span(), "output", snapshot))
    assert parsed == {"stdout": "hi\n", "stderr": "", "returned_value": None}


def test_error_message_keeps_both_ends():
    message = _error_message(ValueError("a" * 3_000 + "b" * 3_000))
    assert message.startswith("a" * 2_500) and message.endswith("b" * 2_500)
    assert "[... 1,000 chars omitted; full output not kept]" in message
    assert _error_message(ValueError("short")) == "short"


# ---- hook-level: real spans through OpenInferenceHooks (review r7, #579) ----------


def _hooks():
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from nooa.tracing._hooks_impl import OpenInferenceHooks

    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return OpenInferenceHooks(provider.get_tracer("test")), exporter


class _Agent:
    pass


def _run_code(code, result):
    hooks, exporter = _hooks()
    agent = _Agent()
    context = hooks.before_code_execution(agent=agent, code=code, execution_id="exec-1")
    hooks.after_code_execution(
        agent=agent,
        code=code,
        result=result,
        exception=None,
        context=context,
        execution_id="exec-1",
    )
    (span,) = exporter.get_finished_spans()
    return span


def test_long_code_preview_is_cut_by_the_snapshot_and_says_so():
    code = "#" + "x" * 10_000  # 10,001 chars: one over the code preview's string limit
    span = _run_code(code, ExecutionResult(stdout="ok"))
    attributes = span.attributes
    parsed = json.loads(attributes[SpanAttributes.INPUT_VALUE])  # valid JSON
    assert parsed == {"code": code[:10_000]}
    assert attributes["code.length"] == 10_001
    assert attributes["nooa.input.preview.incomplete"] is True
    assert attributes["nooa.input.preview.paths"] == ("/code",)
    assert attributes["nooa.input.preview.reasons"] == ("max_length",)
    # The tool-call arguments are the same JSON.
    assert json.loads(attributes["tool_call.function.arguments"]) == parsed


def test_code_at_the_limit_is_complete():
    code = "y" * 10_000
    attributes = _run_code(code, ExecutionResult()).attributes
    assert json.loads(attributes[SpanAttributes.INPUT_VALUE]) == {"code": code}
    assert attributes["nooa.input.preview.incomplete"] is False


def test_error_status_description_is_bounded():
    from opentelemetry.trace import StatusCode

    from nooa.tracing._hooks_impl import _record_error

    hooks, exporter = _hooks()
    with hooks.tracer.start_as_current_span("error") as span:
        _record_error(span, ValueError("e" * 1_000_000))
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code == StatusCode.ERROR
    assert span.status.description == span.attributes["error.message"]
    assert len(span.status.description) < 5_100  # 5,000 kept plus the cut line
    assert "chars omitted; full output not kept]" in span.status.description


def test_a_result_like_object_is_captured_without_calling_its_properties():
    calls = []

    class LooksLikeAResult:
        def __init__(self):
            self.name = "plain data"

        @property
        def stdout(self):
            calls.append("stdout")
            raise RuntimeError("never read me")

        @property
        def returned_value(self):
            calls.append("returned_value")
            return 1

    span = _run_code("value", LooksLikeAResult())
    assert calls == []
    # Present and valid JSON: an ordinary capture of its public attributes.
    preview = json.loads(span.attributes[SpanAttributes.OUTPUT_VALUE])
    assert "plain data" in json.dumps(preview) and "stdout" not in json.dumps(preview)
