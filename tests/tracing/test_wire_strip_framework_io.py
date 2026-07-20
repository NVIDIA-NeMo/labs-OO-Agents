# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The journal/wire strip must NOT drop input.value/output.value from framework spans.

When traces are sent to a viewer the journal exporter strips
``input.value``/``output.value`` because, for LLM spans, the viewer reconstructs
them from the message journal. Framework spans (AGENT / CHAIN / TOOL) are NOT in
the journal — those attributes are the sole carrier of the method I/O and executed
code, so they must survive the wire. Otherwise consumers that read traces via the
viewer (e.g. the capability scorers' code/methodology judges) see empty inputs and
outputs.

This guards the span-kind-aware strip in ``build_resource_spans`` /
``span_to_otlp``.
"""

from __future__ import annotations

from types import SimpleNamespace

from opentelemetry.sdk.resources import Resource
from opentelemetry.trace import SpanKind

from nooa.tracing._otlp_http_exporter import OtlpJsonHttpExporter
from nooa.tracing._otlp_serialize import build_resource_spans

_RESOURCE = Resource(attributes={})


def _fake_span(kind_value: str, name: str):
    """A minimal ReadableSpan-like object carrying input.value/output.value."""
    return SimpleNamespace(
        context=SimpleNamespace(trace_id=0x1234, span_id=0x5678),
        parent=None,
        name=name,
        kind=SpanKind.INTERNAL,
        start_time=1,
        end_time=2,
        status=None,
        events=None,
        resource=_RESOURCE,
        instrumentation_scope=SimpleNamespace(name="test", version=None),
        attributes={
            "openinference.span.kind": kind_value,
            "input.value": f"INPUT_{name}",
            "output.value": f"OUTPUT_{name}",
            "input.mime_type": "application/json",
        },
    )


def _attrs_for(resource_spans: list[dict], span_name: str) -> dict[str, str]:
    for rs in resource_spans:
        for ss in rs.get("scopeSpans", []):
            for sp in ss.get("spans", []):
                if sp.get("name") == span_name:
                    return {
                        a["key"]: a["value"].get("stringValue") for a in sp.get("attributes", [])
                    }
    raise AssertionError(f"span {span_name!r} not found")


def test_value_keys_stripped_from_llm_spans_only():
    """With the LLM-only exclusion, ``input.value``/``output.value`` are dropped
    from LLM spans (reconstructed by the viewer) but kept on framework spans."""
    llm = _fake_span("LLM", "acompletion")
    tool = _fake_span("TOOL", "code_execution")
    chain = _fake_span("CHAIN", "generation")
    agent = _fake_span("AGENT", "method.run")

    resource_spans = build_resource_spans(
        [llm, tool, chain, agent],
        exclude_attr_prefixes=("llm.input_messages.", "llm.output_messages."),
        exclude_attr_prefixes_llm_only=("input.value", "output.value"),
    )

    # LLM span: value keys stripped (the viewer reconstructs them).
    llm_attrs = _attrs_for(resource_spans, "acompletion")
    assert "input.value" not in llm_attrs
    assert "output.value" not in llm_attrs

    # Framework spans: value keys preserved (sole carrier of their I/O).
    for span_name in ("code_execution", "generation", "method.run"):
        attrs = _attrs_for(resource_spans, span_name)
        assert attrs.get("input.value") == f"INPUT_{span_name}", span_name
        assert attrs.get("output.value") == f"OUTPUT_{span_name}", span_name
        # non-value keys are untouched on all spans
        assert attrs.get("input.mime_type") == "application/json"


def test_journal_exporter_strips_llm_messages_but_keeps_framework_values():
    """End-to-end through the HTTP exporter's strip path (strip_llm_messages=True)."""
    exporter = OtlpJsonHttpExporter(strip_llm_messages=True)
    captured: dict = {}

    def _fake_send(spans, payload):
        captured["payload"] = payload
        from opentelemetry.sdk.trace.export import SpanExportResult

        return SpanExportResult.SUCCESS

    exporter._send_payload = _fake_send  # type: ignore[method-assign]
    exporter.export([_fake_span("LLM", "acompletion"), _fake_span("TOOL", "code_execution")])

    rs = captured["payload"]["resourceSpans"]
    assert "input.value" not in _attrs_for(rs, "acompletion")
    assert _attrs_for(rs, "code_execution").get("input.value") == "INPUT_code_execution"
