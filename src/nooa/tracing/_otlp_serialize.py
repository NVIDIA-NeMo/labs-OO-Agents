# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared OTLP JSON serialization helpers.

Converts OpenTelemetry SDK span objects to the OTLP JSON wire format
(``ExportTraceServiceRequest`` / ``TracesData`` schema).  Used by both
:class:`OtlpJsonHttpExporter` and :class:`OtlpJsonFileExporter`.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.trace import SpanKind, StatusCode


def format_trace_id(trace_id: int) -> str:
    return format(trace_id, "032x")


def format_span_id(span_id: int) -> str:
    return format(span_id, "016x")


def value_to_otlp(value: object) -> dict:
    """Convert a Python value to an OTLP ``AnyValue`` dict."""
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [value_to_otlp(v) for v in value]}}
    if isinstance(value, dict):
        return {
            "kvlistValue": {
                "values": [{"key": k, "value": value_to_otlp(v)} for k, v in value.items()]
            }
        }
    if isinstance(value, bytes):
        import base64

        return {"bytesValue": base64.b64encode(value).decode()}
    return {"stringValue": str(value)}


def attrs_to_otlp(
    attributes: Mapping[str, Any] | None,
    exclude_prefixes: tuple[str, ...] = (),
) -> list[dict]:
    """Convert an attribute mapping to a list of OTLP ``KeyValue`` dicts.

    When *exclude_prefixes* is non-empty, attributes whose key starts with any
    of the given prefixes are silently dropped.
    """
    if not attributes:
        return []
    if exclude_prefixes:
        return [
            {"key": k, "value": value_to_otlp(v)}
            for k, v in attributes.items()
            if not k.startswith(exclude_prefixes)
        ]
    return [{"key": k, "value": value_to_otlp(v)} for k, v in attributes.items()]


_SPAN_KIND_MAP = {
    SpanKind.INTERNAL: 1,
    SpanKind.SERVER: 2,
    SpanKind.CLIENT: 3,
    SpanKind.PRODUCER: 4,
    SpanKind.CONSUMER: 5,
}

_STATUS_CODE_MAP = {
    StatusCode.UNSET: 0,
    StatusCode.OK: 1,
    StatusCode.ERROR: 2,
}


def span_to_otlp(
    span: ReadableSpan,
    exclude_attr_prefixes: tuple[str, ...] = (),
    exclude_attr_prefixes_llm_only: tuple[str, ...] = (),
) -> dict:
    """Convert a :class:`ReadableSpan` to an OTLP JSON span dict.

    *exclude_attr_prefixes* are dropped from every span. *exclude_attr_prefixes_llm_only*
    are dropped only from ``LLM``-kind spans — used for ``input.value`` /
    ``output.value``, which the viewer reconstructs from the message journal for
    LLM spans but which are the sole carrier of I/O on framework spans (AGENT /
    CHAIN / TOOL), so they must survive the wire there.
    """
    ctx = span.context
    trace_id = format_trace_id(ctx.trace_id) if ctx else "0" * 32
    span_id = format_span_id(ctx.span_id) if ctx else "0" * 16
    effective_exclude = exclude_attr_prefixes
    if exclude_attr_prefixes_llm_only:
        attrs = span.attributes or {}
        if attrs.get("openinference.span.kind") == "LLM":
            effective_exclude = (*exclude_attr_prefixes, *exclude_attr_prefixes_llm_only)
    result: dict[str, Any] = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": span.name,
        "kind": _SPAN_KIND_MAP.get(span.kind, 0),
        "startTimeUnixNano": str(span.start_time),
        "endTimeUnixNano": str(span.end_time or span.start_time),
        "attributes": attrs_to_otlp(span.attributes, exclude_prefixes=effective_exclude),
        "status": {},
    }

    if span.parent and span.parent.span_id:
        result["parentSpanId"] = format_span_id(span.parent.span_id)

    if span.status:
        result["status"]["code"] = _STATUS_CODE_MAP.get(span.status.status_code, 0)
        if span.status.description:
            result["status"]["message"] = span.status.description

    if span.events:
        result["events"] = [
            {
                "timeUnixNano": str(evt.timestamp),
                "name": evt.name,
                "attributes": attrs_to_otlp(evt.attributes),
            }
            for evt in span.events
        ]

    return result


def build_resource_spans(
    spans: Sequence[ReadableSpan],
    resource_attrs_override: dict[str, Any] | None = None,
    exclude_attr_prefixes: tuple[str, ...] = (),
    exclude_attr_prefixes_llm_only: tuple[str, ...] = (),
) -> list[dict]:
    """Build the ``resourceSpans`` array for an OTLP ``TracesData`` envelope.

    Groups *spans* by resource and instrumentation scope.  If
    *resource_attrs_override* is provided, those key-value pairs are merged
    into (and override) the resource attributes of every group.

    When *exclude_attr_prefixes* is non-empty, span attributes whose key starts
    with any of the given prefixes are dropped from every span.
    *exclude_attr_prefixes_llm_only* are dropped only from ``LLM``-kind spans (see
    :func:`span_to_otlp`).
    """
    resource_groups: dict[int, list[ReadableSpan]] = defaultdict(list)
    for span in spans:
        resource_groups[id(span.resource)].append(span)

    resource_spans: list[dict] = []
    for _res_id, group in resource_groups.items():
        resource = group[0].resource
        attrs_dict = dict(resource.attributes) if resource else {}
        if resource_attrs_override:
            attrs_dict.update(resource_attrs_override)
        resource_attrs = attrs_to_otlp(attrs_dict)

        scope_groups: dict[str, list[ReadableSpan]] = defaultdict(list)
        for span in group:
            scope_key = span.instrumentation_scope.name if span.instrumentation_scope else ""
            scope_groups[scope_key].append(span)

        scope_spans: list[dict] = []
        for scope_name, scope_group in scope_groups.items():
            scope_obj: dict[str, str] = {"name": scope_name}
            if scope_group[0].instrumentation_scope:
                ver = scope_group[0].instrumentation_scope.version
                if ver:
                    scope_obj["version"] = ver

            scope_spans.append(
                {
                    "scope": scope_obj,
                    "spans": [
                        span_to_otlp(
                            s,
                            exclude_attr_prefixes=exclude_attr_prefixes,
                            exclude_attr_prefixes_llm_only=exclude_attr_prefixes_llm_only,
                        )
                        for s in scope_group
                    ],
                }
            )

        resource_spans.append(
            {
                "resource": {"attributes": resource_attrs},
                "scopeSpans": scope_spans,
            }
        )

    return resource_spans
