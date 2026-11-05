# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Time-indexed, completed-call accounting from local OpenInference traces.

This is a report of recorded spans, not a provider bill or an estimate of tokens
generated partway through a call. No prompt, response, or reasoning text is exported.
"""

from __future__ import annotations

import csv
import io
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from typing import Any

from nooa.trace_explorer.explorer import _load_spans


@dataclass
class UsageTotals:
    """Known counts and missing-data indicators for completed recorded calls."""

    completed_calls: int = 0
    known_input_tokens: int = 0
    known_output_tokens: int = 0
    known_total_tokens: int = 0
    known_cached_input_tokens: int = 0
    missing_usage_calls: int = 0
    missing_cached_input_calls: int = 0

    def _add(self, call: _Call) -> None:
        """Accumulate known values without treating absent fields as reported zero."""
        self.completed_calls += 1
        self.known_input_tokens += call.input_tokens or 0
        self.known_output_tokens += call.output_tokens or 0
        self.known_total_tokens += call.total_tokens or 0
        self.known_cached_input_tokens += call.cached_input_tokens or 0
        self.missing_usage_calls += any(
            value is None for value in (call.input_tokens, call.output_tokens, call.total_tokens)
        )
        self.missing_cached_input_calls += call.cached_input_tokens is None


@dataclass(frozen=True)
class UsagePoint:
    """One completed call and a snapshot of cumulative known usage after it."""

    elapsed_seconds: float
    model: str
    trace_id: str
    span_id: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cached_input_tokens: int | None
    cumulative: UsageTotals


@dataclass
class UsageReport:
    """Versioned accounting for a file, with an optional inclusive time cutoff."""

    start_time_unix_nano: str
    recorded_duration_seconds: float
    until_seconds: float | None
    in_flight_calls: int = 0
    deduplicated_spans: int = 0
    totals: UsageTotals = field(default_factory=UsageTotals)
    by_model: dict[str, UsageTotals] = field(default_factory=dict)
    timeline: list[UsagePoint] = field(default_factory=list)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible data, preserving absent per-call counts as null."""
        return asdict(self)

    def to_csv(self) -> str:
        """Return call rows and a final summary row, including for zero-call reports.

        Unknown per-call counts are empty fields. Cumulative fields are known
        counts only. The summary carries file timing and recorded in-flight
        counts. Text fields beginning like spreadsheet formulas are apostrophe-
        prefixed; JSON preserves the original identifiers and model names.
        """
        output = io.StringIO(newline="")
        total_fields = list(asdict(self.totals))
        columns = [
            "record_type",
            "elapsed_seconds",
            "model",
            "trace_id",
            "span_id",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_input_tokens",
            *total_fields,
            "in_flight_calls",
            "deduplicated_spans",
            "start_time_unix_nano",
            "recorded_duration_seconds",
        ]
        writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for point in self.timeline:
            row = asdict(point)
            row.update(row.pop("cumulative"))
            for key in ("model", "trace_id", "span_id"):
                value = row[key]
                if value.lstrip().startswith(("=", "+", "-", "@")) or value.startswith(
                    ("\t", "\r", "\n")
                ):
                    row[key] = "'" + value
            writer.writerow({"record_type": "call", **row})
        writer.writerow(
            {
                "record_type": "summary",
                "elapsed_seconds": self.until_seconds
                if self.until_seconds is not None
                else self.recorded_duration_seconds,
                **asdict(self.totals),
                "in_flight_calls": self.in_flight_calls,
                "deduplicated_spans": self.deduplicated_spans,
                "start_time_unix_nano": self.start_time_unix_nano,
                "recorded_duration_seconds": self.recorded_duration_seconds,
            }
        )
        return output.getvalue()


@dataclass(frozen=True)
class _Call:
    """Accounting-relevant fields used for identity checks and stable ordering."""

    trace_id: str
    span_id: str
    model: str
    start_ns: int
    end_ns: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cached_input_tokens: int | None


def _count(attrs: dict[str, Any], key: str) -> int | None:
    """Read a non-negative integer, distinguishing missing values from zero."""
    value = attrs.get(key)
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError(f"{key} must be a non-negative integer, got {value!r}")
    return value


def _recorded_call(span: dict[str, Any]) -> _Call:
    """Validate timing and extract accounting metadata from strictly loaded spans."""
    trace_id, span_id = span["trace_id"], span["span_id"]
    trace_id.encode("utf-8")
    span_id.encode("utf-8")
    start, end = span["start_time"], span["end_time"]
    if start <= 0 or (end and end < start):
        raise ValueError(f"Invalid LLM timing for span {span_id}: start={start}, end={end}")
    attrs = span["attributes"]
    model = attrs.get("llm.model_name")
    if model is None or model == "":
        model = "unknown"
    if not isinstance(model, str):
        raise ValueError(f"llm.model_name must be a string for span {span_id}")
    model.encode("utf-8")
    prompt = _count(attrs, "llm.token_count.prompt")
    completion = _count(attrs, "llm.token_count.completion")
    total = _count(attrs, "llm.token_count.total")
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return _Call(
        trace_id,
        span_id,
        model,
        start,
        end,
        prompt,
        completion,
        total,
        _count(attrs, "llm.token_count.prompt_details.cache_read"),
    )


def _cutoff_ns(seconds: float) -> int:
    """Convert finite non-negative decimal seconds, avoiding float boundary drift."""
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
        raise ValueError("until_seconds must be a finite non-negative number")
    value = Decimal(str(seconds))
    if not value.is_finite() or value < 0:
        raise ValueError("until_seconds must be a finite non-negative number")
    numerator, denominator = value.as_integer_ratio()
    return numerator * 1_000_000_000 // denominator


def load_usage_report(trace_path: str | Path, *, until_seconds: float | None = None) -> UsageReport:
    """Report completed OpenInference LLM spans from a local OTLP JSONL file.

    Time zero is the earliest positive start timestamp in the file, including
    orchestrator/tool spans. The cutoff is inclusive by completion time. Calls
    recorded as crossing it (or lacking an end timestamp) are counted separately,
    without assigning partial token consumption. Exporters may not record running
    calls at all, so that count is not proof every in-flight call is represented.

    Missing usage is null per call and tracked in aggregate counters. A missing
    total is derived only when both input and output counts are available; a
    reported total takes precedence. Cache/reasoning subsets are never added.
    Repeated trace/span IDs with identical accounting are counted once; conflicting
    records raise. Invalid/truncated JSON and portable-journal envelopes also
    raise rather than silently producing incomplete accounting. No journal codec,
    live viewer, cost estimation, or provider request is involved.
    """
    offset = _cutoff_ns(until_seconds) if until_seconds is not None else None
    spans = _load_spans(trace_path, strict=True)
    starts = [s["start_time"] for s in spans if s["start_time"] > 0]
    if not starts:
        raise ValueError("Trace contains no spans with a positive start timestamp")
    origin = min(starts)
    last = max(max(s["start_time"], s["end_time"]) for s in spans)
    report = UsageReport(str(origin), (last - origin) / 1e9, until_seconds)
    calls: dict[tuple[str, str], _Call] = {}
    for span in spans:
        if span["attributes"].get("openinference.span.kind") != "LLM":
            continue
        call = _recorded_call(span)
        key = (call.trace_id, call.span_id)
        if key in calls:
            if calls[key] != call:
                raise ValueError(f"Conflicting LLM accounting for trace/span {key}")
            report.deduplicated_spans += 1
        else:
            calls[key] = call
    for call in sorted(calls.values(), key=lambda c: (c.end_ns, c.trace_id, c.span_id)):
        if offset is not None and call.start_ns > origin + offset:
            continue
        if not call.end_ns or (offset is not None and call.end_ns > origin + offset):
            report.in_flight_calls += 1
            continue
        report.totals._add(call)
        report.by_model.setdefault(call.model, UsageTotals())._add(call)
        report.timeline.append(
            UsagePoint(
                (call.end_ns - origin) / 1e9,
                call.model,
                call.trace_id,
                call.span_id,
                call.input_tokens,
                call.output_tokens,
                call.total_tokens,
                call.cached_input_tokens,
                replace(report.totals),
            )
        )
    return report
