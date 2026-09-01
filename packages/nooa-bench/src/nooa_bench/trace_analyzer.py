# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Trace analyzer for extracting usage statistics from execution traces.

This module reads JSONL trace files produced by the agent framework and
extracts per-model token counts, LLM call latencies, and total runtime.
"""

import json
import logging
from pathlib import Path
from typing import Any

from nooa_bench.protocol import (
    ModelUsageStats,
    TaskUsageStats,
)

logger = logging.getLogger(__name__)


_MODEL_ATTRS = (
    "llm.model_name",
    "llm.model",
    "gen_ai.request.model",
    "gen_ai.response.model",
    "model",
)
_PROMPT_TOKEN_ATTRS = (
    "llm.token_count.prompt",
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.prompt_tokens",
    "input_tokens",
)
_COMPLETION_TOKEN_ATTRS = (
    "llm.token_count.completion",
    "gen_ai.usage.output_tokens",
    "gen_ai.usage.completion_tokens",
    "output_tokens",
)
_TOTAL_TOKEN_ATTRS = (
    "llm.token_count.total",
    "gen_ai.usage.total_tokens",
    "total_tokens",
)


def _extract_any_value(value_obj: dict[str, Any]) -> Any:
    """Extract a Python value from an OTLP AnyValue object."""
    if "stringValue" in value_obj:
        return value_obj["stringValue"]
    if "intValue" in value_obj:
        return int(value_obj["intValue"])
    if "doubleValue" in value_obj:
        return float(value_obj["doubleValue"])
    if "boolValue" in value_obj:
        return value_obj["boolValue"]
    if "bytesValue" in value_obj:
        return value_obj["bytesValue"]
    if "arrayValue" in value_obj:
        return [_extract_any_value(v) for v in value_obj["arrayValue"].get("values", [])]
    if "kvlistValue" in value_obj:
        return {
            kv["key"]: _extract_any_value(kv.get("value", {}))
            for kv in value_obj["kvlistValue"].get("values", [])
        }
    return None


def _otlp_attrs_to_dict(attrs: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert OTLP ``[{key, value}]`` attributes to a flat dict."""
    result: dict[str, Any] = {}
    for attr in attrs:
        key = attr.get("key", "")
        if key:
            result[key] = _extract_any_value(attr.get("value", {}))
    return result


def _parse_time_ns(value: Any) -> int:
    """Parse a timestamp value in nanoseconds, returning 0 when absent."""
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _first_attr(attrs: dict[str, Any], names: tuple[str, ...], default: Any = None) -> Any:
    """Return the first present attribute from ``names``."""
    for name in names:
        value = attrs.get(name)
        if value is not None:
            return value
    return default


def _iter_trace_spans(doc: dict[str, Any]):
    """Yield trace spans from current OTLP envelopes and legacy flat JSONL."""
    if "resourceSpans" in doc:
        for resource_spans in doc.get("resourceSpans", []):
            for scope_spans in resource_spans.get("scopeSpans", []):
                for span in scope_spans.get("spans", []):
                    yield {
                        "name": span.get("name", ""),
                        "start_ns": _parse_time_ns(span.get("startTimeUnixNano")),
                        "end_ns": _parse_time_ns(span.get("endTimeUnixNano")),
                        "attributes": _otlp_attrs_to_dict(span.get("attributes", [])),
                    }
        return

    attrs = doc.get("attributes", {})
    if isinstance(attrs, list):
        attrs = _otlp_attrs_to_dict(attrs)
    elif not isinstance(attrs, dict):
        attrs = {}

    yield {
        "name": doc.get("name", ""),
        "start_ns": _parse_time_ns(
            doc.get("start_time_unix_nano", doc.get("startTimeUnixNano", doc.get("start_time")))
        ),
        "end_ns": _parse_time_ns(
            doc.get("end_time_unix_nano", doc.get("endTimeUnixNano", doc.get("end_time")))
        ),
        "attributes": attrs,
    }


class TraceAnalyzer:
    """
    Analyzes OTel traces to extract usage statistics.

    Implements the TraceAnalyzer protocol: reads .jsonl trace files and
    extracts token counts per model, LLM call latencies, and total runtime.
    """

    def analyze_trace(self, trace_path: str) -> TaskUsageStats:
        """
        Analyze a trace file and extract usage statistics.

        Implements the TraceAnalyzer protocol to extract:
        - Token counts per model
        - LLM call latencies
        - Total runtime
        - Model usage breakdown

        Args:
            trace_path: Path to .jsonl trace file

        Returns:
            TaskUsageStats with extracted metrics
        """
        path = Path(trace_path)
        if not path.exists():
            # Return empty stats for missing trace
            return TaskUsageStats(
                task_id=path.stem,
                total_runtime_seconds=0.0,
                models_used=[],
                total_llm_calls=0,
            )

        # Track stats per model
        model_stats: dict[str, ModelUsageStats] = {}
        trace_start_ns: int | None = None
        trace_end_ns: int | None = None
        total_llm_calls = 0

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)

                    for span in _iter_trace_spans(doc):
                        start_ns = span["start_ns"]
                        end_ns = span["end_ns"]

                        # Track overall trace timing
                        if start_ns > 0 and (trace_start_ns is None or start_ns < trace_start_ns):
                            trace_start_ns = start_ns
                        if end_ns > 0 and (trace_end_ns is None or end_ns > trace_end_ns):
                            trace_end_ns = end_ns

                        # Look for LLM spans
                        attrs = span["attributes"]
                        span_name = span["name"]

                        # Check if this is an LLM call span
                        if self._is_llm_span(span_name, attrs):
                            total_llm_calls += 1

                            # Extract model name
                            model_name = _first_attr(attrs, _MODEL_ATTRS, "unknown")

                            # Initialize model stats if not seen before
                            if model_name not in model_stats:
                                model_stats[model_name] = ModelUsageStats(model_name=model_name)

                            stats = model_stats[model_name]

                            # Extract token counts with validation
                            prompt_tokens = _first_attr(attrs, _PROMPT_TOKEN_ATTRS, 0)
                            completion_tokens = _first_attr(attrs, _COMPLETION_TOKEN_ATTRS, 0)
                            total_tokens = _first_attr(attrs, _TOTAL_TOKEN_ATTRS, None)

                            # Validate and parse token counts
                            prompt_count = self._parse_token_count(
                                prompt_tokens, "prompt_tokens", path
                            )
                            completion_count = self._parse_token_count(
                                completion_tokens, "completion_tokens", path
                            )
                            total_count = self._parse_token_count(
                                total_tokens, "total_tokens", path
                            )
                            combined_count = prompt_count + completion_count

                            stats.prompt_tokens += prompt_count
                            stats.completion_tokens += completion_count
                            stats.total_tokens += combined_count or total_count
                            stats.call_count += 1

                            # Calculate latency in milliseconds
                            if start_ns > 0 and end_ns > 0:
                                latency_ms = (end_ns - start_ns) / 1e6
                                stats.latencies_ms.append(latency_ms)

                except json.JSONDecodeError:
                    continue

        # Calculate total runtime
        runtime_seconds = 0.0
        if trace_start_ns and trace_end_ns:
            runtime_seconds = (trace_end_ns - trace_start_ns) / 1e9

        return TaskUsageStats(
            task_id=path.stem,
            total_runtime_seconds=runtime_seconds,
            models_used=list(model_stats.values()),
            total_llm_calls=total_llm_calls,
        )

    def _is_llm_span(self, span_name: str, attributes: dict[str, Any]) -> bool:
        """Check if a span represents an LLM API call."""
        # Check span name
        llm_span_names = ["llm", "chat", "completion", "generation", "inference"]
        if any(name in span_name.lower() for name in llm_span_names):
            return True

        # Check for LLM-related attributes
        llm_attrs = [
            "llm.model_name",
            "llm.model",
            "llm.token_count.prompt",
            "llm.token_count.completion",
            "llm.token_count.total",
            "gen_ai.request.model",
            "gen_ai.response.model",
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.output_tokens",
            "gen_ai.usage.total_tokens",
        ]
        return any(attr in attributes for attr in llm_attrs)

    def _parse_token_count(self, value: Any, field_name: str, trace_path: Path) -> int:
        """Parse a token count value with validation.

        Args:
            value: The raw value from the trace (could be int, str, None, etc.)
            field_name: Name of the field for logging
            trace_path: Path to trace file for logging context

        Returns:
            Parsed integer token count, or 0 if invalid
        """
        if value is None:
            return 0

        if isinstance(value, int):
            return value

        if isinstance(value, float):
            return int(value)

        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                logger.warning(
                    f"Invalid {field_name} value '{value}' in trace {trace_path.name}, using 0"
                )
                return 0

        logger.warning(
            f"Unexpected type {type(value).__name__} for {field_name} in trace "
            f"{trace_path.name}, using 0"
        )
        return 0
