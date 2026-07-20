# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write evaluation results as spans to trace files or OTLP.

This module creates OpenTelemetry-compatible spans containing eval results
that can be rendered by the trace viewer's EvalPlugin. Supports both
JSONL trace files and OTLP HTTP (viewer API).
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eval_pipeline.eval_types import ScoreDetail

log = logging.getLogger(__name__)


def _otlp_value(v: str | float | bool | int) -> dict:
    """Encode a scalar as OTLP AnyValue."""
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    return {"stringValue": str(v)}


def _attrs_to_otlp(attributes: dict[str, str | int | float | bool]) -> list[dict]:
    """Convert flat attributes dict to OTLP KeyValue list."""
    return [{"key": k, "value": _otlp_value(v)} for k, v in attributes.items()]


# Keys that are set explicitly by the eval span functions.
# extra_metadata entries with these names raise ValueError to prevent collisions.
_RESERVED_EVAL_KEYS = frozenset(
    {
        "passed",
        "weighted_score",
        "score",
        "model",
        "test_id",
        "agent_class",
        "method",
        "test_name",
        "display_name",
        "tier",
        "variant",
        "run_id",
    }
)


def post_eval_span_to_otlp(
    *,
    session_id: str,
    experiment: str,
    test_id: str,
    passed: bool,
    weighted_score: float,
    model: str,
    agent_class: str,
    method: str,
    scores: dict[str, ScoreDetail],
    duration_ns: int | None = None,
    test_name: str | None = None,
    display_name: str | None = None,
    tier: str | None = None,
    variant: str | None = None,
    run_id: int | None = None,
    extra_metadata: dict[str, str | int | float | bool] | None = None,
    endpoint: str | None = None,
) -> bool:
    """Post an eval result as an OTLP trace to the viewer API.

    Uses the same session_id as the agent run so the viewer shows one session
    with both agent spans and eval outcome. Uses OTLP_ENDPOINT env if endpoint
    is not provided.
    """
    endpoint = endpoint or os.getenv("OTLP_ENDPOINT", "http://localhost:5001/v1/traces")
    endpoint = endpoint.rstrip("/")
    if not endpoint.endswith("/v1/traces"):
        endpoint = f"{endpoint}/v1/traces"
    trace_id = uuid.uuid4().hex
    span_id = uuid.uuid4().hex[:16]
    now_ns = time.time_ns()
    duration = duration_ns if duration_ns is not None else 0

    resource_attrs = [
        {"key": "session.id", "value": {"stringValue": session_id}},
        {"key": "experiment", "value": {"stringValue": experiment}},
        {"key": "eval.model", "value": {"stringValue": model}},
        {"key": "eval.test_id", "value": {"stringValue": test_id}},
        {"key": "eval.agent_class", "value": {"stringValue": agent_class}},
        {"key": "eval.method", "value": {"stringValue": method}},
    ]
    if test_name:
        resource_attrs.append({"key": "eval.test_name", "value": {"stringValue": test_name}})
    if display_name:
        resource_attrs.append({"key": "eval.display_name", "value": {"stringValue": display_name}})
    if tier:
        resource_attrs.append({"key": "eval.tier", "value": {"stringValue": tier}})
    if variant:
        resource_attrs.append({"key": "eval.variant", "value": {"stringValue": variant}})
    if run_id is not None:
        resource_attrs.append({"key": "eval.run_id", "value": {"intValue": str(run_id)}})
    # Append arbitrary eval metadata (from config, test, or task level)
    if extra_metadata:
        collisions = set(extra_metadata) & _RESERVED_EVAL_KEYS
        if collisions:
            raise ValueError(
                f"eval_metadata keys collide with built-in eval attributes: {sorted(collisions)}. "
                f"Rename them in your config.yaml or task JSONL metadata."
            )
        for meta_key, meta_val in extra_metadata.items():
            resource_attrs.append({"key": f"eval.{meta_key}", "value": _otlp_value(meta_val)})

    span_attrs = [
        {"key": "eval.passed", "value": {"boolValue": passed}},
        {"key": "eval.weighted_score", "value": {"doubleValue": weighted_score}},
        {"key": "eval.score", "value": {"doubleValue": weighted_score}},
    ]
    for scorer_name, detail in scores.items():
        prefix = f"eval.scorer.{scorer_name}"
        span_attrs.append({"key": f"{prefix}.score", "value": _otlp_value(detail.score)})
        span_attrs.append({"key": f"{prefix}.passed", "value": {"boolValue": detail.passed}})
        if detail.reasoning:
            span_attrs.append(
                {"key": f"{prefix}.reasoning", "value": {"stringValue": detail.reasoning}}
            )
    if duration_ns is not None:
        span_attrs.append({"key": "duration_ns", "value": {"intValue": str(duration_ns)}})

    payload = {
        "resourceSpans": [
            {
                "resource": {"attributes": resource_attrs},
                "scopeSpans": [
                    {
                        "scope": {"name": "eval_pipeline"},
                        "spans": [
                            {
                                "traceId": trace_id,
                                "spanId": span_id,
                                "name": "eval",
                                "kind": 1,
                                "startTimeUnixNano": str(now_ns - duration),
                                "endTimeUnixNano": str(now_ns),
                                "attributes": span_attrs,
                                "status": {"code": 1 if passed else 2, "message": ""},
                            }
                        ],
                    }
                ],
            }
        ]
    }
    try:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status < 300
    except Exception as e:
        log.debug("Failed to post eval span to OTLP: %s", e)
        return False


def write_eval_span_to_trace(
    trace_file: Path | None,
    test_id: str,
    passed: bool,
    weighted_score: float,
    model: str,
    agent_class: str,
    method: str,
    scores: dict[str, ScoreDetail],
    duration_ns: int | None = None,
    additional_attributes: dict[str, str | float | bool] | None = None,
    run_id: int | None = None,
    extra_metadata: dict[str, str | int | float | bool] | None = None,
) -> None:
    """Write an eval span to the trace file.

    Creates a span with name 'eval' and attributes that the trace viewer's
    EvalPlugin expects. The span is appended to the existing trace file.

    Args:
        trace_file: Path to trace file. No-op if None or doesn't exist.
        test_id: Unique test identifier (e.g., "test_001_gpt4_run1")
        passed: Overall pass/fail status
        weighted_score: Weighted score across all scorers (0.0-1.0)
        model: Model identifier
        agent_class: Agent class name
        method: Method name that was evaluated
        scores: Dict of scorer name -> ScoreDetail with per-scorer results
        duration_ns: Optional duration in nanoseconds (for timing display)
        additional_attributes: Optional dict of additional eval.* attributes
            (e.g., {"eval.expected_output": "...", "eval.actual_output": "..."})

    Raises:
        ValueError: If duration_ns is negative
    """
    # Validate duration_ns
    if duration_ns is not None and duration_ns < 0:
        raise ValueError(f"duration_ns must be non-negative, got {duration_ns}")

    if trace_file is None or not trace_file.exists():
        return

    # Build span attributes
    attributes: dict[str, str | int | float | bool] = {
        "eval.test_id": test_id,
        "eval.passed": passed,
        "eval.weighted_score": weighted_score,
        "eval.model": model,
        "eval.agent_class": agent_class,
        "eval.method": method,
    }

    # Add per-scorer attributes
    for scorer_name, detail in scores.items():
        prefix = f"eval.scorer.{scorer_name}"
        attributes[f"{prefix}.score"] = detail.score
        attributes[f"{prefix}.passed"] = detail.passed
        if detail.reasoning:
            attributes[f"{prefix}.reasoning"] = detail.reasoning

    if run_id is not None:
        attributes["eval.run_id"] = run_id

    # Append arbitrary eval metadata (from config, test, or task level)
    if extra_metadata:
        collisions = set(extra_metadata) & _RESERVED_EVAL_KEYS
        if collisions:
            raise ValueError(
                f"eval_metadata keys collide with built-in eval attributes: {sorted(collisions)}. "
                f"Rename them in your config.yaml or task JSONL metadata."
            )
        for meta_key, meta_val in extra_metadata.items():
            attributes[f"eval.{meta_key}"] = meta_val

    # Always add duration_ns to attributes when provided
    if duration_ns is not None:
        attributes["duration_ns"] = duration_ns

    # Merge additional attributes if provided
    if additional_attributes:
        attributes.update(additional_attributes)

    # Append one OTLP TracesData line (same format as file exporter) so the trace file stays all-OTLP.
    now_ns = time.time_ns()
    span_duration = duration_ns if duration_ns is not None else 0
    start_ns = now_ns - span_duration
    end_ns = now_ns

    payload = {
        "resourceSpans": [
            {
                "resource": {"attributes": []},
                "scopeSpans": [
                    {
                        "scope": {"name": "eval_pipeline"},
                        "spans": [
                            {
                                "traceId": uuid.uuid4().hex,
                                "spanId": uuid.uuid4().hex[:16],
                                "name": "eval",
                                "kind": 1,
                                "startTimeUnixNano": str(start_ns),
                                "endTimeUnixNano": str(end_ns),
                                "attributes": _attrs_to_otlp(attributes),
                                "status": {"code": 1 if passed else 2, "message": ""},
                            }
                        ],
                    }
                ],
            }
        ]
    }
    with open(trace_file, "a") as f:
        f.write(json.dumps(payload, separators=(",", ":"), default=str) + "\n")
