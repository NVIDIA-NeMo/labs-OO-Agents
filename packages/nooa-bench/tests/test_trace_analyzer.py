# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for nooa-bench trace usage extraction."""

from __future__ import annotations

import json
from pathlib import Path

from nooa_bench.trace_analyzer import TraceAnalyzer


def _attr(key: str, value: object) -> dict:
    if isinstance(value, bool):
        otlp_value = {"boolValue": value}
    elif isinstance(value, int):
        otlp_value = {"intValue": str(value)}
    elif isinstance(value, float):
        otlp_value = {"doubleValue": value}
    else:
        otlp_value = {"stringValue": str(value)}
    return {"key": key, "value": otlp_value}


def _write_jsonl(path: Path, *records: dict) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_current_otlp_jsonl_extracts_usage(tmp_path: Path) -> None:
    trace_path = tmp_path / "current_otlp.jsonl"
    _write_jsonl(
        trace_path,
        {
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "name": "completion",
                                    "startTimeUnixNano": "1000000000",
                                    "endTimeUnixNano": "1500000000",
                                    "attributes": [
                                        _attr("llm.model_name", "model-a"),
                                        _attr("llm.token_count.prompt", 10),
                                        _attr("llm.token_count.completion", 5),
                                    ],
                                }
                            ]
                        }
                    ]
                }
            ]
        },
    )

    stats = TraceAnalyzer().analyze_trace(str(trace_path))

    assert stats.total_runtime_seconds == 0.5
    assert stats.total_llm_calls == 1
    assert len(stats.models_used) == 1
    model_stats = stats.models_used[0]
    assert model_stats.model_name == "model-a"
    assert model_stats.prompt_tokens == 10
    assert model_stats.completion_tokens == 5
    assert model_stats.total_tokens == 15
    assert model_stats.call_count == 1
    assert model_stats.latencies_ms == [500.0]


def test_current_otlp_jsonl_aggregates_multiple_batches_and_models(tmp_path: Path) -> None:
    trace_path = tmp_path / "multi_batch.jsonl"
    _write_jsonl(
        trace_path,
        {
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "name": "acompletion",
                                    "startTimeUnixNano": "1000000000",
                                    "endTimeUnixNano": "1200000000",
                                    "attributes": [
                                        _attr("llm.model_name", "model-a"),
                                        _attr("llm.token_count.prompt", 3),
                                        _attr("llm.token_count.completion", 2),
                                    ],
                                },
                                {
                                    "name": "tool",
                                    "startTimeUnixNano": "1250000000",
                                    "endTimeUnixNano": "1300000000",
                                    "attributes": [_attr("tool.name", "shell")],
                                },
                            ]
                        }
                    ]
                }
            ]
        },
        {
            "resourceSpans": [
                {
                    "scopeSpans": [
                        {
                            "spans": [
                                {
                                    "name": "responses",
                                    "startTimeUnixNano": "2000000000",
                                    "endTimeUnixNano": "2400000000",
                                    "attributes": [
                                        _attr("llm.model", "model-a"),
                                        _attr("gen_ai.usage.input_tokens", 7),
                                        _attr("gen_ai.usage.output_tokens", 4),
                                    ],
                                },
                                {
                                    "name": "inference",
                                    "startTimeUnixNano": "3000000000",
                                    "endTimeUnixNano": "3100000000",
                                    "attributes": [
                                        _attr("gen_ai.response.model", "model-b"),
                                        _attr("llm.token_count.total", 9),
                                    ],
                                },
                            ]
                        }
                    ]
                }
            ]
        },
    )

    stats = TraceAnalyzer().analyze_trace(str(trace_path))
    by_model = {model.model_name: model for model in stats.models_used}

    assert stats.total_runtime_seconds == 2.1
    assert stats.total_llm_calls == 3
    assert by_model["model-a"].prompt_tokens == 10
    assert by_model["model-a"].completion_tokens == 6
    assert by_model["model-a"].total_tokens == 16
    assert by_model["model-a"].call_count == 2
    assert by_model["model-a"].latencies_ms == [200.0, 400.0]
    assert by_model["model-b"].prompt_tokens == 0
    assert by_model["model-b"].completion_tokens == 0
    assert by_model["model-b"].total_tokens == 9
    assert by_model["model-b"].call_count == 1


def test_legacy_flat_jsonl_still_extracts_usage(tmp_path: Path) -> None:
    trace_path = tmp_path / "legacy_flat.jsonl"
    _write_jsonl(
        trace_path,
        {
            "name": "completion",
            "start_time_unix_nano": 1000000000,
            "end_time_unix_nano": 1500000000,
            "attributes": {
                "llm.model": "legacy-model",
                "llm.token_count.prompt": 10,
                "llm.token_count.completion": 5,
            },
        },
    )

    stats = TraceAnalyzer().analyze_trace(str(trace_path))

    assert stats.total_runtime_seconds == 0.5
    assert stats.total_llm_calls == 1
    assert stats.models_used[0].model_name == "legacy-model"
    assert stats.models_used[0].total_tokens == 15


def test_missing_trace_returns_empty_stats(tmp_path: Path) -> None:
    trace_path = tmp_path / "missing.jsonl"

    stats = TraceAnalyzer().analyze_trace(str(trace_path))

    assert stats.task_id == "missing"
    assert stats.total_runtime_seconds == 0.0
    assert stats.total_llm_calls == 0
    assert stats.models_used == []
