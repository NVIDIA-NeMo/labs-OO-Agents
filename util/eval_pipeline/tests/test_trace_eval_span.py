# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for eval span creation in traces."""

import json
from pathlib import Path

import pytest

from eval_pipeline.eval_types import ScoreDetail
from eval_pipeline.trace_eval_span import write_eval_span_to_trace
from tests.otlp_helpers import _otlp_attrs_to_dict


def _read_first_span_from_otlp_file(trace_file: Path) -> tuple[str, dict]:
    """Parse first OTLP TracesData line and return (span_name, flat_attributes)."""
    line = trace_file.read_text().strip()
    assert line, "Trace file is empty"
    payload = json.loads(line)
    res = payload["resourceSpans"][0]
    scope = res["scopeSpans"][0]
    span = scope["spans"][0]
    attrs = _otlp_attrs_to_dict(span.get("attributes", []))
    return span.get("name", ""), attrs


class TestWriteEvalSpanToTrace:
    """Test eval span writing to trace files."""

    def test_writes_span_with_eval_attributes(self, tmp_path: Path):
        """Eval span has correct name and attributes (OTLP format)."""
        trace_file = tmp_path / "test.jsonl"
        trace_file.write_text("")  # Create empty trace file

        write_eval_span_to_trace(
            trace_file=trace_file,
            test_id="test_001_gpt4_run1",
            passed=True,
            weighted_score=0.85,
            model="gpt-4",
            agent_class="TestAgent",
            method="run",
            scores={
                "exact_match": ScoreDetail(
                    score=1.0,
                    passed=True,
                    reasoning="Output matches expected",
                ),
                "llm_judge": ScoreDetail(
                    score=0.7,
                    passed=True,
                    reasoning="Good quality response",
                ),
            },
        )

        content = trace_file.read_text()
        lines = [ln for ln in content.strip().split("\n") if ln]
        assert len(lines) == 1, f"Expected 1 line, got {len(lines)}"

        name, attrs = _read_first_span_from_otlp_file(trace_file)
        assert name == "eval"

        assert attrs["eval.test_id"] == "test_001_gpt4_run1"
        assert attrs["eval.passed"] is True
        assert attrs["eval.weighted_score"] == 0.85
        assert attrs["eval.model"] == "gpt-4"
        assert attrs["eval.agent_class"] == "TestAgent"
        assert attrs["eval.method"] == "run"
        assert attrs["eval.scorer.exact_match.score"] == 1.0
        assert attrs["eval.scorer.exact_match.passed"] is True
        assert attrs["eval.scorer.exact_match.reasoning"] == "Output matches expected"
        assert attrs["eval.scorer.llm_judge.score"] == 0.7
        assert attrs["eval.scorer.llm_judge.passed"] is True
        assert attrs["eval.scorer.llm_judge.reasoning"] == "Good quality response"

    def test_handles_missing_reasoning(self, tmp_path: Path):
        """Scorer without reasoning still creates attributes."""
        trace_file = tmp_path / "test.jsonl"
        trace_file.write_text("")

        write_eval_span_to_trace(
            trace_file=trace_file,
            test_id="test_002",
            passed=False,
            weighted_score=0.3,
            model="gpt-3.5",
            agent_class="Agent",
            method="execute",
            scores={
                "basic": ScoreDetail(score=0.3, passed=False, reasoning=None),
            },
        )

        _, attrs = _read_first_span_from_otlp_file(trace_file)
        assert attrs["eval.scorer.basic.score"] == 0.3
        assert attrs["eval.scorer.basic.passed"] is False
        assert "eval.scorer.basic.reasoning" not in attrs

    def test_does_nothing_if_trace_file_is_none(self, tmp_path: Path):
        """No error if trace_file is None."""
        # Should not raise
        write_eval_span_to_trace(
            trace_file=None,
            test_id="test_003",
            passed=True,
            weighted_score=1.0,
            model="gpt-4",
            agent_class="Agent",
            method="run",
            scores={},
        )

    def test_does_nothing_if_trace_file_does_not_exist(self, tmp_path: Path):
        """No error if trace file doesn't exist."""
        trace_file = tmp_path / "nonexistent.jsonl"

        # Should not raise
        write_eval_span_to_trace(
            trace_file=trace_file,
            test_id="test_004",
            passed=True,
            weighted_score=1.0,
            model="gpt-4",
            agent_class="Agent",
            method="run",
            scores={},
        )

    def test_includes_duration_when_provided(self, tmp_path: Path):
        """Duration is included in span when provided."""
        trace_file = tmp_path / "test.jsonl"
        trace_file.write_text("")

        write_eval_span_to_trace(
            trace_file=trace_file,
            test_id="test_with_duration",
            passed=True,
            weighted_score=1.0,
            model="gpt-4",
            agent_class="Agent",
            method="run",
            scores={},
            duration_ns=5000000000,  # 5 seconds
        )

        _, attrs = _read_first_span_from_otlp_file(trace_file)
        assert attrs["duration_ns"] == 5000000000

    def test_rejects_negative_duration(self, tmp_path: Path):
        """Negative duration raises ValueError."""
        trace_file = tmp_path / "test.jsonl"
        trace_file.write_text("")

        with pytest.raises(ValueError, match="duration_ns must be non-negative"):
            write_eval_span_to_trace(
                trace_file=trace_file,
                test_id="test_negative",
                passed=True,
                weighted_score=1.0,
                model="gpt-4",
                agent_class="Agent",
                method="run",
                scores={},
                duration_ns=-1000,
            )
