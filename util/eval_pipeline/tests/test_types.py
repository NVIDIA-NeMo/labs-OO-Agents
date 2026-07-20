# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for eval_pipeline types."""

from pathlib import Path

import pytest

from eval_pipeline.models import ExecutionResult, ScoreResult, ScoringContext, Task


class TestTask:
    def test_basic_creation(self):
        task = Task(id="t1", input="hello", expected="world")
        assert task.id == "t1"
        assert task.input == "hello"
        assert task.expected == "world"
        assert task.metadata == {}

    def test_with_metadata(self):
        task = Task(id="t1", input="x", expected="y", metadata={"key": "value"})
        assert task.metadata["key"] == "value"


class TestExecutionResult:
    def test_success(self):
        result = ExecutionResult(
            task_id="t1",
            input="hello",
            expected="world",
            actual="world",
            trace_file=Path("/tmp/trace.jsonl"),
            latency_ms=100.5,
        )
        assert result.task_id == "t1"
        assert result.actual == "world"
        assert result.error is None

    def test_with_error(self):
        result = ExecutionResult(
            task_id="t1",
            input="hello",
            expected="world",
            actual=None,
            trace_file=Path("/tmp/trace.jsonl"),
            latency_ms=50.0,
            error="Connection failed",
        )
        assert result.error == "Connection failed"
        assert result.actual is None


class TestScoringContext:
    def test_minimal(self):
        ctx = ScoringContext(
            task_id="t1",
            input="hello",
            expected="world",
            actual="world",
        )
        assert ctx.trace is None

    def test_full(self):
        ctx = ScoringContext(
            task_id="t1",
            input="hello",
            expected="world",
            actual="world",
            trace=None,
            latency_ms=100.0,
            input_tokens=30,
            output_tokens=20,
            total_tokens=50,
            metadata={"difficulty": "easy"},
        )
        assert ctx.total_tokens == 50
        assert ctx.metadata == {"difficulty": "easy"}


class TestScoreResult:
    def test_valid_score(self):
        result = ScoreResult(score=0.75, reasoning="Good match")
        assert result.score == 0.75
        assert result.reasoning == "Good match"

    def test_perfect_score(self):
        result = ScoreResult(score=1.0, reasoning="Perfect")
        assert result.score == 1.0

    def test_zero_score(self):
        result = ScoreResult(score=0.0, reasoning="No match")
        assert result.score == 0.0

    def test_invalid_score_too_high(self):
        with pytest.raises(ValueError, match="must be 0.0-1.0"):
            ScoreResult(score=1.5, reasoning="Invalid")

    def test_invalid_score_negative(self):
        with pytest.raises(ValueError, match="must be 0.0-1.0"):
            ScoreResult(score=-0.1, reasoning="Invalid")

    def test_with_metadata(self):
        result = ScoreResult(
            score=0.8,
            reasoning="Close match",
            metadata={"diff": "world vs worl"},
        )
        assert result.metadata["diff"] == "world vs worl"
