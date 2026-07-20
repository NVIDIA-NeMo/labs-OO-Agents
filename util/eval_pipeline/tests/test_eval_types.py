# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for eval_types and eval_parser.

Tests cover:
- Round-trip serialization (write -> read -> compare)
- Version validation (reject unsupported versions)
- Unknown _type raises EvalParseError
- Annotation lines parsed correctly
- Incomplete files (no completion line) handled
- JSON parse errors raise EvalParseError
- Pydantic validation errors propagate
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from eval_pipeline.eval_parser import EvalFileParser, EvalParseError, get_experiment_status
from eval_pipeline.eval_types import (
    SUPPORTED_VERSIONS,
    EvalAnnotationLine,
    EvalCompletionLine,
    EvalMetadata,
    EvalMetadataLine,
    EvalTestResult,
    ModelSpec,
    ScoreDetail,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_metadata() -> EvalMetadataLine:
    """Create a sample metadata line."""
    return EvalMetadataLine(
        metadata=EvalMetadata(
            timestamp=datetime.now().isoformat(),
            suite_name="test_suite",
            models=["gpt-4", "claude-3"],
            config_file="config.yaml",
            tests=["test_a", "test_b"],
            runs=3,
        )
    )


@pytest.fixture
def sample_result() -> EvalTestResult:
    """Create a sample test result."""
    return EvalTestResult(
        test_id="sentiment_001_gpt4_run1",
        base_test_id="sentiment_001_gpt4",
        run_id=1,
        agent_class="SentimentAgent",
        method="classify_single",
        display_name="Classify single sentiment",
        model="openai/gpt-4",
        variant="run1",
        passed=True,
        scores={
            "accuracy": ScoreDetail(score=1.0, passed=True, reasoning="Correct"),
            "quality": ScoreDetail(score=0.8, passed=True, reasoning="Good quality"),
        },
        input={"text": "I love this!"},
        output="positive",
        expected="positive",
        trace_file="/path/to/trace.jsonl",
    )


@pytest.fixture
def sample_completion() -> EvalCompletionLine:
    """Create a sample completion line."""
    return EvalCompletionLine(
        status="completed",
        completed_at=datetime.now().isoformat(),
        result_count=10,
        passed_count=8,
        success_rate=0.8,
        duration_seconds=120.5,
    )


@pytest.fixture
def sample_annotation() -> EvalAnnotationLine:
    """Create a sample annotation line."""
    return EvalAnnotationLine(
        test_id="sentiment_001_gpt4_run1",
        annotation="Needs review - edge case",
        timestamp=datetime.now().isoformat(),
        author="reviewer@example.com",
    )


# =============================================================================
# Round-trip Serialization Tests
# =============================================================================


class TestRoundTrip:
    """Test that models serialize and deserialize correctly."""

    def test_metadata_round_trip(self, sample_metadata: EvalMetadataLine):
        """Metadata line serializes with _type and deserializes back."""
        json_str = sample_metadata.model_dump_json(by_alias=True)
        data = json.loads(json_str)

        # Check _type is present
        assert data["_type"] == "metadata"
        assert data["version"] == "1"

        # Round-trip
        parsed = EvalMetadataLine.model_validate(data)
        assert parsed.metadata.suite_name == sample_metadata.metadata.suite_name
        assert parsed.metadata.models == sample_metadata.metadata.models

    def test_result_round_trip(self, sample_result: EvalTestResult):
        """Test result serializes with _type and deserializes back."""
        json_str = sample_result.model_dump_json(by_alias=True)
        data = json.loads(json_str)

        # Check _type is present
        assert data["_type"] == "result"

        # Round-trip
        parsed = EvalTestResult.model_validate(data)
        assert parsed.test_id == sample_result.test_id
        assert parsed.passed == sample_result.passed
        assert parsed.scores["accuracy"].score == 1.0

    def test_completion_round_trip(self, sample_completion: EvalCompletionLine):
        """Completion line serializes with _type and deserializes back."""
        json_str = sample_completion.model_dump_json(by_alias=True)
        data = json.loads(json_str)

        # Check _type is present
        assert data["_type"] == "completion"

        # Round-trip
        parsed = EvalCompletionLine.model_validate(data)
        assert parsed.status == sample_completion.status
        assert parsed.result_count == sample_completion.result_count

    def test_annotation_round_trip(self, sample_annotation: EvalAnnotationLine):
        """Annotation line serializes with _type and deserializes back."""
        json_str = sample_annotation.model_dump_json(by_alias=True)
        data = json.loads(json_str)

        # Check _type is present
        assert data["_type"] == "annotation"

        # Round-trip
        parsed = EvalAnnotationLine.model_validate(data)
        assert parsed.test_id == sample_annotation.test_id
        assert parsed.annotation == sample_annotation.annotation


# =============================================================================
# Parser Tests
# =============================================================================


class TestEvalFileParser:
    """Test EvalFileParser functionality."""

    def test_parse_metadata_line(self, sample_metadata: EvalMetadataLine):
        """Parser correctly handles metadata line."""
        parser = EvalFileParser()
        json_str = sample_metadata.model_dump_json(by_alias=True)

        parsed = parser.parse_line(json_str)
        assert isinstance(parsed, EvalMetadataLine)
        assert parser.version == "1"

    def test_parse_result_line(self, sample_result: EvalTestResult):
        """Parser correctly handles result line."""
        parser = EvalFileParser()
        json_str = sample_result.model_dump_json(by_alias=True)

        parsed = parser.parse_line(json_str)
        assert isinstance(parsed, EvalTestResult)
        assert parsed.test_id == sample_result.test_id

    def test_parse_completion_line(self, sample_completion: EvalCompletionLine):
        """Parser correctly handles completion line."""
        parser = EvalFileParser()
        json_str = sample_completion.model_dump_json(by_alias=True)

        parsed = parser.parse_line(json_str)
        assert isinstance(parsed, EvalCompletionLine)
        assert parsed.status == "completed"

    def test_parse_annotation_line(self, sample_annotation: EvalAnnotationLine):
        """Parser correctly handles annotation line."""
        parser = EvalFileParser()
        json_str = sample_annotation.model_dump_json(by_alias=True)

        parsed = parser.parse_line(json_str)
        assert isinstance(parsed, EvalAnnotationLine)
        assert parsed.annotation == sample_annotation.annotation


class TestParserErrorHandling:
    """Test parser error handling."""

    def test_invalid_json_raises_parse_error(self):
        """Invalid JSON raises EvalParseError."""
        parser = EvalFileParser()

        with pytest.raises(EvalParseError) as exc_info:
            parser.parse_line("not valid json", line_number=5)

        assert "Invalid JSON" in str(exc_info.value)
        assert exc_info.value.line_number == 5

    def test_unknown_type_raises_parse_error(self):
        """Unknown _type raises EvalParseError."""
        parser = EvalFileParser()
        data = json.dumps({"_type": "unknown_type", "foo": "bar"})

        with pytest.raises(EvalParseError) as exc_info:
            parser.parse_line(data, line_number=10)

        assert "Unknown line type" in str(exc_info.value)
        assert "'unknown_type'" in str(exc_info.value)

    def test_unsupported_version_raises_parse_error(self):
        """Unsupported version raises EvalParseError."""
        parser = EvalFileParser()
        data = json.dumps(
            {
                "_type": "metadata",
                "version": "999",
                "metadata": {
                    "timestamp": "2024-01-01T00:00:00",
                    "suite_name": "test",
                    "models": [],
                },
            }
        )

        with pytest.raises(EvalParseError) as exc_info:
            parser.parse_line(data)

        assert "Unsupported" in str(exc_info.value)
        assert "999" in str(exc_info.value)

    def test_pydantic_validation_error_propagates(self):
        """Pydantic ValidationError is not wrapped."""
        parser = EvalFileParser()
        # Missing required fields
        data = json.dumps({"_type": "result", "test_id": "test_001"})

        with pytest.raises(ValidationError):
            parser.parse_line(data)


class TestFileParser:
    """Test parsing complete files."""

    def test_parse_complete_file(
        self,
        sample_metadata: EvalMetadataLine,
        sample_result: EvalTestResult,
        sample_completion: EvalCompletionLine,
        sample_annotation: EvalAnnotationLine,
    ):
        """Parser correctly parses a complete file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(sample_metadata.model_dump_json(by_alias=True) + "\n")
            f.write(sample_result.model_dump_json(by_alias=True) + "\n")
            f.write(sample_completion.model_dump_json(by_alias=True) + "\n")
            f.write(sample_annotation.model_dump_json(by_alias=True) + "\n")
            path = Path(f.name)

        try:
            parser = EvalFileParser()
            metadata, results, completion, annotations = parser.parse_file(path)

            assert metadata.metadata.suite_name == sample_metadata.metadata.suite_name
            assert len(results) == 1
            assert results[0].test_id == sample_result.test_id
            assert completion is not None
            assert completion.status == "completed"
            assert len(annotations) == 1
            assert annotations[0].annotation == sample_annotation.annotation
        finally:
            path.unlink()

    def test_parse_incomplete_file(
        self,
        sample_metadata: EvalMetadataLine,
        sample_result: EvalTestResult,
    ):
        """Parser handles files without completion line (still running)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(sample_metadata.model_dump_json(by_alias=True) + "\n")
            f.write(sample_result.model_dump_json(by_alias=True) + "\n")
            path = Path(f.name)

        try:
            parser = EvalFileParser()
            metadata, results, completion, annotations = parser.parse_file(path)

            assert metadata is not None
            assert len(results) == 1
            assert completion is None  # No completion line
            assert len(annotations) == 0
        finally:
            path.unlink()

    def test_parse_file_without_metadata_raises(self, sample_result: EvalTestResult):
        """File without metadata line raises EvalParseError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(sample_result.model_dump_json(by_alias=True) + "\n")
            path = Path(f.name)

        try:
            parser = EvalFileParser()
            with pytest.raises(EvalParseError) as exc_info:
                parser.parse_file(path)

            assert "No metadata line found" in str(exc_info.value)
        finally:
            path.unlink()

    def test_parse_empty_lines_skipped(self, sample_metadata: EvalMetadataLine):
        """Empty lines in file are skipped."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(sample_metadata.model_dump_json(by_alias=True) + "\n")
            f.write("\n")  # Empty line
            f.write("   \n")  # Whitespace-only line
            path = Path(f.name)

        try:
            parser = EvalFileParser()
            metadata, results, _completion, _annotations = parser.parse_file(path)
            assert metadata is not None
            assert len(results) == 0
        finally:
            path.unlink()


# =============================================================================
# Status Derivation Tests
# =============================================================================


class TestGetExperimentStatus:
    """Test get_experiment_status function."""

    def test_completed_status(self, sample_completion: EvalCompletionLine):
        """Completion line with completed status returns 'completed'."""
        status = get_experiment_status(sample_completion, file_mtime=0)
        assert status == "completed"

    def test_failed_status(self):
        """Completion line with failed status returns 'failed'."""
        completion = EvalCompletionLine(
            status="failed",
            completed_at=datetime.now().isoformat(),
            result_count=5,
        )
        status = get_experiment_status(completion, file_mtime=0)
        assert status == "failed"

    def test_running_status_recent_file(self):
        """No completion with recent file returns 'running'."""
        import time

        recent_mtime = time.time() - 30  # 30 seconds ago
        status = get_experiment_status(None, file_mtime=recent_mtime)
        assert status == "running"

    def test_stale_status_old_file(self):
        """No completion with old file returns 'stale'."""
        import time

        old_mtime = time.time() - 120  # 2 minutes ago
        status = get_experiment_status(None, file_mtime=old_mtime)
        assert status == "stale"

    def test_custom_stale_threshold(self):
        """Custom stale threshold is respected."""
        import time

        mtime = time.time() - 10  # 10 seconds ago
        # Default threshold (60s) - should be running
        assert get_experiment_status(None, file_mtime=mtime) == "running"
        # Custom threshold (5s) - should be stale
        assert get_experiment_status(None, file_mtime=mtime, stale_threshold_seconds=5) == "stale"


# =============================================================================
# Model-specific Tests
# =============================================================================


class TestModelSpec:
    """Test ModelSpec model."""

    def test_full_model_spec(self):
        """ModelSpec with all fields."""
        spec = ModelSpec(
            id="gpt4",
            model_name="openai/gpt-4",
            endpoint="https://api.openai.com/v1",
            api_key_env="OPENAI_API_KEY",
            max_tokens=4096,
        )
        data = spec.model_dump()
        assert data["id"] == "gpt4"
        assert data["endpoint"] == "https://api.openai.com/v1"

    def test_minimal_model_spec(self):
        """ModelSpec with only required fields."""
        spec = ModelSpec(id="gpt4", model_name="openai/gpt-4")
        assert spec.endpoint is None
        assert spec.api_key_env is None


class TestScoreDetail:
    """Test ScoreDetail model."""

    def test_full_score_detail(self):
        """ScoreDetail with all fields."""
        score = ScoreDetail(
            score=0.95,
            passed=True,
            reasoning="Excellent output",
            metrics={"precision": 0.98, "recall": 0.92},
        )
        assert score.score == 0.95
        assert score.metrics["precision"] == 0.98

    def test_minimal_score_detail(self):
        """ScoreDetail with only required fields."""
        score = ScoreDetail(score=0.5, passed=False)
        assert score.reasoning is None
        assert score.metrics is None


class TestSupportedVersions:
    """Test version constants."""

    def test_version_1_supported(self):
        """Version 1 is in supported versions."""
        assert "1" in SUPPORTED_VERSIONS


# =============================================================================
# Integration Tests - ExperimentWriter with Real File I/O
# =============================================================================


class TestExperimentWriterIntegration:
    """Integration tests for ExperimentWriter with real file I/O."""

    def test_writer_produces_parseable_output(self, tmp_path):
        """ExperimentWriter output can be parsed by EvalFileParser."""
        from eval_pipeline.experiment_writer import ExperimentWriter

        # Write using ExperimentWriter
        writer = ExperimentWriter(output_dir=tmp_path, experiment_name="test")
        writer.start(
            suite_name="integration_test",
            models=["gpt-4", "claude-3"],
            tests=["test_a", "test_b"],
            runs=2,
        )

        # Append some results
        result1 = EvalTestResult(
            test_id="test_001_gpt4_run1",
            agent_class="TestAgent",
            method="run",
            model="gpt-4",
            variant="run1",
            passed=True,
            scores={"eval": ScoreDetail(score=1.0, passed=True, reasoning="OK")},
            input={"x": 1},
            output="result",
            expected="result",
        )
        result2 = EvalTestResult(
            test_id="test_002_gpt4_run1",
            agent_class="TestAgent",
            method="run",
            model="gpt-4",
            variant="run1",
            passed=False,
            scores={"eval": ScoreDetail(score=0.0, passed=False, reasoning="Wrong")},
            input={"x": 2},
            output="wrong",
            expected="right",
        )
        writer.append_result(result1)
        writer.append_result(result2)
        writer.finalize()

        # Parse using EvalFileParser
        parser = EvalFileParser()
        metadata, results, completion, annotations = parser.parse_file(writer.file_path)

        # Verify metadata
        assert metadata.metadata.suite_name == "integration_test"
        assert metadata.metadata.models == ["gpt-4", "claude-3"]
        assert metadata.version == "1"

        # Verify results
        assert len(results) == 2
        assert results[0].test_id == "test_001_gpt4_run1"
        assert results[0].passed is True
        assert results[1].test_id == "test_002_gpt4_run1"
        assert results[1].passed is False

        # Verify completion
        assert completion is not None
        assert completion.status == "completed"
        assert completion.result_count == 2
        assert completion.passed_count == 1
        assert completion.success_rate == 0.5

        # Verify no annotations
        assert len(annotations) == 0

    def test_writer_tracks_counts(self, tmp_path):
        """ExperimentWriter correctly tracks result and passed counts."""
        from eval_pipeline.experiment_writer import ExperimentWriter

        writer = ExperimentWriter(output_dir=tmp_path, experiment_name="counts")
        writer.start(suite_name="counts_test", models=[])

        # Append mixed results
        for i, passed in enumerate([True, True, False, True, False]):
            result = EvalTestResult(
                test_id=f"test_{i}",
                agent_class="Agent",
                method="run",
                model="model",
                variant="v1",
                passed=passed,
                scores={"s": ScoreDetail(score=1.0 if passed else 0.0, passed=passed)},
                input=i,
                output=i,
                expected=i,
            )
            writer.append_result(result)

        assert writer.result_count == 5
        assert writer.passed_count == 3

        writer.finalize()

        # Verify completion has correct counts
        parser = EvalFileParser()
        _, _, completion, _ = parser.parse_file(writer.file_path)
        assert completion.result_count == 5
        assert completion.passed_count == 3
        assert completion.success_rate == 0.6

    def test_writer_incomplete_file_no_completion(self, tmp_path):
        """Incomplete file (no finalize) has no completion line."""
        from eval_pipeline.experiment_writer import ExperimentWriter

        writer = ExperimentWriter(output_dir=tmp_path, experiment_name="incomplete")
        writer.start(suite_name="incomplete_test", models=[])

        result = EvalTestResult(
            test_id="test_1",
            agent_class="Agent",
            method="run",
            model="model",
            variant="v1",
            passed=True,
            scores={"s": ScoreDetail(score=1.0, passed=True)},
            input=1,
            output=1,
            expected=1,
        )
        writer.append_result(result)
        # Note: NOT calling writer.finalize()

        # Parse - should have no completion
        parser = EvalFileParser()
        _metadata, results, completion, _annotations = parser.parse_file(writer.file_path)

        assert len(results) == 1
        assert completion is None  # No completion line
