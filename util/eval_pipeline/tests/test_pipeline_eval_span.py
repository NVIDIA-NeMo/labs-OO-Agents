# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for eval span integration in pipeline."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from eval_pipeline.pipeline import PipelineConfig, process_sample
from tests.otlp_helpers import _otlp_attrs_to_dict


class TestProcessSampleEvalSpan:
    """Test that process_sample writes eval span to trace."""

    @pytest.fixture
    def mock_sample(self):
        """Create a mock sample for testing."""
        sample = MagicMock()
        sample.task.id = "test_001"
        sample.task.run_id = 1
        sample.model = "test-model"
        sample.agent_class = "TestAgent"
        sample.method = "run"
        sample.display_name = "Test"
        sample.test_name = "test_process_sample_writes_eval_span"
        sample.tier = "stable"
        sample.scorers = []

        # Mock agent factory
        mock_agent = MagicMock()
        sample.agent_factory = MagicMock(return_value=mock_agent)

        return sample

    @pytest.fixture
    def mock_writer(self):
        """Create a mock writer."""
        writer = MagicMock()
        writer.append_result = MagicMock()
        return writer

    @pytest.mark.asyncio
    async def test_process_sample_writes_eval_span(
        self, tmp_path: Path, mock_sample, mock_writer, monkeypatch
    ):
        """process_sample writes eval span to trace file."""
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()

        # Set sample_id so we can predict trace filename
        mock_sample.sample_id = "test_001"

        # Mock execute_task to return a result and create trace file
        mock_result = MagicMock()
        mock_result.error = None

        async def mock_execute_task(**kwargs):
            # Create the trace file that the pipeline expects
            trace_file = kwargs.get("trace_file")
            if trace_file:
                trace_file.write_text("")
            return mock_result

        monkeypatch.setattr("eval_pipeline.pipeline.execute_task", mock_execute_task)

        # Mock build_scoring_context
        mock_ctx = MagicMock()
        mock_ctx.input = "test input"
        mock_ctx.actual = "test output"
        mock_ctx.expected = "expected"
        mock_ctx.error = None
        mock_ctx.input_tokens = 10
        mock_ctx.output_tokens = 20
        mock_ctx.total_tokens = 30

        monkeypatch.setattr(
            "eval_pipeline.pipeline.build_scoring_context",
            lambda *args, **kwargs: mock_ctx,
        )

        # Mock score_task to return scores
        async def mock_score_task(ctx, scorers):
            return {
                "test_scorer": {
                    "score": 0.9,
                    "reasoning": "Test reasoning",
                    "weight": 1.0,
                }
            }

        monkeypatch.setattr("eval_pipeline.pipeline.score_task", mock_score_task)

        # Mock tracing exports (no-op)
        try:
            monkeypatch.setattr(
                "openinference_instrumentation_nooa.flush_traces",
                lambda: None,
            )
        except (ImportError, AttributeError):
            pass  # Tracing not installed, that's fine

        config = PipelineConfig(trace_dir=trace_dir, pass_threshold=0.5)

        # Run the pipeline
        await process_sample(
            sample=mock_sample,
            config=config,
            writer=mock_writer,
        )

        # Find the generated trace file
        trace_files = [
            f
            for f in trace_dir.glob("*.jsonl")
            if not f.name.endswith((".annotations.jsonl", ".noo-eval.jsonl"))
        ]
        assert len(trace_files) == 1, f"Expected 1 trace file, found {len(trace_files)}"
        trace_file = trace_files[0]

        # Verify trace file has eval span (OTLP format: one TracesData line per write)
        content = trace_file.read_text()
        lines = [line for line in content.strip().split("\n") if line]

        eval_attrs = None
        for line in lines:
            obj = json.loads(line)
            if "resourceSpans" not in obj:
                continue
            for res in obj.get("resourceSpans", []):
                for scope in res.get("scopeSpans", []):
                    for span in scope.get("spans", []):
                        if span.get("name") == "eval":
                            eval_attrs = _otlp_attrs_to_dict(span.get("attributes", []))
                            break

        assert eval_attrs is not None, (
            f"Expected 1 eval span in trace file, found in {len(lines)} lines"
        )

        assert "eval.test_id" in eval_attrs
        assert eval_attrs["eval.passed"] is True
        assert eval_attrs["eval.weighted_score"] == 0.9
        assert eval_attrs["eval.model"] == "test-model"
        assert eval_attrs["eval.agent_class"] == "TestAgent"
        assert eval_attrs["eval.method"] == "run"
        assert eval_attrs["eval.scorer.test_scorer.score"] == 0.9
