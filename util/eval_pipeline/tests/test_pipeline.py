# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for eval_pipeline using capability tests as test suite."""

from pathlib import Path
from typing import Any

import pytest

from eval_pipeline.eval_types import EvalTestResult, Tier
from eval_pipeline.models import ScoreResult, ScoringContext, Task
from eval_pipeline.pipeline import PipelineConfig, Sample, process_sample, run_evaluation
from eval_pipeline.scoring import ScorerConfig

# =============================================================================
# Mock agents for unit testing (no LLM required)
# =============================================================================


class MockSentimentAgent:
    """Mock agent that returns canned responses."""

    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self.default_response = "positive"

    async def run(self, input: str) -> str:
        return self.responses.get(input, self.default_response)


class MockWriter:
    """Mock experiment writer for testing."""

    def __init__(self):
        self.results: list[EvalTestResult] = []
        self.started = False
        self.finalized = False

    def start(
        self,
        suite_name: str | None = None,
        models: list | None = None,
        config_file: str | None = None,
        tests: list[str] | None = None,
        runs: int = 1,
        variants: list[str] | None = None,
    ) -> Path:
        self.started = True
        return Path("/tmp/mock.noo-eval.jsonl")

    def append_result(self, result: EvalTestResult | dict[str, Any]) -> None:
        if isinstance(result, dict):
            result = EvalTestResult.model_validate(result)
        self.results.append(result)

    def finalize(
        self,
        status: str = "completed",
        extra_metrics: dict[str, Any] | None = None,
    ) -> None:
        self.finalized = True


# =============================================================================
# Simple scorers for testing
# =============================================================================


class ExactMatchScorer:
    """Simple exact match scorer."""

    def score(self, ctx: ScoringContext) -> ScoreResult:
        expected = str(ctx.expected).lower().strip()
        actual = str(ctx.actual).lower().strip()
        match = expected == actual
        return ScoreResult(
            score=1.0 if match else 0.0,
            reasoning=f"{'Match' if match else 'No match'}",
        )


# =============================================================================
# Unit tests for pipeline components
# =============================================================================


class TestProcessSample:
    @pytest.mark.asyncio
    async def test_successful_sample(self, tmp_path):
        """Test processing a single sample through the pipeline."""
        agent = MockSentimentAgent(responses={"I love this!": "positive"})
        writer = MockWriter()

        sample = Sample(
            task=Task(id="t1", input="I love this!", expected="positive"),
            method="classify",
            agent_class="SentimentAgent",
            scorers=[ScorerConfig(name="exact", weight=1.0, scorer=ExactMatchScorer())],
            agent_factory=lambda: agent,
        )

        config = PipelineConfig(trace_dir=tmp_path / "traces")

        result = await process_sample(sample, config, writer)

        # test_id includes model and run suffix: "t1_default_run1"
        assert result.test_id.startswith("t1")
        assert result.passed is True
        assert result.output == "positive"
        assert result.scores["exact"].score == 1.0
        assert len(writer.results) == 1

    @pytest.mark.asyncio
    async def test_failed_sample(self, tmp_path):
        """Test a sample that fails scoring."""
        agent = MockSentimentAgent(responses={"I hate this!": "positive"})  # Wrong!
        writer = MockWriter()

        sample = Sample(
            task=Task(id="t1", input="I hate this!", expected="negative"),
            method="classify",
            agent_class="SentimentAgent",
            scorers=[ScorerConfig(name="exact", weight=1.0, scorer=ExactMatchScorer())],
            agent_factory=lambda: agent,
            tier=Tier.STABLE,
        )

        config = PipelineConfig(trace_dir=tmp_path / "traces")

        result = await process_sample(sample, config, writer)

        assert result.passed is False
        assert result.output == "positive"
        assert result.expected == "negative"
        assert result.scores["exact"].score == 0.0


class TestRunEvaluation:
    @pytest.mark.asyncio
    async def test_multiple_samples_sequential(self, tmp_path):
        """Test running multiple samples sequentially."""
        agent = MockSentimentAgent(
            responses={
                "I love this!": "positive",
                "I hate this!": "negative",
                "It's okay.": "neutral",
            }
        )
        writer = MockWriter()

        samples = [
            Sample(
                task=Task(id="t1", input="I love this!", expected="positive"),
                method="classify",
                agent_class="SentimentAgent",
                scorers=[ScorerConfig(name="exact", weight=1.0, scorer=ExactMatchScorer())],
                agent_factory=lambda: agent,
            ),
            Sample(
                task=Task(id="t2", input="I hate this!", expected="negative"),
                method="classify",
                agent_class="SentimentAgent",
                scorers=[ScorerConfig(name="exact", weight=1.0, scorer=ExactMatchScorer())],
                agent_factory=lambda: agent,
            ),
            Sample(
                task=Task(id="t3", input="It's okay.", expected="neutral"),
                method="classify",
                agent_class="SentimentAgent",
                scorers=[ScorerConfig(name="exact", weight=1.0, scorer=ExactMatchScorer())],
                agent_factory=lambda: agent,
            ),
        ]

        config = PipelineConfig(trace_dir=tmp_path / "traces")

        results = await run_evaluation(samples, config, writer, max_concurrent=1)

        assert len(results) == 3
        assert all(r.passed for r in results)
        assert len(writer.results) == 3

    @pytest.mark.asyncio
    async def test_multiple_samples_parallel(self, tmp_path):
        """Test running multiple samples in parallel."""
        agent = MockSentimentAgent(
            responses={
                "I love this!": "positive",
                "I hate this!": "negative",
            }
        )
        writer = MockWriter()

        samples = [
            Sample(
                task=Task(id="t1", input="I love this!", expected="positive"),
                method="classify",
                agent_class="SentimentAgent",
                scorers=[ScorerConfig(name="exact", weight=1.0, scorer=ExactMatchScorer())],
                agent_factory=lambda: agent,
            ),
            Sample(
                task=Task(id="t2", input="I hate this!", expected="negative"),
                method="classify",
                agent_class="SentimentAgent",
                scorers=[ScorerConfig(name="exact", weight=1.0, scorer=ExactMatchScorer())],
                agent_factory=lambda: agent,
            ),
        ]

        config = PipelineConfig(trace_dir=tmp_path / "traces")

        results = await run_evaluation(samples, config, writer, max_concurrent=5)

        assert len(results) == 2
        assert len(writer.results) == 2


# =============================================================================
# Integration test with real capability test data
# =============================================================================


class TestCapabilityTestsIntegration:
    """Integration tests using capability test data format."""

    @pytest.mark.asyncio
    async def test_sentiment_single_format(self, tmp_path):
        """Test that we can run sentiment_single capability test format."""
        # This mimics the test_sentiment_single test function format
        text = "I absolutely love this product, it exceeded all my expectations!"
        expected = "positive"

        # Mock agent that returns correct answer
        agent = MockSentimentAgent(responses={text: expected})
        writer = MockWriter()

        sample = Sample(
            task=Task(id="sentiment_single_001", input=text, expected=expected),
            method="classify_single",
            agent_class="SentimentAgent",
            scorers=[ScorerConfig(name="exact_match", weight=1.0, scorer=ExactMatchScorer())],
            agent_factory=lambda: agent,
            tier=Tier.STABLE,
        )

        config = PipelineConfig(trace_dir=tmp_path / "traces")

        result = await process_sample(sample, config, writer)

        assert result.passed is True
        assert result.output == "positive"
        assert result.scores["exact_match"].score == 1.0

    @pytest.mark.asyncio
    async def test_calculate_single_format(self, tmp_path):
        """Test calculate capability test format."""
        a, b = 17, 23
        expected = a * b

        # Mock agent that returns correct answer
        class MockCalculateAgent:
            async def run(self, input: str) -> int:
                return 391  # 17 * 23

        writer = MockWriter()

        sample = Sample(
            task=Task(id="calculate_single_001", input=f"{a} * {b}", expected=expected),
            method="calculate",
            agent_class="CalculateAgent",
            scorers=[ScorerConfig(name="exact_match", weight=1.0, scorer=ExactMatchScorer())],
            agent_factory=MockCalculateAgent,
            tier=Tier.STABLE,
        )

        config = PipelineConfig(trace_dir=tmp_path / "traces")

        result = await process_sample(sample, config, writer)

        assert result.passed is True
        assert result.output == 391


class TestSubprocessOnComplete:
    """Tests for subprocess_on_complete error handling (the MR #132 fixes)."""

    def _make_sample(self, task_id: str, run_id: int = 1) -> Sample:
        return Sample(
            task=Task(id=task_id, input="test", expected="x", run_id=run_id),
            method="classify",
            agent_class="DummyAgent",
            scorers=[ScorerConfig(name="exact", weight=1.0, scorer=ExactMatchScorer())],
            agent_factory=lambda: None,
            model="test",
            test_name="crash_test",
            sample_id=task_id,
            tier=Tier.STABLE,
            agent_module="eval_pipeline._test_fixtures",
            client_config={},
            scorer_specs=[
                {
                    "name": "exact",
                    "weight": 1.0,
                    "scorer_class": "eval_pipeline.scoring.ExactMatchScorer",
                    "scorer_kwargs": {},
                }
            ],
        )

    def test_on_progress_fires_even_when_writer_raises(self, tmp_path):
        """on_progress is called even if writer.append_result raises."""
        import asyncio
        from unittest.mock import patch

        progress_calls: list = []

        class FailingWriter(MockWriter):
            def append_result(self, result):
                raise OSError("disk full")

        sample = self._make_sample("crash_001")
        writer = FailingWriter()
        config = PipelineConfig(
            trace_dir=tmp_path,
            engine_type="subprocess",
            pass_threshold=0.5,
            on_progress=lambda completed, total, result: progress_calls.append(result),
        )

        async def fake_run_tasks(task_ids, task_data, config, on_task_complete=None):
            for tid in task_ids:
                if on_task_complete:
                    on_task_complete(tid, RuntimeError("subprocess crashed"))
            return [RuntimeError("subprocess crashed")]

        async def run():
            with patch("eval_pipeline.pipeline.SubprocessEngine") as MockEngine:
                MockEngine.return_value.run_tasks = fake_run_tasks
                await run_evaluation([sample], config, writer, max_concurrent=1)

        asyncio.run(run())

        assert len(progress_calls) == 1, "on_progress must fire even when writer raises"
        assert progress_calls[0].passed is False
        assert progress_calls[0].error_type == "SubprocessError"

    def test_crash_stub_uses_correct_run_id(self, tmp_path):
        """Crash stub EvalTestResult uses run_id from the sample, not hardcoded 1."""
        import asyncio
        from unittest.mock import patch

        sample = self._make_sample("crash_run2", run_id=2)
        writer = MockWriter()
        config = PipelineConfig(
            trace_dir=tmp_path,
            engine_type="subprocess",
            pass_threshold=0.5,
        )

        async def fake_run_tasks(task_ids, task_data, config, on_task_complete=None):
            for tid in task_ids:
                if on_task_complete:
                    on_task_complete(tid, RuntimeError("subprocess crashed"))
            return [RuntimeError("subprocess crashed")]

        async def run():
            with patch("eval_pipeline.pipeline.SubprocessEngine") as MockEngine:
                MockEngine.return_value.run_tasks = fake_run_tasks
                await run_evaluation([sample], config, writer, max_concurrent=1)

        asyncio.run(run())

        assert len(writer.results) == 1
        stub = writer.results[0]
        assert stub.run_id == 2, f"Expected run_id=2, got {stub.run_id}"
        assert stub.variant == "run2", f"Expected variant='run2', got {stub.variant}"
        assert stub.passed is False
        assert stub.error_type == "SubprocessError"


class TestParallelExecution:
    """Tests for parallel execution of samples."""

    @pytest.mark.asyncio
    async def test_parallel_execution_is_faster(self, tmp_path):
        """Verify that parallel execution actually runs samples concurrently.

        Creates 4 samples that each take 0.5s to execute.
        Sequential: 4 * 0.5s = 2.0s
        Parallel (4): ~0.5s (all run together)

        We verify parallel is significantly faster than sequential.
        """
        import asyncio
        import time

        execution_times: list[float] = []

        class SlowAgent:
            """Agent that takes 0.5 seconds to respond."""

            async def run(self, input: str) -> str:
                start = time.time()
                await asyncio.sleep(0.5)
                execution_times.append(time.time() - start)
                return "done"

        writer = MockWriter()
        config = PipelineConfig(trace_dir=tmp_path / "traces")

        # Create 4 samples
        samples = [
            Sample(
                task=Task(id=f"slow_{i}", input=f"task {i}", expected="done"),
                method="run",
                agent_class="SlowAgent",
                scorers=[ScorerConfig(name="exact", weight=1.0, scorer=ExactMatchScorer())],
                agent_factory=SlowAgent,
                tier=Tier.STABLE,
            )
            for i in range(4)
        ]

        # Run in parallel
        start = time.time()
        results = await run_evaluation(samples, config, writer, max_concurrent=4)
        parallel_time = time.time() - start

        assert len(results) == 4
        assert all(r.passed for r in results)

        # Parallel should take ~0.5-0.6s, not ~2.0s
        # Allow some overhead, but should be well under 1.5s
        assert parallel_time < 1.5, f"Parallel took {parallel_time:.2f}s, expected < 1.5s"

    @pytest.mark.asyncio
    async def test_sequential_execution_takes_longer(self, tmp_path):
        """Verify that sequential execution (max_concurrent=1) is slower."""
        import asyncio
        import time

        class SlowAgent:
            async def run(self, input: str) -> str:
                await asyncio.sleep(0.2)
                return "done"

        writer = MockWriter()
        config = PipelineConfig(trace_dir=tmp_path / "traces")

        samples = [
            Sample(
                task=Task(id=f"slow_{i}", input=f"task {i}", expected="done"),
                method="run",
                agent_class="SlowAgent",
                scorers=[ScorerConfig(name="exact", weight=1.0, scorer=ExactMatchScorer())],
                agent_factory=SlowAgent,
                tier=Tier.STABLE,
            )
            for i in range(4)
        ]

        # Run sequentially
        start = time.time()
        results = await run_evaluation(samples, config, writer, max_concurrent=1)
        sequential_time = time.time() - start

        assert len(results) == 4
        # Sequential should take ~0.8s (4 * 0.2s)
        assert sequential_time >= 0.7, f"Sequential took {sequential_time:.2f}s, expected >= 0.7s"
