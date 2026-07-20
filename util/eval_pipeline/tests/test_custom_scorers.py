# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for custom scorer support: dynamic import, kwargs forwarding, metadata flow."""

import sys
import textwrap

import pytest

from eval_pipeline.models import ExecutionResult, ScoreResult, ScoringContext
from eval_pipeline.scoring import ScorerConfig, build_scoring_context, score_task

# ============================================================
# Metadata flow: Task.metadata → ScoringContext.metadata
# ============================================================


class TestMetadataFlow:
    """Verify that task metadata reaches ScoringContext."""

    def test_build_scoring_context_passes_metadata(self, tmp_path):
        """Metadata dict is forwarded into ScoringContext."""
        trace_file = tmp_path / "trace.jsonl"
        trace_file.write_text("")

        metadata = {"difficulty": "hard", "rubric": "Check billing category"}

        result = ExecutionResult(
            task_id="t1",
            input="hello",
            expected="billing",
            actual="billing",
            trace_file=trace_file,
            latency_ms=50.0,
        )

        ctx = build_scoring_context(result, metadata=metadata)

        assert ctx.metadata == metadata
        assert ctx.metadata["difficulty"] == "hard"
        assert ctx.metadata["rubric"] == "Check billing category"

    def test_build_scoring_context_none_metadata_becomes_empty_dict(self, tmp_path):
        """Explicitly passing metadata=None normalizes to {}."""
        trace_file = tmp_path / "trace.jsonl"
        trace_file.write_text("")

        result = ExecutionResult(
            task_id="t_none",
            input="hello",
            expected="world",
            actual="world",
            trace_file=trace_file,
            latency_ms=50.0,
        )

        ctx = build_scoring_context(result, metadata=None)
        assert ctx.metadata == {}

    def test_build_scoring_context_defaults_to_empty_metadata(self, tmp_path):
        """When no metadata is passed, ScoringContext.metadata defaults to {}."""
        trace_file = tmp_path / "trace.jsonl"
        trace_file.write_text("")

        result = ExecutionResult(
            task_id="t2",
            input="hello",
            expected="world",
            actual="world",
            trace_file=trace_file,
            latency_ms=50.0,
        )

        ctx = build_scoring_context(result)

        assert ctx.metadata == {}

    def test_scorer_can_read_metadata(self):
        """A scorer can access per-task metadata from ctx.metadata."""

        class MetadataScorer:
            def score(self, ctx: ScoringContext) -> ScoreResult:
                rubric = ctx.metadata.get("rubric", "")
                return ScoreResult(
                    score=1.0 if rubric else 0.0,
                    reasoning=f"rubric={'present' if rubric else 'missing'}",
                )

        ctx = ScoringContext(
            task_id="t3",
            input="hello",
            expected="billing",
            actual="billing",
            metadata={"rubric": "Check billing category", "difficulty": "hard"},
        )

        result = MetadataScorer().score(ctx)
        assert result.score == 1.0
        assert "present" in result.reasoning

    @pytest.mark.asyncio
    async def test_score_task_with_metadata_scorer(self):
        """Full scoring pipeline with a metadata-aware scorer."""

        class DifficultyWeightedScorer:
            """Scores 1.0 for exact match, 0.5 bonus info from metadata."""

            def score(self, ctx: ScoringContext) -> ScoreResult:
                match = ctx.expected == ctx.actual
                difficulty = ctx.metadata.get("difficulty", "easy")
                return ScoreResult(
                    score=1.0 if match else 0.0,
                    reasoning=f"match={match}, difficulty={difficulty}",
                    metadata={"difficulty": difficulty},
                )

        ctx = ScoringContext(
            task_id="t4",
            input="hello",
            expected="positive",
            actual="positive",
            metadata={"difficulty": "hard", "domain": "sentiment"},
        )

        scorers = [
            ScorerConfig(name="weighted", weight=1.0, scorer=DifficultyWeightedScorer()),
        ]

        scores = await score_task(ctx, scorers)
        assert scores["weighted"]["score"] == 1.0
        assert scores["weighted"]["metadata"]["difficulty"] == "hard"


# ============================================================
# Dynamic import of custom scorers from YAML config
# ============================================================


class TestDynamicScorerImport:
    """Test that the config loader dynamically imports custom scorers."""

    def test_dynamic_import_no_args(self, tmp_path):
        """Custom scorer with no constructor params is imported and instantiated."""
        # Create a scorer module on disk
        scorer_pkg = tmp_path / "my_scorers"
        scorer_pkg.mkdir()
        (scorer_pkg / "__init__.py").write_text("")
        (scorer_pkg / "basic.py").write_text(
            textwrap.dedent("""\
            from eval_pipeline.models import ScoreResult, ScoringContext

            class AlwaysPassScorer:
                def score(self, ctx: ScoringContext) -> ScoreResult:
                    return ScoreResult(score=1.0, reasoning="always pass")
        """)
        )

        sys.path.insert(0, str(tmp_path))
        try:
            from eval_pipeline.config import _create_scorers

            scorer_configs = [
                {
                    "name": "always_pass",
                    "class": "my_scorers.basic.AlwaysPassScorer",
                    "weight": 1.0,
                },
            ]
            scorers = _create_scorers(scorer_configs, model_specs={})

            assert len(scorers) == 1
            assert scorers[0].name == "always_pass"

            # Verify it actually works
            ctx = ScoringContext(
                task_id="t1",
                input="x",
                expected="y",
                actual="y",
            )
            result = scorers[0].scorer.score(ctx)
            assert result.score == 1.0
        finally:
            sys.path.remove(str(tmp_path))
            sys.modules.pop("my_scorers", None)
            sys.modules.pop("my_scorers.basic", None)

    def test_dynamic_import_with_kwargs(self, tmp_path):
        """Extra YAML keys are forwarded as kwargs to the custom scorer."""
        scorer_pkg = tmp_path / "my_scorers2"
        scorer_pkg.mkdir()
        (scorer_pkg / "__init__.py").write_text("")
        (scorer_pkg / "configurable.py").write_text(
            textwrap.dedent("""\
            from eval_pipeline.models import ScoreResult, ScoringContext

            class ThresholdScorer:
                def __init__(self, threshold=0.5, categories=None):
                    self.threshold = threshold
                    self.categories = categories or []

                def score(self, ctx: ScoringContext) -> ScoreResult:
                    match = ctx.expected == ctx.actual
                    return ScoreResult(
                        score=1.0 if match else 0.0,
                        reasoning=f"threshold={self.threshold}",
                    )
        """)
        )

        sys.path.insert(0, str(tmp_path))
        try:
            from eval_pipeline.config import _create_scorers

            scorer_configs = [
                {
                    "name": "threshold",
                    "class": "my_scorers2.configurable.ThresholdScorer",
                    "weight": 1.0,
                    "threshold": 0.8,
                    "categories": ["billing", "technical"],
                },
            ]
            scorers = _create_scorers(scorer_configs, model_specs={})

            assert len(scorers) == 1
            assert scorers[0].scorer.threshold == 0.8
            assert scorers[0].scorer.categories == ["billing", "technical"]
        finally:
            sys.path.remove(str(tmp_path))
            sys.modules.pop("my_scorers2", None)
            sys.modules.pop("my_scorers2.configurable", None)

    def test_unknown_unqualified_name_raises(self):
        """Unqualified names that aren't built-in raise ValueError."""
        from eval_pipeline.config import _create_scorers

        scorer_configs = [
            {"name": "bad", "class": "NoSuchScorer", "weight": 1.0},
        ]
        with pytest.raises(ValueError, match="Unknown scorer class: NoSuchScorer"):
            _create_scorers(scorer_configs, model_specs={})

    def test_nonexistent_module_raises_with_context(self):
        """Dotted path to nonexistent module gives a clear error."""
        from eval_pipeline.config import _create_scorers

        scorer_configs = [
            {"name": "bad", "class": "nonexistent.module.Scorer", "weight": 1.0},
        ]
        with pytest.raises(ValueError, match="cannot import module 'nonexistent.module'"):
            _create_scorers(scorer_configs, model_specs={})

    def test_nonexistent_class_in_module_raises_with_context(self):
        """Dotted path to existing module but missing class gives a clear error."""
        from eval_pipeline.config import _create_scorers

        scorer_configs = [
            {"name": "bad", "class": "os.NonExistentClass", "weight": 1.0},
        ]
        with pytest.raises(ValueError, match="module 'os' has no class 'NonExistentClass'"):
            _create_scorers(scorer_configs, model_specs={})

    def test_bad_kwargs_raises_type_error(self, tmp_path):
        """Extra kwargs not accepted by the scorer constructor raise TypeError."""
        scorer_pkg = tmp_path / "my_scorers3"
        scorer_pkg.mkdir()
        (scorer_pkg / "__init__.py").write_text("")
        (scorer_pkg / "strict.py").write_text(
            textwrap.dedent("""\
            class StrictScorer:
                def __init__(self):
                    pass
                def score(self, ctx):
                    return {"score": 1.0, "reasoning": "ok"}
        """)
        )

        sys.path.insert(0, str(tmp_path))
        try:
            from eval_pipeline.config import _create_scorers

            scorer_configs = [
                {
                    "name": "strict",
                    "class": "my_scorers3.strict.StrictScorer",
                    "weight": 1.0,
                    "bogus_key": 42,
                },
            ]
            with pytest.raises(TypeError):
                _create_scorers(scorer_configs, model_specs={})
        finally:
            sys.path.remove(str(tmp_path))
            sys.modules.pop("my_scorers3", None)
            sys.modules.pop("my_scorers3.strict", None)

    def test_builtin_scorers_still_work(self):
        """Built-in scorers (ExactMatchScorer, TypeMatchScorer) still load."""
        from eval_pipeline.config import _create_scorers

        scorer_configs = [
            {"name": "exact", "class": "ExactMatchScorer", "weight": 1.0},
            {"name": "type", "class": "TypeMatchScorer", "weight": 0.5},
        ]
        scorers = _create_scorers(scorer_configs, model_specs={})

        assert len(scorers) == 2
        assert scorers[0].name == "exact"
        assert scorers[1].name == "type"


# ============================================================
# Per-task rubric pattern (metadata-driven custom scoring)
# ============================================================


class TestPerTaskRubric:
    """Demonstrate the per-task rubric pattern using metadata."""

    def test_per_task_rubric_via_metadata(self):
        """Custom scorer reads a per-task rubric from ctx.metadata."""

        class RubricScorer:
            def __init__(self, default_rubric="be correct"):
                self.default_rubric = default_rubric

            def score(self, ctx: ScoringContext) -> ScoreResult:
                rubric = ctx.metadata.get("rubric", self.default_rubric)
                match = ctx.expected == ctx.actual
                return ScoreResult(
                    score=1.0 if match else 0.0,
                    reasoning=f"rubric='{rubric}', match={match}",
                    metadata={"rubric_used": rubric},
                )

        # Task 1: has a custom rubric in metadata
        ctx1 = ScoringContext(
            task_id="task_001",
            input="query",
            expected="billing",
            actual="billing",
            metadata={"rubric": "Must identify billing category AND cite pricing"},
        )
        r1 = RubricScorer(default_rubric="fallback").score(ctx1)
        assert r1.score == 1.0
        assert r1.metadata["rubric_used"] == "Must identify billing category AND cite pricing"

        # Task 2: no rubric in metadata, falls back to default
        ctx2 = ScoringContext(
            task_id="task_002",
            input="query",
            expected="technical",
            actual="technical",
            metadata={},
        )
        r2 = RubricScorer(default_rubric="fallback").score(ctx2)
        assert r2.metadata["rubric_used"] == "fallback"
