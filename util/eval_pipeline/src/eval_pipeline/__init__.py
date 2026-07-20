# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluation pipeline for running and scoring agent tests.

This package provides a high-level API for evaluating agents:

Example (Python - recommended):
    from eval_pipeline import Evaluator, ExactMatchScorer
    from nooa.unifiedllm import CompletionClient

    # Create evaluator with your LLM clients
    evaluator = Evaluator(
        models={
            "gpt-4": CompletionClient(model="openai/gpt-4", ...),
            "claude": CompletionClient(model="anthropic/claude-3", ...),
        },
        output_dir="experiments/results",
    )

    # Add tests programmatically
    evaluator.add_test(
        name="sentiment",
        agent_class=SentimentAgent,
        method="classify_single",
        data=[
            {"kwargs": {"text": "I love this"}, "expected": "positive"},
            {"kwargs": {"text": "I hate this"}, "expected": "negative"},
        ],
        scorers=[ExactMatchScorer()],
    )

    # Run with specific models and multiple runs
    results = await evaluator.run(models=["gpt-4"], runs=3)
    print(results.summary())  # "15/18 passed (83.3%)"

Example (from YAML config):
    from eval_pipeline import Evaluator

    evaluator = Evaluator.from_config("experiments/config.yaml")
    results = await evaluator.run(quiet=True)

CLI usage:
    python -m eval_pipeline --config config.yaml --runs 3 -q
"""

# High-level API
# Config loading
from .config import EvalConfig, evaluator_from_config, load_config

# Eval types (canonical schema for .noo-eval.jsonl files)
from .eval_parser import EvalFileParser, EvalParseError, get_experiment_status
from .eval_types import (
    SUPPORTED_VERSIONS,
    EvalAnnotationLine,
    EvalCompletionLine,
    EvalLine,
    EvalMetadata,
    EvalMetadataLine,
    EvalTestResult,
    ModelSpec,
    ScoreDetail,
    Tier,
)
from .evaluator import EvalResults, EvalTest, Evaluator

# Low-level primitives (for custom pipelines)
from .execute import execute_task
from .experiment_writer import ExperimentWriter
from .models import ExecutionResult, ScoreResult, ScoringContext, Task
from .pipeline import PipelineConfig, Sample, process_sample, run_evaluation

# Scorers
from .scoring import (
    ExactMatchScorer,
    LLMJudgeScorer,
    LLMMethodologyScorer,
    ModeSelectionScorer,
    ScorerConfig,
    build_scoring_context,
    score_task,
)
from .trace_eval_span import write_eval_span_to_trace

__all__ = [
    "EvalAnnotationLine",
    "EvalCompletionLine",
    "EvalConfig",
    "EvalFileParser",
    "EvalLine",
    "EvalMetadata",
    "EvalMetadataLine",
    "EvalParseError",
    "EvalResults",
    "EvalTest",
    "EvalTestResult",
    "Evaluator",
    "ExactMatchScorer",
    "ExecutionResult",
    "ExperimentWriter",
    "LLMJudgeScorer",
    "ModeSelectionScorer",
    "LLMMethodologyScorer",
    "ModelSpec",
    "PipelineConfig",
    "Sample",
    "ScoreDetail",
    "ScoreResult",
    "ScorerConfig",
    "ScoringContext",
    "SUPPORTED_VERSIONS",
    "Task",
    "Tier",
    "build_scoring_context",
    "evaluator_from_config",
    "execute_task",
    "get_experiment_status",
    "load_config",
    "process_sample",
    "run_evaluation",
    "score_task",
    "write_eval_span_to_trace",
]
