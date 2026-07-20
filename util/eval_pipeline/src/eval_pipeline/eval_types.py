# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical Pydantic types for .noo-eval.jsonl files.

This module defines the schema for evaluation result files.
Used by both producers (eval_pipeline, e2e_optimization) and
consumers (prompt-opt viewer).

File format (JSONL):
- Line 1: EvalMetadataLine (version, experiment config)
- Lines 2-N: EvalTestResult (one per test)
- Line N+1: EvalCompletionLine (marks complete, optional if still running)
- Optional: EvalAnnotationLine (viewer-added notes, can appear anywhere after metadata)

Example:
    {"_type": "metadata", "version": "1", "metadata": {...}}
    {"_type": "result", "test_id": "test_001", ...}
    {"_type": "result", "test_id": "test_002", ...}
    {"_type": "completion", "status": "completed", ...}
    {"_type": "annotation", "test_id": "test_001", "annotation": "Needs review"}
"""

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

# Supported schema versions
SUPPORTED_VERSIONS = {"1"}


# =============================================================================
# Test Tier
# =============================================================================


class Tier(StrEnum):
    """Tier of a test.

    - STABLE: Must pass
    - FRONTIER: At the edge of capability
    - HORIZON: Aspirational - cannot yet handle
    """

    STABLE = "stable"
    FRONTIER = "frontier"
    HORIZON = "horizon"


# =============================================================================
# Supporting Types
# =============================================================================


class ModelSpec(BaseModel):
    """Model configuration for traceability."""

    id: str  # Short ID (e.g., "gpt4")
    model_name: str  # Full model name (e.g., "openai/gpt-4")
    endpoint: str | None = None  # API endpoint URL
    api_key_env: str | None = None  # Env var for API key
    max_tokens: int | None = None  # Max tokens limit
    # Sampling parameters (forwarded to the client / litellm)
    temperature: float | None = None  # Sampling temperature
    top_p: float | None = None  # Nucleus sampling probability
    # Reasoning model parameters
    reasoning_effort: str | None = None  # "low", "medium", "high" for reasoning models
    max_thinking_tokens: int | None = None  # Cap on reasoning tokens (nvext)
    # Retry configuration (for handling transient failures)
    max_retries: int | None = None  # Max retry attempts (default: 3 if retry enabled)
    retry_on_empty_content: bool = False  # Retry when reasoning models return empty content
    # Cache control (NVIDIA inference API caches by default, bad for pass@k diversity)
    no_cache: bool = False
    # Client type: "completion" (default) or "responses" (Responses API)
    client_type: str | None = None


class ScoreDetail(BaseModel):
    """Result from a single scorer."""

    score: float  # 0.0 - 1.0 normalized score
    passed: bool  # Whether this scorer passed
    reasoning: str | None = None  # Human-readable explanation
    metrics: dict[str, Any] | None = None  # Scorer-specific data


# =============================================================================
# Metadata Line (First Line)
# =============================================================================


class EvalMetadata(BaseModel):
    """Experiment metadata (nested in metadata line)."""

    timestamp: str  # ISO format start time
    suite_name: str  # Experiment suite name
    models: list[ModelSpec | str]  # Models used (can be ID strings or full specs)

    # Optional config
    config_file: str | None = None  # Path to config file used
    tests: list[str] | None = None  # List of test names to run
    runs: int = 1  # Number of runs per test
    variants: list[str] | None = None  # Variant names


class EvalMetadataLine(BaseModel):
    """First line of .noo-eval.jsonl - experiment metadata with version."""

    type_: Literal["metadata"] = Field(default="metadata", alias="_type")
    version: Literal["1"] = "1"  # Schema version for compatibility
    metadata: EvalMetadata

    model_config = {"populate_by_name": True}


# =============================================================================
# Test Result Line (Lines 2-N)
# =============================================================================


class EvalTestResult(BaseModel):
    """Individual test result line."""

    type_: Literal["result"] = Field(default="result", alias="_type")

    # Identity
    test_id: str  # Unique ID (e.g., "sentiment_001_gpt4_run1")
    base_test_id: str | None = None  # For grouping runs (e.g., "sentiment_001_gpt4")
    run_id: int = 1  # Which run (1-based) for self-consistency
    test_case: str | None = None  # Logical test case ID (e.g., "sentiment_001"), model/run agnostic

    # Test info
    agent_class: str  # Agent class name (e.g., "SentimentAgent")
    method: str  # Method name (e.g., "classify_single")
    test_name: str | None = None  # Config test name (e.g., "json_qa_lookup") for unique grouping
    display_name: str | None = None  # Human-readable (e.g., "Classify sentiment")
    tier: Tier = Field(default=Tier.STABLE)  # Test tier

    # Model/variant
    model: str  # Model ID (e.g., "nvidia/llama-3.1-8b")
    variant: str  # Variant ID (e.g., "run1", "smart_run1")

    # Results
    passed: bool  # Overall pass/fail
    scores: dict[str, ScoreDetail]  # Per-scorer results

    # I/O
    input: Any  # Input passed to agent
    output: Any  # Actual output from agent
    expected: Any  # Expected output for comparison

    # Tracing
    trace_file: str | None = None  # Path to trace .jsonl

    # Token usage
    input_tokens: int | None = None  # Number of input (prompt) tokens
    output_tokens: int | None = None  # Number of output (completion) tokens
    total_tokens: int | None = None  # Total tokens used

    # Errors
    error: str | None = None  # Error message if failed
    error_type: str | None = None  # Error class name (e.g., "ValueError")
    memory_diag_file: str | None = None  # Path to memory diagnostics if OOM
    peak_rss_mb: float | None = None  # Peak RSS in MB (when memory monitoring enabled)

    # Timing
    started_at: str | None = None  # ISO format start time
    duration_seconds: float | None = None  # How long sample took to run
    worker_id: int | None = None  # Worker slot ID (1 to max_concurrent) for parallel execution

    # Arbitrary eval metadata — merged from config-level, test-level, and task-level.
    # Each key becomes an eval.{key} attribute on the OTLP span and a dynamic column
    # in the trace viewer.  Example: {"difficulty": "hard", "dataset": "kdd-cup-2026"}
    eval_metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


# =============================================================================
# Completion Line (Last Line)
# =============================================================================


class EvalCompletionLine(BaseModel):
    """Final line - marks experiment complete."""

    type_: Literal["completion"] = Field(default="completion", alias="_type")
    status: Literal["completed", "failed"]
    completed_at: str  # ISO format completion time
    result_count: int  # Number of test results written

    # Optional aggregate metrics
    passed_count: int | None = None
    success_rate: float | None = None
    duration_seconds: float | None = None

    # Token usage totals (summed across all results)
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None
    total_tokens: int | None = None

    model_config = {"populate_by_name": True}


# =============================================================================
# Annotation Line (Appended by Viewer)
# =============================================================================


class EvalAnnotationLine(BaseModel):
    """Annotation line - appended by viewer or humans."""

    type_: Literal["annotation"] = Field(default="annotation", alias="_type")
    test_id: str  # Links to EvalTestResult.test_id
    annotation: str  # Free-form annotation text
    timestamp: str | None = None  # When annotation was added (ISO format)
    author: str | None = None  # Who added it (optional)

    model_config = {"populate_by_name": True}


# =============================================================================
# Union Type for Parsing
# =============================================================================

EvalLine = EvalMetadataLine | EvalTestResult | EvalCompletionLine | EvalAnnotationLine


# =============================================================================
# Subprocess Boundary Types (JSON in → JSON out)
# =============================================================================


class AgentSpec(BaseModel):
    """Serializable recipe for constructing an agent in a subprocess."""

    agent_module: str  # e.g. "tests.capability.agents.sentiment"
    agent_class: str  # e.g. "SentimentAgent"
    method: str  # e.g. "classify_single"
    client_config: dict[str, Any]  # CompletionClient kwargs
    agent_file: str | None = None  # File path for file-based agents (--agent flag)


class ScorerSpec(BaseModel):
    """Serializable recipe for constructing a scorer in a subprocess."""

    name: str
    weight: float = 1.0
    scorer_class: str  # e.g. "eval_pipeline.scoring.ExactMatchScorer"
    scorer_kwargs: dict[str, Any] = Field(default_factory=dict)


class TaskSpec(BaseModel):
    """JSON-safe version of Task for subprocess boundary."""

    id: str
    input: Any  # (args_tuple, kwargs_dict)
    expected: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    run_id: int = 1


class SubprocessTaskInput(BaseModel):
    """Everything a subprocess worker needs. Pydantic → JSON on stdin."""

    agent_spec: AgentSpec
    task: TaskSpec
    scorers: list[ScorerSpec]
    sample_id: str

    # Sample metadata (for building EvalTestResult)
    model: str
    test_name: str | None = None
    display_name: str | None = None
    tier: str = "stable"
    agent_label: str | None = None

    # Pipeline config
    trace_dir: str
    use_otlp: bool = False
    write_trace_file: bool = True
    otlp_endpoint: str = "http://localhost:5001/v1/traces"
    viewer_endpoint: str | None = None  # external viewer /v1/traces URL (if running)
    experiment_name: str | None = None
    pass_threshold: float = 0.5
    timeout_seconds: float | None = None
    memory_limit_mb: int | None = None

    # Merged eval metadata (config + test + task levels) for eval.* span attributes
    eval_metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)
