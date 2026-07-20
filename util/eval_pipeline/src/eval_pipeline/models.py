# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core data types for the evaluation pipeline.

This module defines the fundamental data structures used throughout the
evaluation pipeline:

- Task: Input specification for a single evaluation
- ExecutionResult: Raw output from running an agent
- ScoringContext: All data needed for scoring
- ScoreResult: Output from a scorer
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nooa.trace_explorer import TraceExplorer

# Type alias for task input: (args, kwargs) tuple for method invocation
TaskInput = tuple[tuple[Any, ...], dict[str, Any]]


@dataclass
class Task:
    """A single evaluation task to run.

    Attributes:
        id: Unique identifier for this task (e.g., "sentiment_001")
        input: Tuple of (args, kwargs) to pass to the agent method
        expected: Expected output for scoring comparison
        metadata: Optional metadata about the task (source, difficulty, etc.)
        run_id: Which run this is (1-based) for self-consistency scoring.
                When running the same task multiple times to measure
                consistency, this tracks which run we're on.
    """

    id: str
    input: TaskInput
    expected: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    run_id: int = 1


@dataclass
class ExecutionResult:
    """Result from executing a task (before scoring).

    Attributes:
        task_id: ID of the task that was executed
        input: The input that was passed to the agent
        expected: Expected output (copied from Task)
        actual: Actual output from the agent
        trace_file: Path to the trace file for this execution, or None when using OTLP (caller supplies path after fetch).
        latency_ms: Time taken for execution in milliseconds
        error: Error message if execution failed, None otherwise
        session_id: Session ID used for judge trace routing
    """

    task_id: str
    input: TaskInput
    expected: Any
    actual: Any
    trace_file: Path | None
    latency_ms: float
    error: str | None = None
    session_id: str | None = None


@dataclass
class ScoringContext:
    """All data available for scoring a task.

    Scorers receive this context and extract what they need.
    Some fields are optional and may not be populated depending
    on the pipeline configuration.

    Attributes:
        task_id: ID of the task being scored
        input: The input that was passed to the agent
        expected: Expected output for comparison
        actual: Actual output from the agent
        trace: TraceExplorer instance for span extraction, or None if unavailable
        latency_ms: Execution time in milliseconds
        input_tokens: Number of input (prompt) tokens used
        output_tokens: Number of output (completion) tokens generated
        total_tokens: Total number of tokens used
        metadata: Task metadata from the task JSON (difficulty, tags, etc.)
        error: Error message if execution failed
        session_id: Session ID used for judge trace routing
        use_otlp: When True, scorers should use set_session_id instead of set_trace_file for judge spans
    """

    task_id: str
    input: TaskInput
    expected: Any
    actual: Any

    trace: TraceExplorer | None = None
    latency_ms: float | None = None

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    # Task metadata from the task JSON — forwarded from Task.metadata.
    # Use this for per-task rubrics, difficulty tags, or any data a custom
    # scorer needs that varies per task.  Example task JSON:
    #
    #   {"id": "001", "expected": "billing", "metadata": {"rubric": "...", "difficulty": "hard"}}
    #
    # Then in a scorer:  rubric = ctx.metadata.get("rubric", "default")
    metadata: dict[str, Any] = field(default_factory=dict)

    # For error cases
    error: str | None = None

    # API mode: for judge trace routing
    session_id: str | None = None
    use_otlp: bool = False


@dataclass
class ScoreResult:
    """Result from a single scorer.

    Attributes:
        score: Normalized score from 0.0 (worst) to 1.0 (best)
        reasoning: Human-readable explanation of the score
        metadata: Additional scorer-specific data

    Raises:
        ValueError: If score is not in [0.0, 1.0] range
    """

    score: float
    reasoning: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be 0.0-1.0, got {self.score}")
