# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Core protocol definitions for the task execution infrastructure.

This module defines the execution-engine and trace-analysis protocols along with
the usage-statistics data structures shared across the evaluation runner.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar

# ===========================
# Layer 0/1/2 Protocols
# ===========================
# These protocols define the lower-level execution and analysis infrastructure
# that the task runner builds on.

R = TypeVar("R", covariant=True)


@dataclass
class EngineConfig:
    """Configuration for execution engines (Layer 0).

    Generic configuration that all execution engines understand.
    Engine-specific configs can extend this.
    """

    max_concurrent: int = 10
    timeout_seconds: float | None = None
    enable_checkpointing: bool = True


@dataclass
class TaskState:
    """Checkpoint state for a single task.

    Used by execution engines to resume from crashes or skip completed tasks.
    """

    task_id: str
    completed: bool
    result: Any | None = None
    error: str | None = None
    timestamp: float | None = None


class ExecutionEngine(Protocol):
    """Protocol for swappable execution engines (Layer 0).

    Different implementations provide different execution strategies:
    - AsyncIOEngine: Async I/O with semaphore (for I/O-bound tasks like LLM APIs)
    - MultiprocessEngine: Process pool (for CPU-bound tasks like local models)
    - RayEngine: Distributed execution across cluster (for massive scale)
    - NemoRunEngine: Slurm/cloud batch submission (for HPC clusters)

    All engines share the same interface, allowing Layer 2 (Runner) to be
    agnostic to the execution strategy.
    """

    async def run_tasks(
        self,
        task_ids: list[str],
        task_fn: Callable[[str], Awaitable[R]],
        config: EngineConfig,
        checkpoint_state: list[TaskState] | None = None,
        on_task_complete: Callable[[str, R | Exception], None] | None = None,
    ) -> list[R]:
        """Execute tasks using engine-specific strategy.

        Args:
            task_ids: List of task identifiers
            task_fn: Async function that takes task_id and returns result
            config: Engine configuration
            checkpoint_state: Previous run state for resume (optional)
            on_task_complete: Callback for each completed task (optional)

        Returns:
            List of results in same order as task_ids
        """
        ...


@dataclass
class ModelUsageStats:
    """Usage statistics for a single model."""

    model_name: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def p95_latency_ms(self) -> float | None:
        """95th percentile latency in milliseconds."""
        if not self.latencies_ms:
            return None
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[idx]


@dataclass
class TaskUsageStats:
    """Usage statistics extracted from a single task's trace."""

    task_id: str
    total_runtime_seconds: float
    models_used: list[ModelUsageStats] = field(default_factory=list)
    total_llm_calls: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens across all models."""
        return sum(m.total_tokens for m in self.models_used)

    @property
    def total_prompt_tokens(self) -> int:
        """Total prompt tokens across all models."""
        return sum(m.prompt_tokens for m in self.models_used)

    @property
    def total_completion_tokens(self) -> int:
        """Total completion tokens across all models."""
        return sum(m.completion_tokens for m in self.models_used)


@dataclass
class AggregateUsageStats:
    """Aggregate usage statistics across multiple tasks."""

    num_tasks: int
    per_task_stats: list[TaskUsageStats] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        """Total tokens across all tasks."""
        return sum(s.total_tokens for s in self.per_task_stats)

    @property
    def total_runtime_seconds(self) -> float:
        """Total runtime across all tasks."""
        return sum(s.total_runtime_seconds for s in self.per_task_stats)

    @property
    def models_breakdown(self) -> dict[str, ModelUsageStats]:
        """Aggregate stats per model across all tasks."""
        by_model: dict[str, ModelUsageStats] = {}

        for task_stats in self.per_task_stats:
            for model_stats in task_stats.models_used:
                if model_stats.model_name not in by_model:
                    by_model[model_stats.model_name] = ModelUsageStats(
                        model_name=model_stats.model_name
                    )

                agg = by_model[model_stats.model_name]
                agg.prompt_tokens += model_stats.prompt_tokens
                agg.completion_tokens += model_stats.completion_tokens
                agg.total_tokens += model_stats.total_tokens
                agg.call_count += model_stats.call_count
                agg.latencies_ms.extend(model_stats.latencies_ms)

        return by_model


class TraceAnalyzer(Protocol):
    """Protocol for analyzing OTel traces to extract usage statistics.

    Implementations read .jsonl trace files and extract:
    - Token counts per model
    - LLM call latencies
    - Total runtime

    This avoids duplicating trace data in IntermediateSteps.
    """

    def analyze_trace(self, trace_path: str) -> TaskUsageStats:
        """Analyze a single trace file and extract usage statistics.

        Args:
            trace_path: Path to .jsonl trace file

        Returns:
            TaskUsageStats with extracted metrics
        """
        ...
