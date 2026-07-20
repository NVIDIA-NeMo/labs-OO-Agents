# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Utility for writing .noo-eval.jsonl experiment files.

Provides API for evaluation runners to write results incrementally using
typed Pydantic models from eval_types.

Usage:
    from eval_pipeline.experiment_writer import ExperimentWriter
    from eval_pipeline.eval_types import EvalTestResult, ScoreDetail

    writer = ExperimentWriter(output_dir="results", experiment_name="sentiment")
    writer.start(
        suite_name="sentiment",
        models=["gpt-4"],
        config_file="config.yaml",
    )

    for result in run_tests():
        writer.append_result(result)  # Pass EvalTestResult or dict

    writer.finalize()  # Marks as completed
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from .eval_types import (
    EvalCompletionLine,
    EvalMetadata,
    EvalMetadataLine,
    EvalTestResult,
    ModelSpec,
)


class NullExperimentWriter:
    """No-op writer used when --no-files suppresses file output.

    Satisfies the ExperimentWriter interface but writes nothing to disk.
    Tracks pass/fail counts in memory so the rest of the pipeline is unchanged.
    """

    def __init__(self) -> None:
        self.file_path: Path | None = None
        self.result_count: int = 0
        self.passed_count: int = 0

    def start(self, **kwargs) -> Path | None:
        return self.file_path

    def append_result(self, result) -> None:
        self.result_count += 1
        passed = (
            result.get("passed", False)
            if isinstance(result, dict)
            else getattr(result, "passed", False)
        )
        if passed:
            self.passed_count += 1

    def finalize(self, status: str = "completed", extra_metrics=None) -> None:
        pass


class ExperimentWriter:
    """Manages writing .noo-eval.jsonl files incrementally.

    Uses typed Pydantic models for validation and consistent serialization.
    """

    def __init__(
        self,
        output_dir: str | Path,
        experiment_name: str,
        timestamp: datetime | None = None,
    ):
        """Initialize experiment writer.

        Args:
            output_dir: Directory to write results
            experiment_name: Name of the experiment (used in filename)
            timestamp: Optional timestamp (defaults to now)
        """
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_name = experiment_name.replace("_", "").replace(" ", "")
        self.timestamp = timestamp or datetime.now()
        self.timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")

        # Generate filename
        self.file_path = (
            self.output_dir / f"{self.experiment_name}_{self.timestamp_str}.noo-eval.jsonl"
        )
        self.started = False
        self.finalized = False
        self.result_count = 0
        self.passed_count = 0
        self._start_time: datetime | None = None

        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0

    def start(
        self,
        suite_name: str | None = None,
        models: list[ModelSpec | str] | None = None,
        config_file: str | None = None,
        tests: list[str] | None = None,
        runs: int = 1,
        variants: list[str] | None = None,
    ) -> Path:
        """Initialize the experiment file with metadata line.

        Args:
            suite_name: Experiment suite name (defaults to experiment_name)
            models: List of models used (ModelSpec or string IDs)
            config_file: Path to config file used
            tests: List of test names
            runs: Number of runs per test
            variants: List of variant names

        Returns:
            Path to the created file
        """
        if self.started:
            raise RuntimeError("Experiment already started")

        self._start_time = datetime.now()

        # Build typed metadata
        metadata = EvalMetadata(
            timestamp=self._start_time.isoformat(),
            suite_name=suite_name or self.experiment_name,
            models=models or [],
            config_file=config_file,
            tests=tests,
            runs=runs,
            variants=variants,
        )

        # Build metadata line with version
        metadata_line = EvalMetadataLine(metadata=metadata)

        # Write first line
        with open(self.file_path, "w") as f:
            f.write(metadata_line.model_dump_json(by_alias=True))
            f.write("\n")

        self.started = True
        return self.file_path

    def append_result(self, result: EvalTestResult | dict[str, Any]) -> None:
        """Append a test result to the file.

        Args:
            result: EvalTestResult instance or dict (validated as EvalTestResult)

        The result is written as a single JSONL line for crash-safe incremental updates.

        Raises:
            RuntimeError: If start() not called or already finalized
            pydantic.ValidationError: If dict doesn't match EvalTestResult schema
        """
        if not self.started:
            raise RuntimeError("Call start() before appending results")
        if self.finalized:
            raise RuntimeError("Cannot append after finalize()")

        # Convert dict to EvalTestResult if needed (validates schema)
        if isinstance(result, dict):
            result = EvalTestResult.model_validate(result)

        # Write as JSONL line with _type alias
        with open(self.file_path, "a") as f:
            f.write(result.model_dump_json(by_alias=True))
            f.write("\n")

        self.result_count += 1
        if result.passed:
            self.passed_count += 1

        # Accumulate token counts
        if result.input_tokens is not None:
            self.total_input_tokens += result.input_tokens
        if result.output_tokens is not None:
            self.total_output_tokens += result.output_tokens
        if result.total_tokens is not None:
            self.total_tokens += result.total_tokens

    def finalize(
        self,
        status: str = "completed",
        extra_metrics=None,
    ) -> None:
        """Mark experiment as completed by appending a completion record.

        Args:
            status: Final status ("completed" or "failed")
            extra_metrics: Unused; accepted for protocol compatibility.
        """
        if not self.started:
            raise RuntimeError("Cannot finalize before starting")
        if self.finalized:
            return  # Already finalized

        completed_at = datetime.now()
        duration_seconds = None
        if self._start_time:
            duration_seconds = (completed_at - self._start_time).total_seconds()

        success_rate = None
        if self.result_count > 0:
            success_rate = self.passed_count / self.result_count

        # Build typed completion line
        # Note: status is typed as Literal["completed", "failed"] so we cast
        completion = EvalCompletionLine(
            status=status if status in ("completed", "failed") else "completed",
            completed_at=completed_at.isoformat(),
            result_count=self.result_count,
            passed_count=self.passed_count,
            success_rate=success_rate,
            duration_seconds=duration_seconds,
            total_input_tokens=self.total_input_tokens if self.total_input_tokens > 0 else None,
            total_output_tokens=self.total_output_tokens if self.total_output_tokens > 0 else None,
            total_tokens=self.total_tokens if self.total_tokens > 0 else None,
        )

        # Write completion line (exclude None values to keep JSON clean)
        with open(self.file_path, "a") as f:
            f.write(completion.model_dump_json(by_alias=True, exclude_none=True))
            f.write("\n")

        self.finalized = True
