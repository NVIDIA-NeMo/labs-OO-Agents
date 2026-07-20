# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parser for .noo-eval.jsonl files.

Provides stateful parsing of eval files with version checking and error handling.

Usage:
    from eval_pipeline.eval_parser import EvalFileParser

    parser = EvalFileParser()
    metadata, results, completion, annotations = parser.parse_file(Path("experiment.noo-eval.jsonl"))

    # Or parse lines incrementally
    for line in file:
        parsed = parser.parse_line(line)
"""

import json
import time
from pathlib import Path

from .eval_types import (
    SUPPORTED_VERSIONS,
    EvalAnnotationLine,
    EvalCompletionLine,
    EvalLine,
    EvalMetadataLine,
    EvalTestResult,
)


class EvalParseError(Exception):
    """Error parsing eval file line.

    Raised for:
    - Invalid JSON
    - Unknown line type
    - Unsupported schema version

    Pydantic ValidationError is NOT wrapped - it propagates with detailed field info.
    """

    def __init__(
        self,
        message: str,
        line_number: int | None = None,
        raw_line: str | None = None,
    ):
        self.line_number = line_number
        self.raw_line = raw_line
        super().__init__(message)

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.line_number is not None:
            parts.append(f"(line {self.line_number})")
        return " ".join(parts)


class EvalFileParser:
    """Stateful parser for .noo-eval.jsonl files.

    Tracks schema version from metadata line and validates subsequent lines.
    """

    def __init__(self):
        self.version: str | None = None

    def parse_line(self, line: str, line_number: int | None = None) -> EvalLine:
        """Parse a single JSONL line.

        Args:
            line: Raw JSON line string
            line_number: Optional line number for error messages

        Returns:
            Parsed EvalLine (one of the line type models)

        Raises:
            EvalParseError: For JSON errors, unknown types, or unsupported versions
            pydantic.ValidationError: For schema validation failures (propagates)
        """
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            raise EvalParseError(f"Invalid JSON: {e}", line_number, line) from e

        line_type = data.get("_type")

        if line_type == "metadata":
            self.version = data.get("version", "1")
            if self.version not in SUPPORTED_VERSIONS:
                raise EvalParseError(
                    f"Unsupported eval file version: {self.version}",
                    line_number,
                )
            return EvalMetadataLine.model_validate(data)

        elif line_type == "result":
            return EvalTestResult.model_validate(data)

        elif line_type == "completion":
            return EvalCompletionLine.model_validate(data)

        elif line_type == "annotation":
            return EvalAnnotationLine.model_validate(data)

        else:
            raise EvalParseError(
                f"Unknown line type: {line_type!r}",
                line_number,
                line,
            )

    def parse_file(
        self, path: Path
    ) -> tuple[
        EvalMetadataLine,
        list[EvalTestResult],
        EvalCompletionLine | None,
        list[EvalAnnotationLine],
    ]:
        """Parse entire file into structured components.

        Args:
            path: Path to .noo-eval.jsonl file

        Returns:
            Tuple of (metadata, results, completion, annotations)
            - metadata: Always present (raises if missing)
            - results: List of test results
            - completion: None if experiment still running
            - annotations: List of annotations (may be empty)

        Raises:
            EvalParseError: If no metadata line found or other parse errors
            pydantic.ValidationError: For schema validation failures
            FileNotFoundError: If file doesn't exist
        """
        metadata: EvalMetadataLine | None = None
        results: list[EvalTestResult] = []
        completion: EvalCompletionLine | None = None
        annotations: list[EvalAnnotationLine] = []

        with open(path) as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                parsed = self.parse_line(line, line_number)

                if isinstance(parsed, EvalMetadataLine):
                    metadata = parsed
                elif isinstance(parsed, EvalTestResult):
                    results.append(parsed)
                elif isinstance(parsed, EvalCompletionLine):
                    completion = parsed
                elif isinstance(parsed, EvalAnnotationLine):
                    annotations.append(parsed)

        if metadata is None:
            raise EvalParseError("No metadata line found in file")

        return metadata, results, completion, annotations


def get_experiment_status(
    completion: EvalCompletionLine | None,
    file_mtime: float,
    stale_threshold_seconds: float = 60.0,
) -> str:
    """Derive experiment status from completion line and file age.

    Args:
        completion: Completion line if present, None if experiment may be running
        file_mtime: File modification time (from os.stat().st_mtime)
        stale_threshold_seconds: Seconds without modification before marking stale

    Returns:
        Status string: "completed", "failed", "running", or "stale"
    """
    if completion:
        return completion.status

    # No completion line - check if still being written
    seconds_since_modified = time.time() - file_mtime
    if seconds_since_modified > stale_threshold_seconds:
        return "stale"
    return "running"
