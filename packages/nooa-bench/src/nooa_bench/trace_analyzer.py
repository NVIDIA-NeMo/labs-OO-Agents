# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Trace analyzer for extracting usage statistics from execution traces.

This module reads JSONL trace files produced by the agent framework and
extracts per-model token counts, LLM call latencies, and total runtime.
"""

import json
import logging
from pathlib import Path
from typing import Any

from nooa_bench.protocol import (
    ModelUsageStats,
    TaskUsageStats,
)

logger = logging.getLogger(__name__)


class TraceAnalyzer:
    """
    Analyzes OTel traces to extract usage statistics.

    Implements the TraceAnalyzer protocol: reads .jsonl trace files and
    extracts token counts per model, LLM call latencies, and total runtime.
    """

    def analyze_trace(self, trace_path: str) -> TaskUsageStats:
        """
        Analyze a trace file and extract usage statistics.

        Implements the TraceAnalyzer protocol to extract:
        - Token counts per model
        - LLM call latencies
        - Total runtime
        - Model usage breakdown

        Args:
            trace_path: Path to .jsonl trace file

        Returns:
            TaskUsageStats with extracted metrics
        """
        path = Path(trace_path)
        if not path.exists():
            # Return empty stats for missing trace
            return TaskUsageStats(
                task_id=path.stem,
                total_runtime_seconds=0.0,
                models_used=[],
                total_llm_calls=0,
            )

        # Track stats per model
        model_stats: dict[str, ModelUsageStats] = {}
        trace_start_time: float | None = None
        trace_end_time: float | None = None
        total_llm_calls = 0

        with open(path) as f:
            for line in f:
                try:
                    span = json.loads(line.strip())

                    # Track overall trace timing
                    start_time = span.get("start_time_unix_nano", 0) / 1e9
                    end_time = span.get("end_time_unix_nano", 0) / 1e9

                    if trace_start_time is None or start_time < trace_start_time:
                        trace_start_time = start_time
                    if trace_end_time is None or end_time > trace_end_time:
                        trace_end_time = end_time

                    # Look for LLM spans
                    attrs = span.get("attributes", {})
                    span_name = span.get("name", "")

                    # Check if this is an LLM call span
                    if self._is_llm_span(span_name, attrs):
                        total_llm_calls += 1

                        # Extract model name
                        model_name = attrs.get(
                            "llm.model", attrs.get("gen_ai.request.model", "unknown")
                        )

                        # Initialize model stats if not seen before
                        if model_name not in model_stats:
                            model_stats[model_name] = ModelUsageStats(model_name=model_name)

                        stats = model_stats[model_name]

                        # Extract token counts with validation
                        prompt_tokens = attrs.get(
                            "llm.token_count.prompt", attrs.get("gen_ai.usage.input_tokens", 0)
                        )
                        completion_tokens = attrs.get(
                            "llm.token_count.completion", attrs.get("gen_ai.usage.output_tokens", 0)
                        )

                        # Validate and parse token counts
                        prompt_count = self._parse_token_count(prompt_tokens, "prompt_tokens", path)
                        completion_count = self._parse_token_count(
                            completion_tokens, "completion_tokens", path
                        )

                        stats.prompt_tokens += prompt_count
                        stats.completion_tokens += completion_count
                        stats.total_tokens += prompt_count + completion_count
                        stats.call_count += 1

                        # Calculate latency in milliseconds
                        if start_time > 0 and end_time > 0:
                            latency_ms = (end_time - start_time) * 1000
                            stats.latencies_ms.append(latency_ms)

                except json.JSONDecodeError:
                    continue

        # Calculate total runtime
        runtime_seconds = 0.0
        if trace_start_time and trace_end_time:
            runtime_seconds = trace_end_time - trace_start_time

        return TaskUsageStats(
            task_id=path.stem,
            total_runtime_seconds=runtime_seconds,
            models_used=list(model_stats.values()),
            total_llm_calls=total_llm_calls,
        )

    def _is_llm_span(self, span_name: str, attributes: dict[str, Any]) -> bool:
        """Check if a span represents an LLM API call."""
        # Check span name
        llm_span_names = ["llm", "chat", "completion", "generation", "inference"]
        if any(name in span_name.lower() for name in llm_span_names):
            return True

        # Check for LLM-related attributes
        llm_attrs = [
            "llm.model",
            "llm.token_count.prompt",
            "llm.token_count.completion",
            "gen_ai.request.model",
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.output_tokens",
        ]
        return any(attr in attributes for attr in llm_attrs)

    def _parse_token_count(self, value: Any, field_name: str, trace_path: Path) -> int:
        """Parse a token count value with validation.

        Args:
            value: The raw value from the trace (could be int, str, None, etc.)
            field_name: Name of the field for logging
            trace_path: Path to trace file for logging context

        Returns:
            Parsed integer token count, or 0 if invalid
        """
        if value is None:
            return 0

        if isinstance(value, int):
            return value

        if isinstance(value, float):
            return int(value)

        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                logger.warning(
                    f"Invalid {field_name} value '{value}' in trace {trace_path.name}, using 0"
                )
                return 0

        logger.warning(
            f"Unexpected type {type(value).__name__} for {field_name} in trace "
            f"{trace_path.name}, using 0"
        )
        return 0
