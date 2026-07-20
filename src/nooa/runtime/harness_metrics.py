# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Harness-level telemetry for OTLP spans.

Tracks all places where the harness silently "helps" the model:
fence removal, import stripping, response fixups, error recovery, etc.

Uses a ContextVar for per-task isolation (same pattern as _generation_id_stack_var).
Set in _execute_with_generation(), flushed to span in the finally block.

Usage from instrumented code::

    from nooa.runtime.harness_metrics import get_harness_metrics

    hm = get_harness_metrics()
    hm.fence_removal("```python")

``get_harness_metrics()`` never returns None -- outside a generation session
it returns a no-op singleton, so call sites never need an ``if`` guard.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, ClassVar, NamedTuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Limits ──────────────────────────────────────────────────────────
_MAX_LIST_ITEMS = 20
_MAX_STRING_CHARS = 500
_MAX_DIAG_CHARS = 1000  # _diagnose_death output is 400-700+ chars
_MAX_CODE_PREVIEW_CHARS = 200


def _truncate(s: str, limit: int = _MAX_STRING_CHARS) -> str:
    """Truncate a string to *limit* chars, appending '...' if truncated."""
    if len(s) <= limit:
        return s
    return s[:limit] + "..."


# ── Structured sub-models ───────────────────────────────────────────


class ErrorRecord(BaseModel):
    """A single correlated error record (exec error or validation error)."""

    error_type: str
    message: str
    turn: int = 0
    code_preview: str = ""


class TimingStat(BaseModel):
    """Tracks min/max/avg/total for a repeated timing measurement."""

    count: int = 0
    total_s: float = 0.0
    min_s: float = 0.0
    max_s: float = 0.0
    samples: list[float] = Field(default_factory=list)

    def record(self, elapsed: float) -> None:
        self.count += 1
        self.total_s += elapsed
        if self.count == 1 or elapsed < self.min_s:
            self.min_s = elapsed
        if elapsed > self.max_s:
            self.max_s = elapsed
        if len(self.samples) < _MAX_LIST_ITEMS:
            self.samples.append(round(elapsed, 6))

    @property
    def avg_s(self) -> float:
        return self.total_s / self.count if self.count else 0.0


# ── Span attribute schema entry ─────────────────────────────────────


class SchemaEntry(NamedTuple):
    """Describes one span attribute for both OTLP export and display."""

    key: str
    label: str
    category: str
    value_fn: Callable[[HarnessMetrics], Any]
    is_detail: bool = False


# ── HarnessMetrics ──────────────────────────────────────────────────


class HarnessMetrics(BaseModel):
    """Collects harness telemetry during a single generation session.

    All convenience methods are safe to call at any time -- they silently
    cap list sizes and truncate strings.  Call ``flush_to_span()`` once
    at the end to write everything as flat OTLP span attributes.
    """

    model_config: ClassVar[dict[str, Any]] = {"arbitrary_types_allowed": True}

    # ── Code Sanitization ──
    fence_removals: list[str] = Field(default_factory=list)
    xml_wrappers_stripped: list[str] = Field(default_factory=list)
    nested_wrapper_iterations: int = 0

    # ── Import Handling ──
    imports_stripped: list[str] = Field(default_factory=list)
    blocked_modules_removed: list[str] = Field(default_factory=list)

    # ── Response Format Fixups ──
    text_to_synthetic_count: int = 0
    stop_to_return_result_count: int = 0
    stop_to_return_result_previews: list[str] = Field(default_factory=list)
    text_only_loop_aborts_count: int = 0
    content_prepended_as_comment_count: int = 0
    empty_response_count: int = 0
    gpt4o_double_quote_fix_count: int = 0
    gpt4o_double_quote_fix_previews: list[str] = Field(default_factory=list)
    variable_refs_resolved: list[str] = Field(default_factory=list)
    constructor_strings_coerced: list[str] = Field(default_factory=list)
    json_auto_parse_methods: list[str] = Field(default_factory=list)
    args_normalized_count: int = 0

    # ── Tool Call Translation ──
    tool_calls_translated: list[str] = Field(default_factory=list)

    # ── Return Value Handling ──
    explicit_return_auto_completed: int = 0
    implicit_return_transformed: int = 0

    # ── Error Recovery ──
    validation_errors: list[ErrorRecord] = Field(default_factory=list)
    predict_retries: list[str] = Field(default_factory=list)
    block_syntax_errors: list[str] = Field(default_factory=list)
    llm_api_errors: list[str] = Field(default_factory=list)

    # ── Content/Reasoning (from unifiedllm via callback) ──
    think_tags_extracted: int = 0
    malformed_think_tag_fixed: int = 0
    content_to_reasoning_fallback: int = 0
    reasoning_as_structured_output: int = 0

    # ── JSON Cleanup (from unifiedllm via callback) ──
    json_fence_removed: int = 0
    json_control_chars_removed: int = 0
    json_escape_fixed: int = 0
    json_nested_extraction: int = 0
    json_double_decoded: int = 0

    # ── Code Execution ──
    exec_python_total: int = 0
    exec_python_success: int = 0
    exec_errors: list[ErrorRecord] = Field(default_factory=list)

    # ── Tool Usage (shell/repo) ──
    shell_variant: str = ""  # which self.shell implementation is active (bake-off tag)
    shell_failures: list[ErrorRecord] = Field(default_factory=list)
    shell_deaths: list[ErrorRecord] = Field(default_factory=list)
    repo_failures: list[ErrorRecord] = Field(default_factory=list)
    tool_avoidance: list[str] = Field(default_factory=list)

    # ── Code Validation ──
    missing_awaits_detected: list[str] = Field(default_factory=list)
    infinite_loops_detected: int = 0
    return_types_redefined: list[str] = Field(default_factory=list)

    # ── Prefill ──
    prefill_type: str = ""

    # ── Context Limits (triggered when context/event budgets are hit) ──
    context_limits_blocks_evicted: int = 0
    context_limits_events_collapsed: int = 0
    context_limits_tokens_archived: int = 0

    # ── Timing (TimingStat tracks count / total / min / max / samples) ──
    # Per-session (typically one sample)
    time_session_init: TimingStat = Field(default_factory=TimingStat)
    time_prefill: TimingStat = Field(default_factory=TimingStat)
    # Per-turn (accumulated across turns)
    time_prepare_context: TimingStat = Field(default_factory=TimingStat)
    time_render_context: TimingStat = Field(default_factory=TimingStat)
    time_llm_call: TimingStat = Field(default_factory=TimingStat)
    time_code_validation: TimingStat = Field(default_factory=TimingStat)
    time_code_execution: TimingStat = Field(default_factory=TimingStat)
    time_tracing_overhead: TimingStat = Field(default_factory=TimingStat)
    turn_count: int = 0

    # ── Helpers ──────────────────────────────────────────────────────

    def _append(self, lst: list[str], value: str, limit: int = _MAX_STRING_CHARS) -> None:
        if len(lst) < _MAX_LIST_ITEMS:
            lst.append(_truncate(value, limit))

    def _append_error(self, lst: list[ErrorRecord], rec: ErrorRecord) -> None:
        if len(lst) < _MAX_LIST_ITEMS:
            lst.append(rec)

    # ── Convenience record methods ──────────────────────────────────
    # Method names follow a consistent pattern: match the field name
    # in singular form so call sites read clearly and are greppable.
    #   hm.fence_removal("```python")  →  appends to fence_removals
    #   hm.think_tag_extracted()       →  increments think_tags_extracted

    # Code Sanitization
    def fence_removal(self, detail: str) -> None:
        self._append(self.fence_removals, detail)

    def xml_wrapper_stripped(self, tag_name: str) -> None:
        self._append(self.xml_wrappers_stripped, tag_name)

    def nested_wrapper_iteration(self, count: int) -> None:
        self.nested_wrapper_iterations = max(self.nested_wrapper_iterations, count)

    # Import Handling
    def import_stripped(self, detail: str) -> None:
        self._append(self.imports_stripped, detail)

    def blocked_module_removed(self, module_name: str) -> None:
        self._append(self.blocked_modules_removed, module_name)

    # Response Format Fixups
    def text_to_synthetic(self) -> None:
        self.text_to_synthetic_count += 1

    def stop_to_return_result(self, content: str | None) -> None:
        self.stop_to_return_result_count += 1
        if content:
            self._append(self.stop_to_return_result_previews, content, _MAX_CODE_PREVIEW_CHARS)

    def text_only_loop_abort(self) -> None:
        self.text_only_loop_aborts_count += 1

    def content_prepended_as_comment(self) -> None:
        self.content_prepended_as_comment_count += 1

    def empty_response(self) -> None:
        self.empty_response_count += 1

    def gpt4o_double_quote_fix(self, original_preview: str = "") -> None:
        self.gpt4o_double_quote_fix_count += 1
        if original_preview:
            self._append(
                self.gpt4o_double_quote_fix_previews, original_preview, _MAX_CODE_PREVIEW_CHARS
            )

    def variable_ref_resolved(self, var_name: str) -> None:
        self._append(self.variable_refs_resolved, var_name)

    def constructor_string_coerced(self, class_name: str) -> None:
        self._append(self.constructor_strings_coerced, class_name)

    def json_auto_parsed(self, method: str) -> None:
        self._append(self.json_auto_parse_methods, method)

    def args_normalized(self) -> None:
        self.args_normalized_count += 1

    # Tool Call Translation
    def tool_call_translated(self, tool_name: str) -> None:
        self._append(self.tool_calls_translated, tool_name)

    # Return Value Handling
    def explicit_return_completed(self) -> None:
        self.explicit_return_auto_completed += 1

    def implicit_return(self) -> None:
        self.implicit_return_transformed += 1

    # Error Recovery
    def validation_error(self, error_type: str, message: str) -> None:
        self._append_error(
            self.validation_errors,
            ErrorRecord(error_type=_truncate(error_type), message=_truncate(message)),
        )

    def predict_retry(self, error_detail: str) -> None:
        self._append(self.predict_retries, error_detail)

    def block_syntax_error(self, detail: str) -> None:
        self._append(self.block_syntax_errors, detail)

    def llm_api_error(self, detail: str) -> None:
        self._append(self.llm_api_errors, detail)

    # Content/Reasoning (dispatched from unifiedllm callback)
    # Note: methods use record_ prefix to avoid Pydantic field/method name collisions.
    def record_think_tag_extracted(self) -> None:
        self.think_tags_extracted += 1

    def record_malformed_think_tag(self) -> None:
        self.malformed_think_tag_fixed += 1

    def record_content_to_reasoning_fallback(self) -> None:
        self.content_to_reasoning_fallback += 1

    def record_reasoning_as_structured_output(self) -> None:
        self.reasoning_as_structured_output += 1

    # JSON Cleanup (dispatched from unifiedllm callback)
    def record_json_fence_removed(self) -> None:
        self.json_fence_removed += 1

    def record_json_control_chars_removed(self) -> None:
        self.json_control_chars_removed += 1

    def record_json_escape_fixed(self) -> None:
        self.json_escape_fixed += 1

    def record_json_nested_extraction(self) -> None:
        self.json_nested_extraction += 1

    def record_json_double_decoded(self) -> None:
        self.json_double_decoded += 1

    # Code Execution
    def exec_python(self, *, success: bool) -> None:
        self.exec_python_total += 1
        if success:
            self.exec_python_success += 1

    def exec_error(
        self, error_type: str, message: str, turn: int = 0, code_preview: str = ""
    ) -> None:
        self._append_error(
            self.exec_errors,
            ErrorRecord(
                error_type=_truncate(error_type),
                message=_truncate(message),
                turn=turn,
                code_preview=_truncate(code_preview, _MAX_CODE_PREVIEW_CHARS),
            ),
        )

    # Tool Usage (shell/repo)
    def set_shell_variant(self, variant: str) -> None:
        """Tag which shell implementation is active (for the shell bake-off)."""
        self.shell_variant = _truncate(variant)

    def shell_failure(self, tool_method: str, message: str, code_preview: str = "") -> None:
        """Record a shell tool failure (non-zero exit, timeout, file not found, etc.)."""
        self._append_error(
            self.shell_failures,
            ErrorRecord(
                error_type=_truncate(tool_method),
                message=_truncate(message),
                code_preview=_truncate(code_preview, _MAX_CODE_PREVIEW_CHARS),
            ),
        )

    def shell_death(self, context: str, diagnostics: str) -> None:
        """Record a bash process death event with diagnostics."""
        self._append_error(
            self.shell_deaths,
            ErrorRecord(
                error_type=_truncate(context),
                message=_truncate(diagnostics, _MAX_DIAG_CHARS),
            ),
        )

    def repo_failure(self, tool_method: str, message: str, code_preview: str = "") -> None:
        """Record a repo tool failure (file not found, no results, timeout, etc.)."""
        self._append_error(
            self.repo_failures,
            ErrorRecord(
                error_type=_truncate(tool_method),
                message=_truncate(message),
                code_preview=_truncate(code_preview, _MAX_CODE_PREVIEW_CHARS),
            ),
        )

    def tool_avoided(self, detail: str) -> None:
        """Record when the LLM bypasses a higher-level tool with a raw command.

        E.g. using shell.bash("sed ...") instead of shell.edit, or
        shell.bash("cat ...") instead of shell.view.
        """
        self._append(self.tool_avoidance, detail)

    # Code Validation
    def missing_await(self, method_name: str) -> None:
        self._append(self.missing_awaits_detected, method_name)

    def infinite_loop(self) -> None:
        self.infinite_loops_detected += 1

    def return_type_redefined(self, type_name: str) -> None:
        self._append(self.return_types_redefined, type_name)

    # Prefill
    def prefill(self, prefill_type: str) -> None:
        self.prefill_type = prefill_type

    # Timing
    @contextmanager
    def timer(self, field: str):
        """Record wall-clock time into a TimingStat field.

        Tracks count, total, min, max, and raw samples per call::

            with hm.timer("time_prepare_context"):
                await _prepare_context(...)
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            stat = getattr(self, field, None)
            if isinstance(stat, TimingStat):
                stat.record(elapsed)

    def tracing_overhead(self, elapsed_s: float) -> None:
        """Record wall-clock time spent inside installed tracing hook callbacks."""
        self.time_tracing_overhead.record(elapsed_s)

    def record_turn(self) -> None:
        self.turn_count += 1

    # ── OTLP export ─────────────────────────────────────────────────

    def to_span_attributes(self) -> dict[str, Any]:
        """Convert to flat OTLP span attribute dict. Only non-default values.

        Skips zeros, empty lists, empty strings, and False — all falsy in Python.
        """
        attrs: dict[str, Any] = {}
        for entry in _SPAN_SCHEMA:
            value = entry.value_fn(self)
            if value:
                attrs[entry.key] = value
        return attrs

    def flush_to_span(self) -> None:
        """Write all collected metrics as flat OTLP span attributes.

        Safe to call even when OpenTelemetry is not installed.
        """
        try:
            from opentelemetry import trace as otel_trace

            span = otel_trace.get_current_span()
            if not span or not span.is_recording():
                logger.debug("[HARNESS_METRICS] No active recording span, skipping flush")
                return

            attrs = self.to_span_attributes()
            for key, value in attrs.items():
                span.set_attribute(key, value)

            logger.debug("[HARNESS_METRICS] Flushed %d attributes to span", len(attrs))
        except ImportError:
            logger.debug("[HARNESS_METRICS] OpenTelemetry not available, skipping flush")
        except Exception as e:
            logger.debug("[HARNESS_METRICS] Failed to flush to span: %s", e)


# ── Span attribute schema (module-level constant) ───────────────────
# Single source of truth for OTLP attribute keys, labels, and categories.
# Used by both to_span_attributes() and trace_explorer display.


def _timing_schema_entries(
    prefix: str,
    label: str,
    category: str,
    stat_fn: Callable[[HarnessMetrics], TimingStat],
) -> list[SchemaEntry]:
    """Generate schema entries for a TimingStat field: total, min, max, avg, count, samples."""
    return [
        SchemaEntry(
            f"{prefix}.total_s",
            f"{label} (total)",
            category,
            lambda m, f=stat_fn: round(f(m).total_s, 4) if f(m).count else 0,
        ),
        SchemaEntry(
            f"{prefix}.min_s",
            f"{label} (min)",
            category,
            lambda m, f=stat_fn: round(f(m).min_s, 4) if f(m).count else 0,
        ),
        SchemaEntry(
            f"{prefix}.max_s",
            f"{label} (max)",
            category,
            lambda m, f=stat_fn: round(f(m).max_s, 4) if f(m).count else 0,
        ),
        SchemaEntry(
            f"{prefix}.avg_s",
            f"{label} (avg)",
            category,
            lambda m, f=stat_fn: round(f(m).avg_s, 4) if f(m).count else 0,
        ),
        SchemaEntry(
            f"{prefix}.count", f"{label} (count)", category, lambda m, f=stat_fn: f(m).count
        ),
        SchemaEntry(
            f"{prefix}.samples",
            f"{label} (samples)",
            category,
            lambda m, f=stat_fn: f(m).samples,
            True,
        ),
    ]


_SPAN_SCHEMA: tuple[SchemaEntry, ...] = (
    # Code Sanitization
    SchemaEntry(
        "harness.fence_removal.count",
        "Fence removals",
        "Code Sanitization",
        lambda m: len(m.fence_removals),
    ),
    SchemaEntry(
        "harness.fence_removal.details",
        "Fence removal details",
        "Code Sanitization",
        lambda m: m.fence_removals,
        True,
    ),
    SchemaEntry(
        "harness.xml_wrapper_stripped.count",
        "XML wrappers stripped",
        "Code Sanitization",
        lambda m: len(m.xml_wrappers_stripped),
    ),
    SchemaEntry(
        "harness.xml_wrapper_stripped.details",
        "XML wrapper details",
        "Code Sanitization",
        lambda m: m.xml_wrappers_stripped,
        True,
    ),
    SchemaEntry(
        "harness.nested_wrapper_iterations",
        "Nested wrapper iterations",
        "Code Sanitization",
        lambda m: m.nested_wrapper_iterations,
    ),
    # Import Handling
    SchemaEntry(
        "harness.imports_stripped.count",
        "Imports stripped",
        "Import Handling",
        lambda m: len(m.imports_stripped),
    ),
    SchemaEntry(
        "harness.imports_stripped.details",
        "Import details",
        "Import Handling",
        lambda m: m.imports_stripped,
        True,
    ),
    SchemaEntry(
        "harness.blocked_modules_removed.count",
        "Blocked modules removed",
        "Import Handling",
        lambda m: len(m.blocked_modules_removed),
    ),
    SchemaEntry(
        "harness.blocked_modules_removed.details",
        "Blocked module details",
        "Import Handling",
        lambda m: m.blocked_modules_removed,
        True,
    ),
    # Response Format Fixups
    SchemaEntry(
        "harness.text_to_synthetic.count",
        "Text-to-synthetic comment",
        "Response Format Fixups",
        lambda m: m.text_to_synthetic_count,
    ),
    SchemaEntry(
        "harness.stop_to_return_result.count",
        "Stop signal routed through return_result",
        "Response Format Fixups",
        lambda m: m.stop_to_return_result_count,
    ),
    SchemaEntry(
        "harness.text_only_loop_aborts.count",
        "Text-only loop aborts",
        "Response Format Fixups",
        lambda m: m.text_only_loop_aborts_count,
    ),
    SchemaEntry(
        "harness.content_prepended_as_comment.count",
        "Content prepended as comment",
        "Response Format Fixups",
        lambda m: m.content_prepended_as_comment_count,
    ),
    SchemaEntry(
        "harness.empty_response.count",
        "Empty responses",
        "Response Format Fixups",
        lambda m: m.empty_response_count,
    ),
    SchemaEntry(
        "harness.gpt4o_double_quote_fix.count",
        "GPT-4o double-quote fixes",
        "Response Format Fixups",
        lambda m: m.gpt4o_double_quote_fix_count,
    ),
    SchemaEntry(
        "harness.gpt4o_double_quote_fix.previews",
        "GPT-4o double-quote fix previews",
        "Response Format Fixups",
        lambda m: m.gpt4o_double_quote_fix_previews,
        True,
    ),
    SchemaEntry(
        "harness.variable_ref_resolved.count",
        "Variable refs resolved",
        "Response Format Fixups",
        lambda m: len(m.variable_refs_resolved),
    ),
    SchemaEntry(
        "harness.variable_ref_resolved.details",
        "Variable ref details",
        "Response Format Fixups",
        lambda m: m.variable_refs_resolved,
        True,
    ),
    SchemaEntry(
        "harness.constructor_string_coerced.count",
        "Constructor string coerced",
        "Response Format Fixups",
        lambda m: len(m.constructor_strings_coerced),
    ),
    SchemaEntry(
        "harness.constructor_string_coerced.details",
        "Constructor coercion details",
        "Response Format Fixups",
        lambda m: m.constructor_strings_coerced,
        True,
    ),
    SchemaEntry(
        "harness.json_auto_parsed.count",
        "JSON string auto-parsed",
        "Response Format Fixups",
        lambda m: len(m.json_auto_parse_methods),
    ),
    SchemaEntry(
        "harness.json_auto_parsed.methods",
        "JSON parse methods",
        "Response Format Fixups",
        lambda m: m.json_auto_parse_methods,
        True,
    ),
    SchemaEntry(
        "harness.args_normalized.count",
        "Args normalized",
        "Response Format Fixups",
        lambda m: m.args_normalized_count,
    ),
    # Tool Call Translation
    SchemaEntry(
        "harness.tool_call_translated.count",
        "Tool calls translated",
        "Tool Call Translation",
        lambda m: len(m.tool_calls_translated),
    ),
    SchemaEntry(
        "harness.tool_call_translated.details",
        "Translated tool details",
        "Tool Call Translation",
        lambda m: m.tool_calls_translated,
        True,
    ),
    # Return Value Handling
    SchemaEntry(
        "harness.explicit_return_auto_completed",
        "Explicit return auto-completed",
        "Return Value Handling",
        lambda m: m.explicit_return_auto_completed,
    ),
    SchemaEntry(
        "harness.implicit_return_transformed",
        "Implicit return transformed",
        "Return Value Handling",
        lambda m: m.implicit_return_transformed,
    ),
    # Error Recovery
    SchemaEntry(
        "harness.validation_error.count",
        "Validation errors",
        "Error Recovery",
        lambda m: len(m.validation_errors),
    ),
    SchemaEntry(
        "harness.validation_error.types",
        "Validation error types",
        "Error Recovery",
        lambda m: [e.error_type for e in m.validation_errors],
        True,
    ),
    SchemaEntry(
        "harness.predict_retry.count",
        "Predict retries",
        "Error Recovery",
        lambda m: len(m.predict_retries),
    ),
    SchemaEntry(
        "harness.predict_retry.errors",
        "Predict retry errors",
        "Error Recovery",
        lambda m: m.predict_retries,
        True,
    ),
    SchemaEntry(
        "harness.block_syntax_error.count",
        "Block syntax errors",
        "Error Recovery",
        lambda m: len(m.block_syntax_errors),
    ),
    SchemaEntry(
        "harness.block_syntax_error.details",
        "Block syntax error details",
        "Error Recovery",
        lambda m: m.block_syntax_errors,
        True,
    ),
    SchemaEntry(
        "harness.llm_api_error.count",
        "LLM API errors",
        "Error Recovery",
        lambda m: len(m.llm_api_errors),
    ),
    SchemaEntry(
        "harness.llm_api_error.details",
        "LLM API error details",
        "Error Recovery",
        lambda m: m.llm_api_errors,
        True,
    ),
    # Content/Reasoning
    SchemaEntry(
        "harness.think_tags_extracted",
        "Think tags extracted",
        "Content/Reasoning",
        lambda m: m.think_tags_extracted,
    ),
    SchemaEntry(
        "harness.malformed_think_tag_fixed",
        "Malformed think tag fixed",
        "Content/Reasoning",
        lambda m: m.malformed_think_tag_fixed,
    ),
    SchemaEntry(
        "harness.content_to_reasoning_fallback",
        "Content-to-reasoning fallback",
        "Content/Reasoning",
        lambda m: m.content_to_reasoning_fallback,
    ),
    SchemaEntry(
        "harness.reasoning_as_structured_output",
        "Reasoning as structured output",
        "Content/Reasoning",
        lambda m: m.reasoning_as_structured_output,
    ),
    # JSON Cleanup
    SchemaEntry(
        "harness.json_fence_removed",
        "JSON fence removed",
        "JSON Cleanup",
        lambda m: m.json_fence_removed,
    ),
    SchemaEntry(
        "harness.json_control_chars_removed",
        "JSON control chars removed",
        "JSON Cleanup",
        lambda m: m.json_control_chars_removed,
    ),
    SchemaEntry(
        "harness.json_escape_fixed",
        "JSON escape fixed",
        "JSON Cleanup",
        lambda m: m.json_escape_fixed,
    ),
    SchemaEntry(
        "harness.json_nested_extraction",
        "JSON nested extraction",
        "JSON Cleanup",
        lambda m: m.json_nested_extraction,
    ),
    SchemaEntry(
        "harness.json_double_decoded",
        "JSON double-decoded",
        "JSON Cleanup",
        lambda m: m.json_double_decoded,
    ),
    # Code Execution
    SchemaEntry(
        "harness.exec_python.total",
        "Total executions",
        "Execute Python",
        lambda m: m.exec_python_total,
    ),
    SchemaEntry(
        "harness.exec_python.success",
        "Successful",
        "Execute Python",
        lambda m: m.exec_python_success,
    ),
    SchemaEntry(
        "harness.exec_error.count", "Errors", "Execute Python", lambda m: len(m.exec_errors)
    ),
    SchemaEntry(
        "harness.exec_error.types",
        "Error types",
        "Execute Python",
        lambda m: [e.error_type for e in m.exec_errors],
        True,
    ),
    # Tool Usage (shell/repo)
    SchemaEntry(
        "harness.shell_variant",
        "Active shell implementation",
        "Tool Usage",
        lambda m: m.shell_variant,
    ),
    SchemaEntry(
        "harness.shell_failure.count",
        "Shell failures",
        "Tool Usage",
        lambda m: len(m.shell_failures),
    ),
    SchemaEntry(
        "harness.shell_failure.methods",
        "Shell failure methods",
        "Tool Usage",
        lambda m: [e.error_type for e in m.shell_failures],
        True,
    ),
    SchemaEntry(
        "harness.shell_failure.messages",
        "Shell failure messages",
        "Tool Usage",
        lambda m: [e.message for e in m.shell_failures],
        True,
    ),
    SchemaEntry(
        "harness.shell_death.count",
        "Shell process deaths",
        "Tool Usage",
        lambda m: len(m.shell_deaths),
    ),
    SchemaEntry(
        "harness.shell_death.contexts",
        "Shell death contexts",
        "Tool Usage",
        lambda m: [e.error_type for e in m.shell_deaths],
        True,
    ),
    SchemaEntry(
        "harness.shell_death.diagnostics",
        "Shell death diagnostics",
        "Tool Usage",
        lambda m: [e.message for e in m.shell_deaths],
        True,
    ),
    SchemaEntry(
        "harness.repo_failure.count",
        "Repo failures",
        "Tool Usage",
        lambda m: len(m.repo_failures),
    ),
    SchemaEntry(
        "harness.repo_failure.methods",
        "Repo failure methods",
        "Tool Usage",
        lambda m: [e.error_type for e in m.repo_failures],
        True,
    ),
    SchemaEntry(
        "harness.repo_failure.messages",
        "Repo failure messages",
        "Tool Usage",
        lambda m: [e.message for e in m.repo_failures],
        True,
    ),
    SchemaEntry(
        "harness.tool_avoidance.count",
        "Tool avoidance",
        "Tool Usage",
        lambda m: len(m.tool_avoidance),
    ),
    SchemaEntry(
        "harness.tool_avoidance.details",
        "Tool avoidance details",
        "Tool Usage",
        lambda m: m.tool_avoidance,
        True,
    ),
    # Code Validation
    SchemaEntry(
        "harness.missing_await.count",
        "Missing awaits detected",
        "Code Validation",
        lambda m: len(m.missing_awaits_detected),
    ),
    SchemaEntry(
        "harness.missing_await.details",
        "Missing await details",
        "Code Validation",
        lambda m: m.missing_awaits_detected,
        True,
    ),
    SchemaEntry(
        "harness.infinite_loops_detected",
        "Infinite loops detected",
        "Code Validation",
        lambda m: m.infinite_loops_detected,
    ),
    # Prefill
    SchemaEntry("harness.prefill_type", "Prefill type", "Prefill", lambda m: m.prefill_type),
    # Timing — each timer emits total/min/max/avg/count + raw samples
    *_timing_schema_entries(
        "harness.time.session_init", "Session init", "Timing", lambda m: m.time_session_init
    ),
    *_timing_schema_entries("harness.time.prefill", "Prefill", "Timing", lambda m: m.time_prefill),
    *_timing_schema_entries(
        "harness.time.prepare_context",
        "Prepare context",
        "Timing",
        lambda m: m.time_prepare_context,
    ),
    *_timing_schema_entries(
        "harness.time.render_context", "Render context", "Timing", lambda m: m.time_render_context
    ),
    *_timing_schema_entries(
        "harness.time.llm_call", "LLM call", "Timing", lambda m: m.time_llm_call
    ),
    *_timing_schema_entries(
        "harness.time.code_validation",
        "Code validation",
        "Timing",
        lambda m: m.time_code_validation,
    ),
    *_timing_schema_entries(
        "harness.time.code_execution", "Code execution", "Timing", lambda m: m.time_code_execution
    ),
    *_timing_schema_entries(
        "harness.time.tracing_overhead",
        "Tracing overhead",
        "Timing",
        lambda m: m.time_tracing_overhead,
    ),
    SchemaEntry("harness.turn_count", "Turns", "Timing", lambda m: m.turn_count),
    # Context Limits (triggered when budgets are hit)
    SchemaEntry(
        "harness.context_limits.blocks_evicted",
        "Context blocks evicted",
        "L4 Eviction",
        lambda m: m.context_limits_blocks_evicted,
    ),
    SchemaEntry(
        "harness.context_limits.events_collapsed",
        "Events collapsed",
        "L4 Eviction",
        lambda m: m.context_limits_events_collapsed,
    ),
    SchemaEntry(
        "harness.context_limits.tokens_archived",
        "Tokens archived",
        "L4 Eviction",
        lambda m: m.context_limits_tokens_archived,
    ),
)


def get_span_schema() -> tuple[SchemaEntry, ...]:
    """Public accessor for the span attribute schema.

    Used by trace_explorer to drive display without importing internals.
    """
    return _SPAN_SCHEMA


# ── Null object (no-op outside generation sessions) ─────────────────


def _noop(*_args: Any, **_kwargs: Any) -> None:
    pass


@contextmanager
def _noop_timer(*_args: Any, **_kwargs: Any):
    yield


class _NullMetrics:
    """No-op stand-in returned by ``get_harness_metrics()`` outside a generation session.

    Every attribute access returns a no-op callable, so call sites never
    need an ``if`` guard.  ``timer()`` returns a no-op context manager.
    Trade-off: typos in method names (e.g. ``hm.fense_removal()``) are
    silently ignored outside generation sessions but will raise
    ``AttributeError`` inside one.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        if name == "timer":
            return _noop_timer
        return _noop


_NULL_METRICS = _NullMetrics()


# ── ContextVar lifecycle ────────────────────────────────────────────

_harness_metrics_var: ContextVar[HarnessMetrics | _NullMetrics] = ContextVar(
    "harness_metrics", default=_NULL_METRICS
)


def get_harness_metrics() -> HarnessMetrics:
    """Get the HarnessMetrics for the current generation session.

    Never returns None. Outside a generation session, returns a no-op
    singleton whose methods silently discard all calls.  The return type
    is annotated as ``HarnessMetrics`` because callers should never need
    to distinguish — the null object is an implementation detail.
    """
    return _harness_metrics_var.get()  # type: ignore[return-value]


def start_harness_metrics() -> tuple[HarnessMetrics, HarnessMetrics | _NullMetrics]:
    """Create and activate a new HarnessMetrics for the current context.

    Returns ``(new_metrics, previous)`` so the caller can restore the
    parent's metrics after flushing (handles nested generation sessions).
    """
    prev = _harness_metrics_var.get()
    metrics = HarnessMetrics()
    _harness_metrics_var.set(metrics)
    return metrics, prev


def restore_harness_metrics(prev: HarnessMetrics | _NullMetrics) -> None:
    """Restore the previous HarnessMetrics (called in finally blocks)."""
    _harness_metrics_var.set(prev)


@contextmanager
def harness_metrics_session():
    """Context manager for a harness metrics generation session.

    Usage::

        with harness_metrics_session() as hm:
            ...  # hm is a fresh HarnessMetrics
        # on exit: flushes to span, restores previous metrics
    """
    metrics, prev = start_harness_metrics()
    try:
        yield metrics
    finally:
        try:
            metrics.flush_to_span()
        finally:
            restore_harness_metrics(prev)
