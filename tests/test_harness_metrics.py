# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for HarnessMetrics telemetry collection."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from nooa.runtime.harness_metrics import (
    _MAX_CODE_PREVIEW_CHARS,
    _MAX_LIST_ITEMS,
    _MAX_STRING_CHARS,
    _SPAN_SCHEMA,
    ErrorRecord,
    HarnessMetrics,
    SchemaEntry,
    TimingStat,
    _NullMetrics,
    _truncate,
    get_harness_metrics,
    get_span_schema,
    harness_metrics_session,
    restore_harness_metrics,
    start_harness_metrics,
)


@pytest.fixture(autouse=True)
def _clean_harness_metrics():
    """Ensure harness metrics ContextVar is clean before and after each test."""
    from nooa.runtime.harness_metrics import _NULL_METRICS, _harness_metrics_var

    token = _harness_metrics_var.set(_NULL_METRICS)
    yield
    _harness_metrics_var.reset(token)


# ── Recording tests ─────────────────────────────────────────────────


class TestRecording:
    """Test all convenience record methods individually."""

    # Code Sanitization
    def test_fence_removal(self):
        m = HarnessMetrics()
        m.fence_removal("```python")
        assert m.fence_removals == ["```python"]

    def test_xml_wrapper_stripped(self):
        m = HarnessMetrics()
        m.xml_wrapper_stripped("tool_code")
        assert m.xml_wrappers_stripped == ["tool_code"]

    def test_nested_wrapper_iteration(self):
        m = HarnessMetrics()
        m.nested_wrapper_iteration(3)
        m.nested_wrapper_iteration(2)  # should keep max
        assert m.nested_wrapper_iterations == 3

    # Import Handling
    def test_import_stripped(self):
        m = HarnessMetrics()
        m.import_stripped("from typing import Literal")
        m.import_stripped("import pandas as pd")
        assert len(m.imports_stripped) == 2

    def test_blocked_module_removed(self):
        m = HarnessMetrics()
        m.blocked_module_removed("os")
        m.blocked_module_removed("subprocess")
        assert m.blocked_modules_removed == ["os", "subprocess"]

    # Response Format Fixups
    def test_text_to_synthetic(self):
        m = HarnessMetrics()
        m.text_to_synthetic()
        m.text_to_synthetic()
        assert m.text_to_synthetic_count == 2

    def test_content_prepended_as_comment(self):
        m = HarnessMetrics()
        m.content_prepended_as_comment()
        assert m.content_prepended_as_comment_count == 1

    def test_empty_response(self):
        m = HarnessMetrics()
        m.empty_response()
        m.empty_response()
        assert m.empty_response_count == 2

    def test_gpt4o_double_quote_fix(self):
        m = HarnessMetrics()
        m.gpt4o_double_quote_fix()
        assert m.gpt4o_double_quote_fix_count == 1
        assert m.gpt4o_double_quote_fix_previews == []

    def test_gpt4o_double_quote_fix_with_preview(self):
        m = HarnessMetrics()
        m.gpt4o_double_quote_fix('"x = 1\\n"')
        assert m.gpt4o_double_quote_fix_count == 1
        assert m.gpt4o_double_quote_fix_previews == ['"x = 1\\n"']

    def test_variable_ref_resolved(self):
        m = HarnessMetrics()
        m.variable_ref_resolved("results")
        assert m.variable_refs_resolved == ["results"]

    def test_json_auto_parsed(self):
        m = HarnessMetrics()
        m.json_auto_parsed("json")
        m.json_auto_parsed("literal_eval")
        assert m.json_auto_parse_methods == ["json", "literal_eval"]

    def test_args_normalized(self):
        m = HarnessMetrics()
        m.args_normalized()
        assert m.args_normalized_count == 1

    # Tool Call Translation
    def test_tool_call_translated(self):
        m = HarnessMetrics()
        m.tool_call_translated("self.bash.run")
        assert m.tool_calls_translated == ["self.bash.run"]

    # Return Value Handling
    def test_explicit_return_completed(self):
        m = HarnessMetrics()
        m.explicit_return_completed()
        assert m.explicit_return_auto_completed == 1

    def test_implicit_return(self):
        m = HarnessMetrics()
        m.implicit_return()
        assert m.implicit_return_transformed == 1

    # Error Recovery
    def test_validation_error(self):
        m = HarnessMetrics()
        m.validation_error("PydanticValidationError", "field required")
        assert len(m.validation_errors) == 1
        assert m.validation_errors[0].error_type == "PydanticValidationError"
        assert m.validation_errors[0].message == "field required"

    def test_predict_retry(self):
        m = HarnessMetrics()
        m.predict_retry("ValueError: bad")
        assert m.predict_retries == ["ValueError: bad"]

    def test_block_syntax_error(self):
        m = HarnessMetrics()
        m.block_syntax_error("block 'plan': SyntaxError")
        assert m.block_syntax_errors == ["block 'plan': SyntaxError"]

    def test_llm_api_error(self):
        m = HarnessMetrics()
        m.llm_api_error("RateLimitError: 429")
        assert m.llm_api_errors == ["RateLimitError: 429"]

    # Content/Reasoning
    def test_think_tag_extracted(self):
        m = HarnessMetrics()
        m.record_think_tag_extracted()
        assert m.think_tags_extracted == 1

    def test_malformed_think_tag_fixed(self):
        m = HarnessMetrics()
        m.record_malformed_think_tag()
        assert m.malformed_think_tag_fixed == 1

    def test_content_to_reasoning_fallback(self):
        m = HarnessMetrics()
        m.record_content_to_reasoning_fallback()
        assert m.content_to_reasoning_fallback == 1

    def test_reasoning_as_structured_output(self):
        m = HarnessMetrics()
        m.record_reasoning_as_structured_output()
        assert m.reasoning_as_structured_output == 1

    # JSON Cleanup
    def test_json_counters(self):
        m = HarnessMetrics()
        m.record_json_fence_removed()
        m.record_json_control_chars_removed()
        m.record_json_escape_fixed()
        m.record_json_nested_extraction()
        m.record_json_double_decoded()
        assert m.json_fence_removed == 1
        assert m.json_control_chars_removed == 1
        assert m.json_escape_fixed == 1
        assert m.json_nested_extraction == 1
        assert m.json_double_decoded == 1

    # Code Execution
    def test_exec_python(self):
        m = HarnessMetrics()
        m.exec_python(success=True)
        m.exec_python(success=False)
        m.exec_python(success=True)
        assert m.exec_python_total == 3
        assert m.exec_python_success == 2

    def test_exec_error(self):
        m = HarnessMetrics()
        m.exec_error("NameError", "name 'df' is not defined", 3, "df.head()")
        assert len(m.exec_errors) == 1
        assert m.exec_errors[0].error_type == "NameError"
        assert m.exec_errors[0].message == "name 'df' is not defined"
        assert m.exec_errors[0].turn == 3
        assert m.exec_errors[0].code_preview == "df.head()"

    # Code Validation
    def test_missing_await(self):
        m = HarnessMetrics()
        m.missing_await("analyze")
        assert m.missing_awaits_detected == ["analyze"]

    def test_infinite_loop(self):
        m = HarnessMetrics()
        m.infinite_loop()
        assert m.infinite_loops_detected == 1

    # Prefill
    def test_prefill(self):
        m = HarnessMetrics()
        m.prefill("inspect_inputs")
        assert m.prefill_type == "inspect_inputs"

    # Edge case: empty strings
    def test_record_methods_accept_empty_strings(self):
        m = HarnessMetrics()
        m.fence_removal("")
        m.exec_error("", "", 0, "")
        assert m.fence_removals == [""]
        assert m.exec_errors[0].error_type == ""


# ── Capping tests ───────────────────────────────────────────────────


class TestListCapping:
    """Test that detail lists are capped at _MAX_LIST_ITEMS."""

    def test_list_capped_at_max(self):
        m = HarnessMetrics()
        for i in range(_MAX_LIST_ITEMS + 10):
            m.fence_removal(f"```python_{i}")
        assert len(m.fence_removals) == _MAX_LIST_ITEMS

    def test_error_record_list_capped(self):
        m = HarnessMetrics()
        for i in range(_MAX_LIST_ITEMS + 5):
            m.exec_error("Error", "msg", i, "code")
        assert len(m.exec_errors) == _MAX_LIST_ITEMS

    def test_validation_error_list_capped_and_correlated(self):
        m = HarnessMetrics()
        for i in range(_MAX_LIST_ITEMS + 5):
            m.validation_error("TypeError", f"error {i}")
        assert len(m.validation_errors) == _MAX_LIST_ITEMS
        for rec in m.validation_errors:
            assert rec.error_type == "TypeError"
            assert rec.message.startswith("error ")


# ── String truncation tests ─────────────────────────────────────────


class TestStringTruncation:
    """Test that strings are truncated correctly."""

    def test_long_string_truncated(self):
        m = HarnessMetrics()
        long_msg = "x" * (_MAX_STRING_CHARS + 100)
        m.import_stripped(long_msg)
        assert len(m.imports_stripped[0]) == _MAX_STRING_CHARS + 3  # +3 for "..."
        assert m.imports_stripped[0].endswith("...")

    def test_short_string_not_truncated(self):
        m = HarnessMetrics()
        m.import_stripped("short")
        assert m.imports_stripped[0] == "short"

    def test_exec_error_code_preview_truncated_at_200_chars(self):
        m = HarnessMetrics()
        long_code = "x" * (_MAX_CODE_PREVIEW_CHARS + 50)
        m.exec_error("E", "m", 1, long_code)
        assert len(m.exec_errors[0].code_preview) == _MAX_CODE_PREVIEW_CHARS + 3
        assert m.exec_errors[0].code_preview.endswith("...")
        # error_type and message use the 500-char limit
        long_msg = "y" * (_MAX_STRING_CHARS + 50)
        m.exec_error(long_msg, long_msg, 2, "short")
        assert len(m.exec_errors[1].error_type) == _MAX_STRING_CHARS + 3
        assert m.exec_errors[1].code_preview == "short"

    def test_validation_error_truncates_long_fields(self):
        m = HarnessMetrics()
        long_val = "z" * (_MAX_STRING_CHARS + 50)
        m.validation_error(long_val, long_val)
        assert len(m.validation_errors[0].error_type) == _MAX_STRING_CHARS + 3
        assert len(m.validation_errors[0].message) == _MAX_STRING_CHARS + 3

    def test_truncate_helper(self):
        assert _truncate("hello", 3) == "hel..."
        assert _truncate("hi", 5) == "hi"
        assert _truncate("", 0) == ""
        assert _truncate("a", 0) == "..."


# ── Span attributes tests ──────────────────────────────────────────


class TestSpanAttributes:
    """Test to_span_attributes produces correct flat OTLP attributes."""

    def test_empty_metrics_no_attributes(self):
        m = HarnessMetrics()
        attrs = m.to_span_attributes()
        assert "harness.fence_removal.count" not in attrs
        assert "harness.exec_error.count" not in attrs

    def test_populated_metrics_correct_attributes(self):
        m = HarnessMetrics()
        m.fence_removal("```python")
        m.fence_removal("```")
        m.import_stripped("from typing import Literal")
        m.empty_response()
        m.exec_python(success=True)
        m.exec_python(success=False)
        m.exec_error("NameError", "name not defined", 3, "x = y")

        attrs = m.to_span_attributes()

        assert attrs["harness.fence_removal.count"] == 2
        assert attrs["harness.fence_removal.details"] == ["```python", "```"]
        assert attrs["harness.imports_stripped.count"] == 1
        assert attrs["harness.imports_stripped.details"] == ["from typing import Literal"]
        assert attrs["harness.empty_response.count"] == 1
        assert attrs["harness.exec_python.total"] == 2
        assert attrs["harness.exec_python.success"] == 1
        assert attrs["harness.exec_error.count"] == 1
        assert attrs["harness.exec_error.types"] == ["NameError"]

    def test_zero_counts_not_in_attributes(self):
        m = HarnessMetrics()
        m.fence_removal("```python")
        attrs = m.to_span_attributes()
        assert "harness.imports_stripped.count" not in attrs
        assert "harness.exec_error.count" not in attrs

    def test_validation_error_types_in_attributes(self):
        m = HarnessMetrics()
        m.validation_error("ValueError", "bad value")
        m.validation_error("TypeError", "type mismatch")
        attrs = m.to_span_attributes()
        assert attrs["harness.validation_error.count"] == 2
        assert attrs["harness.validation_error.types"] == ["ValueError", "TypeError"]


# ── Schema tests ────────────────────────────────────────────────────


def _populate_all_fields(m: HarnessMetrics) -> None:
    """Populate every field of a HarnessMetrics instance."""
    m.fence_removal("```")
    m.xml_wrapper_stripped("tag")
    m.nested_wrapper_iteration(2)
    m.import_stripped("import x")
    m.blocked_module_removed("os")
    m.text_to_synthetic()
    m.content_prepended_as_comment()
    m.empty_response()
    m.gpt4o_double_quote_fix('"x\\n"')
    m.variable_ref_resolved("x")
    m.json_auto_parsed("json")
    m.args_normalized()
    m.tool_call_translated("tool")
    m.explicit_return_completed()
    m.implicit_return()
    m.validation_error("E", "m")
    m.predict_retry("err")
    m.block_syntax_error("detail")
    m.llm_api_error("err")
    m.record_think_tag_extracted()
    m.record_malformed_think_tag()
    m.record_content_to_reasoning_fallback()
    m.record_reasoning_as_structured_output()
    m.record_json_fence_removed()
    m.record_json_control_chars_removed()
    m.record_json_escape_fixed()
    m.record_json_nested_extraction()
    m.record_json_double_decoded()
    m.exec_python(success=True)
    m.exec_error("E", "m", 1, "c")
    m.missing_await("method")
    m.infinite_loop()
    m.prefill("inspect_inputs")
    # Timing fields
    m.time_session_init.record(0.1)
    m.time_prefill.record(0.05)
    m.time_prepare_context.record(0.2)
    m.time_render_context.record(0.03)
    m.time_llm_call.record(1.5)
    m.time_code_validation.record(0.01)
    m.time_code_execution.record(0.4)
    m.tracing_overhead(0.02)
    m.record_turn()


class TestSchema:
    """Test span attribute schema is well-formed and complete."""

    def test_schema_is_tuple_of_named_tuples(self):
        assert isinstance(_SPAN_SCHEMA, tuple)
        for entry in _SPAN_SCHEMA:
            assert isinstance(entry, SchemaEntry)

    def test_schema_entries_have_required_fields(self):
        for entry in _SPAN_SCHEMA:
            assert entry.key.startswith("harness.")
            assert entry.label
            assert entry.category
            assert callable(entry.value_fn)

    def test_get_span_schema_returns_same_object(self):
        assert get_span_schema() is _SPAN_SCHEMA

    def test_every_to_span_attributes_key_is_in_schema(self):
        """Forward direction: to_span_attributes() keys ⊆ schema keys."""
        m = HarnessMetrics()
        _populate_all_fields(m)
        attrs = m.to_span_attributes()
        schema_keys = {entry.key for entry in _SPAN_SCHEMA}
        for key in attrs:
            assert key in schema_keys, f"Key {key!r} from to_span_attributes() not in schema"

    def test_every_non_detail_schema_entry_appears_when_populated(self):
        """Inverse direction: when all fields populated, every non-detail schema entry with a
        truthy value should appear in to_span_attributes()."""
        m = HarnessMetrics()
        _populate_all_fields(m)
        attrs = m.to_span_attributes()
        for entry in _SPAN_SCHEMA:
            value = entry.value_fn(m)
            if value:  # should be truthy since we populated everything
                assert entry.key in attrs, (
                    f"Schema entry {entry.key!r} has value {value!r} but missing from to_span_attributes()"
                )


# ── ContextVar lifecycle tests ──────────────────────────────────────


class TestContextVarLifecycle:
    """Test ContextVar-based lifecycle."""

    def test_default_is_null_metrics(self):
        result = get_harness_metrics()
        assert isinstance(result, _NullMetrics)

    def test_null_metrics_is_noop(self):
        """Calling methods on NullMetrics should not raise."""
        null = _NullMetrics()
        null.fence_removal("```python")
        null.exec_python(success=True)
        null.validation_error("E", "m")

    def test_null_metrics_is_not_harness_metrics_instance(self):
        assert not isinstance(_NullMetrics(), HarnessMetrics)

    def test_null_metrics_attribute_returns_noop(self):
        null = _NullMetrics()
        result = null.fence_removal("test")
        assert result is None

    def test_start_creates_instance(self):
        metrics, prev = start_harness_metrics()
        try:
            assert isinstance(get_harness_metrics(), HarnessMetrics)
            assert get_harness_metrics() is metrics
        finally:
            restore_harness_metrics(prev)

    def test_restore_resets_to_previous(self):
        _metrics, prev = start_harness_metrics()
        restore_harness_metrics(prev)
        assert isinstance(get_harness_metrics(), _NullMetrics)

    def test_nested_sessions_preserve_parent(self):
        m1, prev0 = start_harness_metrics()
        m1.fence_removal("parent")
        m2, prev1 = start_harness_metrics()
        m2.fence_removal("child")
        assert get_harness_metrics() is m2
        assert prev1 is m1
        restore_harness_metrics(prev1)
        assert get_harness_metrics() is m1
        assert m1.fence_removals == ["parent"]
        restore_harness_metrics(prev0)

    def test_triple_nested_sessions(self):
        m1, prev0 = start_harness_metrics()
        m1.fence_removal("L1")
        m2, prev1 = start_harness_metrics()
        m2.fence_removal("L2")
        m3, prev2 = start_harness_metrics()
        m3.fence_removal("L3")

        assert get_harness_metrics() is m3
        restore_harness_metrics(prev2)
        assert get_harness_metrics() is m2
        assert m2.fence_removals == ["L2"]
        restore_harness_metrics(prev1)
        assert get_harness_metrics() is m1
        assert m1.fence_removals == ["L1"]
        restore_harness_metrics(prev0)
        assert isinstance(get_harness_metrics(), _NullMetrics)


# ── Context manager tests ──────────────────────────────────────────


class TestContextManager:
    """Test harness_metrics_session context manager."""

    def test_context_manager_creates_and_restores(self):
        assert isinstance(get_harness_metrics(), _NullMetrics)
        with harness_metrics_session() as hm:
            assert isinstance(hm, HarnessMetrics)
            assert get_harness_metrics() is hm
            hm.fence_removal("test")
        assert isinstance(get_harness_metrics(), _NullMetrics)

    def test_context_manager_restores_on_exception(self):
        assert isinstance(get_harness_metrics(), _NullMetrics)
        with pytest.raises(ValueError, match="boom"):
            with harness_metrics_session():
                raise ValueError("boom")
        assert isinstance(get_harness_metrics(), _NullMetrics)

    def test_nested_context_managers(self):
        with harness_metrics_session() as hm1:
            hm1.fence_removal("outer")
            with harness_metrics_session() as hm2:
                hm2.fence_removal("inner")
                assert get_harness_metrics() is hm2
            assert get_harness_metrics() is hm1
            assert hm1.fence_removals == ["outer"]

    def test_restores_even_when_flush_raises(self):
        """restore_harness_metrics must run even if flush_to_span raises."""
        from unittest.mock import patch

        assert isinstance(get_harness_metrics(), _NullMetrics)
        with patch.object(HarnessMetrics, "flush_to_span", side_effect=RuntimeError("boom")):
            with pytest.raises(RuntimeError, match="boom"):
                with harness_metrics_session():
                    pass  # exit triggers flush (which raises), but restore still runs
        # restore_harness_metrics was called despite the exception
        assert isinstance(get_harness_metrics(), _NullMetrics)


# ── flush_to_span tests ────────────────────────────────────────────


class TestFlushToSpan:
    """Test flush_to_span with and without OpenTelemetry."""

    def test_flush_without_opentelemetry_does_not_raise(self):
        m = HarnessMetrics()
        m.fence_removal("```python")
        m.flush_to_span()

    def test_flush_is_idempotent(self):
        """Calling flush_to_span twice should not crash or corrupt state."""
        m = HarnessMetrics()
        m.fence_removal("```python")
        m.flush_to_span()
        m.flush_to_span()
        assert m.fence_removals == ["```python"]

    def test_flush_sets_attributes_on_span(self):
        m = HarnessMetrics()
        m.fence_removal("```python")
        m.empty_response()
        m.exec_python(success=True)

        mock_span = MagicMock()
        mock_span.is_recording.return_value = True

        mock_trace = MagicMock()
        mock_trace.get_current_span.return_value = mock_span

        mock_otel = MagicMock()
        mock_otel.trace = mock_trace

        saved = {}
        for key in list(sys.modules.keys()):
            if key == "opentelemetry" or key.startswith("opentelemetry."):
                saved[key] = sys.modules.pop(key)

        sys.modules["opentelemetry"] = mock_otel
        sys.modules["opentelemetry.trace"] = mock_trace
        try:
            m.flush_to_span()
        finally:
            for key in ["opentelemetry", "opentelemetry.trace"]:
                sys.modules.pop(key, None)
            sys.modules.update(saved)

        call_args = {call[0][0]: call[0][1] for call in mock_span.set_attribute.call_args_list}
        assert call_args["harness.fence_removal.count"] == 1
        assert call_args["harness.fence_removal.details"] == ["```python"]
        assert call_args["harness.empty_response.count"] == 1
        assert call_args["harness.exec_python.total"] == 1
        assert call_args["harness.exec_python.success"] == 1


# ── ErrorRecord tests ──────────────────────────────────────────────


class TestErrorRecord:
    """Test ErrorRecord model."""

    def test_fields(self):
        rec = ErrorRecord(error_type="ValueError", message="bad")
        assert rec.error_type == "ValueError"
        assert rec.message == "bad"
        assert rec.turn == 0
        assert rec.code_preview == ""

    def test_with_all_fields(self):
        rec = ErrorRecord(
            error_type="NameError", message="x not defined", turn=5, code_preview="print(x)"
        )
        assert rec.turn == 5
        assert rec.code_preview == "print(x)"

    def test_model_dump(self):
        rec = ErrorRecord(error_type="E", message="m", turn=3, code_preview="c")
        d = rec.model_dump()
        assert d == {"error_type": "E", "message": "m", "turn": 3, "code_preview": "c"}


# ── Pydantic serialization tests ───────────────────────────────────


class TestPydanticSerialization:
    """Test model_dump / model_validate round-trip."""

    def test_empty_round_trip(self):
        m = HarnessMetrics()
        d = m.model_dump()
        m2 = HarnessMetrics.model_validate(d)
        assert m2.fence_removals == []
        assert m2.exec_python_total == 0

    def test_populated_round_trip(self):
        m = HarnessMetrics()
        _populate_all_fields(m)
        d = m.model_dump()
        m2 = HarnessMetrics.model_validate(d)
        assert m2.fence_removals == m.fence_removals
        assert m2.think_tags_extracted == m.think_tags_extracted
        assert len(m2.validation_errors) == len(m.validation_errors)
        assert m2.validation_errors[0].error_type == m.validation_errors[0].error_type
        assert m2.prefill_type == "inspect_inputs"

    def test_json_round_trip(self):
        """Full JSON serialization round-trip (catches float('inf') and similar issues)."""
        m = HarnessMetrics()
        _populate_all_fields(m)
        json_str = m.model_dump_json()
        m2 = HarnessMetrics.model_validate_json(json_str)
        assert m2.fence_removals == m.fence_removals
        assert m2.time_llm_call.count == m.time_llm_call.count
        assert m2.time_llm_call.min_s == pytest.approx(m.time_llm_call.min_s)

    def test_empty_json_round_trip(self):
        """Empty HarnessMetrics must survive JSON round-trip (all TimingStats at defaults)."""
        m = HarnessMetrics()
        json_str = m.model_dump_json()
        m2 = HarnessMetrics.model_validate_json(json_str)
        assert m2.turn_count == 0
        assert m2.time_llm_call.count == 0


# ── UnifiedLLM callback protocol tests ─────────────────────────────


class TestLLMMetricsCallback:
    """Test the ContextVar callback protocol used by unifiedllm."""

    def test_record_llm_metric_noop_when_no_callback(self):
        from nooa.unifiedllm.unifiedllm import _record_llm_metric

        _record_llm_metric("think_tag_extracted")

    def test_record_llm_metric_dispatches_to_callback(self):
        from nooa.unifiedllm.unifiedllm import _llm_metrics_callback, _record_llm_metric

        events: list[tuple[str, Any]] = []

        def capture(event: str, detail: Any = None) -> None:
            events.append((event, detail))

        token = _llm_metrics_callback.set(capture)
        try:
            _record_llm_metric("think_tag_extracted")
            _record_llm_metric("json_fence_removed")
            assert len(events) == 2
            assert events[0][0] == "think_tag_extracted"
            assert events[1][0] == "json_fence_removed"
        finally:
            _llm_metrics_callback.reset(token)


# ── Bridge tests ───────────────────────────────────────────────────


class TestLLMMetricsBridge:
    """Test the _make_llm_metrics_bridge function in actor.py."""

    def test_bridge_dispatches_all_events(self):
        from nooa.runtime.actor import _make_llm_metrics_bridge

        hm = HarnessMetrics()
        bridge = _make_llm_metrics_bridge(hm)

        events_to_test = [
            ("think_tag_extracted", "think_tags_extracted"),
            ("malformed_think_tag_fixed", "malformed_think_tag_fixed"),
            ("json_fence_removed", "json_fence_removed"),
            ("json_control_chars_removed", "json_control_chars_removed"),
            ("json_escape_fixed", "json_escape_fixed"),
            ("json_nested_extraction", "json_nested_extraction"),
            ("json_double_decoded", "json_double_decoded"),
            ("reasoning_as_structured_output", "reasoning_as_structured_output"),
        ]
        for event_name, _ in events_to_test:
            bridge(event_name)

        data = hm.model_dump()
        for event_name, field_name in events_to_test:
            assert data[field_name] == 1, (
                f"Field {field_name} not incremented by event {event_name}"
            )

    def test_bridge_ignores_unknown_events(self):
        from nooa.runtime.actor import _make_llm_metrics_bridge

        hm = HarnessMetrics()
        bridge = _make_llm_metrics_bridge(hm)
        bridge("totally_unknown_event")
        bridge("another_unknown", detail="some_detail")

    def test_end_to_end_callback_to_harness_metrics(self):
        """Full path: set callback via ContextVar, record metric, verify HarnessMetrics."""
        from nooa.runtime.actor import _make_llm_metrics_bridge
        from nooa.unifiedllm.unifiedllm import _llm_metrics_callback, _record_llm_metric

        hm = HarnessMetrics()
        bridge = _make_llm_metrics_bridge(hm)
        token = _llm_metrics_callback.set(bridge)
        try:
            _record_llm_metric("think_tag_extracted")
            _record_llm_metric("json_fence_removed")
            _record_llm_metric("json_double_decoded")
            data = hm.model_dump()
            assert data["think_tags_extracted"] == 1
            assert data["json_fence_removed"] == 1
            assert data["json_double_decoded"] == 1
        finally:
            _llm_metrics_callback.reset(token)


# ── Timing tests ───────────────────────────────────────────────────


class TestTimingStat:
    """Test TimingStat sub-model."""

    def test_record_single(self):
        ts = TimingStat()
        ts.record(0.5)
        assert ts.count == 1
        assert ts.total_s == pytest.approx(0.5)
        assert ts.min_s == pytest.approx(0.5)
        assert ts.max_s == pytest.approx(0.5)
        assert ts.avg_s == pytest.approx(0.5)
        assert ts.samples == [0.5]

    def test_record_multiple(self):
        ts = TimingStat()
        ts.record(0.1)
        ts.record(0.5)
        ts.record(0.2)
        assert ts.count == 3
        assert ts.total_s == pytest.approx(0.8)
        assert ts.min_s == pytest.approx(0.1)
        assert ts.max_s == pytest.approx(0.5)
        assert ts.avg_s == pytest.approx(0.8 / 3)
        assert len(ts.samples) == 3

    def test_samples_capped(self):
        ts = TimingStat()
        for i in range(_MAX_LIST_ITEMS + 5):
            ts.record(float(i))
        assert len(ts.samples) == _MAX_LIST_ITEMS
        assert ts.count == _MAX_LIST_ITEMS + 5  # count is not capped

    def test_avg_zero_when_empty(self):
        ts = TimingStat()
        assert ts.avg_s == 0.0

    def test_record_zero_elapsed(self):
        ts = TimingStat()
        ts.record(0.0)
        assert ts.count == 1
        assert ts.total_s == 0.0
        assert ts.min_s == 0.0
        assert ts.max_s == 0.0
        assert ts.avg_s == 0.0
        assert ts.samples == [0.0]

    def test_model_dump(self):
        ts = TimingStat()
        ts.record(0.123)
        d = ts.model_dump()
        assert d["count"] == 1
        assert d["total_s"] == pytest.approx(0.123)
        assert d["samples"] == [0.123]

    def test_json_round_trip_empty(self):
        """Empty TimingStat must survive JSON serialization (no float('inf'))."""
        ts = TimingStat()
        json_str = ts.model_dump_json()
        ts2 = TimingStat.model_validate_json(json_str)
        assert ts2.count == 0
        assert ts2.min_s == 0.0

    def test_json_round_trip_populated(self):
        ts = TimingStat()
        ts.record(0.1)
        ts.record(0.5)
        json_str = ts.model_dump_json()
        ts2 = TimingStat.model_validate_json(json_str)
        assert ts2.count == 2
        assert ts2.min_s == pytest.approx(0.1)
        assert ts2.max_s == pytest.approx(0.5)

    def test_min_is_correct_for_first_sample(self):
        """Verify min_s is set correctly on the first sample (not stuck at 0.0)."""
        ts = TimingStat()
        ts.record(0.5)
        assert ts.min_s == pytest.approx(0.5)
        ts.record(0.1)
        assert ts.min_s == pytest.approx(0.1)
        ts.record(0.3)
        assert ts.min_s == pytest.approx(0.1)  # unchanged


class TestTiming:
    """Test timing instrumentation on HarnessMetrics."""

    def test_timer_records_to_timing_stat(self):
        m = HarnessMetrics()
        with m.timer("time_prepare_context"):
            pass
        assert m.time_prepare_context.count == 1
        assert m.time_prepare_context.total_s > 0

    def test_timer_with_invalid_field_is_silent(self):
        m = HarnessMetrics()
        with m.timer("nonexistent_field"):
            pass  # should not raise

    def test_timer_with_non_timing_field_is_silent(self):
        m = HarnessMetrics()
        with m.timer("fence_removals"):  # list field, not TimingStat
            pass  # should not raise
        assert m.fence_removals == []  # unchanged

    def test_timer_accumulates_across_calls(self):
        import time as _time

        m = HarnessMetrics()
        with m.timer("time_llm_call"):
            _time.sleep(0.01)
        with m.timer("time_llm_call"):
            _time.sleep(0.01)
        assert m.time_llm_call.count == 2
        assert m.time_llm_call.total_s > 0.02
        assert m.time_llm_call.min_s > 0
        assert m.time_llm_call.max_s >= m.time_llm_call.min_s

    def test_tracing_overhead_records_to_timing_stat(self):
        m = HarnessMetrics()

        m.tracing_overhead(0.01)
        m.tracing_overhead(0.03)

        assert m.time_tracing_overhead.count == 2
        assert m.time_tracing_overhead.total_s == pytest.approx(0.04)
        assert m.time_tracing_overhead.min_s == pytest.approx(0.01)
        assert m.time_tracing_overhead.max_s == pytest.approx(0.03)

    def test_record_turn(self):
        m = HarnessMetrics()
        m.record_turn()
        m.record_turn()
        m.record_turn()
        assert m.turn_count == 3

    def test_timing_in_span_attributes(self):
        m = HarnessMetrics()
        m.time_prepare_context.record(0.1)
        m.time_prepare_context.record(0.3)
        m.time_llm_call.record(2.5)
        m.tracing_overhead(0.02)
        m.turn_count = 5
        attrs = m.to_span_attributes()
        assert attrs["harness.time.prepare_context.total_s"] == 0.4
        assert attrs["harness.time.prepare_context.min_s"] == 0.1
        assert attrs["harness.time.prepare_context.max_s"] == 0.3
        assert attrs["harness.time.prepare_context.avg_s"] == 0.2
        assert attrs["harness.time.prepare_context.count"] == 2
        assert attrs["harness.time.prepare_context.samples"] == [0.1, 0.3]
        assert attrs["harness.time.llm_call.total_s"] == 2.5
        assert attrs["harness.time.tracing_overhead.total_s"] == 0.02
        assert attrs["harness.time.tracing_overhead.count"] == 1
        assert attrs["harness.turn_count"] == 5

    def test_zero_timing_not_in_attributes(self):
        m = HarnessMetrics()
        attrs = m.to_span_attributes()
        assert "harness.time.prepare_context.total_s" not in attrs
        assert "harness.time.llm_call.total_s" not in attrs
        assert "harness.turn_count" not in attrs

    def test_null_metrics_timer_is_noop_context_manager(self):
        null = _NullMetrics()
        with null.timer("time_prepare_context"):
            pass

    def test_null_metrics_record_turn_is_noop(self):
        null = _NullMetrics()
        null.record_turn()


# ── Tool Usage (shell/repo) metrics tests ────────────────────────────


class TestToolUsageMetrics:
    """Test tool failure and avoidance recording methods."""

    def test_shell_failure(self):
        m = HarnessMetrics()
        m.shell_failure("bash:exit_1", "command not found", "xyz --foo")
        assert len(m.shell_failures) == 1
        assert m.shell_failures[0].error_type == "bash:exit_1"
        assert m.shell_failures[0].message == "command not found"
        assert m.shell_failures[0].code_preview == "xyz --foo"

    def test_shell_failure_truncates(self):
        m = HarnessMetrics()
        long_msg = "x" * 1000
        m.shell_failure("bash:timeout", long_msg, "cmd")
        assert len(m.shell_failures[0].message) <= _MAX_STRING_CHARS + 3  # +3 for "..."

    def test_shell_failure_respects_max_items(self):
        m = HarnessMetrics()
        for i in range(_MAX_LIST_ITEMS + 5):
            m.shell_failure(f"bash:exit_{i}", f"msg_{i}")
        assert len(m.shell_failures) == _MAX_LIST_ITEMS

    def test_repo_failure(self):
        m = HarnessMetrics()
        m.repo_failure("filemap:file_not_found", "File not found: foo.py", "foo.py")
        assert len(m.repo_failures) == 1
        assert m.repo_failures[0].error_type == "filemap:file_not_found"
        assert m.repo_failures[0].message == "File not found: foo.py"

    def test_repo_failure_respects_max_items(self):
        m = HarnessMetrics()
        for i in range(_MAX_LIST_ITEMS + 5):
            m.repo_failure(f"method_{i}", f"msg_{i}")
        assert len(m.repo_failures) == _MAX_LIST_ITEMS

    def test_tool_avoided(self):
        m = HarnessMetrics()
        m.tool_avoided("bash(sed -i 's/foo/bar/g' file.py) -> should use shell.edit")
        assert len(m.tool_avoidance) == 1
        assert "sed -i" in m.tool_avoidance[0]

    def test_tool_avoided_respects_max_items(self):
        m = HarnessMetrics()
        for i in range(_MAX_LIST_ITEMS + 5):
            m.tool_avoided(f"avoidance_{i}")
        assert len(m.tool_avoidance) == _MAX_LIST_ITEMS

    def test_tool_usage_span_attributes(self):
        m = HarnessMetrics()
        m.shell_failure("bash:exit_2", "No such file")
        m.repo_failure("search_symbol:no_results", "No matches for 'foo'")
        m.tool_avoided("bash(cat foo.py) -> should use shell.view")
        attrs = m.to_span_attributes()
        assert attrs["harness.shell_failure.count"] == 1
        assert attrs["harness.repo_failure.count"] == 1
        assert attrs["harness.tool_avoidance.count"] == 1
        assert attrs["harness.shell_failure.methods"] == ["bash:exit_2"]
        assert attrs["harness.repo_failure.methods"] == ["search_symbol:no_results"]

    def test_tool_usage_empty_not_in_span_attributes(self):
        m = HarnessMetrics()
        attrs = m.to_span_attributes()
        assert "harness.shell_failure.count" not in attrs
        assert "harness.repo_failure.count" not in attrs
        assert "harness.tool_avoidance.count" not in attrs

    def test_null_metrics_tool_methods_are_noop(self):
        null = _NullMetrics()
        null.shell_failure("bash:exit_1", "msg")
        null.repo_failure("filemap:file_not_found", "msg")
        null.tool_avoided("detail")
