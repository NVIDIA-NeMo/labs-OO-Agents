# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for enable_tracing() idempotency.

Verifies that enable_tracing() is idempotent — multiple calls are no-ops
and don't accumulate span processors.
"""

import tempfile
from pathlib import Path

from otlp_test_helpers import read_all_otlp_jsonl_spans as read_otlp_jsonl_spans_from_dir
from otlp_test_helpers import read_otlp_jsonl_spans


class TestEnableTracingIdempotency:
    """Tests for enable_tracing() idempotency behaviour."""

    def test_no_arg_calls_are_noop_after_first(self):
        """No-arg enable_tracing() is a no-op once tracing is enabled."""
        import nooa.tracing as module
        from nooa.tracing import enable_tracing, exporters

        with tempfile.TemporaryDirectory() as tmpdir:
            enable_tracing(exporters=[exporters.jsonl(tmpdir)])
            assert module._enabled is True

            provider_after_first = module._provider

            # No-arg call — should be ignored
            enable_tracing()

            assert module._provider is provider_after_first

    def test_explicit_exporters_replace_after_enabled(self):
        """Calling with explicit exporters after enabled replaces the old ones."""
        from nooa.tracing import enable_tracing, exporters

        with (
            tempfile.TemporaryDirectory() as dir1,
            tempfile.TemporaryDirectory() as dir2,
        ):
            enable_tracing(exporters=[exporters.jsonl(dir1)])

            # Second call with explicit exporters — should REPLACE, not add
            enable_tracing(exporters=[exporters.jsonl(dir2)])

            # Emit a span — it should appear ONLY in dir2
            from opentelemetry import trace as otel_trace

            tracer = otel_trace.get_tracer("test")
            with tracer.start_as_current_span("replace_test_span"):
                pass

            from nooa.tracing import flush_traces

            flush_traces()

            spans1 = read_otlp_jsonl_spans_from_dir(dir1)
            spans2 = read_otlp_jsonl_spans_from_dir(dir2)

            assert not any(s["name"] == "replace_test_span" for s in spans1), (
                f"Span should NOT be in first (replaced) exporter dir. Files: {list(Path(dir1).iterdir())}"
            )
            assert any(s["name"] == "replace_test_span" for s in spans2), (
                f"Span not found in second (replacement) exporter dir. Files: {list(Path(dir2).iterdir())}"
            )

    def test_span_written_once_per_span(self):
        """Each span should be written exactly once with isolated TracerProvider."""
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        from nooa.tracing._otlp_file_exporter import OtlpJsonFileExporter
        from nooa.tracing._session import set_session
        from nooa.tracing._session_processor import SessionSpanProcessor

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = OtlpJsonFileExporter(tmpdir)
            tracer_provider = TracerProvider()
            tracer_provider.add_span_processor(SessionSpanProcessor())
            tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

            set_session("test-session")

            tracer = tracer_provider.get_tracer(__name__)
            with tracer.start_as_current_span("test_idempotent_span") as span:
                span.set_attribute("test.attribute", "test_value")

            tracer_provider.force_flush()

            trace_files = list(Path(tmpdir).glob("*.jsonl"))
            assert len(trace_files) >= 1

            session_file = Path(tmpdir) / "test-session.jsonl"
            assert session_file.exists()

            spans = read_otlp_jsonl_spans(session_file)
            test_spans = [s for s in spans if s["name"] == "test_idempotent_span"]
            assert len(test_spans) == 1, f"Expected 1 span, found {len(test_spans)}"


class TestFlushAndShutdown:
    """Tests for flush_traces() and shutdown_traces()."""

    def test_flush_traces_no_op_when_not_enabled(self):
        """flush_traces() should be safe to call when tracing is not enabled."""
        from nooa.tracing import flush_traces

        flush_traces()  # Should not raise

    def test_shutdown_traces_no_op_when_not_enabled(self):
        """shutdown_traces() should be safe to call when tracing is not enabled."""
        from nooa.tracing import shutdown_traces

        shutdown_traces()  # Should not raise

    def test_flush_traces_after_enable(self):
        """flush_traces() should work after enable_tracing()."""
        from nooa.tracing import enable_tracing, exporters, flush_traces

        with tempfile.TemporaryDirectory() as tmpdir:
            enable_tracing(exporters=[exporters.jsonl(tmpdir)])
            flush_traces()  # Should not raise
