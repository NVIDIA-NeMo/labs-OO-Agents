# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SessionSpanProcessor."""

import tempfile
from pathlib import Path

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from otlp_test_helpers import read_otlp_jsonl_spans

from nooa.tracing._otlp_file_exporter import OtlpJsonFileExporter
from nooa.tracing._session import set_session
from nooa.tracing._session_processor import SessionSpanProcessor


class TestSessionSpanProcessor:
    def test_stamps_session_id_on_span(self):
        """SessionSpanProcessor should set session.id as a span attribute."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = OtlpJsonFileExporter(tmpdir)
            provider = TracerProvider()
            provider.add_span_processor(SessionSpanProcessor())
            provider.add_span_processor(SimpleSpanProcessor(exporter))

            set_session("my-session-42")

            tracer = provider.get_tracer(__name__)
            with tracer.start_as_current_span("test"):
                pass

            provider.force_flush()

            session_file = Path(tmpdir) / "my-session-42.jsonl"
            assert session_file.exists()

            spans = read_otlp_jsonl_spans(session_file)
            assert len(spans) >= 1
            assert spans[0]["attributes"]["session.id"] == "my-session-42"

    def test_no_session_uses_default_file(self):
        """Without set_session(), spans go to the default trace file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = OtlpJsonFileExporter(tmpdir)
            provider = TracerProvider()
            provider.add_span_processor(SessionSpanProcessor())
            provider.add_span_processor(SimpleSpanProcessor(exporter))

            set_session(None)

            tracer = provider.get_tracer(__name__)
            with tracer.start_as_current_span("test"):
                pass

            provider.force_flush()

            # Should go to the default timestamp-based file
            assert exporter.default_file is not None
            assert exporter.default_file.exists()
            spans = read_otlp_jsonl_spans(exporter.default_file)
            assert len(spans) >= 1
            assert "session.id" not in spans[0]["attributes"]

    def test_session_routes_to_different_files(self):
        """Different sessions should route to different files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = OtlpJsonFileExporter(tmpdir)
            provider = TracerProvider()
            provider.add_span_processor(SessionSpanProcessor())
            provider.add_span_processor(SimpleSpanProcessor(exporter))

            tracer = provider.get_tracer(__name__)

            set_session("session-a")
            with tracer.start_as_current_span("span-a"):
                pass

            set_session("session-b")
            with tracer.start_as_current_span("span-b"):
                pass

            provider.force_flush()

            file_a = Path(tmpdir) / "session-a.jsonl"
            file_b = Path(tmpdir) / "session-b.jsonl"

            assert file_a.exists()
            assert file_b.exists()

            spans_a = read_otlp_jsonl_spans(file_a)
            spans_b = read_otlp_jsonl_spans(file_b)

            assert spans_a[0]["name"] == "span-a"
            assert spans_a[0]["attributes"]["session.id"] == "session-a"
            assert spans_b[0]["name"] == "span-b"
            assert spans_b[0]["attributes"]["session.id"] == "session-b"

    def test_force_flush_returns_true(self):
        proc = SessionSpanProcessor()
        assert proc.force_flush() is True

    def test_shutdown_is_noop(self):
        proc = SessionSpanProcessor()
        proc.shutdown()  # Should not raise

    def test_session_propagates_via_otel_context_to_openinference_spans(self):
        """Regression: acompletion spans created by LiteLLMInstrumentor
        were landing in ``unknown_*`` sessions because our ContextVar was
        a separate propagation channel from OTel's context — and
        OpenInference's OITracer reads ``session.id`` out of the OTel
        context at span creation, not from our ContextVar.

        This test uses :class:`OITracer` directly (same tracer class
        LiteLLMInstrumentor uses) to confirm ``set_session()`` makes it
        all the way onto a span without relying on
        :class:`SessionSpanProcessor`.
        """
        from openinference.instrumentation import OITracer, TraceConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = OtlpJsonFileExporter(tmpdir)
            provider = TracerProvider()
            # Note: deliberately NOT adding SessionSpanProcessor so this
            # proves the OTel context channel works on its own.
            provider.add_span_processor(SimpleSpanProcessor(exporter))

            set_session("via-otel-ctx")
            tracer = OITracer(provider.get_tracer(__name__), config=TraceConfig())
            with tracer.start_as_current_span("oi-span"):
                pass
            provider.force_flush()

            session_file = Path(tmpdir) / "via-otel-ctx.jsonl"
            # If this file doesn't exist, session.id wasn't propagated:
            # the exporter routes by session.id attribute.
            assert session_file.exists(), (
                "session.id must reach OITracer-created spans via the OTel context"
            )
            spans = read_otlp_jsonl_spans(session_file)
            assert spans[0]["attributes"]["session.id"] == "via-otel-ctx"
