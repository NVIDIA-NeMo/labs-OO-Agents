# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for tracing with metadata."""

import tempfile
from pathlib import Path

import pytest
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from otlp_test_helpers import read_otlp_jsonl_spans

from nooa.tracing._metadata import get_all_metadata
from nooa.tracing._otlp_file_exporter import OtlpJsonFileExporter
from nooa.tracing._session import set_session
from nooa.tracing._session_processor import SessionSpanProcessor


class TestTracingIntegration:
    """Integration tests for tracing with metadata.

    Creates isolated TracerProviders because OTel's global TracerProvider
    cannot be reset once set.
    """

    def test_metadata_attached_to_spans(self):
        """Metadata is attached to spans as Resource Attributes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = OtlpJsonFileExporter(tmpdir)
            metadata = get_all_metadata()
            resource = Resource(attributes=metadata)

            tracer_provider = TracerProvider(resource=resource)
            tracer_provider.add_span_processor(SessionSpanProcessor())
            tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

            set_session("test_session")

            tracer = tracer_provider.get_tracer(__name__)
            with tracer.start_as_current_span("test_span") as span:
                span.set_attribute("test.attribute", "test_value")

            tracer_provider.force_flush()

            # Session-routed file
            session_file = Path(tmpdir) / "test_session.jsonl"
            assert session_file.exists(), (
                f"Expected {session_file}, found: {list(Path(tmpdir).iterdir())}"
            )

            spans = read_otlp_jsonl_spans(session_file)
            assert len(spans) >= 1

            span_data = spans[0]

            # session.id should be in span attributes
            assert span_data["attributes"].get("session.id") == "test_session"

            # Environment metadata should be in resource attributes
            resource_attrs = span_data["resource_attributes"]
            assert "nooa.version" in resource_attrs
            assert "python.version" in resource_attrs
            assert "hostname" in resource_attrs

    def test_metadata_includes_git_info_when_available(self):
        """Git metadata is included when in a git repo."""
        import subprocess

        try:
            subprocess.check_output(["git", "rev-parse", "--git-dir"], stderr=subprocess.DEVNULL)
            in_git_repo = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            in_git_repo = False

        if not in_git_repo:
            pytest.skip("Not in a git repository")

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = OtlpJsonFileExporter(tmpdir)
            metadata = get_all_metadata()
            resource = Resource(attributes=metadata)

            tracer_provider = TracerProvider(resource=resource)
            tracer_provider.add_span_processor(SessionSpanProcessor())
            tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

            set_session("test_session")

            tracer = tracer_provider.get_tracer(__name__)
            with tracer.start_as_current_span("test_span"):
                pass

            tracer_provider.force_flush()

            session_file = Path(tmpdir) / "test_session.jsonl"
            assert session_file.exists()

            spans = read_otlp_jsonl_spans(session_file)
            assert len(spans) >= 1

            resource_attrs = spans[0]["resource_attributes"]
            assert "git.commit" in resource_attrs
            assert "git.branch" in resource_attrs
            assert "git.dirty" in resource_attrs
            assert len(resource_attrs["git.commit"]) == 7
