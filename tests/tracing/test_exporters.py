# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for exporter factory functions."""

import tempfile
from pathlib import Path

import pytest

from nooa.tracing import exporters
from nooa.tracing._otlp_file_exporter import OtlpJsonFileExporter


class TestJsonlFactory:
    def test_creates_otlp_file_exporter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            exp = exporters.jsonl(tmpdir)
            assert isinstance(exp, OtlpJsonFileExporter)
            assert exp.trace_dir == Path(tmpdir)

    def test_creates_dir_if_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sub = Path(tmpdir) / "sub" / "traces"
            exp = exporters.jsonl(sub)
            assert sub.exists()
            assert isinstance(exp, OtlpJsonFileExporter)

    def test_auto_detect_with_env_var(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("TRACE_DIR", tmpdir)
            exp = exporters.jsonl()
            assert isinstance(exp, OtlpJsonFileExporter)
            assert exp.trace_dir == Path(tmpdir)


class TestJournalFileFactory:
    def test_creates_opt_in_journal_file_exporter(self):
        from nooa.tracing._journal_file_exporter import JournalFileExporter

        with tempfile.TemporaryDirectory() as tmpdir:
            exp = exporters.journal_file(tmpdir)
            try:
                assert isinstance(exp, JournalFileExporter)
                assert exp.trace_dir == Path(tmpdir)
            finally:
                exp.shutdown()


class TestOtlpFactory:
    def test_raises_import_error_for_http(self):
        # This test assumes the OTLP HTTP exporter may or may not be installed
        try:
            exp = exporters.otlp("http://localhost:4318/v1/traces")
            # If installed, it should be a SpanExporter
            from opentelemetry.sdk.trace.export import SpanExporter

            assert isinstance(exp, SpanExporter)
        except ImportError as e:
            assert "OTLP HTTP exporter" in str(e)


class TestLangfuseFactory:
    def test_requires_credentials(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_HOST", raising=False)
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)

        with pytest.raises(ValueError, match="Langfuse requires"):
            exporters.langfuse()


class TestLocalOtlpFactory:
    def test_creates_otlp_json_http_exporter(self, monkeypatch):
        from nooa.tracing._otlp_http_exporter import OtlpJsonHttpExporter

        monkeypatch.delenv("OTLP_ENDPOINT", raising=False)
        exp = exporters.local_otlp()
        assert isinstance(exp, OtlpJsonHttpExporter)
        assert exp._endpoint == "http://localhost:5001/v1/traces"

    def test_custom_endpoint(self):
        from nooa.tracing._otlp_http_exporter import OtlpJsonHttpExporter

        exp = exporters.local_otlp(endpoint="http://custom:9999/v1/traces")
        assert isinstance(exp, OtlpJsonHttpExporter)
        assert exp._endpoint == "http://custom:9999/v1/traces"

    def test_env_var_override(self, monkeypatch):
        from nooa.tracing._otlp_http_exporter import OtlpJsonHttpExporter

        monkeypatch.setenv("OTLP_ENDPOINT", "http://env-host:8080/v1/traces")
        exp = exporters.local_otlp()
        assert isinstance(exp, OtlpJsonHttpExporter)
        assert exp._endpoint == "http://env-host:8080/v1/traces"


class TestConsoleFactory:
    def test_creates_console_exporter(self):
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        exp = exporters.console()
        assert isinstance(exp, ConsoleSpanExporter)
