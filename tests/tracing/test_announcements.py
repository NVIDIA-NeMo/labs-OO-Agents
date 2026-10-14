# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Announcement policy hides startup notices, not setup, exports or errors."""

import asyncio
from contextlib import nullcontext
from contextvars import Context
from unittest.mock import Mock

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

import nooa.tracing as tracing
from nooa.runtime.hooks import get_hooks, set_hooks


class RecordingExporter(SpanExporter):
    synchronous = True

    def __init__(self):
        self.spans = []

    def describe(self):
        return "journal:https://internal-traces.test"

    def export(self, spans):
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS


@pytest.fixture
def trace_setup(monkeypatch):
    """Use real setup/processors with an isolated provider and no external I/O."""
    provider = TracerProvider(shutdown_on_exit=False)
    exporter = RecordingExporter()
    metadata = Mock(return_value={"service.name": "test"})
    instrumentation = Mock()
    monkeypatch.setattr(tracing, "_enabled", False)
    monkeypatch.setattr(tracing, "_provider", None)
    monkeypatch.setattr(tracing, "_hooks", None)
    monkeypatch.setattr(tracing, "_probe_failed", False)
    monkeypatch.setattr(tracing.trace, "get_tracer_provider", lambda: provider)
    monkeypatch.setattr(tracing, "get_all_metadata", metadata)
    monkeypatch.setattr(tracing, "_instrument_litellm", instrumentation)
    monkeypatch.setattr(tracing, "_default_exporters", lambda: [exporter])
    previous_hooks = get_hooks()
    set_hooks(None)
    try:
        yield provider, exporter, metadata, instrumentation
    finally:
        set_hooks(previous_hooks)
        provider.shutdown()


@pytest.mark.parametrize("fail", [False, True])
def test_setup_and_reconfiguration_are_quiet_and_policy_restores(trace_setup, capsys, fail):
    provider, exporter, metadata, instrumentation = trace_setup
    with pytest.raises(ValueError) if fail else nullcontext():
        with tracing.suppress_tracing_announcements():
            tracing.enable_tracing()
            assert tracing._enabled and tracing._provider is provider
            assert tracing._hooks is not None
            with tracing.suppress_tracing_announcements():
                tracing.enable_tracing(exporters=[exporter], experiment="nested")
            tracing.enable_tracing(exporters=[exporter])
            with provider.get_tracer("test").start_as_current_span("still exported"):
                pass
            if fail:
                raise ValueError("genuine failure")
    assert capsys.readouterr().out == ""
    assert [span.name for span in exporter.spans] == ["still exported"]
    metadata.assert_called_once()
    instrumentation.assert_called_once_with(provider)
    tracing.enable_tracing(exporters=[exporter], experiment="outside")
    assert capsys.readouterr().out == (
        "OTel tracing enabled: journal:https://internal-traces.test (experiment=outside)\n"
    )


@pytest.mark.asyncio
async def test_policy_is_context_local_and_inherited_by_child_tasks(trace_setup, capsys):
    tracing.enable_tracing()
    capsys.readouterr()

    async def enable():
        tracing.enable_tracing(exporters=[trace_setup[1]])

    with tracing.suppress_tracing_announcements():
        await asyncio.create_task(enable())
        assert capsys.readouterr().out == ""
        # An unrelated context still announces, even while this one is quiet.
        Context().run(tracing.enable_tracing, exporters=[trace_setup[1]])
        assert "OTel tracing enabled:" in capsys.readouterr().out
        tracing.enable_tracing(exporters=[trace_setup[1]])
        assert capsys.readouterr().out == ""


def test_endpoint_startup_notice_is_quiet_but_setup_error_is_not_suppressed(monkeypatch, capsys):
    monkeypatch.setattr(tracing, "resolve_otlp_endpoint", lambda endpoint: None)
    monkeypatch.setattr(tracing, "_probe_failed", False)
    monkeypatch.setenv("OTLP_ENDPOINT", "https://unreachable.test/v1/traces")
    with tracing.suppress_tracing_announcements():
        assert tracing._default_exporters() is None
    assert capsys.readouterr().err == ""
    assert tracing._probe_failed is True
    assert tracing._default_exporters() is None
    assert "is not reachable — tracing disabled" in capsys.readouterr().err
    monkeypatch.setattr(tracing, "_enabled", False)
    monkeypatch.setattr(tracing, "get_all_metadata", Mock(side_effect=RuntimeError("setup failed")))
    with pytest.raises(RuntimeError, match="setup failed"):
        with tracing.suppress_tracing_announcements():
            tracing.enable_tracing(exporters=[RecordingExporter()])
    assert tracing._announce_trace_target.get() is True


def test_file_target_tip_is_part_of_the_suppressed_announcement(trace_setup, tmp_path, capsys):
    exporter = tracing.OtlpJsonFileExporter(tmp_path)
    with tracing.suppress_tracing_announcements():
        tracing.enable_tracing(exporters=[exporter])
    assert capsys.readouterr().out == ""
    tracing.enable_tracing(exporters=[exporter])
    output = capsys.readouterr().out
    assert "OTel tracing enabled:" in output
    assert "Tip: Run `nooa start-dev`" in output


@pytest.mark.parametrize("quiet", [False, True])
def test_explicit_enable_unreachable_notice_obeys_scope_only(monkeypatch, capsys, caplog, quiet):
    monkeypatch.setattr(tracing, "_enabled", False)
    monkeypatch.setattr(tracing, "_provider", None)
    monkeypatch.setattr(tracing, "_probe_failed", False)
    monkeypatch.setenv("OTLP_ENDPOINT", "https://unreachable.test/v1/traces")
    resolve = Mock(return_value=None)
    monkeypatch.setattr(tracing, "resolve_otlp_endpoint", resolve)
    with tracing.suppress_tracing_announcements() if quiet else nullcontext():
        tracing.enable_tracing()
        tracing.logger.warning("unrelated warning stays visible")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        ""
        if quiet
        else "OTLP_ENDPOINT (https://unreachable.test/v1/traces) is not reachable — tracing disabled.\n"
    )
    resolve.assert_called_once_with("https://unreachable.test/v1/traces")
    assert tracing._probe_failed is True
    assert tracing._enabled is False
    assert "unrelated warning stays visible" in caplog.text
    assert tracing._announce_trace_target.get() is True
