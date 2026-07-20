# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for OtlpJsonHttpExporter drop counting and logging.

Verifies that:
  - Successful exports leave counters at zero and produce no log output.
  - HTTP errors, timeouts, and non-2xx responses increment the drop counter
    and emit a log.ERROR per failure.
  - shutdown() is silent when no spans were dropped.
  - shutdown() emits log.WARNING when drops occurred, including the count.
"""

import urllib.error
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace.export import SpanExportResult

from nooa.tracing._otlp_http_exporter import OtlpJsonHttpExporter

# Patch build_resource_spans throughout — we're testing HTTP/drop behaviour,
# not span serialization.
_SERIALIZE_PATH = "nooa.tracing._otlp_http_exporter.build_resource_spans"
_FAKE_RESOURCE_SPANS = [{"spans": []}]


@pytest.fixture()
def exporter():
    return OtlpJsonHttpExporter(endpoint="http://localhost:5001/v1/traces")


@pytest.fixture(autouse=True)
def mock_serialize():
    with patch(_SERIALIZE_PATH, return_value=_FAKE_RESOURCE_SPANS):
        yield


def _spans(n: int = 3, session_id: str | None = None):
    """Return *n* placeholder span objects (content irrelevant — serialization is mocked)."""
    spans = []
    for _ in range(n):
        span = MagicMock()
        if session_id is not None:
            span.attributes = {"session.id": session_id}
        else:
            span.attributes = {}
        spans.append(span)
    return spans


def _ok_response():
    resp = MagicMock()
    resp.status = 200
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


class TestSuccessfulExport:
    def test_returns_success(self, exporter):
        with patch("urllib.request.urlopen", return_value=_ok_response()):
            result = exporter.export(_spans())
        assert result == SpanExportResult.SUCCESS

    def test_no_drops_on_success(self, exporter):
        with patch("urllib.request.urlopen", return_value=_ok_response()):
            exporter.export(_spans(5))
        assert exporter._dropped_spans == 0
        assert exporter._failed_batches == 0

    def test_shutdown_silent_when_no_drops(self, exporter, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            exporter.shutdown()
        assert caplog.text == ""


class TestFailedExport:
    def test_timeout_increments_drop_counter(self, exporter):
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            result = exporter.export(_spans(4))
        assert result == SpanExportResult.FAILURE
        assert exporter._dropped_spans == 4
        assert exporter._failed_batches == 1

    def test_http_error_increments_drop_counter(self, exporter):
        err = urllib.error.HTTPError(
            url="http://localhost:5001/v1/traces",
            code=503,
            msg="Service Unavailable",
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )
        with patch("urllib.request.urlopen", side_effect=err):
            exporter.export(_spans(2))
        assert exporter._dropped_spans == 2
        assert exporter._failed_batches == 1

    def test_non_2xx_response_increments_drop_counter(self, exporter):
        resp = MagicMock()
        resp.status = 500
        resp.__enter__ = lambda s: s
        resp.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=resp):
            result = exporter.export(_spans(3))
        assert result == SpanExportResult.FAILURE
        assert exporter._dropped_spans == 3

    def test_failure_logs_error(self, exporter, caplog):
        import logging

        with (
            patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")),
            caplog.at_level(logging.ERROR),
        ):
            exporter.export(_spans(2))

        errors = [r for r in caplog.records if r.levelname == "ERROR"]
        assert errors, "Expected at least one ERROR log entry"
        assert any("DROP" in r.message for r in errors)
        assert any("2" in r.message for r in errors)

    def test_accumulated_drops_across_batches(self, exporter):
        with patch("urllib.request.urlopen", side_effect=TimeoutError()):
            exporter.export(_spans(3))
            exporter.export(_spans(2))
        assert exporter._dropped_spans == 5
        assert exporter._failed_batches == 2


class TestSessionFromSpanAttributes:
    """Exporter must extract session.id from span attributes, not from the ContextVar.

    BatchSpanProcessor calls export() from a worker thread whose ContextVar context
    is a frozen copy from thread-start time — before set_session(session_id) is ever
    called in the eval task.  get_session() would return None in that thread, routing
    all spans to unknown_* sessions.  The exporter must read session.id from span
    attributes instead (where SessionSpanProcessor correctly stamped it).
    """

    def test_session_extracted_from_first_span_with_session_attr(self, exporter):
        captured = {}

        def fake_build(spans, resource_attrs_override=None, **_kwargs):
            captured["override"] = resource_attrs_override
            return _FAKE_RESOURCE_SPANS

        with (
            patch(_SERIALIZE_PATH, side_effect=fake_build),
            patch("urllib.request.urlopen", return_value=_ok_response()),
        ):
            exporter.export(_spans(3, session_id="my-session-123"))

        assert captured["override"] == {"session.id": "my-session-123"}

    def test_no_override_when_spans_have_no_session(self, exporter):
        captured = {}

        def fake_build(spans, resource_attrs_override=None, **_kwargs):
            captured["override"] = resource_attrs_override
            return _FAKE_RESOURCE_SPANS

        with (
            patch(_SERIALIZE_PATH, side_effect=fake_build),
            patch("urllib.request.urlopen", return_value=_ok_response()),
        ):
            exporter.export(_spans(3, session_id=None))

        assert captured["override"] is None

    def test_groups_spans_by_session_id(self, exporter):
        """Spans from different sessions are sent as separate HTTP requests."""
        call_overrides = []

        def fake_build(spans, resource_attrs_override=None, **_kwargs):
            call_overrides.append(resource_attrs_override)
            return _FAKE_RESOURCE_SPANS

        spans = _spans(2, session_id="first-session") + _spans(2, session_id="second-session")
        with (
            patch(_SERIALIZE_PATH, side_effect=fake_build),
            patch("urllib.request.urlopen", return_value=_ok_response()),
        ):
            result = exporter.export(spans)

        assert result == SpanExportResult.SUCCESS
        assert len(call_overrides) == 2
        session_ids = {o["session.id"] for o in call_overrides if o}
        assert session_ids == {"first-session", "second-session"}


class TestShutdownWarning:
    def test_shutdown_warns_when_drops_occurred(self, exporter, caplog):
        import logging

        with patch("urllib.request.urlopen", side_effect=TimeoutError()):
            exporter.export(_spans(7))

        with caplog.at_level(logging.WARNING):
            exporter.shutdown()

        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert warnings, "Expected a WARNING log on shutdown after drops"
        assert any("7" in r.message for r in warnings)

    def test_shutdown_warning_includes_batch_count(self, exporter, caplog):
        import logging

        with patch("urllib.request.urlopen", side_effect=TimeoutError()):
            exporter.export(_spans(1))
            exporter.export(_spans(1))

        with caplog.at_level(logging.WARNING):
            exporter.shutdown()

        warning_text = " ".join(r.message for r in caplog.records if r.levelname == "WARNING")
        assert "2" in warning_text  # 2 failed batches
