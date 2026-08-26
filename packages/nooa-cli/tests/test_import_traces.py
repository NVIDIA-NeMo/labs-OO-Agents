# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for reliable ``nooa import-traces`` ingestion."""

from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import pytest
from click.testing import CliRunner
from nooa_cli.commands import _otlp_helpers, delete_traces, import_harbor, import_traces


def _otlp_body(index: int) -> dict:
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": []},
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": f"{index:032x}",
                                "spanId": f"{index:016x}",
                                "name": f"span-{index}",
                            }
                        ]
                    }
                ],
            }
        ]
    }


def _write_trace(path: Path, count: int, *extra_records: dict) -> None:
    records = [_otlp_body(index) for index in range(count)]
    records.extend(extra_records)
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def _patch_viewer_preflight(
    monkeypatch: pytest.MonkeyPatch,
    *,
    stored_span_count: int = 1,
) -> None:
    monkeypatch.setattr(import_traces, "check_endpoint_reachable", lambda _endpoint: True)
    monkeypatch.setattr(
        import_traces,
        "session_exists",
        lambda _endpoint, _session_id: False,
    )
    monkeypatch.setattr(
        import_traces,
        "get_session_span_count",
        lambda _endpoint, _session_id: stored_span_count,
    )


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_post_trace_retries_503_with_exponential_backoff(monkeypatch: pytest.MonkeyPatch):
    calls = 0
    sleeps: list[float] = []

    def urlopen(request, *, timeout):
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise urllib.error.HTTPError(
                request.full_url,
                503,
                "Service Unavailable",
                {},
                io.BytesIO(b'{"error":"ingest queue is full"}'),
            )
        return _Response()

    monkeypatch.setattr(_otlp_helpers.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(_otlp_helpers.time, "sleep", sleeps.append)

    _otlp_helpers.post_trace_with_retry(
        "http://viewer:5001",
        _otlp_body(1),
        max_retries=2,
        initial_backoff=0.1,
    )

    assert calls == 3
    assert sleeps == [0.1, 0.2]


def test_post_trace_exposes_http_status_and_response_body(monkeypatch: pytest.MonkeyPatch):
    def urlopen(request, *, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            413,
            "Content Too Large",
            {},
            io.BytesIO(b'{"detail":"ingest body too large"}'),
        )

    monkeypatch.setattr(_otlp_helpers.urllib.request, "urlopen", urlopen)

    with pytest.raises(_otlp_helpers.OtlpRequestError) as exc_info:
        _otlp_helpers.post_trace_with_retry(
            "http://viewer:5001",
            _otlp_body(1),
        )

    message = str(exc_info.value)
    assert "HTTP 413 Content Too Large" in message
    assert "ingest body too large" in message
    assert "after 1 attempt" in message


def test_post_trace_does_not_replay_ambiguous_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = 0
    sleeps: list[float] = []

    def urlopen(_request, *, timeout):
        nonlocal calls
        calls += 1
        raise urllib.error.URLError(TimeoutError("response lost"))

    monkeypatch.setattr(_otlp_helpers.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(_otlp_helpers.time, "sleep", sleeps.append)

    with pytest.raises(_otlp_helpers.OtlpRequestError) as exc_info:
        _otlp_helpers.post_trace_with_retry(
            "http://viewer:5001",
            _otlp_body(1),
            max_retries=5,
        )

    assert calls == 1
    assert sleeps == []
    assert "after 1 attempt" in str(exc_info.value)


def test_endpoint_probe_exposes_authentication_failure(monkeypatch: pytest.MonkeyPatch):
    def urlopen(request, *, timeout):
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            {},
            io.BytesIO(b'{"detail":"viewer authorization required"}'),
        )

    monkeypatch.setattr(_otlp_helpers.urllib.request, "urlopen", urlopen)

    with pytest.raises(_otlp_helpers.OtlpRequestError) as exc_info:
        _otlp_helpers.check_endpoint_reachable("http://viewer:5001")

    assert exc_info.value.status_code == 401
    assert "viewer authorization required" in str(exc_info.value)


def test_remote_import_requests_use_configured_viewer_auth(
    monkeypatch: pytest.MonkeyPatch,
):
    requests: list[urllib.request.Request] = []

    def urlopen(request, *, timeout):
        requests.append(request)
        return _Response()

    monkeypatch.setenv("NOOA_VIEWER_AUTH_TOKEN", "test-viewer-token")
    monkeypatch.setattr(_otlp_helpers.urllib.request, "urlopen", urlopen)

    assert _otlp_helpers.check_endpoint_reachable("http://viewer:5001")
    assert _otlp_helpers.session_exists("http://viewer:5001", "new-session")
    _otlp_helpers.post_trace_with_retry("http://viewer:5001", _otlp_body(1))
    _otlp_helpers.sync_ingest("http://viewer:5001")
    assert _otlp_helpers.post_annotations("http://viewer:5001", [{"label": "good"}]) == 1
    assert _otlp_helpers.post_journal_record(
        "http://viewer:5001",
        {"type": "blocks", "blocks": []},
        "new-session",
    )

    assert len(requests) == 6
    assert all(
        request.get_header("Authorization") == "Bearer test-viewer-token" for request in requests
    )


def test_remote_cleanup_requests_use_configured_viewer_auth(
    monkeypatch: pytest.MonkeyPatch,
):
    requests: list[urllib.request.Request] = []

    class DeleteResponse(_Response):
        def read(self):
            return b'{"deleted":1}'

    def urlopen(request, *, timeout):
        requests.append(request)
        return DeleteResponse()

    monkeypatch.setenv("NOOA_VIEWER_AUTH_TOKEN", "test-viewer-token")
    monkeypatch.setattr(delete_traces.urllib.request, "urlopen", urlopen)

    result = CliRunner().invoke(
        delete_traces.command,
        ["--batch-id", "batch-1", "--endpoint", "http://viewer:5001"],
    )

    assert result.exit_code == 0, result.output
    assert "Deleted 1 trace(s)" in result.output
    assert len(requests) == 2
    assert all(
        request.get_header("Authorization") == "Bearer test-viewer-token" for request in requests
    )


def test_remote_cleanup_reports_authentication_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    def fail_probe(_endpoint):
        raise _otlp_helpers.OtlpRequestError(
            "HTTP 401 Unauthorized",
            status_code=401,
        )

    monkeypatch.setattr(delete_traces, "check_endpoint_reachable", fail_probe)

    result = CliRunner().invoke(
        delete_traces.command,
        ["--batch-id", "batch-1", "--endpoint", "http://viewer:5001"],
    )

    assert result.exit_code == 1
    assert "HTTP 401 Unauthorized" in result.output
    assert "Check NOOA_VIEWER_AUTH_TOKEN" in result.output
    assert "Is it running?" not in result.output


def test_harbor_import_reports_authentication_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def fail_probe(_endpoint):
        raise _otlp_helpers.OtlpRequestError(
            "HTTP 403 Forbidden",
            status_code=403,
        )

    monkeypatch.setattr(import_harbor, "check_endpoint_reachable", fail_probe)

    result = CliRunner().invoke(
        import_harbor.command,
        [str(tmp_path), "--endpoint", "http://viewer:5001"],
    )

    assert result.exit_code == 1
    assert "HTTP 403 Forbidden" in result.output
    assert "Check NOOA_VIEWER_AUTH_TOKEN" in result.output
    assert "Is it running?" not in result.output


def test_import_batches_179_records_and_syncs_before_annotations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    annotation_record = {"annotations": [{"session_id": "session", "label": "good"}]}
    trace_file = tmp_path / "session.jsonl"
    _write_trace(trace_file, 179, annotation_record)
    _patch_viewer_preflight(monkeypatch, stored_span_count=179)

    batch_sizes: list[int] = []
    events: list[str] = []

    def post_batch(_endpoint, bodies, *, max_retries):
        assert max_retries == 5
        batch_sizes.append(len(bodies))
        events.append("post")

    def sync(_endpoint):
        events.append("sync")

    def post_annotations(_endpoint, annotations):
        events.append("annotations")
        return len(annotations)

    monkeypatch.setattr(import_traces, "post_traces_batch_with_retry", post_batch)
    monkeypatch.setattr(import_traces, "sync_ingest", sync)
    monkeypatch.setattr(import_traces, "post_annotations", post_annotations)

    result = CliRunner().invoke(
        import_traces.command,
        [
            str(trace_file),
            "--endpoint",
            "http://viewer:5001",
            "--batch-id",
            "batch-1",
            "--batch-lines",
            "50",
            "--batch-bytes",
            "10000000",
        ],
    )

    assert result.exit_code == 0, result.output
    assert batch_sizes == [50, 50, 50, 29]
    assert events == ["post", "post", "post", "post", "sync", "annotations"]
    assert "1 imported, 0 skipped" in result.output


def test_import_injects_batch_and_session_attributes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    trace_file = tmp_path / "my-session.nooa.jsonl"
    body = _otlp_body(1)
    body["resourceSpans"][0]["resource"]["attributes"] = [
        {"key": "batch_id", "value": {"stringValue": "old-batch"}},
        {"key": "session.id", "value": {"stringValue": "old-session"}},
    ]
    trace_file.write_text(json.dumps(body) + "\n")
    _patch_viewer_preflight(monkeypatch)
    posted: list[dict] = []

    def post_batch(_endpoint, bodies, *, max_retries):
        posted.extend(bodies)

    monkeypatch.setattr(import_traces, "post_traces_batch_with_retry", post_batch)
    monkeypatch.setattr(import_traces, "sync_ingest", lambda _endpoint: None)

    result = CliRunner().invoke(
        import_traces.command,
        [str(trace_file), "--batch-id", "batch-1"],
    )

    assert result.exit_code == 0, result.output
    attributes = posted[0]["resourceSpans"][0]["resource"]["attributes"]
    values = {attribute["key"]: attribute["value"] for attribute in attributes}
    assert values["batch_id"] == {"stringValue": "batch-1"}
    assert values["session.id"] == {"stringValue": "my-session"}


def test_import_flushes_before_crossing_batch_byte_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    records = [_otlp_body(1), _otlp_body(2)]
    serialized = [json.dumps(record) for record in records]
    trace_file = tmp_path / "session.jsonl"
    trace_file.write_text("\n".join(serialized) + "\n")
    _patch_viewer_preflight(monkeypatch, stored_span_count=2)
    batch_sizes: list[int] = []

    def post_batch(_endpoint, bodies, *, max_retries):
        batch_sizes.append(len(bodies))

    monkeypatch.setattr(import_traces, "post_traces_batch_with_retry", post_batch)
    monkeypatch.setattr(import_traces, "sync_ingest", lambda _endpoint: None)

    result = CliRunner().invoke(
        import_traces.command,
        [
            str(trace_file),
            "--batch-id",
            "batch-1",
            "--batch-bytes",
            str(len(serialized[0].encode("utf-8")) + 1),
        ],
    )

    assert result.exit_code == 0, result.output
    assert batch_sizes == [1, 1]


def test_import_exits_nonzero_when_stored_span_count_is_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    trace_file = tmp_path / "session.jsonl"
    _write_trace(trace_file, 2)
    _patch_viewer_preflight(monkeypatch, stored_span_count=1)
    monkeypatch.setattr(
        import_traces,
        "post_traces_batch_with_retry",
        lambda _endpoint, _bodies, *, max_retries: None,
    )
    monkeypatch.setattr(import_traces, "sync_ingest", lambda _endpoint: None)

    result = CliRunner().invoke(
        import_traces.command,
        [str(trace_file), "--batch-id", "batch-1"],
    )

    assert result.exit_code == 1
    assert "viewer stored 1/2 spans" in result.output
    assert "Import incomplete" in result.output


def test_annotation_only_file_is_imported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    trace_file = tmp_path / "session.jsonl"
    trace_file.write_text(
        json.dumps({"annotations": [{"session_id": "session", "name": "quality"}]}) + "\n"
    )
    _patch_viewer_preflight(monkeypatch)
    monkeypatch.setattr(
        import_traces,
        "post_annotations",
        lambda _endpoint, annotations: len(annotations),
    )

    result = CliRunner().invoke(
        import_traces.command,
        [str(trace_file), "--batch-id", "batch-1"],
    )

    assert result.exit_code == 0, result.output
    assert "1 imported, 0 skipped" in result.output
    assert "1 annotation(s) imported" in result.output


def test_journal_only_file_is_imported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    trace_file = tmp_path / "session.nooa.jsonl"
    trace_file.write_text(
        json.dumps(
            {
                "nooaJournal": {
                    "format": "nooa.message_journal",
                    "version": 1,
                    "type": "blocks",
                    "blocks": [],
                }
            }
        )
        + "\n"
    )
    _patch_viewer_preflight(monkeypatch)
    monkeypatch.setattr(
        import_traces,
        "post_journal_record",
        lambda _endpoint, _record, _session_id: True,
    )

    result = CliRunner().invoke(
        import_traces.command,
        [str(trace_file), "--batch-id", "batch-1"],
    )

    assert result.exit_code == 0, result.output
    assert "1 imported, 0 skipped" in result.output


def test_import_exits_nonzero_and_prints_cleanup_on_batch_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    trace_file = tmp_path / "session.jsonl"
    _write_trace(trace_file, 2)
    _patch_viewer_preflight(monkeypatch)
    sync_calls: list[str] = []

    def fail_batch(_endpoint, _bodies, *, max_retries):
        raise _otlp_helpers.OtlpRequestError(
            'HTTP 503 Service Unavailable: {"error":"ingest queue is full"}',
            status_code=503,
            retryable=True,
        )

    monkeypatch.setattr(import_traces, "post_traces_batch_with_retry", fail_batch)
    monkeypatch.setattr(import_traces, "sync_ingest", sync_calls.append)

    result = CliRunner().invoke(
        import_traces.command,
        [str(trace_file), "--endpoint", "http://viewer:5001", "--batch-id", "batch-1"],
    )

    assert result.exit_code == 1
    assert sync_calls == ["http://viewer:5001"]
    assert "HTTP 503 Service Unavailable" in result.output
    assert "1 failed" in result.output
    assert "Import incomplete" in result.output
    assert "nooa delete-traces --batch-id batch-1" in result.output


def test_import_exits_nonzero_when_sync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    trace_file = tmp_path / "session.jsonl"
    _write_trace(trace_file, 1)
    _patch_viewer_preflight(monkeypatch)

    monkeypatch.setattr(
        import_traces,
        "post_traces_batch_with_retry",
        lambda _endpoint, _bodies, *, max_retries: None,
    )

    def fail_sync(_endpoint):
        raise _otlp_helpers.OtlpRequestError(
            'HTTP 503 Service Unavailable: {"error":"timeout waiting for queue drain"}',
            status_code=503,
            retryable=True,
        )

    monkeypatch.setattr(import_traces, "sync_ingest", fail_sync)

    result = CliRunner().invoke(
        import_traces.command,
        [str(trace_file), "--batch-id", "batch-1"],
    )

    assert result.exit_code == 1
    assert "failed to sync viewer ingest" in result.output
    assert "timeout waiting for queue drain" in result.output
