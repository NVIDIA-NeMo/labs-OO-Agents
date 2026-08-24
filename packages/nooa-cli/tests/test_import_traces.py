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
from nooa_cli.commands import _otlp_helpers, import_traces


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


def _patch_viewer_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(import_traces, "check_endpoint_reachable", lambda _endpoint: True)
    monkeypatch.setattr(
        import_traces,
        "session_exists",
        lambda _endpoint, _session_id: False,
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


def test_import_batches_179_records_and_syncs_before_annotations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    annotation_record = {"annotations": [{"session_id": "session", "label": "good"}]}
    trace_file = tmp_path / "session.jsonl"
    _write_trace(trace_file, 179, annotation_record)
    _patch_viewer_preflight(monkeypatch)

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
    _write_trace(trace_file, 1)
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
