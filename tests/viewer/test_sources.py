# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source isolation, lazy loading, and cache publication using synthetic data."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from nooa.viewer import main, otlp_store, sources


class ExampleSource:
    version = "1"

    def __init__(self):
        self.loads = 0

    def list_sessions(self):
        return [
            {
                "id": "example-task",
                "name": "Task",
                "experiment": "Example",
                "modified": 1,
                "span_count": None,
                "eval": {"passed": False},
            }
        ]

    def load_session(self, session_id):
        self.loads += 1
        return [
            {
                "nooaJournal": {
                    "type": "blocks",
                    "session_id": session_id,
                    "blocks": [{"hash": "text", "content": "stdout"}],
                }
            },
            {
                "nooaJournal": {
                    "type": "call",
                    "session_id": session_id,
                    "call": {
                        "call_id": "call",
                        "session_id": session_id,
                        "span_id": "a" * 16,
                        "input_skeleton": [{"role": "user", "parts": [{"block_hash": "text"}]}],
                        "output_messages": [],
                        "tokens": {},
                    },
                }
            },
            {
                "resourceSpans": [
                    {
                        "resource": {
                            "attributes": [
                                {"key": "session.id", "value": {"stringValue": session_id}},
                                {"key": "experiment", "value": {"stringValue": "Example"}},
                            ]
                        },
                        "scopeSpans": [
                            {
                                "spans": [
                                    {
                                        "traceId": "b" * 32,
                                        "spanId": "a" * 16,
                                        "name": "llm_call",
                                        "startTimeUnixNano": "1000000000",
                                        "endTimeUnixNano": "2000000000",
                                        "attributes": [],
                                    }
                                ]
                            }
                        ],
                    }
                ]
            },
        ]


@pytest.fixture
def source_store(tmp_path, monkeypatch):
    monkeypatch.setattr(otlp_store, "DB_PATH", tmp_path / "traces.db")
    otlp_store.init_db()
    source = ExampleSource()
    monkeypatch.setattr(sources, "configured_sources", lambda: {"example": source})
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr(main, "_write_executor", executor)
        yield source


def test_catalog_is_lazy_and_materialization_is_idempotent(source_store):
    assert sources.list_experiments() == ["Example"]
    assert sources.list_sessions()[0]["span_count"] is None
    assert not otlp_store.session_exists("example-task")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(sources.ensure_session, ["example-task"] * 4))
    assert source_store.loads == 1
    assert len(otlp_store.get_session_spans("example-task")) == 1
    assert sources.list_sessions()[0]["span_count"] == 1
    assert (
        otlp_store.get_session_calls("example-task")[0]["input_skeleton"][0]["parts"][0][
            "block_hash"
        ]
        == "text"
    )
    assert otlp_store.get_session_blocks("example-task")["text"] == "stdout"


def test_source_cannot_overwrite_local_session(source_store):
    otlp_store.ingest(source_store.load_session("example-task")[-1])
    with pytest.raises(ValueError, match="overwrite"):
        sources.ensure_session("example-task")


def test_native_publication_during_fetch_is_preserved(source_store, monkeypatch):
    original = source_store.load_session

    def concurrent_publication(session_id):
        records = original(session_id)
        otlp_store.ingest(records[-1])
        return records

    monkeypatch.setattr(source_store, "load_session", concurrent_publication)
    with pytest.raises(ValueError, match="overwrite"):
        sources.ensure_session("example-task")
    assert len(otlp_store.get_session_spans("example-task")) == 1
    assert "viewer.source.name" not in otlp_store.get_session_resource("example-task")
    assert otlp_store.get_session_calls("example-task") == []


def test_invalid_identity_does_not_publish_trace(source_store, monkeypatch):
    monkeypatch.setattr(
        source_store, "load_session", lambda sid: ExampleSource().load_session("other")
    )
    with pytest.raises(ValueError, match="identity"):
        sources.ensure_session("example-task")
    assert not otlp_store.session_exists("other")
    assert not otlp_store.session_exists("example-task")


def test_version_change_preserves_cached_data(source_store):
    sources.ensure_session("example-task")
    source_store.version = "2"
    with pytest.raises(ValueError, match="version changed"):
        sources.ensure_session("example-task")
    assert len(otlp_store.get_session_spans("example-task")) == 1


def test_cached_detail_does_not_need_remote_catalog(source_store, monkeypatch):
    sources.ensure_session("example-task")

    def offline():
        raise RuntimeError("offline")

    monkeypatch.setattr(source_store, "list_sessions", offline)
    sources.ensure_session("example-task")
    assert len(otlp_store.get_session_spans("example-task")) == 1


def test_duplicate_source_identity_rejected(source_store, monkeypatch):
    monkeypatch.setattr(
        sources, "configured_sources", lambda: {"a": source_store, "b": source_store}
    )
    with pytest.raises(ValueError, match="collision|namespace"):
        sources.list_sessions()


def test_journal_failure_can_retry_without_duplicate_spans(source_store, monkeypatch):
    original = otlp_store.ingest_journal_call

    def fail(call):
        raise RuntimeError("interrupted")

    monkeypatch.setattr(otlp_store, "ingest_journal_call", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        sources.ensure_session("example-task")
    assert not otlp_store.session_exists("example-task")
    monkeypatch.setattr(otlp_store, "ingest_journal_call", original)
    sources.ensure_session("example-task")
    assert len(otlp_store.get_session_spans("example-task")) == 1
