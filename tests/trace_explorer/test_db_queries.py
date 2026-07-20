# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for incremental DB queries for trace explorer."""

from unittest.mock import patch

import pytest

from nooa.viewer import otlp_store


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary SQLite DB with trace data."""
    db_path = tmp_path / "test_traces.db"
    with patch.object(otlp_store, "DB_PATH", db_path):
        otlp_store.init_db()
        # Ingest a sample trace
        body = _make_sample_trace()
        otlp_store.ingest(body)
        yield db_path


def _make_sample_trace():
    """Create a sample OTLP trace with multiple spans."""
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": []},
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "aaaa",
                                "spanId": "span001",
                                "name": "RootAgent.handle",
                                "kind": 1,
                                "startTimeUnixNano": "1000000000",
                                "endTimeUnixNano": "5000000000",
                                "attributes": [
                                    {
                                        "key": "openinference.span.kind",
                                        "value": {"stringValue": "AGENT"},
                                    },
                                    {"key": "agent.name", "value": {"stringValue": "RootAgent"}},
                                    {"key": "agent.method", "value": {"stringValue": "handle"}},
                                    {"key": "session.id", "value": {"stringValue": "test-session"}},
                                ],
                                "status": {"code": 1},
                            },
                            {
                                "traceId": "aaaa",
                                "spanId": "span002",
                                "parentSpanId": "span001",
                                "name": "ChildAgent.run",
                                "kind": 1,
                                "startTimeUnixNano": "2000000000",
                                "endTimeUnixNano": "4000000000",
                                "attributes": [
                                    {
                                        "key": "openinference.span.kind",
                                        "value": {"stringValue": "AGENT"},
                                    },
                                    {"key": "agent.name", "value": {"stringValue": "ChildAgent"}},
                                    {"key": "agent.method", "value": {"stringValue": "run"}},
                                    {"key": "session.id", "value": {"stringValue": "test-session"}},
                                ],
                                "status": {"code": 2, "message": "Something failed"},
                            },
                        ],
                    }
                ],
            }
        ],
    }


class TestSessionSummaryQuery:
    """Test DB-level session summary queries."""

    def test_get_session_summary_returns_span_counts(self, temp_db):
        """Should return session-level summary without loading all spans."""
        with patch.object(otlp_store, "DB_PATH", temp_db):
            summary = otlp_store.get_session_summary("test-session")
        assert summary is not None
        assert summary["session_id"] == "test-session"
        assert summary["span_count"] >= 2
        assert "duration_ms" in summary

    def test_get_session_summary_not_found(self, temp_db):
        """Should return None for nonexistent session."""
        with patch.object(otlp_store, "DB_PATH", temp_db):
            summary = otlp_store.get_session_summary("nonexistent")
        assert summary is None


class TestAgentSpansQuery:
    """Test DB-level queries for agent spans (tree structure)."""

    def test_get_agent_spans_returns_only_agent_kind(self, temp_db):
        """Should return only AGENT spans without loading everything."""
        with patch.object(otlp_store, "DB_PATH", temp_db):
            agent_spans = otlp_store.get_agent_spans("test-session")
        assert len(agent_spans) >= 1
        for span in agent_spans:
            attrs = span.get("attributes", {})
            if isinstance(attrs, list):
                attr_dict = {a["key"]: a["value"] for a in attrs}
                assert "openinference.span.kind" in attr_dict
            else:
                assert attrs.get("openinference.span.kind") == "AGENT"

    def test_get_agent_spans_not_found(self, temp_db):
        """Should return empty list for nonexistent session."""
        with patch.object(otlp_store, "DB_PATH", temp_db):
            spans = otlp_store.get_agent_spans("nonexistent")
        assert spans == []


class TestErrorSpansQuery:
    """Test DB-level queries for error spans."""

    def test_get_error_spans_finds_errors(self, temp_db):
        """Should find spans with error status directly from DB."""
        with patch.object(otlp_store, "DB_PATH", temp_db):
            errors = otlp_store.get_error_spans("test-session")
        assert len(errors) >= 1
        for span in errors:
            assert span.get("status", {}).get("code") == 2

    def test_get_error_spans_empty_when_no_errors(self, temp_db):
        """Should return empty when no error spans exist."""
        # Ingest a trace with no errors
        with patch.object(otlp_store, "DB_PATH", temp_db):
            otlp_store.ingest(
                {
                    "resourceSpans": [
                        {
                            "resource": {"attributes": []},
                            "scopeSpans": [
                                {
                                    "spans": [
                                        {
                                            "traceId": "bbbb",
                                            "spanId": "span_ok",
                                            "name": "OkAgent.run",
                                            "kind": 1,
                                            "startTimeUnixNano": "1000000000",
                                            "endTimeUnixNano": "2000000000",
                                            "attributes": [
                                                {
                                                    "key": "session.id",
                                                    "value": {"stringValue": "ok-session"},
                                                },
                                            ],
                                            "status": {"code": 1},
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            )
            errors = otlp_store.get_error_spans("ok-session")
        assert errors == []


class TestFTSSearch:
    """Test FTS5 full-text search on spans."""

    def test_search_spans_fts_finds_matching_content(self, temp_db):
        """FTS should find spans by attribute content."""
        with patch.object(otlp_store, "DB_PATH", temp_db):
            results = otlp_store.search_spans_fts("test-session", "RootAgent")
        # Should find the span with agent.name = "RootAgent"
        assert len(results) >= 1
        assert any(
            "RootAgent" in r.get("snippet", "") or "RootAgent" in r.get("name", "") for r in results
        )

    def test_search_spans_fts_no_results(self, temp_db):
        """FTS should return empty for non-matching queries."""
        with patch.object(otlp_store, "DB_PATH", temp_db):
            results = otlp_store.search_spans_fts("test-session", "nonexistent_xyz_pattern")
        assert results == []

    def test_search_spans_fts_wrong_session(self, temp_db):
        """FTS should scope results to the given session."""
        with patch.object(otlp_store, "DB_PATH", temp_db):
            results = otlp_store.search_spans_fts("wrong-session", "RootAgent")
        assert results == []
