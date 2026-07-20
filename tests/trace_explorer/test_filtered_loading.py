# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for filtered/paginated span loading."""

from unittest.mock import patch

import pytest

from nooa.viewer import otlp_store


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary SQLite DB with a multi-session trace."""
    db_path = tmp_path / "test_traces.db"
    with patch.object(otlp_store, "DB_PATH", db_path):
        otlp_store.init_db()
        otlp_store.ingest(_make_multi_session_trace())
        yield db_path


def _make_multi_session_trace():
    """Create a trace with a parent AGENT and two child AGENT spans plus LLM spans."""
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": []},
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "traceId": "aaaa",
                                "spanId": "root_span_01",
                                "name": "RootAgent.handle",
                                "kind": 1,
                                "startTimeUnixNano": "1000000000",
                                "endTimeUnixNano": "9000000000",
                                "attributes": [
                                    {
                                        "key": "openinference.span.kind",
                                        "value": {"stringValue": "AGENT"},
                                    },
                                    {"key": "agent.name", "value": {"stringValue": "RootAgent"}},
                                    {"key": "agent.method", "value": {"stringValue": "handle"}},
                                    {"key": "session.id", "value": {"stringValue": "multi-sess"}},
                                ],
                                "status": {"code": 1},
                            },
                            {
                                "traceId": "aaaa",
                                "spanId": "child_span_1",
                                "parentSpanId": "root_span_01",
                                "name": "ChildA.run",
                                "kind": 1,
                                "startTimeUnixNano": "2000000000",
                                "endTimeUnixNano": "4000000000",
                                "attributes": [
                                    {
                                        "key": "openinference.span.kind",
                                        "value": {"stringValue": "AGENT"},
                                    },
                                    {"key": "agent.name", "value": {"stringValue": "ChildA"}},
                                    {"key": "session.id", "value": {"stringValue": "multi-sess"}},
                                ],
                                "status": {"code": 1},
                            },
                            {
                                "traceId": "aaaa",
                                "spanId": "llm_span_001",
                                "parentSpanId": "child_span_1",
                                "name": "acompletion",
                                "kind": 3,
                                "startTimeUnixNano": "2500000000",
                                "endTimeUnixNano": "3500000000",
                                "attributes": [
                                    {
                                        "key": "openinference.span.kind",
                                        "value": {"stringValue": "LLM"},
                                    },
                                    {"key": "session.id", "value": {"stringValue": "multi-sess"}},
                                ],
                                "status": {"code": 1},
                            },
                            {
                                "traceId": "aaaa",
                                "spanId": "child_span_2",
                                "parentSpanId": "root_span_01",
                                "name": "ChildB.run",
                                "kind": 1,
                                "startTimeUnixNano": "5000000000",
                                "endTimeUnixNano": "8000000000",
                                "attributes": [
                                    {
                                        "key": "openinference.span.kind",
                                        "value": {"stringValue": "AGENT"},
                                    },
                                    {"key": "agent.name", "value": {"stringValue": "ChildB"}},
                                    {"key": "session.id", "value": {"stringValue": "multi-sess"}},
                                ],
                                "status": {"code": 1},
                            },
                            {
                                "traceId": "aaaa",
                                "spanId": "llm_span_002",
                                "parentSpanId": "child_span_2",
                                "name": "acompletion",
                                "kind": 3,
                                "startTimeUnixNano": "5500000000",
                                "endTimeUnixNano": "7500000000",
                                "attributes": [
                                    {
                                        "key": "openinference.span.kind",
                                        "value": {"stringValue": "LLM"},
                                    },
                                    {"key": "session.id", "value": {"stringValue": "multi-sess"}},
                                ],
                                "status": {"code": 1},
                            },
                        ],
                    }
                ],
            }
        ],
    }


class TestFilteredSpanLoading:
    """Test loading only spans for a specific sub-tree."""

    def test_get_descendant_spans_returns_subtree(self, temp_db):
        """Should return only spans that are descendants of a given span_id."""
        with patch.object(otlp_store, "DB_PATH", temp_db):
            spans = otlp_store.get_descendant_spans("multi-sess", "child_span_1")
        # Should include child_span_1 itself and its LLM child
        span_ids = [s.get("spanId") for s in spans]
        assert "child_span_1" in span_ids
        assert "llm_span_001" in span_ids
        # Should NOT include spans from the other subtree
        assert "child_span_2" not in span_ids
        assert "llm_span_002" not in span_ids

    def test_get_descendant_spans_root_returns_all(self, temp_db):
        """Getting descendants of root should return all spans."""
        with patch.object(otlp_store, "DB_PATH", temp_db):
            spans = otlp_store.get_descendant_spans("multi-sess", "root_span_01")
        assert len(spans) == 5  # root + 2 children + 2 LLM

    def test_get_descendant_spans_leaf_returns_just_itself(self, temp_db):
        """A leaf span has no children, returns only itself."""
        with patch.object(otlp_store, "DB_PATH", temp_db):
            spans = otlp_store.get_descendant_spans("multi-sess", "llm_span_001")
        assert len(spans) == 1
        assert spans[0].get("spanId") == "llm_span_001"

    def test_get_descendant_spans_unknown_span_returns_empty(self, temp_db):
        """Unknown span_id returns empty list."""
        with patch.object(otlp_store, "DB_PATH", temp_db):
            spans = otlp_store.get_descendant_spans("multi-sess", "nonexistent")
        assert spans == []
