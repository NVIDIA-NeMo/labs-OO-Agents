# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for incremental (subtree-only) session and turn loading."""

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient


def _make_multi_agent_otlp_spans():
    """OTLP spans with a root AGENT, two child AGENTs, and LLM/exec spans under each."""
    return [
        {
            "traceId": "t1",
            "spanId": "root_agent_01",
            "name": "Router.handle",
            "kind": 1,
            "startTimeUnixNano": "1000000000",
            "endTimeUnixNano": "9000000000",
            "attributes": [
                {"key": "openinference.span.kind", "value": {"stringValue": "AGENT"}},
                {"key": "agent.name", "value": {"stringValue": "Router"}},
                {"key": "agent.method", "value": {"stringValue": "handle"}},
                {"key": "agent.call_id", "value": {"stringValue": "call_root"}},
            ],
            "status": {"code": 1},
            "events": [],
            "_resource": {},
        },
        {
            "traceId": "t1",
            "spanId": "child_agent_a",
            "parentSpanId": "root_agent_01",
            "name": "Worker.run",
            "kind": 1,
            "startTimeUnixNano": "2000000000",
            "endTimeUnixNano": "4000000000",
            "attributes": [
                {"key": "openinference.span.kind", "value": {"stringValue": "AGENT"}},
                {"key": "agent.name", "value": {"stringValue": "Worker"}},
                {"key": "agent.method", "value": {"stringValue": "run"}},
                {"key": "agent.call_id", "value": {"stringValue": "call_a"}},
            ],
            "status": {"code": 1},
            "events": [],
            "_resource": {},
        },
        {
            "traceId": "t1",
            "spanId": "child_agent_b",
            "parentSpanId": "root_agent_01",
            "name": "Analyzer.analyze",
            "kind": 1,
            "startTimeUnixNano": "5000000000",
            "endTimeUnixNano": "8000000000",
            "attributes": [
                {"key": "openinference.span.kind", "value": {"stringValue": "AGENT"}},
                {"key": "agent.name", "value": {"stringValue": "Analyzer"}},
                {"key": "agent.method", "value": {"stringValue": "analyze"}},
                {"key": "agent.call_id", "value": {"stringValue": "call_b"}},
            ],
            "status": {"code": 1},
            "events": [],
            "_resource": {},
        },
    ]


@pytest.fixture
def app():
    from fastapi import FastAPI

    from nooa.viewer.explorer_routes import clear_explorer_cache, router

    clear_explorer_cache()
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def mock_store():
    with patch("nooa.viewer.explorer_routes.otlp_store") as m:
        m.session_exists.return_value = True
        m.get_session_spans.return_value = _make_multi_agent_otlp_spans()
        # For the fast path: get_agent_spans returns just AGENT spans
        m.get_agent_spans.return_value = [
            {
                "spanId": "root_agent_01",
                "name": "Router.handle",
                "parentSpanId": None,
                "attributes": [
                    {"key": "openinference.span.kind", "value": {"stringValue": "AGENT"}}
                ],
                "status": {"code": 1},
                "startTimeUnixNano": "1000000000",
                "endTimeUnixNano": "9000000000",
                "traceId": "t1",
                "kind": 1,
            },
            {
                "spanId": "child_agent_a",
                "name": "Worker.run",
                "parentSpanId": "root_agent_01",
                "attributes": [
                    {"key": "openinference.span.kind", "value": {"stringValue": "AGENT"}}
                ],
                "status": {"code": 1},
                "startTimeUnixNano": "2000000000",
                "endTimeUnixNano": "4000000000",
                "traceId": "t1",
                "kind": 1,
            },
            {
                "spanId": "child_agent_b",
                "name": "Analyzer.analyze",
                "parentSpanId": "root_agent_01",
                "attributes": [
                    {"key": "openinference.span.kind", "value": {"stringValue": "AGENT"}}
                ],
                "status": {"code": 1},
                "startTimeUnixNano": "5000000000",
                "endTimeUnixNano": "8000000000",
                "traceId": "t1",
                "kind": 1,
            },
        ]

        # get_descendant_spans returns spans for a specific subtree
        def _descendant_spans(session_id, span_id):
            all_spans = _make_multi_agent_otlp_spans()
            if span_id == "child_agent_a":
                return [s for s in all_spans if s["spanId"] in ("child_agent_a",)]
            elif span_id == "child_agent_b":
                return [s for s in all_spans if s["spanId"] in ("child_agent_b",)]
            elif span_id == "root_agent_01":
                return all_spans
            return []

        m.get_descendant_spans.side_effect = _descendant_spans
        yield m


class TestIncrementalSessionLoading:
    """Test /api/explorer/session-fast endpoint."""

    @pytest.mark.asyncio
    async def test_session_fast_loads_only_subtree(self, app, mock_store):
        """Fast session endpoint should use get_descendant_spans, not get_session_spans."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get(
                "/api/explorer/session-fast",
                params={
                    "session_id": "my-session",
                    "target_session_id": "child_",  # prefix match for child_agent_a
                    "span_id": "child_agent_a",
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "result" in data
        # Should have used get_descendant_spans, NOT get_session_spans
        mock_store.get_descendant_spans.assert_called()
        mock_store.get_session_spans.assert_not_called()

    @pytest.mark.asyncio
    async def test_session_fast_404_unknown_span(self, app, mock_store):
        """Should 404 if span_id not found."""
        mock_store.get_descendant_spans.side_effect = None
        mock_store.get_descendant_spans.return_value = []
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get(
                "/api/explorer/session-fast",
                params={
                    "session_id": "my-session",
                    "target_session_id": "nope",
                    "span_id": "nonexistent",
                },
            )
        assert resp.status_code == 404


class TestIncrementalTurnLoading:
    """Test /api/explorer/turn-fast endpoint."""

    @pytest.mark.asyncio
    async def test_turn_fast_loads_only_subtree(self, app, mock_store):
        """Fast turn endpoint should use get_descendant_spans."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get(
                "/api/explorer/turn-fast",
                params={
                    "session_id": "my-session",
                    "target_session_id": "child_",
                    "span_id": "child_agent_a",
                    "turn_index": 0,
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "result" in data
        mock_store.get_descendant_spans.assert_called()
        mock_store.get_session_spans.assert_not_called()
