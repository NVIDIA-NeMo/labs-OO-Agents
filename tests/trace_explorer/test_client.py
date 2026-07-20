# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for trace explorer thin-client path (explorer_routes + client)."""

from unittest.mock import patch

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from nooa.trace_explorer.client import TraceExplorerClient

# =============================================================================
# Fixtures
# =============================================================================


def _make_otlp_spans() -> list[dict]:
    """Create minimal OTLP-format spans as returned by otlp_store.get_session_spans."""
    agent_span = {
        "traceId": "trace001",
        "spanId": "aabbccdd11223344",
        "name": "TestAgent.handle",
        "kind": 1,
        "startTimeUnixNano": "1000000000",
        "endTimeUnixNano": "2000000000",
        "attributes": [
            {"key": "openinference.span.kind", "value": {"stringValue": "AGENT"}},
            {"key": "agent.name", "value": {"stringValue": "TestAgent"}},
            {"key": "agent.method", "value": {"stringValue": "handle"}},
            {"key": "agent.call_id", "value": {"stringValue": "call_001"}},
        ],
        "status": {"code": 1},
        "events": [],
        "_resource": {},
    }
    return [agent_span]


def _make_otlp_spans_with_reasoning() -> list[dict]:
    """Create a minimal agent/generation/LLM trace with output reasoning."""
    spans = _make_otlp_spans()
    generation_span = {
        "traceId": "trace001",
        "spanId": "gen001",
        "parentSpanId": "aabbccdd11223344",
        "name": "generation",
        "kind": 1,
        "startTimeUnixNano": "1100000000",
        "endTimeUnixNano": "1900000000",
        "attributes": [
            {"key": "openinference.span.kind", "value": {"stringValue": "AGENT"}},
            {"key": "generation.id", "value": {"stringValue": "gen001abcdef"}},
            {"key": "agent.name", "value": {"stringValue": "TestAgent"}},
            {"key": "agent.method", "value": {"stringValue": "handle"}},
            {"key": "agent.call_id", "value": {"stringValue": "call_001"}},
        ],
        "status": {"code": 1},
        "events": [],
        "_resource": {},
    }
    llm_span = {
        "traceId": "trace001",
        "spanId": "llm001",
        "parentSpanId": "gen001",
        "name": "acompletion",
        "kind": 1,
        "startTimeUnixNano": "1200000000",
        "endTimeUnixNano": "1300000000",
        "attributes": [
            {"key": "openinference.span.kind", "value": {"stringValue": "LLM"}},
            {"key": "llm.model_name", "value": {"stringValue": "test-model"}},
            {"key": "llm.input_messages.0.message.role", "value": {"stringValue": "user"}},
            {"key": "llm.input_messages.0.message.content", "value": {"stringValue": "solve"}},
            {"key": "llm.output_messages.0.message.content", "value": {"stringValue": "final"}},
            {
                "key": "llm.output_messages.0.message.reasoning_content",
                "value": {"stringValue": "server-side reasoning"},
            },
        ],
        "status": {"code": 1},
        "events": [],
        "_resource": {},
    }
    return spans + [generation_span, llm_span]


@pytest.fixture
def app():
    """Create a test FastAPI app with explorer routes."""
    from fastapi import FastAPI

    from nooa.viewer.explorer_routes import router

    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def mock_otlp_store():
    """Mock otlp_store functions used by explorer_routes."""
    from nooa.viewer.explorer_routes import clear_explorer_cache

    clear_explorer_cache()
    with patch("nooa.viewer.explorer_routes.otlp_store") as mock_store:
        mock_store.session_exists.return_value = True
        mock_store.get_session_spans.return_value = _make_otlp_spans()
        yield mock_store
    clear_explorer_cache()


# =============================================================================
# Server-side route tests
# =============================================================================


@pytest.mark.asyncio
async def test_overview_endpoint(app, mock_otlp_store):
    """Test /api/explorer/overview returns a text result."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            "/api/explorer/overview",
            params={"session_id": "test-session"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "result" in data
    assert isinstance(data["result"], str)
    assert len(data["result"]) > 0


@pytest.mark.asyncio
async def test_overview_endpoint_session_not_found(app, mock_otlp_store):
    """Test 404 when session doesn't exist."""
    mock_otlp_store.session_exists.return_value = False
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            "/api/explorer/overview",
            params={"session_id": "nonexistent"},
        )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_errors_endpoint(app, mock_otlp_store):
    """Test /api/explorer/errors endpoint."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            "/api/explorer/errors",
            params={"session_id": "test-session"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "result" in data


@pytest.mark.asyncio
async def test_search_endpoint(app, mock_otlp_store):
    """Test /api/explorer/search endpoint."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            "/api/explorer/search",
            params={"session_id": "test-session", "pattern": "TestAgent"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "result" in data


@pytest.mark.asyncio
async def test_session_list_endpoint(app, mock_otlp_store):
    """Test /api/explorer/session-list endpoint."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            "/api/explorer/session-list",
            params={"session_id": "test-session"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "sessions" in data
    assert isinstance(data["sessions"], list)


@pytest.mark.asyncio
async def test_turn_endpoint(app, mock_otlp_store):
    """Test /api/explorer/turn endpoint."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            "/api/explorer/turn",
            params={
                "session_id": "test-session",
                "target_session_id": "aabbcc",
                "turn_index": 0,
            },
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "result" in data


@pytest.mark.asyncio
async def test_timeline_endpoint(app, mock_otlp_store):
    """Test /api/explorer/timeline endpoint."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            "/api/explorer/timeline",
            params={"session_id": "test-session"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "result" in data


@pytest.mark.asyncio
async def test_first_error_endpoint(app, mock_otlp_store):
    """Test /api/explorer/first-error endpoint."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            "/api/explorer/first-error",
            params={"session_id": "test-session"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "result" in data


@pytest.mark.asyncio
async def test_eval_context_endpoint(app, mock_otlp_store):
    """Test /api/explorer/eval-context endpoint."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            "/api/explorer/eval-context",
            params={"session_id": "test-session"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "result" in data


# =============================================================================
# Client tests
# =============================================================================


@pytest.mark.asyncio
async def test_client_connection_error():
    """Test that client raises ConnectionError when server is unreachable."""
    client = TraceExplorerClient("http://localhost:19999", "test-session", timeout=2.0)
    with pytest.raises(ConnectionError):
        await client.get_overview()


@pytest.mark.asyncio
async def test_client_help():
    """Test that help() returns usage text without network calls."""
    client = TraceExplorerClient("http://localhost:5001", "test-session")
    result = await client.help()
    assert "get_overview" in result
    assert "get_session" in result


@pytest.mark.asyncio
async def test_client_repr():
    """Test client repr."""
    client = TraceExplorerClient("http://localhost:5001", "my-session")
    assert "localhost:5001" in repr(client)
    assert "my-session" in repr(client)


@pytest.mark.asyncio
async def test_client_bypasses_env_proxy(monkeypatch):
    """A remote viewer must be reached directly even when HTTP(S)_PROXY is set.

    Without trust_env=False the request would be routed through the env proxy
    (the sandbox proxy), which times out against internal viewers. We assert the
    transport actually chosen for the viewer URL is the client's direct
    transport, not a proxy mount.
    """
    monkeypatch.setenv("HTTP_PROXY", "http://blackhole.invalid:3128")
    monkeypatch.setenv("HTTPS_PROXY", "http://blackhole.invalid:3128")

    client = TraceExplorerClient("http://viewer.internal:5001", "test-session")

    async with httpx.AsyncClient(timeout=client._timeout, trust_env=False) as h:
        url = httpx.URL("http://viewer.internal:5001/api/explorer/overview")
        assert h._transport_for_url(url) is h._transport, (
            "viewer request must go direct, not through the env proxy"
        )


@pytest.mark.asyncio
async def test_session_endpoint_include_reasoning_flag(app, mock_otlp_store):
    """Session endpoint includes reasoning by default and hides it when requested."""
    mock_otlp_store.get_session_spans.return_value = _make_otlp_spans_with_reasoning()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            "/api/explorer/session",
            params={"session_id": "test-session", "target_session_id": "aabbcc"},
        )
        hidden = await client.get(
            "/api/explorer/session",
            params={
                "session_id": "test-session",
                "target_session_id": "aabbcc",
                "include_reasoning": False,
            },
        )

    assert resp.status_code == 200
    assert "server-side reasoning" in resp.json()["result"]
    assert hidden.status_code == 200
    assert "server-side reasoning" not in hidden.json()["result"]


@pytest.mark.asyncio
async def test_turn_endpoint_include_reasoning_flag(app, mock_otlp_store):
    """Turn endpoint includes reasoning by default and hides it when requested."""
    mock_otlp_store.get_session_spans.return_value = _make_otlp_spans_with_reasoning()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            "/api/explorer/turn",
            params={
                "session_id": "test-session",
                "target_session_id": "aabbcc",
                "turn_index": 0,
            },
        )
        hidden = await client.get(
            "/api/explorer/turn",
            params={
                "session_id": "test-session",
                "target_session_id": "aabbcc",
                "turn_index": 0,
                "include_reasoning": False,
            },
        )

    assert resp.status_code == 200
    assert "server-side reasoning" in resp.json()["result"]
    assert hidden.status_code == 200
    assert "server-side reasoning" not in hidden.json()["result"]


@pytest.mark.asyncio
async def test_client_passes_include_reasoning_params(monkeypatch):
    """Thin client forwards include_reasoning to session/turn endpoints."""
    seen: list[tuple[str, dict]] = []

    async def fake_get(self, url, params=None):
        seen.append((str(url), dict(params or {})))
        request = httpx.Request("GET", url, params=params)
        return httpx.Response(200, json={"result": "ok"}, request=request)

    monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)
    client = TraceExplorerClient("http://viewer", "viewer-session")

    await client.get_session("aabbcc", include_reasoning=False)
    await client.get_turn("aabbcc", 0, include_reasoning=False)
    await client.get_session_fast("aabbcc", "span1", include_reasoning=False)
    await client.get_turn_fast("aabbcc", "span1", 0, include_reasoning=False)

    assert [params["include_reasoning"] for _, params in seen] == [False, False, False, False]
    assert seen[0][1]["target_session_id"] == "aabbcc"
    assert seen[1][1]["turn_index"] == 0
    assert seen[2][1]["span_id"] == "span1"
    assert seen[3][1]["span_id"] == "span1"
