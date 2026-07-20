# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TraceExplorer server-side caching."""

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient


def _make_otlp_spans():
    """Minimal OTLP spans fixture."""
    return [
        {
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
    ]


@pytest.fixture
def app():
    from fastapi import FastAPI

    from nooa.viewer.explorer_routes import router

    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the explorer cache before each test."""
    from nooa.viewer.explorer_routes import clear_explorer_cache

    clear_explorer_cache()
    yield
    clear_explorer_cache()


@pytest.fixture
def mock_store():
    with patch("nooa.viewer.explorer_routes.otlp_store") as m:
        m.session_exists.return_value = True
        m.get_session_spans.return_value = _make_otlp_spans()
        yield m


class TestExplorerCache:
    """Tests for LRU caching of TraceExplorer instances."""

    @pytest.mark.asyncio
    async def test_repeated_calls_reuse_cached_explorer(self, app, mock_store):
        """Same session_id should only build TraceExplorer once."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.get("/api/explorer/overview", params={"session_id": "sess-1"})
            await client.get("/api/explorer/errors", params={"session_id": "sess-1"})

        # get_session_spans should only be called once (cached on second call)
        assert mock_store.get_session_spans.call_count == 1

    @pytest.mark.asyncio
    async def test_different_sessions_build_separate_explorers(self, app, mock_store):
        """Different session_ids should each build their own explorer."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.get("/api/explorer/overview", params={"session_id": "sess-1"})
            await client.get("/api/explorer/overview", params={"session_id": "sess-2"})

        assert mock_store.get_session_spans.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_has_max_size(self, app, mock_store):
        """Cache should evict oldest entries when full."""
        from nooa.viewer.explorer_routes import _explorer_cache

        # Fill cache beyond max size, verify it doesn't grow unbounded
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            for i in range(20):
                await client.get("/api/explorer/overview", params={"session_id": f"sess-{i}"})

        assert len(_explorer_cache) <= 16  # default max size

    @pytest.mark.asyncio
    async def test_cache_clear(self, app, mock_store):
        """Cache should be clearable."""
        from nooa.viewer.explorer_routes import clear_explorer_cache

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.get("/api/explorer/overview", params={"session_id": "sess-1"})

        clear_explorer_cache()

        from nooa.viewer.explorer_routes import _explorer_cache

        assert len(_explorer_cache) == 0
