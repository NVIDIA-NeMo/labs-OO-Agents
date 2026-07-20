# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MCPTool._call_tool 401 auto-refresh + flattened errors."""

import httpx
import pytest

from nooa.mcp.tool import MCPTool


def _http_401() -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "https://example/mcp")
    resp = httpx.Response(401, request=req)
    return httpx.HTTPStatusError("401", request=req, response=resp)


class _FakeSession:
    def __init__(self, fail_times: int):
        self._fail_times = fail_times

    async def call_tool(self, name, args):
        if self._fail_times > 0:
            self._fail_times -= 1
            raise ExceptionGroup("tg", [_http_401()])

        class _R:
            content = [type("C", (), {"text": "ok"})()]

        return _R()


class _FakeClient:
    def __init__(self, fail_times: int):
        self._session = _FakeSession(fail_times)

    def connect_to_server(self):
        session = self._session

        class _CM:
            async def __aenter__(self):
                return session

            async def __aexit__(self, *a):
                return False

        return _CM()


def _make_tool(client, refresh_ctx=None):
    t = object.__new__(MCPTool)
    t.__init__(client, "test-server", refresh_ctx=refresh_ctx)
    return t


@pytest.mark.asyncio
async def test_success_no_refresh():
    tool = _make_tool(_FakeClient(fail_times=0))
    assert await tool._call_tool("x", {}) == "ok"


@pytest.mark.asyncio
async def test_401_without_refresh_ctx_raises_actionable():
    tool = _make_tool(_FakeClient(fail_times=1), refresh_ctx=None)
    with pytest.raises(RuntimeError, match="401 Unauthorized"):
        await tool._call_tool("x", {})


@pytest.mark.asyncio
async def test_401_with_refresh_retries_once(monkeypatch):
    # First call 401s; after a successful refresh the retry succeeds.
    client = _FakeClient(fail_times=1)
    tool = _make_tool(client, refresh_ctx={"server_url": "https://example/mcp"})

    async def fake_refresh():
        return True  # pretend the token refreshed; same fake client now succeeds

    monkeypatch.setattr(tool, "_refresh_access_token", fake_refresh)
    assert await tool._call_tool("x", {}) == "ok"


@pytest.mark.asyncio
async def test_401_refresh_fails_raises_actionable(monkeypatch):
    tool = _make_tool(_FakeClient(fail_times=2), refresh_ctx={"server_url": "https://example/mcp"})

    async def fake_refresh():
        return False

    monkeypatch.setattr(tool, "_refresh_access_token", fake_refresh)
    with pytest.raises(RuntimeError, match="re-authenticate|Re-authenticate"):
        await tool._call_tool("x", {})
