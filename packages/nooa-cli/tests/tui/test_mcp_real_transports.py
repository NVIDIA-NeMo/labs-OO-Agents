# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TUI-registry integration tests against real local MCP transports."""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from nooa_cli.tui.mcp_approval import MCPApprovalRequired
from nooa_cli.tui.mcp_registry import MCPRegistry

from nooa.runtime.context_manager import ContextManager

pytest.importorskip("mcp")

SERVER_SOURCE = '''
import os
import sys

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("tui-approval-integration")
last_authorization = "missing"
last_path = "missing"

@mcp.tool()
def probe(value: str) -> str:
    """Echo a value and the fake test token visible to this server."""
    return f"{value}:{os.environ.get('MCP_INTEGRATION_TOKEN', 'missing')}"

@mcp.tool()
def http_probe(value: str) -> str:
    """Echo the latest HTTP authorization header and original request path."""
    return f"{value}:{last_authorization}:{last_path}"

if len(sys.argv) > 1:
    import uvicorn

    inner = mcp.streamable_http_app()

    class CaptureAndRewritePath:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            global last_authorization, last_path
            if scope["type"] == "http":
                last_path = scope["path"]
                headers = {
                    key.decode("latin-1").lower(): value.decode("latin-1")
                    for key, value in scope.get("headers", [])
                }
                last_authorization = headers.get("authorization", "missing")
                scope = dict(scope)
                scope["path"] = "/mcp"
                scope["raw_path"] = b"/mcp"
            await self.app(scope, receive, send)

    uvicorn.run(CaptureAndRewritePath(inner), host="127.0.0.1", port=int(sys.argv[1]))
else:
    mcp.run(transport="stdio")
'''


class _Agent:
    def __init__(self) -> None:
        self.context_manager = ContextManager()


def _registry(tmp_path: Path, servers: dict) -> tuple[MCPRegistry, _Agent]:
    registry = MCPRegistry(
        mcp_file=tmp_path / "absent.mcp.json",
        servers=servers,
        approval_path=tmp_path / "approvals.json",
    )
    agent = _Agent()
    registry.attach(agent)
    agent.mcp = registry
    return registry, agent


def _approve(registry: MCPRegistry, name: str) -> None:
    request = registry._approval_request(name)
    registry._approve(name, request.confirmation)


def _wait_for_server(process: subprocess.Popen[str], port: int) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.communicate()[0]
            pytest.fail(f"MCP HTTP server exited during startup:\n{output}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.05)
    pytest.fail("MCP HTTP server did not start within 10 seconds")


@pytest.fixture
def mcp_server_script(tmp_path: Path) -> Path:
    script = tmp_path / "server.py"
    script.write_text(SERVER_SOURCE)
    return script


@pytest.fixture
def mcp_http_url(mcp_server_script: Path, unused_tcp_port: int) -> Iterator[str]:
    process = subprocess.Popen(
        [sys.executable, str(mcp_server_script), str(unused_tcp_port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_server(process, unused_tcp_port)
        yield f"http://127.0.0.1:{unused_tcp_port}/mcp"
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process.stdout is not None:
            process.stdout.close()


@pytest.mark.asyncio
async def test_approved_stdio_config_discovers_and_invokes_real_server(
    tmp_path: Path, mcp_server_script: Path, monkeypatch
) -> None:
    canary = "stdio-canary"
    monkeypatch.setenv("MCP_INTEGRATION_TOKEN", canary)
    registry, agent = _registry(
        tmp_path,
        {
            "local": {
                "command": sys.executable,
                "args": [str(mcp_server_script)],
                "env": {"MCP_INTEGRATION_TOKEN": "${MCP_INTEGRATION_TOKEN}"},
            }
        },
    )

    with pytest.raises(MCPApprovalRequired):
        await registry.connect(["local"])
    _approve(registry, "local")
    assert await registry.connect(["local"]) == ["local"]

    assert await agent.local.probe(value="stdio-ok") == f"stdio-ok:{canary}"


@pytest.mark.asyncio
async def test_approved_http_config_discovers_and_invokes_real_server(
    tmp_path: Path, mcp_http_url: str, monkeypatch
) -> None:
    canary = "http-canary"
    monkeypatch.setenv("MCP_HTTP_TOKEN", canary)
    registry, agent = _registry(
        tmp_path,
        {
            "loopback": {
                "url": f"{mcp_http_url}/${{MCP_HTTP_TOKEN}}",
                "transport": "streamable-http",
                "headers": {"Authorization": "Bearer ${MCP_HTTP_TOKEN}"},
            }
        },
    )

    with pytest.raises(MCPApprovalRequired):
        await registry.connect(["loopback"])
    _approve(registry, "loopback")
    assert await registry.connect(["loopback"]) == ["loopback"]

    assert await agent.loopback.http_probe(value="http-ok") == (
        f"http-ok:Bearer {canary}:/mcp/{canary}"
    )


@pytest.mark.asyncio
async def test_approved_value_is_not_reexpanded_by_installed_core(
    tmp_path: Path, mcp_server_script: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OUTER_VALUE", "${NESTED_SECRET}")
    monkeypatch.setenv("NESTED_SECRET", "must-not-reach-child")
    registry, agent = _registry(
        tmp_path,
        {
            "local": {
                "command": sys.executable,
                "args": [str(mcp_server_script)],
                "env": {"MCP_INTEGRATION_TOKEN": "${OUTER_VALUE}"},
            }
        },
    )

    _approve(registry, "local")
    assert await registry.connect(["local"]) == ["local"]

    assert await agent.local.probe(value="nested") == "nested:${NESTED_SECRET}"
