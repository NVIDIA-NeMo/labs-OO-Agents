# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests against real local MCP transports.

These tests use the bundled wiki server and require no external services,
credentials, or network access beyond a loopback socket.
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from importlib import import_module
from pathlib import Path

import pytest

pytest.importorskip("mcp")

MCPManager = import_module("nooa.mcp").MCPManager


REPO_ROOT = Path(__file__).resolve().parents[2]
MCP_CONFIG = REPO_ROOT / "examples" / "quickstart" / ".mcp.json"
WIKI_SERVER = REPO_ROOT / "examples" / "assets" / "wiki_mcp_server.py"

HTTP_SERVER_BOOTSTRAP = """
import sys

from examples.assets.wiki_mcp_server import mcp

mcp.settings.host = "127.0.0.1"
mcp.settings.port = int(sys.argv[1])
mcp.run(transport="streamable-http")
"""


def _wait_for_server(process: subprocess.Popen[str], port: int) -> None:
    """Wait until the child accepts loopback connections or fail with its logs."""
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


def _stop_server(process: subprocess.Popen[str]) -> None:
    """Stop the local server and ensure no child survives the test."""
    try:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    finally:
        if process.stdout is not None:
            process.stdout.close()


@pytest.fixture
def wiki_http_url(unused_tcp_port: int) -> Iterator[str]:
    """Run the bundled wiki MCP server over streamable HTTP on loopback."""
    process = subprocess.Popen(
        [sys.executable, "-c", HTTP_SERVER_BOOTSTRAP, str(unused_tcp_port)],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_server(process, unused_tcp_port)
        yield f"http://127.0.0.1:{unused_tcp_port}/mcp"
    finally:
        _stop_server(process)


@pytest.mark.asyncio
async def test_stdio_transport_discovers_and_invokes_real_server() -> None:
    """MCPManager can discover and call a tool through a real stdio subprocess."""
    wiki = MCPManager.create_from_server(
        "wiki",
        command=sys.executable,
        args=[str(WIKI_SERVER)],
        mcp_file=MCP_CONFIG,
    )

    result = await wiki.search_wiki(query="deployment")

    assert "Deployment Best Practices" in result


@pytest.mark.asyncio
async def test_streamable_http_transport_discovers_and_invokes_real_server(
    wiki_http_url: str,
) -> None:
    """MCPManager can discover and call a tool through real loopback HTTP."""
    wiki = MCPManager.create_from_server(
        "wiki-http",
        url=wiki_http_url,
        transport="streamable-http",
    )

    result = await wiki.search_wiki(query="code review")

    assert "Code Review Checklist" in result
