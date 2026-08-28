# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F403,F405
"""Quickstart 11: MCP tools — connect external MCP servers as agent tools.

Requires: uv sync --extra mcp

MCP servers are configured in a ``.mcp.json`` file (same format as VS Code /
Claude Code).  ``MCPManager.create_from_server("name")`` looks the server up
by name, connects, and exposes its tools to the agent.

This example ships a small wiki MCP server (``examples/assets/wiki_mcp_server.py``)
and a companion ``.mcp.json`` that registers it under the name ``"wiki"``.

Usage:
    uv run python examples/quickstart/11_mcp.py
"""

from pathlib import Path

try:
    from nooa.mcp import MCPManager, MCPTool
except ImportError as e:
    raise ImportError("nooa[mcp] not installed. Run: uv sync --extra mcp") from e

from nooa.util.quickstart import *

# .mcp.json lives next to this script — tells MCPManager how to launch "wiki"
MCP_CONFIG = Path(__file__).parent / ".mcp.json"
# Resolve the server script path so the example works from any working directory
WIKI_SERVER = str(Path(__file__).resolve().parent.parent / "assets" / "wiki_mcp_server.py")


class WikiAgent(Agent, llm=llm):
    """Agent with MCP tool access to an internal wiki."""

    wiki: MCPTool

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Each MCPTool owns connection state, so keep it per agent instance.
        self.wiki = MCPManager.create_from_server("wiki", mcp_file=MCP_CONFIG, args=[WIKI_SERVER])

    async def respond(self, prompt: str) -> str:
        """Answer the user's question using the wiki search tool.

        Always search the wiki first, then synthesize an answer from the results.
        """
        ...


@autorun
async def main():
    agent = WikiAgent()
    result = await agent.respond("What are the best practices for deployment and testing?")
    print(result)
