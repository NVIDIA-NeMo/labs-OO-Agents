# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCP (Model Context Protocol) tool integration for NVIDIA OO Agents.

This package provides tools for connecting to MCP servers and using their tools
as agent capabilities.
"""

from .client import (
    MCPBaseClient,
    MCPSSEClient,
    MCPStdioClient,
    MCPStreamableHTTPClient,
    create_mcp_client,
)
from .oauth import OAuthConfig, OAuthHandler, OAuthToken, handle_mcp_oauth
from .tool import MCPManager, MCPTool, MCPToolSpec

__all__ = [
    # Client
    "MCPBaseClient",
    "MCPSSEClient",
    "MCPStdioClient",
    "MCPStreamableHTTPClient",
    "create_mcp_client",
    # OAuth
    "OAuthConfig",
    "OAuthHandler",
    "OAuthToken",
    "handle_mcp_oauth",
    # Tool
    "MCPTool",
    "MCPManager",
    "MCPToolSpec",
]
