# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MCP skills raise the installed SDK's error type, whichever SDK line is installed."""

import mcp.shared.exceptions as sdk_exceptions
from mcp.types import ErrorData
from nooa_atom.skills import mcp_auth, mcp_servers


def test_timeout_error_is_the_installed_sdks_exception_with_its_error_data():
    error = mcp_auth.mcp_error(ErrorData(code=408, message="timed out"))
    installed = getattr(sdk_exceptions, "McpError", None) or sdk_exceptions.MCPError
    assert isinstance(error, installed) and isinstance(error, Exception)
    assert error.error.code == 408 and error.error.message == "timed out"
    assert "timed out" in str(error)


def test_both_mcp_skill_modules_agree_on_the_error_class():
    assert mcp_servers.McpError is mcp_auth.McpError
