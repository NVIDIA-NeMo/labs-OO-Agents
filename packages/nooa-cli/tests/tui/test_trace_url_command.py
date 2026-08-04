# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for /trace-url slash command."""

import os
from unittest.mock import MagicMock, patch

import pytest
from nooa_cli.tui.commands import TraceUrlCommand


@pytest.fixture
def cmd():
    """Create a TraceUrlCommand with minimal mocks."""
    frontend = MagicMock()
    config = MagicMock()
    agent = MagicMock()
    return TraceUrlCommand(frontend, config, agent)


@pytest.mark.asyncio
async def test_trace_url_no_tracing_package(cmd):
    """Returns error when tracing package is not installed."""
    with patch.dict("sys.modules", {"nooa.tracing": None}):
        with patch("builtins.__import__", side_effect=ImportError):
            result = await cmd.execute([])
    assert not result.success


@pytest.mark.asyncio
async def test_trace_url_no_active_session(cmd):
    """Returns error when no trace session is active."""
    with patch("nooa.tracing.get_session", return_value=None):
        result = await cmd.execute([])
    assert not result.success
    assert "No active trace session" in result.outputs[0].content


@pytest.mark.asyncio
async def test_trace_url_default_endpoint(cmd):
    """Constructs URL with default localhost:5001 endpoint."""
    with patch("nooa.tracing.get_session", return_value="tui-20250508-120000-abcdef12"):
        with patch.dict(os.environ, {}, clear=False):
            # Remove OTLP_ENDPOINT if set
            env = os.environ.copy()
            env.pop("OTLP_ENDPOINT", None)
            with patch.dict(os.environ, env, clear=True):
                result = await cmd.execute([])
    assert result.success
    assert (
        "http://localhost:5001/traces/view?session_id=tui-20250508-120000-abcdef12"
        in result.outputs[0].content
    )


@pytest.mark.asyncio
async def test_trace_url_custom_endpoint(cmd):
    """Constructs URL stripping /v1/traces from OTLP_ENDPOINT."""
    with patch("nooa.tracing.get_session", return_value="tui-20250508-120000-abcdef12"):
        with patch.dict(os.environ, {"OTLP_ENDPOINT": "http://myviewer:8080/v1/traces"}):
            result = await cmd.execute([])
    assert result.success
    assert (
        "http://myviewer:8080/traces/view?session_id=tui-20250508-120000-abcdef12"
        in result.outputs[0].content
    )


@pytest.mark.asyncio
async def test_trace_url_endpoint_with_v1_only(cmd):
    """Strips /v1 suffix from endpoint."""
    with patch("nooa.tracing.get_session", return_value="my-session"):
        with patch.dict(os.environ, {"OTLP_ENDPOINT": "http://host:9000/v1"}):
            result = await cmd.execute([])
    assert result.success
    assert "http://host:9000/traces/view?session_id=my-session" in result.outputs[0].content


@pytest.mark.asyncio
async def test_trace_url_endpoint_no_suffix(cmd):
    """Endpoint without /v1/traces suffix is used as-is."""
    with patch("nooa.tracing.get_session", return_value="my-session"):
        with patch.dict(os.environ, {"OTLP_ENDPOINT": "http://host:9000"}):
            result = await cmd.execute([])
    assert result.success
    assert "http://host:9000/traces/view?session_id=my-session" in result.outputs[0].content
