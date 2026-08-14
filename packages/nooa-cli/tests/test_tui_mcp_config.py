# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent MCP add/remove and user-owned approval command paths."""

from types import SimpleNamespace

import yaml
from nooa_cli.tui import settings
from nooa_cli.tui.commands import MCPCommand
from nooa_cli.tui.config import TUIConfig
from nooa_cli.tui.mcp_registry import MCPRegistry

from nooa.skill import get_slash_commands


def _project_settings(monkeypatch, tmp_path):
    path = tmp_path / ".nooa" / "settings.yaml"
    monkeypatch.setattr(settings, "settings_path", lambda scope="project": path)
    return path


def _command(registry):
    return MCPCommand(object(), TUIConfig(), SimpleNamespace(mcp=registry))


async def test_mcp_add_and_remove_round_trip_project_settings(monkeypatch, tmp_path):
    path = _project_settings(monkeypatch, tmp_path)
    registry = MCPRegistry(
        servers={"keep": {"url": "https://keep.example/mcp"}},
        approval_path=tmp_path / "approvals.json",
    )
    command = _command(registry)

    added = await command.execute(["add", "docs", "https://docs.example/mcp"])
    assert added.success
    assert "Added HTTP MCP server 'docs'" in added.outputs[0].content
    data = yaml.safe_load(path.read_text())
    assert data["tui"]["mcp_servers"]["docs"] == {
        "url": "https://docs.example/mcp",
        "transport": "streamable-http",
    }
    assert registry._is_approved("docs") is False

    removed = await command.execute(["remove", "docs"])
    assert removed.success
    data = yaml.safe_load(path.read_text())
    assert "docs" not in data.get("tui", {}).get("mcp_servers", {})
    assert registry._servers == {"keep": {"url": "https://keep.example/mcp"}}


async def test_mcp_add_stdio_command_requires_later_approval(monkeypatch, tmp_path):
    path = _project_settings(monkeypatch, tmp_path)
    registry = MCPRegistry(approval_path=tmp_path / "approvals.json")

    result = await _command(registry).execute(["add", "local", "my-mcp-server"])

    assert result.success
    assert "Review it with /mcp approve local" in result.outputs[0].content
    data = yaml.safe_load(path.read_text())
    assert data["tui"]["mcp_servers"]["local"] == {"command": "my-mcp-server"}
    assert registry._is_approved("local") is False


async def test_mcp_add_rejects_unsafe_name_before_persisting(monkeypatch, tmp_path):
    path = _project_settings(monkeypatch, tmp_path)
    registry = MCPRegistry(approval_path=tmp_path / "approvals.json")

    result = await _command(registry).execute(["add", "evil\x1b[2J", "https://docs.example/mcp"])

    assert result.success is False
    assert not path.exists()
    assert registry.discovered() == []


async def test_mcp_remove_does_not_edit_external_mcp_file(monkeypatch, tmp_path):
    _project_settings(monkeypatch, tmp_path)
    mcp_file = tmp_path / ".mcp.json"
    original = '{"mcpServers":{"external":{"url":"https://example/mcp"}}}'
    mcp_file.write_text(original)
    registry = MCPRegistry(
        mcp_file=mcp_file,
        approval_path=tmp_path / "approvals.json",
    )

    result = await _command(registry).execute(["remove", "external"])

    assert result.success is False
    assert "comes from" in result.outputs[0].content
    assert mcp_file.read_text() == original


async def test_mcp_approve_without_code_only_reviews_configuration(tmp_path):
    registry = MCPRegistry(
        servers={"docs": {"url": "https://docs.example/mcp"}},
        approval_path=tmp_path / "approvals.json",
    )

    result = await _command(registry).execute(["approve", "docs"])

    assert result.success
    assert "Config fingerprint: sha256:" in result.outputs[0].content
    assert "/mcp approve docs " in result.outputs[0].content
    assert registry._is_approved("docs") is False


def test_mcp_registry_exposes_only_agent_assisted_add_command():
    commands = [(meta.name, meta.output_to_agent) for meta, _ in get_slash_commands(MCPRegistry())]
    assert commands == [("mcp-add", True)]
