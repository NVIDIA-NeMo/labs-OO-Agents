# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WorkspaceSettings.remember_mcp/forget_mcp: credential handling and settings scope."""

from types import SimpleNamespace

import pytest
import yaml
from nooa_coder.interactive.mcp_registry import MCPRegistry
from nooa_coder.interactive.options import SessionOptions
from nooa_coder.interactive.workspace_settings import WorkspaceSettings


@pytest.fixture
def workspace_settings(tmp_path, monkeypatch):
    workspace = tmp_path / "project"
    workspace.mkdir()
    project = workspace / ".nooa"
    project.mkdir()
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    monkeypatch.delenv("NEMO_OO_PROJECT_DIR", raising=False)

    registry = MCPRegistry(
        mcp_file=tmp_path / ".mcp.json",
        approval_path=tmp_path / "approvals.json",
        watch_settings=True,
        project_dir=project,
    )
    options = SessionOptions(working_dir=str(workspace))
    ws = WorkspaceSettings(options)
    ws._agent = SimpleNamespace(mcp=registry)
    return ws, registry, workspace


async def test_remember_mcp_rejects_a_literal_secret_header(workspace_settings):
    ws, registry, workspace = workspace_settings
    registry.register(
        "leaky", url="https://example.com/mcp", headers={"Authorization": "Bearer sk-real-secret"}
    )
    with pytest.raises(ValueError, match="placeholder"):
        ws.remember_mcp("leaky")
    assert not (workspace / ".nooa" / "settings.yaml").exists()


async def test_remember_mcp_accepts_a_placeholder_only_header(workspace_settings):
    ws, registry, workspace = workspace_settings
    registry.register(
        "clean", url="https://example.com/mcp", headers={"Authorization": "${MY_TOKEN}"}
    )
    result = ws.remember_mcp("clean")
    assert "Saved" in result
    saved = yaml.safe_load((workspace / ".nooa" / "settings.yaml").read_text())
    assert saved["coding"]["mcp_servers"]["clean"]["headers"]["Authorization"] == "${MY_TOKEN}"


async def test_remember_mcp_does_not_leak_a_users_personal_auto_connect(
    workspace_settings, monkeypatch
):
    ws, registry, workspace = workspace_settings
    user_dir = workspace.parent / "user-config"
    user_dir.mkdir()
    (user_dir / "settings.yaml").write_text(
        yaml.safe_dump({"coding": {"mcp_auto_connect": ["personal-server"]}})
    )
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user_dir))

    registry.register("shared", command="shared-command")
    ws.remember_mcp("shared")

    saved = yaml.safe_load((workspace / ".nooa" / "settings.yaml").read_text())
    persisted = saved["coding"]["mcp_auto_connect"]
    assert "personal-server" not in persisted
    assert "shared" in persisted


async def test_forget_mcp_does_not_leak_a_users_personal_auto_connect(
    workspace_settings, monkeypatch
):
    ws, registry, workspace = workspace_settings
    registry.register("shared", command="shared-command")
    ws.remember_mcp("shared")

    user_dir = workspace.parent / "user-config"
    user_dir.mkdir()
    (user_dir / "settings.yaml").write_text(
        yaml.safe_dump({"coding": {"mcp_auto_connect": ["personal-server"]}})
    )
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user_dir))

    ws.forget_mcp("shared")

    saved = yaml.safe_load((workspace / ".nooa" / "settings.yaml").read_text())
    persisted = saved["coding"]["mcp_auto_connect"]
    assert "personal-server" not in persisted
    assert "shared" not in persisted


async def test_remember_mcp_after_forget_mcp_saves_the_definition_again(workspace_settings):
    """forget_mcp writes a null mask; a later remember_mcp must read past it."""
    ws, registry, workspace = workspace_settings
    registry.register("again", command="again-command")
    ws.remember_mcp("again")
    ws.forget_mcp("again")
    saved = yaml.safe_load((workspace / ".nooa" / "settings.yaml").read_text())
    assert saved["coding"]["mcp_servers"]["again"] is None

    registry.register("again", command="again-command")
    ws.remember_mcp("again")

    saved = yaml.safe_load((workspace / ".nooa" / "settings.yaml").read_text())
    assert saved["coding"]["mcp_servers"]["again"]["command"] == "again-command"
    assert saved["coding"]["mcp_auto_connect"] == ["again"]


async def test_remember_mcp_keeps_a_connected_server_connected(workspace_settings, monkeypatch):
    """Saving normalizes the definition (a bare url gains a transport).

    The registry must adopt the saved form before refreshing, or the refresh
    sees a "changed" definition and disconnects the live server.
    """
    ws, registry, workspace = workspace_settings
    registry.register("live", url="https://example.com/mcp")
    registry._connected["live"] = object()
    detached = []
    monkeypatch.setattr(registry, "deactivate", lambda names: detached.extend(names))
    monkeypatch.setattr(registry, "_detach", lambda name: detached.append(name))

    ws.remember_mcp("live")

    saved = yaml.safe_load((workspace / ".nooa" / "settings.yaml").read_text())
    assert saved["coding"]["mcp_servers"]["live"]["transport"] == "streamable-http"
    assert detached == []
    assert "live" in registry._connected
    assert registry._servers["live"] == saved["coding"]["mcp_servers"]["live"]


@pytest.mark.parametrize(
    "registration",
    [
        {"url": "https://user:hunter2@example.com/mcp"},
        {"url": "https://sk-live-token@example.com/mcp"},
        {"url": "https://example.com/mcp?api_key=sk-live-token"},
        {"url": "https://example.com/mcp?region=us&access_token=sk-live-token"},
        {"command": "server", "args": ["--token", "sk-live-token"]},
        {"command": "server", "args": ["--api-key=sk-live-token"]},
        {"command": "server", "args": ["--password", "hunter2"]},
    ],
)
async def test_remember_mcp_rejects_literal_credentials_in_url_and_args(
    workspace_settings, registration
):
    ws, registry, workspace = workspace_settings
    registry.register("leaky", **registration)
    with pytest.raises(ValueError, match="placeholder"):
        ws.remember_mcp("leaky")
    assert not (workspace / ".nooa" / "settings.yaml").exists()


@pytest.mark.parametrize(
    "registration",
    [
        {"url": "https://${MCP_USER}:${MCP_PASSWORD}@example.com/mcp"},
        {"url": "https://example.com/mcp?api_key=${MCP_KEY}&verbose"},
        {"url": "https://example.com/mcp?region=us-east&format=json"},
        {"command": "server", "args": ["--token", "${MCP_TOKEN}", "--port", "8080"]},
        {"command": "server", "args": ["--api-key=${MCP_KEY}", "--verbose"]},
        {"command": "server", "args": ["--tokenizer", "bpe"]},
    ],
)
async def test_remember_mcp_accepts_placeholders_and_ordinary_args(
    workspace_settings, registration
):
    ws, registry, workspace = workspace_settings
    registry.register("clean", **registration)
    assert "Saved" in ws.remember_mcp("clean")
