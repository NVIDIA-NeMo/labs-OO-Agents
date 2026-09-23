# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WorkspaceSettings.remember_mcp/forget_mcp: credential handling and settings scope."""

from types import SimpleNamespace

import pytest
import yaml
from nooa_cli.interactive.mcp_registry import MCPRegistry
from nooa_cli.interactive.options import SessionOptions
from nooa_cli.interactive.workspace_settings import WorkspaceSettings


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
