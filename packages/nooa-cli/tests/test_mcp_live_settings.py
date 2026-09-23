# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCP lifecycle commands observe saved definitions and approval changes."""

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest
from nooa_cli.interactive.controls import MCPControl
from nooa_cli.interactive.mcp_approval import MCPApprovalRequired
from nooa_cli.interactive.mcp_registry import MCPRegistry
from nooa_cli.interactive.options import SessionOptions

from nooa.mcp import MCPManager


@pytest.fixture
def registry(tmp_path, monkeypatch):
    project = tmp_path / ".nooa"
    project.mkdir()
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    monkeypatch.delenv("NEMO_OO_PROJECT_DIR", raising=False)
    return MCPRegistry(
        mcp_file=tmp_path / ".mcp.json",
        approval_path=tmp_path / "approvals.json",
        watch_settings=True,
        project_dir=project,
    )


def save(registry, command):
    servers = {"probe": {"command": command}} if command is not None else {}
    (registry._project_dir / "settings.yaml").write_text(
        json.dumps({"coding": {"mcp_servers": servers}})
    )


async def test_controls_see_saved_servers_and_preserve_transient_registrations(registry, tmp_path):
    registry.register("transient", command="transient-command")
    control = MCPControl(SimpleNamespace(mcp=registry), SessionOptions(), workspace=tmp_path)
    save(registry, "new-command")
    status = await control.run(["status"])
    assert status.success and "probe" in str(status) and "transient" in str(status)
    review = await control.run(["approve", "probe"])
    assert review.success
    assert registry._approval_request("probe").confirmation in str(review)
    assert "/mcp approve <name>" in await registry.mcp_add_command("probe info")


async def test_changed_definition_requires_reapproval_before_factory(registry, monkeypatch):
    save(registry, "old-command")
    registry.refresh_settings()
    old = registry._approval_request("probe")
    registry._approve("probe", old.confirmation)
    calls = []

    def factory(*args, **kwargs):
        calls.append(kwargs["command"])
        return SimpleNamespace()

    monkeypatch.setattr(MCPManager, "create_from_server", factory)
    assert await registry.connect(["probe"]) == ["probe"]
    save(registry, "new-command")
    with pytest.raises(MCPApprovalRequired):
        await registry.connect(["probe"])
    assert calls == ["old-command"]
    assert registry.connected() == []
    assert registry._approval_request("probe").fingerprint != old.fingerprint


@pytest.mark.parametrize("change", ["replace", "remove", "revoke"])
async def test_pending_connection_cannot_attach_after_config_or_approval_changes(
    registry, monkeypatch, change
):
    save(registry, "old-command")
    registry.refresh_settings()
    registry._approve("probe", registry._approval_request("probe").confirmation)
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    def factory(*args, **kwargs):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5), "test did not release discovery"
        return SimpleNamespace()

    monkeypatch.setattr(MCPManager, "create_from_server", factory)
    task = asyncio.create_task(registry.connect(["probe"]))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        if change == "revoke":
            registry._revoke_approvals("probe")
        else:
            save(registry, "new-command" if change == "replace" else None)
        release.set()
        with pytest.raises(RuntimeError, match="changed during connection"):
            await asyncio.wait_for(task, 2)
        assert registry.connected() == []
        assert registry.activated() == []
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
