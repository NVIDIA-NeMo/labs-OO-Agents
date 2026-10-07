# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCPServers: saved definitions, approval, connecting, and signing in."""

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fake_oauth_mcp import FakeOAuthServer
from nooa_atom.skills.mcp_servers import MCPServers, MCPSignInRequired
from nooa_atom.workspace.controls import MCPControl
from nooa_atom.workspace.mcp_approval import MCPApprovalRequired
from nooa_atom.workspace.options import AtomOptions

from nooa.mcp import MCPManager

PROBE = Path(__file__).parent / "acp_adapter" / "fixtures" / "mcp_probe.py"
_PROXY_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


class _Agent:
    """Only what MCPServers touches: attributes for connected servers."""


@pytest.fixture
def servers(tmp_path, monkeypatch):
    project = tmp_path / ".nooa"
    project.mkdir()
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    monkeypatch.delenv("NEMO_OO_PROJECT_DIR", raising=False)
    made = MCPServers(
        mcp_file=tmp_path / ".mcp.json",
        approval_path=tmp_path / "approvals.json",
        watch_settings=True,
        project_dir=project,
        token_path=tmp_path / "tokens.json",
    )
    notes: list[str] = []
    made.bind(_Agent(), notes.append)
    made.notes = notes
    return made


def save(servers, command):
    definitions = {"probe": {"command": command}} if command is not None else {}
    (servers.project_dir / "settings.yaml").write_text(
        json.dumps({"atom": {"mcp_servers": definitions}})
    )


def approve(servers, name):
    servers.approve(name, servers.approval_request(name).confirmation)


def _control(servers, tmp_path):
    agent = SimpleNamespace(skills=SimpleNamespace(mcp=servers))
    return MCPControl(agent, AtomOptions(), workspace=tmp_path)


async def test_controls_see_saved_servers_and_keep_registered_ones(servers, tmp_path):
    servers.register("transient", command="transient-command")
    control = _control(servers, tmp_path)
    save(servers, "new-command")
    status = await control.run(["status"])
    assert status.success and "probe" in str(status) and "transient" in str(status)
    review = await control.run(["approve", "probe"])
    assert review.success
    assert servers.approval_request("probe").confirmation in str(review)


async def test_the_status_control_shows_server_endpoints(servers, tmp_path):
    servers.register("remote", url="https://mcp.example.com/mcp")
    status = await _control(servers, tmp_path).run(["status"])
    assert status.success
    assert "https://mcp.example.com/mcp" in str(status)
    assert "approval required" in str(status)


async def test_a_changed_definition_needs_approval_again_before_connecting(servers, monkeypatch):
    save(servers, "old-command")
    servers.refresh_settings()
    old = servers.approval_request("probe")
    approve(servers, "probe")
    calls = []

    async def create(name, *, command, **kwargs):
        calls.append(command)
        return SimpleNamespace()

    monkeypatch.setattr(MCPManager, "create_stdio_server", create)
    assert await servers.connect(["probe"]) == ["probe"]
    save(servers, "new-command")
    with pytest.raises(MCPApprovalRequired):
        await servers.connect(["probe"])
    assert calls == ["old-command"]
    assert servers.connected() == []
    assert servers.approval_request("probe").fingerprint != old.fingerprint


@pytest.mark.parametrize("change", ["replace", "remove", "revoke"])
async def test_a_connection_is_not_kept_when_config_or_approval_changes_meanwhile(
    servers, monkeypatch, change
):
    save(servers, "old-command")
    servers.refresh_settings()
    approve(servers, "probe")
    entered, release = asyncio.Event(), asyncio.Event()

    async def create(name, **kwargs):
        entered.set()
        await release.wait()
        return SimpleNamespace()

    monkeypatch.setattr(MCPManager, "create_stdio_server", create)
    task = asyncio.create_task(servers.connect(["probe"]))
    await asyncio.wait_for(entered.wait(), 2)
    if change == "revoke":
        servers.revoke("probe")
    else:
        save(servers, "new-command" if change == "replace" else None)
    release.set()
    with pytest.raises(RuntimeError, match="changed while connecting"):
        await asyncio.wait_for(task, 2)
    assert servers.connected() == []
    assert servers.activated() == []


async def test_a_stdio_server_connects_and_its_tools_are_methods_on_the_agent(servers, tmp_path):
    journal = tmp_path / "journal.jsonl"
    servers.register("probe", command=sys.executable, args=[str(PROBE), "--journal", str(journal)])
    approve(servers, "probe")
    assert await servers.connect(["probe"]) == ["probe"]
    assert servers.state("probe") == "active"
    assert servers.tool_names("probe") == ["probe"]
    reply = await servers.agent.probe.probe(nonce="n-1")
    assert json.loads(reply)["nonce"] == "n-1"
    await servers.deactivate(["probe"])
    assert servers.state("probe") == "connected"
    assert await servers.disconnect(["probe"]) == ["probe"]
    assert servers.state("probe") == "available"
    assert not hasattr(servers.agent, "probe")


def test_discovered_reads_the_config_file_again_only_after_it_changes(servers, monkeypatch):
    servers.mcp_file.write_text(json.dumps({"mcpServers": {"tool": {"command": "run-tool"}}}))
    reads = []
    original = MCPManager.list_servers
    monkeypatch.setattr(
        MCPManager, "list_servers", lambda *a, **k: reads.append(1) or original(*a, **k)
    )
    assert servers.discovered() == ["tool"]
    assert servers.discovered() == ["tool"]
    assert len(reads) == 1
    servers.mcp_file.write_text(json.dumps({"mcpServers": {"other": {"command": "run-other"}}}))
    assert servers.discovered() == ["other"]


@pytest.fixture
def oauth_server(monkeypatch):
    for name in _PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    fake = FakeOAuthServer().start()
    yield fake
    fake.stop()


async def test_a_server_that_needs_sign_in_connects_after_the_pasted_address(servers, oauth_server):
    servers.register("remote", url=oauth_server.url)
    approve(servers, "remote")
    with pytest.raises(MCPSignInRequired) as raised:
        await servers.connect(["remote"])
    url = raised.value.url
    assert url.startswith(f"{oauth_server.base}/authorize?")
    assert "/mcp auth remote" in str(raised.value)
    assert servers.state("remote") == "needs-auth"
    assert servers.sign_in_url("remote") == url

    pasted = await asyncio.to_thread(oauth_server.consent, url)
    message = await servers.complete_sign_in("remote", pasted)
    assert "remote" in message and "echo" in message
    assert servers.state("remote") == "active"
    assert await servers.agent.remote.echo(text="hi") == "hi"
    assert servers.notes and "remote" in servers.notes[-1]

    # Signed in once: the next connection uses the stored token.
    await servers.disconnect(["remote"])
    assert await servers.connect(["remote"]) == ["remote"]
    assert oauth_server.grants == ["authorization_code"]


async def test_a_bad_pasted_address_keeps_the_sign_in_waiting(servers, oauth_server):
    servers.register("remote", url=oauth_server.url)
    approve(servers, "remote")
    with pytest.raises(MCPSignInRequired):
        await servers.connect(["remote"])
    with pytest.raises(ValueError, match="no authorisation code"):
        await servers.complete_sign_in("remote", "http://localhost/callback")
    assert servers.state("remote") == "needs-auth"
    await servers.aclose()
    assert servers.state("remote") == "available"


async def test_completing_a_sign_in_that_is_not_waiting_is_refused(servers):
    with pytest.raises(ValueError, match="No sign-in is waiting"):
        await servers.complete_sign_in("remote", "http://localhost/callback?code=x")


async def test_a_failed_connection_is_reported_in_the_state(servers):
    servers.register("broken", command="/nonexistent/mcp-server")
    approve(servers, "broken")
    with pytest.raises(RuntimeError, match="broken"):
        await servers.connect(["broken"])
    assert servers.state("broken") == "failed"
    assert "broken" in servers.error("broken")
