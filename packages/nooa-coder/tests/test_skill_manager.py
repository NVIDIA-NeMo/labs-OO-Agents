# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SkillManager: one surface for code skills, text skills and MCP servers."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fake_oauth_mcp import FakeOAuthServer
from nooa_coder.coding.agent import CodingAgent
from nooa_coder.skills.manager import SkillManager
from nooa_coder.skills.mcp_servers import MCPServers
from nooa_coder.workspace.controls import MCPControl, SkillsControl
from nooa_coder.workspace.options import CoderOptions

from nooa.events import Notification
from nooa.interactive import AgentMessage
from nooa.skill import Skill
from nooa.unifiedllm import FakeLLMClient

PROBE = Path(__file__).parent / "acp_adapter" / "fixtures" / "mcp_probe.py"
_PROXY_VARIABLES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
BLOCK_TAIL = "self.skills.search('query'); await activate(['name']); read('name') for text skills"


class Greeter(Skill):
    """Greet people by name."""

    def greet(self, name: str) -> str:
        """Return a greeting for ``name``."""
        return f"Hello, {name}"


class _EntryPoint:
    """An installed ``nooa.skills`` entry point; ``loads`` counts imports."""

    def __init__(self, name, target, summary):
        self.name = name
        self.value = f"example_skills:{name}"
        self.dist = SimpleNamespace(metadata={"Summary": summary})
        self.loads = 0
        self._target = target

    def load(self):
        self.loads += 1
        if self._target is None:
            raise ImportError("importing this skill is not allowed in this test")
        return self._target


@pytest.fixture
def installed(monkeypatch, tmp_path):
    points = [
        _EntryPoint("tools.greeter", Greeter, "Greeting helpers"),
        # The agent does not load nemo.memory at start; searching must not either.
        _EntryPoint("nemo.memory", None, "Long-term memory you own and curate"),
    ]
    monkeypatch.setattr("nooa.skill_registry.entry_points", lambda *, group: points)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return {point.name: point for point in points}


@pytest.fixture
async def agent(installed, tmp_path):
    skills_dir = tmp_path / "skills"
    (skills_dir / "review").mkdir(parents=True)
    (skills_dir / "review" / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review a change for defects\n---\n"
        "Read the diff, then list defects by severity.\n"
    )
    made = CodingAgent(llm=FakeLLMClient(), cwd=tmp_path, skills_dirs=[skills_dir])
    servers = MCPServers(
        mcp_file=tmp_path / ".mcp.json",
        approval_path=tmp_path / "approvals.json",
        project_dir=tmp_path / ".nooa",
        token_path=tmp_path / "tokens.json",
    )
    servers.register(
        "probe",
        command=sys.executable,
        args=[str(PROBE), "--journal", str(tmp_path / "journal.jsonl")],
    )
    made.skills.set_mcp_servers(servers)
    yield made
    await made.aclose()


def approve(agent, name):
    servers = agent.skills.mcp
    servers.approve(name, servers.approval_request(name).confirmation)


async def test_self_skills_is_the_skill_manager(agent):
    assert isinstance(agent.skills, SkillManager)
    assert not hasattr(agent, "mcp")


async def test_the_block_names_active_skills_and_counts_the_rest(agent):
    assert agent.skills.status() == (
        "Active: libwriting, methodwriting, repo, shell, todo\n"
        f"4 more (code, text, MCP): {BLOCK_TAIL}"
    )


async def test_the_block_stays_the_same_until_something_is_activated(agent):
    first = agent.skills.status()
    agent.skills.search("greet")
    agent.skills.doc("review")
    assert agent.skills.status() == first
    await agent.skills.activate(["greeter"])
    assert agent.skills.status().startswith("Active: greeter, libwriting,")


async def test_search_matches_names_and_descriptions_of_every_kind(agent):
    assert agent.skills.search("review") == (
        "review (text, available): Review a change for defects"
    )
    assert agent.skills.search("greet people") == "greeter (code, available): Greet people by name."
    assert agent.skills.search("probe").startswith("probe (mcp, available): stdio MCP server")
    assert agent.skills.search("nothing-like-this") == "No skills match 'nothing-like-this'."


async def test_search_does_not_import_a_code_skill(agent, installed):
    assert agent.skills.search("memory") == (
        "memory (code, available): Long-term memory you own and curate"
    )
    assert installed["nemo.memory"].loads == 0


async def test_search_respects_the_limit(agent):
    lines = agent.skills.search("e", limit=2).splitlines()
    assert len(lines) == 3
    assert lines[-1].startswith("... ")


async def test_activating_and_deactivating_a_code_skill(agent):
    assert await agent.skills.activate(["greeter"]) == "greeter: active; use self.greeter"
    assert "tools.greeter" in agent.skills.activated()
    assert agent.greeter.greet("Ada") == "Hello, Ada"
    assert await agent.skills.deactivate(["greeter"]) == "greeter: inactive"
    assert "tools.greeter" not in agent.skills.activated()


async def test_a_name_that_matches_nothing_is_reported(agent):
    assert await agent.skills.activate(["nope"]) == (
        "nope: no such skill; try self.skills.search('nope')"
    )


async def test_activating_a_text_skill_sends_its_doc_to_the_system_messages_channel(agent):
    result = await agent.skills.activate(["review"])
    assert "system_messages" in result
    channel = agent.queue_manager.get_channel("system_messages")
    [item] = channel.drain()
    assert item == agent.skills.doc("review")
    assert "Read the diff, then list defects by severity." in item
    assert "review" not in agent.skills.status().splitlines()[0]


async def test_read_returns_a_text_skills_instructions(agent):
    assert agent.skills.read("review") == "Read the diff, then list defects by severity."
    with pytest.raises(ValueError, match="code skill"):
        agent.skills.read("greeter")


async def test_doc_describes_each_kind(agent, installed):
    assert "def greet(self, name: str) -> str" in agent.skills.doc("greeter")
    review = agent.skills.doc("review")
    assert review.startswith("review (text skill): Review a change for defects")
    probe = agent.skills.doc("probe")
    assert "probe (mcp, available)" in probe and "not connected" in probe
    with pytest.raises(KeyError, match="nope"):
        agent.skills.doc("nope")


async def test_activating_an_mcp_server_connects_it_and_its_tools_are_methods(agent):
    approve(agent, "probe")
    result = await agent.skills.activate(["probe"])
    assert result == "probe: active; its tools are methods of self.probe: probe"
    assert agent.skills.status().splitlines()[0].endswith(", probe (mcp)")
    assert "async def probe(self, nonce: str)" in agent.skills.doc("probe")
    assert await agent.skills.deactivate(["probe"]) == "probe: inactive (still connected)"
    assert "probe" not in agent.skills.status().splitlines()[0]


async def test_an_unapproved_mcp_server_asks_for_approval(agent):
    assert await agent.skills.activate(["probe"]) == (
        "probe: not approved; ask the person to run /mcp approve probe"
    )


@pytest.fixture
def oauth_server(monkeypatch):
    for name in _PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    fake = FakeOAuthServer().start()
    yield fake
    fake.stop()


async def test_a_server_that_needs_sign_in_sends_the_link_and_reports_back(
    agent, oauth_server, tmp_path
):
    agent.skills.mcp.register("remote", url=oauth_server.url)
    approve(agent, "remote")
    result = await agent.skills.activate(["remote"])
    assert result.startswith("remote: needs sign-in;")
    [link_message] = [
        event.content for event in agent.event_manager.values() if isinstance(event, AgentMessage)
    ]
    assert "/mcp auth remote" in link_message
    url = agent.skills.mcp.sign_in_url("remote")
    assert url in link_message
    assert "remote (mcp, needs-auth)" in agent.skills.search("remote")

    pasted = await asyncio.to_thread(oauth_server.consent, url)
    control = MCPControl(agent, CoderOptions(), workspace=tmp_path)
    finished = await control.run(["auth", "remote", pasted])
    assert finished.success, str(finished)
    assert "echo" in str(finished)
    assert agent.skills.status().splitlines()[0].endswith(", remote (mcp)")
    notices = [
        event.description
        for event in agent.event_manager.values()
        if isinstance(event, Notification) and event.source == "skills"
    ]
    assert notices and "remote" in notices[-1]


async def test_a_client_supplied_mcp_server_is_an_mcp_skill(agent):
    class Remote:
        """MCP server 'remote'."""

        async def ping(self) -> str:
            return "pong"

    agent.skills.register("mcp.remote", Remote())
    assert await agent.skills.activate(["mcp.remote"]) == "remote: active; use self.remote"
    assert "remote (mcp, active)" in agent.skills.search("remote")
    assert agent.skills.status().splitlines()[0].endswith(", remote (mcp)")


async def test_the_skills_control_lists_every_kind(agent, tmp_path):
    control = SkillsControl(agent, CoderOptions(), workspace=tmp_path)
    listed = await control.run(["list"])
    assert listed.success
    table = listed.outputs[0]
    assert table.columns == ["Name", "Kind", "State", "Description"]
    rows = {row[0]: row[1:3] for row in table.rows}
    assert rows["shell"] == ["code", "active"]
    assert rows["review"] == ["text", "available"]
    assert rows["probe"] == ["mcp", "available"]


async def test_the_skills_control_activates_by_name(agent, tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / ".nooa"))
    control = SkillsControl(agent, CoderOptions(), workspace=tmp_path)
    result = await control.run(["activate", "greeter"])
    assert result.success, str(result)
    assert "tools.greeter" in agent.skills.activated()


async def test_the_mcp_add_command_hands_the_details_to_the_agent(agent):
    text = await agent.skills.mcp_add_command("docs https://example.com/mcp")
    assert "docs https://example.com/mcp" in text
    assert "/mcp approve <name>" in text


async def test_an_old_snapshot_with_mcp_state_loads_and_the_agent_is_told(
    installed, tmp_path, sessions_dir
):
    """A snapshot from before SkillManager holds the ``<mcp>`` block, ``self.mcp.status()``."""
    import json

    from nooa_coder.coding.factory import create_session_agent
    from nooa_coder.session.options import SessionOptions
    from nooa_coder.session.registry import SessionRegistry
    from nooa_coder.session.store import SessionStore
    from test_experimental_agent import python_cell

    from nooa.storage.json_snapshot import snapshot_to_json

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    options = SessionOptions(
        workspace=workspace,
        agent_spec="nooa_coder.coding.agent:CodingAgent",
        llm=FakeLLMClient(),
        sessions_dir=sessions_dir,
    )
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=create_session_agent)
    try:
        root = await registry.create(options)
        old = snapshot_to_json(root._agent)
        old["context"].append({"key": "mcp", "type": "dynamic", "expr": "self.mcp.status()"})
        root.handle.storage.save_snapshot_json(json.dumps(old), created_at="9999-01-01T00:00:00")
    finally:
        await registry.close_all()

    llm = FakeLLMClient([python_cell("return_result(Done(explanation='x'))", "c1")])
    fresh = SessionRegistry(SessionStore(sessions_dir), agent_factory=create_session_agent)
    try:
        loaded = await fresh.load(root.id, llm=llm)
        assert "mcp" not in loaded._agent.context
        await asyncio.wait_for(loaded.prompt("hello"), 30)
    finally:
        await fresh.close_all()
    prompt = str(llm.calls[0].messages)
    assert "not restored: mcp" in prompt
    assert "self.mcp.status()" not in prompt
