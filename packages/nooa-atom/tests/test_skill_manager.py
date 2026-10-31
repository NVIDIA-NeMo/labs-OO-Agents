# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SkillManager: one surface for code skills, text skills and MCP servers."""

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from atom_test_agents import ATOM_SPEC
from fake_oauth_mcp import FakeOAuthServer
from nooa_atom.agent.base import AtomAgentBase
from nooa_atom.skills.manager import SkillManager
from nooa_atom.skills.mcp_servers import MCPServers
from nooa_atom.workspace.controls import MCPControl, SkillsControl
from nooa_atom.workspace.options import AtomOptions

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
BLOCK_TAIL = (
    "self.skills.search('query') (sync); print(self.skills.doc('name')) for full instructions/APIs "
    "(unconnected MCP: connection status); await self.skills.activate(['name']) (async)"
)


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
    made = AtomAgentBase(llm=FakeLLMClient(), cwd=tmp_path, skills_dirs=[skills_dir])
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


@pytest.mark.parametrize("concise", [False, True])
async def test_manager_hides_internal_read_and_source_dir_only(agent, concise, tmp_path):
    from nooa.agentdoc import doc

    for target in (SkillManager, agent.skills):
        rendered = doc(target, concise=concise)
        assert "source_dir" not in rendered
        assert "def read(" not in rendered
        assert "def doc(self, name: str) -> str" in rendered
    assert agent.skills.source_dir is not None  # functionality stays available to host code
    assert agent.skills.read("review") in agent.skills.doc("review")
    skill = agent.skills.registry["cmd.review"]
    assert skill.source_dir == tmp_path / "skills" / "review"
    assert "source_dir" in doc(skill) and "def read_file(" in doc(skill)
    (skill.source_dir / "guide.txt").write_text("local guide")
    assert skill.read_file("guide.txt") == "local guide"
    extra = tmp_path / "extra_skills"
    (extra / "second").mkdir(parents=True)
    (extra / "second" / "SKILL.md").write_text(
        "---\nname: second\ndescription: Second skill\n---\nSecond full instructions.\n"
    )
    agent.skills.discover_skills_dirs([extra])
    assert "Second full instructions." in agent.skills.doc("second")
    assert "Read the diff, then list defects" in agent.skills.doc("review")


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
    assert agent.skills.search("greet people").splitlines() == [
        "greeter (code, available): Greet people by name.",
        # A loaded code skill's public methods are searched too.
        "greeter has greet (method: self.greeter.greet)",
    ]
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
    assert "print(self.skills.doc('review'))" in result
    assert "self.skills.read" not in result
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
    assert "Instructions:\nRead the diff, then list defects by severity." in review
    assert "self.review.read_file(path)" in review
    probe = agent.skills.doc("probe")
    assert "probe (mcp, available)" in probe and "not connected" in probe
    with pytest.raises(KeyError, match="nope"):
        agent.skills.doc("nope")


async def test_activating_an_mcp_server_connects_it_and_its_tools_are_methods(agent):
    approve(agent, "probe")
    result = await agent.skills.activate(["probe"])
    assert result == "probe: active; its tools are methods of self.probe: probe"
    assert (
        agent.skills.status()
        .splitlines()[0]
        .endswith(", probe (mcp: await self.probe.<tool>(...))")
    )
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
    control = MCPControl(agent, AtomOptions(), workspace=tmp_path)
    finished = await control.run(["auth", "remote", pasted])
    assert finished.success, str(finished)
    assert "echo" in str(finished)
    assert (
        agent.skills.status()
        .splitlines()[0]
        .endswith(", remote (mcp: await self.remote.<tool>(...))")
    )
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
    assert "remote (mcp, active; call: await self.remote.<tool>(...))" in agent.skills.search(
        "remote"
    )
    assert (
        agent.skills.status()
        .splitlines()[0]
        .endswith(", remote (mcp: await self.remote.<tool>(...))")
    )


async def test_the_skills_control_lists_every_kind(agent, tmp_path):
    control = SkillsControl(agent, AtomOptions(), workspace=tmp_path)
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
    control = SkillsControl(agent, AtomOptions(), workspace=tmp_path)
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

    from atom_test_agents import python_cell
    from nooa_atom.agent.factory import create_session_agent
    from nooa_atom.session.options import SessionOptions
    from nooa_atom.session.registry import SessionRegistry
    from nooa_atom.session.store import SessionStore

    from nooa.storage.json_snapshot import snapshot_to_json

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    options = SessionOptions(
        workspace=workspace,
        agent_spec=ATOM_SPEC,
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

    llm = FakeLLMClient([python_cell("return Done(explanation='x')", "c1")])
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


async def test_read_of_a_non_text_skill_names_the_working_doc_call(agent):
    with pytest.raises(ValueError) as error:
        agent.skills.read("probe")
    message = str(error.value)
    assert "print(self.skills.doc('probe'))" in message
    assert "use doc('probe')" not in message
    assert agent.skills.doc("probe").startswith("probe")


@pytest.mark.parametrize(
    "query",
    [
        "timer",
        "schedule",
        "cron",
        "remind",
        "later",
        "after",
        "every",
        "periodic",
        "monitor",
        "watch",
        "tail",
        "follow",
        "Schedule a reading in 120 s",
    ],
)
async def test_search_for_timers_and_tails_points_to_self_jobs(agent, query):
    found = agent.skills.search(query)
    assert "No skills match" not in found
    first = found.splitlines()[0]
    assert first.startswith("self.jobs (built in, no skill needed):")
    for call in (
        "self.jobs.after(",
        "self.jobs.every(",
        'Shell().run("tail -F',
        "progress=Throttle(",
    ):
        assert call in first
    assert "own Shell" in first and "one command at a time" in first
    assert "self.jobs.tail(" not in first
    assert "a delay counts from now: subtract time you have already spent" in first


async def test_search_hint_comes_before_matching_skills(agent):
    assert agent.skills.search("greet later").splitlines() == [
        agent.skills.search("later"),
        "greeter (code, available): Greet people by name.",
        "greeter has greet (method: self.greeter.greet)",
    ]


async def test_search_matches_mcp_tool_names_after_skills(agent):
    class Resources:
        """MCP server 'hcb_resources'."""

        _tool_method_names = ("issue_parking_permit", "list_rooms", "permit_status")

    agent.skills.register("mcp.hcb_resources", Resources())
    await agent.skills.activate(["mcp.hcb_resources"])
    # The skill line (its description lists the tools, cut at 100 characters)
    # comes first, then each matching tool.
    found = agent.skills.search("permit").splitlines()
    assert found[0].startswith("hcb_resources (mcp, active")
    assert found[1:] == [
        "hcb_resources has issue_parking_permit (MCP tool: self.hcb_resources.issue_parking_permit)",
        "hcb_resources has permit_status (MCP tool: self.hcb_resources.permit_status)",
    ]
    # Tool matches are bounded by the limit.
    assert agent.skills.search("permit", limit=1).splitlines()[1:] == [
        "hcb_resources has issue_parking_permit (MCP tool: self.hcb_resources.issue_parking_permit)",
        "... 1 more tools; narrow the query",
    ]
    # Short words do not match tool names.
    assert not any(" has " in line for line in agent.skills.search("is").splitlines())
    # The self.jobs hint still comes first.
    assert agent.skills.search("permit later").splitlines()[0].startswith("self.jobs (built in")


async def test_a_client_server_shown_under_its_key_is_called_on_its_attribute(agent):
    # A code skill already shows as "greeter", so the client server shows as
    # "mcp.greeter"; SkillRegistry still attaches it as self.greeter.
    class Server:
        """MCP server 'greeter'."""

        _tool_method_names = ("wave_hello",)

    agent.skills.register("mcp.greeter", Server())
    await agent.skills.activate(["mcp.greeter"])
    assert agent.greeter is agent.skills.registry["mcp.greeter"]
    found = agent.skills.search("wave")
    assert "mcp.greeter has wave_hello (MCP tool: self.greeter.wave_hello)" in found
    assert "self.mcp." not in found
    assert "self.mcp." not in agent.skills.search("greeter")
    assert "mcp.greeter (mcp: await self.greeter.<tool>(...))" in agent.skills.status()


async def test_search_omits_hidden_methods_of_code_skills(agent):
    from nooa.agentdoc import hidden

    class Stash(Skill):
        """Keep things."""

        def stash_put(self, item: str) -> None:
            """Keep ``item``."""

        @hidden
        def stash_restore(self, data: dict) -> None:
            """Host-only: replace the state."""

    agent.skills.register("tools.stash", Stash())
    found = agent.skills.search("stash")
    assert "stash has stash_put (method: self.stash.stash_put)" in found
    assert "stash_restore" not in found
    # The todo skill's restore hook (@hidden from_dict) is not advertised either.
    assert "todo has from_dict" not in agent.skills.search("from_dict")


def test_share_from_keeps_an_attachable_client_skill_with_its_parent(tmp_path):
    class Bound(Skill):
        """MCP server 'bound'."""

        async def ping(self) -> str:
            return "pong"

    parent = AtomAgentBase(llm=FakeLLMClient(), cwd=tmp_path)
    child = AtomAgentBase(llm=FakeLLMClient(), cwd=tmp_path)
    bound = Bound()
    parent.skills.register("mcp.bound", bound)
    warnings = child.skills.share_from(parent.skills)
    assert warnings == ["MCP server 'bound' was not shared: it is bound to the parent agent"]
    assert getattr(bound, "_agent", None) is parent and "mcp.bound" not in child.skills.loaded()


def test_share_from_keeps_a_childs_skill_at_the_same_attribute(tmp_path):
    class Client:
        """MCP server 'greeter' (a client-provided one)."""

        async def ping(self) -> str:
            return "pong"

    parent = AtomAgentBase(llm=FakeLLMClient(), cwd=tmp_path)
    child = AtomAgentBase(llm=FakeLLMClient(), cwd=tmp_path)
    parent.skills.register("mcp.greeter", Client())
    own = Greeter()
    child.skills.register("tools.greeter", own)
    warnings = child.skills.share_from(parent.skills)
    assert warnings == ["MCP server 'greeter' was not shared: self.greeter is already in use here"]
    assert child.greeter is own and "mcp.greeter" not in child.skills.loaded()


async def test_search_matches_a_connected_mcp_servers_tools(agent):
    approve(agent, "probe")
    await agent.skills.activate(["probe"])
    assert "probe has probe (MCP tool: self.probe.probe)" in agent.skills.search("probe")
