# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Connected MCP servers are named up front, and a tool called without its server is pointed at it."""

import sys
from pathlib import Path

import pytest
from atom_test_agents import AtomModels, cell
from nooa_atom.agent.base import AtomAgentBase
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore
from nooa_atom.skills.mcp_servers import MCPServers

from nooa.interactive import Done
from nooa.runtime.harness_metrics import HarnessMetrics
from nooa.unifiedllm import FakeLLMClient

PROBE = Path(__file__).parent / "acp_adapter" / "fixtures" / "mcp_probe.py"
LINE = (
    "MCP servers (connected, no activation needed): self.hcb_resources (2 tools); "
    "list tools: print(doc(self.hcb_resources)); call: await self.hcb_resources.<tool>(...)"
)


class Resources:
    """MCP server 'hcb_resources' (a client-provided one)."""

    async def lookup_order(self, order_id: int) -> str:
        return f"order {order_id}"

    async def list_items(self) -> list[str]:
        return ["a"]


class Other:
    """Another client server."""

    async def ping(self) -> str:
        return "pong"


def test_no_line_without_a_connected_server(tmp_path):
    agent = AtomAgentBase(llm=FakeLLMClient(), cwd=tmp_path)
    assert agent.skills.mcp_line() == ""
    assert agent.skills.mcp_tool_server("lookup_order") is None


def test_the_line_names_servers_and_counts_tools_without_listing_them(tmp_path):
    agent = AtomAgentBase(llm=FakeLLMClient(), cwd=tmp_path)
    agent.skills.register("mcp.hcb_resources", Resources())
    assert agent.skills.mcp_line() == LINE
    assert "lookup_order" not in agent.skills.mcp_line()
    agent.skills.register("mcp.other", Other())
    assert agent.skills.mcp_line() == (
        "MCP servers (connected, no activation needed): self.hcb_resources (2 tools), "
        "self.other (1 tool); list tools: print(doc(self.<server>)); "
        "call: await self.<server>.<tool>(...)"
    )
    assert agent.skills.mcp_tool_server("lookup_order") == "hcb_resources"
    assert agent.skills.mcp_tool_server("ping") == "other"
    assert agent.skills.mcp_tool_server("nothing") is None


def test_the_line_is_bounded(tmp_path):
    agent = AtomAgentBase(llm=FakeLLMClient(), cwd=tmp_path)
    for i in range(11):
        agent.skills.register(f"mcp.s{i:02d}", Other())
    line = agent.skills.mcp_line()
    assert "self.s07 (1 tool), +3 more: self.skills.search('query');" in line
    assert "self.s08" not in line


async def test_a_connected_configured_server_is_named_active_or_not(tmp_path):
    agent = AtomAgentBase(llm=FakeLLMClient(), cwd=tmp_path)
    servers = MCPServers(
        mcp_file=tmp_path / ".mcp.json",
        approval_path=tmp_path / "approvals.json",
        project_dir=tmp_path / ".nooa",
        token_path=tmp_path / "tokens.json",
    )
    servers.register(
        "probe", command=sys.executable, args=[str(PROBE), "--journal", str(tmp_path / "j")]
    )
    agent.skills.set_mcp_servers(servers)
    try:
        assert agent.skills.mcp_line() == ""  # configured but not connected
        servers.approve("probe", servers.approval_request("probe").confirmation)
        await agent.skills.activate(["probe"])
        expected = (
            "MCP servers (connected, no activation needed): self.probe (1 tool); "
            "list tools: print(doc(self.probe)); call: await self.probe.<tool>(...)"
        )
        assert agent.skills.mcp_line() == expected
        await agent.skills.deactivate(["probe"])  # still connected: still self.probe
        assert agent.skills.mcp_line() == expected and agent.probe is servers.tool("probe")
        assert agent.skills.mcp_tool_server("probe") == "probe"
    finally:
        await agent.aclose()


def _prompt(llm, index):
    return str(llm.calls[index].messages)


@pytest.fixture
async def registry(tmp_path, monkeypatch, tmp_path_factory):
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path_factory.mktemp("user-config")))
    models = AtomModels()
    registry = SessionRegistry(SessionStore(tmp_path / "sessions"), agent_factory=models)
    registry.models = models
    yield registry
    await registry.close_all()


async def test_root_and_child_name_the_server_in_the_first_request_and_hint_its_tools(
    registry, tmp_path, monkeypatch
):
    metrics = HarnessMetrics()  # the turn's own metrics session would hide them
    monkeypatch.setattr("nooa_atom.agent.agent.get_harness_metrics", lambda: metrics)
    models = registry.models
    models.scripts[None] = [
        cell("await lookup_order(order_id=1)"),
        cell("await self.lookup_order(order_id=1)"),
        cell("undefined_thing"),
        cell("await self.not_a_tool()"),
        cell("return_result(Done(explanation='ok'))"),
    ]
    models.scripts["c"] = [cell("return_result(Done(explanation='child ok'))")]
    options = SessionOptions(workspace=tmp_path, agent_spec="nooa_atom.agent:AtomAgent")
    root = await registry.create(options)
    assert await root.register_tools({"mcp.hcb_resources": Resources()}) == {}
    assert await root.prompt("go") == Done(explanation="ok")
    llm = models.llms[None]
    assert LINE in _prompt(llm, 0)  # the first request
    assert "lookup_order" not in _prompt(llm, 0)  # tools are not listed
    name_hint = "`lookup_order` is a tool of MCP server hcb_resources: "
    call = "await self.hcb_resources.lookup_order(...)"
    after_bare, after_self = _prompt(llm, 1), _prompt(llm, 2)
    assert "NameError" in after_bare and name_hint + call in after_bare
    assert "AttributeError" in after_self and name_hint + call in after_self
    assert after_self.count(name_hint) == 2  # the bare call's hint, then the self.<tool> one
    # Unrelated names get no MCP hint.
    assert _prompt(llm, 3).count("is a tool of MCP server") == 2
    assert _prompt(llm, 4).count("is a tool of MCP server") == 2
    assert metrics.mcp_tool_name_hints == ["hcb_resources.lookup_order"] * 2

    child = await registry.create(options.inherit(name="c"), parent_id=root.id)
    assert child._agent.hcb_resources is root._agent.hcb_resources
    assert await child.prompt("go") == Done(explanation="child ok")
    assert LINE in _prompt(models.llms["c"], 0)


async def test_no_line_in_the_request_without_a_server(registry, tmp_path):
    models = registry.models
    models.scripts[None] = [cell("return_result(Done(explanation='ok'))")]
    root = await registry.create(
        SessionOptions(workspace=tmp_path, agent_spec="nooa_atom.agent:AtomAgent")
    )
    assert await root.prompt("go") == Done(explanation="ok")
    assert "MCP servers (connected" not in _prompt(models.llms[None], 0)
