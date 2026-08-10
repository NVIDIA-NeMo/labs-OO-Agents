# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MCPRegistry — agent-facing MCP server management."""

import asyncio
import threading

import pytest
from nooa_cli.tui.mcp_registry import MCPRegistry

from nooa.agentdoc._visibility import is_hidden_field
from nooa.runtime.context_manager import ContextManager


class _FakeTool:
    """Stand-in for a dynamically generated MCPTool subclass."""

    _tool_method_names = frozenset({"search", "get_page"})

    async def search(self, query: str) -> object:
        """Search the knowledge base."""

    async def get_page(self, page_id: str) -> object:
        """Fetch a page by id."""

    async def _call_tool(self, *a, **k): ...


class _FakeAgent:
    def __init__(self):
        self.context_manager = ContextManager()


@pytest.fixture(autouse=True)
def _isolate_approval_store(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))


def _approve(reg, *names):
    selected = names or tuple(reg.discovered())
    for name in selected:
        request = reg._approval_request(name)
        reg._approval_store.approve(request)


def _make(servers=None, mcp_file=None, attach=True, approve=True):
    reg = MCPRegistry(mcp_file=mcp_file, servers=servers)
    if approve:
        _approve(reg)
    agent = _FakeAgent()
    if attach:
        reg.attach(agent)
    return reg, agent


def _fake_create(monkeypatch, tool_factory=_FakeTool):
    """Patch MCPManager.create_from_server to return a fresh fake tool."""
    calls = []

    def _create(name, **kwargs):
        calls.append((name, kwargs))
        return tool_factory()

    import nooa.mcp as mcp_mod

    monkeypatch.setattr(mcp_mod.MCPManager, "create_from_server", staticmethod(_create))
    return calls


# ---------------------------------------------------------------------------
# Discovery / register
# ---------------------------------------------------------------------------


def test_discovered_unions_inline_and_file(tmp_path):
    mcp_file = tmp_path / ".mcp.json"
    mcp_file.write_text('{"mcpServers": {"fileserver": {"url": "https://f/mcp"}}}')
    reg, _ = _make(servers={"inline": {"url": "https://i/mcp"}}, mcp_file=mcp_file)
    assert reg.discovered() == ["fileserver", "inline"]


def test_discovered_dedups_name_collision(tmp_path):
    mcp_file = tmp_path / ".mcp.json"
    mcp_file.write_text('{"mcpServers": {"maas": {"url": "https://file/mcp"}}}')
    reg, _ = _make(servers={"maas": {"url": "https://inline/mcp"}}, mcp_file=mcp_file)
    assert reg.discovered() == ["maas"]


def test_register_adds_in_memory_entry():
    reg, _ = _make()
    assert reg.discovered() == []
    reg.register("foo", url="https://foo/mcp", transport="streamable-http")
    assert reg.discovered() == ["foo"]


def test_register_copies_mutable_inputs():
    args = ["server.py"]
    env = {"TOKEN": "literal"}
    reg, _ = _make()

    reg.register("foo", command="python", args=args, env=env)
    args.append("--changed")
    env["TOKEN"] = "changed"

    assert reg._servers["foo"]["args"] == ["server.py"]
    assert reg._servers["foo"]["env"] == {"TOKEN": "literal"}


@pytest.mark.asyncio
async def test_register_rejects_change_while_connected(monkeypatch):
    _fake_create(monkeypatch)
    reg, _ = _make(servers={"foo": {"url": "https://foo/mcp"}})
    await reg.connect(["foo"])

    with pytest.raises(RuntimeError, match="Disconnect"):
        reg.register("foo", url="https://changed/mcp")


# ---------------------------------------------------------------------------
# Connect
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connect_attaches_and_activates(monkeypatch):
    calls = _fake_create(monkeypatch)
    reg, agent = _make(servers={"maas": {"url": "https://x/mcp"}})
    newly = await reg.connect(["maas"])
    assert newly == ["maas"]
    assert reg.connected() == ["maas"]
    assert reg.activated() == ["maas"]
    assert hasattr(agent, "maas")
    assert calls[0][0] == "maas"


@pytest.mark.asyncio
async def test_connect_without_activate(monkeypatch):
    _fake_create(monkeypatch)
    reg, _ = _make(servers={"maas": {"url": "https://x/mcp"}})
    await reg.connect(["maas"], activate=False)
    assert reg.connected() == ["maas"]
    assert reg.activated() == []


@pytest.mark.asyncio
async def test_connect_is_idempotent(monkeypatch):
    calls = _fake_create(monkeypatch)
    reg, _ = _make(servers={"maas": {"url": "https://x/mcp"}})
    await reg.connect(["maas"])
    again = await reg.connect(["maas"])
    assert again == []
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_connect_glob(monkeypatch):
    _fake_create(monkeypatch)
    reg, _ = _make(servers={"conf-a": {"url": "a"}, "conf-b": {"url": "b"}, "other": {"url": "c"}})
    newly = await reg.connect(["conf-*"])
    assert newly == ["conf-a", "conf-b"]


@pytest.mark.asyncio
async def test_connect_uses_approved_oauth_config_and_bound_prompt(monkeypatch):
    calls = _fake_create(monkeypatch)
    reg, _ = _make(
        servers={
            "maas": {
                "url": "https://x/mcp",
                "oauth_manual": True,
                "oauth_scope": "read",
            }
        }
    )

    async def prompt(url):
        return "code"

    reg._bind_oauth_code_prompt(prompt)
    await reg.connect(["maas"])
    _, kwargs = calls[0]
    assert kwargs["oauth_code_prompt"] is prompt
    assert kwargs["oauth_manual"] is True
    assert kwargs["oauth_scope"] == "read"
    assert kwargs["url"] == "https://x/mcp"
    assert kwargs["transport"] == "streamable-http"
    assert kwargs["servers"] == {"maas": {}}


@pytest.mark.asyncio
async def test_connect_requires_approval_before_factory(monkeypatch):
    from nooa_cli.tui.mcp_approval import MCPApprovalRequired

    calls = _fake_create(monkeypatch)
    reg, _ = _make(
        servers={"maas": {"url": "https://x/mcp", "transport": "streamable-http"}},
        approve=False,
    )

    with pytest.raises(MCPApprovalRequired, match="/mcp approve maas"):
        await reg.connect(["maas"])

    assert calls == []


@pytest.mark.asyncio
async def test_approved_config_expands_only_in_exact_factory_override(monkeypatch):
    canary = "sk-host-secret-canary"
    monkeypatch.setenv("OPENAI_API_KEY", canary)
    calls = _fake_create(monkeypatch)
    config = {
        "url": "https://trusted.example/mcp/${OPENAI_API_KEY}",
        "transport": "streamable-http",
        "headers": {"Authorization": "Bearer ${OPENAI_API_KEY}"},
    }
    reg, _ = _make(servers={"maas": config}, approve=False)
    request = reg._approval_request("maas")
    reg._approve("maas", request.confirmation)

    await reg.connect(["maas"])

    factory_kwargs = calls[0][1]
    assert factory_kwargs["url"] == f"https://trusted.example/mcp/{canary}"
    assert factory_kwargs["headers"]["Authorization"] == f"Bearer {canary}"
    assert factory_kwargs["servers"] == {"maas": {}}
    assert config["url"].endswith("${OPENAI_API_KEY}")


@pytest.mark.asyncio
async def test_resolved_value_is_not_reexpanded_by_older_core(monkeypatch):
    monkeypatch.setenv("OUTER", "${OPENAI_API_KEY}")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-read")
    calls = _fake_create(monkeypatch)
    reg, _ = _make(
        servers={
            "maas": {
                "url": "https://trusted.example/${OUTER}",
                "transport": "streamable-http",
            }
        },
        approve=False,
    )
    request = reg._approval_request("maas")
    reg._approve("maas", request.confirmation)

    await reg.connect(["maas"])

    assert calls[0][1]["url"] == "https://trusted.example/${OPENAI_API_KEY}"
    assert calls[0][1]["servers"] == {"maas": {}}


@pytest.mark.asyncio
async def test_connection_error_redacts_approved_secret(monkeypatch):
    secret = "sk-secret-in-failed-url"
    monkeypatch.setenv("MCP_SECRET", secret)
    reg, _ = _make(
        servers={"maas": {"url": "https://example.test/${MCP_SECRET}"}},
        approve=False,
    )
    _approve(reg, "maas")

    import nooa.mcp as mcp_mod

    def _fail(*args, **kwargs):
        raise RuntimeError(f"Could not connect to https://example.test/{secret}")

    monkeypatch.setattr(mcp_mod.MCPManager, "create_from_server", staticmethod(_fail))

    with pytest.raises(RuntimeError) as caught:
        await reg.connect(["maas"])

    assert secret not in str(caught.value)
    assert "${MCP_SECRET}" in str(caught.value)


@pytest.mark.asyncio
async def test_cancelled_connect_stays_pending_until_worker_finishes(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    import nooa.mcp as mcp_mod

    def _blocking_create(*args, **kwargs):
        started.set()
        assert release.wait(timeout=5)
        return _FakeTool()

    monkeypatch.setattr(mcp_mod.MCPManager, "create_from_server", staticmethod(_blocking_create))
    reg, _ = _make(servers={"maas": {"url": "https://x/mcp"}})
    connect_task = asyncio.create_task(reg.connect(["maas"]))
    assert await asyncio.to_thread(started.wait, 1)

    connect_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await connect_task

    assert "maas" in reg._pending
    with pytest.raises(RuntimeError, match="already in progress"):
        await reg.connect(["maas"])

    release.set()
    for _ in range(100):
        if "maas" not in reg._pending:
            break
        await asyncio.sleep(0.01)
    assert "maas" not in reg._pending
    assert reg.connected() == []


@pytest.mark.asyncio
async def test_config_change_invalidates_approval_before_secret_expansion(monkeypatch):
    from nooa_cli.tui.mcp_approval import MCPApprovalRequired

    monkeypatch.setenv("OPENAI_API_KEY", "sk-host-secret-canary")
    calls = _fake_create(monkeypatch)
    reg, _ = _make(
        servers={
            "maas": {
                "url": "https://trusted.example/mcp",
                "transport": "streamable-http",
                "headers": {"Authorization": "Bearer ${OPENAI_API_KEY}"},
            }
        },
        approve=False,
    )
    request = reg._approval_request("maas")
    reg._approve("maas", request.confirmation)
    reg._servers["maas"]["url"] = "https://attacker.example/mcp"

    with pytest.raises(MCPApprovalRequired):
        await reg.connect(["maas"])

    assert calls == []


@pytest.mark.asyncio
async def test_removed_expand_env_vars_field_is_rejected(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-host-secret-canary")
    calls = _fake_create(monkeypatch)
    reg, _ = _make(
        servers={
            "attacker": {
                "url": "https://attacker.example/${OPENAI_API_KEY}",
                "transport": "streamable-http",
                "expand_env_vars": True,
            }
        },
        approve=False,
    )

    with pytest.raises(ValueError, match="unsupported field.*expand_env_vars"):
        await reg.connect(["attacker"])

    assert calls == []


@pytest.mark.asyncio
async def test_unapproved_stdio_command_never_reaches_factory(monkeypatch):
    from nooa_cli.tui.mcp_approval import MCPApprovalRequired

    calls = _fake_create(monkeypatch)
    reg, _ = _make(
        servers={"local": {"command": "sh", "args": ["-c", "touch /tmp/pwned"]}},
        approve=False,
    )

    with pytest.raises(MCPApprovalRequired, match="execute a local process"):
        await reg.connect(["local"])

    assert calls == []


@pytest.mark.asyncio
async def test_agent_attribute_collision_is_rejected_before_factory(monkeypatch):
    calls = _fake_create(monkeypatch)
    reg, _ = _make(servers={"mcp": {"url": "https://x/mcp", "transport": "streamable-http"}})
    reg._agent.mcp = reg

    with pytest.raises(ValueError, match="overwrite existing agent attribute"):
        await reg.connect(["mcp"])

    assert calls == []


@pytest.mark.asyncio
async def test_server_name_with_spaces_connects_using_underscored_agent_attribute(monkeypatch):
    calls = _fake_create(monkeypatch)
    reg, agent = _make(
        servers={"MaaS Jira": {"url": "https://jira.example/mcp", "transport": "streamable-http"}}
    )

    assert await reg.connect(["MaaS Jira"]) == ["MaaS Jira"]
    assert agent.MaaS_Jira is reg["MaaS Jira"]
    assert calls[0][0] == "MaaS Jira"
    assert "self.MaaS_Jira" in reg.status()

    assert await reg.disconnect(["MaaS Jira"]) == ["MaaS Jira"]
    assert not hasattr(agent, "MaaS_Jira")


@pytest.mark.asyncio
async def test_normalized_server_name_collision_is_rejected(monkeypatch):
    calls = _fake_create(monkeypatch)
    reg, _ = _make(
        servers={
            "foo-bar": {"url": "https://one/mcp", "transport": "streamable-http"},
            "foo_bar": {"url": "https://two/mcp", "transport": "streamable-http"},
        }
    )
    await reg.connect(["foo-bar"])

    with pytest.raises(ValueError, match="conflicts with connected server"):
        await reg.connect(["foo_bar"])

    assert len(calls) == 1


@pytest.mark.asyncio
async def test_connection_time_endpoint_override_is_rejected(monkeypatch):
    calls = _fake_create(monkeypatch)
    reg, _ = _make(servers={"maas": {"url": "https://trusted/mcp", "transport": "streamable-http"}})

    with pytest.raises(TypeError, match="endpoint/config overrides"):
        await reg.connect(["maas"], url="https://attacker/mcp")

    assert calls == []


@pytest.mark.asyncio
async def test_connection_time_oauth_override_is_rejected(monkeypatch):
    calls = _fake_create(monkeypatch)
    reg, _ = _make(servers={"maas": {"url": "https://trusted/mcp"}})

    with pytest.raises(TypeError, match="endpoint/config overrides"):
        await reg.connect(["maas"], oauth_redirect_uri="https://attacker/callback")

    assert calls == []


# ---------------------------------------------------------------------------
# Activate / deactivate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_activate_deactivate_membership(monkeypatch):
    _fake_create(monkeypatch)
    reg, _ = _make(servers={"maas": {"url": "x"}})
    await reg.connect(["maas"], activate=False)
    assert reg.activated() == []
    reg.activate(["maas"])
    assert reg.activated() == ["maas"]
    reg.deactivate(["maas"])
    assert reg.activated() == []


@pytest.mark.asyncio
async def test_deactivate_does_not_close_session(monkeypatch):
    _fake_create(monkeypatch)
    reg, _ = _make(servers={"maas": {"url": "x"}})
    await reg.connect(["maas"])
    tool = reg["maas"]
    reg.deactivate(["maas"])
    reg.activate(["maas"])
    assert reg["maas"] is tool
    assert reg.connected() == ["maas"]


@pytest.mark.asyncio
async def test_activate_glob(monkeypatch):
    _fake_create(monkeypatch)
    reg, _ = _make(servers={"a": {"url": "x"}, "b": {"url": "y"}})
    await reg.connect(["*"], activate=False)
    reg.activate(["*"])
    assert reg.activated() == ["a", "b"]


# ---------------------------------------------------------------------------
# Visibility — self.<server> stays hidden from doc(self) either way
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connected_attr_is_hidden_from_doc(monkeypatch):
    _fake_create(monkeypatch)
    reg, agent = _make(servers={"maas": {"url": "x"}})
    await reg.connect(["maas"])  # activated by default
    assert is_hidden_field(agent, "maas") is True
    reg.deactivate(["maas"])
    assert is_hidden_field(agent, "maas") is True


# ---------------------------------------------------------------------------
# Status / <mcp> block
# ---------------------------------------------------------------------------


def test_status_empty():
    reg, _ = _make()
    assert reg.status() == "No MCP servers configured."


def test_status_configured_only():
    reg, _ = _make(servers={"maas": {"url": "x"}})
    out = reg.status()
    assert "Configured MCP servers" in out
    assert "self.mcp.connect(['name'])" in out
    assert "  maas" in out


def test_status_marks_unapproved_server():
    reg, _ = _make(
        servers={"maas": {"url": "https://x/mcp", "transport": "streamable-http"}},
        approve=False,
    )
    assert "approval required" in reg.status()


def test_status_escapes_control_characters_in_untrusted_server_name():
    reg, _ = _make(
        servers={"evil\x1b[2J": {"url": "https://x/mcp", "transport": "streamable-http"}},
        approve=False,
    )

    status = reg.status()

    assert "\x1b" not in status
    assert "evil\\x1b[2J" in status


@pytest.mark.asyncio
async def test_status_connected_inactive(monkeypatch):
    _fake_create(monkeypatch)
    reg, _ = _make(servers={"maas": {"url": "x"}})
    await reg.connect(["maas"], activate=False)
    out = reg.status()
    assert "Connected but inactive" in out
    assert "self.maas" in out
    assert "search" in out  # tool names summarized
    assert "search(" not in out  # but no full signatures


@pytest.mark.asyncio
async def test_status_active_lists_tools_as_free_functions(monkeypatch):
    _fake_create(monkeypatch)
    reg, _ = _make(servers={"maas": {"url": "x"}})
    await reg.connect(["maas"])
    out = reg.status()
    assert "Active MCP servers" in out
    assert "self.maas" in out
    # tool names are summarized on the server row (docs via doc(self.maas))
    assert "search" in out
    assert "get_page" in out
    assert "(2 tools)" in out


@pytest.mark.asyncio
async def test_status_precedence_active_over_connected(monkeypatch):
    _fake_create(monkeypatch)
    reg, _ = _make(servers={"maas": {"url": "x"}})
    await reg.connect(["maas"])
    out = reg.status()
    # an active server appears once, in the Active section (not Configured/inactive)
    assert "Active MCP servers" in out
    assert "self.maas" in out
    assert "Configured MCP servers" not in out
    assert "Connected but inactive" not in out


@pytest.mark.asyncio
async def test_status_truncates_many_tools(monkeypatch):
    many = {f"tool_{i}" for i in range(50)}

    class _BigTool:
        _tool_method_names = frozenset(many)

    def _make_methods():
        pass

    _fake_create(monkeypatch, tool_factory=_BigTool)
    reg, _ = _make(servers={"big": {"url": "x"}})
    await reg.connect(["big"])
    out = reg.status()
    assert "more" in out
    assert "(50 tools)" in out


# ---------------------------------------------------------------------------
# Disconnect
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_disconnect_removes_attr_and_state(monkeypatch):
    _fake_create(monkeypatch)
    reg, agent = _make(servers={"maas": {"url": "x"}})
    await reg.connect(["maas"])
    out = await reg.disconnect(["maas"])
    assert out == ["maas"]
    assert reg.connected() == []
    assert reg.activated() == []
    assert not hasattr(agent, "maas")


# ---------------------------------------------------------------------------
# attach / detach + context block
# ---------------------------------------------------------------------------


def test_attach_registers_context_block():
    reg, agent = _make(servers={"maas": {"url": "x"}})
    assert "mcp" in agent.context_manager


@pytest.mark.asyncio
async def test_detach_disconnects_and_removes_block(monkeypatch):
    _fake_create(monkeypatch)
    reg, agent = _make(servers={"maas": {"url": "x"}})
    await reg.connect(["maas"])
    reg.detach()
    assert "mcp" not in agent.context_manager
    assert reg.connected() == []


def test_getitem_raises_when_not_connected():
    reg, _ = _make(servers={"maas": {"url": "x"}})
    with pytest.raises(KeyError):
        reg["maas"]


@pytest.mark.asyncio
async def test_status_three_state_inactive_section(monkeypatch):
    """A deactivated-but-connected server stays visible in its own section."""
    _fake_create(monkeypatch)
    reg, _ = _make(servers={"maas": {"url": "x"}, "jira": {"url": "y"}})
    await reg.connect(["maas"])  # active
    await reg.connect(["jira"], activate=False)  # connected, inactive
    out = reg.status()
    assert "Active MCP servers" in out
    assert "Connected but inactive" in out
    assert "self.maas" in out
    assert "self.jira" in out


@pytest.mark.asyncio
async def test_deactivate_moves_to_inactive_section(monkeypatch):
    _fake_create(monkeypatch)
    reg, _ = _make(servers={"maas": {"url": "x"}})
    await reg.connect(["maas"])
    assert "Active MCP servers" in reg.status()
    reg.deactivate(["maas"])
    out = reg.status()
    assert "Connected but inactive" in out
    assert "Active MCP servers" not in out


def test_mcp_lifecycle_command_is_not_agent_exposed():
    """Approval/lifecycle stays in the native TUI command, outside the agent skill."""
    from nooa.skill import get_slash_commands

    reg, _ = _make(servers={"maas": {"url": "x"}})
    assert [meta.name for meta, _method in get_slash_commands(reg)] == ["mcp-add"]


@pytest.mark.asyncio
async def test_mcp_add_slash_command_returns_agent_task(monkeypatch, tmp_path):
    """/mcp-add hands the server details to the agent (output_to_agent=True)."""
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path))
    reg, _ = _make(servers={"maas": {"url": "x"}})
    out = await reg.mcp_add_command(
        "maas-gdrive https://maas.prd.astra.nvidia.com/maas/gdrive/mcp streamable-http"
    )
    assert "maas-gdrive" in out
    assert "tui.mcp_servers" in out
    assert "settings.yaml" in out
    # Includes the currently-configured servers for context.
    assert "maas" in out


@pytest.mark.asyncio
async def test_mcp_add_slash_command_empty_shows_usage():
    reg, _ = _make(servers={})
    out = await reg.mcp_add_command("")
    assert "Usage: /mcp-add" in out


def test_mcp_add_slash_command_outputs_to_agent():
    from nooa.skill import get_slash_commands

    reg, _ = _make(servers={})
    meta = next(m for m, _ in get_slash_commands(reg) if m.name == "mcp-add")
    assert meta.output_to_agent is True
