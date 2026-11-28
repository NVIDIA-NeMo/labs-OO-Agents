# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WorkspaceSettings.set_live_context: saved for future sessions and applied to the running agent."""

from types import SimpleNamespace

import pytest
import yaml
from atom_test_agents import ATOM_SPEC
from nooa_atom.agent.factory import create_session_agent
from nooa_atom.session.options import SessionOptions
from nooa_atom.workspace.options import AtomOptions
from nooa_atom.workspace.workspace_settings import WorkspaceSettings

from nooa.context_blocks import AppendedLiveContext, NoLiveContext, RenderConfig
from nooa.storage import InMemoryStorageManager
from nooa.unifiedllm import FakeLLMClient


@pytest.fixture
def workspace_settings(tmp_path, monkeypatch):
    workspace = tmp_path / "project"
    workspace.mkdir()
    (workspace / ".nooa").mkdir()
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    monkeypatch.delenv("NEMO_OO_PROJECT_DIR", raising=False)
    ws = WorkspaceSettings(AtomOptions(working_dir=str(workspace)))
    agent = SimpleNamespace(
        render_config=RenderConfig(),
        llm=None,
        skills=SimpleNamespace(activated=lambda: [], mcp=SimpleNamespace(connected=lambda: [])),
    )
    ws._agent = agent
    return ws, agent, workspace


def _saved(workspace):
    return yaml.safe_load((workspace / ".nooa" / "settings.yaml").read_text())


def test_set_off_is_saved_and_applied_now(workspace_settings):
    ws, agent, workspace = workspace_settings
    message = ws.set_live_context("off")
    assert _saved(workspace)["atom"]["render_config"]["live_context"] == {"kind": "off"}
    assert agent.render_config.live_context == NoLiveContext()
    assert "off" in message and "next model call" in message
    assert AtomOptions.load(workspace).render_config.live_context == NoLiveContext()


def test_set_append_with_a_cadence_is_saved_and_applied(workspace_settings):
    ws, agent, workspace = workspace_settings
    ws.set_live_context("append", every_n_turns=3)
    assert _saved(workspace)["atom"]["render_config"]["live_context"] == {
        "kind": "append",
        "every_n_turns": 3,
    }
    assert agent.render_config.live_context == AppendedLiveContext(every_n_turns=3)
    assert AtomOptions.load(workspace).render_config.live_context == AppendedLiveContext(
        every_n_turns=3
    )


def test_invalid_requests_are_refused_before_saving(workspace_settings):
    ws, agent, workspace = workspace_settings
    with pytest.raises(ValueError):
        ws.set_live_context("sometimes")
    with pytest.raises(ValueError, match="append"):
        ws.set_live_context("dynamic", every_n_turns=3)
    with pytest.raises(ValueError):
        ws.set_live_context("append", every_n_turns=0)
    assert not (workspace / ".nooa" / "settings.yaml").exists()
    assert agent.render_config.live_context == AppendedLiveContext()


def test_status_shows_the_saved_and_current_policy(workspace_settings):
    ws, agent, workspace = workspace_settings
    assert ws.status()["saved"]["live_context"] is None
    assert ws.status()["current"]["live_context"] == {"kind": "append", "every_n_turns": 5}
    ws.set_live_context("off")
    status = ws.status()
    assert status["saved"]["live_context"] == {"kind": "off"}
    assert status["current"]["live_context"] == {"kind": "off"}


def test_clear_restores_the_default_for_fresh_sessions(workspace_settings):
    ws, agent, workspace = workspace_settings
    ws.set_live_context("off")
    message = ws.clear_live_context()
    assert "unchanged" in message
    assert AtomOptions.load(workspace).render_config is None or (
        AtomOptions.load(workspace).render_config.live_context == AppendedLiveContext()
    )
    assert agent.render_config.live_context == NoLiveContext()  # the running agent keeps its choice


async def test_the_live_context_slash_command_is_listed_and_sets_the_policy(tmp_path, monkeypatch):
    """/live-context on a real Atom agent: listed with help, reports, persists and applies."""
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    options = SessionOptions(workspace=workspace, agent_spec=ATOM_SPEC, llm=FakeLLMClient())
    agent = create_session_agent(options, InMemoryStorageManager())
    try:
        agent.slash_commands.refresh_skill_commands()
        commands = {command.name: command for command in agent.slash_commands.commands()}
        assert "live-context" in commands
        assert commands["live-context"].argument_hint == "<dynamic|off|append> [every_n_turns]"
        assert "live <context>" in commands["live-context"].description
        assert commands["live-context"].output_to_agent is False

        shown = str(await agent.slash_commands.invoke("live-context", ""))
        assert "append every 5 turns" in shown and "default" in shown

        result = str(await agent.slash_commands.invoke("live-context", "off"))
        assert "Live context set to off" in result and "next model call" in result
        assert agent.render_config.live_context == NoLiveContext()
        assert _saved(workspace)["atom"]["render_config"]["live_context"] == {"kind": "off"}

        result = str(await agent.slash_commands.invoke("live-context", "append 3"))
        assert agent.render_config.live_context == AppendedLiveContext(every_n_turns=3)
        assert "Not changed" in str(await agent.slash_commands.invoke("live-context", "dynamic 3"))
        assert agent.render_config.live_context == AppendedLiveContext(every_n_turns=3)
    finally:
        await agent.queue_manager.shutdown()
        await agent.aclose()


def test_switching_kind_over_an_inherited_cadence_is_valid_in_later_sessions(
    workspace_settings, tmp_path
):
    """Severin's case: user scope append/3, then /live-context off at project scope."""
    ws, agent, workspace = workspace_settings
    user_dir = tmp_path / "user"
    user_dir.mkdir(exist_ok=True)
    (user_dir / "settings.yaml").write_text(
        "atom:\n  default_model: my-model\n  render_config:\n    live_context:\n"
        "      kind: append\n      every_n_turns: 3\n"
    )
    assert AtomOptions.load(workspace).render_config.live_context == AppendedLiveContext(
        every_n_turns=3
    )
    ws.set_live_context("off")
    saved = _saved(workspace)["atom"]["render_config"]["live_context"]
    assert saved == {"kind": "off"}  # the complete policy, self-contained
    reloaded = AtomOptions.load(workspace)
    assert reloaded.render_config.live_context == NoLiveContext()
    assert reloaded.default_model == "my-model"
