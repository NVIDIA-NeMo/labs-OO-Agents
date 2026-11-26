# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An Atom agent's RenderConfig comes from the workspace settings, the session or the Gym harness."""

import pytest
from atom_test_agents import ATOM_SPEC
from nooa_atom.agent.factory import create_session_agent
from nooa_atom.gym import AtomHarness
from nooa_atom.session.options import SessionOptions
from nooa_atom.workspace.options import AtomOptions
from nooa_atom.workspace.settings import resolve_behavior_settings

from nooa.context_blocks import AppendedLiveContext, DynamicLiveContext, NoLiveContext, RenderConfig
from nooa.context_blocks.renderers.cached import CachedBlockFormatter
from nooa.storage import InMemoryStorageManager
from nooa.unifiedllm import FakeLLMClient

OFF_SETTINGS = "atom:\n  render_config:\n    live_context:\n      kind: off\n"


@pytest.fixture(autouse=True)
def _no_settings_env(monkeypatch):
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    path = tmp_path / "workspace"
    path.mkdir()
    user_config = tmp_path / "user-config"
    user_config.mkdir()
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user_config))
    return path


async def _close(agent):
    await agent.queue_manager.shutdown()
    await agent.aclose()


def _policy_of(agent):
    formatter = agent.render_config.block_formatter
    assert isinstance(formatter, CachedBlockFormatter)
    return formatter.live_context


def test_workspace_settings_carry_a_render_config():
    values = resolve_behavior_settings(
        {"atom": {"render_config": {"live_context": {"kind": "append", "every_n_turns": 2}}}}
    )
    assert AtomOptions(**values).render_config.live_context == AppendedLiveContext(every_n_turns=2)
    assert AtomOptions().render_config is None


def test_settings_file_is_loaded_into_options(workspace):
    config_dir = workspace / ".nooa"
    config_dir.mkdir()
    (config_dir / "settings.yaml").write_text(OFF_SETTINGS)
    assert AtomOptions.load(workspace).render_config.live_context == NoLiveContext()


def test_session_options_accept_a_mapping_and_children_inherit(workspace):
    options = SessionOptions(workspace=workspace, agent_spec=ATOM_SPEC)
    assert options.render_config is None
    options = SessionOptions(
        workspace=workspace, agent_spec=ATOM_SPEC, render_config={"live_context": "dynamic"}
    )
    assert options.render_config.live_context == DynamicLiveContext()
    child = options.inherit(name="child", kept=False)
    assert child.render_config is options.render_config


async def test_the_session_render_config_wins_over_the_workspace(workspace):
    config_dir = workspace / ".nooa"
    config_dir.mkdir()
    (config_dir / "settings.yaml").write_text(OFF_SETTINGS)
    options = SessionOptions(
        workspace=workspace,
        agent_spec=ATOM_SPEC,
        llm=FakeLLMClient(),
        render_config=RenderConfig(live_context=DynamicLiveContext()),
    )
    agent = create_session_agent(options, InMemoryStorageManager())
    try:
        assert _policy_of(agent) == DynamicLiveContext()
    finally:
        await _close(agent)


async def test_the_workspace_render_config_applies_without_a_session_override(workspace):
    config_dir = workspace / ".nooa"
    config_dir.mkdir()
    (config_dir / "settings.yaml").write_text(OFF_SETTINGS)
    options = SessionOptions(workspace=workspace, agent_spec=ATOM_SPEC, llm=FakeLLMClient())
    agent = create_session_agent(options, InMemoryStorageManager())
    try:
        assert _policy_of(agent) == NoLiveContext()
    finally:
        await _close(agent)


async def test_without_either_the_agent_keeps_its_default(workspace):
    options = SessionOptions(workspace=workspace, agent_spec=ATOM_SPEC, llm=FakeLLMClient())
    agent = create_session_agent(options, InMemoryStorageManager())
    try:
        assert _policy_of(agent) == AppendedLiveContext()
    finally:
        await _close(agent)


def test_the_gym_harness_forwards_the_render_config_into_the_session_options(tmp_path):
    harness = AtomHarness(
        FakeLLMClient(), workdir=tmp_path, render_config={"live_context": {"kind": "off"}}
    )
    assert harness.options.render_config.live_context == NoLiveContext()
    config = RenderConfig(live_context=DynamicLiveContext())
    assert (
        AtomHarness(FakeLLMClient(), workdir=tmp_path, render_config=config).options.render_config
        is config
    )
    assert AtomHarness(FakeLLMClient(), workdir=tmp_path).options.render_config is None
    with pytest.raises(ValueError):
        AtomHarness(FakeLLMClient(), workdir=tmp_path, render_config={"live_context": "sometimes"})


def test_a_non_mapping_render_config_is_an_invalid_setting_that_falls_back(workspace, caplog):
    """A ValueError (not a TypeError) so pydantic reports it and the loader keeps the defaults."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AtomOptions(render_config=42)
    with pytest.raises(ValidationError):
        SessionOptions(workspace=workspace, agent_spec=ATOM_SPEC, render_config=42)
    config_dir = workspace / ".nooa"
    config_dir.mkdir()
    (config_dir / "settings.yaml").write_text("atom:\n  render_config: 42\n")
    assert AtomOptions.load(workspace).render_config is None
    assert "Invalid Atom render_config" in caplog.text  # only that key is dropped


def test_a_nested_override_replaces_a_null_legacy_layer():
    values = resolve_behavior_settings(
        {"tui": {"render_config": None}, "atom": {"render_config": {"live_context": "off"}}}
    )
    assert AtomOptions(**values).render_config.live_context == NoLiveContext()
    # Two dict layers still merge key by key.
    values = resolve_behavior_settings(
        {
            "tui": {"render_config": {"live_context": {"kind": "append", "every_n_turns": 2}}},
            "atom": {"render_config": {}},
        }
    )
    assert AtomOptions(**values).render_config.live_context == AppendedLiveContext(every_n_turns=2)


def test_a_higher_layer_replaces_the_live_context_policy_as_a_whole(tmp_path, monkeypatch):
    """User scope append/3, project scope off: the policy is not field-merged into
    an invalid {kind: off, every_n_turns: 3}; the other settings survive."""
    workspace = tmp_path / "workspace"
    (workspace / ".nooa").mkdir(parents=True)
    user_config = tmp_path / "user-config"
    user_config.mkdir()
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user_config))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    (user_config / "settings.yaml").write_text(
        "atom:\n  default_model: my-model\n  render_config:\n    live_context:\n"
        "      kind: append\n      every_n_turns: 3\n"
    )
    (workspace / ".nooa" / "settings.yaml").write_text(
        "atom:\n  render_config:\n    live_context:\n      kind: off\n"
    )
    options = AtomOptions.load(workspace)
    assert options.render_config.live_context == NoLiveContext()
    assert options.default_model == "my-model"


def test_an_invalid_render_config_alone_keeps_the_other_settings(workspace, caplog):
    config_dir = workspace / ".nooa"
    config_dir.mkdir()
    (config_dir / "settings.yaml").write_text(
        "atom:\n  default_model: my-model\n  render_config:\n    live_context:\n      kind: sometimes\n"
    )
    options = AtomOptions.load(workspace)
    assert options.render_config is None
    assert options.default_model == "my-model"
    assert "Invalid Atom render_config" in caplog.text
