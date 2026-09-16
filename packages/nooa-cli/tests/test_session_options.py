# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from nooa_cli.interactive.options import SessionOptions
from nooa_cli.tui.bootstrap import session_options_from_config
from nooa_cli.tui.config import Config


def test_native_and_acp_resolve_same_workspace_behavior(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    project = workspace / ".nooa"
    project.mkdir(parents=True)
    skills = workspace / "skills"
    skills.mkdir()
    conventional = workspace / ".agents" / "skills"
    conventional.mkdir(parents=True)
    user = tmp_path / "user"
    user.mkdir()
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    monkeypatch.setattr(Path, "home", lambda: user)
    monkeypatch.chdir(tmp_path)  # The host process need not live in the workspace.
    (project / "settings.yaml").write_text("""
tui:
  active_skills: [legacy.skill]
  memory: project
  reflection: true
agent:
  summarization:
    max_tokens: 2000
    policy: none
    preserve_recent: 3
coding:
  additional_skills_dirs: [skills]
  active_skills: [shared.skill]
  inactive_skills: [disabled.skill]
  memory: session
  summarization:
    max_tokens: 4000
""")
    native = session_options_from_config(Config.load(working_dir=str(workspace)))
    acp = SessionOptions.load(workspace)
    assert native == acp
    assert acp.skills_dirs == [skills, conventional]
    assert acp.active_skills == ["shared.skill"]
    assert acp.inactive_skills == ["disabled.skill"]
    assert not hasattr(acp, "memory")
    assert not hasattr(acp, "reflection")
    assert acp.summarization.max_tokens == 4000
    assert acp.summarization.policy == "none"
    assert acp.summarization.preserve_recent == 3


@pytest.mark.parametrize("section", ["tui", "coding"])
def test_retired_keep_going_settings_are_ignored_in_both_hosts(tmp_path, monkeypatch, section):
    from nooa_cli.tui.settings import settings_to_dict

    project = tmp_path / ".nooa"
    project.mkdir()
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    (project / "settings.yaml").write_text(
        f"{section}:\n  keep_going: true\n  keep_going_model: obsolete-judge\n"
    )
    config = Config.load(working_dir=str(tmp_path))
    native = session_options_from_config(config)
    acp = SessionOptions.load(tmp_path)
    assert native == acp
    for key in ("keep_going", "keep_going_model"):
        assert not hasattr(config.tui, key)
        assert not hasattr(acp, key)
        exported = settings_to_dict(config)
        assert key not in exported["coding"]
        assert key not in exported["tui"]


def test_export_keeps_behavior_in_shared_namespace():
    from nooa_cli.tui.settings import settings_to_dict

    data = settings_to_dict(Config())
    assert "reflection" not in data["coding"]
    assert "summarization" in data["coding"]
    assert "reflection" not in data["tui"]
    assert "summarization" not in data["agent"]
    assert "theme" in data["tui"]


@pytest.mark.parametrize("section", ["tui", "coding"])
def test_native_ignores_persisted_agent_spec_and_retired_memory(tmp_path, monkeypatch, section):
    project = tmp_path / ".nooa"
    project.mkdir()
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    (project / "settings.yaml").write_text(
        f"{section}:\n  agent_spec: ./untrusted.py:Agent\n"
        "  memory: project\n  reflection_agents: nooa_cli\n"
    )
    config = Config.load(working_dir=str(tmp_path))
    options = session_options_from_config(config)
    assert options.agent_spec is None
    assert not hasattr(config.tui, "memory")
    assert not hasattr(config.tui, "reflection_agents")
    explicit = Config.load(working_dir=str(tmp_path), agent="./explicit.py:Agent")
    assert session_options_from_config(explicit).agent_spec == "./explicit.py:Agent"
