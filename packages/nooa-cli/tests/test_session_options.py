# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from nooa_cli.interactive.options import SessionOptions
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
    native = SessionOptions.from_native_config(Config.load(working_dir=str(workspace)))
    acp = SessionOptions.load(workspace)
    assert native == acp
    assert acp.skills_dirs == [skills, conventional]
    assert acp.active_skills == ["shared.skill"]
    assert acp.inactive_skills == ["disabled.skill"]
    assert acp.memory == "session"
    assert acp.reflection is True
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
    native = SessionOptions.from_native_config(config)
    acp = SessionOptions.load(tmp_path)
    assert native == acp
    for key in ("keep_going", "keep_going_model"):
        assert not hasattr(config.tui, key)
        assert not hasattr(acp, key)
        exported = settings_to_dict(config)
        assert key not in exported["coding"]
        assert key not in exported["tui"]


@pytest.mark.parametrize("with_canonical", [False, True])
def test_shared_writer_preserves_siblings_and_removes_legacy_aliases(tmp_path, with_canonical):
    import yaml
    from nooa_cli.interactive.settings import delete_settings_value, write_settings_updates

    project = tmp_path / ".nooa"
    project.mkdir()
    (project / "settings.yaml").write_text(
        "tui:\n  memory_agents: {first: session, second: project}\n"
        + (
            "coding:\n  memory_agents: {first: project, second: project}\n"
            if with_canonical
            else ""
        )
    )
    path, data = write_settings_updates(
        {("tui", "memory_agents", "first"): "off"}, workspace=tmp_path
    )
    assert data["coding"]["memory_agents"] == {"first": "off", "second": "project"}
    _, data, deleted = delete_settings_value(
        ("coding", "memory_agents", "first"), workspace=tmp_path
    )
    assert deleted
    assert data["coding"]["memory_agents"] == {"second": "project"}
    assert data["tui"]["memory_agents"] == {"second": "project"}
    assert yaml.safe_load(path.read_text()) == data


def test_export_keeps_behavior_in_shared_namespace():
    from nooa_cli.tui.settings import settings_to_dict

    data = settings_to_dict(Config())
    assert "reflection" in data["coding"]
    assert "summarization" in data["coding"]
    assert "reflection" not in data["tui"]
    assert "summarization" not in data["agent"]
    assert "theme" in data["tui"]
