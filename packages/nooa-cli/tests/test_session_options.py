# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from nooa_cli.interactive.options import SessionOptions
from nooa_cli.tui.config import Config


def test_native_and_acp_resolve_same_workspace_behavior(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    project = workspace / ".nooa"
    project.mkdir(parents=True)
    skills = workspace / "skills"
    skills.mkdir()
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
  keep_going: true
agent:
  summarization:
    max_tokens: 2000
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
    assert acp.skills_dirs == [skills]
    assert acp.active_skills == ["shared.skill"]
    assert acp.inactive_skills == ["disabled.skill"]
    assert acp.memory == "session"
    assert acp.keep_going is True
    assert acp.summarization.max_tokens == 4000
