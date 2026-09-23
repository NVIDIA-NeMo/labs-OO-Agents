# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""/skills activate|deactivate|add must never copy a user's personal
settings.yaml entries into the shared, committed project settings file.
"""

import yaml
from nooa_coder.coding.agent import CodingAgent
from nooa_coder.coding.slash_commands import CodingSlashCommandRegistry
from nooa_coder.interactive.controls import SkillsControl
from nooa_coder.interactive.options import SessionOptions

from nooa.unifiedllm import FakeLLMClient


async def test_deactivate_does_not_leak_a_users_personal_active_skill(tmp_path, monkeypatch):
    workspace = tmp_path / "project"
    workspace.mkdir()

    # A personal, user-scope setting that only this user has -- never
    # committed to the shared project file.
    user_dir = tmp_path / "user-config"
    user_dir.mkdir()
    (user_dir / "settings.yaml").write_text(
        yaml.safe_dump({"coding": {"active_skills": ["personal.secret-skill"]}})
    )
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user_dir))

    agent = CodingAgent(llm=FakeLLMClient(), cwd=workspace)
    try:
        config = SessionOptions(working_dir=str(workspace))
        control = SkillsControl(agent, config, workspace=workspace)

        result = await control.invoke("deactivate nemo.methodwriting")
        assert result.success, str(result)

        project_settings = yaml.safe_load((workspace / ".nooa" / "settings.yaml").read_text())
        persisted_active = project_settings["coding"]["active_skills"]
        assert "personal.secret-skill" not in persisted_active
        assert "nemo.methodwriting" not in persisted_active
    finally:
        await agent.close()


async def test_add_skills_dir_does_not_leak_a_users_personal_directory(tmp_path, monkeypatch):
    workspace = tmp_path / "project"
    workspace.mkdir()
    extra_dir = tmp_path / "extra-skills"
    extra_dir.mkdir()

    user_dir = tmp_path / "user-config"
    user_dir.mkdir()
    (user_dir / "settings.yaml").write_text(
        yaml.safe_dump(
            {"coding": {"additional_skills_dirs": [str(tmp_path / "personal-only-dir")]}}
        )
    )
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user_dir))

    agent = CodingAgent(llm=FakeLLMClient(), cwd=workspace)
    try:
        config = SessionOptions(working_dir=str(workspace))
        command_registry = CodingSlashCommandRegistry(agent)
        control = SkillsControl(
            agent, config, workspace=workspace, command_registry=command_registry
        )

        result = await control.invoke(f"add {extra_dir}")
        assert result.success, str(result)

        project_settings = yaml.safe_load((workspace / ".nooa" / "settings.yaml").read_text())
        persisted_dirs = project_settings["coding"]["additional_skills_dirs"]
        assert not any("personal-only-dir" in entry for entry in persisted_dirs)
        assert any(str(extra_dir) in entry for entry in persisted_dirs)
    finally:
        await agent.close()
