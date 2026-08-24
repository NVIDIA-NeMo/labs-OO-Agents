# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Display preference slash commands."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml
from nooa_cli.tui.commands import ShowDiffsCommand, ShowPythonCommand, SkillsCommand
from nooa_cli.tui.config import TUIConfig


def _command(command_type, config: TUIConfig):
    return command_type(MagicMock(), config, MagicMock())


def test_command_families_have_one_help_row() -> None:
    assert list(SkillsCommand.help_text()) == [
        "/skills <list|add DIR|commands|activate ID|deactivate ID>"
    ]
    assert list(ShowPythonCommand.help_text()) == ["/show-python [status|on|off]"]
    assert list(ShowDiffsCommand.help_text()) == ["/show-diffs [status|on|off]"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command_type", "setting"),
    [(ShowPythonCommand, "show_python"), (ShowDiffsCommand, "show_diffs")],
)
async def test_display_toggle_is_saved_to_project_settings(
    command_type, setting: str, tmp_path, monkeypatch
) -> None:
    project_dir = tmp_path / ".nooa"
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project_dir))
    config = TUIConfig()

    result = await _command(command_type, config).execute(["off"])

    assert result.success is True
    assert getattr(config, setting) is False
    saved = yaml.safe_load((project_dir / "settings.yaml").read_text())
    assert saved["tui"][setting] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("command_type", [ShowPythonCommand, ShowDiffsCommand])
async def test_display_toggle_without_action_reports_status(command_type) -> None:
    result = await _command(command_type, TUIConfig()).execute([])

    assert result.success is True
    assert len(result.outputs) == 1
    assert "on" in result.outputs[0].content or "off" in result.outputs[0].content


@pytest.mark.asyncio
async def test_skills_commands_lists_extensions_without_requiring_skill_registry() -> None:
    registry = SimpleNamespace(
        _user_skills={
            "project-operation": SimpleNamespace(
                argument_hint="<target>", description="Operate on this project"
            )
        }
    )
    command = SkillsCommand(
        MagicMock(), TUIConfig(), SimpleNamespace(), registry=registry, skills_dirs=[]
    )

    result = await command.execute(["commands"])

    assert result.success is True
    assert result.outputs[0].rows == [["/project-operation", "<target>", "Operate on this project"]]


@pytest.mark.asyncio
async def test_skills_add_discovers_immediately_and_persists(tmp_path, monkeypatch) -> None:
    from nooa_cli.tui.commands import CommandRegistry

    from nooa.skill_registry import SkillRegistry

    project_dir = tmp_path / ".nooa"
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project_dir))
    skills_root = tmp_path / "shared-skills"
    skill_dir = skills_root / "review-code"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: review-code\ndescription: Review code\n---\nReview the code.\n"
    )
    library_dir = skills_root / "local_tool"
    library_dir.mkdir()
    (library_dir / "pyproject.toml").write_text(
        '[project]\nname = "local-tool"\n\n'
        '[project.entry-points."nooa.skills"]\n'
        '"local.tool" = "local_tool:LocalTool"\n'
    )
    (library_dir / "__init__.py").write_text(
        'from nooa.skill import Skill\n\nclass LocalTool(Skill):\n    """A local tool."""\n'
    )

    agent = SimpleNamespace(context_manager=MagicMock(), cwd=tmp_path)
    agent.skills = SkillRegistry(agent)
    config = TUIConfig(skills_dirs=[], additional_skills_dirs=[])
    registry = CommandRegistry(
        config=config,
        agent=agent,
        frontend=MagicMock(),
        skills_dirs=config.skills_dirs,
    )
    agent._command_registry = registry

    command = registry.get_command("skills")
    assert command is not None
    result = await command.execute(["add", str(skills_root)])

    assert result.success is True
    assert skills_root.resolve() in registry.skills_dirs
    assert "cmd.review-code" in agent.skills.discovered()
    assert "local.tool" in agent.skills.discovered()
    assert "review-code" in registry._user_skills
    saved = yaml.safe_load((project_dir / "settings.yaml").read_text())
    assert saved["tui"]["additional_skills_dirs"] == [str(skills_root.resolve())]


@pytest.mark.asyncio
async def test_skills_activate_and_deactivate_are_persisted(tmp_path, monkeypatch) -> None:
    from nooa.skill_registry import SkillRegistry

    project_dir = tmp_path / ".nooa"
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project_dir))
    agent = SimpleNamespace(context_manager=MagicMock(), cwd=tmp_path)
    agent.skills = SkillRegistry(agent)
    agent.skills.register("local.reconnect", SimpleNamespace())
    config = TUIConfig(active_skills=[], inactive_skills=["local.reconnect"])
    command = SkillsCommand(MagicMock(), config, agent, skills_dirs=[])

    activated = await command.execute(["activate", "local.reconnect"])

    assert activated.success is True
    assert config.active_skills == ["local.reconnect"]
    assert config.inactive_skills == []
    saved = yaml.safe_load((project_dir / "settings.yaml").read_text())
    assert saved["tui"]["active_skills"] == ["local.reconnect"]
    assert saved["tui"]["inactive_skills"] == []

    deactivated = await command.execute(["deactivate", "local.reconnect"])

    assert deactivated.success is True
    assert config.active_skills == []
    assert config.inactive_skills == ["local.reconnect"]
    saved = yaml.safe_load((project_dir / "settings.yaml").read_text())
    assert saved["tui"]["active_skills"] == []
    assert saved["tui"]["inactive_skills"] == ["local.reconnect"]


@pytest.mark.asyncio
async def test_skills_does_not_persist_an_activation_that_did_not_take_effect(
    tmp_path, monkeypatch
) -> None:
    from nooa.skill_registry import SkillRegistry

    project_dir = tmp_path / ".nooa"
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project_dir))
    agent = SimpleNamespace(context_manager=MagicMock(), cwd=tmp_path)
    agent.skills = SkillRegistry(agent)
    agent.skills.register("local.broken", SimpleNamespace())
    monkeypatch.setattr(agent.skills, "activate", lambda _patterns: None)
    config = TUIConfig(active_skills=[])
    command = SkillsCommand(MagicMock(), config, agent, skills_dirs=[])

    result = await command.execute(["activate", "local.broken"])

    assert result.success is False
    assert result.outputs[0].content == "Failed to activate `local.broken`"
    assert config.active_skills == []
    assert not (project_dir / "settings.yaml").exists()


@pytest.mark.asyncio
async def test_skills_does_not_persist_a_deactivation_that_did_not_take_effect(
    tmp_path, monkeypatch
) -> None:
    from nooa.skill_registry import SkillRegistry

    project_dir = tmp_path / ".nooa"
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project_dir))
    agent = SimpleNamespace(context_manager=MagicMock(), cwd=tmp_path)
    agent.skills = SkillRegistry(agent)
    agent.skills.register("local.stuck", SimpleNamespace())
    agent.skills.activate(["local.stuck"])
    monkeypatch.setattr(agent.skills, "deactivate", lambda _patterns: None)
    config = TUIConfig(active_skills=["local.stuck"], inactive_skills=[])
    command = SkillsCommand(MagicMock(), config, agent, skills_dirs=[])

    result = await command.execute(["deactivate", "local.stuck"])

    assert result.success is False
    assert result.outputs[0].content == "Failed to deactivate `local.stuck`"
    assert config.active_skills == ["local.stuck"]
    assert config.inactive_skills == []
    assert not (project_dir / "settings.yaml").exists()
