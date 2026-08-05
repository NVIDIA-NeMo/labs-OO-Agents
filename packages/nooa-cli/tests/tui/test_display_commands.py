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
    assert result.outputs[0].rows == [
        ["/project-operation", "<target>", "Operate on this project"]
    ]


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
