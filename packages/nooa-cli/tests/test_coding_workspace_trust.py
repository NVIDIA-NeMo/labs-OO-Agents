# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The coding agent in an untrusted workspace (nooa.workspace_trust)."""

import sys
from pathlib import Path

import pytest
from nooa_cli.coding import CodingAgent
from nooa_cli.coding.settings import load_coding_skills_dirs, untrusted_workspace_notice
from nooa_cli.coding.slash_commands import CodingSlashCommandRegistry

from nooa.unifiedllm import FakeLLMClient
from nooa.workspace_trust import HOME_REFUSAL, is_trusted

_SKILL_CLASS = 'from nooa.skill import Skill\n\n\nclass {name}(Skill):\n    """Probe."""\n'


@pytest.fixture
def user_dir(tmp_path, monkeypatch, untrusted_folders):
    path = tmp_path / "user"
    path.mkdir()
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(path))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    return path


@pytest.fixture
def workspace(tmp_path):
    """A workspace with code and text skills in .nooa/skills, .claude/skills and .claude/commands."""
    root = tmp_path / "work"
    evil = root / ".nooa" / "skills" / "evil"
    evil.mkdir(parents=True)
    (evil / "pyproject.toml").write_text('[project]\nname = "evil"\nversion = "0.1.0"\n')
    (evil / "__init__.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(tmp_path / 'evil-marker')!r}).write_text('x')\n"
        + _SKILL_CLASS.format(name="Evil")
    )
    notes = root / ".nooa" / "skills" / "notes"
    notes.mkdir()
    (notes / "SKILL.md").write_text("---\nname: notes\ndescription: Notes.\n---\n\nHi.\n")
    claude = root / ".claude" / "skills"
    claude.mkdir(parents=True)
    (claude / "probe.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(tmp_path / 'probe-marker')!r}).write_text('x')\n"
        + _SKILL_CLASS.format(name="Probe")
    )
    hello = claude / "hello"
    hello.mkdir()
    (hello / "SKILL.md").write_text("---\nname: hello\ndescription: Say hello.\n---\n\nHello.\n")
    yield root
    sys.modules.pop("evil", None)
    skills = str(root / ".nooa" / "skills")
    if skills in sys.path:
        sys.path.remove(skills)


def _agent(root: Path) -> CodingAgent:
    # Built as nooa-acp's _build_agent builds it.
    return CodingAgent(
        llm=FakeLLMClient(),
        cwd=root,
        libs_dir=root / ".nooa" / "skills",
        skills_dirs=load_coding_skills_dirs(root),
    )


_NOTICE = "This folder isn't trusted, so its skills are off. Type /trust to allow."


def _local_and_cmd(agent: CodingAgent) -> list[str]:
    return [n for n in agent.skills.discovered() if n.startswith(("local.", "cmd.", "ext."))]


def test_untrusted_workspace_loads_nothing_from_its_skill_dirs(user_dir, workspace, tmp_path):
    agent = _agent(workspace)
    assert not (tmp_path / "evil-marker").exists()
    assert not (tmp_path / "probe-marker").exists()
    assert str(workspace / ".nooa" / "skills") not in sys.path
    # Not even the SKILL.md text skills.
    assert _local_and_cmd(agent) == []
    assert load_coding_skills_dirs(workspace) == []


def test_the_agent_sees_nothing_about_trust(user_dir, workspace):
    agent = _agent(workspace)
    status = agent.skills.status().lower()
    assert "trust" not in status and "off" not in status
    assert agent.skills.notices() == []
    assert not hasattr(agent, "workspace_untrusted_notice")
    assert not hasattr(agent, "trust_workspace")
    assert not any("trust" in name for name in agent.skills.discovered())


def test_the_user_gets_the_notice(user_dir, workspace, tmp_path):
    assert untrusted_workspace_notice(workspace) == _NOTICE
    (user_dir / "settings.yaml").write_text(f"trusted_folders: [{workspace}]\n")
    assert untrusted_workspace_notice(workspace) is None
    plain = tmp_path / "plain"
    (plain / ".nooa").mkdir(parents=True)
    # nooa-acp never reads the workspace llm_config.yaml/secrets.yaml: no notice for them.
    (plain / ".nooa" / "llm_config.yaml").write_text("models: {}\n")
    assert untrusted_workspace_notice(plain) is None


async def test_skills_the_agent_creates_still_work(user_dir, workspace):
    agent = _agent(workspace)
    reply = await agent.skill_writing.create("fresh", "Fresh.", text="Be fresh.")
    assert reply.startswith("Created text skill")
    assert "local.fresh" in agent.skills.activated()
    assert "local.notes" not in agent.skills.loaded()


def test_project_config_skill_dirs_are_ignored_when_untrusted(user_dir, workspace, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (workspace / ".nooa" / "settings.yaml").write_text(
        f"coding:\n  additional_skills_dirs: [{outside}]\n"
    )
    assert outside.resolve() not in load_coding_skills_dirs(workspace)
    assert outside.resolve() in load_coding_skills_dirs(workspace, trusted=True)


def test_user_level_skill_dirs_load_in_an_untrusted_workspace(user_dir, workspace, tmp_path):
    mine = tmp_path / "my-skills"
    mine.mkdir()
    (user_dir / "settings.yaml").write_text(
        f"coding:\n  additional_skills_dirs: [{mine}, relative/dir]\n"
    )
    (workspace / "relative" / "dir").mkdir(parents=True)
    dirs = load_coding_skills_dirs(workspace)
    assert mine.resolve() in dirs
    assert (workspace / "relative" / "dir").resolve() not in dirs  # names a workspace folder


async def test_trust_command_trusts_the_folder_and_loads_its_skills(user_dir, workspace, tmp_path):
    agent = _agent(workspace)
    commands = CodingSlashCommandRegistry(agent)
    assert commands.get("trust") is not None
    assert commands.get("hello") is None
    result = await commands.invoke("trust", "")
    assert result.text == f"Trusted {workspace.resolve()}. Its skills are loaded now."
    assert is_trusted(workspace)
    assert (tmp_path / "evil-marker").exists()
    assert (tmp_path / "probe-marker").exists()
    assert {"local.evil", "local.notes"} <= set(agent.skills.activated())
    assert {"ext.probe", "cmd.hello"} <= set(agent.skills.discovered())
    assert "trust" not in agent.skills.status().lower()
    again = await commands.invoke("trust", "")
    assert again.text.startswith("This folder is already trusted")


def test_trusted_workspace_loads_skills_as_before(user_dir, workspace, tmp_path):
    (user_dir / "settings.yaml").write_text(f"trusted_folders: [{workspace}]\n")
    agent = _agent(workspace)
    assert (tmp_path / "evil-marker").exists()
    assert "local.evil" in agent.skills.activated()
    assert "cmd.hello" in agent.skills.discovered()
    assert CodingSlashCommandRegistry(agent).get("trust") is not None


def test_symlinked_skill_dirs_in_an_untrusted_workspace_do_not_load(user_dir, tmp_path):
    """Security review #2 (p1): links from the workspace to the rest of the clone."""
    repo = tmp_path / "repo"
    work = repo / "pkg"
    (work / ".claude" / "skills").mkdir(parents=True)
    elsewhere = repo / "elsewhere"
    (elsewhere / "agentsdir").mkdir(parents=True)
    (elsewhere / "viafile.py").write_text(
        f"from pathlib import Path\nPath({str(tmp_path / 'file-marker')!r}).write_text('x')\n"
        + _SKILL_CLASS.format(name="Viafile")
    )
    (elsewhere / "agentsdir" / "a.py").write_text(
        f"from pathlib import Path\nPath({str(tmp_path / 'root-marker')!r}).write_text('x')\n"
        + _SKILL_CLASS.format(name="A")
    )
    (work / ".claude" / "skills" / "linked.py").symlink_to(elsewhere / "viafile.py")
    (work / ".agents").mkdir()
    (work / ".agents" / "skills").symlink_to(elsewhere / "agentsdir")
    assert load_coding_skills_dirs(work) == []
    _agent(work)
    assert not (tmp_path / "file-marker").exists()
    assert not (tmp_path / "root-marker").exists()


def test_home_workspace_keeps_user_skills(user_dir, tmp_path):
    """Security review #6 (p6)."""
    home = tmp_path / "home"
    mine = home / ".claude" / "skills"
    mine.mkdir(parents=True)
    (mine / "mine.py").write_text(
        f"from pathlib import Path\nPath({str(tmp_path / 'mine-marker')!r}).write_text('x')\n"
        + _SKILL_CLASS.format(name="Mine")
    )
    try:
        agent = _agent(home)
        assert (tmp_path / "mine-marker").exists()
        assert "ext.mine" in agent.skills.discovered()
    finally:
        sys.modules.pop("mine", None)


async def test_trust_command_refuses_the_home_folder(user_dir, tmp_path):
    home = tmp_path / "home"
    (home / ".nooa" / "skills" / "x").mkdir(parents=True)
    (home / ".nooa" / "skills" / "x" / "pyproject.toml").write_text('[project]\nname = "x"\n')
    agent = _agent(home)
    assert HOME_REFUSAL in untrusted_workspace_notice(home)
    result = await CodingSlashCommandRegistry(agent).invoke("trust", "")
    assert result.text == HOME_REFUSAL
    assert not (user_dir / "settings.yaml").exists()
