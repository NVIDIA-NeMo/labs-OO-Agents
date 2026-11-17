# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The coding agent in an untrusted workspace (nooa.workspace_trust)."""

import sys
from pathlib import Path

import pytest
from nooa_cli.coding import CodingAgent
from nooa_cli.coding.settings import load_coding_skills_dirs
from nooa_cli.coding.slash_commands import CodingSlashCommandRegistry

from nooa.unifiedllm import FakeLLMClient
from nooa.workspace_trust import HOME_REFUSAL, is_trusted, untrusted_notice

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
    """A workspace with code skills in .nooa/skills and .claude/skills, and a text skill."""
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


def test_untrusted_workspace_loads_text_skills_and_no_code(user_dir, workspace, tmp_path):
    agent = _agent(workspace)
    assert not (tmp_path / "evil-marker").exists()
    assert not (tmp_path / "probe-marker").exists()
    assert str(workspace / ".nooa" / "skills") not in sys.path
    assert "local.notes" in agent.skills.activated()
    assert "local.evil" not in agent.skills.loaded()
    assert "ext.probe" not in agent.skills.discovered()
    notice = untrusted_notice(workspace)
    assert notice == (
        "Code skills in this folder are off because it is not trusted. Type /trust to allow."
    )
    assert agent.workspace_untrusted_notice() == notice
    assert notice in agent.skills.status()


def test_project_config_skill_dirs_are_ignored_when_untrusted(user_dir, workspace, tmp_path):
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (workspace / ".nooa" / "settings.yaml").write_text(
        f"coding:\n  additional_skills_dirs: [{outside}]\n"
    )
    assert outside.resolve() not in load_coding_skills_dirs(workspace)
    assert outside.resolve() in load_coding_skills_dirs(workspace, trusted=True)


async def test_trust_command_trusts_the_folder_and_loads_its_code(user_dir, workspace, tmp_path):
    agent = _agent(workspace)
    commands = CodingSlashCommandRegistry(agent)
    assert commands.get("trust") is not None
    result = await commands.invoke("trust", "")
    assert "Trusted" in result.text
    assert is_trusted(workspace)
    assert (tmp_path / "evil-marker").exists()
    assert (tmp_path / "probe-marker").exists()
    assert "local.evil" in agent.skills.activated()
    assert "ext.probe" in agent.skills.discovered()
    assert agent.workspace_untrusted_notice() is None
    # Quality M1: the notice and the skip list are gone after /trust.
    assert untrusted_notice(workspace) not in agent.skills.status()
    assert agent.skills.skipped_code() == []
    assert result.text == f"Trusted {workspace.resolve()}. Its code skills are loaded now."
    again = await commands.invoke("trust", "")
    assert again.text.startswith("This folder is already trusted")


def test_trusted_workspace_loads_code_as_before(user_dir, workspace, tmp_path):
    (user_dir / "settings.yaml").write_text(f"trusted_folders: [{workspace}]\n")
    agent = _agent(workspace)
    assert (tmp_path / "evil-marker").exists()
    assert "local.evil" in agent.skills.activated()
    assert agent.workspace_untrusted_notice() is None


def test_trust_is_a_slash_command_not_a_model_api(user_dir, workspace):
    """Security review #7 / quality m6, m7: no public trust method or skill; /trust always exists."""
    agent = _agent(workspace)
    assert not hasattr(agent, "trust_workspace")
    assert "workspace_trust" not in agent.skills.status()
    assert not any("trust" in name for name in agent.skills.discovered())
    (user_dir / "settings.yaml").write_text(f"trusted_folders: [{workspace}]\n")
    trusted = CodingSlashCommandRegistry(_agent(workspace))
    assert trusted.get("trust") is not None


def test_symlinked_skills_in_an_untrusted_workspace_do_not_run(user_dir, tmp_path):
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
    dirs = load_coding_skills_dirs(work)
    assert work / ".agents" / "skills" in dirs  # where it was found, not where it points
    agent = _agent(work)
    assert not (tmp_path / "file-marker").exists()
    assert not (tmp_path / "root-marker").exists()
    assert len(agent.skills.skipped_code()) == 2


def test_home_workspace_keeps_user_skills_and_refuses_trust(user_dir, tmp_path):
    """Security review #6 / quality M2 (p6)."""
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
    assert HOME_REFUSAL in agent.workspace_untrusted_notice()
    result = await CodingSlashCommandRegistry(agent).invoke("trust", "")
    assert result.text == HOME_REFUSAL
    assert not (user_dir / "settings.yaml").exists()


def test_model_settings_alone_give_no_notice(user_dir, tmp_path):
    """Quality M3: nooa-acp never reads the workspace llm_config.yaml/secrets.yaml."""
    work = tmp_path / "plain"
    (work / ".nooa").mkdir(parents=True)
    (work / ".nooa" / "llm_config.yaml").write_text("models: {}\n")
    (work / ".nooa" / "secrets.yaml").write_text("env: {}\n")
    assert _agent(work).workspace_untrusted_notice() is None
