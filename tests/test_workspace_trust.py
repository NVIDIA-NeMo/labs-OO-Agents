# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workspace trust (nooa.workspace_trust) and the code gate in SkillRegistry/SkillWriting."""

import sys
from pathlib import Path

import pytest
import yaml

from nooa import workspace_trust
from nooa.layered_config import layered_paths
from nooa.llm_config import llm_config_chain
from nooa.skill_registry import SkillRegistry
from nooa.tools import SkillWriting
from nooa.workspace_trust import (
    HOME_REFUSAL,
    WorkspaceTrustError,
    code_allowed_outside,
    is_inside,
    is_trusted,
    trust,
    user_skill_roots,
)


@pytest.fixture
def user_dir(tmp_path, monkeypatch, untrusted_folders):
    path = tmp_path / "user"
    path.mkdir()
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(path))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    monkeypatch.delenv("NEMO_OO_PROJECT_DIR", raising=False)
    return path


def _code_skill(root: Path, name: str, marker: Path) -> Path:
    """A code skill whose import writes ``marker``."""
    lib = root / name
    lib.mkdir(parents=True)
    (lib / "pyproject.toml").write_text(f'[project]\nname = "{name}"\nversion = "0.1.0"\n')
    (lib / "__init__.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('imported')\n"
        "from nooa.skill import Skill\n\n\n"
        f"class {name.capitalize()}(Skill):\n"
        '    """Probe."""\n'
    )
    return lib


def _text_skill(root: Path, name: str) -> None:
    (root / name).mkdir(parents=True)
    (root / name / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: A text skill.\n---\n\nSay hello.\n"
    )


class _Agent:
    def __init__(self):
        self.skills = SkillRegistry(self)


def test_untrusted_by_default_and_trust_adds_the_folder(user_dir, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (user_dir / "settings.yaml").write_text("atom:\n  default_model: m\n")
    assert not is_trusted(work)
    trust(work)
    assert is_trusted(work)
    data = yaml.safe_load((user_dir / "settings.yaml").read_text())
    assert data["atom"] == {"default_model": "m"}  # other keys kept
    assert data["trusted_folders"] == [str(work.resolve())]
    trust(work)  # idempotent
    saved = yaml.safe_load((user_dir / "settings.yaml").read_text())
    assert saved["trusted_folders"] == [str(work.resolve())]


def test_a_trusted_parent_trusts_its_subfolders(user_dir, tmp_path):
    trust(tmp_path / "parent")
    assert is_trusted(tmp_path / "parent" / "child" / "deeper")
    assert not is_trusted(tmp_path / "parentx")
    assert not is_trusted(tmp_path)


def test_the_project_layer_cannot_trust_its_own_folder(user_dir, tmp_path, monkeypatch):
    work = tmp_path / "work"
    (work / ".nooa").mkdir(parents=True)
    (work / ".nooa" / "settings.yaml").write_text(f"trusted_folders: [{work}]\n")
    monkeypatch.chdir(work)
    assert not is_trusted(work)


def test_nemo_oo_settings_trusts_a_folder(user_dir, tmp_path, monkeypatch):
    work = tmp_path / "ci-checkout"
    work.mkdir()
    settings = tmp_path / "ci.yaml"
    settings.write_text(f"trusted_folders:\n  - {work}\n")
    monkeypatch.setenv("NEMO_OO_SETTINGS", str(settings))
    assert is_trusted(work)


def test_untrusted_skill_writing_loads_text_skills_but_imports_no_code(user_dir, tmp_path):
    skills = tmp_path / "work" / ".nooa" / "skills"
    marker = tmp_path / "marker"
    _code_skill(skills, "evil", marker)
    _text_skill(skills, "notes")
    agent = _Agent()
    agent.skills.code_allowed = code_allowed_outside(tmp_path / "work")
    writing = SkillWriting(agent, path=skills)

    assert not marker.exists()
    assert str(skills) not in sys.path
    assert "local.notes" in agent.skills.loaded()
    assert "local.evil" not in agent.skills.loaded()
    assert agent.skills.skipped_code() == [skills / "evil"]

    agent.skills.code_allowed = None
    assert agent.skills.skipped_code() == []  # a new policy clears the skip list
    writing.enable_code()
    try:
        assert marker.read_text() == "imported"
        assert "local.evil" in agent.skills.activated()
        assert str(skills) in sys.path
    finally:
        sys.path.remove(str(skills))
        sys.modules.pop("evil", None)


async def test_untrusted_skill_writing_refuses_to_reload_a_code_skill(user_dir, tmp_path):
    skills = tmp_path / "skills"
    marker = tmp_path / "marker"
    _code_skill(skills, "evil", marker)
    agent = _Agent()
    agent.skills.code_allowed = code_allowed_outside(tmp_path)
    writing = SkillWriting(agent, path=skills)
    report = await writing.reload("evil")
    assert "not trusted" in report
    assert not marker.exists()


def test_registry_refuses_loose_python_and_packages_under_an_untrusted_root(user_dir, tmp_path):
    root = tmp_path / "work"
    skills = root / ".claude" / "skills"
    skills.mkdir(parents=True)
    marker = tmp_path / "py-marker"
    (skills / "probe.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('x')\n"
    )
    _code_skill(skills, "pkg", tmp_path / "pkg-marker")
    _text_skill(skills, "hello")
    agent = _Agent()
    agent.skills.code_allowed = code_allowed_outside(root)
    agent.skills.discover_skills_dirs([skills])
    assert not marker.exists()
    assert not (tmp_path / "pkg-marker").exists()
    assert "cmd.hello" in agent.skills.discovered()
    assert len(agent.skills.skipped_code()) == 2


def test_project_layer_can_be_left_out_of_layered_config(user_dir, tmp_path, monkeypatch):
    project = tmp_path / "work" / ".nooa"
    project.mkdir(parents=True)
    (project / "llm_config.yaml").write_text("models: {}\n")
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project))
    assert layered_paths("llm_config.yaml") == [(project / "llm_config.yaml").resolve()]
    assert llm_config_chain(include_project=False) == []


def test_launch_project_trust_follows_the_cwd(user_dir, tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    (work / "pyproject.toml").write_text("")
    monkeypatch.chdir(work)
    assert not workspace_trust.launch_project_trusted()
    trust(work)
    assert workspace_trust.launch_project_trusted()


# ---- review fixes (#231 fix round) ------------------------------------------


def _loose_skill(path: Path, marker: Path) -> None:
    path.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('x')\n"
        'from nooa.skill import Skill\n\n\nclass Probe(Skill):\n    """Probe."""\n'
    )


def test_symlinks_in_the_workspace_that_point_outside_are_still_refused(user_dir, tmp_path):
    """Security review #2 (p1): a link found in the workspace is workspace code."""
    repo = tmp_path / "repo"
    work = repo / "pkg"
    skills = work / ".claude" / "skills"
    skills.mkdir(parents=True)
    elsewhere = repo / "elsewhere"
    elsewhere.mkdir()
    _loose_skill(elsewhere / "viafile.py", tmp_path / "file-marker")
    _code_skill(elsewhere, "viapkg", tmp_path / "pkg-marker")
    (skills / "linked.py").symlink_to(elsewhere / "viafile.py")
    (skills / "linkedpkg").symlink_to(elsewhere / "viapkg")
    agentsdir = elsewhere / "agentsdir"
    agentsdir.mkdir()
    _loose_skill(agentsdir / "a.py", tmp_path / "root-marker")
    (work / ".agents").mkdir()
    (work / ".agents" / "skills").symlink_to(agentsdir)

    agent = _Agent()
    agent.skills.code_allowed = code_allowed_outside(work)
    agent.skills.discover_skills_dirs([work / ".agents" / "skills", skills])
    assert not (tmp_path / "file-marker").exists()
    assert not (tmp_path / "pkg-marker").exists()
    assert not (tmp_path / "root-marker").exists()
    assert len(agent.skills.skipped_code()) == 3


def test_a_path_outside_that_points_into_the_workspace_is_refused(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "x.py").write_text("")
    (tmp_path / "link.py").symlink_to(work / "x.py")
    assert is_inside(tmp_path / "link.py", work)
    assert not is_inside(tmp_path / "other.py", work)
    assert not code_allowed_outside(work)(tmp_path / "link.py")


def test_user_skill_roots_load_code_even_inside_the_workspace(user_dir, tmp_path, monkeypatch):
    """Security review #6 / quality M2: workspace == $HOME keeps the user's own skills."""
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    mine = home / ".claude" / "skills"
    mine.mkdir(parents=True)
    _loose_skill(mine / "mine.py", tmp_path / "mine-marker")
    extra = tmp_path / "home" / "my-skills"
    extra.mkdir()
    roots = user_skill_roots([extra, "relative/dir"])
    assert mine in roots and extra in roots
    assert all(root.is_absolute() for root in roots) and len(roots) == 4

    agent = _Agent()
    agent.skills.code_allowed = code_allowed_outside(home, user_roots=roots)
    agent.skills.discover_skills_dirs([mine])
    assert (tmp_path / "mine-marker").exists()
    assert agent.skills.skipped_code() == []
    assert not code_allowed_outside(home, user_roots=roots)(home / ".nooa" / "skills" / "x")


def test_trust_refuses_the_home_folder_and_filesystem_roots(user_dir, tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    for folder in (home, Path("/")):
        with pytest.raises(WorkspaceTrustError, match="home folder"):
            trust(folder)
    assert str(HOME_REFUSAL).startswith("Open a project folder instead")
    assert not (user_dir / "settings.yaml").exists()
    trust(home / "project")  # a folder below home is fine
    assert is_trusted(home / "project")


def test_trust_refuses_a_malformed_settings_file_and_leaves_it(user_dir, tmp_path):
    """Security review #4 (p5)."""
    settings = user_dir / "settings.yaml"
    broken = "# my settings\natom:\n  model: big\nbroken: [unclosed\n"
    settings.write_text(broken)
    with pytest.raises(WorkspaceTrustError, match="not valid YAML"):
        trust(tmp_path / "work")
    assert settings.read_text() == broken
    settings.write_text("- a list\n")
    with pytest.raises(WorkspaceTrustError, match="not a YAML mapping"):
        trust(tmp_path / "work")


def test_trust_keeps_comments_when_it_appends(user_dir, tmp_path):
    settings = user_dir / "settings.yaml"
    settings.write_text("# keep me\natom:\n  model: big  # inline\n")
    trust(tmp_path / "work")
    text = settings.read_text()
    assert text.startswith("# keep me\natom:\n  model: big  # inline\n")
    assert yaml.safe_load(text)["trusted_folders"] == [str((tmp_path / "work").resolve())]


def test_relative_trusted_folders_entries_are_ignored(user_dir, tmp_path, monkeypatch, caplog):
    """Security review #5 / quality m11 (p5)."""
    (tmp_path / "proj").mkdir()
    monkeypatch.chdir(tmp_path)
    (user_dir / "settings.yaml").write_text("trusted_folders: [proj, ~/abs]\n")
    with caplog.at_level("WARNING"):
        folders = workspace_trust.trusted_folders()
    assert folders == [(Path.home() / "abs").resolve()]
    assert not is_trusted(tmp_path / "proj")
    assert "relative trusted_folders entry 'proj'" in caplog.text


def test_remove_notice():
    agent = _Agent()
    agent.skills.add_notice("a")
    agent.skills.remove_notice("a")
    agent.skills.remove_notice("missing")
    assert agent.skills.notices() == []
