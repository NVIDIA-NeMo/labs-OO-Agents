# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``Config.load`` skills_dirs ordering.

The scan order matters: ``SkillManager.install()`` attaches a skill the
first time it sees a given attribute name, then skips subsequent matches.
So whichever directory appears first in ``cfg.tui.skills_dirs`` wins.

Precedence:

    1. user-explicit ``--skills-dir`` values
    2. paths persisted by ``/skills add``
    3. default locations (``~/.claude/commands``, ``.claude/skills``, …)

Installed packages contribute skills through ``nooa.skills`` rather than a
TUI-specific directory entry-point group.
"""

from pathlib import Path

import pytest
from nooa_cli.tui.config import Config


@pytest.fixture(autouse=True)
def _isolate_settings(tmp_path_factory, monkeypatch):
    """Pin the layered settings.yaml dirs so Config.load() can't read a real
    user/project settings.yaml (e.g. one written by `/config set` at the
    repo root) and perturb skills_dirs assertions."""
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path_factory.mktemp("settings-user")))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path_factory.mktemp("settings-proj")))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)


def _skill_dir(base: Path, name: str) -> Path:
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "dummy-skill").mkdir(exist_ok=True)
    (d / "dummy-skill" / "SKILL.md").write_text("---\nname: dummy\ndescription: x\n---\n\nbody")
    return d


def test_explicit_skills_dir_precedes_defaults(tmp_path, monkeypatch):
    """--skills-dir comes before the default ~/.claude/commands etc."""
    explicit = _skill_dir(tmp_path, "user-explicit")
    default = _skill_dir(tmp_path, "fake-home/.claude/commands")

    # Force cwd and $HOME defaults to point at existing dirs for the test
    monkeypatch.chdir(tmp_path / "fake-home")
    monkeypatch.setenv("HOME", str(tmp_path / "fake-home"))

    cfg = Config.load(skills_dir=[explicit])
    dirs = cfg.tui.skills_dirs
    assert explicit in dirs
    assert default in dirs
    assert dirs.index(explicit) < dirs.index(default), (
        "user --skills-dir must appear before default dirs"
    )


def test_string_skills_dir_not_iterated_as_characters(tmp_path, monkeypatch):
    """A single string passed as ``skills_dir=`` is treated as one path,
    not iterated character-by-character (``list("/path")`` → ['/', 'p',
    'a', …] bug from the pre-fix code)."""
    explicit = _skill_dir(tmp_path, "single-string")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))

    # Pass a bare string (not a list) as skills_dir.
    cfg = Config.load(skills_dir=str(explicit))

    assert explicit in cfg.tui.skills_dirs
    # No character-fragments should show up
    assert not any(len(str(d)) == 1 for d in cfg.tui.skills_dirs)


def test_same_explicit_dir_passed_twice_is_deduped(tmp_path, monkeypatch):
    shared = _skill_dir(tmp_path, "shared")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))

    cfg = Config.load(skills_dir=[shared, shared])

    assert cfg.tui.skills_dirs.count(shared) == 1


def test_persisted_skills_dir_precedes_defaults(tmp_path, monkeypatch):
    persisted = _skill_dir(tmp_path, "persisted")
    default = _skill_dir(tmp_path, "fake-home/.claude/commands")
    project_dir = tmp_path / "project-settings"
    project_dir.mkdir()
    (project_dir / "settings.yaml").write_text(
        f"tui:\n  additional_skills_dirs:\n    - {persisted}\n"
    )
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project_dir))
    monkeypatch.chdir(tmp_path / "fake-home")
    monkeypatch.setenv("HOME", str(tmp_path / "fake-home"))

    cfg = Config.load()

    assert cfg.tui.additional_skills_dirs == [persisted]
    assert cfg.tui.skills_dirs.index(persisted) < cfg.tui.skills_dirs.index(default)
