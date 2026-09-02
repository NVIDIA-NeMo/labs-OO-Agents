# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for nooa.Skill — path-based loading."""

import math
from pathlib import Path

import pytest

from nooa import Skill, SkillFile, TextSkill
from nooa.agentdoc import doc
from nooa.tools import ShellTools


@pytest.fixture
def skill_dir(tmp_path):
    d = tmp_path / "git-workflow"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: git-workflow\ndescription: Best practices for Git\n---\n"
        "# Git Workflow Guide\n\n1. Always create feature branches\n"
    )
    return d


# ── constructor paths ──────────────────────────────────────────────────────────


def test_skill_path_loads_id(skill_dir):
    assert TextSkill(path=skill_dir).id == "git-workflow"


def test_skill_path_accepts_explicit_id(skill_dir):
    assert TextSkill(path=skill_dir, id="custom-id").id == "custom-id"


def test_skill_path_creates_dynamic_subclass(skill_dir):
    skill = TextSkill(path=skill_dir)
    assert type(skill).__name__ != "Skill"
    assert isinstance(skill, Skill)
    assert not isinstance(skill, TextSkill)
    assert "Best practices for Git" in (type(skill).__doc__ or "")


def test_skill_content_constructor():
    skill = Skill(content="A helpful skill.")
    assert "A helpful skill." in (type(skill).__doc__ or "")


def test_skill_obj_constructor():
    skill = Skill(math)
    assert isinstance(skill, Skill)


def test_skill_no_args_raises():
    with pytest.raises(ValueError, match="requires one of"):
        Skill()


def test_skill_multiple_args_raises():
    with pytest.raises(ValueError, match="exactly one of"):
        Skill(math, content="extra")


def test_skill_path_raises_for_missing_skill_md(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(ValueError, match=r"SKILL\.md not found"):
        TextSkill(path=d)


# ── properties ────────────────────────────────────────────────────────────────


def test_skill_description_property(skill_dir):
    assert TextSkill(path=skill_dir).description == "Best practices for Git"


def test_skill_dir_forwards_to_wrapped_obj():
    skill = Skill(math)
    assert "sqrt" in dir(skill)


def test_skill_dir_on_path_skill(skill_dir):
    # Non-obj skill: __dir__ falls back to base (no _skill_obj)
    skill = TextSkill(path=skill_dir)
    assert "id" in dir(skill)


# ── ShellTools and packaged files ─────────────────────────────────────────────


@pytest.fixture
def skill_with_scripts(skill_dir):
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "greet.py").write_text("print('hello from script')")
    assets_dir = skill_dir / "assets"
    assets_dir.mkdir()
    (assets_dir / "prompt.txt").write_text("prompt")
    return skill_dir


def test_text_skill_injects_skill_root_shell(skill_dir):
    skill = TextSkill(path=skill_dir)
    assert isinstance(skill.shell, ShellTools)
    assert skill.shell.cwd == skill_dir.resolve()


@pytest.mark.asyncio
async def test_text_skill_reads_files_through_shell(skill_dir):
    skill = TextSkill(path=skill_dir)
    result = await skill.shell.read("SKILL.md")
    assert "Best practices for Git" in result.text


@pytest.mark.asyncio
async def test_text_skill_runs_scripts_through_shell(skill_with_scripts):
    skill = TextSkill(path=skill_with_scripts)
    try:
        output = await skill.shell.run("python3 scripts/greet.py")
        assert output.success
        assert "hello from script" in output.stdout
    finally:
        await skill.detach()


def test_text_skill_files_are_relative_and_stably_sorted(skill_with_scripts):
    skill = TextSkill(path=skill_with_scripts)
    assert skill.files == [
        SkillFile(path="SKILL.md"),
        SkillFile(path="assets/prompt.txt"),
        SkillFile(path="scripts/greet.py"),
    ]


def test_text_skill_manifest_excludes_external_symlinks(skill_dir, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    (skill_dir / "outside-link.txt").symlink_to(outside)

    skill = TextSkill(path=skill_dir)

    assert SkillFile(path="outside-link.txt") not in skill.files


def test_text_skill_removes_legacy_helper_surface(skill_dir):
    skill = TextSkill(path=skill_dir)
    assert not hasattr(skill, "read_file")
    assert not hasattr(skill, "run_script")


def test_text_skill_documentation_exposes_shell_and_files(skill_dir):
    rendered = doc(TextSkill(path=skill_dir))
    assert "Best practices for Git" in rendered
    assert "shell: ShellTools" in rendered
    assert "files: list[SkillFile]" in rendered
    assert "read_file" not in rendered
    assert "run_script" not in rendered


# ── source_dir ────────────────────────────────────────────────────────────────


def test_text_skill_source_dir(skill_dir):
    skill = TextSkill(path=skill_dir)
    assert skill.source_dir == skill_dir


def test_skill_subclass_source_dir():
    class MySkill(Skill):
        """A custom skill."""

        pass

    skill = MySkill()
    # source_dir should point to this test file's directory
    assert skill.source_dir == Path(__file__).resolve().parent


def test_skill_source_dir_explicit_override(skill_dir):
    class MySkill(Skill):
        """A custom skill."""

        pass

    skill = MySkill()
    skill._source_dir = skill_dir
    assert skill.source_dir == skill_dir


def test_skill_obj_wrapper_source_dir():
    skill = Skill(math)
    # Skill(obj) creates a dynamic class — inspect.getfile may fail on it
    # source_dir should gracefully return None or a valid path
    result = skill.source_dir
    assert result is None or result.is_dir()
