# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A skill written with self.skill_writing is saved in .nooa/skills and active in a later session."""

import subprocess
import sys
import textwrap

import pytest

SOURCE = """\
\"\"\"Tiny statistics.\"\"\"
from nooa.skill import Skill


class Stats(Skill):
    \"\"\"Mean of numbers.\"\"\"

    def mean(self, values: list[float]) -> float:
        return sum(values) / len(values)
"""

LATER_SESSION = textwrap.dedent(
    """
    import asyncio, sys
    from pathlib import Path
    from nooa_atom.agent.agent import AtomAgent
    from nooa.unifiedllm import FakeLLMClient

    async def main():
        root = Path(sys.argv[1])
        agent = AtomAgent(
            llm=FakeLLMClient([]), cwd=root, libs_dir=root / ".nooa" / "skills", skills_dirs=[]
        )
        try:
            print("LIST", await agent.skill_writing.list())
            print("ACTIVE", sorted(n for n in agent.skills.activated() if n.startswith("local.")))
            for line in sys.argv[2:]:
                print("EVAL", line, "=", repr(eval(line, {"agent": agent})))
        finally:
            await agent.aclose()

    asyncio.run(main())
    """
)


def _later_session(root, *exprs: str) -> str:
    """Run a fresh agent on the same workspace in a new process; return its output."""
    done = subprocess.run(
        [sys.executable, "-c", LATER_SESSION, str(root), *exprs],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def _agent(root):
    from nooa_atom.agent.agent import AtomAgent

    from nooa.unifiedllm import FakeLLMClient

    return AtomAgent(
        llm=FakeLLMClient([]), cwd=root, libs_dir=root / ".nooa" / "skills", skills_dirs=[]
    )


async def test_a_code_skill_is_saved_in_nooa_skills_and_active_in_a_new_session(tmp_path):
    skills = tmp_path / ".nooa" / "skills"
    agent = _agent(tmp_path)
    try:
        await agent.skill_writing.create("stats", "Tiny statistics.")
        path = agent.skill_writing.path("stats", "__init__.py")
        assert path == str(skills / "stats" / "__init__.py")
        (skills / "stats" / "__init__.py").write_text(SOURCE)
        result = await agent.skill_writing.reload("stats")
        assert "error" not in result.lower(), result
        assert agent.stats.mean([2.0, 4.0]) == 3.0
    finally:
        await agent.aclose()
    assert (skills / "stats" / "pyproject.toml").is_file()
    assert not (tmp_path / ".nooa" / "libs").exists()

    out = _later_session(
        tmp_path, "agent.stats.mean([1.0, 2.0, 3.0])", "agent.skills.search('mean')"
    )
    assert "LIST ['stats']" in out
    assert "ACTIVE ['local.stats']" in out
    assert "agent.stats.mean([1.0, 2.0, 3.0]) = 2.0" in out
    assert "stats (code, active): Mean of numbers." in out


async def test_a_text_skill_is_a_skill_md_found_and_active_in_a_new_session(tmp_path):
    skills = tmp_path / ".nooa" / "skills"
    agent = _agent(tmp_path)
    try:
        result = await agent.skill_writing.create(
            "release_steps", "How to cut a release: tag, then publish.", text="1. Tag.\n2. Publish."
        )
        assert "SKILL.md" in result
        # Active in this session at once, with its instructions as its doc.
        assert "local.release_steps" in agent.skills.activated()
        assert "2. Publish." in type(agent.release_steps).__doc__
    finally:
        await agent.aclose()
    folder = skills / "release_steps"
    assert sorted(p.name for p in folder.iterdir()) == ["SKILL.md"]
    assert (folder / "SKILL.md").read_text() == (
        "---\n"
        'name: "release_steps"\n'
        'description: "How to cut a release: tag, then publish."\n'
        "---\n\n1. Tag.\n2. Publish.\n"
    )

    out = _later_session(
        tmp_path,
        "type(agent.release_steps).__doc__.splitlines()[-1]",
        "agent.skills.search('release')",
    )
    assert "LIST ['release_steps']" in out
    assert "ACTIVE ['local.release_steps']" in out
    assert "= '2. Publish.'" in out
    assert "release_steps (code, active): How to cut a release: tag, then publish." in out


async def test_a_skill_in_the_older_nooa_libs_still_loads(tmp_path):
    libs = tmp_path / ".nooa" / "libs"
    for name, doc in (("old_stats", "Old statistics."), ("both", "From libs.")):
        (libs / name).mkdir(parents=True)
        (libs / name / "pyproject.toml").write_text(f'[project]\nname = "{name}"\n')
        (libs / name / "__init__.py").write_text(
            SOURCE.replace("Mean of numbers.", doc).replace("class Stats", "class Lib")
        )
    # The same name in .nooa/skills wins.
    newer = tmp_path / ".nooa" / "skills" / "both"
    newer.mkdir(parents=True)
    (newer / "SKILL.md").write_text("---\nname: both\ndescription: From skills.\n---\nNewer.\n")

    out = _later_session(
        tmp_path, "agent.old_stats.mean([1.0, 3.0])", "type(agent.both).__doc__.splitlines()[0]"
    )
    assert "LIST ['both', 'old_stats']" in out
    assert "ACTIVE ['local.both', 'local.old_stats']" in out
    assert "agent.old_stats.mean([1.0, 3.0]) = 2.0" in out
    assert "= 'From skills.'" in out
    # New skills still go to .nooa/skills.
    agent = _agent(tmp_path)
    try:
        await agent.skill_writing.create("fresh", "Fresh one.", text="Hi.")
    finally:
        await agent.aclose()
    assert (tmp_path / ".nooa" / "skills" / "fresh" / "SKILL.md").is_file()
    assert not (libs / "fresh").exists()


async def test_an_older_skill_with_another_entry_point_does_not_replace_a_newer_one(tmp_path):
    old = tmp_path / ".nooa" / "libs" / "both"
    old.mkdir(parents=True)
    (old / "pyproject.toml").write_text(
        '[project]\nname = "both"\n\n[project.entry-points."nooa.skills"]\n"other.both" = "both"\n'
    )
    (old / "__init__.py").write_text(
        SOURCE.replace("Mean of numbers.", "From libs.").replace("class Stats", "class Lib")
    )
    newer = tmp_path / ".nooa" / "skills" / "both"
    newer.mkdir(parents=True)
    (newer / "SKILL.md").write_text("---\nname: both\ndescription: From skills.\n---\nNewer.\n")
    agent = _agent(tmp_path)
    try:
        assert type(agent.both).__doc__.splitlines()[0] == "From skills."
        assert "other.both" not in agent.skills.registry.loaded()
    finally:
        await agent.aclose()


async def test_a_text_skill_does_not_reuse_a_code_skill_directory(tmp_path):
    agent = _agent(tmp_path)
    try:
        await agent.skill_writing.create("stats", "Tiny statistics.")
        result = await agent.skill_writing.create("stats", "Steps.", text="1. Tag.")
        assert result.startswith("Not created")
    finally:
        await agent.aclose()
    assert not (tmp_path / ".nooa" / "skills" / "stats" / "SKILL.md").exists()


async def test_self_libs_is_gone(tmp_path):
    agent = _agent(tmp_path)
    try:
        with pytest.raises(AttributeError):
            agent.libs  # noqa: B018
    finally:
        await agent.aclose()


async def test_a_text_skill_named_like_an_agent_attribute_is_refused(tmp_path):
    agent = _agent(tmp_path)
    try:
        shell = agent.shell
        result = await agent.skill_writing.create("shell", "Shell tips.", text="Use rg.")
        assert result.startswith("Not created")
        assert "self.shell is already an attribute" in result
        assert "choose another name" in result
        # reload() does not import the directory and replace self.shell either.
        skill_dir = tmp_path / ".nooa" / "skills" / "shell"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("---\nname: shell\ndescription: x\n---\nHi.\n")
        reloaded = await agent.skill_writing.reload("shell")
        assert "OK — written and loaded" not in reloaded
        assert "choose another name" in reloaded
        assert agent.shell is shell
        assert "local.shell" not in agent.skills.registry.loaded()
        # Control: the same SKILL.md under a free name does load, so the refusal
        # above is the collision, not the file.
        free_dir = tmp_path / ".nooa" / "skills" / "shell_tips"
        free_dir.mkdir(parents=True)
        (free_dir / "SKILL.md").write_text("---\nname: shell_tips\ndescription: x\n---\nHi.\n")
        assert "OK — written and loaded" in await agent.skill_writing.reload("shell_tips")
        assert "local.shell_tips" in agent.skills.registry.loaded()
    finally:
        await agent.aclose()


async def test_a_text_skill_with_a_free_name_is_created_and_loaded(tmp_path):
    agent = _agent(tmp_path)
    try:
        result = await agent.skill_writing.create("release_notes", "Notes.", text="1. Tag.")
        assert result.startswith("Created text skill 'release_notes'")
        assert "local.release_notes" in agent.skills.registry.loaded()
    finally:
        await agent.aclose()


async def test_a_skill_name_with_a_path_is_refused(tmp_path):
    agent = _agent(tmp_path)
    try:
        with pytest.raises(ValueError, match="choose another name"):
            await agent.skill_writing.create("../../escaped", "x", text="Hi.")
    finally:
        await agent.aclose()
    assert not (tmp_path / "escaped").exists()
    assert not (tmp_path.parent / "escaped").exists()
