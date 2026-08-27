# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SlimTextSkillTranslator."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from nooa import Agent
from nooa.agentdoc import doc
from nooa.context_blocks import DynamicContext
from nooa.skill_registry import SkillRegistry
from nooa.tools.slim_skill_translator import SlimTextSkillTranslator


class _Agent:
    pass


def _write_skill_md(
    skill_dir: Path,
    *,
    name: str = "slim-skill",
    body: str = "Use this skill to test slim translation.\n",
) -> None:
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: Slim translator test\n"
        "---\n"
        f"{body}",
        encoding="utf-8",
    )


def test_slim_translator_omits_argparse_scripts(tmp_path):
    skill_dir = tmp_path / "search-skill"
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    _write_skill_md(skill_dir, name="search-skill")
    (scripts_dir / "search.py").write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--query', required=True)\n"
        "args = parser.parse_args()\n"
        "print(args.query)\n",
        encoding="utf-8",
    )
    translator = SlimTextSkillTranslator()

    result = translator.translate(skill_dir, tmp_path / "libs")

    assert [(script.script_path, script.reason) for script in result.omitted_scripts] == [
        ("scripts/search.py", "No import-safe public Python functions could be inferred.")
    ]
    generated_text = (result.package_dir / "src" / "search_skill" / "__init__.py").read_text()
    assert "def search(" not in generated_text
    assert "run_resource_script" not in generated_text


def test_slim_translator_does_not_advertise_omitted_scripts(tmp_path):
    skill_dir = tmp_path / "search-skill"
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    _write_skill_md(
        skill_dir,
        name="search-skill",
        body=(
            "Use this skill to answer search questions.\n"
            "Run `scripts/search.py` with the requested query.\n"
        ),
    )
    (scripts_dir / "search.py").write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--query', required=True)\n"
        "args = parser.parse_args()\n"
        "print(args.query)\n",
        encoding="utf-8",
    )

    result = SlimTextSkillTranslator().translate(skill_dir, tmp_path / "libs")

    generated_text = (result.package_dir / "src" / "search_skill" / "__init__.py").read_text()
    assert "scripts/search.py" not in generated_text
    assert "search()" not in generated_text
    assert "corresponding LibrarySkill API" not in generated_text
    assert "the relevant LibrarySkill guidance" in generated_text


def test_slim_translator_adapts_resource_script_path_guidance(tmp_path):
    skill_dir = tmp_path / "browser-skill"
    _write_skill_md(
        skill_dir,
        name="browser-skill",
        body=(
            "Run `npx ts-node <path-to-this-skill>/measure-cls.ts http://localhost:3000` "
            "before and after the fix.\n"
        ),
    )
    (skill_dir / "measure-cls.ts").write_text("console.log('cls')\n", encoding="utf-8")

    result = SlimTextSkillTranslator().translate(skill_dir, tmp_path / "libs")

    generated_text = (result.package_dir / "src" / "browser_skill" / "__init__.py").read_text()
    assert "<path-to-this-skill>" not in generated_text
    assert "write script text to a workspace file before running it" in generated_text
    assert "a file containing the text returned by `measure_cls()`" in generated_text


def test_slim_translator_does_not_depend_on_legacy_translator():
    source = Path(inspect.getsourcefile(SlimTextSkillTranslator) or "")
    text = source.read_text(encoding="utf-8")

    assert "nooa.tools.skill_translator" not in text
    assert "class SlimTextSkillTranslator(TextSkillTranslator)" not in text


def test_slim_translator_ignores_argparse_for_function_scripts(tmp_path):
    skill_dir = tmp_path / "calc-skill"
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    _write_skill_md(skill_dir, name="calc-skill")
    (scripts_dir / "math_tools.py").write_text(
        "def scale(value: int) -> int:\n"
        "    return value * 2\n",
        encoding="utf-8",
    )

    result = SlimTextSkillTranslator().translate(skill_dir, tmp_path / "libs")

    assert "src/calc_skill/_impl/_scripts_math_tools.py" in result.files_written


def test_slim_translator_normalizes_default_registry_name(tmp_path):
    skill_dir = tmp_path / "odd-skill"
    _write_skill_md(skill_dir, name="Odd Skill!!")

    result = SlimTextSkillTranslator().translate(skill_dir, tmp_path / "libs")

    assert result.registry_name == "local.odd-skill"


@pytest.mark.asyncio
async def test_slim_translator_preserves_private_sibling_helpers(tmp_path):
    skill_dir = tmp_path / "calc-skill"
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    _write_skill_md(skill_dir, name="calc-skill")
    (scripts_dir / "helpers.py").write_text(
        "FACTOR = 3\n"
        "\n"
        "def _offset() -> int:\n"
        "    return 1\n",
        encoding="utf-8",
    )
    (scripts_dir / "math_tools.py").write_text(
        "from helpers import FACTOR, _offset\n"
        "\n"
        "def scale(value: int) -> int:\n"
        "    return value * FACTOR + _offset()\n",
        encoding="utf-8",
    )
    translator = SlimTextSkillTranslator()

    result = translator.translate(skill_dir, tmp_path / "libs")

    assert translator.validate_package(result.package_dir).ok
    assert "src/calc_skill/_impl/_scripts_helpers.py" in result.files_written
    assert [script.script_path for script in result.omitted_scripts] == []

    registry = SkillRegistry(_Agent())
    registry.discover_libs(result.package_dir.parent)
    try:
        skill = registry[result.registry_name]
        visible_doc = doc(skill)
        assert skill.scale(4) == 13
        assert "helpers" not in visible_doc
        assert "impl_scripts_helpers" not in visible_doc
    finally:
        await registry.aclose()


@pytest.mark.asyncio
async def test_slim_translator_keeps_functions_resources_and_context(tmp_path):
    skill_dir = tmp_path / "calc-skill"
    scripts_dir = skill_dir / "scripts"
    refs_dir = skill_dir / "references"
    scripts_dir.mkdir(parents=True)
    refs_dir.mkdir()
    _write_skill_md(skill_dir, name="calc-skill")
    (scripts_dir / "math_tools.py").write_text(
        "SCALE = 2\n"
        "\n"
        "def scale(value: int) -> int:\n"
        "    \"\"\"Scale a value.\"\"\"\n"
        "    return value * SCALE\n",
        encoding="utf-8",
    )
    (refs_dir / "notes.txt").write_text("reference notes\n", encoding="utf-8")
    translator = SlimTextSkillTranslator()

    result = translator.translate(skill_dir, tmp_path / "libs")
    report = translator.validate_package(result.package_dir)

    assert report.ok
    assert "src/calc_skill/_impl/_scripts_math_tools.py" in result.files_written
    assert "src/calc_skill/resources/scripts/math_tools.py" not in result.files_written

    registry = SkillRegistry(_Agent())
    registry.discover_libs(result.package_dir.parent)
    try:
        skill = registry[result.registry_name]
        visible_doc = doc(skill)
        assert "def scale(" in visible_doc
        assert "references_notes" in visible_doc
        assert skill.scale(4) == 8
        assert skill.references_notes() == "reference notes\n"
    finally:
        await registry.aclose()

    agent = Agent(llm=object())
    registry = SkillRegistry(agent)
    registry.discover_libs(result.package_dir.parent)
    registry.activate([result.registry_name])
    try:
        context_key = f"skill:{result.registry_name}"
        raw_block = dict(agent.context_manager._raw_items())[context_key]
        assert isinstance(raw_block, DynamicContext)
        assert raw_block.expr == "self.calc_skill.format_guidance()"
    finally:
        await registry.aclose()
