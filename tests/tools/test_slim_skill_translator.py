# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SlimTextSkillTranslator."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from nooa import Agent
from nooa.agentdoc import doc
from nooa.context_blocks import DynamicContext
from nooa.skill_registry import SkillRegistry
from nooa.tools.slim_skill_translator import SlimTextSkillTranslator


class _Agent:
    pass


def _write_skill_md(skill_dir: Path, *, name: str = "slim-skill") -> None:
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "description: Slim translator test\n"
        "---\n"
        "Use this skill to test slim translation.\n",
        encoding="utf-8",
    )


def _run_generated_tests(package_dir: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{package_dir / 'src'}:{env.get('PYTHONPATH', '')}"
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(package_dir / "tests"), "-q"],
        text=True,
        capture_output=True,
        timeout=60,
        env=env,
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
    generated_tests = _run_generated_tests(result.package_dir)

    assert report.ok
    assert generated_tests.returncode == 0, generated_tests.stdout + generated_tests.stderr
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
