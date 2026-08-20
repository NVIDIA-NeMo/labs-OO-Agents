# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TextSkillTranslator."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from nooa import Agent
from nooa.skill import TextSkill
from nooa.skill_registry import SkillRegistry
from nooa.tools.skill_translator import TextSkillTranslator
from nooa.unifiedllm import LLMResponse, ToolCall


class _Agent:
    pass


class _ScriptedCodeActLLM:
    def __init__(self, code: str):
        self.code = code
        self.call_count = 0

    async def acall(self, messages, tools=None, **kwargs):
        self.call_count += 1
        return LLMResponse(
            raw_response=None,
            content="",
            tool_calls=[
                ToolCall(
                    id=f"call_{self.call_count}",
                    name="execute_python",
                    arguments=json.dumps({"code": self.code}),
                )
            ],
            finish_reason="tool_calls",
            assistant_message={},
        )


def _make_text_skill(tmp_path: Path) -> Path:
    skill_dir = tmp_path / "hello-skill"
    scripts_dir = skill_dir / "scripts"
    refs_dir = skill_dir / "references"
    scripts_dir.mkdir(parents=True)
    refs_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: hello-skill\n"
        "description: Say hello\n"
        "---\n"
        "Use this skill to greet people.\n",
        encoding="utf-8",
    )
    (scripts_dir / "hello.py").write_text(
        "import sys\nprint('hello ' + ' '.join(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    (refs_dir / "notes.txt").write_text("reference notes\n", encoding="utf-8")
    return skill_dir


def _make_argparse_skill(tmp_path: Path) -> Path:
    skill_dir = tmp_path / "search-skill"
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: search-skill\n"
        "description: Search records\n"
        "---\n"
        "Use this skill to search records.\n",
        encoding="utf-8",
    )
    (scripts_dir / "search.py").write_text(
        "import argparse\n"
        "import json\n"
        "\n"
        "parser = argparse.ArgumentParser(description='Search records')\n"
        "parser.add_argument('path')\n"
        "parser.add_argument('--query', '-q', required=True)\n"
        "parser.add_argument('--limit', type=int, default=10)\n"
        "parser.add_argument('--dry-run', action='store_true')\n"
        "args = parser.parse_args()\n"
        "print(json.dumps(vars(args), sort_keys=True))\n",
        encoding="utf-8",
    )
    return skill_dir


def _make_function_skill(tmp_path: Path) -> Path:
    skill_dir = tmp_path / "math-skill"
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "name: math-skill\n"
        "description: Math helpers\n"
        "---\n"
        "Use this skill for small math helpers.\n",
        encoding="utf-8",
    )
    (scripts_dir / "math_tools.py").write_text(
        "from typing import Optional\n"
        "\n"
        "SCALE = 2\n"
        "OFFSETS = [(-1, 0), (1, 0)]\n"
        "\n"
        "def add(x: int, y: int = 1) -> int:\n"
        "    \"\"\"Add two numbers.\"\"\"\n"
        "    return x + y\n"
        "\n"
        "def label(value: str) -> str:\n"
        "    return f'label:{value}'\n"
        "\n"
        "def neighbors(x: int) -> list[tuple[int, int]]:\n"
        "    return [(x + dx, dy) for dx, dy in OFFSETS]\n"
        "\n"
        "def optional_label(value: Optional[str] = None) -> str:\n"
        "    return value or 'missing'\n"
        "\n"
        "def scale(baseMVA: float) -> float:\n"
        "    return baseMVA * SCALE\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    print(add(1, 2))\n",
        encoding="utf-8",
    )
    return skill_dir


def _run_generated_tests(package_dir: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{package_dir / 'src'}:{env.get('PYTHONPATH', '')}"
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(package_dir / "tests"), "-q"],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )


def test_inspect_text_skill_inventory(tmp_path):
    skill_dir = _make_text_skill(tmp_path)
    translator = TextSkillTranslator()

    inventory = translator.inspect_text_skill(skill_dir)

    assert inventory.skill_name == "hello-skill"
    assert inventory.description == "Say hello"
    assert inventory.body == "Use this skill to greet people."
    assert {file.path for file in inventory.files} == {
        "SKILL.md",
        "references/notes.txt",
        "scripts/hello.py",
    }
    script = inventory.scripts[0]
    assert script.path == "scripts/hello.py"
    assert script.kind == "script"
    assert script.sha256


def test_plan_conversion_creates_package_skill_names(tmp_path):
    skill_dir = _make_text_skill(tmp_path)
    translator = TextSkillTranslator()
    inventory = translator.inspect_text_skill(skill_dir)

    plan = translator.plan_conversion(inventory)

    assert plan.package_name == "hello_skill"
    assert plan.project_name == "hello-skill"
    assert plan.registry_name == "local.hello-skill"
    assert plan.class_name == "HelloSkill"
    assert plan.script_methods[0].method_name == "run_hello"
    assert plan.script_methods[0].interpreter == "sys.executable"


def test_plan_conversion_infers_argparse_api_method(tmp_path):
    skill_dir = _make_argparse_skill(tmp_path)
    translator = TextSkillTranslator()
    inventory = translator.inspect_text_skill(skill_dir)

    plan = translator.plan_conversion(inventory)

    method = plan.script_methods[0]
    assert method.method_name == "run_search"
    assert method.api_method_name == "search"
    assert [(arg.param_name, arg.cli_name, arg.required, arg.annotation, arg.action) for arg in method.arguments] == [
        ("path", None, True, "str", "store"),
        ("query", "--query", True, "str", "store"),
        ("limit", "--limit", False, "int", "store"),
        ("dry_run", "--dry-run", False, "bool", "store_true"),
    ]


def test_plan_conversion_infers_importable_function_methods(tmp_path):
    skill_dir = _make_function_skill(tmp_path)
    translator = TextSkillTranslator()
    inventory = translator.inspect_text_skill(skill_dir)

    plan = translator.plan_conversion(inventory)

    method = plan.script_methods[0]
    assert method.method_name == "run_math_tools"
    assert [function.method_name for function in method.function_methods] == [
        "add",
        "label",
        "neighbors",
        "optional_label",
        "scale",
    ]
    assert [(param.param_name, param.annotation, param.required, param.default) for param in method.function_methods[0].parameters] == [
        ("x", "int", True, None),
        ("y", "int", False, 1),
    ]
    assert method.function_methods[2].return_annotation == "list[tuple[int, int]]"
    assert [
        (param.param_name, param.annotation, param.required, param.default)
        for param in method.function_methods[3].parameters
    ] == [("value", "str | None", False, None)]


@pytest.mark.asyncio
async def test_translate_writes_valid_package_and_script_wrapper_runs(tmp_path):
    skill_dir = _make_text_skill(tmp_path)
    translator = TextSkillTranslator()

    result = translator.translate(skill_dir, tmp_path / "libs")
    report = translator.validate_package(result.package_dir)

    assert report.ok
    assert result.package_name == "hello_skill"
    assert result.registry_name == "local.hello-skill"
    assert "src/hello_skill/resources/SKILL.md" in result.files_written
    assert "src/hello_skill/resources/scripts/hello.py" in result.files_written
    assert (result.package_dir / "pyproject.toml").exists()
    assert (result.package_dir / "src" / "hello_skill" / "__init__.py").exists()
    generated_tests = _run_generated_tests(result.package_dir)
    assert generated_tests.returncode == 0, generated_tests.stdout + generated_tests.stderr

    agent = _Agent()
    registry = SkillRegistry(agent)
    registry.discover_libs(result.package_dir.parent)
    try:
        skill = registry[result.registry_name]
        assert "references/notes.txt" in skill.list_resources()
        assert skill.read_resource("references/notes.txt") == "reference notes\n"
        output = await skill.run_hello("world")
        assert output.strip() == "hello world"
    finally:
        await registry.aclose()


@pytest.mark.asyncio
async def test_translated_importable_script_has_function_api_methods(tmp_path):
    skill_dir = _make_function_skill(tmp_path)
    translator = TextSkillTranslator()

    result = translator.translate(skill_dir, tmp_path / "libs")
    report = translator.validate_package(result.package_dir)
    generated_tests = _run_generated_tests(result.package_dir)

    assert report.ok
    assert generated_tests.returncode == 0, generated_tests.stdout + generated_tests.stderr

    registry = SkillRegistry(_Agent())
    registry.discover_libs(result.package_dir.parent)
    try:
        skill = registry[result.registry_name]
        assert skill.add(3, y=4) == 7
        assert skill.label("item") == "label:item"
        assert skill.neighbors(10) == [(9, 0), (11, 0)]
        assert skill.optional_label() == "missing"
        assert skill.scale(baseMVA=5.0) == 10.0
        assert (await skill.run_math_tools()).strip() == "3"
    finally:
        await registry.aclose()


@pytest.mark.asyncio
async def test_translated_argparse_script_has_named_api_method(tmp_path):
    skill_dir = _make_argparse_skill(tmp_path)
    translator = TextSkillTranslator()

    result = translator.translate(skill_dir, tmp_path / "libs")
    report = translator.validate_package(result.package_dir)
    generated_tests = _run_generated_tests(result.package_dir)

    assert report.ok
    assert generated_tests.returncode == 0, generated_tests.stdout + generated_tests.stderr
    init_source = (result.package_dir / "src" / "search_skill" / "__init__.py").read_text()
    assert "async def search(" in init_source
    assert "async def run_search(" in init_source

    registry = SkillRegistry(_Agent())
    registry.discover_libs(result.package_dir.parent)
    try:
        skill = registry[result.registry_name]
        output = await skill.search("records.jsonl", "needle", limit=3, dry_run=True)
        assert json.loads(output) == {
            "dry_run": True,
            "limit": 3,
            "path": "records.jsonl",
            "query": "needle",
        }
    finally:
        await registry.aclose()


@pytest.mark.asyncio
async def test_agent_can_use_text_skill_and_translated_package_skill_equivalently(tmp_path):
    skill_dir = _make_text_skill(tmp_path)
    translator = TextSkillTranslator()
    result = translator.translate(skill_dir, tmp_path / "libs")
    code = (
        "import sys\n"
        "if hasattr(self.hello, 'run_hello'):\n"
        "    output = await self.hello.run_hello('world')\n"
        "else:\n"
        "    output = await self.hello.run_script('hello.py', 'world', interpreter=sys.executable)\n"
        "return_result(output)\n"
    )

    class SkillUsingAgent(Agent):
        def __init__(self, hello, **kwargs):
            super().__init__(**kwargs)
            self.hello = hello

        async def greet(self) -> str:
            """Use self.hello to greet world."""
            ...

    async def run_with_skill(skill) -> str:
        llm = _ScriptedCodeActLLM(code)
        agent = SkillUsingAgent(hello=skill, llm=llm)
        output = await agent.greet()
        assert llm.call_count == 1
        return output

    registry = SkillRegistry(_Agent())
    registry.discover_libs(result.package_dir.parent)
    try:
        text_output = await run_with_skill(TextSkill(path=skill_dir))
        package_output = await run_with_skill(registry[result.registry_name])
        assert text_output == package_output == "hello world"
    finally:
        await registry.aclose()


@pytest.mark.asyncio
async def test_script_wrapper_preserves_text_skill_root_cwd(tmp_path):
    skill_dir = _make_text_skill(tmp_path)
    (skill_dir / "scripts" / "read_ref.py").write_text(
        "from pathlib import Path\nprint(Path('references/notes.txt').read_text().strip())\n",
        encoding="utf-8",
    )
    translator = TextSkillTranslator()
    result = translator.translate(skill_dir, tmp_path / "libs")

    agent = _Agent()
    registry = SkillRegistry(agent)
    registry.discover_libs(result.package_dir.parent)
    try:
        skill = registry[result.registry_name]
        output = await skill.run_read_ref()
        assert output.strip() == "reference notes"
    finally:
        await registry.aclose()


def test_translate_requires_overwrite_for_existing_package(tmp_path):
    skill_dir = _make_text_skill(tmp_path)
    translator = TextSkillTranslator()
    output_dir = tmp_path / "libs"

    first = translator.translate(skill_dir, output_dir)
    marker = first.package_dir / "marker.txt"
    marker.write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError):
        translator.translate(skill_dir, output_dir)

    second = translator.translate(skill_dir, output_dir, overwrite=True)
    assert second.package_dir == first.package_dir
    assert not marker.exists()


def test_symlink_files_are_not_copied(tmp_path):
    skill_dir = _make_text_skill(tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_text("do not bundle\n", encoding="utf-8")
    try:
        os.symlink(secret, skill_dir / "references" / "leak.txt")
    except OSError:
        pytest.skip("symlink creation is not available")

    translator = TextSkillTranslator()
    inventory = translator.inspect_text_skill(skill_dir)
    result = translator.translate(skill_dir, tmp_path / "libs")

    assert "references/leak.txt" not in {file.path for file in inventory.files}
    assert "src/hello_skill/resources/references/leak.txt" not in result.files_written


def test_write_package_rejects_escaping_project_name(tmp_path):
    skill_dir = _make_text_skill(tmp_path)
    translator = TextSkillTranslator()
    plan = translator.plan_conversion(translator.inspect_text_skill(skill_dir))
    bad_plan = plan.model_copy(update={"project_name": "../outside"})

    with pytest.raises(ValueError, match="escapes"):
        translator.write_package(bad_plan, tmp_path / "libs", overwrite=True)

    assert not (tmp_path / "outside").exists()


def test_trailing_backslash_guidance_generates_valid_package(tmp_path):
    skill_dir = _make_text_skill(tmp_path)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: slash-skill\ndescription: Backslash skill\n---\nEnds with slash \\\\",
        encoding="utf-8",
    )
    translator = TextSkillTranslator()

    result = translator.translate(skill_dir, tmp_path / "libs")
    report = translator.validate_package(result.package_dir)

    assert report.ok


def test_generated_tests_pass_for_skill_without_scripts(tmp_path):
    skill_dir = tmp_path / "guide-skill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: guide-skill\ndescription: Guidance only\n---\nRead the guidance.\n",
        encoding="utf-8",
    )
    translator = TextSkillTranslator()
    result = translator.translate(skill_dir, tmp_path / "libs")

    generated_tests = _run_generated_tests(result.package_dir)

    assert generated_tests.returncode == 0, generated_tests.stdout + generated_tests.stderr


def test_unknown_non_executable_script_has_no_specific_wrapper(tmp_path):
    skill_dir = _make_text_skill(tmp_path)
    (skill_dir / "scripts" / "query.sql").write_text("select 1;\n", encoding="utf-8")
    translator = TextSkillTranslator()

    plan = translator.plan_conversion(translator.inspect_text_skill(skill_dir))

    assert "scripts/query.sql" not in {method.script_path for method in plan.script_methods}


def test_non_executable_python_shebang_script_gets_python_wrapper(tmp_path):
    skill_dir = _make_text_skill(tmp_path)
    script = skill_dir / "scripts" / "hello.py"
    script.write_text("#!/usr/bin/env python3\nimport sys\nprint('hello ' + sys.argv[1])\n", encoding="utf-8")
    script.chmod(0o644)
    translator = TextSkillTranslator()

    plan = translator.plan_conversion(translator.inspect_text_skill(skill_dir))

    method = next(method for method in plan.script_methods if method.script_path == "scripts/hello.py")
    assert method.method_name == "run_hello"
    assert method.interpreter == "sys.executable"
