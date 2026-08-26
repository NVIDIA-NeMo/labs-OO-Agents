# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Slim TextSkill-to-LibrarySkill translation policy."""

from __future__ import annotations

import ast
from pathlib import Path

from nooa.tools.skill_translator import (
    ConversionPlan,
    OmittedScriptPlan,
    ScriptMethodPlan,
    TextSkillInventory,
    TextSkillTranslator,
    _build_docstring,
    _class_name,
    _infer_script_functions,
    _normalize_identifier,
    _resource_method_plans,
)


class SlimTextSkillTranslator(TextSkillTranslator):
    """Translate TextSkills using only stable LibrarySkill-native primitives.

    This translator keeps the package writer, validator, guidance rendering, and
    resource handling from ``TextSkillTranslator``, but narrows script planning
    to import-safe Python functions. It deliberately does not synthesize native
    APIs from argparse or CLI-shaped scripts.
    """

    def plan_conversion(
        self,
        inventory: TextSkillInventory,
        *,
        package_name: str | None = None,
        registry_name: str | None = None,
        class_name: str | None = None,
    ) -> ConversionPlan:
        """Create a compact package-skill conversion plan from an inventory."""
        package = _normalize_identifier(package_name or inventory.skill_name)
        project = package.replace("_", "-")
        registry = registry_name or f"local.{project}"
        cls_name = class_name or _class_name(package)

        used_script_names: set[str] = set()
        used_api_names: set[str] = set()
        script_methods: list[ScriptMethodPlan] = []
        script_methods_by_path: dict[str, ScriptMethodPlan] = {}
        omitted_scripts: list[OmittedScriptPlan] = []
        for file in inventory.scripts:
            if not file.path.lower().endswith(".py"):
                omitted_scripts.append(
                    OmittedScriptPlan(
                        script_path=file.path,
                        reason="No import-safe Python API could be inferred.",
                    )
                )
                continue

            script_path = inventory.source_dir / file.path
            function_methods = _infer_script_functions(script_path, used_api_names)
            if not function_methods:
                omitted_scripts.append(
                    OmittedScriptPlan(
                        script_path=file.path,
                        reason="No import-safe public Python functions could be inferred.",
                    )
                )
                continue

            script_methods.append(
                ScriptMethodPlan(
                    script_path=file.path,
                    method_name=_implementation_method_name(file.path, used_script_names),
                    interpreter=None,
                    function_methods=function_methods,
                )
            )
            script_methods_by_path[file.path] = script_methods[-1]

        implementation_only_paths = _sibling_dependency_closure(
            inventory.source_dir,
            set(script_methods_by_path),
            {file.path for file in inventory.scripts},
        ) - set(script_methods_by_path)
        for script_path in sorted(implementation_only_paths):
            script_methods.append(
                ScriptMethodPlan(
                    script_path=script_path,
                    method_name=_implementation_method_name(script_path, used_script_names),
                    interpreter=None,
                    implementation_only=True,
                )
            )
        if implementation_only_paths:
            omitted_scripts = [
                omitted
                for omitted in omitted_scripts
                if omitted.script_path not in implementation_only_paths
            ]

        resource_methods = _resource_method_plans(inventory, used_api_names)
        docstring = _build_docstring(inventory, script_methods, resource_methods, omitted_scripts)

        return ConversionPlan(
            source_dir=inventory.source_dir,
            package_name=package,
            project_name=project,
            registry_name=registry,
            class_name=cls_name,
            description=inventory.description,
            docstring=docstring,
            script_methods=script_methods,
            omitted_scripts=omitted_scripts,
            resource_methods=resource_methods,
        )


def _implementation_method_name(script_path: str, used_names: set[str]) -> str:
    name = f"impl_{_normalize_identifier(Path(script_path).with_suffix('').as_posix())}"
    base = name
    index = 2
    while name in used_names:
        name = f"{base}_{index}"
        index += 1
    used_names.add(name)
    return name


def _sibling_dependency_closure(source_dir: Path, roots: set[str], script_paths: set[str]) -> set[str]:
    dependencies: set[str] = set()
    pending = list(roots)
    seen: set[str] = set()
    while pending:
        script_path = pending.pop()
        if script_path in seen:
            continue
        seen.add(script_path)
        for dependency in _sibling_python_imports(source_dir, script_path, script_paths):
            if dependency not in dependencies:
                dependencies.add(dependency)
                pending.append(dependency)
    return dependencies


def _sibling_python_imports(source_dir: Path, script_path: str, script_paths: set[str]) -> set[str]:
    path = source_dir / script_path
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return set()

    script_dir = Path(script_path).parent
    imports: set[str] = set()
    for statement in tree.body:
        candidate_names: list[str] = []
        if isinstance(statement, ast.Import):
            candidate_names.extend(alias.name for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom) and statement.level == 0 and statement.module:
            candidate_names.append(statement.module)
        for name in candidate_names:
            if "." in name:
                continue
            candidate = (script_dir / f"{name}.py").as_posix()
            if candidate in script_paths and (source_dir / candidate).is_file():
                imports.add(candidate)
    return imports
