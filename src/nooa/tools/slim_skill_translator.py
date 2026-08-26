# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Slim TextSkill-to-LibrarySkill translation policy."""

from __future__ import annotations

from nooa.tools.skill_translator import (
    ConversionPlan,
    OmittedScriptPlan,
    ScriptMethodPlan,
    TextSkillInventory,
    TextSkillTranslator,
    _build_docstring,
    _class_name,
    _default_interpreter,
    _infer_script_functions,
    _normalize_identifier,
    _resource_method_plans,
    _script_method_name,
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
        registry = registry_name or f"local.{inventory.skill_name}"
        cls_name = class_name or _class_name(package)

        used_script_names: set[str] = set()
        used_api_names: set[str] = set()
        script_methods: list[ScriptMethodPlan] = []
        omitted_scripts: list[OmittedScriptPlan] = []
        for file in inventory.scripts:
            interpreter = _default_interpreter(file)
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
                    method_name=_script_method_name(file.path, used_script_names),
                    interpreter=interpreter,
                    function_methods=function_methods,
                )
            )

        resource_methods = _resource_method_plans(inventory, used_api_names)
        docstring = _build_docstring(inventory, script_methods, resource_methods)

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
