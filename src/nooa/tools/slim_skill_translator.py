# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone slim TextSkill-to-LibrarySkill translator."""

from __future__ import annotations

import ast
import json
import keyword
import re
import shutil
import textwrap
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from nooa.skill import Skill, _find_skill_md, _parse_frontmatter

_RESOURCE_DOCSTRING_INLINE_LIMIT = 1000
_RESERVED_METHOD_NAMES = {
    "read_resource",
    "read_resource_bytes",
    "list_resources",
    "format_guidance",
    "_resource_root",
    *(name for name in dir(Skill) if not name.startswith("_")),
}


@dataclass(frozen=True)
class SkillFile:
    path: str


@dataclass(frozen=True)
class TextSkillInventory:
    source_dir: Path
    skill_name: str
    description: str
    body: str
    scripts: list[SkillFile] = field(default_factory=list)
    resources: list[SkillFile] = field(default_factory=list)


@dataclass(frozen=True)
class FunctionParameterPlan:
    param_name: str
    annotation: str = "object"
    required: bool = True
    default: str | int | float | bool | None = None


@dataclass(frozen=True)
class ScriptFunctionPlan:
    function_name: str
    method_name: str
    parameters: list[FunctionParameterPlan] = field(default_factory=list)
    return_annotation: str = "object"
    docstring: str = ""


@dataclass(frozen=True)
class ScriptMethodPlan:
    script_path: str
    function_methods: list[ScriptFunctionPlan] = field(default_factory=list)
    implementation_only: bool = False


@dataclass(frozen=True)
class OmittedScriptPlan:
    script_path: str
    reason: str

    def model_dump(self, *args, **kwargs) -> dict[str, str]:
        return {"script_path": self.script_path, "reason": self.reason}


@dataclass(frozen=True)
class ResourceMethodPlan:
    resource_path: str
    method_name: str
    return_annotation: Literal["str", "bytes"]
    docstring: str


@dataclass(frozen=True)
class ConversionPlan:
    source_dir: Path
    package_name: str
    project_name: str
    registry_name: str
    class_name: str
    description: str
    docstring: str
    script_methods: list[ScriptMethodPlan] = field(default_factory=list)
    omitted_scripts: list[OmittedScriptPlan] = field(default_factory=list)
    resource_methods: list[ResourceMethodPlan] = field(default_factory=list)
    resource_prefix: str = "resources"


@dataclass(frozen=True)
class PackageTranslationResult:
    package_dir: Path
    package_name: str
    registry_name: str
    class_name: str
    files_written: list[str]
    omitted_scripts: list[OmittedScriptPlan] = field(default_factory=list)


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    package_dir: Path
    registry_name: str | None = None
    loaded: bool = False
    importable: bool = False
    errors: list[str] = field(default_factory=list)

    def model_dump(self, *args, mode: str | None = None, **kwargs) -> dict[str, object]:
        package_dir: str | Path = str(self.package_dir) if mode == "json" else self.package_dir
        return {
            "ok": self.ok,
            "package_dir": package_dir,
            "registry_name": self.registry_name,
            "loaded": self.loaded,
            "importable": self.importable,
            "errors": list(self.errors),
        }


class SlimTextSkillTranslator(Skill):
    """Translate TextSkills with a small, LibrarySkill-native policy."""
    def inspect_text_skill(self, path: str | Path) -> TextSkillInventory:
        source_dir = Path(path).resolve()
        skill_md = _find_skill_md(source_dir)
        if skill_md is None:
            raise ValueError(f"SKILL.md not found in {source_dir}")

        frontmatter_raw, body = _parse_frontmatter(skill_md.read_text(encoding="utf-8"))
        if "name" not in frontmatter_raw:
            raise ValueError("Missing required frontmatter field: name")
        if "description" not in frontmatter_raw:
            raise ValueError("Missing required frontmatter field: description")

        scripts: list[SkillFile] = []
        resources: list[SkillFile] = []
        for file_path in _iter_skill_files(source_dir):
            rel = file_path.relative_to(source_dir).as_posix()
            if file_path == skill_md:
                continue
            file = SkillFile(path=rel)
            if rel.startswith("scripts/"):
                scripts.append(file)
            else:
                resources.append(file)

        return TextSkillInventory(
            source_dir=source_dir,
            skill_name=str(frontmatter_raw["name"]).strip(),
            description=str(frontmatter_raw["description"]).strip(),
            body=body,
            scripts=scripts,
            resources=resources,
        )

    def plan_conversion(
        self,
        inventory: TextSkillInventory,
        *,
        package_name: str | None = None,
        registry_name: str | None = None,
        class_name: str | None = None,
    ) -> ConversionPlan:
        package = _normalize_identifier(package_name or inventory.skill_name)
        project = package.replace("_", "-")
        registry = registry_name or f"local.{project}"
        cls_name = class_name or _class_name(package)

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

            function_methods = _infer_script_functions(inventory.source_dir / file.path, used_api_names)
            if not function_methods:
                omitted_scripts.append(
                    OmittedScriptPlan(
                        script_path=file.path,
                        reason="No import-safe public Python functions could be inferred.",
                    )
                )
                continue

            script_method = ScriptMethodPlan(
                script_path=file.path,
                function_methods=function_methods,
            )
            script_methods.append(script_method)
            script_methods_by_path[file.path] = script_method

        implementation_only_paths = _sibling_dependency_closure(
            inventory.source_dir,
            set(script_methods_by_path),
            {file.path for file in inventory.scripts},
        ) - set(script_methods_by_path)
        for script_path in sorted(implementation_only_paths):
            script_methods.append(
                ScriptMethodPlan(
                    script_path=script_path,
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

    def write_package(
        self,
        plan: ConversionPlan,
        output_dir: str | Path,
        *,
        overwrite: bool = False,
    ) -> PackageTranslationResult:
        root = Path(output_dir).resolve()
        _validate_identifier(plan.package_name, "package_name")
        _validate_class_name(plan.class_name)
        package_dir = _safe_child(root, plan.project_name)
        if package_dir.exists():
            if not overwrite:
                raise FileExistsError(f"{package_dir} already exists; pass overwrite=True")
            shutil.rmtree(package_dir)

        package_src = _safe_child(package_dir / "src", plan.package_name)
        resources_dir = _safe_child(package_src, plan.resource_prefix)
        resources_dir.mkdir(parents=True)

        written: list[str] = []
        _write(package_dir / "pyproject.toml", _render_pyproject(plan), package_dir, written)
        if plan.script_methods:
            (package_src / "_impl").mkdir(parents=True, exist_ok=True)
            _write(package_src / "_impl" / "__init__.py", "", package_dir, written)
            for method in plan.script_methods:
                _write(
                    package_src / "_impl" / f"{_implementation_module_name(method.script_path)}.py",
                    _render_implementation_module(plan.source_dir, method.script_path),
                    package_dir,
                    written,
                )
        _write(package_src / "__init__.py", _render_init(plan), package_dir, written)

        for source_file in _iter_package_resource_files(plan):
            rel = source_file.relative_to(plan.source_dir)
            dest = resources_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, dest)
            written.append(dest.relative_to(package_dir).as_posix())

        return PackageTranslationResult(
            package_dir=package_dir,
            package_name=plan.package_name,
            registry_name=plan.registry_name,
            class_name=plan.class_name,
            files_written=sorted(written),
            omitted_scripts=plan.omitted_scripts,
        )

    def translate(
        self,
        text_skill_dir: str | Path,
        output_dir: str | Path,
        *,
        package_name: str | None = None,
        registry_name: str | None = None,
        class_name: str | None = None,
        overwrite: bool = False,
    ) -> PackageTranslationResult:
        inventory = self.inspect_text_skill(text_skill_dir)
        plan = self.plan_conversion(
            inventory,
            package_name=package_name,
            registry_name=registry_name,
            class_name=class_name,
        )
        return self.write_package(plan, output_dir, overwrite=overwrite)

    def validate_package(self, package_dir: str | Path) -> ValidationReport:
        package_path = Path(package_dir).resolve()
        errors: list[str] = []
        registry_name = _read_registry_name(package_path)
        if registry_name is None:
            errors.append('No [project.entry-points."nooa.skills"] entry point found')
            return ValidationReport(ok=False, package_dir=package_path, errors=errors)

        importable = False
        loaded = False
        try:
            for py_file in package_path.rglob("*.py"):
                source = py_file.read_text(encoding="utf-8")
                compile(source, str(py_file), "exec")
            importable = True
        except Exception as exc:
            errors.append(f"Python compile failed: {exc}")

        if importable:
            try:
                loaded, loaded_names = _validate_registry_load(package_path, registry_name)
                if not loaded:
                    errors.append(f"{registry_name!r} was not loaded; loaded={loaded_names}")
            except Exception as exc:
                errors.append(f"SkillRegistry discovery failed: {exc}")

        return ValidationReport(
            ok=importable and loaded and not errors,
            package_dir=package_path,
            registry_name=registry_name,
            loaded=loaded,
            importable=importable,
            errors=errors,
        )


def _iter_skill_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"}
    )


def _iter_package_resource_files(plan: ConversionPlan) -> list[Path]:
    resources: list[Path] = []
    for path in _iter_skill_files(plan.source_dir):
        rel = path.relative_to(plan.source_dir).as_posix()
        if rel == "SKILL.md" or rel == "skill.md" or rel.startswith("scripts/"):
            continue
        resources.append(path)
    return resources


def _normalize_identifier(value: str) -> str:
    normalized = re.sub(r"\W+", "_", value.strip().lower()).strip("_")
    if not normalized:
        normalized = "translated_skill"
    if normalized[0].isdigit():
        normalized = f"skill_{normalized}"
    if keyword.iskeyword(normalized):
        normalized = f"{normalized}_skill"
    return normalized


def _validate_identifier(value: str, field_name: str) -> None:
    if not value.isidentifier() or keyword.iskeyword(value):
        raise ValueError(f"{field_name} must be a valid Python identifier, got {value!r}")


def _validate_class_name(value: str) -> None:
    _validate_identifier(value, "class_name")
    if not value[:1].isupper():
        raise ValueError(f"class_name should be PascalCase, got {value!r}")


def _safe_child(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if candidate == resolved_root or not candidate.is_relative_to(resolved_root):
        raise ValueError(f"Path {relative!r} escapes {resolved_root}")
    return candidate


def _class_name(value: str) -> str:
    parts = [part for part in re.split(r"[^0-9A-Za-z]+", value) if part]
    name = "".join(part[:1].upper() + part[1:] for part in parts) or "TranslatedSkill"
    if name[0].isdigit():
        name = f"Skill{name}"
    return name


def _function_method_name(function_name: str, used_names: set[str]) -> str:
    return _unique_name(_normalize_identifier(function_name), used_names, suffix="function")


def _resource_method_name(resource_path: str, used_names: set[str]) -> str:
    rel = Path(resource_path)
    return _unique_name(_normalize_identifier(rel.with_suffix("").as_posix()), used_names, suffix="resource")


def _unique_name(name: str, used_names: set[str], *, suffix: str) -> str:
    if name in _RESERVED_METHOD_NAMES:
        name = f"{name}_{suffix}"
    base = name
    index = 2
    while name in used_names:
        name = f"{base}_{index}"
        index += 1
    used_names.add(name)
    return name


def _resource_method_plans(inventory: TextSkillInventory, used_names: set[str]) -> list[ResourceMethodPlan]:
    methods: list[ResourceMethodPlan] = []
    for file in inventory.resources:
        path = inventory.source_dir / file.path
        text = _read_resource_text_for_docstring(path)
        return_annotation: Literal["str", "bytes"] = "str" if text is not None else "bytes"
        methods.append(
            ResourceMethodPlan(
                resource_path=file.path,
                method_name=_resource_method_name(file.path, used_names),
                return_annotation=return_annotation,
                docstring=_resource_method_docstring(
                    resource_path=file.path,
                    return_annotation=return_annotation,
                    text=text,
                ),
            )
        )
    return methods


def _read_resource_text_for_docstring(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if "\x00" in text:
        return None
    return text


def _resource_method_docstring(
    *,
    resource_path: str,
    return_annotation: Literal["str", "bytes"],
    text: str | None,
) -> str:
    if return_annotation == "bytes":
        return f"Return bundled binary resource `{resource_path}` as bytes."
    assert text is not None
    if len(text) <= _RESOURCE_DOCSTRING_INLINE_LIMIT:
        content = text
    else:
        content = (
            text[:_RESOURCE_DOCSTRING_INLINE_LIMIT].rstrip()
            + "\n\n[Truncated in docstring; call this method for the full resource.]"
        )
    return f"Return bundled text resource `{resource_path}`.\n\nResource contents:\n{content}"


def _infer_script_functions(path: Path, used_names: set[str]) -> list[ScriptFunctionPlan]:
    if path.suffix.lower() != ".py":
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    if not _module_top_level_is_import_safe(tree):
        return []

    methods: list[ScriptFunctionPlan] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name.startswith("_") or node.name in {"main", "app", "get_args"}:
            continue
        parameters = _function_parameters(node)
        if parameters is None:
            continue
        methods.append(
            ScriptFunctionPlan(
                function_name=node.name,
                method_name=_function_method_name(node.name, used_names),
                parameters=parameters,
                return_annotation=_safe_annotation(node.returns),
                docstring=ast.get_docstring(node) or "",
            )
        )
    return methods


def _module_top_level_is_import_safe(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            continue
        if _is_module_docstring(node) or _is_main_guard(node):
            continue
        if isinstance(node, ast.Assign) and _literal_container(node.value):
            continue
        if isinstance(node, ast.AnnAssign) and _literal_container(node.value):
            continue
        return False
    return True


def _is_module_docstring(node: ast.AST) -> bool:
    return isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)


def _is_main_guard(node: ast.AST) -> bool:
    test = node.test if isinstance(node, ast.If) else None
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def _literal_container(node: ast.AST | None) -> bool:
    if node is None:
        return True
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        return False
    return _literal_value(value)


def _literal_value(value: object) -> bool:
    if isinstance(value, (str, int, float, bool, type(None))):
        return True
    if isinstance(value, (list, tuple, set)):
        return all(_literal_value(item) for item in value)
    if isinstance(value, dict):
        return all(_literal_value(key) and _literal_value(item) for key, item in value.items())
    return False


def _function_parameters(node: ast.FunctionDef) -> list[FunctionParameterPlan] | None:
    args = node.args
    if args.posonlyargs or args.vararg or args.kwonlyargs or args.kwarg:
        return None
    defaults = list(args.defaults)
    required_count = len(args.args) - len(defaults)
    padded_defaults: list[ast.AST | None] = [None] * required_count + defaults
    parameters: list[FunctionParameterPlan] = []
    used_names: set[str] = set()
    for arg, default_node in zip(args.args, padded_defaults, strict=True):
        if not arg.arg or arg.arg in used_names:
            return None
        used_names.add(arg.arg)
        default, default_supported = _function_default(default_node)
        if not default_supported:
            return None
        parameters.append(
            FunctionParameterPlan(
                param_name=arg.arg,
                annotation=_safe_annotation(arg.annotation),
                required=default_node is None,
                default=default,
            )
        )
    return parameters


def _function_default(node: ast.AST | None) -> tuple[str | int | float | bool | None, bool]:
    if node is None:
        return None, True
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError):
        return None, False
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value, True
    return None, False


def _safe_annotation(node: ast.AST | None) -> str:
    if node is None:
        return "object"
    try:
        rendered = ast.unparse(node).strip()
    except Exception:
        return "object"
    return rendered if rendered and "\n" not in rendered else "object"


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


def _build_docstring(
    inventory: TextSkillInventory,
    script_methods: list[ScriptMethodPlan],
    resource_methods: list[ResourceMethodPlan],
    omitted_scripts: list[OmittedScriptPlan],
) -> str:
    title = inventory.description.strip() or inventory.skill_name
    lines = [
        title,
        "",
        "LibrarySkill-native guidance.",
        "Use the public Python APIs on this skill.",
    ]
    public_methods = _public_method_guidance(script_methods)
    if public_methods:
        lines.extend(
            [
                "",
                "Generated public APIs:",
                *public_methods,
                "",
                "Use these public Python methods as the supported capability interface.",
            ]
        )
    else:
        lines.extend(["", "Use the guidance and bundled resource APIs when relevant."])
    adapted_guidance = _adapt_skill_guidance(inventory.body, script_methods, resource_methods, omitted_scripts)
    if adapted_guidance:
        lines.extend(["", "Guidance:", adapted_guidance])
    if resource_methods:
        lines.extend(["", "Bundled resource APIs:", "Resource methods return bundled file contents; write script text to a workspace file before running it."])
        for resource in resource_methods:
            lines.append(
                f"- {resource.method_name}() -> {resource.return_annotation}: "
                f"returns `{resource.resource_path}` from package data."
            )
    return "\n".join(lines)


def _adapt_skill_guidance(
    body: str,
    script_methods: list[ScriptMethodPlan],
    resource_methods: list[ResourceMethodPlan],
    omitted_scripts: list[OmittedScriptPlan],
) -> str:
    guidance = body.strip()
    if not guidance:
        return ""

    guidance = re.sub(r"\bUse this skill\b", "Use this LibrarySkill", guidance)
    guidance = re.sub(r"\bthis skill\b", "this LibrarySkill", guidance)
    guidance = re.sub(r"\bthe skill\b", "the LibrarySkill", guidance)
    guidance = re.sub(r"\bSKILL\.md\b", "this LibrarySkill guidance", guidance)

    replacements: list[tuple[str, str]] = []
    for method in script_methods:
        if method.implementation_only:
            continue
        public_names = [function.method_name for function in method.function_methods]
        if not public_names:
            continue
        replacement = " or ".join(f"`{name}()`" for name in public_names)
        replacements.append((method.script_path, replacement))
        replacements.append((Path(method.script_path).name, replacement))

    for resource in resource_methods:
        replacement = f"`{resource.method_name}()`"
        file_replacement = f"a file containing the text returned by {replacement}"
        replacements.append((f"<path-to-this-skill>/{resource.resource_path}", file_replacement))
        replacements.append((f"<path-to-this-skill>/{Path(resource.resource_path).name}", file_replacement))
        replacements.append((resource.resource_path, replacement))
        replacements.append((Path(resource.resource_path).name, replacement))

    for old, new in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        guidance = _replace_reference(guidance, old, new)

    omitted_replacements: list[tuple[str, str]] = []
    for omitted in omitted_scripts:
        replacement = "the relevant LibrarySkill guidance"
        omitted_replacements.append((omitted.script_path, replacement))
        omitted_replacements.append((Path(omitted.script_path).name, replacement))
    for old, new in sorted(omitted_replacements, key=lambda item: len(item[0]), reverse=True):
        guidance = _replace_reference(guidance, old, new)

    return re.sub(r"`?scripts/[^`\s),.;:]+" + "`?", "the relevant LibrarySkill guidance", guidance)


def _replace_reference(text: str, old: str, new: str) -> str:
    return re.sub(rf"`?{re.escape(old)}`?", new, text)


def _public_method_guidance(script_methods: list[ScriptMethodPlan]) -> list[str]:
    lines: list[str] = []
    for method in script_methods:
        if method.implementation_only:
            continue
        for function in method.function_methods:
            parameters = ", ".join(_guidance_parameter(parameter) for parameter in function.parameters)
            lines.append(
                f"- {function.method_name}({parameters}) -> {function.return_annotation}: "
                "returns the Python value from the library implementation."
            )
    return lines


def _guidance_parameter(parameter: FunctionParameterPlan) -> str:
    suffix = "" if parameter.required else f" = {parameter.default!r}"
    return f"{parameter.param_name}: {parameter.annotation}{suffix}"


def _render_pyproject(plan: ConversionPlan) -> str:
    return textwrap.dedent(f"""\
        [project]
        name = {json.dumps(plan.project_name)}
        version = "0.1.0"
        description = {json.dumps(plan.description)}
        dependencies = ["nooa"]

        [build-system]
        requires = ["setuptools>=68"]
        build-backend = "setuptools.build_meta"

        [tool.setuptools.packages.find]
        where = ["src"]

        [tool.setuptools.package-data]
        {plan.package_name} = ["resources/**"]

        [project.entry-points."nooa.skills"]
        {json.dumps(plan.registry_name)} = {json.dumps(f"{plan.package_name}:{plan.class_name}")}
    """)


def _render_implementation_module(source_dir: Path, script_path: str) -> str:
    path = source_dir / script_path
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        raise ValueError(f"Cannot render implementation module for {path}: {exc}") from exc
    body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.FunctionDef) and node.name == "main")
        and not _is_main_guard(node)
    ]
    module = ast.Module(body=body, type_ignores=[])
    module = _rewrite_sibling_script_imports(module, source_dir, script_path)
    ast.fix_missing_locations(module)
    return ast.unparse(module) + "\n"


def _implementation_module_name(script_path: str) -> str:
    return f"_{_normalize_identifier(Path(script_path).with_suffix('').as_posix())}"


def _rewrite_sibling_script_imports(module: ast.Module, source_dir: Path, script_path: str) -> ast.Module:
    script_dir = Path(script_path).parent
    body: list[ast.stmt] = []
    for statement in module.body:
        replacement = _rewrite_sibling_import(statement, source_dir, script_dir)
        if replacement is None:
            body.append(statement)
        else:
            body.extend(replacement)
    module.body = body
    return module


def _rewrite_sibling_import(statement: ast.stmt, source_dir: Path, script_dir: Path) -> list[ast.stmt] | None:
    if isinstance(statement, ast.Import):
        rewritten: list[ast.stmt] = []
        unchanged: list[ast.alias] = []
        for alias in statement.names:
            module_name = _sibling_impl_module(source_dir, script_dir, alias.name)
            if module_name is None:
                unchanged.append(alias)
            else:
                rewritten.append(
                    ast.ImportFrom(
                        module="",
                        names=[ast.alias(name=module_name, asname=alias.asname or alias.name)],
                        level=1,
                    )
                )
        if unchanged:
            rewritten.insert(0, ast.Import(names=unchanged))
        return rewritten if rewritten else None
    if isinstance(statement, ast.ImportFrom) and statement.level == 0 and statement.module:
        module_name = _sibling_impl_module(source_dir, script_dir, statement.module)
        if module_name is not None:
            return [ast.ImportFrom(module=module_name, names=statement.names, level=1)]
    return None


def _sibling_impl_module(source_dir: Path, script_dir: Path, import_name: str) -> str | None:
    if "." in import_name:
        return None
    sibling = script_dir / f"{import_name}.py"
    if not (source_dir / sibling).is_file():
        return None
    return _implementation_module_name(sibling.as_posix())


def _render_init(plan: ConversionPlan) -> str:
    resource_methods = "\n".join(_render_resource_method(resource) for resource in plan.resource_methods)
    function_methods = "\n".join(
        _render_function_method(method, function)
        for method in plan.script_methods
        if not method.implementation_only
        for function in method.function_methods
    )
    methods = "\n".join(part for part in (resource_methods, function_methods) if part)
    if methods:
        methods = "\n" + methods
    docstring = textwrap.indent(repr(plan.docstring), "    ")
    context_key = f"skill:{plan.registry_name}"
    attr_name = _normalize_identifier(plan.registry_name.split(".")[-1])
    template = textwrap.dedent(f'''\
        from __future__ import annotations

        from pathlib import Path

        from nooa.agentdoc import hidden
        from nooa.skill import Skill


        class {plan.class_name}(Skill):
        __DOCSTRING__

            context_block = ({context_key!r}, "self.{attr_name}.format_guidance()")

            def _resource_root(self):
                return Path(__file__).parent / "{plan.resource_prefix}"

            @hidden
            def format_guidance(self) -> str:
                """Return the LibrarySkill-native guidance."""
                return type(self).__doc__ or ""

        __METHODS__
    ''')
    return template.replace("__DOCSTRING__", docstring).replace("__METHODS__", methods)


def _render_resource_method(resource: ResourceMethodPlan) -> str:
    body = (
        f"return (self._resource_root() / {resource.resource_path!r}).read_text(encoding='utf-8')"
        if resource.return_annotation == "str"
        else f"return (self._resource_root() / {resource.resource_path!r}).read_bytes()"
    )
    lines = [
        f"def {resource.method_name}(self) -> {resource.return_annotation}:",
        f"    {resource.docstring!r}",
        f"    {body}",
        "",
    ]
    return textwrap.indent("\n".join(lines), "    ")


def _render_function_method(method: ScriptMethodPlan, function: ScriptFunctionPlan) -> str:
    signature_parts = [_render_function_parameter(parameter) for parameter in function.parameters]
    signature = ", ".join(signature_parts)
    if signature:
        signature = f", {signature}"
    call_args = ", ".join(f"{parameter.param_name}={parameter.param_name}" for parameter in function.parameters)
    docstring = function.docstring.strip()
    if not docstring:
        docstring = (
            "Return the Python value from the library implementation."
            if function.return_annotation == "object"
            else f"Return `{function.return_annotation}` from the library implementation."
        )
    lines = [
        f"def {function.method_name}(self{signature}) -> {function.return_annotation}:",
        f"    {docstring!r}",
        f"    from ._impl import {_implementation_module_name(method.script_path)} as module",
        f"    return module.{function.function_name}({call_args})",
        "",
    ]
    return textwrap.indent("\n".join(lines), "    ")


def _render_function_parameter(parameter: FunctionParameterPlan) -> str:
    if parameter.required:
        return f"{parameter.param_name}: {parameter.annotation}"
    return f"{parameter.param_name}: {parameter.annotation} = {parameter.default!r}"


def _write(path: Path, content: str, package_dir: Path, written: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    written.append(path.relative_to(package_dir).as_posix())


def _read_registry_name(package_dir: Path) -> str | None:
    import tomllib

    pyproject = package_dir / "pyproject.toml"
    if not pyproject.exists():
        return None
    with pyproject.open("rb") as handle:
        data = tomllib.load(handle)
    entries = data.get("project", {}).get("entry-points", {}).get("nooa.skills", {})
    if not entries:
        return None
    return str(next(iter(entries.keys())))


def _validate_registry_load(package_path: Path, registry_name: str) -> tuple[bool, list[str]]:
    result: dict[str, object] = {}
    error: list[BaseException] = []

    def worker() -> None:
        try:
            result["value"] = _validate_registry_load_sync(package_path, registry_name)
        except BaseException as exc:
            error.append(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=30)
    if thread.is_alive():
        raise TimeoutError(f"SkillRegistry discovery timed out for {package_path}")
    if error:
        raise error[0]
    value = result.get("value")
    if not isinstance(value, tuple):
        raise RuntimeError("SkillRegistry validation did not return a result")
    loaded, names = value
    return bool(loaded), list(names)


def _validate_registry_load_sync(package_path: Path, registry_name: str) -> tuple[bool, list[str]]:
    from nooa.skill_registry import SkillRegistry
    class Agent:
        pass

    registry = SkillRegistry(Agent())
    try:
        registry.discover_libs(package_path.parent)
        loaded_names = registry.loaded()
        return registry_name in loaded_names, loaded_names
    finally:
        registry.close()
