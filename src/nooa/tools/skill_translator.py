# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate SKILL.md TextSkills into package-backed nooa Skill libraries."""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import keyword
import re
import shutil
import textwrap
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from nooa.skill import Skill, _find_skill_md, _parse_frontmatter

_RESERVED_METHOD_NAMES = {
    "run_resource_script",
    "read_resource",
    "read_resource_bytes",
    "list_resources",
    "format_guidance",
    "_resource_root",
    "_list_resources",
    "_read_resource",
    "_read_resource_bytes",
    "_format_resource_index",
    "_RESOURCE_METHODS",
    *(name for name in dir(Skill) if not name.startswith("_")),
}

_RESOURCE_DOCSTRING_INLINE_LIMIT = 1000


class TextSkillFile(BaseModel):
    """One file found inside a TextSkill directory."""

    path: str
    kind: Literal["skill", "script", "resource"]
    size_bytes: int
    sha256: str
    executable: bool = False
    shebang: str | None = None


class TextSkillInventory(BaseModel):
    """Parsed inventory for a SKILL.md TextSkill directory."""

    source_dir: Path
    skill_name: str
    description: str
    frontmatter: dict[str, str] = Field(default_factory=dict)
    body: str
    files: list[TextSkillFile] = Field(default_factory=list)

    @property
    def scripts(self) -> list[TextSkillFile]:
        return [file for file in self.files if file.kind == "script"]


class ScriptArgumentPlan(BaseModel):
    """One inferred parameter for a script-backed package API."""

    param_name: str
    cli_name: str | None = None
    positional: bool = False
    required: bool = False
    annotation: Literal["str", "int", "float", "bool"] = "str"
    default: str | int | float | bool | None = None
    action: Literal["store", "store_true", "store_false"] = "store"


class FunctionParameterPlan(BaseModel):
    """One inferred Python function parameter for a generated module API."""

    param_name: str
    annotation: str = "object"
    required: bool = True
    default: str | int | float | bool | None = None


class ScriptFunctionPlan(BaseModel):
    """Plan for one generated wrapper around a Python function in a script."""

    function_name: str
    method_name: str
    parameters: list[FunctionParameterPlan] = Field(default_factory=list)
    return_annotation: str = "object"
    docstring: str = ""


class ScriptMethodPlan(BaseModel):
    """Plan for one generated wrapper method around a bundled script."""

    script_path: str
    method_name: str
    interpreter: str | None = None
    api_method_name: str | None = None
    arguments: list[ScriptArgumentPlan] = Field(default_factory=list)
    function_methods: list[ScriptFunctionPlan] = Field(default_factory=list)


class OmittedScriptPlan(BaseModel):
    """One script intentionally left out of the generated package API."""

    script_path: str
    reason: str


class ResourceMethodPlan(BaseModel):
    """Plan for one named method exposing a bundled non-script resource."""

    resource_path: str
    method_name: str
    return_annotation: Literal["str", "bytes"]
    size_bytes: int
    docstring: str


class ConversionPlan(BaseModel):
    """Deterministic plan for converting a TextSkill into a package skill."""

    source_dir: Path
    package_name: str
    project_name: str
    registry_name: str
    class_name: str
    description: str
    docstring: str
    script_methods: list[ScriptMethodPlan] = Field(default_factory=list)
    omitted_scripts: list[OmittedScriptPlan] = Field(default_factory=list)
    resource_methods: list[ResourceMethodPlan] = Field(default_factory=list)
    resource_prefix: str = "resources"


class PackageTranslationResult(BaseModel):
    """Result of writing a package skill to disk."""

    package_dir: Path
    package_name: str
    registry_name: str
    class_name: str
    files_written: list[str]
    omitted_scripts: list[OmittedScriptPlan] = Field(default_factory=list)


class ValidationReport(BaseModel):
    """Validation outcome for a generated package skill."""

    ok: bool
    package_dir: Path
    registry_name: str | None = None
    loaded: bool = False
    importable: bool = False
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def __str__(self) -> str:
        status = "OK" if self.ok else "ERROR"
        details = [status]
        if self.registry_name:
            details.append(f"registry_name={self.registry_name}")
        if self.errors:
            details.extend(f"error: {error}" for error in self.errors)
        if self.warnings:
            details.extend(f"warning: {warning}" for warning in self.warnings)
        return "\n".join(details)


@dataclass(frozen=True)
class NativeArgparseExecution:
    """Native method rendering plan for one argparse-backed script."""

    args_name: str
    statements: list[ast.stmt]
    import_lines: list[str]
    needs_module: bool
    implementation_body: list[ast.stmt] | None = None


class TextSkillTranslator(Skill):
    """Translate traditional SKILL.md TextSkills into package-backed Skill libraries.

    The translator builds a package-style library from TextSkill content. It
    preserves non-script resources and only bundles scripts that back inferred
    public package APIs.

    Typical flow:

        inventory = self.skill_translator.inspect_text_skill("skills/frontend")
        plan = self.skill_translator.plan_conversion(inventory)
        result = self.skill_translator.write_package(plan, "libs")
        report = self.skill_translator.validate_package(result.package_dir)

    The generated package contains:
    - pyproject.toml with a nooa.skills entry point
    - src/<package_name>/__init__.py exporting a Skill subclass
    - copied non-script resources under src/<package_name>/resources/
    - baseline pytest tests under tests/

    Conversion is intentionally conservative. Script files are copied unchanged
    only when they are needed as private implementation details for inferred
    argparse-backed or import-safe function APIs.
    """

    def inspect_text_skill(self, path: str | Path) -> TextSkillInventory:
        """Parse a TextSkill directory into structured metadata and file inventory."""
        source_dir = Path(path).resolve()
        skill_md = _find_skill_md(source_dir)
        if skill_md is None:
            raise ValueError(f"SKILL.md not found in {source_dir}")

        frontmatter_raw, body = _parse_frontmatter(skill_md.read_text(encoding="utf-8"))
        if "name" not in frontmatter_raw:
            raise ValueError("Missing required frontmatter field: name")
        if "description" not in frontmatter_raw:
            raise ValueError("Missing required frontmatter field: description")

        files: list[TextSkillFile] = []
        for file_path in _iter_skill_files(source_dir):
            rel = file_path.relative_to(source_dir).as_posix()
            kind: Literal["skill", "script", "resource"]
            if file_path == skill_md:
                kind = "skill"
            elif rel.startswith("scripts/"):
                kind = "script"
            else:
                kind = "resource"
            files.append(
                TextSkillFile(
                    path=rel,
                    kind=kind,
                    size_bytes=file_path.stat().st_size,
                    sha256=_sha256(file_path),
                    executable=_is_executable(file_path),
                    shebang=_read_shebang(file_path),
                )
            )

        frontmatter = {str(key): _stringify_frontmatter_value(value) for key, value in frontmatter_raw.items()}
        return TextSkillInventory(
            source_dir=source_dir,
            skill_name=frontmatter["name"].strip(),
            description=frontmatter["description"].strip(),
            frontmatter=frontmatter,
            body=body,
            files=files,
        )

    def plan_conversion(
        self,
        inventory: TextSkillInventory,
        *,
        package_name: str | None = None,
        registry_name: str | None = None,
        class_name: str | None = None,
    ) -> ConversionPlan:
        """Create a deterministic package-skill conversion plan from an inventory."""
        package = _normalize_identifier(package_name or inventory.skill_name)
        project = package.replace("_", "-")
        registry = registry_name or f"local.{inventory.skill_name}"
        cls_name = class_name or _class_name(package)

        used_names: set[str] = set()
        used_api_names: set[str] = set()
        script_methods: list[ScriptMethodPlan] = []
        omitted_scripts: list[OmittedScriptPlan] = []
        for file in inventory.scripts:
            interpreter = _default_interpreter(file)
            if interpreter is None and not file.executable:
                omitted_scripts.append(
                    OmittedScriptPlan(
                        script_path=file.path,
                        reason="No supported Python or shell entry point was detected.",
                    )
                )
                continue
            script_path = inventory.source_dir / file.path

            arguments = _infer_script_arguments(script_path)
            can_render_argparse = bool(arguments) and _can_render_native_argparse_api(script_path)
            api_method_name = (
                _api_method_name(file.path, used_api_names)
                if can_render_argparse
                else None
            )
            function_methods = _infer_script_functions(script_path, used_api_names)
            if not api_method_name and not function_methods:
                omitted_scripts.append(
                    OmittedScriptPlan(
                        script_path=file.path,
                        reason=_omission_reason(script_path),
                    )
                )
                continue
            script_methods.append(
                ScriptMethodPlan(
                    script_path=file.path,
                    method_name=_script_method_name(file.path, used_names),
                    interpreter=interpreter,
                    api_method_name=api_method_name,
                    arguments=arguments,
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

    def write_package(
        self,
        plan: ConversionPlan,
        output_dir: str | Path,
        *,
        overwrite: bool = False,
    ) -> PackageTranslationResult:
        """Write the planned package skill under output_dir and copy resources."""
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
        tests_dir = package_dir / "tests"
        resources_dir.mkdir(parents=True)
        tests_dir.mkdir(parents=True)

        written: list[str] = []
        for relative in ("pyproject.toml", "README.md", f"tests/test_{plan.package_name}.py"):
            (package_dir / relative).parent.mkdir(parents=True, exist_ok=True)

        _write(package_dir / "pyproject.toml", _render_pyproject(plan), package_dir, written)
        _write(package_dir / "README.md", _render_readme(plan), package_dir, written)
        implementation_modules = _implementation_modules(plan)
        if implementation_modules:
            (package_src / "_impl").mkdir(parents=True, exist_ok=True)
            _write(package_src / "_impl" / "__init__.py", "", package_dir, written)
            for method in implementation_modules:
                _write(
                    package_src / "_impl" / f"{_implementation_module_name(method.script_path)}.py",
                    _render_implementation_module(plan.source_dir, method.script_path),
                    package_dir,
                    written,
                )
        _write(package_src / "__init__.py", _render_init(plan), package_dir, written)
        _write(
            tests_dir / f"test_{plan.package_name}.py",
            _render_tests(plan),
            package_dir,
            written,
        )

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
        """Inspect, plan, and write a package skill in one deterministic call."""
        inventory = self.inspect_text_skill(text_skill_dir)
        plan = self.plan_conversion(
            inventory,
            package_name=package_name,
            registry_name=registry_name,
            class_name=class_name,
        )
        return self.write_package(plan, output_dir, overwrite=overwrite)

    def validate_package(self, package_dir: str | Path) -> ValidationReport:
        """Validate that a generated package imports and loads through SkillRegistry."""
        package_path = Path(package_dir).resolve()
        errors: list[str] = []
        warnings: list[str] = []
        registry_name = _read_registry_name(package_path)
        if registry_name is None:
            errors.append("No [project.entry-points.\"nooa.skills\"] entry point found")
            return ValidationReport(ok=False, package_dir=package_path, errors=errors)

        importable = False
        loaded = False
        try:
            for py_file in package_path.rglob("*.py"):
                source = py_file.read_text(encoding="utf-8")
                compile(source, str(py_file), "exec")
            importable = True
        except Exception as exc:  # pragma: no cover - exact py_compile errors vary
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
            warnings=warnings,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        if rel == "SKILL.md" or rel.startswith("scripts/"):
            continue
        resources.append(path)
    return resources


def _is_executable(path: Path) -> bool:
    return bool(path.stat().st_mode & 0o111)


def _read_shebang(path: Path) -> str | None:
    try:
        with path.open("rb") as handle:
            line = handle.readline(200)
    except OSError:
        return None
    if line.startswith(b"#!"):
        return line.decode("utf-8", errors="replace").strip()
    return None


def _stringify_frontmatter_value(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


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


def _script_method_name(script_path: str, used_names: set[str]) -> str:
    stem = Path(script_path).stem
    name = f"run_{_normalize_identifier(stem)}"
    if name in {"run_resource_script", "read_resource", "list_resources"}:
        name = f"{name}_script"
    base = name
    index = 2
    while name in used_names:
        name = f"{base}_{index}"
        index += 1
    used_names.add(name)
    return name


def _resource_method_name(resource_path: str, used_names: set[str]) -> str:
    rel = Path(resource_path)
    without_suffix = rel.with_suffix("").as_posix()
    name = _normalize_identifier(without_suffix)
    if name in _RESERVED_METHOD_NAMES:
        name = f"{name}_resource"
    base = name
    index = 2
    while name in used_names:
        name = f"{base}_{index}"
        index += 1
    used_names.add(name)
    return name


def _api_method_name(script_path: str, used_names: set[str]) -> str:
    name = _normalize_identifier(Path(script_path).stem)
    if name in _RESERVED_METHOD_NAMES:
        name = f"{name}_api"
    base = name
    index = 2
    while name in used_names:
        name = f"{base}_{index}"
        index += 1
    used_names.add(name)
    return name


def _resource_method_plans(inventory: TextSkillInventory, used_names: set[str]) -> list[ResourceMethodPlan]:
    methods: list[ResourceMethodPlan] = []
    for file in inventory.files:
        if file.kind != "resource":
            continue
        path = inventory.source_dir / file.path
        text = _read_resource_text_for_docstring(path)
        return_annotation: Literal["str", "bytes"] = "str" if text is not None else "bytes"
        method_name = _resource_method_name(file.path, used_names)
        methods.append(
            ResourceMethodPlan(
                resource_path=file.path,
                method_name=method_name,
                return_annotation=return_annotation,
                size_bytes=file.size_bytes,
                docstring=_resource_method_docstring(
                    resource_path=file.path,
                    return_annotation=return_annotation,
                    size_bytes=file.size_bytes,
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
    size_bytes: int,
    text: str | None,
) -> str:
    if return_annotation == "bytes":
        return (
            f"Return bundled binary resource `{resource_path}` as bytes.\n\n"
            f"Size: {size_bytes} bytes."
        )
    assert text is not None
    if len(text) <= _RESOURCE_DOCSTRING_INLINE_LIMIT:
        content = text
    else:
        content = (
            text[:_RESOURCE_DOCSTRING_INLINE_LIMIT].rstrip()
            + "\n\n[Truncated in docstring; call this method for the full resource.]"
        )
    return (
        f"Return bundled text resource `{resource_path}`.\n\n"
        "Resource contents:\n"
        f"{content}"
    )


def _default_interpreter(file: TextSkillFile) -> str | None:
    if file.shebang and file.executable:
        return None
    suffix = Path(file.path).suffix.lower()
    if suffix == ".py":
        return "sys.executable"
    if suffix in {".sh", ".bash"}:
        return '"bash"'
    return None


def _infer_script_arguments(path: Path) -> list[ScriptArgumentPlan]:
    """Infer a named package API from simple argparse declarations."""
    if path.suffix.lower() != ".py":
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []

    constants = _literal_module_constants(tree)
    arguments: list[ScriptArgumentPlan] = []
    used_names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add_subparsers":
            return []
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        argument = _argument_from_add_argument(node, used_names, constants)
        if argument is not None:
            arguments.append(argument)
        else:
            return []
    return arguments


def _infer_script_functions(path: Path, used_names: set[str]) -> list[ScriptFunctionPlan]:
    """Infer direct package methods for safe top-level Python functions."""
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
        method_name = _function_method_name(node.name, used_names)
        methods.append(
            ScriptFunctionPlan(
                function_name=node.name,
                method_name=method_name,
                parameters=parameters,
                return_annotation=_safe_annotation(node.returns),
                docstring=ast.get_docstring(node) or "",
            )
        )
    return methods


def _module_top_level_is_import_safe(tree: ast.Module) -> bool:
    """Return true only when importing the script should not execute work.

    Function wrappers import the original script as a resource module. We keep
    this deliberately conservative: imports, definitions, literal constants,
    module docstrings, and `if __name__ == "__main__"` blocks are fine; arbitrary
    top-level calls are not.
    """
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
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if not isinstance(test, ast.Compare):
        return False
    if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
        return False
    if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
        return False
    if len(test.comparators) != 1:
        return False
    comparator = test.comparators[0]
    return isinstance(comparator, ast.Constant) and comparator.value == "__main__"


def _literal_container(node: ast.AST | None) -> bool:
    if node is None:
        return True
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (str, int, float, bool, type(None)))
    if _literal_number(node) is not None:
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_literal_container(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            _literal_container(key) and _literal_container(value)
            for key, value in zip(node.keys, node.values, strict=True)
        )
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
        param_name = arg.arg
        if not param_name or param_name in used_names:
            return None
        used_names.add(param_name)
        default, default_supported = _function_default(default_node)
        if not default_supported:
            return None
        parameters.append(
            FunctionParameterPlan(
                param_name=param_name,
                annotation=_safe_annotation(arg.annotation),
                required=default_node is None,
                default=default,
            )
        )
    return parameters


def _function_method_name(function_name: str, used_names: set[str]) -> str:
    name = _normalize_identifier(function_name)
    if name in _RESERVED_METHOD_NAMES:
        name = f"{name}_function"
    base = name
    index = 2
    while name in used_names:
        name = f"{base}_{index}"
        index += 1
    used_names.add(name)
    return name


def _safe_annotation(node: ast.AST | None) -> str:
    if isinstance(node, ast.Name) and node.id in {"str", "int", "float", "bool"}:
        return node.id
    if isinstance(node, ast.Constant) and node.value is None:
        return "None"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _safe_annotation(node.left)
        right = _safe_annotation(node.right)
        if "object" in {left, right}:
            return "object"
        return f"{left} | {right}"
    if isinstance(node, ast.Subscript):
        base = _annotation_name(node.value)
        if base == "Optional":
            inner = _safe_annotation(node.slice)
            return "object | None" if inner == "object" else f"{inner} | None"
        if base == "Union":
            elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            rendered = [_safe_annotation(element) for element in elements]
            if any(annotation == "object" for annotation in rendered):
                return "object"
            return " | ".join(rendered)
        builtin_base = {"List": "list", "Tuple": "tuple", "Dict": "dict", "Set": "set"}.get(base, base)
        if builtin_base not in {"list", "tuple", "dict", "set"}:
            return "object"
        elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        rendered = [_safe_annotation(element) for element in elements]
        if any(annotation == "object" for annotation in rendered):
            return "object"
        return f"{builtin_base}[{', '.join(rendered)}]"
    return "object"


def _annotation_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return None


def _argument_from_add_argument(
    node: ast.Call, used_names: set[str], constants: dict[str, str | int | float | bool | None]
) -> ScriptArgumentPlan | None:
    raw_names = [_literal_string(arg) for arg in node.args]
    names = [name for name in raw_names if name]
    if not names:
        return None
    kwargs = {
        keyword.arg: keyword.value
        for keyword in node.keywords
        if keyword.arg is not None
    }
    if "nargs" in kwargs:
        return None

    positional = not any(name.startswith("-") for name in names)
    cli_name = None if positional else _choose_cli_name(names)
    dest = _literal_string(kwargs.get("dest")) if "dest" in kwargs else None
    param_source = dest or (names[0] if positional else cli_name)
    if param_source is None:
        return None
    param_name = _normalize_identifier(param_source.lstrip("-").replace("-", "_"))
    if not param_name or param_name in used_names:
        return None
    used_names.add(param_name)

    action = _literal_string(kwargs.get("action")) if "action" in kwargs else None
    if action in {"store_true", "store_false"}:
        annotation: Literal["str", "int", "float", "bool"] = "bool"
        default: str | int | float | bool | None = action == "store_false"
        normalized_action: Literal["store", "store_true", "store_false"] = action
    elif action not in {None, "store"}:
        return None
    else:
        annotation = _annotation_from_type(kwargs.get("type"))
        if annotation is None:
            return None
        default, _default_supported = _literal_argument_value(kwargs.get("default"), constants)
        normalized_action = "store"

    required = positional
    if "required" in kwargs:
        required_value, required_supported = _literal_argument_value(kwargs["required"], constants)
        if not required_supported or not isinstance(required_value, bool):
            return None
        required = positional or required_value
    return ScriptArgumentPlan(
        param_name=param_name,
        cli_name=cli_name,
        positional=positional,
        required=required,
        annotation=annotation,
        default=default,
        action=normalized_action,
    )


def _literal_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _literal_default(node: ast.AST | None) -> str | int | float | bool | None:
    value, supported = _literal_argument_value(node, {})
    if not supported or not isinstance(value, (str, int, float, bool)):
        return None
    return value


def _literal_argument_value(
    node: ast.AST | None, constants: dict[str, str | int | float | bool | None]
) -> tuple[str | int | float | bool | None, bool]:
    if node is None:
        return None, True
    if isinstance(node, ast.Name) and node.id in constants:
        return constants[node.id], True
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool, type(None))):
        return node.value, True
    number = _literal_number(node)
    if number is not None:
        return number, True
    return None, False


def _literal_module_constants(tree: ast.Module) -> dict[str, str | int | float | bool | None]:
    constants: dict[str, str | int | float | bool | None] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            value, supported = _literal_argument_value(node.value, constants)
            if supported:
                constants[node.targets[0].id] = value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            value, supported = _literal_argument_value(node.value, constants)
            if supported:
                constants[node.target.id] = value
    return constants


def _function_default(node: ast.AST | None) -> tuple[str | int | float | bool | None, bool]:
    if node is None:
        return None, True
    if isinstance(node, ast.Constant) and node.value is None:
        return None, True
    default = _literal_default(node)
    return default, default is not None


def _literal_number(node: ast.AST | None) -> int | float | None:
    if not isinstance(node, ast.UnaryOp) or not isinstance(node.op, (ast.USub, ast.UAdd)):
        return None
    if not isinstance(node.operand, ast.Constant) or not isinstance(node.operand.value, (int, float)):
        return None
    return -node.operand.value if isinstance(node.op, ast.USub) else node.operand.value


def _choose_cli_name(names: list[str]) -> str | None:
    long_names = [name for name in names if name.startswith("--")]
    if long_names:
        return max(long_names, key=len)
    option_names = [name for name in names if name.startswith("-")]
    return option_names[0] if option_names else None


def _annotation_from_type(node: ast.AST | None) -> Literal["str", "int", "float", "bool"] | None:
    if node is None:
        return "str"
    if isinstance(node, ast.Name) and node.id in {"str", "int", "float"}:
        return node.id  # type: ignore[return-value]
    return None


def _omission_reason(path: Path) -> str:
    if _has_argparse_api_shape(path):
        return "Argparse usage was detected, but the script shape is not safe to translate into a native method."
    return "No import-safe public functions or supported argparse API could be inferred."


def _has_argparse_api_shape(path: Path) -> bool:
    if path.suffix.lower() != ".py":
        return False
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return False
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"add_argument", "parse_args", "add_subparsers"}
        for node in ast.walk(tree)
    )


def _build_docstring(
    inventory: TextSkillInventory,
    script_methods: list[ScriptMethodPlan],
    resource_methods: list[ResourceMethodPlan],
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
        lines.extend(
            [
                "",
                "Use the guidance and bundled resource APIs when relevant.",
            ]
        )
    if resource_methods:
        lines.extend(["", "Bundled resource APIs:"])
        for resource in resource_methods:
            lines.append(
                f"- {resource.method_name}() -> {resource.return_annotation}: "
                f"returns `{resource.resource_path}` from package data."
            )
    adapted_guidance = _adapt_skill_guidance(inventory.body, script_methods, resource_methods)
    if adapted_guidance:
        lines.extend(["", "Guidance:", adapted_guidance])
    return "\n".join(lines)


def _adapt_skill_guidance(
    body: str,
    script_methods: list[ScriptMethodPlan],
    resource_methods: list[ResourceMethodPlan],
) -> str:
    """Render source guidance as LibrarySkill-native instructions.

    This preserves task-specific details while rewriting script/resource
    references to generated package API names. The raw body is not copied as a
    provenance block.
    """
    guidance = body.strip()
    if not guidance:
        return ""

    guidance = re.sub(r"\bUse this skill\b", "Use this LibrarySkill", guidance)
    guidance = re.sub(r"\bthis skill\b", "this LibrarySkill", guidance)
    guidance = re.sub(r"\bthe skill\b", "the LibrarySkill", guidance)
    guidance = re.sub(r"\bSKILL\.md\b", "this LibrarySkill guidance", guidance)
    guidance = re.sub(
        r"\b(run|execute|invoke)\s+(?:the\s+)?(?:script\s+)?`?scripts/",
        "call the corresponding LibrarySkill API `scripts/",
        guidance,
        flags=re.IGNORECASE,
    )

    replacements: list[tuple[str, str]] = []
    for method in script_methods:
        public_names = _script_public_api_names(method)
        if not public_names:
            continue
        replacement = " or ".join(f"`{name}()`" for name in public_names)
        replacements.append((method.script_path, replacement))
        replacements.append((Path(method.script_path).name, replacement))

    for resource in resource_methods:
        replacement = f"`{resource.method_name}()`"
        replacements.append((resource.resource_path, replacement))
        replacements.append((Path(resource.resource_path).name, replacement))

    for old, new in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        guidance = _replace_reference(guidance, old, new)

    return re.sub(r"`?scripts/[^`\s),.;:]+" + "`?", "the corresponding LibrarySkill API", guidance)


def _script_public_api_names(method: ScriptMethodPlan) -> list[str]:
    names: list[str] = []
    if method.api_method_name:
        names.append(method.api_method_name)
    names.extend(function.method_name for function in method.function_methods)
    return names


def _replace_reference(text: str, old: str, new: str) -> str:
    escaped = re.escape(old)
    return re.sub(rf"`?{escaped}`?", new, text)


def _public_method_guidance(script_methods: list[ScriptMethodPlan]) -> list[str]:
    lines: list[str] = []
    for method in script_methods:
        if method.api_method_name:
            parameters = ", ".join(_guidance_argument(argument) for argument in method.arguments)
            lines.append(f"- {method.api_method_name}({parameters}) -> str: returns captured text output.")
        for function in method.function_methods:
            parameters = ", ".join(_guidance_parameter(parameter) for parameter in function.parameters)
            lines.append(
                f"- {function.method_name}({parameters}) -> {function.return_annotation}: "
                "returns the Python value from the library implementation."
            )
    return lines


def _guidance_argument(argument: ScriptArgumentPlan) -> str:
    return _render_api_parameter(argument, required=argument.required and argument.action == "store")


def _guidance_parameter(parameter: FunctionParameterPlan) -> str:
    suffix = "" if parameter.required else f" = {parameter.default!r}"
    return f"{parameter.param_name}: {parameter.annotation}{suffix}"


def _method_return_guidance(function: ScriptFunctionPlan) -> str:
    if function.return_annotation == "object":
        return "Return the Python value from the library implementation."
    return f"Return `{function.return_annotation}` from the library implementation."


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _write(path: Path, content: str, package_dir: Path, written: list[str]) -> None:
    path.write_text(content, encoding="utf-8")
    written.append(path.relative_to(package_dir).as_posix())


def _render_pyproject(plan: ConversionPlan) -> str:
    return textwrap.dedent(f"""\
        [project]
        name = {_toml_string(plan.project_name)}
        version = "0.1.0"
        description = {_toml_string(plan.description)}
        dependencies = ["nooa"]

        [build-system]
        requires = ["setuptools>=68"]
        build-backend = "setuptools.build_meta"

        [tool.setuptools.packages.find]
        where = ["src"]

        [tool.setuptools.package-data]
        {plan.package_name} = ["resources/**"]

        [project.entry-points."nooa.skills"]
        {_toml_string(plan.registry_name)} = {_toml_string(f"{plan.package_name}:{plan.class_name}")}
    """)


def _render_readme(plan: ConversionPlan) -> str:
    return textwrap.dedent(f"""\
        # {plan.project_name}

        NOOA LibrarySkill package.

        Registry name: `{plan.registry_name}`

        Use the public Python APIs exposed by the Skill class.

        ## LibrarySkill-native guidance

        {textwrap.indent(plan.docstring, "        ")}
    """)


def _implementation_modules(plan: ConversionPlan) -> list[ScriptMethodPlan]:
    methods: list[ScriptMethodPlan] = []
    for method in plan.script_methods:
        execution = _native_argparse_execution(plan.source_dir / method.script_path)
        if method.function_methods or (execution is not None and execution.needs_module):
            methods.append(method)
    return methods


def _implementation_module_name(script_path: str) -> str:
    return f"_{_normalize_identifier(Path(script_path).with_suffix('').as_posix())}"


def _render_implementation_module(source_dir: Path, script_path: str) -> str:
    path = source_dir / script_path
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        raise ValueError(f"Cannot render implementation module for {path}: {exc}") from exc
    execution = _native_argparse_execution(path)
    if execution is not None and execution.needs_module and execution.implementation_body is not None:
        body = execution.implementation_body
        extra_used_names = _loaded_names(execution.statements)
    else:
        body = [
            node
            for node in tree.body
            if not (isinstance(node, ast.FunctionDef) and node.name == "main")
            and not _is_main_guard(node)
            and not _is_argparse_setup_statement(node)
            and _parse_args_target(node) is None
        ]
        extra_used_names = set()
    module = ast.Module(body=body, type_ignores=[])
    module = _rewrite_sibling_script_imports(module, source_dir, script_path)
    module = _prune_unused_imports(module, extra_used_names=extra_used_names)
    ast.fix_missing_locations(module)
    return ast.unparse(module) + "\n"


def _rewrite_sibling_script_imports(module: ast.Module, source_dir: Path, script_path: str) -> ast.Module:
    script_rel = Path(script_path)
    script_dir = script_rel.parent
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
            return [
                ast.ImportFrom(
                    module=module_name,
                    names=statement.names,
                    level=1,
                )
            ]
    return None


def _sibling_impl_module(source_dir: Path, script_dir: Path, import_name: str) -> str | None:
    if "." in import_name:
        return None
    sibling = script_dir / f"{import_name}.py"
    if not (source_dir / sibling).is_file():
        return None
    return _implementation_module_name(sibling.as_posix())


def _prune_unused_imports(module: ast.Module, *, extra_used_names: set[str] | None = None) -> ast.Module:
    used_names = {
        node.id
        for statement in module.body
        if not isinstance(statement, (ast.Import, ast.ImportFrom))
        for node in ast.walk(statement)
        if isinstance(node, ast.Name)
    }
    if extra_used_names:
        used_names.update(extra_used_names)
    body: list[ast.stmt] = []
    for statement in module.body:
        if isinstance(statement, ast.ImportFrom):
            if statement.module == "__future__":
                body.append(statement)
                continue
            names = [
                alias
                for alias in statement.names
                if (alias.asname or alias.name) in used_names or alias.name == "*"
            ]
            if names:
                statement.names = names
                body.append(statement)
        elif isinstance(statement, ast.Import):
            names = [
                alias
                for alias in statement.names
                if (alias.asname or alias.name.split(".", 1)[0]) in used_names
            ]
            if names:
                statement.names = names
                body.append(statement)
        else:
            body.append(statement)
    module.body = body
    return module


def _loaded_names(statements: list[ast.stmt]) -> set[str]:
    return {
        node.id
        for statement in statements
        for node in ast.walk(statement)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }


def _render_init(plan: ConversionPlan) -> str:
    # Generated skills expose package-style APIs backed by private modules.
    resource_methods = "\n".join(_render_resource_method(resource) for resource in plan.resource_methods)
    methods = "\n".join(
        rendered
        for rendered in (
            resource_methods,
            *(
                rendered
                for method in plan.script_methods
                for rendered in (
                    _render_api_method(plan, method),
                    *(_render_function_method(method, function) for function in method.function_methods),
                )
                if rendered
            ),
        )
        if rendered
    )
    if methods:
        methods = "\n" + methods
    docstring = textwrap.indent(_triple_quoted(plan.docstring), "    ")
    context_key = f"skill:{plan.registry_name}"
    attr_name = plan.registry_name.split(".")[-1].replace("-", "_")
    resource_methods_tuple = repr(
        tuple(
            (resource.method_name, resource.resource_path, resource.return_annotation, resource.size_bytes)
            for resource in plan.resource_methods
        )
    )
    template = textwrap.dedent(f'''\
        from __future__ import annotations

        from importlib import resources
        from pathlib import Path

        from nooa.agentdoc import hidden
        from nooa.skill import Skill


        class {plan.class_name}(Skill):
        __DOCSTRING__

            context_block = ({context_key!r}, "self.{attr_name}.format_guidance()")
            _RESOURCE_METHODS = {resource_methods_tuple}

            def _resource_root(self):
                return resources.files(__package__) / "{plan.resource_prefix}"

            def _list_resources(self) -> list[str]:
                """Return all bundled resource paths."""
                root = self._resource_root()
                return sorted(
                    path.relative_to(root).as_posix()
                    for path in Path(root).rglob("*")
                    if path.is_file()
                )

            def _read_resource(self, path: str) -> str:
                """Read a bundled resource as text."""
                return self._read_resource_bytes(path).decode()

            def _read_resource_bytes(self, path: str) -> bytes:
                """Read a bundled resource as bytes."""
                root = Path(self._resource_root()).resolve()
                resolved = (root / path).resolve()
                if not resolved.is_relative_to(root):
                    raise ValueError(f"Path {{path!r}} escapes package resources")
                if not resolved.is_file():
                    raise FileNotFoundError(path)
                return resolved.read_bytes()

            @hidden
            def format_guidance(self) -> str:
                """Return the LibrarySkill-native guidance and bundled resource API index."""
                resource_index = self._format_resource_index()
                if resource_index:
                    return type(self).__doc__ + "\\n\\nBundled resource APIs:\\n" + resource_index
                return type(self).__doc__ or ""

            def _format_resource_index(self) -> str:
                return "\\n".join(
                    f"- {{method}}() -> {{kind}}: {{path}} ({{size}} bytes)"
                    for method, path, kind, size in self._RESOURCE_METHODS
                )

        __METHODS__
    ''')
    return template.replace("__DOCSTRING__", docstring).replace("__METHODS__", methods)


def _render_resource_method(resource: ResourceMethodPlan) -> str:
    body = (
        f"return self._read_resource({resource.resource_path!r})"
        if resource.return_annotation == "str"
        else f"return self._read_resource_bytes({resource.resource_path!r})"
    )
    lines = [
        f"def {resource.method_name}(self) -> {resource.return_annotation}:",
        f"    {_triple_quoted(resource.docstring)}",
        f"    {body}",
        "",
    ]
    return textwrap.indent("\n".join(lines), "    ")


def _render_api_method(plan: ConversionPlan, method: ScriptMethodPlan) -> str:
    if not method.api_method_name or not method.arguments:
        return ""

    required_args = [
        argument for argument in method.arguments if argument.required and argument.action == "store"
    ]
    optional_args = [argument for argument in method.arguments if argument not in required_args]
    signature_parts = [_render_api_parameter(argument, required=True) for argument in required_args]
    signature_parts.extend(_render_api_parameter(argument, required=False) for argument in optional_args)
    signature = ", ".join(signature_parts)
    signature_suffix = f", {signature}" if signature else ""
    native_body = _render_native_argparse_body(plan.source_dir / method.script_path, method)
    if native_body is None:
        return ""

    lines: list[str] = [
        f"def {method.api_method_name}(self{signature_suffix}) -> str:",
        '    """Run the translated package implementation and return captured text output."""',
    ]
    lines.extend(f"    {line}" if line else "" for line in native_body)
    lines.append("")
    return textwrap.indent("\n".join(lines), "    ")


def _render_function_method(method: ScriptMethodPlan, function: ScriptFunctionPlan) -> str:
    signature_parts = [_render_function_parameter(parameter) for parameter in function.parameters]
    signature = ", ".join(signature_parts)
    if signature:
        signature = f", {signature}"
    call_args = ", ".join(f"{parameter.param_name}={parameter.param_name}" for parameter in function.parameters)
    docstring = function.docstring.strip() or _method_return_guidance(function)
    lines = [
        f"def {function.method_name}(self{signature}) -> {function.return_annotation}:",
        f"    {_triple_quoted(docstring)}",
        f"    from ._impl import {_implementation_module_name(method.script_path)} as module",
        f"    return module.{function.function_name}({call_args})",
        "",
    ]
    return textwrap.indent("\n".join(lines), "    ")


def _can_render_native_argparse_api(path: Path) -> bool:
    return _native_argparse_execution(path) is not None


def _render_native_argparse_body(path: Path, method: ScriptMethodPlan) -> list[str] | None:
    execution = _native_argparse_execution(path)
    if execution is None:
        return None
    args_name = execution.args_name
    statements = execution.statements
    param_names = {argument.param_name for argument in method.arguments}
    rewritten = _ArgparseMethodRewriter(
        args_name=args_name,
        param_names=param_names,
        prefix_globals=execution.needs_module,
        local_names=param_names | {args_name} | _assigned_names(statements),
    ).visit_statements(statements)
    if not rewritten:
        return None

    lines: list[str] = []
    if execution.needs_module:
        lines.append(f"from ._impl import {_implementation_module_name(method.script_path)} as module")
    else:
        lines.extend(execution.import_lines)
    lines.extend(
        [
            "import contextlib",
            "import io",
            "import os",
            "import types",
            f"{args_name} = types.SimpleNamespace({_namespace_kwargs(method.arguments)})",
            "buffer = io.StringIO()",
            "cwd = os.getcwd()",
            "try:",
            "    os.chdir(self._resource_root())",
            "    with contextlib.redirect_stdout(buffer):",
        ]
    )
    for statement in rewritten:
        rendered = ast.unparse(statement).splitlines()
        lines.extend(f"        {line}" if line else "" for line in rendered)
    lines.extend(
        [
            "finally:",
            "    os.chdir(cwd)",
        ]
    )
    lines.append("return buffer.getvalue().rstrip('\\n')")
    return lines


def _namespace_kwargs(arguments: list[ScriptArgumentPlan]) -> str:
    return ", ".join(f"{argument.param_name}={argument.param_name}" for argument in arguments)


def _native_argparse_execution(path: Path) -> NativeArgparseExecution | None:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    if _has_sibling_script_imports(path, tree):
        return None

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            if not _can_render_implementation_module(tree):
                return None
            parsed = _body_after_parse_args(node.body)
            if parsed is not None:
                args_name, statements, previous_statements = parsed
                implementation_body = _main_implementation_body(tree, previous_statements)
                if implementation_body is None:
                    return None
                return NativeArgparseExecution(
                    args_name=args_name,
                    statements=statements,
                    import_lines=[],
                    needs_module=bool(implementation_body),
                    implementation_body=implementation_body or None,
                )

    return _top_level_body_after_parse_args(tree.body)


def _has_sibling_script_imports(path: Path, tree: ast.Module) -> bool:
    script_dir = path.parent
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level = alias.name.split(".", 1)[0]
                if _sibling_module_exists(script_dir, top_level):
                    return True
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            top_level = node.module.split(".", 1)[0]
            if _sibling_module_exists(script_dir, top_level):
                return True
    return False


def _sibling_module_exists(script_dir: Path, module_name: str) -> bool:
    return (script_dir / f"{module_name}.py").exists() or (script_dir / module_name / "__init__.py").exists()


def _can_render_implementation_module(tree: ast.Module) -> bool:
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    for node in tree.body:
        if _is_module_docstring(node) or _is_main_guard(node):
            continue
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.Assign) and _safe_module_assignment_value(node.value, functions):
            continue
        if isinstance(node, ast.AnnAssign) and _safe_module_assignment_value(node.value, functions):
            continue
        return False
    return True


def _main_implementation_body(tree: ast.Module, previous_statements: list[ast.stmt]) -> list[ast.stmt] | None:
    local_body = [
        statement
        for statement in previous_statements
        if not _is_module_docstring(statement) and not _is_argparse_setup_statement(statement)
    ]
    if not all(_is_safe_implementation_statement(statement) for statement in local_body):
        return None
    module_body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.FunctionDef) and node.name == "main")
        and not _is_main_guard(node)
        and not _is_argparse_setup_statement(node)
        and _parse_args_target(node) is None
    ]
    return module_body + local_body


def _top_level_body_after_parse_args(statements: list[ast.stmt]) -> NativeArgparseExecution | None:
    for index, statement in enumerate(statements):
        args_name = _parse_args_target(statement)
        if args_name is not None:
            previous_statements = [
                previous for previous in statements[:index] if not _is_module_docstring(previous)
            ]
            import_lines = [ast.unparse(node) for node in previous_statements if isinstance(node, (ast.Import, ast.ImportFrom))]
            implementation_body = [
                previous
                for previous in previous_statements
                if not _is_argparse_setup_statement(previous)
            ]
            needs_module = any(
                not isinstance(previous, (ast.Import, ast.ImportFrom))
                and not _is_argparse_setup_statement(previous)
                for previous in previous_statements
            )
            if needs_module and not all(_is_safe_implementation_statement(previous) for previous in implementation_body):
                return None
            if not needs_module and not all(
                isinstance(previous, (ast.Import, ast.ImportFrom)) or _is_argparse_setup_statement(previous)
                for previous in previous_statements
            ):
                return None
            tail = statements[index + 1 :]
            if not tail:
                return None
            return NativeArgparseExecution(
                args_name=args_name,
                statements=tail,
                import_lines=import_lines,
                needs_module=needs_module,
                implementation_body=implementation_body if needs_module else None,
            )
    return None


def _is_safe_implementation_statement(statement: ast.stmt) -> bool:
    if isinstance(statement, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
        return True
    if isinstance(statement, ast.Assign):
        return _literal_container(statement.value)
    if isinstance(statement, ast.AnnAssign):
        return _literal_container(statement.value)
    return False


def _safe_module_assignment_value(value: ast.AST | None, functions: dict[str, ast.FunctionDef]) -> bool:
    if _literal_container(value):
        return True
    if not isinstance(value, ast.Call) or value.args or value.keywords:
        return False
    if not isinstance(value.func, ast.Name):
        return False
    function = functions.get(value.func.id)
    return function is not None and _is_safe_path_resolver_function(function)


def _is_safe_path_resolver_function(function: ast.FunctionDef) -> bool:
    if function.args.args or function.args.posonlyargs or function.args.kwonlyargs:
        return False
    for node in ast.walk(function):
        if isinstance(
            node,
            (
                ast.AsyncFunctionDef,
                ast.Await,
                ast.Delete,
                ast.For,
                ast.Global,
                ast.Import,
                ast.ImportFrom,
                ast.Lambda,
                ast.Nonlocal,
                ast.Raise,
                ast.Try,
                ast.While,
                ast.With,
                ast.Yield,
                ast.YieldFrom,
            ),
        ):
            return False
        if isinstance(node, ast.Call) and not _is_safe_path_resolver_call(node):
            return False
    return True


def _is_safe_path_resolver_call(node: ast.Call) -> bool:
    if isinstance(node.func, ast.Name) and node.func.id == "Path":
        return True
    return isinstance(node.func, ast.Attribute) and node.func.attr in {"exists", "resolve"}


def _body_after_parse_args(statements: list[ast.stmt]) -> tuple[str, list[ast.stmt], list[ast.stmt]] | None:
    for index, statement in enumerate(statements):
        args_name = _parse_args_target(statement)
        if args_name is not None:
            tail = statements[index + 1 :]
            return (args_name, tail, statements[:index]) if tail else None
    return None


def _parse_args_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    if not isinstance(target, ast.Name):
        return None
    value = statement.value
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "parse_args"
    ):
        return target.id
    return None


def _is_argparse_setup_statement(statement: ast.stmt) -> bool:
    if isinstance(statement, ast.Assign):
        return _is_argparse_parser_assignment(statement)
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return False
    call = statement.value
    return isinstance(call.func, ast.Attribute) and call.func.attr == "add_argument"


def _is_argparse_parser_assignment(statement: ast.Assign) -> bool:
    if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
        return False
    value = statement.value
    return (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and value.func.attr == "ArgumentParser"
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id == "argparse"
    )


def _assigned_names(statements: list[ast.stmt]) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.Module(body=statements, type_ignores=[])):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            names.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


class _ArgparseMethodRewriter(ast.NodeTransformer):
    def __init__(
        self,
        *,
        args_name: str,
        param_names: set[str],
        prefix_globals: bool,
        local_names: set[str],
    ) -> None:
        self.args_name = args_name
        self.param_names = param_names
        self.prefix_globals = prefix_globals
        self.local_names = local_names | {"module"}
        self.builtin_names = set(dir(builtins))

    def visit_statements(self, statements: list[ast.stmt]) -> list[ast.stmt]:
        rewritten = [self.visit(statement) for statement in statements]
        ast.fix_missing_locations(ast.Module(body=rewritten, type_ignores=[]))
        return rewritten

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if isinstance(node.value, ast.Name) and node.value.id == self.args_name and node.attr in self.param_names:
            return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
        node = self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if (
            self.prefix_globals
            and isinstance(node.ctx, ast.Load)
            and node.id not in self.local_names
            and node.id not in self.builtin_names
        ):
            return ast.copy_location(
                ast.Attribute(value=ast.Name(id="module", ctx=ast.Load()), attr=node.id, ctx=node.ctx),
                node,
            )
        return node


def _render_function_parameter(parameter: FunctionParameterPlan) -> str:
    if parameter.required:
        return f"{parameter.param_name}: {parameter.annotation}"
    default = repr(parameter.default)
    return f"{parameter.param_name}: {parameter.annotation} = {default}"


def _render_api_parameter(argument: ScriptArgumentPlan, *, required: bool) -> str:
    annotation = argument.annotation
    if argument.action in {"store_true", "store_false"}:
        default = "False" if argument.action == "store_true" else "True"
        return f"{argument.param_name}: bool = {default}"
    if required:
        return f"{argument.param_name}: {annotation}"
    default = repr(argument.default) if argument.default is not None else "None"
    return f"{argument.param_name}: {annotation} | None = {default}"


def _render_argument_append(argument: ScriptArgumentPlan) -> list[str]:
    name = argument.param_name
    if argument.positional:
        return [f"    args.append(str({name}))"]
    if argument.cli_name is None:
        return []
    if argument.action == "store_true":
        return [
            f"    if {name}:",
            f"        args.append({argument.cli_name!r})",
        ]
    if argument.action == "store_false":
        return [
            f"    if not {name}:",
            f"        args.append({argument.cli_name!r})",
        ]
    if argument.required:
        return [f"    args.extend([{argument.cli_name!r}, str({name})])"]
    return [
        f"    if {name} is not None:",
        f"        args.extend([{argument.cli_name!r}, str({name})])",
    ]


def _render_tests(plan: ConversionPlan) -> str:
    attr_name = plan.registry_name.split(".")[-1].replace("-", "_")
    lines = [
        "from pathlib import Path",
        "",
        "from nooa import Agent",
        "from nooa.agentdoc import doc",
        "from nooa.context_blocks import DynamicContext",
        "from nooa.skill_registry import SkillRegistry",
        f"from {plan.package_name} import {plan.class_name}",
        "",
        "",
        "def test_skill_instantiates_and_lists_resources():",
        f"    skill = {plan.class_name}()",
        "    visible_doc = doc(skill)",
        "    assert 'list_resources' not in visible_doc",
        "    assert 'read_resource' not in visible_doc",
        "    assert 'LibrarySkill-native guidance' in visible_doc",
        "    assert 'run_resource_script' not in visible_doc",
        "    assert isinstance(skill.format_guidance(), str)",
    ]
    for resource in plan.resource_methods:
        lines.extend(
            [
                f"    assert hasattr(skill, {resource.method_name!r})",
                f"    assert {resource.method_name!r} in visible_doc",
                f"    assert {resource.resource_path!r} in visible_doc",
            ]
        )
        if resource.return_annotation == "str":
            lines.append(f"    assert isinstance(skill.{resource.method_name}(), str)")
        else:
            lines.append(f"    assert isinstance(skill.{resource.method_name}(), bytes)")
    if plan.script_methods:
        for method in plan.script_methods:
            lines.extend(_test_assertions(method))
    else:
        lines.append("    assert isinstance(skill._list_resources(), list)")
    lines.extend(
        [
            "",
            "",
            "def test_skill_registry_loads_package():",
            "    class Agent:",
            "        pass",
            "",
            "    package_dir = Path(__file__).resolve().parents[1]",
            "    registry = SkillRegistry(Agent())",
            "    try:",
            "        registry.discover_libs(package_dir.parent)",
            f"        assert {plan.registry_name!r} in registry.loaded()",
            "    finally:",
            "        registry.close()",
            "",
            "",
            "def test_skill_registry_activation_registers_context_block():",
            "    package_dir = Path(__file__).resolve().parents[1]",
            "    agent = Agent(llm=object())",
            "    registry = SkillRegistry(agent)",
            "    try:",
            "        registry.discover_libs(package_dir.parent)",
            f"        registry.activate([{plan.registry_name!r}])",
            f"        context_key = {f'skill:{plan.registry_name}'!r}",
            "        assert context_key in agent.context_manager",
            "        raw_block = dict(agent.context_manager._raw_items())[context_key]",
            "        assert isinstance(raw_block, DynamicContext)",
            f"        assert raw_block.expr == {f'self.{attr_name}.format_guidance()'!r}",
            "    finally:",
            "        registry.close()",
        ]
    )
    return "\n".join(lines) + "\n"


def _test_assertions(method: ScriptMethodPlan) -> list[str]:
    assertions = [
        f"    assert {method.script_path!r} not in skill._list_resources()",
        f"    assert {method.method_name!r} not in visible_doc",
    ]
    if method.api_method_name:
        assertions.extend(
            [
                f"    assert hasattr(skill, {method.api_method_name!r})",
                f"    assert {method.api_method_name!r} in visible_doc",
            ]
        )
    for function in method.function_methods:
        assertions.extend(
            [
                f"    assert hasattr(skill, {function.method_name!r})",
                f"    assert {function.method_name!r} in visible_doc",
            ]
        )
    return assertions


def _triple_quoted(value: str) -> str:
    return repr(value)


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
    """Run registry validation in a clean thread so close() can always clean up."""
    result: dict[str, object] = {}
    error: list[BaseException] = []

    def worker() -> None:
        try:
            result["value"] = _validate_registry_load_sync(package_path, registry_name)
        except BaseException as exc:  # pragma: no cover - defensive thread bridge
            error.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    if error:
        raise error[0]
    return result["value"]  # type: ignore[return-value]


def _validate_registry_load_sync(package_path: Path, registry_name: str) -> tuple[bool, list[str]]:
    from nooa.skill_registry import SkillRegistry

    class _ValidationAgent:
        pass

    registry = SkillRegistry(_ValidationAgent())
    try:
        registry.discover_libs(package_path.parent)
        loaded_names = registry.loaded()
        return registry_name in loaded_names, loaded_names
    finally:
        registry.close()
