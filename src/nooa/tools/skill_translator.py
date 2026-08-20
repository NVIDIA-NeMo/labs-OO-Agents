# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate SKILL.md TextSkills into package-backed nooa Skill libraries."""

from __future__ import annotations

import ast
import hashlib
import json
import keyword
import re
import shutil
import textwrap
import threading
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from nooa.skill import Skill, _find_skill_md, _parse_frontmatter


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
    """One inferred command-line argument for a generated script API."""

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
    resource_prefix: str = "resources"


class PackageTranslationResult(BaseModel):
    """Result of writing a package skill to disk."""

    package_dir: Path
    package_name: str
    registry_name: str
    class_name: str
    files_written: list[str]


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


class TextSkillTranslator(Skill):
    """Translate traditional SKILL.md TextSkills into package-backed Skill libraries.

    The translator preserves TextSkill content first, then opportunistically
    exposes package-style APIs around bundled resources and scripts. The
    generated package should therefore keep TextSkill behavior available even
    when no richer API can be inferred.

    Typical flow:

        inventory = self.skill_translator.inspect_text_skill("skills/frontend")
        plan = self.skill_translator.plan_conversion(inventory)
        result = self.skill_translator.write_package(plan, "libs")
        report = self.skill_translator.validate_package(result.package_dir)

    The generated package contains:
    - pyproject.toml with a nooa.skills entry point
    - src/<package_name>/__init__.py exporting a Skill subclass
    - copied resources under src/<package_name>/resources/
    - baseline pytest tests under tests/

    Conversion is intentionally conservative. Script files are copied unchanged
    and wrapped with async methods instead of being rewritten.
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
        docstring = _build_docstring(inventory)

        used_names: set[str] = set()
        used_api_names: set[str] = set()
        script_methods: list[ScriptMethodPlan] = []
        for file in inventory.scripts:
            interpreter = _default_interpreter(file)
            if interpreter is None and not file.executable:
                continue
            script_path = inventory.source_dir / file.path

            # Every supported script gets a raw runner. The inferred APIs below
            # are additive ergonomics: a CLI-shaped method for argparse scripts
            # and direct methods for import-safe top-level Python functions.
            arguments = _infer_script_arguments(script_path)
            api_method_name = _api_method_name(file.path, used_api_names) if arguments else None
            function_methods = _infer_script_functions(script_path, used_api_names)
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

        return ConversionPlan(
            source_dir=inventory.source_dir,
            package_name=package,
            project_name=project,
            registry_name=registry,
            class_name=cls_name,
            description=inventory.description,
            docstring=docstring,
            script_methods=script_methods,
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
        _write(package_src / "__init__.py", _render_init(plan), package_dir, written)
        _write(
            tests_dir / f"test_{plan.package_name}.py",
            _render_tests(plan),
            package_dir,
            written,
        )

        for source_file in _iter_skill_files(plan.source_dir):
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


def _api_method_name(script_path: str, used_names: set[str]) -> str:
    name = _normalize_identifier(Path(script_path).stem)
    if name in {"run_resource_script", "read_resource", "list_resources", "_resource_root"}:
        name = f"{name}_api"
    base = name
    index = 2
    while name in used_names:
        name = f"{base}_{index}"
        index += 1
    used_names.add(name)
    return name


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
    """Infer a named package method from simple argparse declarations."""
    if path.suffix.lower() != ".py":
        return []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []

    arguments: list[ScriptArgumentPlan] = []
    used_names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        argument = _argument_from_add_argument(node, used_names)
        if argument is not None:
            arguments.append(argument)
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
    if name in {"run_resource_script", "read_resource", "list_resources", "_resource_root"}:
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
    node: ast.Call, used_names: set[str]
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
        default = _literal_default(kwargs.get("default"))
        normalized_action = "store"

    required = positional or bool(_literal_default(kwargs.get("required")))
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
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool)):
        return node.value
    number = _literal_number(node)
    if number is not None:
        return number
    return None


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


def _annotation_from_type(node: ast.AST | None) -> Literal["str", "int", "float", "bool"]:
    if isinstance(node, ast.Name) and node.id in {"str", "int", "float"}:
        return node.id  # type: ignore[return-value]
    return "str"


def _build_docstring(inventory: TextSkillInventory) -> str:
    title = inventory.description.strip() or inventory.skill_name
    body = inventory.body.strip()
    if body:
        return f"{title}\n\nOriginal SKILL.md guidance:\n\n{body}"
    return title


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

        Package skill translated from `{plan.source_dir}`.

        Registry name: `{plan.registry_name}`

        The original TextSkill files are bundled under package resources.
    """)


def _render_init(plan: ConversionPlan) -> str:
    # Generated skills have three API layers: resource access, raw script
    # runners that preserve TextSkill behavior, and any inferred ergonomic
    # methods planned from argparse declarations or import-safe functions.
    methods = "\n".join(
        rendered
        for method in plan.script_methods
        for rendered in (
            _render_script_method(method),
            _render_api_method(method),
            *(_render_function_method(method, function) for function in method.function_methods),
        )
        if rendered
    )
    if methods:
        methods = "\n" + methods
    docstring = textwrap.indent(_triple_quoted(plan.docstring), "    ")
    template = textwrap.dedent(f'''\
        from __future__ import annotations

        import importlib.util
        import sys
        from importlib import resources
        from pathlib import Path

        from nooa.skill import Skill
        from nooa.tools._bash_session import BashSession


        class {plan.class_name}(Skill):
        __DOCSTRING__

            def _resource_root(self):
                return resources.files(__package__) / "{plan.resource_prefix}"

            def _load_resource_module(self, path: str):
                root = Path(self._resource_root()).resolve()
                module_path = (root / path).resolve()
                if not module_path.is_relative_to(root):
                    raise ValueError(f"Path {{path!r}} escapes package resources")
                if not module_path.is_file():
                    raise FileNotFoundError(path)
                cache = getattr(self, "_module_cache", None)
                if cache is None:
                    cache = {{}}
                    self._module_cache = cache
                if path in cache:
                    return cache[path]
                module_name = f"{{__package__}}._resource_{{abs(hash(path))}}"
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Cannot import resource module {{path!r}}")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                cache[path] = module
                return module

            def list_resources(self) -> list[str]:
                """Return all bundled resource paths."""
                root = self._resource_root()
                return sorted(
                    path.relative_to(root).as_posix()
                    for path in Path(root).rglob("*")
                    if path.is_file()
                )

            def read_resource(self, path: str) -> str:
                """Read a bundled resource as text."""
                root = Path(self._resource_root()).resolve()
                resolved = (root / path).resolve()
                if not resolved.is_relative_to(root):
                    raise ValueError(f"Path {{path!r}} escapes package resources")
                if not resolved.is_file():
                    raise FileNotFoundError(path)
                return resolved.read_text()

            async def run_resource_script(
                self,
                path: str,
                *args: str,
                interpreter: str | None = None,
                timeout: float = 30.0,
            ) -> str:
                """Run a bundled script resource."""
                root = Path(self._resource_root()).resolve()
                script = (root / path).resolve()
                if not script.is_relative_to(root):
                    raise ValueError(f"Path {{path!r}} escapes package resources")
                if not script.is_file():
                    raise FileNotFoundError(path)
                quoted_script = _quote(str(script))
                quoted_args = " ".join(_quote(arg) for arg in args)
                if interpreter:
                    command = f"{{interpreter}} {{quoted_script}} {{quoted_args}}".strip()
                else:
                    command = f"{{quoted_script}} {{quoted_args}}".strip()
                session = BashSession(cwd=root)
                try:
                    await session.start()
                    stdout, stderr, exit_code = await session.run(command, timeout=timeout)
                finally:
                    await session.close()
                output = stdout
                if stderr:
                    output += f"\\n[stderr]\\n{{stderr}}"
                if exit_code != 0:
                    output += f"\\n[exit code: {{exit_code}}]"
                return output
        __METHODS__


        def _quote(value: str) -> str:
            import shlex

            return shlex.quote(value)
    ''')
    return template.replace("__DOCSTRING__", docstring).replace("__METHODS__", methods)


def _render_script_method(method: ScriptMethodPlan) -> str:
    if method.interpreter is None:
        interpreter_arg = "None"
    else:
        interpreter_arg = method.interpreter
    return textwrap.indent(
        textwrap.dedent(f'''\
            async def {method.method_name}(self, *args: str, timeout: float = 30.0) -> str:
                """Run bundled script `{method.script_path}`."""
                return await self.run_resource_script(
                    {method.script_path!r},
                    *args,
                    interpreter={interpreter_arg},
                    timeout=timeout,
                )
        '''),
        "    ",
    )


def _render_api_method(method: ScriptMethodPlan) -> str:
    if not method.api_method_name or not method.arguments:
        return ""

    timeout_name = "timeout"
    arg_names = {argument.param_name for argument in method.arguments}
    if timeout_name in arg_names:
        timeout_name = "execution_timeout"

    required_args = [
        argument for argument in method.arguments if argument.required and argument.action == "store"
    ]
    optional_args = [argument for argument in method.arguments if argument not in required_args]
    signature_parts = [_render_api_parameter(argument, required=True) for argument in required_args]
    signature_parts.extend(_render_api_parameter(argument, required=False) for argument in optional_args)
    signature_parts.append(f"{timeout_name}: float = 30.0")
    signature = ", ".join(signature_parts)

    lines: list[str] = [
        f"async def {method.api_method_name}(self, {signature}) -> str:",
        f'    """Run `{method.script_path}` with named command-line arguments."""',
        "    args: list[str] = []",
    ]
    for argument in method.arguments:
        lines.extend(_render_argument_append(argument))
    lines.extend(
        [
            f"    return await self.{method.method_name}(*args, timeout={timeout_name})",
            "",
        ]
    )
    return textwrap.indent("\n".join(lines), "    ")


def _render_function_method(method: ScriptMethodPlan, function: ScriptFunctionPlan) -> str:
    signature_parts = [_render_function_parameter(parameter) for parameter in function.parameters]
    signature = ", ".join(signature_parts)
    if signature:
        signature = f", {signature}"
    call_args = ", ".join(f"{parameter.param_name}={parameter.param_name}" for parameter in function.parameters)
    docstring = function.docstring.strip() or f"Call `{function.function_name}` from `{method.script_path}`."
    lines = [
        f"def {function.method_name}(self{signature}) -> {function.return_annotation}:",
        f"    {_triple_quoted(docstring)}",
        f"    module = self._load_resource_module({method.script_path!r})",
        f"    return module.{function.function_name}({call_args})",
        "",
    ]
    return textwrap.indent("\n".join(lines), "    ")


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
    lines = [
        f"from {plan.package_name} import {plan.class_name}",
        "",
        "",
        "def test_skill_instantiates_and_lists_resources():",
        f"    skill = {plan.class_name}()",
    ]
    if plan.script_methods:
        for method in plan.script_methods:
            lines.extend(_test_assertions(method))
    else:
        lines.append("    assert isinstance(skill.list_resources(), list)")
    return "\n".join(lines) + "\n"


def _test_assertions(method: ScriptMethodPlan) -> list[str]:
    assertions = [f"    assert {method.script_path!r} in skill.list_resources()"]
    if method.api_method_name:
        assertions.append(f"    assert hasattr(skill, {method.api_method_name!r})")
    for function in method.function_methods:
        assertions.append(f"    assert hasattr(skill, {function.method_name!r})")
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
