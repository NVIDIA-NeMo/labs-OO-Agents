# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Settings snap to the existing config objects (AGENTS.md, "Configuration").

An agent behaviour is configured through the config object that owns the concern
(``RenderConfig``, ``TruncationConfig``, ``SummarizationConfig``, ``HttpConfig``,
``RetryConfig``, ``ReasoningConfig``), and the surfaces that construct agents pass
those objects through. This check walks every importable ``nooa.Agent`` subclass's
``__init__`` and Atom's ``SessionOptions`` / ``AtomOptions`` and fails when a
parameter or field carries the NAME of a config-object field without carrying a
config object: that is a setting mirrored as a loose kwarg or settings key.

A collision that is deliberate (same word, different concept) goes into
``ALLOWED`` with a reason; a new setting that belongs on a config object does not.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import warnings
from pathlib import Path

import pytest

from nooa import Agent
from nooa.config.truncation_config import TruncationConfig
from nooa.context_blocks.render_config import RenderConfig
from nooa.interactive import SummarizationConfig
from nooa.unifiedllm.http_config import HttpConfig
from nooa.unifiedllm.reasoning import ReasoningConfig
from nooa.unifiedllm.retry_config import RetryConfig

REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIG_OBJECTS: tuple[type, ...] = (
    RenderConfig,
    TruncationConfig,
    SummarizationConfig,
    HttpConfig,
    RetryConfig,
    ReasoningConfig,
)

# (owner qualified name, parameter or field name) -> why the shared name is not a mirrored setting.
ALLOWED: dict[tuple[str, str], str] = {
    (
        "nooa_atom.agent.agent.AtomAgent",
        "policy",
    ): "Atom's TurnPolicy (which turn handler runs); unrelated to SummarizationConfig.policy",
}

# Modules whose import has side effects or optional dependencies; skipped, not failed.
SKIP_MODULE_PARTS = (".tests", ".__main__", ".viewer")


def _package_roots() -> list[str]:
    roots = ["nooa"]
    for package_dir in sorted((REPO_ROOT / "packages").iterdir()):
        src = package_dir / "src"
        if src.is_dir():
            roots.extend(p.name for p in src.iterdir() if (p / "__init__.py").exists())
    return roots


def _import_everything() -> list[str]:
    """Import every module under the package roots so ``Agent.__subclasses__`` is complete.

    Walks packages by hand instead of ``pkgutil.walk_packages``: that helper imports a
    package before the caller can skip it, and ``nooa.viewer`` raises at import time
    when no frontend build is present. A package matching ``SKIP_MODULE_PARTS`` is
    neither imported nor traversed.
    """
    skipped: list[str] = []

    def import_module(name: str) -> object | None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return importlib.import_module(name)
        except BaseException as exc:  # noqa: BLE001 - optional deps, platform guards
            skipped.append(f"{name}: {type(exc).__name__}: {str(exc)[:80]}")
            return None

    def walk(package: object, prefix: str) -> None:
        for info in pkgutil.iter_modules(getattr(package, "__path__", []), prefix=prefix):
            if any(part in info.name for part in SKIP_MODULE_PARTS):
                continue
            module = import_module(info.name)
            if module is not None and info.ispkg:
                walk(module, f"{info.name}.")

    for root in _package_roots():
        package = import_module(root)
        if package is not None:
            walk(package, f"{root}.")
    return skipped


def _all_subclasses(cls: type) -> set[type]:
    seen: set[type] = set()
    stack = [cls]
    while stack:
        for sub in stack.pop().__subclasses__():
            if sub not in seen:
                seen.add(sub)
                stack.append(sub)
    return seen


def _config_field_owners() -> dict[str, list[str]]:
    owners: dict[str, list[str]] = {}
    for config in CONFIG_OBJECTS:
        for field in config.model_fields:
            owners.setdefault(field, []).append(config.__name__)
    return owners


def _carries_config_object(annotation: object) -> bool:
    text = str(annotation)
    return any(config.__name__ in text for config in CONFIG_OBJECTS)


OPTION_MODELS: tuple[tuple[str, str], ...] = (
    ("nooa_atom.session.options", "SessionOptions"),
    ("nooa_atom.workspace.options", "AtomOptions"),
)


def _option_models(skipped: list[str]) -> list[tuple[str, type]]:
    """Atom's option models as (owner name, class); an absent package is recorded, not failed."""
    models: list[tuple[str, type]] = []
    for module_name, class_name in OPTION_MODELS:
        try:
            model = getattr(importlib.import_module(module_name), class_name)
        except Exception as exc:  # noqa: BLE001 - the Atom package may be absent
            skipped.append(f"{module_name}.{class_name}: {type(exc).__name__}: {exc}")
            continue
        models.append((f"{module_name}.{class_name}", model))
    return models


def _violations() -> tuple[list[str], list[str]]:
    skipped = _import_everything()
    owners = _config_field_owners()
    findings: list[str] = []

    def check(owner: str, name: str, annotation: object) -> None:
        if name not in owners or _carries_config_object(annotation):
            return
        if (owner, name) in ALLOWED:
            return
        findings.append(
            f"{owner}: `{name}` ({annotation}) mirrors {', '.join(owners[name])}.{name}; "
            "accept or pass the config object instead, or add it to ALLOWED with a reason"
        )

    for cls in sorted(_all_subclasses(Agent), key=lambda c: f"{c.__module__}.{c.__qualname__}"):
        if (
            any(part in cls.__module__ for part in SKIP_MODULE_PARTS)
            or cls.__module__ == "__main__"
        ):
            continue
        try:
            signature = inspect.signature(cls.__init__)
        except (TypeError, ValueError):
            continue
        for name, parameter in signature.parameters.items():
            if name == "self" or parameter.kind in (
                parameter.VAR_POSITIONAL,
                parameter.VAR_KEYWORD,
            ):
                continue
            check(f"{cls.__module__}.{cls.__qualname__}", name, parameter.annotation)

    for owner, model in _option_models(skipped):
        for name, field in model.model_fields.items():
            check(owner, name, field.annotation)

    return findings, skipped


def test_settings_snap_to_config_objects() -> None:
    findings, skipped = _violations()
    if skipped:
        print("modules skipped (not imported):", *skipped, sep="\n  ")
    assert not findings, "\n".join(findings)


def test_allow_list_entries_are_still_needed() -> None:
    """A stale ALLOWED entry hides nothing today but would hide a future regression."""
    skipped = _import_everything()
    owners = _config_field_owners()
    present: set[tuple[str, str]] = set()
    for cls in _all_subclasses(Agent):
        try:
            signature = inspect.signature(cls.__init__)
        except (TypeError, ValueError):
            continue
        owner = f"{cls.__module__}.{cls.__qualname__}"
        present.update((owner, name) for name in signature.parameters if name in owners)
    for owner, model in _option_models(skipped):
        present.update((owner, name) for name in model.model_fields if name in owners)
    stale = sorted(key for key in ALLOWED if key not in present)
    assert not stale, f"ALLOWED entries no longer match any parameter or field: {stale}"


@pytest.mark.parametrize("config", CONFIG_OBJECTS)
def test_config_objects_have_fields(config: type) -> None:
    assert config.model_fields, f"{config.__name__} declares no fields; the check would be vacuous"
