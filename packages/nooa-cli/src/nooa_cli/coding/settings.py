# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workspace-aware settings shared by interactive coding-agent hosts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from nooa.layered_config import load_layered_yaml

_SETTINGS_FILENAME = "settings.yaml"
_SETTINGS_ENV_VAR = "NEMO_OO_SETTINGS"
_WORKSPACE_SKILL_DIRS = (
    Path(".agents/skills"),
    Path(".cursor/skills"),
    Path(".claude/skills"),
    Path(".claude/commands"),
)
_USER_SKILL_DIRS = (
    Path(".agents/skills"),
    Path(".claude/skills"),
    Path(".claude/commands"),
)


def load_coding_skills_dirs(
    workspace: str | Path,
    *,
    explicit: Iterable[str | Path] = (),
) -> list[Path]:
    """Return existing skill roots for one coding-agent workspace.

    Reads ``coding.additional_skills_dirs``. Relative configured paths are
    resolved against the active workspace, not the ACP server process's
    checkout.
    """
    root = Path(workspace).expanduser().resolve()
    settings = load_layered_yaml(
        _SETTINGS_FILENAME,
        _SETTINGS_ENV_VAR,
        project_dir=root / ".nooa",
    )

    configured = _setting_paths(settings, "coding")

    candidates: list[Path] = []
    candidates.extend(_resolve_configured(path, root) for path in explicit)
    candidates.extend(_resolve_configured(path, root) for path in configured)
    candidates.extend(root / path for path in _WORKSPACE_SKILL_DIRS)
    candidates.extend(Path.home() / path for path in _USER_SKILL_DIRS)

    result: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        result.append(resolved)
    return result


def _setting_paths(settings: Mapping[str, Any], section: str) -> list[str | Path]:
    value = settings.get(section)
    if not isinstance(value, Mapping):
        return []
    paths = value.get("additional_skills_dirs")
    if isinstance(paths, (str, Path)):
        return [paths]
    if not isinstance(paths, list):
        return []
    return [path for path in paths if isinstance(path, (str, Path))]


def _resolve_configured(path: str | Path, workspace: Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else workspace / candidate


__all__ = ["load_coding_skills_dirs"]
