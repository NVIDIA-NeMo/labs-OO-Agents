# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workspace-aware settings shared by interactive coding-agent hosts."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from nooa.layered_config import load_layered_yaml

_SETTINGS_FILENAME = "settings.yaml"
_SETTINGS_ENV_VAR = "NEMO_OO_SETTINGS"
_LEGACY_CONFIG_FILENAME = "config.toml"
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

logger = logging.getLogger(__name__)


def load_coding_skills_dirs(
    workspace: str | Path,
    *,
    explicit: Iterable[str | Path] = (),
) -> list[Path]:
    """Return existing skill roots for one coding-agent workspace.

    The shared ``coding.additional_skills_dirs`` setting is preferred for new
    configuration. ``tui.additional_skills_dirs`` remains supported while the
    TUI migrates to the shared section. Relative configured paths are resolved
    against the active workspace, not the ACP server process's checkout.
    """
    root = Path(workspace).expanduser().resolve()
    settings = load_layered_yaml(
        _SETTINGS_FILENAME,
        _SETTINGS_ENV_VAR,
        project_dir=root / ".nooa",
    )

    configured: list[str | Path] = []
    configured.extend(_setting_paths(settings, "coding"))
    configured.extend(_setting_paths(settings, "tui"))
    # A workspace's old config.toml is still a workspace layer. Do not let an
    # unrelated user-level settings.yaml silently suppress it. A modern
    # workspace settings file supersedes the legacy file, and an explicit
    # NEMO_OO_SETTINGS file remains a full override.
    project_settings = _read_project_settings(root / ".nooa" / _SETTINGS_FILENAME)
    # Presence of the key, not truthiness of its value: `additional_skills_dirs: []`
    # is an explicit "none", and treating it as absent silently resurrected the
    # legacy paths the user had just emptied.
    modern_key_set = any(
        isinstance(project_settings.get(section), Mapping)
        and "additional_skills_dirs" in project_settings[section]
        for section in ("coding", "tui")
    )
    if not os.environ.get(_SETTINGS_ENV_VAR) and not modern_key_set:
        configured.extend(_legacy_project_paths(root / ".nooa" / _LEGACY_CONFIG_FILENAME))

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


def _read_project_settings(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    try:
        import yaml

        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Failed to read coding settings %s: %s", path, exc)
        return {}
    return value if isinstance(value, Mapping) else {}


def _legacy_project_paths(path: Path) -> list[str | Path]:
    """Read the pre-settings-YAML ``[tui].libs_dirs`` compatibility key."""
    if not path.is_file():
        return []
    try:
        import tomllib

        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.warning("Failed to read legacy coding settings %s: %s", path, exc)
        return []
    tui = data.get("tui")
    if not isinstance(tui, Mapping):
        return []
    value = tui.get("libs_dirs")
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


__all__ = ["load_coding_skills_dirs"]
