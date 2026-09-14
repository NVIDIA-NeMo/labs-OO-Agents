# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical behavior settings and persistence for interactive hosts."""

from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from nooa.layered_config import load_layered_yaml

SETTINGS_FILENAME = "settings.yaml"
SETTINGS_ENV_VAR = "NEMO_OO_SETTINGS"
logger = logging.getLogger(__name__)


def behavior_fields() -> frozenset[str]:
    from .options import SessionOptions

    return frozenset(SessionOptions.model_fields) - {
        "working_dir",
        "legacy_agent",
        "skills_dirs",
        "agent_spec",
    }


def load_settings_data(workspace: str | Path | None = None) -> dict[str, Any]:
    project = Path(workspace).expanduser().resolve() / ".nooa" if workspace is not None else None
    return load_layered_yaml(SETTINGS_FILENAME, SETTINGS_ENV_VAR, project_dir=project)


def resolve_behavior_settings(data: dict[str, Any]) -> dict[str, Any]:
    """Resolve legacy aliases and partial nested overrides identically for both hosts."""
    from .options import SessionOptions

    values: dict[str, Any] = {}
    agent = data.get("agent", {})
    if isinstance(agent, dict) and isinstance(agent.get("summarization"), dict):
        values["summarization"] = deepcopy(agent["summarization"])
    for name in ("tui", "coding"):
        section = data.get(name)
        if not isinstance(section, dict):
            continue
        if name == "tui" and section:
            logger.warning("Reading legacy tui settings; move shared behavior settings to coding")
        for key, value in section.items():
            if key == "agent_spec":
                logger.warning(
                    "Ignoring %s.agent_spec in settings; select custom agents explicitly "
                    "through the host CLI or SessionOptions overrides",
                    name,
                )
            if key not in behavior_fields():
                if name == "coding" and key != "agent_spec":
                    logger.warning("Ignoring unsupported coding setting %r", key)
                continue
            if key == "summarization" and isinstance(value, dict):
                values.setdefault(key, {}).update(value)
            else:
                values[key] = deepcopy(value)
    # Read historical per-agent keys at the configuration boundary. Explicit
    # canonical values win when both spellings are present.
    from nooa_cli.coding.identity import CODING_AGENT, EXPERIMENTAL_CODING_AGENT, LEGACY_AGENT_SPECS

    for field in ("memory_agents", "memory_owner_agents", "reflection_agents"):
        preferences = values.get(field)
        if isinstance(preferences, dict):
            for old, new in LEGACY_AGENT_SPECS.items():
                if old in preferences:
                    key = CODING_AGENT if new == EXPERIMENTAL_CODING_AGENT else new
                    logger.warning("Migrating legacy %s agent key %r to %r", field, old, key)
                    preferences.setdefault(key, preferences[old])
                    del preferences[old]
    return SessionOptions(**values).model_dump(exclude_unset=True)


def canonical_setting_path(path: tuple[str, ...]) -> tuple[str, ...]:
    if len(path) >= 2 and (
        (path[0] == "tui" and path[1] in behavior_fields())
        or (path[0] == "agent" and path[1] == "summarization")
    ):
        return ("coding", *path[1:])
    return path


def settings_path(
    scope: Literal["project", "user"] = "project", *, workspace: str | Path | None = None
) -> Path:
    """Return the writable ``settings.yaml`` path for *scope*."""
    from nooa.paths import get_project_dir, get_user_dir

    if scope == "project":
        return (
            Path(workspace).expanduser().resolve() / ".nooa" / SETTINGS_FILENAME
            if workspace is not None
            else get_project_dir(SETTINGS_FILENAME)
        )
    if scope == "user":
        return get_user_dir(SETTINGS_FILENAME)
    raise ValueError(f"Unknown settings scope: {scope!r}")


def write_settings_updates(
    updates: dict[tuple[str, ...], Any],
    *,
    scope: Literal["project", "user"] = "project",
    dry_run: bool = False,
    workspace: str | Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Apply nested setting updates to one writable settings file.

    ``updates`` maps dotted-path tuples like ``("coding", "default_model")``
    to YAML-friendly values. Existing sibling keys are preserved. When
    ``dry_run`` is true, the returned data is what would be written.
    """
    import yaml

    path = settings_path(scope, workspace=workspace)
    data: dict[str, Any] = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text())
        if isinstance(loaded, dict):
            data = loaded

    for setting_path, value in updates.items():
        canonical = canonical_setting_path(setting_path)
        if len(canonical) > 2 and canonical[0] == "coding":
            # A first nested canonical write must retain siblings that only
            # exist under the legacy alias (e.g. another agent's memory mode).
            coding = data.get("coding")
            if not isinstance(coding, dict) or canonical[1] not in coding:
                inherited = resolve_behavior_settings(data).get(canonical[1])
                if isinstance(inherited, dict):
                    _set_mapping_path(data, list(canonical[:2]), deepcopy(inherited))
        _set_mapping_path(data, list(canonical), value)

    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path, data


def _set_mapping_path(data: dict[str, Any], path: list[str], value: Any) -> None:
    """Set ``data[path[0]]...[path[-1]]`` creating dictionaries as needed."""
    current = data
    for part in path[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            child = {}
            current[part] = child
        current = child
    current[path[-1]] = value
