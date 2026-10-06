# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The workspace's settings: behaviour settings, skill directories, and persistence.

One layered ``settings.yaml`` (user, then project ``.nooa/``, or the file
``NEMO_OO_SETTINGS`` names) holds them for every interactive host.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from nooa.layered_config import load_layered_yaml

SETTINGS_FILENAME = "settings.yaml"
SETTINGS_ENV_VAR = "NEMO_OO_SETTINGS"
logger = logging.getLogger(__name__)

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


def load_skills_dirs(
    workspace: str | Path,
    *,
    explicit: Iterable[str | Path] = (),
) -> list[Path]:
    """Return existing skill roots for one Atom workspace.

    The shared ``coding.additional_skills_dirs`` setting is preferred for new
    configuration. ``tui.additional_skills_dirs`` remains supported while the
    TUI migrates to the shared section. Relative configured paths are resolved
    against the active workspace, not the ACP server process's checkout.
    """
    root = Path(workspace).expanduser().resolve()
    settings = load_layered_yaml(
        SETTINGS_FILENAME,
        SETTINGS_ENV_VAR,
        project_dir=root / ".nooa",
    )

    configured: list[str | Path] = []
    coding = settings.get("coding")
    section = (
        "coding" if isinstance(coding, Mapping) and "additional_skills_dirs" in coding else "tui"
    )
    if section == "tui" and isinstance(settings.get("tui"), Mapping):
        if "additional_skills_dirs" in settings["tui"]:
            logger.warning(
                "Reading legacy tui.additional_skills_dirs; use coding.additional_skills_dirs"
            )
    configured.extend(_setting_paths(settings, section))
    # A workspace's old config.toml is still a workspace layer. Do not let an
    # unrelated user-level settings.yaml silently suppress it. A modern
    # workspace settings file supersedes the legacy file, and an explicit
    # NEMO_OO_SETTINGS file remains a full override.
    project_settings = _read_project_settings(root / ".nooa" / SETTINGS_FILENAME)
    # Presence of the key, not truthiness of its value: `additional_skills_dirs: []`
    # is an explicit "none", and treating it as absent silently resurrected the
    # legacy paths the user had just emptied.
    modern_key_set = any(
        isinstance(project_settings.get(section), Mapping)
        and "additional_skills_dirs" in project_settings[section]
        for section in ("coding", "tui")
    )
    if not os.environ.get(SETTINGS_ENV_VAR) and not modern_key_set:
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
    import yaml

    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        logger.warning("Failed to read coding settings %s: %s", path, exc)
        return {}
    return value if isinstance(value, Mapping) else {}


def _legacy_project_paths(path: Path) -> list[str | Path]:
    """Read the pre-settings-YAML ``[tui].libs_dirs`` compatibility key."""
    if not path.is_file():
        return []
    import tomllib

    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        logger.warning("Failed to read legacy coding settings %s: %s", path, exc)
        return []
    tui = data.get("tui")
    if not isinstance(tui, Mapping):
        return []
    value = tui.get("libs_dirs")
    if isinstance(value, (str, list)):
        logger.warning(
            "Reading legacy config.toml tui.libs_dirs; use settings.yaml coding.additional_skills_dirs"
        )
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


@lru_cache(maxsize=1)
def _warn_ignored_agent_spec() -> None:
    """Report the process-wide agent-selection policy once despite repeated loads."""
    logger.warning(
        "Ignoring coding.agent_spec and tui.agent_spec in all settings layers "
        "(including user and project settings); select custom agents explicitly "
        "through the host CLI or AtomOptions overrides"
    )


def behavior_fields() -> frozenset[str]:
    from .options import AtomOptions

    return frozenset(AtomOptions.model_fields) - {
        "working_dir",
        "skills_dirs",
        "agent_spec",
    }


def load_settings_data(workspace: str | Path | None = None) -> dict[str, Any]:
    project = Path(workspace).expanduser().resolve() / ".nooa" if workspace is not None else None
    return load_layered_yaml(SETTINGS_FILENAME, SETTINGS_ENV_VAR, project_dir=project)


def load_behavior_settings(workspace: str | Path) -> dict[str, Any]:
    """The workspace's resolved behaviour settings; the defaults if they are invalid.

    A value of the wrong type (``active_skills: "one"``) must not stop a
    session from starting: a warning names the settings file and the error,
    and every setting falls back to its default.
    """
    from pydantic import ValidationError

    try:
        return resolve_behavior_settings(load_settings_data(workspace))
    except ValidationError as exc:
        errors = "; ".join(
            f"{'.'.join(str(part) for part in error['loc'])}: {error['msg']}"
            for error in exc.errors()
        )
        files = ", ".join(_invalid_settings_files(workspace)) or "the merged settings"
        logger.warning("Invalid coding settings in %s (%s); using the defaults", files, errors)
        return {}


def _invalid_settings_files(workspace: str | Path) -> list[str]:
    """The settings files whose own values do not validate."""
    import yaml
    from pydantic import ValidationError

    from nooa.layered_config import layered_paths

    project = Path(workspace).expanduser().resolve() / ".nooa"
    invalid = []
    for path in layered_paths(SETTINGS_FILENAME, SETTINGS_ENV_VAR, project_dir=project):
        try:
            data = yaml.safe_load(path.read_text())
        except (OSError, UnicodeError, yaml.YAMLError):
            continue  # load_layered_yaml already skipped and reported it
        if not isinstance(data, dict):
            continue
        try:
            resolve_behavior_settings(data)
        except ValidationError:
            invalid.append(str(path))
    return invalid


def resolve_behavior_settings(data: dict[str, Any]) -> dict[str, Any]:
    """Resolve legacy aliases and partial nested overrides identically for both hosts."""
    from .options import AtomOptions

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
                _warn_ignored_agent_spec()
            if key not in behavior_fields():
                if name == "coding" and key != "agent_spec":
                    logger.warning("Ignoring unsupported coding setting %r", key)
                continue
            if key == "summarization" and isinstance(value, dict):
                values.setdefault(key, {}).update(value)
            else:
                values[key] = deepcopy(value)
    servers = values.get("mcp_servers")
    if isinstance(servers, dict):
        # forget_mcp() writes ``name: null`` to mask an inherited definition;
        # a masked server is simply absent from the resolved options.
        values["mcp_servers"] = {k: v for k, v in servers.items() if v is not None}
    return AtomOptions(**values).model_dump(exclude_unset=True)


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
            # exist under the legacy alias (e.g. another MCP server definition).
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
