# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Layered YAML settings for the NOOA TUI.

This is the TUI half of the project's "one config story": TUI settings
live in ``settings.yaml`` next to ``llm_config.yaml`` and ``secrets.yaml``,
share the same directories, and are discovered through the same
:func:`nooa.layered_config.load_layered_yaml` helper.

The file is a direct serialisation of the :class:`Config` model tree
(``tui:`` / ``agent:`` sections, Pydantic field names), so it
round-trips: :func:`dump_settings` writes a config and :func:`load_settings`
reads it back identically.

Precedence (low → high, last wins) is the shared layered chain:

1. Model defaults (in code).
2. ``~/.config/nooa/settings.yaml`` (user).
3. ``<project-root>/.nooa/settings.yaml`` (project).
4. ``NEMO_OO_SETTINGS`` env var — comma-separated YAML paths.

CLI flags are layered on top of this by :meth:`Config.load`.

.. note::
   This module lives in the CLI package rather than core because it
   binds to :class:`Config`/:class:`TUIConfig`, which are defined here;
   core cannot import them without a circular dependency. The *generic*
   layered-loading machinery is in
   :mod:`nooa.layered_config`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from .config import Config

logger = logging.getLogger(__name__)

SETTINGS_FILENAME = "settings.yaml"
SETTINGS_ENV_VAR = "NEMO_OO_SETTINGS"

# Per-field coercion when reading YAML scalars into config fields. We set
# fields directly (no assignment-time validation), so Path-typed fields are
# coerced here. Dotted path (section.field) → callable; unlisted = set as-is.
_COERCE: dict[str, Any] = {
    "tui.mcp_file": lambda v: Path(v),
    "tui.trace_dir": lambda v: Path(v) if v is not None else None,
    "tui.additional_skills_dirs": lambda v: [Path(item) for item in (v or [])],
}

# Fields that are computed / runtime-only and must NOT be persisted or
# applied from file (skills_dirs is derived from discovery + CLI in
# Config.load; the no_* flags are per-invocation).
_SKIP_FIELDS = {"tui.skills_dirs", "no_splash", "no_trace"}


def load_settings(cfg: Config) -> Config:
    """Apply layered ``settings.yaml`` onto *cfg* in place and return it.

    Reads the merged settings dict (user → project → env, last wins,
    ``null`` deletes) and sets matching config fields. Unknown keys
    are warned about and skipped so a stale file never crashes startup.
    """
    from nooa.layered_config import load_layered_yaml

    data = load_layered_yaml(SETTINGS_FILENAME, SETTINGS_ENV_VAR)
    for section in ("tui", "agent"):
        sect = data.get(section)
        if isinstance(sect, dict):
            _apply_section(getattr(cfg, section), sect, section)
    return cfg


def _apply_section(obj: Any, data: dict[str, Any], prefix: str) -> None:
    """Recursively set config-model fields on *obj* from *data*."""
    for key, value in data.items():
        dotted = f"{prefix}.{key}"
        if dotted in _SKIP_FIELDS:
            continue
        if not hasattr(obj, key):
            logger.warning("Unknown settings key %r — ignoring", dotted)
            continue
        current = getattr(obj, key)
        if isinstance(value, dict) and isinstance(current, BaseModel):
            _apply_section(current, value, dotted)
            continue
        coerce = _COERCE.get(dotted)
        setattr(obj, key, coerce(value) if coerce else value)


def settings_to_dict(cfg: Config) -> dict[str, Any]:
    """Serialise the persistable fields of *cfg* to a YAML-friendly dict.

    Inverse of :func:`load_settings`: ``load_settings(Config()) ==``
    a config built by applying ``settings_to_dict(Config())`` back.
    Paths become strings; ``skills_dirs`` and runtime flags are omitted.
    """
    return {
        "tui": _model_to_dict(cfg.tui, "tui"),
        "agent": _model_to_dict(cfg.agent, "agent"),
    }


def _model_to_dict(obj: BaseModel, prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in type(obj).model_fields:
        dotted = f"{prefix}.{name}"
        if dotted in _SKIP_FIELDS:
            continue
        value = getattr(obj, name)
        if isinstance(value, BaseModel):
            out[name] = _model_to_dict(value, dotted)
        elif isinstance(value, Path):
            out[name] = str(value)
        elif isinstance(value, list):
            out[name] = [str(v) if isinstance(v, Path) else v for v in value]
        else:
            out[name] = value
    return out


def dump_settings(cfg: Config) -> str:
    """Return the YAML text for *cfg* (round-trips with :func:`load_settings`)."""
    import yaml

    return yaml.safe_dump(settings_to_dict(cfg), sort_keys=False)


def settings_present() -> bool:
    """True if a ``settings.yaml`` exists in any layer (user/project/env)."""
    from nooa.layered_config import layered_paths

    return bool(layered_paths(SETTINGS_FILENAME, SETTINGS_ENV_VAR))


def settings_path(scope: Literal["project", "user"] = "project") -> Path:
    """Return the writable ``settings.yaml`` path for *scope*."""
    from nooa.paths import get_project_dir, get_user_dir

    if scope == "project":
        return get_project_dir(SETTINGS_FILENAME)
    if scope == "user":
        return get_user_dir(SETTINGS_FILENAME)
    raise ValueError(f"Unknown settings scope: {scope!r}")


def write_settings_updates(
    updates: dict[tuple[str, ...], Any],
    *,
    scope: Literal["project", "user"] = "project",
    dry_run: bool = False,
) -> tuple[Path, dict[str, Any]]:
    """Apply nested setting updates to one writable settings file.

    ``updates`` maps dotted-path tuples like ``("tui", "default_model")``
    to YAML-friendly values. Existing sibling keys are preserved. When
    ``dry_run`` is true, the returned data is what would be written.
    """
    import yaml

    path = settings_path(scope)
    data: dict[str, Any] = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text())
        if isinstance(loaded, dict):
            data = loaded

    for setting_path, value in updates.items():
        _set_mapping_path(data, list(setting_path), value)

    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path, data


def delete_settings_value(
    setting_path: tuple[str, ...],
    *,
    scope: Literal["project", "user"] = "project",
    dry_run: bool = False,
) -> tuple[Path, dict[str, Any], bool]:
    """Delete one nested setting while preserving all sibling settings."""
    import yaml

    path = settings_path(scope)
    data: dict[str, Any] = {}
    if path.exists():
        loaded = yaml.safe_load(path.read_text())
        if isinstance(loaded, dict):
            data = loaded

    current: dict[str, Any] = data
    parents: list[tuple[dict[str, Any], str]] = []
    for part in setting_path[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            return path, data, False
        parents.append((current, part))
        current = child
    deleted = current.pop(setting_path[-1], None) is not None
    if not deleted:
        return path, data, False

    for parent, key in reversed(parents):
        child = parent.get(key)
        if isinstance(child, dict) and not child:
            parent.pop(key)
        else:
            break
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path, data, True


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


# Commented scaffold written on first run. Everything is commented out so
# the file documents the schema without overriding any defaults.
SETTINGS_TEMPLATE = """\
# NVIDIA Labs Object Oriented Agents (NOOA) — TUI settings
#
# Layered, last wins:
#   1. built-in defaults
#   2. this file (user:    ~/.config/nooa/settings.yaml)
#   3. project file:        .nooa/settings.yaml
#   4. $NEMO_OO_SETTINGS    (comma-separated YAML paths)
#
# All keys are optional; uncomment only what you want to change.
# `null` removes a key inherited from a lower layer.

tui:
  # LLM model alias (from the unifiedllm registry) or a litellm model name.
  # default_model: {default_model}

  # Show the agent's Python execution panels.
  # show_python: false

  # Show bounded unified diffs when coding tools edit files.
  # show_diffs: true

  # Audit DONE turns and continue when autonomous work remains.
  # Configure the judge model before enabling this.
  # keep_going: false
  # keep_going_model: nemotron3-nano-30b

  # Long-term memory. "project" shares one store across project sessions;
  # "session" uses a sidecar database for only the current session.
  # Prefer /memory so the choice is persisted per agent.
  # memory: off                 # off | session | project
  # memory_path: .nooa/memory/memory.sqlite
  # memory_owner: coding-agent

  # Consolidate memory during idle windows. Prefer /reflection so the choice
  # is persisted per agent. Generative reflection uses the current model.
  # reflection: false
  # reflection_generative: true
  # reflection_debounce_s: 10.0
  # reflection_grace_s: 0.5

  # Vi keybindings in the input prompt.
  # vi_mode: false

  # Additional skill roots. Prefer /skills add <directory> so these are
  # discovered immediately and saved here for future TUI runs.
  # additional_skills_dirs: []

  # Write trace files here (relative to project root, or ":project:").
  # trace_dir: .nooa/traces

  # Ordered toolbar items. Built-ins: time, model, cwd, context, session.
  # toolbar_items: [time, model, context, session]

  # MCP servers, declared inline (preferred over a separate .mcp.json).
  # Keep secrets in the host environment. The TUI resolves ${VAR} only after
  # you approve the exact server definition with `/mcp approve <name> <code>`.
  # Repository config cannot approve itself; any config change invalidates trust.
  # mcp_auto_connect: [maas]
  # mcp_servers:
  #   maas:
  #     url: https://maas.stg.astra.nvidia.com/maas/confluence/mcp
  #     transport: streamable-http
  #     headers:
  #       Authorization: "Bearer ${MAAS_API_KEY}"

# agent:
#   working_dir: "."
#   summarization:
#     policy: token_budget   # token_budget | none
#     max_tokens: null       # null = 80% of the model's context window
"""


def render_settings_template(cfg: Config) -> str:
    """Render the commented first-run scaffold for *cfg*.

    Uses ``str.replace`` rather than ``str.format`` because the template
    contains literal braces (e.g. ``${MAAS_API_KEY}`` in the gated MCP example)
    that ``format()`` would treat as fields and raise ``KeyError`` on.
    """
    return SETTINGS_TEMPLATE.replace("{default_model}", cfg.tui.default_model)
