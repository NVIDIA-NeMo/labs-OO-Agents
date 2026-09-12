# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Layered YAML settings for the NOOA TUI.

This is the TUI half of the project's "one config story": TUI settings
live in ``settings.yaml`` next to ``llm_config.yaml`` and ``secrets.yaml``,
share the same directories, and are discovered through the same
:func:`nooa.layered_config.load_layered_yaml` helper.

Behavior is persisted under ``coding:`` for both interactive hosts;
presentation stays under ``tui:``. Legacy ``tui:`` behavior and
``agent.summarization`` remain readable. :func:`dump_settings` writes a config
and :func:`load_settings` reads it back identically.

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
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

# Compatibility exports: behavior writes now use the shared canonical namespace.
from nooa_cli.interactive.settings import (
    delete_settings_value as delete_settings_value,
)
from nooa_cli.interactive.settings import (
    settings_path as settings_path,
)
from nooa_cli.interactive.settings import (
    write_settings_updates as write_settings_updates,
)

if TYPE_CHECKING:
    from .config import Config

logger = logging.getLogger(__name__)

SETTINGS_FILENAME = "settings.yaml"
SETTINGS_ENV_VAR = "NEMO_OO_SETTINGS"

# Per-field coercion when reading YAML scalars into config fields. We set
# fields directly (no assignment-time validation), so Path-typed fields are
# coerced here. Dotted path (section.field) → callable; unlisted = set as-is.
_COERCE: dict[str, Any] = {
    "tui.display_mode": lambda v: _coerce_display_mode(v),
    "tui.theme": lambda v: _coerce_theme(v),
    "tui.mcp_file": lambda v: Path(v),
    "tui.trace_dir": lambda v: Path(v) if v is not None else None,
    "tui.additional_skills_dirs": lambda v: [Path(item) for item in (v or [])],
    "tui.active_skills": lambda v: [str(item) for item in (v or [])],
    "tui.inactive_skills": lambda v: [str(item) for item in (v or [])],
}


def _coerce_display_mode(value: Any) -> Any:
    if value is None:
        return None
    from .config import DisplayMode

    return DisplayMode(value)


def _coerce_theme(value: Any) -> str:
    from .theme import THEMES, reload_themes

    reload_themes()
    name = str(value).lower()
    if name not in THEMES:
        logger.warning("Unknown persisted theme %r; falling back to mocha", value)
        return "mocha"
    return name


# Fields that are computed / runtime-only and must NOT be persisted or
# applied from file (skills_dirs is derived from discovery + CLI in
# Config.load; the no_* flags are per-invocation).
_SKIP_FIELDS = {"tui.skills_dirs", "no_splash", "no_trace", "legacy_agent"}


def load_settings(cfg: Config, *, workspace: str | Path | None = None) -> Config:
    """Apply layered ``settings.yaml`` onto *cfg* in place and return it.

    Reads the merged settings dict (user → project → env, last wins,
    ``null`` deletes) and sets matching config fields. Unknown keys
    are warned about and skipped so a stale file never crashes startup.
    """
    from nooa_cli.interactive.settings import (
        behavior_fields,
        load_settings_data,
        resolve_behavior_settings,
    )

    data = load_settings_data(workspace)
    for section in ("tui", "agent"):
        sect = data.get(section)
        if isinstance(sect, dict):
            excluded = behavior_fields() if section == "tui" else {"summarization"}
            sect = {key: value for key, value in sect.items() if key not in excluded}
            _apply_section(getattr(cfg, section), sect, section)
    behavior = resolve_behavior_settings(data)
    summarization = behavior.pop("summarization", None)
    if isinstance(summarization, dict):
        _apply_section(cfg.agent.summarization, summarization, "agent.summarization")
    _apply_section(cfg.tui, behavior, "tui")
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
    from nooa_cli.interactive.settings import behavior_fields

    tui = _model_to_dict(cfg.tui, "tui")
    agent = _model_to_dict(cfg.agent, "agent")
    coding = {key: tui.pop(key) for key in list(tui) if key in behavior_fields()}
    coding["summarization"] = agent.pop("summarization")
    return {"tui": tui, "agent": agent, "coding": coding}


def _model_to_dict(obj: BaseModel, prefix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in type(obj).model_fields:
        dotted = f"{prefix}.{name}"
        if dotted in _SKIP_FIELDS or dotted == "tui.full_screen":
            continue
        value = getattr(obj, name)
        if dotted == "tui.display_mode" and value is None:
            # Persist the effective mode without importing config at module load
            # time (config imports this module from Config.load()).
            from .config import resolve_display_mode

            value = resolve_display_mode(obj).value
        if isinstance(value, BaseModel):
            out[name] = _model_to_dict(value, dotted)
        elif isinstance(value, Path):
            out[name] = str(value)
        elif isinstance(value, Enum):
            out[name] = value.value
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


# Commented scaffold written on first run. Everything is commented out so
# the file documents the schema without overriding any defaults.
SETTINGS_TEMPLATE = """\
# NVIDIA Labs Object Oriented Agents (NOOA) — interactive settings
#
# Layered, last wins:
#   1. built-in defaults
#   2. this file (user:    ~/.config/nooa/settings.yaml)
#   3. project file:        .nooa/settings.yaml
#   4. $NEMO_OO_SETTINGS    (comma-separated YAML paths)
#
# All keys are optional; uncomment only what you want to change.
# `null` removes a key inherited from a lower layer.

coding:
  # LLM model alias (from the unifiedllm registry) or a litellm model name.
  # default_model: {default_model}

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

  # Additional skill roots. Prefer /skills add <directory> so these are
  # discovered immediately and saved here for both interactive hosts.
  # additional_skills_dirs: []
  # active_skills: []    # re-activate these before SessionResumed hooks run
  # inactive_skills: []  # keep explicitly deactivated skills inactive on restart

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

  # summarization:
  #   policy: token_budget   # token_budget | none
  #   max_tokens: null       # null = 80% of the model's context window

tui:
  # Color palette. Use /theme to browse; /theme <id> applies directly.
  # Custom Base16/Base24 YAML themes load from user/project .nooa/themes/.
  # theme: mocha

  # Show the agent's Python execution panels.
  # show_python: false

  # Show bounded unified diffs when coding tools edit files.
  # show_diffs: true

  # Vi keybindings in the input prompt.
  # vi_mode: false

  # Write trace files here (relative to project root, or ":project:").
  # trace_dir: .nooa/traces

  # Ordered toolbar items. Built-ins: time, model, cwd, context, session.
  # toolbar_items: [time, model, context, session]
"""


def render_settings_template(cfg: Config) -> str:
    """Render the commented first-run scaffold for *cfg*.

    Uses ``str.replace`` rather than ``str.format`` because the template
    contains literal braces (e.g. ``${MAAS_API_KEY}`` in the gated MCP example)
    that ``format()`` would treat as fields and raise ``KeyError`` on.
    """
    return SETTINGS_TEMPLATE.replace("{default_model}", cfg.tui.default_model)
