# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM-registry config-file discovery.

The ``unifiedllm`` package is intentionally ignorant of project-wide
filesystem conventions (``paths.py``). This module is where the
project's path infrastructure combines with the LLM-config discovery
rules into a single helper:

- :func:`llm_config_chain` returns a priority-ordered list of YAML
  files (lowest priority first). Pass it to
  :func:`nooa.unifiedllm.reload_registry` to populate the
  registry.
- :func:`bundled_config_paths` returns the on-disk paths of all
  bundled-default YAMLs registered via the
  ``nooa.bundled_configs`` entry-point group. External
  packages (e.g. ``nemo-oo-agents-nvidia``) register their YAML there
  and the core picks them up automatically; install / don't install
  to opt in / out.

Priority (last wins):

1. Bundled defaults — every entry-point in
   ``nooa.bundled_configs`` (lowest)
2. ``get_user_dir("llm_config.yaml")`` — user-global default
3. ``get_project_dir("llm_config.yaml")`` — project-local config
4. ``NEMO_OO_LLM_CONFIG`` env var (comma-separated paths) — global
   override; highest priority so a shell session can override
   project / user files without editing them.
"""

from __future__ import annotations

import logging
from importlib import metadata
from pathlib import Path

from nooa.layered_config import layered_paths

logger = logging.getLogger(__name__)

_CONFIG_FILENAME = "llm_config.yaml"
_CONFIG_ENV_VAR = "NEMO_OO_LLM_CONFIG"
_BUNDLED_ENTRY_POINT_GROUP = "nooa.bundled_configs"


def bundled_config_paths() -> list[Path]:
    """Return on-disk paths for every registered bundled-default YAML.

    Iterates entry-points in the ``nooa.bundled_configs``
    group, calls each (they're zero-arg callables returning
    ``Path | None``), and returns the resulting list with ``None``\\ s
    filtered out. Entries are sorted by entry-point name for stable
    ordering.

    External packages register their bundled defaults like this::

        # pyproject.toml
        [project.entry-points."nooa.bundled_configs"]
        nvidia = "nooa_nvidia:get_default_config_path"

    The callable receives no arguments and returns the absolute path
    of its YAML, or ``None`` if the resource can't be materialised
    (zipped wheels, exotic install layouts).
    """
    out: list[Path] = []
    try:
        eps = metadata.entry_points(group=_BUNDLED_ENTRY_POINT_GROUP)
    except Exception:  # noqa: BLE001 — importlib.metadata can raise on broken installs
        logger.warning("Failed to enumerate bundled-config entry points", exc_info=True)
        return out

    for ep in sorted(eps, key=lambda e: e.name):
        try:
            func = ep.load()
            path = func()
        except Exception:  # noqa: BLE001 — third-party entry-point code, be defensive
            logger.warning(
                "Bundled-config entry-point %r failed to load",
                ep.name,
                exc_info=True,
            )
            continue
        if path is None:
            continue
        out.append(Path(path))
    return out


def llm_config_chain() -> list[Path]:
    """Return YAML config files for the LLM registry, lowest priority first.

    Layers, in order:

    1. Bundled defaults — every package that registers under the
       ``nooa.bundled_configs`` entry-point group (in
       entry-point-name order). Install ``nemo-oo-agents-nvidia`` for
       the NVIDIA-gateway aliases; install nothing for an OSS-only
       registry.
    2. ``get_user_dir("llm_config.yaml")`` — user-global default
       (``~/.config/nooa/llm_config.yaml``).
    3. ``get_project_dir("llm_config.yaml")`` — project-local config
       (``<project-root>/.nooa/llm_config.yaml``).
    4. ``NEMO_OO_LLM_CONFIG`` env var — comma-separated YAML paths.
       Highest priority: an explicit env var always wins so users can
       override project / user config from a shell session without
       editing files.

    Only files that actually exist are returned. Each path is
    canonicalised via :meth:`pathlib.Path.resolve` so duplicates and
    symlink aliases collapse: if the same resolved path appears in
    multiple layers, the **lower-priority occurrence is dropped and
    the higher-priority position is kept** — so the last-wins merge
    in :func:`nooa.unifiedllm.reload_registry` lines up
    with the layering described here.

    Paths from ``NEMO_OO_LLM_CONFIG`` that don't exist log a warning.
    User and project layer paths are silently skipped when missing.

    The user → project → env-var walk (and resolved-path dedup) is the
    shared :func:`nooa.layered_config.layered_paths` logic;
    this function only adds the bundled-defaults layer on top.
    """
    return layered_paths(
        _CONFIG_FILENAME,
        _CONFIG_ENV_VAR,
        prepend=bundled_config_paths(),
    )


__all__ = ["bundled_config_paths", "llm_config_chain"]
