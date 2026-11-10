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

Priority (last wins):

1. ``get_user_dir("llm_config.yaml")`` — user-global default (lowest)
2. ``get_project_dir("llm_config.yaml")`` — project-local config
3. ``NEMO_OO_LLM_CONFIG`` env var (comma-separated paths) — global
   override; highest priority so a shell session can override
   project / user files without editing them.
"""

from __future__ import annotations

import logging
from pathlib import Path

from nooa.layered_config import layered_paths

logger = logging.getLogger(__name__)

_CONFIG_FILENAME = "llm_config.yaml"
_CONFIG_ENV_VAR = "NEMO_OO_LLM_CONFIG"


def llm_config_chain() -> list[Path]:
    """Return YAML config files for the LLM registry, lowest priority first.

    Layers, in order:

    1. ``get_user_dir("llm_config.yaml")`` — user-global default
       (``~/.config/nooa/llm_config.yaml``).
    2. ``get_project_dir("llm_config.yaml")`` — project-local config
       (``<project-root>/.nooa/llm_config.yaml``).
    3. ``NEMO_OO_LLM_CONFIG`` env var — comma-separated YAML paths.
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
    shared :func:`nooa.layered_config.layered_paths` logic.
    """
    return layered_paths(_CONFIG_FILENAME, _CONFIG_ENV_VAR)


__all__ = ["llm_config_chain"]
