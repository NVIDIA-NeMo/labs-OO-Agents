# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared layered-config discovery for NVIDIA OO Agents.

One mental model for *every* config file the project ships — LLM aliases
(``llm_config.yaml``), TUI settings (``settings.yaml``), and secrets
(``secrets.yaml``). All three live in the same directories, share the
same precedence rules, and are discovered through the helpers here.

Precedence (low → high, last wins):

1. **Bundled defaults** — supplied by the caller as *prepend* paths
   (e.g. the entry-point YAMLs for LLM config). Nothing is bundled for
   settings or secrets.
2. **User** — ``get_user_dir(filename)``
   (``~/.config/nooa/<filename>``).
3. **Project** — ``get_project_dir(filename)``
   (``<project-root>/.nooa/<filename>``).
4. **Env-var path override** — a comma-separated list of YAML paths in
   the named environment variable (``NEMO_OO_LLM_CONFIG``,
   ``NEMO_OO_SETTINGS``, ``NEMO_OO_SECRETS``). Highest priority so a
   shell session can override files without editing them.

Two entry points:

- :func:`layered_paths` returns the priority-ordered, deduplicated list
  of existing YAML paths (lowest priority first). Use it when a
  downstream consumer wants the *paths* (e.g. the LLM registry, which
  does its own last-wins merge).
- :func:`load_layered_yaml` reads those paths and deep-merges them into
  a single dict (last wins; ``null`` deletes a key). Use it for settings
  and secrets.

Resolved-path deduplication: each path is canonicalised via
:meth:`pathlib.Path.resolve` so symlink aliases and repeated entries
collapse. When the same resolved path appears in multiple layers, the
**lower-priority occurrence is dropped and the higher-priority position
is kept** — matching the last-wins semantics described above.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from nooa.paths import get_project_dir, get_user_dir

logger = logging.getLogger(__name__)


def _resolved_if_exists(path: Path) -> Path | None:
    """Return ``path.resolve()`` if the file exists, else ``None``."""
    try:
        if path.exists():
            return path.resolve()
    except OSError:
        # Permission or stat failure — treat as missing.
        return None
    return None


def _env_paths(env_var: str) -> list[tuple[Path, Path | None]]:
    """Parse a comma-separated env var into ``(raw, resolved)`` pairs.

    ``resolved`` is ``None`` when the file doesn't exist (the caller
    decides whether to warn).
    """
    raw = os.environ.get(env_var, "").strip()
    if not raw:
        return []
    out: list[tuple[Path, Path | None]] = []
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        p = Path(entry).expanduser()
        out.append((p, _resolved_if_exists(p)))
    return out


def layered_paths(
    filename: str,
    env_var: str | None = None,
    *,
    prepend: Iterable[Path] = (),
) -> list[Path]:
    """Return existing config paths for *filename*, lowest priority first.

    Args:
        filename: Bare filename looked up under the user and project
            directories (e.g. ``"llm_config.yaml"``).
        env_var: Optional environment variable holding a comma-separated
            list of override paths (highest priority). ``None`` disables
            the env-var layer.
        prepend: Lowest-priority paths placed before the user layer
            (e.g. entry-point bundled defaults). Already in priority
            order; non-existent entries are dropped.

    Only files that exist are returned, each canonicalised via
    :meth:`pathlib.Path.resolve`. Env-var paths that don't exist log a
    warning; user / project paths are silently skipped when missing.
    """
    # Walk layers in priority order and keep the *latest* (highest
    # priority) occurrence of any resolved path.
    chain: list[Path] = []
    seen: dict[Path, int] = {}

    def _push(resolved: Path) -> None:
        if resolved in seen:
            chain.pop(seen[resolved])
            for k, v in seen.items():
                if v > seen[resolved]:
                    seen[k] = v - 1
            seen.pop(resolved)
        seen[resolved] = len(chain)
        chain.append(resolved)

    for p in prepend:
        resolved = _resolved_if_exists(p)
        if resolved is not None:
            _push(resolved)

    user = _resolved_if_exists(get_user_dir(filename))
    if user is not None:
        _push(user)

    project = _resolved_if_exists(get_project_dir(filename))
    if project is not None:
        _push(project)

    if env_var is not None:
        for raw, resolved in _env_paths(env_var):
            if resolved is None:
                logger.warning("%s path does not exist: %s", env_var, raw)
                continue
            _push(resolved)

    return chain


def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Last-wins deep merge of *overlay* onto *base* (returns *base*).

    Rules:

    - A ``None`` value in *overlay* **deletes** the key from the result.
    - When both sides hold a dict, merge recursively.
    - Otherwise the overlay value replaces the base value.
    """
    for key, value in overlay.items():
        if value is None:
            base.pop(key, None)
        elif isinstance(value, Mapping) and isinstance(base.get(key), dict):
            base[key] = _deep_merge(base[key], value)
        elif isinstance(value, Mapping):
            base[key] = _deep_merge({}, value)
        else:
            base[key] = value
    return base


def load_layered_yaml(
    filename: str,
    env_var: str | None = None,
    *,
    prepend: Iterable[Path] = (),
) -> dict[str, Any]:
    """Load and deep-merge every layer of *filename* into a single dict.

    Walks the same chain as :func:`layered_paths` and merges last-wins.
    ``null`` values delete keys. A file whose YAML doesn't parse to a mapping
    is skipped with a warning so one bad layer never takes down the rest.

    *prepend* supplies lowest-priority paths (e.g. entry-point bundled
    defaults) just like :func:`layered_paths`; omit it for settings / secrets.
    """
    import yaml

    merged: dict[str, Any] = {}
    for path in layered_paths(filename, env_var, prepend=prepend):
        try:
            data = yaml.safe_load(path.read_text())
        except (OSError, yaml.YAMLError) as e:
            logger.warning("Failed to load config file %s: %s", path, e)
            continue
        if data is None:
            continue
        if not isinstance(data, dict):
            logger.warning("Config file %s is not a YAML mapping; skipping", path)
            continue
        merged = _deep_merge(merged, data)
    return merged


__all__ = ["layered_paths", "load_layered_yaml"]
