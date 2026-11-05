# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The model registry as one workspace sees it.

Core's ``llm_config_chain()`` takes its project layer from the directory of
the installed ``nooa`` package, not from the workspace a session works in.
A session in workspace ``W`` also reads ``W/.nooa/llm_config.yaml``, the file
``nooa connect`` run in ``W`` writes. Priority, lowest first:

1. bundled defaults, the user file and the package's project file (core's chain);
2. ``W/.nooa/llm_config.yaml``;
3. the files in ``NEMO_OO_LLM_CONFIG`` (an explicit environment variable still wins).

The merged entries are kept per workspace and are never written into the
process-wide ``nooa.unifiedllm.registry.MODELS``, so one process serving
several workspaces resolves each session's aliases against its own
workspace only. They are read again when a file in the chain is added,
removed or changed.
"""

from __future__ import annotations

import logging
import os
from contextlib import suppress
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ENV_VAR = "NEMO_OO_LLM_CONFIG"

_cache: dict[Path, tuple[tuple[Any, ...], dict[str, dict[str, Any]]]] = {}
"""By resolved workspace: the files' stamp and the merged entries read from them."""


def workspace_config_file(workspace: Path) -> Path:
    """The workspace's own model configuration file (it may not exist)."""
    return Path(workspace) / ".nooa" / "llm_config.yaml"


def llm_config_files(workspace: Path) -> list[Path]:
    """The model configuration files of ``workspace``, lowest priority first.

    Core's chain with the workspace file placed below the
    ``NEMO_OO_LLM_CONFIG`` files. Only existing files, resolved; a file
    that appears twice keeps its higher-priority place.
    """
    from nooa.llm_config import llm_config_chain

    chain = llm_config_chain()
    own = workspace_config_file(workspace)
    if not own.is_file():
        return chain
    own = own.resolve()
    env = {
        Path(entry.strip()).expanduser().resolve()
        for entry in os.environ.get(_ENV_VAR, "").split(",")
        if entry.strip()
    }
    if own in env:
        return chain
    rest = [path for path in chain if path != own]
    at = next((index for index, path in enumerate(rest) if path in env), len(rest))
    return [*rest[:at], own, *rest[at:]]


def _stamp(files: list[Path]) -> tuple[Any, ...]:
    stamp: list[tuple[str, int, int]] = []
    for path in files:
        with suppress(OSError):
            status = path.stat()
            stamp.append((str(path), status.st_mtime_ns, status.st_size))
    return tuple(stamp)


def _read_models(path: Path) -> dict[str, Any]:
    """The ``models`` mapping of one file; empty (with a warning) when unreadable."""
    import yaml

    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        logger.warning("Could not read model configuration %s: %s", path, exc)
        return {}
    models = data.get("models") if isinstance(data, dict) else None
    return models if isinstance(models, dict) else {}


def workspace_models(workspace: Path) -> dict[str, dict[str, Any]]:
    """The merged model entries of ``workspace``, by alias.

    Merged as core's ``reload_registry`` does: a later file replaces an
    alias's whole entry, and ``null`` removes it. Read again only when a
    file was added, removed or changed since the last call.
    """
    files = llm_config_files(workspace)
    stamp = _stamp(files)
    key = Path(workspace).resolve()
    cached = _cache.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    merged: dict[str, dict[str, Any]] = {}
    for path in files:
        for name, entry in _read_models(path).items():
            if isinstance(entry, dict):
                merged[name] = entry
            else:
                merged.pop(name, None)
    _cache[key] = (stamp, merged)
    return merged


def default_model(workspace: Path) -> str | None:
    """The first model alias ``workspace`` configures, or ``None`` when it has none."""
    return next(iter(workspace_models(workspace)), None)


def workspace_llm_client(
    name: str, workspace: Path, *, client_type: str | None = None, **overrides: Any
) -> Any:
    """Build the client for ``name`` as ``workspace`` configures it.

    Like ``nooa.unifiedllm.get_llm_client``, with the workspace's merged
    entries in place of the process-wide registry: a name that is not an
    alias there is passed to litellm as a model name.
    """
    from nooa.unifiedllm.registry import client_from_config

    config = workspace_models(workspace).get(name, {})
    return client_from_config(name, dict(config), client_type=client_type, **overrides)
