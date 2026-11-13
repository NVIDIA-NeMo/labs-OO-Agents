# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model sources: installed packages that add models and a sign-in.

A package adds a source with an entry point in the
``nooa_atom.model_sources`` group::

    [project.entry-points."nooa_atom.model_sources"]
    hub = "my_package.hub:source"

The entry point loads to a :class:`ModelSource`, or to a callable with no
arguments that returns one. Atom lists the source's models with the models
of the configuration files (which win for the same alias), and an ACP
client that runs terminal auth offers the source's sign-in next to
``nooa connect``.

A source that cannot be loaded or is not valid is logged and skipped:
Atom always starts.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "nooa_atom.model_sources"

_ID = re.compile(r"[a-z0-9][a-z0-9-]*")

_sources: list[ModelSource] | None = None
"""The loaded sources, once per process."""


@runtime_checkable
class ModelSource(Protocol):
    """What a model source provides."""

    id: str
    """Stable id: lowercase letters, digits and ``-``."""
    name: str
    """Shown in the client's sign-in picker, e.g. ``"Example Hub"``."""
    description: str
    """One line under the name in the picker."""

    def models(self, workspace: Path) -> dict[str, dict[str, Any]]:
        """The source's model entries by alias, in the ``llm_config.yaml`` ``models:`` format.

        Called often: it must be fast, so the source caches what it fetches.
        """
        ...

    def sign_in(self) -> str | None:
        """Set the source up in a real terminal (it may prompt).

        Returns the alias to switch to, or ``None`` when nothing was set up.
        """
        ...


def _check(source: Any) -> str | None:
    """Why ``source`` is not a usable model source, or ``None`` when it is."""
    for field in ("id", "name", "description"):
        if not isinstance(getattr(source, field, None), str):
            return f"it has no {field!r} text"
    for method in ("models", "sign_in"):
        if not callable(getattr(source, method, None)):
            return f"it has no {method}() method"
    if not _ID.fullmatch(source.id):
        return f"its id {source.id!r} is not lowercase letters, digits and '-'"
    return None


def _load() -> list[ModelSource]:
    from importlib.metadata import entry_points

    sources: list[ModelSource] = []
    seen: set[str] = set()
    try:
        found = list(entry_points(group=ENTRY_POINT_GROUP))
    except Exception:
        logger.warning("Could not list the model sources", exc_info=True)
        return sources
    for entry in found:
        try:
            source = entry.load()
            # A class or a factory function: call it for the source.
            if isinstance(source, type) or (_check(source) is not None and callable(source)):
                source = source()
        except Exception:
            logger.warning("Skipped model source %s: it failed to load", entry.name, exc_info=True)
            continue
        reason = _check(source)
        if reason is None and source.id in seen:
            reason = f"another source has the id {source.id!r}"
        if reason is not None:
            logger.warning("Skipped model source %s: %s", entry.name, reason)
            continue
        seen.add(source.id)
        sources.append(source)
    return sources


def model_sources() -> list[ModelSource]:
    """The installed model sources, loaded once per process."""
    global _sources
    if _sources is None:
        _sources = _load()
    return list(_sources)


def reset_model_sources() -> None:
    """Forget the loaded sources, so the next call loads them again (for tests)."""
    global _sources
    _sources = None


def find_source(source_id: str) -> ModelSource | None:
    """The installed source with this id, or ``None``."""
    return next((source for source in model_sources() if source.id == source_id), None)


def source_models(workspace: Path) -> dict[str, dict[str, Any]]:
    """The model entries of all sources, by alias; a later source wins for the same alias.

    A source whose ``models()`` fails is logged and skipped, and so is an
    entry that is not a mapping.
    """
    merged: dict[str, dict[str, Any]] = {}
    for source in model_sources():
        try:
            models = source.models(Path(workspace))
        except Exception:
            logger.warning("Model source %s could not list its models", source.id, exc_info=True)
            continue
        if not isinstance(models, dict):
            logger.warning("Model source %s listed its models as %s", source.id, type(models))
            continue
        for alias, entry in models.items():
            if isinstance(alias, str) and isinstance(entry, dict):
                merged[alias] = entry
    return merged


__all__ = [
    "ENTRY_POINT_GROUP",
    "ModelSource",
    "find_source",
    "model_sources",
    "reset_model_sources",
    "source_models",
]
