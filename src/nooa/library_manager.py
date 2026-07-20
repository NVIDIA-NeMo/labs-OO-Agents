# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LibraryManager — scan, load, and hot-reload persistent agent libraries."""

import importlib
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nooa.skill import Skill

logger = logging.getLogger(__name__)


class LibraryManager:
    """Scans a libs directory and attaches each Python package as an agent attribute.

    Each subdirectory containing a pyproject.toml is imported and attached
    as ``agent.<lib_name>`` (i.e. ``self.<lib_name>`` from the agent's
    perspective). Libraries should export a ``Skill`` subclass in their
    ``__init__.py`` — this gives the agent full method discovery via
    ``doc(self.<lib_name>)``.

    Libraries that don't export a Skill get a ``Skill(module)`` fallback
    that exposes the module's docstring and attributes.

    Library modules are NOT injected as globals — the agent accesses them
    exclusively through ``self.<lib_name>``.

        mgr = LibraryManager.install(agent, libs_dir=Path("libs"))
        # agent.stats, agent.utils, ... are now set as self.stats, self.utils

        mgr.reload()              # reload every installed library

        LibraryManager.discover(path)  # list library names without loading
    """

    def __init__(self, agent: Any, libs_path: Path) -> None:
        self._agent = agent
        self._libs_path = libs_path
        self._installed: list[str] = []

    @classmethod
    def install(cls, agent: Any, *, libs_dir: Path) -> "LibraryManager":
        """Scan libs_dir and attach all libraries to *agent*. Returns the manager."""
        manager = cls(agent, libs_dir)
        manager._scan()
        return manager

    def _scan(self) -> None:
        if not self._libs_path.is_dir():
            return
        for lib_dir in sorted(self._libs_path.iterdir()):
            if not (lib_dir.is_dir() and (lib_dir / "pyproject.toml").exists()):
                continue
            lib_name = lib_dir.name
            if hasattr(self._agent, lib_name):
                continue
            try:
                self._attach(lib_dir, lib_name)
                self._installed.append(lib_name)
            except Exception:
                logger.warning("Library %s skipped", lib_name, exc_info=True)

    def _reload(self, lib_name: str) -> "Skill":
        """Re-import and re-attach a library from disk."""
        lib_dir = self._libs_path / lib_name
        return self._attach(lib_dir, lib_name)

    def _import_module(self, lib_dir: Path) -> Any:
        """Cache-bust and (re-)import a library package from disk."""
        lib_name = lib_dir.name
        prefix = lib_name + "."
        for key in [k for k in sys.modules if k == lib_name or k.startswith(prefix)]:
            del sys.modules[key]
        return importlib.import_module(lib_name)

    def _attach(self, lib_dir: Path, lib_name: str) -> "Skill":
        """Import the library and attach the best Skill handle to the agent.

        Resolution:
        1. Cache-bust and import the module.
        2. Look for a user-defined ``Skill`` export (instance or subclass).
        3. Fall back to ``Skill(module, name=lib_name)`` if none found.

        The Skill (or fallback wrapper) is set as ``agent.<lib_name>``
        so the agent accesses it via ``self.<lib_name>``.
        """
        from nooa.skill import Skill
        from nooa.skill_registry import skill_from_module

        module = self._import_module(lib_dir)

        # Prefer a user-defined Skill export.
        user_skill = skill_from_module(module, lib_name, source=f"Library {lib_name!r}")
        if user_skill is not None:
            attached: Skill = user_skill
            logger.info("Library %s: attached %s", lib_name, type(user_skill).__name__)
        else:
            attached = Skill(module, name=lib_name)
            logger.info("Library %s: no Skill export, using Skill(module) fallback", lib_name)

        attached._source_dir = lib_dir
        setattr(self._agent, lib_name, attached)
        attached.attach(self._agent)
        # Hot-reload slash commands: if the TUI's CommandRegistry is reachable,
        # re-discover @slash_command methods so new commands are immediately available.
        registry = getattr(self._agent, "_command_registry", None)
        if registry is not None and hasattr(registry, "refresh_skill_commands"):
            try:
                registry.refresh_skill_commands()
            except Exception:
                pass
        return attached

    def reload(self) -> None:
        """Reload all installed libraries from disk."""
        for lib_name in self._installed:
            try:
                self._reload(lib_name)
            except Exception:
                logger.warning("Reload failed for %s", lib_name, exc_info=True)

    @staticmethod
    def discover(path: Path) -> list[str]:
        """Return sorted library names (directories with a pyproject.toml) under *path*."""
        return sorted(p.parent.name for p in path.glob("*/pyproject.toml") if p.parent.is_dir())
