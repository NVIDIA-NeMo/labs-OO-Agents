# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structured, extensible statusbar items for the terminal UI.

Plugins expose a callable through the ``nooa_cli.tui.statusbar_items`` entry-point
group. The entry-point name is the item name and the callable receives a
:class:`StatusbarContext`; it returns display text or ``None`` to hide itself.
"""

from __future__ import annotations

import datetime
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STATUSBAR_ENTRY_POINT = "nooa_cli.tui.statusbar_items"
LEGACY_TOOLBAR_ENTRY_POINT = "nooa_cli.tui.toolbar_items"
StatusbarProvider = Callable[["StatusbarContext"], str | None]


@dataclass(frozen=True, slots=True)
class StatusbarContext:
    """Read-only state supplied to each statusbar provider."""

    model: str
    working_directory: Path
    context_usage: str
    session_id: str | None = None
    session_title: str | None = None
    agent: Any = None


class StatusbarRegistry:
    """Registry of named built-in and installed statusbar providers."""

    def __init__(self, *, load_plugins: bool = True) -> None:
        self._providers: dict[str, StatusbarProvider] = {
            "time": lambda _: datetime.datetime.now().strftime("%H:%M"),
            "model": lambda context: _short_model_name(context.model),
            "cwd": lambda context: context.working_directory.name or str(context.working_directory),
            "context": lambda context: context.context_usage,
            "session": _session_label,
        }
        if load_plugins:
            self._load_plugins()

    def register(self, name: str, provider: StatusbarProvider) -> None:
        """Register *provider* under a case-insensitive, whitespace-free name."""
        normalized = name.strip().lower()
        if not normalized or any(character.isspace() for character in normalized):
            raise ValueError(f"Invalid statusbar item name {name!r}")
        if not callable(provider):
            raise TypeError(f"Statusbar provider {name!r} must be callable")
        self._providers[normalized] = provider

    def names(self) -> tuple[str, ...]:
        """Return available item names in stable order."""
        return tuple(sorted(self._providers))

    def render(self, names: Iterable[str], context: StatusbarContext) -> str:
        """Render configured providers, isolating missing or broken plugins."""
        values: list[str] = []
        for name in names:
            provider = self._providers.get(name)
            if provider is None:
                continue
            try:
                value = provider(context)
            except Exception:
                logger.debug("Statusbar item %r failed", name, exc_info=True)
                continue
            if value:
                values.append(str(value))
        return " · ".join(values)

    def _load_plugins(self) -> None:
        # Load the deprecated group first so the current group deterministically
        # replaces providers with the same normalized name.
        for group in (LEGACY_TOOLBAR_ENTRY_POINT, STATUSBAR_ENTRY_POINT):
            try:
                plugins = entry_points(group=group)
            except Exception:
                logger.debug("Statusbar entry-point discovery failed for %r", group, exc_info=True)
                continue
            for plugin in plugins:
                try:
                    self.register(plugin.name, plugin.load())
                except Exception:
                    logger.warning(
                        "Statusbar provider %r could not be loaded", plugin.name, exc_info=True
                    )


def _short_model_name(model: str) -> str:
    return model.split("/")[-1].replace("claude-", "")


def _session_label(context: StatusbarContext) -> str:
    short_id = (context.session_id or "")[:8]
    if context.session_title and short_id:
        return f"{context.session_title} [{short_id}]"
    return f"[{short_id}]" if short_id else ""
