# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structured and extensible terminal toolbar items."""

from __future__ import annotations

import datetime
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TOOLBAR_ENTRY_POINT = "nooa_cli.tui.toolbar_items"
ToolbarProvider = Callable[["ToolbarContext"], str | None]


@dataclass(frozen=True, slots=True)
class ToolbarContext:
    model: str
    working_directory: Path
    context_usage: str
    session_id: str | None = None
    session_title: str | None = None
    agent: Any = None


class ToolbarRegistry:
    """Named toolbar providers with optional package entry-point extensions."""

    def __init__(self, *, load_plugins: bool = True) -> None:
        self._providers: dict[str, ToolbarProvider] = {
            "time": lambda _: datetime.datetime.now().strftime("%H:%M"),
            "model": lambda context: _short_model_name(context.model),
            "cwd": lambda context: context.working_directory.name or str(context.working_directory),
            "context": lambda context: context.context_usage,
            "session": _session_label,
        }
        if load_plugins:
            self._load_plugins()

    def register(self, name: str, provider: ToolbarProvider) -> None:
        normalized = name.strip().lower()
        if not normalized or any(character.isspace() for character in normalized):
            raise ValueError(f"Invalid toolbar item name {name!r}")
        self._providers[normalized] = provider

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))

    def render(self, names: Iterable[str], context: ToolbarContext) -> str:
        values: list[str] = []
        for name in names:
            provider = self._providers.get(name)
            if provider is None:
                continue
            try:
                value = provider(context)
            except Exception:
                logger.debug("Toolbar item %r failed", name, exc_info=True)
                continue
            if value:
                values.append(str(value))
        return " · ".join(values)

    def _load_plugins(self) -> None:
        try:
            plugins = entry_points(group=TOOLBAR_ENTRY_POINT)
        except Exception:
            logger.debug("Toolbar entry-point discovery failed", exc_info=True)
            return
        for plugin in plugins:
            try:
                self.register(plugin.name, plugin.load())
            except Exception:
                logger.warning(
                    "Toolbar provider %r could not be loaded", plugin.name, exc_info=True
                )


def _short_model_name(model: str) -> str:
    return model.split("/")[-1].replace("claude-", "")


def _session_label(context: ToolbarContext) -> str:
    short_id = (context.session_id or "")[:8]
    if context.session_title and short_id:
        return f"{context.session_title} [{short_id}]"
    return f"[{short_id}]" if short_id else ""
