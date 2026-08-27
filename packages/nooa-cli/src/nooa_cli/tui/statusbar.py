# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structured statusbar items, including capabilities supplied by loaded skills."""

from __future__ import annotations

import datetime
import logging
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
StatusbarProvider = Callable[["StatusbarContext"], Any]
_SELF_PATH = re.compile(r"self(?:\.[A-Za-z][A-Za-z0-9_]*)+")


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
    """Registry of built-ins and statusbar capabilities from loaded skills."""

    def __init__(self, agent: Any = None) -> None:
        self._agent = agent
        self._providers: dict[str, StatusbarProvider] = {}
        self.refresh_skills()

    def register(self, name: str, provider: StatusbarProvider) -> None:
        """Register *provider* under a case-insensitive, whitespace-free name."""
        normalized = name.strip().lower()
        if not normalized or any(character.isspace() for character in normalized):
            raise ValueError(f"Invalid statusbar item name {name!r}")
        if not callable(provider):
            raise TypeError(f"Statusbar provider {name!r} must be callable")
        self._providers[normalized] = provider

    def refresh_skills(self) -> None:
        """Rebuild providers from built-ins and all currently loaded skills."""
        self._providers = {
            "time": lambda _: datetime.datetime.now().strftime("%H:%M"),
            "model": lambda context: _short_model_name(context.model),
            "cwd": lambda context: context.working_directory.name or str(context.working_directory),
            "context": lambda context: context.context_usage,
            "session": _session_label,
        }
        skills = getattr(self._agent, "skills", None)
        if skills is None or not callable(getattr(skills, "loaded", None)):
            return
        for skill_name in skills.loaded():
            try:
                skill = skills[skill_name]
                items = getattr(skill, "statusbar_items", None)
            except Exception:
                logger.debug(
                    "Could not inspect statusbar capability for %r", skill_name, exc_info=True
                )
                continue
            if not isinstance(items, Mapping):
                continue
            for item_name, value in items.items():
                try:
                    self.register(str(item_name), _skill_provider(skill, value))
                except (TypeError, ValueError):
                    logger.warning("Invalid statusbar item %r from skill %r", item_name, skill_name)

    def names(self) -> tuple[str, ...]:
        """Return available item names in stable order."""
        return tuple(sorted(self._providers))

    def accepts(self, name: str) -> bool:
        """Whether a configured name is a provider or a safe ``self.x`` path."""
        return name.lower() in self._providers or _SELF_PATH.fullmatch(name) is not None

    def render(self, names: Iterable[str], context: StatusbarContext) -> str:
        """Render configured providers, isolating missing or broken capabilities."""
        values: list[str] = []
        for raw_name in names:
            name = raw_name.lower()
            provider = self._providers.get(name)
            try:
                if provider is not None:
                    value = provider(context)
                elif _SELF_PATH.fullmatch(raw_name):
                    value = _resolve_self_path(context.agent, raw_name)
                else:
                    continue
            except Exception:
                logger.debug("Statusbar item %r failed", raw_name, exc_info=True)
                continue
            if value is not None and value != "":
                values.append(str(value))
        return " · ".join(values)


def _skill_provider(skill: Any, value: Any) -> StatusbarProvider:
    if callable(value):
        # Skill capability callbacks are intentionally simple and synchronous.
        return value
    if isinstance(value, str) and _SELF_PATH.fullmatch(value):
        return lambda context: _resolve_self_path(context.agent, value)
    raise TypeError("skill statusbar values must be callables or safe self.<attribute> paths")


def _resolve_self_path(agent: Any, path: str) -> Any:
    """Resolve a public dotted agent attribute without evaluating expressions."""
    if agent is None or _SELF_PATH.fullmatch(path) is None:
        raise ValueError(f"Unsafe statusbar attribute path {path!r}")
    value = agent
    for segment in path.split(".")[1:]:
        if segment.startswith("_"):
            raise ValueError(f"Private statusbar attribute path {path!r}")
        value = getattr(value, segment)
    return value


def _short_model_name(model: str) -> str:
    return model.split("/")[-1].replace("claude-", "")


def _session_label(context: StatusbarContext) -> str:
    short_id = (context.session_id or "")[:8]
    if context.session_title and short_id:
        return f"{context.session_title} [{short_id}]"
    return f"[{short_id}]" if short_id else ""
