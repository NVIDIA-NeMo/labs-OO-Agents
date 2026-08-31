# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-neutral discovery and dispatch for coding-agent skill commands."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from nooa.skill import Skill, get_slash_commands
from nooa.slash_dispatch import SlashCommandResult, parse_typed_args

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CodingSlashCommand:
    """One user-invocable command exposed by a loaded coding skill."""

    name: str
    description: str
    argument_hint: str | None
    completions: tuple[str, ...]
    output_to_agent: bool
    _method: Any = field(repr=False, compare=False)


class CodingSlashCommandRegistry:
    """Discover and invoke ``@slash_command`` methods without UI coupling."""

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self._commands: dict[str, CodingSlashCommand] = {}
        self._on_change: Callable[[tuple[CodingSlashCommand, ...]], None] | None = None
        self._previous_registry = getattr(agent, "_command_registry", None)
        agent._command_registry = self
        self.refresh_skill_commands()

    def commands(self) -> tuple[CodingSlashCommand, ...]:
        return tuple(self._commands[name] for name in sorted(self._commands))

    def get(self, name: str) -> CodingSlashCommand | None:
        return self._commands.get(name.lower())

    def set_on_change(
        self,
        callback: Callable[[tuple[CodingSlashCommand, ...]], None] | None,
        *,
        emit: bool = False,
    ) -> None:
        self._on_change = callback
        if emit and callback is not None:
            callback(self.commands())

    def refresh_skill_commands(self) -> None:
        """Rebuild the command map after skill load, activation, or reload."""
        previous = self.commands()
        commands: dict[str, CodingSlashCommand] = {}
        skills = getattr(self.agent, "skills", None)
        loaded = skills.loaded() if skills is not None and hasattr(skills, "loaded") else []
        for skill_name in loaded:
            try:
                skill = skills[skill_name]
            except (AttributeError, KeyError):
                continue
            if not isinstance(skill, Skill):
                continue
            for meta, method in get_slash_commands(skill):
                name = meta.name.lower()
                if name in commands:
                    logger.warning(
                        "Ignoring duplicate slash command /%s from skill %s",
                        name,
                        skill_name,
                    )
                    continue
                doc = inspect.getdoc(method) or ""
                description = doc.splitlines()[0] if doc else f"Run /{name}"
                commands[name] = CodingSlashCommand(
                    name=name,
                    description=description,
                    argument_hint=meta.argument_hint,
                    completions=tuple(meta.completions),
                    output_to_agent=meta.output_to_agent,
                    _method=method,
                )
        self._commands = commands
        if self._on_change is not None and self.commands() != previous:
            self._on_change(self.commands())

    async def invoke(self, name: str, raw_args: str) -> SlashCommandResult:
        command = self.get(name)
        if command is None:
            raise KeyError(name)
        kwargs = parse_typed_args(command._method, raw_args)
        # Commands execute on the agent's event loop, matching the native TUI.
        # Several existing synchronous skills call QueueManager.spawn(), which
        # requires that loop. Async commands can cancel cooperatively; a truly
        # blocking sync command needs the future per-agent process boundary to
        # provide a safe kill mechanism.
        value = command._method(**kwargs)
        if inspect.isawaitable(value):
            value = await value
        return SlashCommandResult(
            command=command.name,
            args=raw_args,
            value=value,
            text=str(value) if value is not None else None,
            output_to_agent=command.output_to_agent,
        )

    def close(self) -> None:
        self._on_change = None
        if getattr(self.agent, "_command_registry", None) is self:
            if self._previous_registry is None:
                delattr(self.agent, "_command_registry")
            else:
                self.agent._command_registry = self._previous_registry


__all__ = ["CodingSlashCommand", "CodingSlashCommandRegistry"]
