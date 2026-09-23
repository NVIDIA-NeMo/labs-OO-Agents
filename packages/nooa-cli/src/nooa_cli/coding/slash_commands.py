# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-neutral discovery and dispatch for coding-agent skill commands."""

from __future__ import annotations

import inspect
import logging
import shlex
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nooa.skill import Skill
from nooa.slash_dispatch import SlashCommandResult, parse_typed_args

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CodingSlashCommand:
    """Metadata for a user-invocable skill slash command."""

    name: str
    description: str
    body: str = ""
    argument_hint: str | None = None
    completions: tuple[str, ...] = ()
    output_to_agent: bool = True
    is_control: bool = False
    _method: Any = field(default=None, repr=False)

    def help_entry(self) -> tuple[str, str]:
        hint = self.argument_hint or ""
        key = f"/{self.name} {hint}".strip()
        return key, self.description

    def make_agent_message(self, args: list[str]) -> str:
        body = self.body
        if args:
            joined = " ".join(args)
            if "$ARGUMENTS" in body:
                return body.replace("$ARGUMENTS", joined)
            return f"{body}\n\nArguments: {joined}"
        return body


# Shared behavior names stay reserved across hosts. Presentation-only commands
# belong in each host's explicit `reserved` argument.
RESERVED_COMMAND_NAMES = frozenset(
    "help exit quit clear compact context edit connect model models skills session resume restart jobs events todos mcp reasoning".split()
)


def discover_markdown_commands(skills_dirs, reserved=RESERVED_COMMAND_NAMES):
    skills: dict[str, CodingSlashCommand] = {}
    if not skills_dirs:
        return skills
    try:
        import yaml
    except ImportError:
        return skills
    for skills_dir in skills_dirs:
        skills_dir = Path(skills_dir)
        if not skills_dir.is_dir():
            continue
        for skill_md in sorted(skills_dir.rglob("SKILL.md")):
            entry = skill_md.parent
            try:
                content = skill_md.read_text(encoding="utf-8")
                if not content.startswith("---"):
                    continue
                parts = content.split("---", 2)
                if len(parts) < 3:
                    continue
                try:
                    meta = yaml.safe_load(parts[1]) or {}
                    if not isinstance(meta, dict):
                        raise ValueError("not a mapping")
                except Exception:
                    # Fallback: line-by-line regex for invalid-YAML values like
                    # argument-hint: "<action>" [issue-id]  (Claude Code style).
                    # Parse each scalar individually so "false" → False (not "false").
                    import re

                    meta = {}
                    for line in parts[1].splitlines():
                        m = re.match(r"^([a-zA-Z][a-zA-Z0-9_-]*):\s*(.+)$", line)
                        if not m:
                            continue
                        raw = m.group(2).strip()
                        try:
                            parsed = yaml.safe_load(raw)
                        except Exception:
                            parsed = raw
                        # Keep the raw text for anything but the scalar/list
                        # shapes the callers below actually handle (bool/str/
                        # None display as-is; a list gets bracket-notation
                        # reconstruction). A dict (e.g. "description: Fix: the
                        # bug" parsing as {"Fix": "the bug"}) would otherwise
                        # surface its Python repr instead of the source text.
                        if isinstance(parsed, bool | str | list) or parsed is None:
                            meta[m.group(1)] = parsed
                        else:
                            meta[m.group(1)] = raw
                if not isinstance(meta, dict):
                    continue
                # CC convention: user-invocable defaults to true.
                # Opt out with user-invocable: false.
                # install-as: command is honored for backward compat.
                if meta.get("user-invocable") is False:
                    continue
                raw_name = str(meta.get("name") or "").strip()
                cmd_name = raw_name.lower()
                if not cmd_name or cmd_name in reserved or cmd_name in skills:
                    continue
                description = str(meta.get("description", "")).strip()
                body = parts[2].strip()
                hint = meta.get("argument-hint")
                if isinstance(hint, list):
                    # YAML parses [label] as a list; reconstruct bracket notation
                    hint = "[" + ", ".join(str(x) for x in hint) + "]"
                elif hint is not None:
                    hint = str(hint)
                skills[cmd_name] = CodingSlashCommand(
                    name=cmd_name,
                    body=body,
                    description=description,
                    argument_hint=hint,
                )
            except Exception as e:
                logger.warning("Failed to load skill from %s: %s", entry, e)
    return skills


def discover_python_commands(agent, reserved=RESERVED_COMMAND_NAMES):
    skills: dict[str, CodingSlashCommand] = {}
    try:
        from nooa.skill import get_slash_commands
    except ImportError:
        return skills

    for attr_name in dir(agent):
        if attr_name.startswith("_"):
            continue
        try:
            obj = getattr(agent, attr_name)
        except Exception:
            continue
        if not isinstance(obj, Skill):
            continue
        for meta, method in get_slash_commands(obj):
            cmd_name = meta.name.lower()
            if cmd_name in reserved or cmd_name in skills:
                continue
            description = (method.__doc__ or "").strip().split("\n")[0]
            skills[cmd_name] = CodingSlashCommand(
                name=cmd_name,
                body="",
                description=description,
                argument_hint=meta.argument_hint,
                completions=getattr(meta, "completions", ()),
                output_to_agent=getattr(meta, "output_to_agent", True),
                _method=method,
            )
    return skills


class CodingSlashCommandRegistry:
    """Discover and invoke Markdown and Python skill commands without UI coupling."""

    def __init__(
        self, agent: Any, *, skills_dirs=(), reserved=(), controls=(), bind_registry: bool = True
    ) -> None:
        self.agent = agent
        self.skills_dirs = tuple(skills_dirs)
        self.controls = {command.name: command for command in controls}
        self.reserved = RESERVED_COMMAND_NAMES | frozenset(reserved)
        self._commands: dict[str, CodingSlashCommand] = {}
        self._on_change: Callable[[tuple[CodingSlashCommand, ...]], None] | None = None
        self._previous_registry = getattr(agent, "_command_registry", None)
        if bind_registry:
            agent._command_registry = self
        self.refresh_skill_commands()

    def commands(self) -> tuple[CodingSlashCommand, ...]:
        return tuple(self._commands[name] for name in sorted(self._commands))

    def skill_commands(self):
        return {name: command for name, command in self._commands.items() if not command.is_control}

    def add_skills_dir(self, path: Path) -> bool:
        path = path.expanduser().resolve()
        added = path not in self.skills_dirs
        if added:
            self.skills_dirs = (*self.skills_dirs, path)
        self.agent.skills.discover_skills_dirs([path])
        self.refresh_skill_commands()
        return added

    def set_controls(self, controls):
        self.controls = {command.name: command for command in controls}
        self.refresh_skill_commands()

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
        commands = discover_markdown_commands(self.skills_dirs, self.reserved)
        commands.update(discover_python_commands(self.agent, self.reserved))
        commands.update(self.controls)
        self._commands = commands
        if self._on_change is not None and self.commands() != previous:
            self._on_change(self.commands())

    async def invoke(self, name: str, raw_args: str) -> SlashCommandResult:
        command = self.get(name)
        if command is None:
            raise KeyError(name)
        return await invoke_skill_command(command, raw_args, agent=self.agent)

    def close(self) -> None:
        self._on_change = None
        if getattr(self.agent, "_command_registry", None) is self:
            if self._previous_registry is None:
                delattr(self.agent, "_command_registry")
            else:
                self.agent._command_registry = self._previous_registry


async def invoke_skill_command(
    command: CodingSlashCommand, raw_args: str, *, agent: Any
) -> SlashCommandResult:
    """Prepare a skill result on the agent loop with identical argument semantics."""
    from .mentions import expand_mentions

    if command._method is None:
        try:
            args = shlex.split(raw_args)
        except ValueError:
            args = raw_args.split()
        value = command.make_agent_message(args)
    else:
        kwargs = parse_typed_args(command._method, raw_args)
        value = command._method(**kwargs)
        if inspect.isawaitable(value):
            value = await value
    text = str(value) if value is not None else None
    if text is not None and command.output_to_agent:
        text = expand_mentions(text, base_dir=getattr(agent, "cwd", None))
    return SlashCommandResult(
        command=command.name,
        args=raw_args,
        value=value,
        text=text,
        output_to_agent=command.output_to_agent,
    )


__all__ = ["CodingSlashCommand", "CodingSlashCommandRegistry"]
