# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavior controls shared by native and ACP presentation adapters."""

from __future__ import annotations

import inspect
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ControlMessage:
    content: str
    style: str = "info"


@dataclass(frozen=True)
class ControlTable:
    columns: list[str]
    rows: list[list[str]]
    title: str

    @property
    def content(self) -> str:
        lines = [self.title, " | ".join(self.columns)]
        lines.extend(" | ".join(row) for row in self.rows)
        return "\n".join(lines)


@dataclass(frozen=True)
class ControlResult:
    success: bool
    outputs: tuple[ControlMessage | ControlTable, ...]

    @classmethod
    def ok(cls, *outputs: ControlMessage | ControlTable) -> ControlResult:
        return cls(True, outputs)

    @classmethod
    def err(cls, message: str) -> ControlResult:
        return cls(False, (ControlMessage(message, "error"),))

    def __str__(self) -> str:
        return "\n".join(output.content for output in self.outputs)


class BehaviorControl:
    def __init__(
        self,
        agent: Any,
        config: Any,
        *,
        configure_memory: Callable | None = None,
        workspace: Path | None = None,
        command_registry: Any = None,
    ):
        self.agent = agent
        self.config = config
        self._configure_memory = configure_memory
        self.workspace = workspace
        self._registry = command_registry

    @property
    def skills_dirs(self):
        return getattr(self._registry, "skills_dirs", self.config.skills_dirs)

    def _persist_settings(self, updates: dict[str, object]) -> Path:
        from .settings import write_settings_updates

        path, _ = write_settings_updates(
            {("coding", key): value for key, value in updates.items()}, workspace=self.workspace
        )
        return path

    async def agent_run_async(self, fn):
        # The host dispatches the whole operation onto the agent's owner loop.
        value = fn()
        return await value if inspect.isawaitable(value) else value

    def _configure(self):
        if self._configure_memory is None:
            raise RuntimeError("the session memory configuration is unavailable")
        return self._configure_memory()

    def _persist_setting(self, field: str, value: object) -> Path:
        from .settings import write_settings_updates

        path, _ = write_settings_updates({("coding", field): value}, workspace=self.workspace)
        return path

    def _persist_agent_preference(self, field: str, value: object) -> Path:
        from .settings import write_settings_updates

        key = getattr(self.agent, "_tui_memory_key", None)
        key = key or f"{type(self.agent).__module__}:{type(self.agent).__qualname__}"
        path, _ = write_settings_updates({("coding", field, key): value}, workspace=self.workspace)
        return path

    async def run(self, args: list[str]) -> ControlResult:
        valid, error = self.validate_args(args)
        if not valid:
            return ControlResult.err(error or "Invalid command arguments")
        try:
            return await self.execute(args)
        except Exception as exc:
            return ControlResult.err(f"/{self.name} failed: {exc}")

    async def invoke(self, args: str) -> ControlResult:
        try:
            parsed = shlex.split(args)
        except ValueError as exc:
            return ControlResult.err(f"/{self.name}: {exc}")
        return await self.run(parsed)


class SkillsControl(BehaviorControl):
    """Discover skills and save workspace activation preferences."""

    @property
    def name(self) -> str:
        return "skills"

    @classmethod
    def help_text(cls) -> dict[str, str]:
        return {
            "/skills <list|add DIR|commands|activate ID|deactivate ID>": (
                "List and manage skills or show their slash commands"
            ),
        }

    def validate_args(self, args: list[str]) -> tuple[bool, str | None]:
        if not args:
            return False, "Usage: /skills <list|add|activate|deactivate|commands>"
        if args[0].lower() not in ("list", "add", "activate", "deactivate", "commands"):
            return False, f"Unknown subcommand `{args[0]}`"
        if args[0].lower() == "add" and len(args) != 2:
            return False, "Usage: /skills add <directory>"
        if args[0].lower() in ("activate", "deactivate") and len(args) != 2:
            return False, f"Usage: /skills {args[0]} <skill_id>"
        return True, None

    async def execute(self, args: list[str]) -> ControlResult:
        subcmd = args[0].lower()
        subargs = args[1:]

        if subcmd == "add":
            if self._registry is None:
                return ControlResult.err("The command registry is unavailable.")
            raw_path = Path(subargs[0]).expanduser()
            base = Path(getattr(self.agent, "cwd", Path.cwd()))
            path = (base / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve()
            if not path.is_dir():
                return ControlResult.err(f"Skills directory not found: {path}")

            before_skills = set(
                getattr(getattr(self.agent, "skills", None), "discovered", lambda: [])()
            )
            before_commands = set(self._registry.skill_commands())
            try:
                added = self._registry.add_skills_dir(path)
            except Exception as exc:
                return ControlResult.err(f"Failed to add skills directory {path}: {exc}")

            persisted = list(
                dict.fromkeys(
                    (base / Path(item).expanduser()).resolve()
                    for item in getattr(self.config, "additional_skills_dirs", [])
                )
            )
            if path not in persisted:
                persisted.append(path)
                self.config.additional_skills_dirs = persisted
            try:
                settings_path = self._persist_setting(
                    "additional_skills_dirs", [str(item) for item in persisted]
                )
            except Exception as exc:
                return ControlResult.ok(
                    ControlMessage(f"Added skills directory: {path}", "success"),
                    ControlMessage(f"Could not save the skills directory: {exc}", "warning"),
                )

            after_skills = set(
                getattr(getattr(self.agent, "skills", None), "discovered", lambda: [])()
            )
            after_commands = set(self._registry.skill_commands())
            detail = (
                f"Discovered {len(after_skills - before_skills)} skill(s) and "
                f"{len(after_commands - before_commands)} slash command(s)."
            )
            verb = "Added" if added else "Already using"
            return ControlResult.ok(
                ControlMessage(f"{verb} skills directory: {path}", "success"),
                ControlMessage(detail, "info"),
                ControlMessage(f"Saved in {settings_path}", "status"),
            )

        if subcmd == "commands":
            user_skills = self._registry.skill_commands() if self._registry else {}
            rows_cmd = [
                [f"/{name}", skill.argument_hint or "", skill.description]
                for name, skill in sorted(user_skills.items())
            ]
            if rows_cmd:
                return ControlResult.ok(
                    ControlTable(
                        columns=["Command", "Args", "Description"],
                        rows=rows_cmd,
                        title="Skill slash commands",
                    ),
                    ControlMessage(f"Searched: {self.skills_dirs}", "status"),
                )
            return ControlResult.ok(
                ControlMessage("No user-invocable skill commands found.", "info"),
                ControlMessage(f"Searched: {self.skills_dirs}", "status"),
            )

        from nooa.skill_registry import SkillRegistry

        registry = getattr(self.agent, "skills", None)
        if not isinstance(registry, SkillRegistry):
            return ControlResult.err(
                "Agent has no SkillRegistry. Skills require self.skills = SkillRegistry(self)."
            )

        if subcmd == "list":
            all_names = registry.discovered()
            activated = set(registry.activated())
            if not all_names:
                return ControlResult.ok(ControlMessage("No skills found", "info"))
            rows = [[name, "\u2713" if name in activated else "", ""] for name in all_names]
            return ControlResult.ok(
                ControlTable(columns=["ID", "Active", "Description"], rows=rows, title="Skills"),
            )

        if subcmd == "activate":
            skill_id = subargs[0]
            if skill_id not in registry.discovered():
                return ControlResult.err(f"Skill `{skill_id}` not found. Use /skills list.")
            try:
                if skill_id not in registry.activated():
                    registry.activate([skill_id])
            except Exception as e:
                return ControlResult.err(f"Failed to activate `{skill_id}`: {e}")
            if skill_id not in registry.activated():
                return ControlResult.err(f"Failed to activate `{skill_id}`")
            active = list(dict.fromkeys([*self.config.active_skills, skill_id]))
            inactive = [name for name in self.config.inactive_skills if name != skill_id]
            self.config.active_skills = active
            self.config.inactive_skills = inactive
            try:
                self._persist_settings({"active_skills": active, "inactive_skills": inactive})
            except Exception as exc:
                return ControlResult.ok(
                    ControlMessage(f"Skill `{skill_id}` activated", "success"),
                    ControlMessage(f"Could not save skill activation: {exc}", "warning"),
                )
            return ControlResult.ok(ControlMessage(f"Skill `{skill_id}` activated", "success"))

        # deactivate
        skill_id = subargs[0]
        if skill_id not in registry.discovered():
            return ControlResult.err(f"Skill `{skill_id}` not found. Use /skills list.")
        try:
            if skill_id in registry.activated():
                registry.deactivate([skill_id])
        except Exception as e:
            return ControlResult.err(f"Failed to deactivate `{skill_id}`: {e}")
        if skill_id in registry.activated():
            return ControlResult.err(f"Failed to deactivate `{skill_id}`")
        active = [name for name in self.config.active_skills if name != skill_id]
        inactive = list(dict.fromkeys([*self.config.inactive_skills, skill_id]))
        self.config.active_skills = active
        self.config.inactive_skills = inactive
        try:
            self._persist_settings({"active_skills": active, "inactive_skills": inactive})
        except Exception as exc:
            return ControlResult.ok(
                ControlMessage(f"Skill `{skill_id}` deactivated", "success"),
                ControlMessage(f"Could not save skill deactivation: {exc}", "warning"),
            )
        return ControlResult.ok(ControlMessage(f"Skill `{skill_id}` deactivated", "success"))


_MEMORY_MODES = {"on": "project", "local": "session", "off": "off"}
_SCOPE_LABELS = {
    "project": "on (shared across sessions, project-wide)",
    "session": "local (this session only)",
    "off": "off",
}


class MemoryControl(BehaviorControl):
    """Configure long-term memory for this agent."""

    @property
    def name(self) -> str:
        return "memory"

    def help_text(self) -> dict[str, str]:  # type: ignore[override]
        return {
            "/memory [on|local|off]": (
                "Configure long-term memory: on shares a project store; "
                f"local uses this session (currently {self._scope_label()})"
            )
        }

    def validate_args(self, args: list[str]) -> tuple[bool, str | None]:
        if len(args) > 1 or (args and args[0].lower() not in {*_MEMORY_MODES, "status"}):
            return False, "Usage: /memory [on|local|off]"
        return True, None

    def _agent_key(self) -> str:
        return getattr(
            self.agent,
            "_tui_memory_key",
            f"{type(self.agent).__module__}:{type(self.agent).__qualname__}",
        )

    def _scope(self) -> str:
        return self.config.memory_agents.get(self._agent_key(), self.config.memory)

    def _scope_label(self) -> str:
        scope = self._scope()
        return _SCOPE_LABELS.get(scope, scope)

    async def execute(self, args: list[str]) -> ControlResult:
        if not args or args[0].lower() == "status":
            line = f"Memory: {self._scope_label()}"
            skill = getattr(self.agent, "memory", None)
            manager = getattr(skill, "_mgr", None) if skill is not None else None
            if manager is not None:
                line += f" — you are {manager.owner} · store: {manager.store.path}"
            return ControlResult.ok(ControlMessage(line, "info"))

        scope = _MEMORY_MODES[args[0].lower()]
        self.config.memory = scope
        self.config.memory_agents[self._agent_key()] = scope
        self._persist_agent_preference("memory_agents", scope)
        try:
            await self.agent_run_async(self._configure)
        except Exception as exc:
            return ControlResult.err(f"Failed to configure memory: {exc}")

        if scope == "off":
            return ControlResult.ok(ControlMessage("Memory disabled for this agent.", "success"))
        return ControlResult.ok(
            ControlMessage(f"Memory {_SCOPE_LABELS[scope]} enabled for this agent.", "success")
        )


class ReflectionControl(MemoryControl):
    """Configure idle consolidation for the current agent's memory."""

    @property
    def name(self) -> str:
        return "reflection"

    def help_text(self) -> dict[str, str]:  # type: ignore[override]
        state = "on" if self._enabled() else "off"
        return {
            "/reflection [on|off|now]": (
                f"Configure idle memory reflection (currently {state}); now runs immediately"
            )
        }

    def validate_args(self, args: list[str]) -> tuple[bool, str | None]:
        if len(args) > 1 or (args and args[0].lower() not in {"on", "off", "status", "now"}):
            return False, "Usage: /reflection [on|off|now]"
        return True, None

    def _enabled(self) -> bool:
        return bool(self.config.reflection_agents.get(self._agent_key(), self.config.reflection))

    def _runner(self):
        return getattr(self.agent, "_tui_reflection_runner", None)

    def _status_output(self) -> ControlMessage:
        state = "on" if self._enabled() else "off"
        runner = self._runner()
        if runner is None:
            return ControlMessage(f"Idle reflection: {state} (memory is not attached)", "info")
        line = f"Idle reflection: {state} | dirty: {runner.dirty}"
        report = runner.last_report
        if report is not None:
            stopped = f"interrupted @ {report.stopped_in}, " if report.interrupted else ""
            line += (
                f" | last: merged {report.merged}, +{report.edges_added} edges, "
                f"rescored {report.rescored}, pruned {report.pruned}, "
                f"created {report.created} ({stopped}{report.duration_ms / 1000:.1f}s)"
            )
        return ControlMessage(line, "info")

    async def execute(self, args: list[str]) -> ControlResult:
        if not args or args[0].lower() == "status":
            return ControlResult.ok(self._status_output())

        if args[0].lower() == "now":
            runner = self._runner()
            if runner is None:
                return ControlResult.err("Memory is not attached. Enable it with /memory first.")
            started = await self.agent_run_async(runner.run_now)
            if not started:
                return ControlResult.ok(
                    ControlMessage("A reflection run is already pending.", "info")
                )
            return ControlResult.ok(
                ControlMessage("Reflection started; /reflection shows the report.", "success")
            )

        enabled = args[0].lower() == "on"
        if enabled and getattr(self.agent, "memory", None) is None:
            return ControlResult.err("Memory is not attached. Enable it with /memory first.")

        self.config.reflection = enabled
        self.config.reflection_agents[self._agent_key()] = enabled
        self._persist_agent_preference("reflection_agents", enabled)
        runner = self._runner()
        if runner is not None and not enabled:
            await self.agent_run_async(runner.interrupt)
        try:
            await self.agent_run_async(self._configure)
        except Exception as exc:
            return ControlResult.err(f"Failed to configure reflection: {exc}")
        state = "enabled" if enabled else "disabled"
        return ControlResult.ok(
            ControlMessage(f"Idle reflection {state} for this agent.", "success")
        )


CONTROL_TYPES = {
    "skills": SkillsControl,
    "memory": MemoryControl,
    "reflection": ReflectionControl,
}


def behavior_commands(
    agent: Any, config: Any, *, configure_memory: Callable, workspace: Path, command_registry: Any
):
    """Adapt shared operations to the host-neutral command catalog."""
    from nooa_cli.coding.slash_commands import CodingSlashCommand

    result = []
    for control_type in CONTROL_TYPES.values():
        control = control_type(
            agent,
            config,
            configure_memory=configure_memory,
            workspace=workspace,
            command_registry=command_registry,
        )
        result.append(
            CodingSlashCommand(
                name=control.name,
                description=control_type.__doc__ or "",
                argument_hint={
                    "skills": "<list|commands|add DIR|activate ID|deactivate ID>",
                    "memory": "[status|on|local|off]",
                    "reflection": "[status|on|off|now]",
                }[control.name],
                output_to_agent=False,
                is_control=True,
                _method=control.invoke,
            )
        )
    return result
