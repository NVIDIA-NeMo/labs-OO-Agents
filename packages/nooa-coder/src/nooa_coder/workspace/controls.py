# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavior controls shared by native and ACP presentation adapters."""

from __future__ import annotations

import shlex
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
        workspace: Path | None = None,
        command_registry: Any = None,
    ):
        self.agent = agent
        self.config = config
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

    def _persist_setting(self, field: str, value: object) -> Path:
        from .settings import write_settings_updates

        path, _ = write_settings_updates({("coding", field): value}, workspace=self.workspace)
        return path

    def _project_scope_settings(self) -> dict[str, Any]:
        """Read only the project-scope settings.yaml, never the user/env layers.

        _persist_settings()/_persist_setting() always write project scope.
        Basing a project-scope write on the fully layered user+project merge
        (as read elsewhere for display) would copy a user's own personal
        settings.yaml entries into the shared, committed project file.
        """
        import yaml

        from .settings import settings_path

        path = settings_path("project", workspace=self.workspace)
        if not path.exists():
            return {}
        try:
            data = yaml.safe_load(path.read_text())
        except (OSError, UnicodeError, yaml.YAMLError):
            return {}
        return data if isinstance(data, dict) else {}

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
        if self.workspace is None:
            cwd = getattr(self.agent, "cwd", None)
            if isinstance(cwd, (str, Path)):
                self.workspace = Path(cwd)
        # Another session (or the agent's persistence skill) may have saved
        # choices since this command's configuration was constructed.
        # activate/deactivate/add always write project scope (see
        # _persist_settings), so the basis for computing what to write must
        # be project scope alone, not the fully layered user+project merge
        # used for display elsewhere -- building the new list from the
        # layered merge and writing it back to project scope would copy a
        # user's own personal settings.yaml entries into the shared,
        # committed project file.
        from .settings import resolve_behavior_settings

        project_saved = resolve_behavior_settings(self._project_scope_settings())

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
                    for item in project_saved.get("additional_skills_dirs", [])
                )
            )
            if path not in persisted:
                persisted.append(path)
            self.config.additional_skills_dirs = list(
                dict.fromkeys([*self.config.additional_skills_dirs, path])
            )
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
            active = list(dict.fromkeys([*project_saved.get("active_skills", []), skill_id]))
            inactive = [
                name for name in project_saved.get("inactive_skills", []) if name != skill_id
            ]
            self.config.active_skills = list(dict.fromkeys([*self.config.active_skills, skill_id]))
            self.config.inactive_skills = [
                name for name in self.config.inactive_skills if name != skill_id
            ]
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
        active = [name for name in project_saved.get("active_skills", []) if name != skill_id]
        inactive = list(dict.fromkeys([*project_saved.get("inactive_skills", []), skill_id]))
        self.config.active_skills = [name for name in self.config.active_skills if name != skill_id]
        self.config.inactive_skills = list(dict.fromkeys([*self.config.inactive_skills, skill_id]))
        try:
            self._persist_settings({"active_skills": active, "inactive_skills": inactive})
        except Exception as exc:
            return ControlResult.ok(
                ControlMessage(f"Skill `{skill_id}` deactivated", "success"),
                ControlMessage(f"Could not save skill deactivation: {exc}", "warning"),
            )
        return ControlResult.ok(ControlMessage(f"Skill `{skill_id}` deactivated", "success"))


class MCPControl(BehaviorControl):
    """Review, approve, and revoke MCP server configurations."""

    @property
    def name(self) -> str:
        return "mcp"

    @classmethod
    def help_text(cls) -> dict[str, str]:
        return {"/mcp [status|approve NAME [CODE]|revoke NAME]": cls.__doc__ or ""}

    def validate_args(self, args: list[str]) -> tuple[bool, str | None]:
        if not args or (args[0] == "status" and len(args) == 1):
            return True, None
        if args[0] == "approve" and len(args) in (2, 3):
            return True, None
        if args[0] == "revoke" and len(args) == 2:
            return True, None
        return False, "Usage: /mcp [status|approve NAME [CODE]|revoke NAME]"

    async def execute(self, args: list[str]) -> ControlResult:
        from .mcp_approval import _safe_display
        from .mcp_registry import MCPRegistry

        registry = getattr(self.agent, "mcp", None)
        if not isinstance(registry, MCPRegistry):
            return ControlResult.err("This agent has no MCP registry.")
        registry.refresh_settings()
        if not args or args[0] == "status":
            rows = []
            for name in registry.discovered():
                try:
                    approval = "approved" if registry._is_approved(name) else "approval required"
                except ValueError:
                    # build_approval_request() raises for an invalid/unsupported
                    # entry; one bad server config must not blank out /mcp
                    # status for every other, valid server.
                    approval = "invalid configuration"
                rows.append(
                    [
                        _safe_display(name),
                        approval,
                        "connected" if name in registry.connected() else "disconnected",
                    ]
                )
            return ControlResult.ok(
                ControlTable(
                    columns=["Server", "Approval", "Connection"], rows=rows, title="MCP servers"
                )
            )
        name = args[1]
        # Registry APIs accept globs; approval commands name one exact definition.
        pattern = "".join({"[": "[[]", "*": "[*]", "?": "[?]"}.get(c, c) for c in name)
        if args[0] == "revoke":
            # Revoke before disconnect so a transport failure cannot retain permission.
            registry._revoke_approvals(name)
            await registry.disconnect([pattern])
            return ControlResult.ok(
                ControlMessage(f"Revoked approvals for {_safe_display(name)}.", "success")
            )
        if len(args) == 2:
            return ControlResult.ok(ControlMessage(registry._approval_request(name).review_text()))
        registry._approve(name, args[2])
        await registry.connect([pattern])
        return ControlResult.ok(
            ControlMessage(f"Approved and connected to {_safe_display(name)}.", "success")
        )


CONTROL_TYPES = {
    "skills": SkillsControl,
    "mcp": MCPControl,
}


def behavior_commands(agent: Any, config: Any, *, workspace: Path, command_registry: Any):
    """Adapt shared operations to the host-neutral command catalog."""
    from nooa_coder.coding.slash_commands import CodingSlashCommand

    result = []
    for control_type in CONTROL_TYPES.values():
        control = control_type(
            agent,
            config,
            workspace=workspace,
            command_registry=command_registry,
        )
        result.append(
            CodingSlashCommand(
                name=control.name,
                description=control_type.__doc__ or "",
                argument_hint={
                    "skills": "<list|commands|add DIR|activate ID|deactivate ID>",
                    "mcp": "[status|approve NAME [CODE]|revoke NAME]",
                }[control.name],
                output_to_agent=False,
                is_control=True,
                _method=control.invoke,
            )
        )
    return result
