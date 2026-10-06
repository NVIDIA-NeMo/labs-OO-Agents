# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workspace-scoped agent settings shared by the native and ACP hosts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from nooa.interactive import DEFAULT_MODEL, SummarizationConfig
from nooa_atom.workspace.settings import load_skills_dirs


class AtomOptions(BaseModel):
    """Behavioral options; terminal presentation settings stay with the TUI.

    These are the Atom agent's own settings, read from the workspace and
    user settings files. They are not the Session layer's
    ``nooa_atom.session.options.SessionOptions``, which say how a session
    is built and run.
    """

    working_dir: str = "."
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig)
    agent_spec: str | None = None
    skills_dirs: list[Path] = Field(default_factory=list)
    additional_skills_dirs: list[Path] = Field(default_factory=list)
    default_model: str = DEFAULT_MODEL
    active_skills: list[str] = Field(default_factory=list)
    inactive_skills: list[str] = Field(default_factory=list)
    mcp_file: Path = Path(".mcp.json")
    mcp_servers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    mcp_auto_connect: list[str] = Field(default_factory=list)

    @classmethod
    def load(cls, workspace: str | Path, **overrides: Any) -> AtomOptions:
        """Load legacy ``tui`` and shared ``coding`` settings for this workspace.

        Invalid settings are reported and replaced by the defaults.
        """
        root = Path(workspace).expanduser().resolve()
        from .settings import load_behavior_settings

        values = load_behavior_settings(root)
        values.update({key: value for key, value in overrides.items() if value is not None})
        values["working_dir"] = str(root)
        values["skills_dirs"] = load_skills_dirs(root)
        return cls(**values)


async def connect_session_mcp(agent: Any, options: AtomOptions) -> list[str]:
    """Connect remembered servers, preserving exact-configuration approvals."""
    warnings = []
    for name in dict.fromkeys(options.mcp_auto_connect):
        try:
            await agent.skills.mcp.connect([name])
        except Exception as exc:
            warnings.append(f"MCP server {name!r} was not connected: {exc}")
    return warnings


def drop_stale_memory_context(agent: Any) -> None:
    """Drop memory prompts an older shared-host snapshot restored.

    The memory skill itself is excluded from snapshots, but its context
    blocks were not. Restoring is additive, so this runs after the restore
    (``AtomAgent.after_restore``). An agent that attached its own
    ``memory`` keeps them.
    """
    if not hasattr(agent, "memory"):
        for key in ("memory_system", "recalled_memories"):
            if key in agent.context:
                del agent.context[key]


def configure_session_skills(agent: Any, options: AtomOptions) -> list[str]:
    """Attach the workspace's MCP servers and explicit skills before resume events.

    Return actionable warnings for either host to display. Discovering a skill
    does not activate it; negative activation preferences override positives.
    """
    from nooa_atom.skills.mcp_servers import MCPServers
    from nooa_atom.workspace.workspace_settings import WorkspaceSettings

    skills = getattr(agent, "skills", None)
    if skills is None:
        return []
    root = Path(options.working_dir)
    mcp_file = options.mcp_file.expanduser()
    servers = MCPServers(
        mcp_file=mcp_file if mcp_file.is_absolute() else root / mcp_file,
        servers=options.mcp_servers,
        watch_settings=True,
        project_dir=root / ".nooa",
    )
    skills.set_mcp_servers(servers)
    skills.register("nooa.workspace_settings", WorkspaceSettings(options))
    skills.registry.activate(["nooa.workspace_settings"])
    warnings: list[str] = []
    discover = getattr(skills, "discover_skills_dirs", None)
    if options.active_skills and callable(discover):
        try:
            discover(options.skills_dirs)
        except Exception as exc:
            warnings.append(f"Could not discover configured skills: {exc}")
    # Saved names are registered names (nemo.web) or skill names (web).
    for name in options.active_skills:
        entry = skills.entry(name)
        if entry is None or entry.kind == "mcp":
            warnings.append(f"Configured skill not found: {name}")
            continue
        try:
            skills.registry.activate([entry.key])
            if entry.kind == "code" and entry.key not in skills.activated():
                warnings.append(f"Could not activate skill {name}")
        except Exception as exc:
            warnings.append(f"Could not activate skill {name}: {exc}")
    for name in options.inactive_skills:
        entry = skills.entry(name)
        if entry is not None and entry.key in skills.activated():
            name = entry.key
            try:
                skills.registry.deactivate([name])
                if name in skills.activated():
                    warnings.append(f"Could not deactivate skill {name}")
            except Exception as exc:
                warnings.append(f"Could not deactivate skill {name}: {exc}")
    return warnings
