# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workspace-scoped agent settings shared by the native and ACP hosts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from nooa.interactive import DEFAULT_MODEL, SummarizationConfig
from nooa_cli.coding.settings import load_coding_skills_dirs


class SessionOptions(BaseModel):
    """Behavioral options; terminal presentation settings stay with the TUI."""

    working_dir: str = "."
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig)
    legacy_agent: bool = False
    agent_spec: str | None = None
    skills_dirs: list[Path] = Field(default_factory=list)
    additional_skills_dirs: list[Path] = Field(default_factory=list)
    default_model: str = DEFAULT_MODEL
    active_skills: list[str] = Field(default_factory=list)
    inactive_skills: list[str] = Field(default_factory=list)
    mcp_file: Path = Path(".mcp.json")
    mcp_servers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    mcp_auto_connect: list[str] = Field(default_factory=list)
    memory: Literal["off", "session", "project"] = "off"
    memory_agents: dict[str, Literal["off", "session", "project"]] = Field(default_factory=dict)
    memory_path: Path | None = None
    memory_owner: str | None = None
    memory_owner_agents: dict[str, str] = Field(default_factory=dict)
    reflection: bool = False
    reflection_agents: dict[str, bool] = Field(default_factory=dict)
    reflection_generative: bool = True
    reflection_debounce_s: float = 10.0
    reflection_grace_s: float = 0.5

    def policy_config(self) -> Any:
        """Provide the legacy settings shape while policy consumers migrate."""
        from types import SimpleNamespace

        return SimpleNamespace(tui=self, agent=self, legacy_agent=self.legacy_agent)

    @classmethod
    def load(cls, workspace: str | Path, **overrides: Any) -> SessionOptions:
        """Load legacy ``tui`` and shared ``coding`` settings for this workspace."""
        root = Path(workspace).expanduser().resolve()
        from .settings import load_settings_data, resolve_behavior_settings

        values = resolve_behavior_settings(load_settings_data(root))
        values.update({key: value for key, value in overrides.items() if value is not None})
        values["working_dir"] = str(root)
        values["skills_dirs"] = load_coding_skills_dirs(root)
        return cls(**values)

    @classmethod
    def from_native_config(cls, config: Any) -> SessionOptions:
        """Project native configuration onto the shared behavior contract."""
        values = config.tui.model_dump()
        values.update(
            working_dir=str(Path(config.agent.working_dir).expanduser().resolve()),
            summarization=config.agent.summarization,
            legacy_agent=config.legacy_agent,
            skills_dirs=load_coding_skills_dirs(
                config.agent.working_dir, explicit=config.tui.skills_dirs
            ),
        )
        return cls(**values)


def configure_session_skills(agent: Any, options: SessionOptions) -> list[str]:
    """Attach the same MCP registry and explicit skills before resume events.

    Return actionable warnings for either host to display. Discovering a skill
    does not activate it; negative activation preferences override positives.
    """
    from nooa_cli.interactive.mcp_registry import MCPRegistry

    skills = getattr(agent, "skills", None)
    if skills is None:
        return []
    root = Path(options.working_dir)
    mcp_file = options.mcp_file.expanduser()
    skills.register(
        "nemo.mcp",
        MCPRegistry(
            mcp_file=mcp_file if mcp_file.is_absolute() else root / mcp_file,
            servers=options.mcp_servers,
            watch_settings=True,
            project_dir=root / ".nooa",
        ),
    )
    skills.activate(["nemo.mcp"])
    warnings: list[str] = []
    discover = getattr(skills, "discover_skills_dirs", None)
    if options.active_skills and callable(discover):
        try:
            discover(options.skills_dirs)
        except Exception as exc:
            warnings.append(f"Could not discover configured skills: {exc}")
    discovered = set(skills.discovered())
    for name in options.active_skills:
        if name not in discovered:
            warnings.append(f"Configured skill not found: {name}")
            continue
        try:
            skills.activate([name])
            if name not in skills.activated():
                warnings.append(f"Could not activate skill {name}")
        except Exception as exc:
            warnings.append(f"Could not activate skill {name}: {exc}")
    for name in options.inactive_skills:
        if name in skills.activated():
            try:
                skills.deactivate([name])
                if name in skills.activated():
                    warnings.append(f"Could not deactivate skill {name}")
            except Exception as exc:
                warnings.append(f"Could not deactivate skill {name}: {exc}")
    return warnings
