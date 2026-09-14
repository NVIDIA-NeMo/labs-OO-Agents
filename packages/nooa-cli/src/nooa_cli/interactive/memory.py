# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared memory ownership and reflection setup for interactive sessions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from nooa import Agent

logger = logging.getLogger(__name__)


def tui_agent_memory_key(agent: Agent, config: Any) -> str:
    """Return the stable settings key for an agent's memory preferences."""
    if config.tui.agent_spec and not config.legacy_agent:
        return config.tui.agent_spec
    # Keep the historical built-in key so existing memory/reflection preferences
    # continue to apply when the single-tool implementation becomes the default.
    return "nooa_cli.tui.agent:TUIAgent"


def resolve_tui_memory_owner(agent: Agent, config: Any) -> str:
    """Return the role portion of this agent's hierarchical memory owner."""
    key = tui_agent_memory_key(agent, config)
    per_agent = config.tui.memory_owner_agents.get(key)
    if per_agent:
        return per_agent
    if config.tui.memory_owner:
        return config.tui.memory_owner
    if key == "nooa_cli.tui.agent:TUIAgent":
        return "TUIAgent"
    return type(agent).__name__


def resolve_tui_memory_scope(agent: Agent, config: Any) -> str:
    """Return the effective memory scope for *agent*."""
    key = tui_agent_memory_key(agent, config)
    return config.tui.memory_agents.get(key, config.tui.memory)


def resolve_tui_reflection_enabled(agent: Agent, config: Any) -> bool:
    """Return whether idle reflection is enabled for *agent*."""
    key = tui_agent_memory_key(agent, config)
    return bool(config.tui.reflection_agents.get(key, config.tui.reflection))


def _teardown_tui_reflection(agent: Agent) -> None:
    """Tear down the reflection runner before detaching or replacing memory."""
    runner = getattr(agent, "_tui_reflection_runner", None)
    if runner is not None:
        runner.teardown()
        del agent._tui_reflection_runner


def configure_tui_memory(
    agent: Agent,
    config: Any,
    *,
    agent_db: Path | None,
    session_id: str | None,
) -> None:
    """Install or remove the memory skill according to the TUI configuration."""
    key = tui_agent_memory_key(agent, config)
    agent._tui_memory_key = key  # type: ignore[attr-defined]
    scope = resolve_tui_memory_scope(agent, config)

    _teardown_tui_reflection(agent)
    existing = getattr(agent, "memory", None)
    if existing is not None and hasattr(existing, "detach"):
        try:
            existing.detach()
        except Exception:
            logger.debug("Could not detach the previous memory skill", exc_info=True)

    if scope == "off":
        skills = getattr(agent, "skills", None)
        if skills is not None:
            try:
                skills.deactivate(["nemo.memory"])
            except Exception:
                logger.debug("Could not deactivate memory", exc_info=True)
        if hasattr(agent, "memory"):
            try:
                delattr(agent, "memory")
            except Exception:
                logger.debug("Could not remove memory from agent", exc_info=True)
        return

    if scope not in {"session", "project"}:
        raise ValueError(
            f"Unsupported TUI memory scope {scope!r}; use 'off', 'session', or 'project'."
        )

    from nooa_memory import MemoryConfig
    from nooa_memory.memory_skill import MemorySkill

    project_dir = (Path(config.agent.working_dir) / ".nooa").resolve()
    if config.tui.memory_path is not None:
        if config.tui.memory_path.is_absolute():
            raise ValueError("tui.memory_path must be relative to the project directory")
        memory_path = (project_dir / config.tui.memory_path).resolve()
        if project_dir not in memory_path.parents and memory_path != project_dir:
            raise ValueError("tui.memory_path must stay under the project directory")
    elif scope == "project":
        memory_path = (
            Path(config.agent.working_dir) / ".nooa" / "memory" / "memory.sqlite"
        ).resolve()
    else:
        if session_id is None or agent_db is None:
            raise RuntimeError("session-scoped memory requires a durable session")
        memory_path = Path(agent_db).with_name(f"{session_id}-memory.db")

    reflection_enabled = resolve_tui_reflection_enabled(agent, config)
    memory_kwargs: dict[str, object] = {}
    if reflection_enabled:
        from nooa_memory.config import ReflectionPolicy

        # ReflectionRunner owns consolidation while idle; do not also run it
        # inline in the memory middleware after every response.
        memory_kwargs["reflection"] = ReflectionPolicy(trigger="manual")

    owner_role = resolve_tui_memory_owner(agent, config)
    owner = f"{owner_role}@{session_id[:8]}" if session_id else owner_role
    memory_config = MemoryConfig(
        enabled=True,
        path=str(memory_path),
        owner=owner,
        **memory_kwargs,
    )

    skill_kwargs: dict[str, object] = {}
    episode_writer = None
    if reflection_enabled and config.tui.reflection_generative:
        from nooa_memory.generative import llm_episode_writer, llm_reasoner, llm_reconciler

        def _session_llm() -> object:
            return agent._llm  # type: ignore[attr-defined]

        skill_kwargs = {
            "reasoner": llm_reasoner(_session_llm),
            "reconciler": llm_reconciler(_session_llm),
        }
        episode_writer = llm_episode_writer(_session_llm)

    skills = getattr(agent, "skills", None)
    if skills is None:
        raise RuntimeError("memory requires an agent with a SkillRegistry")
    skills.register("nemo.memory", MemorySkill(memory_config, **skill_kwargs))
    skills.activate(["nemo.memory"])

    manager = agent.memory._mgr  # type: ignore[attr-defined]
    if key != owner_role:
        renamed = manager.store.rename_owner(key, owner_role)
        if renamed:
            manager.store.log_maintenance(
                "rename_owner",
                {"from": key, "to": owner_role, "rows": renamed},
            )
    manager.session_ref = session_id

    from .reflection_runner import ReflectionRunner

    agent._tui_reflection_runner = ReflectionRunner(  # type: ignore[attr-defined]
        agent,
        manager,
        config.tui,
        enabled=reflection_enabled,
        episode_writer=episode_writer,
    )
