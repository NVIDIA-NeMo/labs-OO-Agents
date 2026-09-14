# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared memory ownership and reflection setup for interactive sessions."""

from __future__ import annotations

import logging
from pathlib import Path

from nooa import Agent
from nooa_cli.coding.identity import CODING_AGENT, EXPERIMENTAL_CODING_AGENT, canonical_agent_spec
from nooa_cli.interactive.options import SessionOptions

logger = logging.getLogger(__name__)


def agent_memory_key(agent: Agent, options: SessionOptions) -> str:
    """Return the stable settings key for an agent's memory preferences."""
    if options.agent_spec and not options.legacy_agent:
        spec = canonical_agent_spec(options.agent_spec)
        if spec not in {CODING_AGENT, EXPERIMENTAL_CODING_AGENT}:
            return spec
    return CODING_AGENT


def resolve_memory_owner(agent: Agent, options: SessionOptions) -> str:
    """Return the role portion of this agent's hierarchical memory owner."""
    key = agent_memory_key(agent, options)
    per_agent = options.memory_owner_agents.get(key)
    if per_agent:
        return per_agent
    if options.memory_owner:
        return options.memory_owner
    if key == CODING_AGENT:
        return "CodingAgent"
    return type(agent).__name__


def resolve_memory_scope(agent: Agent, options: SessionOptions) -> str:
    """Return the effective memory scope for *agent*."""
    key = agent_memory_key(agent, options)
    return options.memory_agents.get(key, options.memory)


def resolve_reflection_enabled(agent: Agent, options: SessionOptions) -> bool:
    """Return whether idle reflection is enabled for *agent*."""
    key = agent_memory_key(agent, options)
    return bool(options.reflection_agents.get(key, options.reflection))


def _teardown_reflection(agent: Agent) -> None:
    """Tear down the reflection runner before detaching or replacing memory."""
    runner = getattr(agent, "_reflection_runner", None)
    if runner is not None:
        runner.teardown()
        del agent._reflection_runner


def configure_session_memory(
    agent: Agent,
    options: SessionOptions,
    *,
    agent_db: Path | None,
    session_id: str | None,
) -> None:
    """Install or remove the memory skill according to the session options."""
    key = agent_memory_key(agent, options)
    agent._memory_key = key  # type: ignore[attr-defined]
    scope = resolve_memory_scope(agent, options)

    _teardown_reflection(agent)
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
        raise ValueError(f"Unsupported memory scope {scope!r}; use 'off', 'session', or 'project'.")

    from nooa_memory import MemoryConfig
    from nooa_memory.memory_skill import MemorySkill

    project_dir = (Path(options.working_dir) / ".nooa").resolve()
    if options.memory_path is not None:
        if options.memory_path.is_absolute():
            raise ValueError("coding.memory_path must be relative to the project directory")
        memory_path = (project_dir / options.memory_path).resolve()
        if project_dir not in memory_path.parents and memory_path != project_dir:
            raise ValueError("coding.memory_path must stay under the project directory")
    elif scope == "project":
        memory_path = (Path(options.working_dir) / ".nooa" / "memory" / "memory.sqlite").resolve()
    else:
        if session_id is None or agent_db is None:
            raise RuntimeError("session-scoped memory requires a durable session")
        memory_path = Path(agent_db).with_name(f"{session_id}-memory.db")

    reflection_enabled = resolve_reflection_enabled(agent, options)
    memory_kwargs: dict[str, object] = {}
    if reflection_enabled:
        from nooa_memory.config import ReflectionPolicy

        # ReflectionRunner owns consolidation while idle; do not also run it
        # inline in the memory middleware after every response.
        memory_kwargs["reflection"] = ReflectionPolicy(trigger="manual")

    owner_role = resolve_memory_owner(agent, options)
    owner = f"{owner_role}@{session_id[:8]}" if session_id else owner_role
    memory_config = MemoryConfig(
        enabled=True,
        path=str(memory_path),
        owner=owner,
        **memory_kwargs,
    )

    skill_kwargs: dict[str, object] = {}
    episode_writer = None
    if reflection_enabled and options.reflection_generative:
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
    if key == CODING_AGENT and owner_role == "CodingAgent":
        # Preserve visibility of memories written under the historical built-in owner.
        old_owners = {
            memory.owner
            for memory in manager.store.iter_memories(include_archived=True, owner="TUIAgent")
            if memory.owner == "TUIAgent" or memory.owner.startswith("TUIAgent@")
        }
        for old_owner in old_owners:
            manager.store.rename_owner(old_owner, owner_role + old_owner[len("TUIAgent") :])
    if key != owner_role:
        renamed = manager.store.rename_owner(key, owner_role)
        if renamed:
            manager.store.log_maintenance(
                "rename_owner",
                {"from": key, "to": owner_role, "rows": renamed},
            )
    manager.session_ref = session_id

    from .reflection_runner import ReflectionRunner

    agent._reflection_runner = ReflectionRunner(  # type: ignore[attr-defined]
        agent,
        manager,
        options,
        enabled=reflection_enabled,
        episode_writer=episode_writer,
    )
