# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared bootstrap for the native terminal UI."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from nooa.sessions import SessionResumed

from .output import Output, TextOutput

if TYPE_CHECKING:
    from nooa import Agent

    from .commands import CommandRegistry
    from .config import Config
    from .frontend import Frontend
    from .health_check import HealthCheckResult
    from .session import Session
    from .session_manager import SessionManager

logger = logging.getLogger(__name__)


def _instantiate_custom_agent(
    agent_cls,
    *,
    llm,
    storage,
    working_directory: str | Path,
    skills_dirs: list[Path],
):
    """Instantiate an extension agent with the host arguments it declares."""
    parameters = inspect.signature(agent_cls).parameters
    kwargs = {"llm": llm, "storage": storage}
    if "cwd" in parameters:
        kwargs["cwd"] = working_directory
    if "skills_dirs" in parameters:
        kwargs["skills_dirs"] = skills_dirs
    return agent_cls(**kwargs)


@dataclass
class BootstrapResult:
    """Everything produced by bootstrap, ready to wire to a frontend."""

    config: Config
    agent: Agent
    session_manager: SessionManager | None
    tracing_enabled: bool
    resumed: bool
    restored: bool
    session_id: str | None
    messages: list[Output] = field(default_factory=list)
    blocking_llm_health: HealthCheckResult | None = None


def _scaffold_settings(config: Config) -> None:
    from nooa.paths import get_user_dir

    from .settings import SETTINGS_FILENAME, render_settings_template, settings_present

    if settings_present():
        return
    target = get_user_dir(SETTINGS_FILENAME)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_settings_template(config))


def tui_agent_memory_key(agent: Agent, config: Config) -> str:
    """Return the stable settings key for an agent's memory preferences."""
    if config.tui.agent_spec:
        return config.tui.agent_spec
    return f"{type(agent).__module__}:{type(agent).__qualname__}"


def resolve_tui_memory_owner(agent: Agent, config: Config) -> str:
    """Return the role portion of this agent's hierarchical memory owner."""
    key = tui_agent_memory_key(agent, config)
    per_agent = config.tui.memory_owner_agents.get(key)
    if per_agent:
        return per_agent
    if config.tui.memory_owner:
        return config.tui.memory_owner
    return type(agent).__name__


def resolve_tui_memory_scope(agent: Agent, config: Config) -> str:
    """Return the effective memory scope for *agent*."""
    key = tui_agent_memory_key(agent, config)
    return config.tui.memory_agents.get(key, config.tui.memory)


def resolve_tui_reflection_enabled(agent: Agent, config: Config) -> bool:
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
    config: Config,
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

    from nooa.paths import get_project_dir

    project_dir = get_project_dir().resolve()
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


def _load_llm_registry(messages: list[Output], explicit_paths: list[Path] | None = None) -> None:
    try:
        from nooa.llm_config import llm_config_chain
        from nooa.secrets import load_secrets_into_env
        from nooa.unifiedllm import reload_registry

        load_secrets_into_env()
        # Explicit host paths come last and therefore override bundled, user,
        # project, and environment layers. The TUI accepts only existing local
        # files; fetching or authenticating to a private registry remains the
        # operator's responsibility outside this public process.
        reload_registry(*llm_config_chain(), *(explicit_paths or []))
    except Exception as exc:
        messages.append(TextOutput(f"Failed to load LLM registry config: {exc}", "warning"))


def _enable_tracing(config: Config, messages: list[Output]):
    if config.no_trace:
        return False, None
    try:
        from nooa.paths import find_project_root, get_project_dir
        from nooa.tracing import enable_tracing, exporters, set_session

        trace_dir = config.tui.trace_dir
        if trace_dir is not None:
            if str(trace_dir) == ":project:":
                trace_dir = get_project_dir("traces")
            elif not trace_dir.is_absolute():
                trace_dir = find_project_root() / trace_dir
            trace_dir.mkdir(parents=True, exist_ok=True)
            enable_tracing(exporters=[exporters.jsonl(trace_dir), exporters.journal()])
        else:
            enable_tracing()
        return True, set_session
    except ImportError:
        messages.append(
            TextOutput(
                "Tracing package not installed (openinference-instrumentation-nooa)",
                "warning",
            )
        )
    except Exception as exc:
        messages.append(TextOutput(f"Failed to enable tracing: {exc}", "warning"))
    return False, None


async def bootstrap(
    config: Config,
    *,
    continue_last: bool = False,
    resume_session_id: str | None = None,
    agent: Agent | None = None,
) -> BootstrapResult:
    """Create the coding agent and its shared durable session."""
    if agent is not None:
        return BootstrapResult(
            config=config,
            agent=agent,
            session_manager=None,
            tracing_enabled=False,
            resumed=False,
            restored=False,
            session_id=None,
        )

    messages: list[Output] = []
    _scaffold_settings(config)
    _load_llm_registry(messages, config.llm_config_paths)
    tracing_enabled, set_trace_session = _enable_tracing(config, messages)

    from .config import UnresolvedModelError, get_llm

    blocking_llm_health = None
    try:
        llm = get_llm(config)
    except UnresolvedModelError as exc:
        from nooa.unifiedllm import FakeLLMClient

        from .health_check import unresolved_model_health

        blocking_llm_health = unresolved_model_health(exc.model)
        messages.append(TextOutput(f"⚠️  {blocking_llm_health.error_message}", "error"))
        if blocking_llm_health.fix_hint:
            messages.append(TextOutput(blocking_llm_health.fix_hint, "info"))
        llm = FakeLLMClient()
    except Exception as exc:
        from nooa.unifiedllm import FakeLLMClient

        from .health_check import HealthCheckResult

        blocking_llm_health = HealthCheckResult(
            ok=False,
            error_message=f"Failed to initialize model '{config.tui.default_model}': {exc}",
            fix_hint=(
                "  • Run `nooa config show` to inspect model configuration\n"
                "  • Use /model <provider/model> to select a different model"
            ),
            blocking=True,
        )
        messages.append(TextOutput(f"⚠️  {blocking_llm_health.error_message}", "error"))
        messages.append(TextOutput(blocking_llm_health.fix_hint, "info"))
        llm = FakeLLMClient()

    from nooa.unifiedllm import FakeLLMClient

    if not isinstance(llm, FakeLLMClient):
        from .health_check import probe_llm

        health = await probe_llm(llm)
        if not health.ok:
            messages.append(TextOutput(f"⚠️  {health.error_message}", "error"))
            if health.fix_hint:
                messages.append(TextOutput(health.fix_hint, "info"))
            if health.blocking:
                blocking_llm_health = health

    from nooa.storage.sqlite import SessionAlreadyActiveError

    from .session_manager import SessionManager, _make_trace_session_name

    resume_id: str | None = None
    if resume_session_id:
        matches = SessionManager.find_by_prefix(resume_session_id)
        if len(matches) == 1:
            resume_id = matches[0]
        elif len(matches) > 1:
            messages.append(
                TextOutput(
                    f"Session prefix '{resume_session_id}' is ambiguous; starting new.", "warning"
                )
            )
        else:
            messages.append(
                TextOutput(f"Session '{resume_session_id}' not found; starting new.", "warning")
            )
    elif continue_last:
        resume_id = next(
            (item.id for item in SessionManager.list_sessions(limit=20) if item.turn_count > 0),
            None,
        )

    resumed = resume_id is not None
    try:
        session_manager = (
            SessionManager.open(resume_id)
            if resume_id is not None
            else SessionManager.create(
                model=config.tui.default_model,
                agent_cls="TUIAgent",
                working_dir=str(config.agent.working_dir),
            )
        )
    except SessionAlreadyActiveError as exc:
        detail = f" (pid {exc.owner_pid})" if exc.owner_pid is not None else ""
        messages.append(
            TextOutput(
                f"Session {str(resume_id)[:8]!r} is active in another process{detail}; starting new.",
                "warning",
            )
        )
        session_manager = SessionManager.create(
            model=config.tui.default_model,
            agent_cls="TUIAgent",
            working_dir=str(config.agent.working_dir),
        )
        resumed = False

    session_id = session_manager.session_id
    if set_trace_session is not None:
        set_trace_session(_make_trace_session_name(session_id))

    from .agent import TUIAgent

    storage_kwargs = {"storage": session_manager._storage}
    if config.tui.agent_spec:
        from .config import load_agent_class
        from .theme import COLORS

        try:
            agent_cls = load_agent_class(config.tui.agent_spec)
            agent = _instantiate_custom_agent(
                agent_cls,
                llm=llm,
                storage=session_manager._storage,
                working_directory=config.agent.working_dir,
                skills_dirs=config.tui.skills_dirs,
            )
            session_manager.update_agent_cls(type(agent).__name__)
            messages.append(
                TextOutput(
                    f"Loaded custom agent: [{COLORS['green']}]{agent_cls.__name__}[/] "
                    f"from {config.tui.agent_spec}",
                    "info",
                )
            )
        except Exception as exc:
            messages.extend(
                (
                    TextOutput(f"Failed to load agent '{config.tui.agent_spec}': {exc}", "error"),
                    TextOutput("Falling back to default coding agent", "info"),
                )
            )
            agent = TUIAgent(
                llm=llm,
                config=config.agent,
                skills_dirs=config.tui.skills_dirs,
                **storage_kwargs,
            )
    else:
        agent = TUIAgent(
            llm=llm,
            config=config.agent,
            skills_dirs=config.tui.skills_dirs,
            **storage_kwargs,
        )

    restored = False
    if resumed:
        try:
            restored = session_manager._storage.restore_latest_snapshot(agent)
            if not restored:
                messages.append(TextOutput("No agent snapshot found in session.", "warning"))
        except Exception as exc:
            messages.append(TextOutput(f"Could not restore agent state: {exc}", "warning"))

    try:
        configure_tui_memory(
            agent,
            config,
            agent_db=session_manager.agent_db_path,
            session_id=session_id,
        )
    except Exception as exc:
        messages.append(TextOutput(f"Could not enable memory: {exc}", "warning"))

    agent._session_manager = session_manager  # type: ignore[attr-defined]
    return BootstrapResult(
        config=config,
        agent=agent,
        session_manager=session_manager,
        tracing_enabled=tracing_enabled,
        resumed=resumed,
        restored=restored,
        session_id=session_id,
        messages=messages,
        blocking_llm_health=blocking_llm_health,
    )


def build_startup_info(result: BootstrapResult) -> Output:
    from .agent import TUIAgent
    from .output import StartupInfo
    from .session import _short_model_name

    config = result.config
    agent = result.agent
    trace_dir: str | None = None
    if result.tracing_enabled and config.tui.trace_dir:
        from nooa.paths import get_project_dir

        configured = config.tui.trace_dir
        trace_dir = str(get_project_dir("traces") if str(configured) == ":project:" else configured)
    return StartupInfo(
        model=config.tui.default_model,
        short_model=_short_model_name(config.tui.default_model),
        working_dir=str(config.agent.working_dir),
        vi_mode=config.tui.vi_mode,
        history_policy=(config.agent.summarization.policy if isinstance(agent, TUIAgent) else None),
        history_limit=(
            config.agent.summarization.max_tokens if isinstance(agent, TUIAgent) else None
        ),
        tracing_enabled=result.tracing_enabled,
        trace_dir=trace_dir,
        custom_agent=(
            type(agent).__name__
            if config.tui.agent_spec and not isinstance(agent, TUIAgent)
            else None
        ),
        llm_ready=result.blocking_llm_health is None,
    )


def build_registry(result: BootstrapResult, frontend: Frontend) -> CommandRegistry:
    from .commands import CommandRegistry
    from .mcp_registry import MCPRegistry

    if hasattr(result.agent, "skills"):
        result.agent.skills.register(  # type: ignore[union-attr]
            "nemo.mcp",
            MCPRegistry(
                mcp_file=result.config.tui.mcp_file,
                servers=result.config.tui.mcp_servers,
                watch_settings=True,
            ),
        )
        result.agent.skills.activate(["nemo.mcp"])  # type: ignore[union-attr]

    skills = getattr(result.agent, "skills", None)
    if skills is not None:
        discovered = set(skills.discovered())
        for skill_name in result.config.tui.active_skills:
            if skill_name not in discovered:
                result.messages.append(
                    TextOutput(f"Configured skill not found: {skill_name}", "warning")
                )
                continue
            try:
                skills.activate([skill_name])
            except Exception as exc:
                result.messages.append(
                    TextOutput(f"Could not activate skill {skill_name}: {exc}", "warning")
                )

    if result.session_id is not None:
        try:
            result.agent.event_manager.add(
                SessionResumed(session_id=result.session_id, restored=result.restored)
            )
        except Exception:
            logger.debug("Failed to emit SessionResumed", exc_info=True)

    registry = CommandRegistry(
        config=result.config.tui,
        agent=result.agent,
        frontend=frontend,
        skills_dirs=result.config.tui.skills_dirs,
        mcp_file=result.config.tui.mcp_file,
        session_manager=result.session_manager,
        root_config=result.config,
    )
    registry.blocking_llm_health = result.blocking_llm_health
    result.agent._command_registry = registry  # type: ignore[attr-defined]
    return registry


def build_session(
    result: BootstrapResult,
    frontend: Frontend,
    registry: CommandRegistry,
    initial_outputs: list[Output] | None = None,
) -> Session:
    from .session import Session

    return Session(
        frontend=frontend,
        agent=result.agent,
        config=result.config,
        registry=registry,
        session_manager=result.session_manager,
        initial_outputs=initial_outputs,
    )
