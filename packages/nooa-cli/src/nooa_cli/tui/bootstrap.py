# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared bootstrap for the native terminal UI."""

from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from nooa.sessions import SessionResumed
from nooa_cli.interactive.memory import (
    configure_tui_memory as configure_tui_memory,
)
from nooa_cli.interactive.memory import (
    resolve_tui_memory_owner as resolve_tui_memory_owner,
)
from nooa_cli.interactive.memory import (
    resolve_tui_memory_scope as resolve_tui_memory_scope,
)
from nooa_cli.interactive.memory import (
    resolve_tui_reflection_enabled as resolve_tui_reflection_enabled,
)
from nooa_cli.interactive.memory import (
    tui_agent_memory_key as tui_agent_memory_key,
)

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

# The runtime's agent_call middleware coverage diagnostic targets runtime
# developers: it fires once per session for synchronous agent methods that no
# registered guard can wrap. In the native TUI those helpers (spawn, session
# titling, status, delegation labels) are benign, so silence the diagnostic for
# TUI users. Developers who explicitly run with warnings-as-errors
# (``-W error``) still see it: the filter is only installed when no error-style
# warning option is active, and the emitter deliberately lets the promoted
# exception propagate.
if not any(option.startswith("error") for option in sys.warnoptions):
    warnings.filterwarnings(
        "ignore",
        message=r"agent_call middleware is registered",
        category=RuntimeWarning,
    )


def _instantiate_custom_agent(
    agent_cls,
    *,
    llm,
    storage,
    working_directory: str | Path,
    skills_dirs: list[Path],
    summarization=None,
):
    """Instantiate an extension agent with the host arguments it declares."""
    from types import SimpleNamespace

    from nooa_cli.coding.factory import create_session_agent

    options = SimpleNamespace(
        working_dir=working_directory, skills_dirs=skills_dirs, summarization=summarization
    )
    return create_session_agent(llm=llm, storage=storage, options=options, agent_cls=agent_cls)


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

    from .config import DEFAULT_MODEL, UnresolvedModelError, get_llm

    blocking_llm_health = None
    llm = None
    try:
        from nooa.unifiedllm import MODELS, FakeLLMClient

        if config.tui.default_model == DEFAULT_MODEL and DEFAULT_MODEL not in MODELS:
            from .health_check import no_models_configured_health

            blocking_llm_health = no_models_configured_health()
            llm = FakeLLMClient()
    except Exception:
        logger.debug("Could not inspect loaded LLM registry", exc_info=True)

    try:
        if llm is None:
            llm = get_llm(config)
    except UnresolvedModelError as exc:
        from nooa.unifiedllm import FakeLLMClient

        from .health_check import unresolved_model_health

        if exc.model == DEFAULT_MODEL:
            from .health_check import no_models_configured_health

            blocking_llm_health = no_models_configured_health()
        else:
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
        from .health_check import HealthCheckResult

        blocking_llm_health = HealthCheckResult(
            ok=False,
            error_message=f"Checking LLM endpoint for model '{config.tui.default_model}'.",
            fix_hint="Slash commands and !shell commands are available while the check runs.",
            blocking=True,
            pending=True,
        )

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
                agent_cls=("TUIAgent" if config.legacy_agent else "ExperimentalTUIAgent"),
                working_dir=str(config.agent.working_dir),
            )
        )
    except SessionAlreadyActiveError as exc:
        messages.append(
            TextOutput(
                f"Could not resume session {str(resume_id)[:8]!r}: {exc} Starting new.",
                "warning",
            )
        )
        session_manager = SessionManager.create(
            model=config.tui.default_model,
            agent_cls=("TUIAgent" if config.legacy_agent else "ExperimentalTUIAgent"),
            working_dir=str(config.agent.working_dir),
        )
        resumed = False

    session_id = session_manager.session_id
    if set_trace_session is not None:
        set_trace_session(_make_trace_session_name(session_id))

    from nooa_cli.coding.factory import create_session_agent
    from nooa_cli.interactive.options import SessionOptions

    options = SessionOptions.from_native_config(config)
    try:
        agent = create_session_agent(llm=llm, storage=session_manager._storage, options=options)
    except Exception as exc:
        if not options.agent_spec or options.legacy_agent:
            raise
        messages.append(TextOutput(f"Failed to load agent '{options.agent_spec}': {exc}", "error"))
        messages.append(TextOutput("Falling back to default coding agent", "info"))
        options.agent_spec = None
        agent = create_session_agent(llm=llm, storage=session_manager._storage, options=options)
    session_manager.update_agent_cls(type(agent).__name__)

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
    from nooa_cli.coding.agent import CodingAgent

    from .agent import TUIAgent
    from .experimental_agent import ExperimentalTUIAgent
    from .output import StartupInfo
    from .session import _short_model_name

    config = result.config
    agent = result.agent
    health = result.blocking_llm_health
    from .health_check import is_no_models_configured_health

    no_models_configured = is_no_models_configured_health(health)
    if health is None:
        llm_status = "ready"
    elif no_models_configured:
        llm_status = "not_connected"
    elif getattr(health, "pending", False):
        llm_status = "checking"
    else:
        llm_status = "unavailable"
    trace_dir: str | None = None
    if result.tracing_enabled and config.tui.trace_dir:
        from nooa.paths import get_project_dir

        configured = config.tui.trace_dir
        trace_dir = str(get_project_dir("traces") if str(configured) == ":project:" else configured)
    return StartupInfo(
        model="run /connect to configure one" if no_models_configured else config.tui.default_model,
        short_model="No LLM connected"
        if no_models_configured
        else _short_model_name(config.tui.default_model),
        working_dir=str(config.agent.working_dir),
        vi_mode=config.tui.vi_mode,
        history_policy=(
            config.agent.summarization.policy if isinstance(agent, CodingAgent) else None
        ),
        history_limit=(
            config.agent.summarization.max_tokens if isinstance(agent, CodingAgent) else None
        ),
        tracing_enabled=result.tracing_enabled,
        trace_dir=trace_dir,
        custom_agent=(
            type(agent).__name__
            if config.tui.agent_spec
            and not config.legacy_agent
            and not isinstance(agent, (TUIAgent, ExperimentalTUIAgent))
            else None
        ),
        llm_ready=llm_status == "ready",
        llm_status=llm_status,
    )


def build_registry(result: BootstrapResult, frontend: Frontend) -> CommandRegistry:
    from nooa_cli.interactive.options import SessionOptions, configure_session_skills

    from .commands import CommandRegistry

    options = SessionOptions.from_native_config(result.config)
    warnings = configure_session_skills(result.agent, options, live_config=result.config.tui)
    result.messages.extend(TextOutput(message, "warning") for message in warnings)

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
        skills_dirs=options.skills_dirs,
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
