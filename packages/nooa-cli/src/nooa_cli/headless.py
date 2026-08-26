# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-shot, renderer-free host for the NOOA coding agent."""

from __future__ import annotations

import asyncio
import inspect
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from nooa.events import LLMComplete
from nooa.interactive import AgentMessage, RespondReason
from nooa.sessions import SessionHandle, SessionResumed, SessionStore
from nooa.storage.in_memory import InMemoryStorageManager
from nooa_cli.coding import CodingAgent, load_coding_skills_dirs
from nooa_cli.interactive.dispatcher import InteractiveSessionDispatcher


@dataclass(slots=True)
class HeadlessUsage:
    """Aggregate model usage across all calls made by one headless turn."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0


@dataclass(slots=True)
class HeadlessResult:
    """Stable result returned by :func:`run_headless`."""

    session_id: str | None
    status: Literal["done", "blocked", "cancelled", "failed"]
    messages: list[str]
    explanation: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    usage: HeadlessUsage = field(default_factory=HeadlessUsage)
    error: dict[str, str] | None = None
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _TurnFinished(Exception):
    """Carry a terminal result through the common cleanup path."""

    def __init__(self, result: HeadlessResult) -> None:
        self.result = result


def resolve_session(
    store: SessionStore,
    *,
    working_directory: Path,
    model: str,
    agent_name: str,
    continue_session: bool = False,
    resume_session_id: str | None = None,
) -> tuple[SessionHandle, bool]:
    """Create a session or resolve one unambiguous durable resume target."""
    if continue_session and resume_session_id:
        raise ValueError("--continue and --resume cannot be used together")

    session_id: str | None = None
    if resume_session_id:
        matches = store.find_by_prefix(resume_session_id)
        if len(matches) != 1:
            detail = "was not found" if not matches else "is ambiguous"
            raise ValueError(f"Session prefix {resume_session_id!r} {detail}")
        session_id = matches[0]
    elif continue_session:
        workspace = working_directory.resolve()
        session_id = next(
            (
                info.id
                for info in store.list(limit=None)
                if Path(info.working_directory).expanduser().resolve() == workspace
                and info.turn_count > 0
            ),
            None,
        )
        if session_id is None:
            raise ValueError(f"No resumable session found for {workspace}")

    if session_id is not None:
        return store.open(session_id, check_same_thread=False), True
    return (
        store.create(
            model=model,
            agent=agent_name,
            working_directory=str(working_directory),
            host="run",
            check_same_thread=False,
        ),
        False,
    )


def _instantiate_agent(agent_cls: type, *, config: Any, llm: Any, storage: Any) -> Any:
    parameters = inspect.signature(agent_cls).parameters
    kwargs: dict[str, Any] = {"llm": llm, "storage": storage}
    if "cwd" in parameters:
        kwargs["cwd"] = config.agent.working_dir
    elif "config" in parameters:
        kwargs["config"] = config.agent
    if "skills_dirs" in parameters:
        kwargs["skills_dirs"] = load_coding_skills_dirs(
            config.agent.working_dir,
            explicit=config.tui.skills_dirs,
        )
    return agent_cls(**kwargs)


def _configure_skills(agent: Any, config: Any) -> None:
    """Apply configured skills and attach the approved MCP registry."""
    skills = getattr(agent, "skills", None)
    if skills is None:
        return

    from nooa_cli.tui.mcp_registry import MCPRegistry

    mcp_file = Path(config.tui.mcp_file).expanduser()
    if not mcp_file.is_absolute():
        mcp_file = Path(config.agent.working_dir).resolve() / mcp_file
    skills.register(
        "nemo.mcp",
        MCPRegistry(
            mcp_file=mcp_file,
            servers=config.tui.mcp_servers,
            watch_settings=False,
        ),
    )
    skills.activate(["nemo.mcp"])
    discover = getattr(skills, "discover_skills_dirs", None)
    if callable(discover):
        discover(config.tui.skills_dirs)
    discovered = set(skills.discovered())
    for name in config.tui.active_skills:
        if name not in discovered:
            raise ValueError(f"Configured skill not found: {name}")
        skills.activate([name])
    for name in config.tui.inactive_skills:
        if name in skills.activated():
            skills.deactivate([name])


async def _connect_startup_mcp(agent: Any, config: Any) -> None:
    """Connect only explicitly configured startup servers without prompting."""
    names = list(dict.fromkeys(config.tui.mcp_auto_connect))
    if not names:
        return
    registry = getattr(agent, "mcp", None)
    if registry is None:
        raise RuntimeError("MCP auto-connect requested but the agent has no MCP registry")
    configured = set(registry.discovered())
    unknown = [name for name in names if name not in configured]
    if unknown:
        raise ValueError(f"Unknown MCP auto-connect server(s): {', '.join(unknown)}")
    for name in names:
        await registry.connect([name])


async def run_headless(
    prompt: str,
    *,
    config: Any,
    ephemeral: bool = False,
    continue_session: bool = False,
    resume_session_id: str | None = None,
    store: SessionStore | None = None,
    llm: Any | None = None,
    agent_cls: type | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> HeadlessResult:
    """Run one prompt without terminal UI ownership and return structured output."""
    if ephemeral and (continue_session or resume_session_id):
        raise ValueError("--ephemeral cannot be combined with --continue or --resume")

    from nooa_cli.tui.config import get_llm, load_agent_class

    workspace = Path(config.agent.working_dir).expanduser().resolve()
    handle: SessionHandle | None = None
    resumed = False
    session_id: str | None = None
    run_id = str(uuid.uuid4())
    storage: Any = InMemoryStorageManager()
    agent: Any | None = None
    dispatcher: InteractiveSessionDispatcher | None = None
    messages: list[str] = []
    usage = HeadlessUsage()
    unsubscribers: list[Any] = []
    final: HeadlessResult | None = None
    cancellation: asyncio.CancelledError | None = None

    def emit(event_type: str, **payload: Any) -> None:
        if on_event is not None:
            on_event(
                {
                    "schema_version": 1,
                    "type": event_type,
                    "timestamp": datetime.now(UTC).isoformat(),
                    "session_id": session_id,
                    "run_id": run_id,
                    **payload,
                }
            )

    try:
        llm = llm or get_llm(config)
        if agent_cls is None:
            agent_cls = (
                load_agent_class(config.tui.agent_spec) if config.tui.agent_spec else CodingAgent
            )
        if not ephemeral:
            store = store or SessionStore(workspace / ".nooa" / "sessions")
            handle, resumed = resolve_session(
                store,
                working_directory=workspace,
                model=config.tui.default_model,
                agent_name=agent_cls.__name__,
                continue_session=continue_session,
                resume_session_id=resume_session_id,
            )
            session_id = handle.id
            storage = handle.storage
        emit("session.started", resumed=resumed, ephemeral=ephemeral)
        from nooa_cli.tui.bootstrap import _enable_tracing
        from nooa_cli.tui.session_manager import _make_trace_session_name

        _tracing_enabled, set_trace_session = _enable_tracing(config, [])
        if set_trace_session is not None:
            set_trace_session(_make_trace_session_name(session_id or run_id))
        agent = _instantiate_agent(agent_cls, config=config, llm=llm, storage=storage)
        _configure_skills(agent, config)
        if handle is not None:
            if resumed:
                handle.storage.restore_latest_snapshot(agent)
            agent._session_manager = handle

        def on_message(event: Any) -> None:
            if isinstance(event, AgentMessage):
                messages.append(event.content)
                emit("agent.message", content=event.content)

        def on_usage(event: Any) -> None:
            if not isinstance(event, LLMComplete):
                return
            usage.prompt_tokens += event.prompt_tokens
            usage.completion_tokens += event.completion_tokens
            usage.cached_tokens += event.cached_tokens
            usage.reasoning_tokens += event.reasoning_tokens
            usage.cost_usd += event.cost_usd
            emit("usage.updated", usage=asdict(usage))

        unsubscribers.extend(
            (
                agent.event_manager.on("AgentMessage", on_message),
                agent.event_manager.on("LLMComplete", on_usage),
            )
        )
        if session_id is not None:
            agent.event_manager.add(SessionResumed(session_id=session_id, restored=resumed))
        if handle is not None:
            handle.record_user_message(prompt)

        try:
            await _connect_startup_mcp(agent, config)
        except Exception as exc:
            from nooa_cli.tui.mcp_approval import MCPApprovalRequired

            if not isinstance(exc, MCPApprovalRequired):
                raise
            messages.append(str(exc))
            blocked_result = HeadlessResult(
                session_id=session_id,
                run_id=run_id,
                status="blocked",
                messages=messages,
                explanation="MCP approval required",
                usage=usage,
            )
            raise _TurnFinished(blocked_result) from None

        emit("turn.started")
        dispatcher = InteractiveSessionDispatcher(agent)
        result = await dispatcher.submit(prompt)
        if result is None:
            final = HeadlessResult(
                session_id=session_id,
                run_id=run_id,
                status="cancelled",
                messages=messages,
                explanation="cancelled",
                usage=usage,
            )
            raise _TurnFinished(final)
        blocked = result.kind in {RespondReason.NEED_INPUT, RespondReason.GET_USER_INPUT}
        final = HeadlessResult(
            session_id=session_id,
            run_id=run_id,
            status="blocked" if blocked else "done",
            messages=messages,
            explanation=result.explanation,
            usage=usage,
        )
        raise _TurnFinished(final)
    except _TurnFinished as finished:
        final = finished.result
    except asyncio.CancelledError as exc:
        cancellation = exc
        final = HeadlessResult(
            session_id=session_id,
            run_id=run_id,
            status="cancelled",
            messages=messages,
            explanation="cancelled",
            usage=usage,
        )
    except Exception as exc:
        final = HeadlessResult(
            session_id=session_id,
            run_id=run_id,
            status="failed",
            messages=messages,
            explanation=str(exc),
            usage=usage,
            error={"type": type(exc).__name__, "message": str(exc)},
        )
    finally:
        cleanup_error: Exception | None = None

        def capture_cleanup_error(exc: Exception) -> None:
            nonlocal cleanup_error
            if cleanup_error is None:
                cleanup_error = exc

        for unsubscribe in unsubscribers:
            try:
                unsubscribe()
            except Exception as exc:
                capture_cleanup_error(exc)
        if handle is not None and agent is not None:
            try:
                handle.storage.save_snapshot(agent)
            except Exception as exc:
                capture_cleanup_error(exc)
        try:
            if dispatcher is not None:
                await dispatcher.close()
            elif agent is not None:
                await agent.close()
            elif llm is not None:
                await llm.aclose()
        except Exception as exc:
            capture_cleanup_error(exc)
        if handle is not None:
            try:
                handle.close()
            except Exception as exc:
                capture_cleanup_error(exc)

        if cleanup_error is not None and cancellation is None:
            final = HeadlessResult(
                session_id=session_id,
                run_id=run_id,
                status="failed",
                messages=messages,
                explanation=f"Resource cleanup failed: {cleanup_error}",
                usage=usage,
                error={
                    "type": type(cleanup_error).__name__,
                    "message": str(cleanup_error),
                },
            )

    assert final is not None
    terminal_type = {
        "done": "turn.completed",
        "blocked": "turn.blocked",
        "cancelled": "turn.cancelled",
        "failed": "turn.failed",
    }[final.status]
    emit(terminal_type, result=final.to_dict())
    if cancellation is not None:
        raise cancellation
    return final


__all__ = ["HeadlessResult", "HeadlessUsage", "resolve_session", "run_headless"]
