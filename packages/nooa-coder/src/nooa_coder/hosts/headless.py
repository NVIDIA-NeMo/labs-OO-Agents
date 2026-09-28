# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Headless host: one in-process session tree, no transport.

Used by benchmarks and scripts::

    async with open_tree(options) as tree:
        outcome = await tree.root.prompt(task_description)

``run_task`` runs one task unattended to its end and reports it::

    run = await run_task(options, task_description)
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from nooa.interactive import Done
from nooa_coder.session.items import (
    AgentEventUpdate,
    SessionEvent,
    TaskResult,
    TurnCancelledOutcome,
    TurnEndedUpdate,
    Usage,
)
from nooa_coder.session.loader import AgentFactory
from nooa_coder.session.options import SessionOptions
from nooa_coder.session.registry import LLMFactory, SessionRegistry
from nooa_coder.session.session import Session, TurnFailedError
from nooa_coder.session.store import SessionStore, sessions_root

DEFAULT_MAX_TURNS = 10


@dataclass(frozen=True)
class Tree:
    """The registry of an open tree and its root session."""

    registry: SessionRegistry
    root: Session


@asynccontextmanager
async def open_tree(
    options: SessionOptions,
    *,
    agent_factory: AgentFactory | None = None,
    llm_factory: LLMFactory | None = None,
) -> AsyncIterator[Tree]:
    """Create a registry and a root session; close every session on exit.

    Sessions are closed children first, on normal exit and on an
    exception, so a crashed run leaves no live claim on any session file.
    Sessions are stored in ``sessions_root(options.workspace,
    options.sessions_dir)``. ``agent_factory`` builds each agent (default:
    ``create_session_agent``). ``llm_factory`` builds the model client of a
    session whose options carry none (see ``SessionRegistry``). The root's
    turn loop is traced when ``nooa.tracing`` is enabled.
    """
    if agent_factory is None:
        from nooa_coder.coding.factory import create_session_agent

        agent_factory = create_session_agent
    store = SessionStore(sessions_root(options.workspace, options.sessions_dir))
    registry = SessionRegistry(store, agent_factory=agent_factory, llm_factory=llm_factory)
    try:
        root = await registry.create(options, prepare=_trace_turns)
        yield Tree(registry=registry, root=root)
    finally:
        await registry.close_all()


async def _trace_turns(session: Session) -> None:
    """Register the tracing hooks in the session's turn loop, whose context is fresh.

    Without them a traced run records no agent or model spans. Spans carry
    the session's id as their trace session.
    """
    try:
        import nooa.tracing as tracing
    except ImportError:
        return

    def enter() -> None:
        tracing.set_session(session.id)
        tracing.register_hooks_in_current_context()

    session.add_loop_context_hook(enter)


@dataclass
class TaskRun:
    """How a ``run_task`` run ended.

    ``done`` is the final ``Done``, or ``None`` when the run stopped
    without one; ``stopped`` then says why (a limit, a cancel, a failed
    turn). ``usage`` is the root session's usage, its children's
    attributed usage included. ``events`` are the root agent's events in
    order, for trajectory exports.
    """

    session_id: str
    done: Done | None
    stopped: str | None
    turns: int
    usage: Usage
    events: list[Any]

    @property
    def result(self) -> TaskResult | None:
        """The ``TaskResult`` of the final ``Done``, if it has one."""
        value = self.done.result if self.done is not None else None
        if isinstance(value, TaskResult) or value is None:
            return value
        try:
            return TaskResult.model_validate(value)
        except ValueError:
            return None


async def run_task(
    options: SessionOptions,
    task: str,
    *,
    max_turns: int = DEFAULT_MAX_TURNS,
    timeout: float | None = None,
    agent_factory: AgentFactory | None = None,
    llm_factory: LLMFactory | None = None,
) -> TaskRun:
    """Run ``task`` unattended in a new session tree until the agent is done.

    The root session runs ``handle_batch`` turns, so it never asks a
    question. A turn that ends ``Waiting`` is followed by the turn its
    job's delivery starts. The run stops early when ``max_turns`` turns
    have ended and the last one is waiting, or after ``timeout`` seconds;
    the running turn is then cancelled. The session is stored like any
    other, so it can be opened afterwards. ``llm_factory`` defaults to
    ``default_llm_factory()``: a session without a client (a root without
    ``options.llm``, or a child on another model) gets one for its model.
    """
    if llm_factory is None:
        from nooa_coder.coding.factory import default_llm_factory

        llm_factory = default_llm_factory()
    options = options.model_copy(update={"turn_method": "handle_batch"})
    events: list[Any] = []
    turns = 0
    waiting_on: list[str] = []
    limit_reached = asyncio.Event()

    def listen(update: SessionEvent) -> None:
        nonlocal turns
        if isinstance(update, AgentEventUpdate):
            events.append(update.event)
        elif isinstance(update, TurnEndedUpdate):
            turns += 1
            if update.outcome_kind == "waiting" and turns >= max_turns:
                waiting_on[:] = update.outcome.get("on", [])
                limit_reached.set()

    async with open_tree(options, agent_factory=agent_factory, llm_factory=llm_factory) as tree:
        root = tree.root
        root.subscribe(listen)
        prompt = asyncio.ensure_future(root.prompt(task))
        limit = asyncio.ensure_future(limit_reached.wait())
        try:
            finished, _ = await asyncio.wait(
                {prompt, limit}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            limit.cancel()
        stopped: str | None = None
        if prompt not in finished:
            if limit_reached.is_set():
                stopped = (
                    f"turn limit ({max_turns}) reached while waiting on {', '.join(waiting_on)}"
                )
            else:
                stopped = f"time limit ({timeout:g} s) reached"
            await root.cancel(by="headless")
        done: Done | None = None
        try:
            outcome = await prompt
        except TurnFailedError as exc:
            stopped = stopped or f"turn failed: {exc}"
        else:
            if isinstance(outcome, Done):
                done = outcome
            elif isinstance(outcome, TurnCancelledOutcome):
                stopped = stopped or f"cancelled by {outcome.by}"
            else:
                stopped = stopped or f"ended with {type(outcome).__name__}"
        usage = root.info.usage.model_copy()
        session_id = root.id
    return TaskRun(
        session_id=session_id,
        done=done,
        stopped=stopped,
        turns=turns,
        usage=usage,
        events=events,
    )
