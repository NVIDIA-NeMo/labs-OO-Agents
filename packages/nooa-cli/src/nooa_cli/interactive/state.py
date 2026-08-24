# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Python-native state and control contract for interactive agents."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class AgentLifecycle(StrEnum):
    STARTING = "starting"
    IDLE = "idle"
    THINKING = "thinking"
    WAITING = "waiting"
    STOPPED = "stopped"
    FAILED = "failed"


class CancellationState(StrEnum):
    NONE = "none"
    REQUESTED = "requested"


class AgentJobState(StrEnum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class AgentJobSummary:
    name: str
    label: str
    state: AgentJobState
    queued: int
    values: tuple[str, ...] = ()
    job_id: str = ""


@dataclass(frozen=True, slots=True)
class AgentWorkspaceState:
    pending_inputs: tuple[str, ...] = ()
    pending_commands: tuple[str, ...] = ()
    working_directory: str | None = None
    context_summary: str | None = None
    todo_summary: str | None = None
    memory_summary: str | None = None
    cancellation: CancellationState = CancellationState.NONE
    jobs: tuple[AgentJobSummary, ...] = ()


@dataclass(frozen=True, slots=True)
class AgentState:
    """Deeply immutable presentation snapshot of an interactive agent."""

    agent_id: str
    display_name: str
    parent_agent_id: str | None
    lifecycle: AgentLifecycle
    workspace: AgentWorkspaceState = AgentWorkspaceState()

    @property
    def pending_inputs(self) -> tuple[str, ...]:
        return self.workspace.pending_inputs

    @property
    def pending_commands(self) -> tuple[str, ...]:
        return self.workspace.pending_commands

    @property
    def working_directory(self) -> str | None:
        return self.workspace.working_directory

    @property
    def context_summary(self) -> str | None:
        return self.workspace.context_summary

    @property
    def todo_summary(self) -> str | None:
        return self.workspace.todo_summary

    @property
    def memory_summary(self) -> str | None:
        return self.workspace.memory_summary


class UIScheduler(Protocol):
    """Marshal one callback onto the UI owner; may run inline or raise."""

    def schedule(self, callback: Callable[[], None]) -> None: ...


class Observation(Protocol):
    """A non-owning, idempotently closeable state subscription.

    Closing prevents a queued callback from starting; a callback already claimed
    by a scheduler may finish. Closing an observation never stops its agent.
    ``failure`` records terminal scheduler/listener failure and ``on_terminated``
    receives that failure exactly once, allowing owners to gate stale controls.
    """

    @property
    def failure(self) -> BaseException | None: ...
    def close(self) -> None: ...


class InteractiveAgent(Protocol):
    """Direct Python interface consumed by an interactive frontend.

    ``state`` is a deeply immutable, thread-safe latest snapshot. ``observe``
    atomically registers against that state and schedules an initial delivery;
    implementations coalesce later updates per observation and serialize its
    callbacks. They invoke neither scheduler nor callback while holding the
    agent state lock.

    A true command result means admission, not completion. The admitted change
    is visible through ``state`` before return. ``stop`` is non-blocking and an
    accepted stop eventually publishes ``STOPPED``; observation ownership is
    independent from agent lifecycle ownership.
    """

    @property
    def state(self) -> AgentState: ...
    def observe(
        self,
        callback: Callable[[AgentState], None],
        scheduler: UIScheduler,
        on_terminated: Callable[[BaseException], None] | None = None,
    ) -> Observation: ...
    def submit(self, text: str) -> bool: ...
    def interrupt(self) -> bool: ...
    def withdraw_pending_input(self) -> str | None: ...
    def stop(self) -> bool: ...
