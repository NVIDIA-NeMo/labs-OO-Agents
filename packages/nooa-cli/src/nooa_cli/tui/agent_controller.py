# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transactional ownership of one UI-facing agent observation."""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

from nooa_cli.interactive.state import AgentState, InteractiveAgent, Observation, UIScheduler

logger = logging.getLogger(__name__)


class AgentController:
    """Observe one agent and route direct Python control calls to it.

    Installation is transactional.  A transition first reserves command routing,
    then waits for any running callback *without* holding the routing lock.  A
    callback that attempts a command during that interval is rejected instead of
    participating in a lock cycle.  Cleanup failures are reported only after the
    controller has restored all bookkeeping invariants.
    """

    def __init__(
        self, scheduler: UIScheduler, on_change: Callable[[AgentState | None], None] | None = None
    ) -> None:
        self._scheduler, self._on_change = scheduler, on_change
        self._condition = threading.Condition(threading.RLock())
        self._installation_lock = threading.RLock()
        self._transitioning = False
        self._agent: InteractiveAgent | None = None
        self._observation: Observation | None = None
        self._state: AgentState | None = None
        self._failure: BaseException | None = None
        self._generation = 0
        self._delivering_thread: int | None = None
        self._notification_pending = False
        self._pending_state: AgentState | None = None

    @property
    def state(self) -> AgentState | None:
        with self._condition:
            self._drop_failed_observation_locked()
            return self._state

    @property
    def failure(self) -> BaseException | None:
        """Terminal failure of the active observation, if it disconnected."""
        with self._condition:
            self._drop_failed_observation_locked()
            return self._failure

    def observe(self, agent: InteractiveAgent) -> AgentState:
        """Transactionally replace the current agent and observation."""
        self._check_not_in_callback()
        self._begin_transition()
        observation: Observation | None = None
        committed = False
        callback: Callable[[AgentState | None], None] | None = None
        old: Observation | None = None
        close_error: BaseException | None = None
        try:
            with self._condition:
                generation = self._generation + 1
            candidate = agent.state
            active = False

            def changed(state: AgentState) -> None:
                nonlocal candidate
                queued: Callable[[AgentState | None], None] | None = None
                with self._condition:
                    candidate = state
                    if active and generation == self._generation:
                        self._state = state
                        queued = self._queue_callback_locked(state)
                if queued is not None:
                    self._drain_callbacks(queued)

            def terminated(failure: BaseException | None) -> None:
                self._observation_terminated(generation, failure)

            observation = agent.observe(changed, self._scheduler, terminated)
            with self._condition:
                self._wait_for_delivery_locked()
                if observation.failure is not None:
                    raise observation.failure
                old = self._observation
                self._generation = generation
                active = True
                self._agent, self._observation, self._state = agent, observation, candidate
                self._failure = None
                callback = self._queue_callback_locked(candidate)
                committed = True
            if old is not None:
                try:
                    old.close()
                except BaseException as exc:
                    close_error = exc
        except BaseException:
            if observation is not None and not committed:
                try:
                    observation.close()
                except BaseException:
                    logger.warning("candidate observation cleanup failed", exc_info=True)
            raise
        finally:
            self._end_transition()
            if callback is not None:
                self._drain_callbacks(callback)
        if close_error is not None:
            raise close_error
        return candidate

    def close(self) -> None:
        """Stop observing without stopping the agent."""
        self._check_not_in_callback()
        self._begin_transition()
        callback: Callable[[AgentState | None], None] | None = None
        close_error: BaseException | None = None
        try:
            with self._condition:
                self._wait_for_delivery_locked()
                old = self._observation
                self._generation += 1
                self._agent = self._observation = self._state = None
                self._failure = None
                callback = self._queue_callback_locked(None)
            if old is not None:
                try:
                    old.close()
                except BaseException as exc:
                    close_error = exc
        finally:
            self._end_transition()
            if callback is not None:
                self._drain_callbacks(callback)
        if close_error is not None:
            raise close_error

    def submit(self, text: str) -> bool:
        return self._route(lambda agent: agent.submit(text))

    def interrupt(self) -> bool:
        return self._route(lambda agent: agent.interrupt())

    def withdraw_pending_input(self) -> str | None:
        return self._route(lambda agent: agent.withdraw_pending_input())

    def stop(self) -> bool:
        return self._route(lambda agent: agent.stop())

    def _route(self, command: Callable[[InteractiveAgent], object]):
        with self._installation_lock:
            if self._transitioning:
                raise RuntimeError("agent transition is in progress")
            with self._condition:
                self._drop_failed_observation_locked()
                agent = self._agent
            if agent is None:
                raise RuntimeError("frontend is not observing an agent")
            return command(agent)

    def _begin_transition(self) -> None:
        with self._installation_lock:
            if self._transitioning:
                raise RuntimeError("another agent transition is in progress")
            self._transitioning = True

    def _end_transition(self) -> None:
        with self._installation_lock:
            self._transitioning = False

    def _observation_terminated(self, generation: int, failure: BaseException | None) -> None:
        if failure is None:
            return
        callback: Callable[[AgentState | None], None] | None = None
        with self._condition:
            if generation != self._generation:
                return
            self._generation += 1
            self._agent = self._observation = self._state = None
            self._failure = failure
            callback = self._queue_callback_locked(None)
        logger.error(
            "interactive-agent observation terminated",
            exc_info=(type(failure), failure, failure.__traceback__),
        )
        if callback is not None:
            try:
                self._drain_callbacks(callback)
            except BaseException:
                logger.exception("agent termination callback failed")

    def _drop_failed_observation_locked(self) -> None:
        observation = self._observation
        if observation is not None and observation.failure is not None:
            self._generation += 1
            self._failure = observation.failure
            self._agent = self._observation = self._state = None

    def _check_not_in_callback(self) -> None:
        with self._condition:
            if self._delivering_thread == threading.get_ident():
                raise RuntimeError("agent transitions cannot run inside an observer callback")

    def _wait_for_delivery_locked(self) -> None:
        while self._delivering_thread is not None:
            self._condition.wait()

    def _queue_callback_locked(
        self, state: AgentState | None
    ) -> Callable[[AgentState | None], None] | None:
        callback = self._on_change
        if callback is None:
            return None
        self._pending_state = state
        self._notification_pending = True
        if self._delivering_thread is not None:
            return None
        self._delivering_thread = threading.get_ident()
        return callback

    def _drain_callbacks(self, callback: Callable[[AgentState | None], None]) -> None:
        """Deliver the latest state serially, including reentrant publications."""
        try:
            while True:
                with self._condition:
                    if not self._notification_pending:
                        return
                    state = self._pending_state
                    self._notification_pending = False
                callback(state)
        finally:
            with self._condition:
                self._delivering_thread = None
                self._condition.notify_all()
