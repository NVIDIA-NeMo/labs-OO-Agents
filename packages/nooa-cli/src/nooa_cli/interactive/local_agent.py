# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Python-native boundary for running and observing an in-process NOOA agent.

All knowledge of the current concrete agent queues is confined here.
``LocalAgentRunner`` owns the optional worker event loop;
observations are read-only leases and never own that lifecycle.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import threading
from collections.abc import Awaitable, Callable
from concurrent.futures import Future as ConcurrentFuture
from dataclasses import replace
from typing import Any

from nooa_cli.interactive.runtime import JobSnapshot
from nooa_cli.interactive.state import (
    AgentJobState,
    AgentJobSummary,
    AgentLifecycle,
    AgentState,
    AgentWorkspaceState,
    CancellationState,
    Observation,
)

logger = logging.getLogger(__name__)

_CALLBACK_OWNER_ATTRIBUTE = "_nooa_local_agent_runner_owner"
_CALLBACK_LEASE_LOCK = threading.RLock()


async def _stop_litellm_worker() -> None:
    """Stop litellm's global worker before its owning loop is torn down."""
    try:
        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

        await GLOBAL_LOGGING_WORKER.stop()
    except Exception:
        pass


class _LocalObservation:
    def __init__(self, owner, callback, scheduler, on_terminated):
        self._owner, self._callback, self._scheduler = owner, callback, scheduler
        self._on_terminated = on_terminated
        self._revision = 0
        self._delivered = -1
        self._scheduled = self._running = self._closed = False
        self._failure = None

    @property
    def failure(self):
        with self._owner._state_lock:
            return self._failure

    def close(self):
        with self._owner._state_lock:
            if self._closed:
                return
            self._closed = True
            self._owner._observations.discard(self)

    def _fail(self, exc):
        with self._owner._state_lock:
            if self._failure is not None:
                return
            self._failure = exc
            self._scheduled = self._running = False
            self._closed = True
            self._owner._observations.discard(self)
        if self._on_terminated is not None:
            try:
                self._on_terminated(exc)
            except BaseException:
                logger.exception("interactive-agent termination listener failed")

    def _changed_locked(self):
        if self._closed or self._failure is not None:
            return False
        self._revision += 1
        if self._scheduled or self._running:
            return False
        self._scheduled = True
        return True

    def _schedule(self, initial=False):
        try:
            self._scheduler.schedule(self._drain)
        except BaseException as exc:
            self._fail(exc)
            if initial:
                raise
            logger.error("interactive-agent scheduler failed", exc_info=exc)

    def _drain(self):
        while True:
            with self._owner._state_lock:
                self._scheduled = False
                if self._closed or self._failure is not None or self._delivered == self._revision:
                    self._running = False
                    return
                self._running = True
                revision, state = self._revision, self._owner._state
            try:
                self._callback(state)
            except BaseException as exc:
                self._fail(exc)
                logger.error("interactive-agent observer failed", exc_info=exc)
                return
            with self._owner._state_lock:
                self._delivered = revision
                if self._closed or self._delivered == self._revision:
                    self._running = False
                    return


class LocalAgentRunner:
    """Own the local agent callback bridge, dispatch task, and worker loop.

    The runner is deliberately a composition-root service rather than part of a
    renderer. ``activate`` opts into a dedicated worker loop; without it tests
    and embedding hosts retain same-loop behavior.
    """

    @property
    def state(self) -> AgentState:
        with self._state_lock:
            return self._state

    def observe(self, callback, scheduler, on_terminated=None) -> Observation:
        observation = _LocalObservation(self, callback, scheduler, on_terminated)
        with self._state_lock:
            self._observations.add(observation)
            observation._changed_locked()
        observation._schedule(initial=True)
        return observation

    def interrupt(self) -> bool:
        return self.request_cancel()

    def withdraw_pending_input(self) -> str | None:
        accepted, text = self._withdraw_pending_input()
        return text if accepted else None

    def stop(self) -> bool:
        return self.request_stop()

    def _commit_state(self, transform: Callable[[AgentState], AgentState]) -> bool:
        """Apply one state transition atomically and schedule observers after unlock."""
        with self._state_lock:
            state = transform(self._state)
            if state == self._state:
                return False
            self._state = state
            schedules = tuple(o for o in self._observations if o._changed_locked())
        for observation in schedules:
            observation._schedule()
        return True

    def _set_lifecycle(self, lifecycle: AgentLifecycle) -> None:
        def transition(state: AgentState) -> AgentState:
            if state.lifecycle is AgentLifecycle.STOPPED or state.lifecycle is lifecycle:
                return state
            return replace(state, lifecycle=lifecycle)

        self._commit_state(transition)

    def _update_pending_inputs(self, pending_inputs: tuple[str, ...]) -> None:
        self._commit_state(
            lambda state: replace(
                state, workspace=replace(state.workspace, pending_inputs=pending_inputs)
            )
        )

    def _update_runtime_projection(
        self,
        *,
        cancellation: CancellationState,
        jobs: tuple[AgentJobSummary, ...],
    ) -> None:
        self._commit_state(
            lambda state: replace(
                state,
                workspace=replace(state.workspace, cancellation=cancellation, jobs=jobs),
            )
        )

    def _update_agent_workspace(
        self, *, pending_inputs: tuple[str, ...], working_directory: str | None
    ) -> None:
        self._commit_state(
            lambda state: replace(
                state,
                workspace=replace(
                    state.workspace,
                    pending_inputs=pending_inputs,
                    working_directory=working_directory,
                    cancellation=CancellationState.NONE,
                    jobs=(),
                ),
            )
        )

    def _close_state(self) -> None:
        self._commit_state(
            lambda state: replace(
                state,
                lifecycle=AgentLifecycle.STOPPED,
                workspace=replace(state.workspace, cancellation=CancellationState.NONE, jobs=()),
            )
        )

    def __init__(
        self,
        agent: Any,
        *,
        emit_text: Callable[[str], None],
        agent_id: str,
        display_name: str | None = None,
        parent_agent_id: str | None = None,
        on_state_change: Callable[[], None] | None = None,
        on_stop_reason: Callable[[Any, str], Awaitable[None] | None] | None = None,
        on_before_handle: Callable[[Any], Awaitable[None] | None] | None = None,
        on_after_handle: Callable[[Any, Any], Awaitable[None] | None] | None = None,
        on_notification: Callable[[dict[str, list[Any]]], None] | None = None,
        dispatcher_exit: type[BaseException] | None = None,
        on_cancelled: Callable[[], None] | None = None,
        bind_callbacks: bool = True,
    ) -> None:
        self._agent = agent
        self._queue_manager = agent.queue_manager
        self._previous_notify_callback = getattr(self._queue_manager, "_notify_callback", None)
        self._user_messages = agent._user_messages_in
        self._emit_text = emit_text
        self._on_state_change = on_state_change
        self._on_stop_reason = on_stop_reason
        self._on_before_handle = on_before_handle
        self._on_after_handle = on_after_handle
        self._on_notification = on_notification
        self._dispatcher_exit = dispatcher_exit
        self._on_cancelled = on_cancelled
        self._ui_loop: asyncio.AbstractEventLoop | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._shutdown_coordinator: threading.Thread | None = None
        self._ready = threading.Event()
        self._task: asyncio.Future[Any] | None = None
        self._source_future: ConcurrentFuture[Any] | None = None
        self._source_task: asyncio.Task[Any] | None = None
        self._cancel_requested = False
        self._notify_cancelled = False
        self._suspend_restart = 0
        self._in_handle = False
        self._waiting_for_input = False
        self._closed = False
        self._bound = False
        self._lifecycle_state = "active"
        self._binding_generation = 0
        self._shutdown_started = False
        self._shutdown_complete = threading.Event()
        self._shutdown_error: BaseException | None = None
        self._stop_task: asyncio.Task[None] | None = None
        self._resources_shutdown = False
        self._lifecycle_lock = threading.RLock()
        self._callback_lock = threading.RLock()
        self._previous_user_on_get = getattr(self._user_messages, "_on_get", None)
        self._owned_user_on_get: Callable[[str], None] | None = None
        self._user_message_accepted_callback: Callable[[str], None] | None = None
        self._pending_user_messages = tuple(
            item for item in self._user_messages.snapshot() if isinstance(item, str)
        )
        self._state_lock = threading.RLock()
        self._observations: set[_LocalObservation] = set()
        self._state = AgentState(
            agent_id,
            display_name or type(agent).__name__,
            parent_agent_id,
            AgentLifecycle.IDLE,
            workspace=AgentWorkspaceState(
                pending_inputs=self._pending_user_messages,
                working_directory=str(agent.cwd)
                if getattr(agent, "cwd", None) is not None
                else None,
            ),
        )
        if bind_callbacks:
            self.bind()

    @property
    def closed(self) -> bool:
        with self._lifecycle_lock:
            return self._closed

    @property
    def is_thinking(self) -> bool:
        with self._lifecycle_lock:
            return self._cancel_requested or (
                self._task is not None and not self._task.done() and self._in_handle
            )

    @property
    def is_quiescent(self) -> bool:
        """Return whether all currently admitted agent work has settled."""
        with self._lifecycle_lock:
            task_active = self._task is not None and not self._task.done()
            return (
                not self._in_handle
                and not self._cancel_requested
                and (not task_active or self._waiting_for_input)
                and self._user_messages.qsize() == 0
                and not self.has_pending_work()
            )

    async def wait_quiescent(self) -> None:
        """Wait on the owner loop until no admitted work remains."""
        while True:
            if self.is_quiescent:
                # A queue getter can consume an item before its dispatcher
                # continuation runs.  Yield on this same loop and require a
                # second idle observation so that claimed work becomes visible.
                await asyncio.sleep(0)
                if self.is_quiescent:
                    return
            await asyncio.sleep(0.01)

    @property
    def cancel_requested(self) -> bool:
        with self._lifecycle_lock:
            return self._cancel_requested

    @property
    def task(self) -> asyncio.Future[Any] | None:
        with self._lifecycle_lock:
            return self._task

    @property
    def in_handle(self) -> bool:
        with self._lifecycle_lock:
            return self._in_handle

    def bind(self) -> None:
        """Exclusively acquire the concrete agent's callback bridge."""
        with self._lifecycle_lock, self._callback_lock:
            if self._lifecycle_state != "active":
                raise RuntimeError("local agent runner is not active")
            self._acquire_callbacks()

    def _acquire_callbacks(self) -> None:
        """Acquire callbacks while lifecycle and callback locks are held."""
        with _CALLBACK_LEASE_LOCK:
            self._acquire_callbacks_locked()

    def _acquire_callbacks_locked(self) -> None:
        """Acquire the process-local callback lease under its global lock."""
        if self._bound:
            return
        owner = getattr(self._queue_manager, _CALLBACK_OWNER_ATTRIBUTE, None)
        if owner is not None and owner is not self:
            raise RuntimeError("local agent already has an active runner")
        self._previous_notify_callback = getattr(self._queue_manager, "_notify_callback", None)
        self._previous_user_on_get = getattr(self._user_messages, "_on_get", None)
        setattr(self._queue_manager, _CALLBACK_OWNER_ATTRIBUTE, self)
        try:
            self._queue_manager.set_notify_callback(self._on_queue_notify)
            generation = self._binding_generation
            user_messages = self._user_messages
            previous_user_on_get = self._previous_user_on_get

            def on_user_message_get(text: str) -> None:
                self._on_user_message_get(
                    text,
                    generation=generation,
                    user_messages=user_messages,
                    previous=previous_user_on_get,
                )

            self._owned_user_on_get = on_user_message_get
            self._user_messages.set_on_get(on_user_message_get)
        except BaseException:
            current_notify = getattr(self._queue_manager, "_notify_callback", None)
            if getattr(current_notify, "__self__", None) is self:
                self._queue_manager.set_notify_callback(self._previous_notify_callback)
            current_on_get = getattr(self._user_messages, "_on_get", None)
            if current_on_get is self._owned_user_on_get:
                self._user_messages.set_on_get(self._previous_user_on_get)
            self._owned_user_on_get = None
            if getattr(self._queue_manager, _CALLBACK_OWNER_ATTRIBUTE, None) is self:
                delattr(self._queue_manager, _CALLBACK_OWNER_ATTRIBUTE)
            raise
        self._bound = True

    def set_dispatch_hooks(
        self,
        *,
        on_state_change: Callable[[], None] | None = None,
        on_before_handle: Callable[[Any], Awaitable[None] | None] | None = None,
        on_after_handle: Callable[[Any, Any], Awaitable[None] | None] | None = None,
        on_notification: Callable[[dict[str, list[Any]]], None] | None = None,
        dispatcher_exit: type[BaseException] | None = None,
        on_cancelled: Callable[[], None] | None = None,
    ) -> None:
        """Install host policy hooks without exposing concrete queues to the host."""
        self._on_state_change = on_state_change
        self._on_before_handle = on_before_handle
        self._on_after_handle = on_after_handle
        self._on_notification = on_notification
        if dispatcher_exit is not None:
            self._dispatcher_exit = dispatcher_exit
        if on_cancelled is not None:
            self._on_cancelled = on_cancelled

    def activate(self, ui_loop: asyncio.AbstractEventLoop) -> None:
        """Establish the UI owner before any asyncio dispatcher is created."""
        with self._lifecycle_lock:
            if self._lifecycle_state != "active":
                raise RuntimeError("local agent runner is not active")
            self._ui_loop = ui_loop
            self._ensure_loop()
        if self._user_messages.qsize() > 0 or self.has_pending_work():
            self._marshal_to_ui_owner(lambda: self.ensure_dispatcher(start_with_race=True))

    def _present(self, text: str) -> None:
        """Publish visible runtime text through the renderer-owned sink."""
        # close()/swap_agent() take the same lock before restoring callbacks,
        # so no callback can begin after either operation returns.
        with self._callback_lock:
            if self._closed:
                return
            self._emit_text(text)

    def _on_ui_owner(self) -> bool:
        loop = self._ui_loop
        if loop is None or not loop.is_running():
            return True
        try:
            return asyncio.get_running_loop() is loop
        except RuntimeError:
            return False

    def _marshal_to_ui_owner(self, callback: Callable[[], None]) -> bool | None:
        """Run now, enqueue on the owner, or return ``None`` if enqueue fails."""
        loop = self._ui_loop
        if loop is None:
            try:
                candidate = asyncio.get_running_loop()
            except RuntimeError:
                return True
            with self._lifecycle_lock:
                if self._ui_loop is None:
                    self._ui_loop = candidate
                loop = self._ui_loop
        if not loop.is_running():
            # A recorded owner that has stopped cannot accept work.  Treat this
            # as rejection; callers with transactional state can roll back.
            return None
        if self._on_ui_owner():
            callback()
            return False
        try:
            loop.call_soon_threadsafe(callback)
        except RuntimeError:
            # The loop may close after ``is_running()``. Callers that require
            # admission guarantees can roll back or choose another coordinator.
            return None
        return True

    def submit(self, text: str) -> bool:
        """Atomically admit input only when its dispatcher wakeup is viable."""
        with self._lifecycle_lock:
            if self._lifecycle_state != "active":
                return False
            before = tuple(self._user_messages.snapshot())
            tail = self._user_messages.pop_last()
            if isinstance(tail, str):
                self._user_messages.put(f"{tail}\n{text}")
            else:
                if tail is not None:
                    self._user_messages.put(tail)
                self._user_messages.put(text)
            pending_inputs = tuple(
                item for item in self._user_messages.snapshot() if isinstance(item, str)
            )
            task_running = self._task is not None and not self._task.done()
            if not task_running and self._marshal_to_ui_owner(self.ensure_dispatcher) is None:
                # ``call_soon_threadsafe`` can lose a race with loop closure.
                # No dispatcher existed, so restoring this queue snapshot is
                # safe and turns the action into a correlated rejection.
                self._user_messages.flush()
                for item in before:
                    self._user_messages.put(item)
                self._pending_user_messages = tuple(
                    item for item in before if isinstance(item, str)
                )
                return False
            self._pending_user_messages = pending_inputs
            # Publish while dequeue is still excluded by the lifecycle lock.
            # Otherwise a newer empty dequeue snapshot can be overwritten by
            # this older submit snapshot after the lock is released.
            self._update_pending_inputs(pending_inputs)
        return True

    def ensure_dispatcher(self, *, start_with_race: bool = False) -> None:
        if (
            self._lifecycle_state != "active"
            or (self._task is not None and not self._task.done())
            or self._cancel_requested
        ):
            return
        if self._loop is None:
            self._task = asyncio.ensure_future(self._dispatch(start_with_race=start_with_race))
            self._source_task = self._task
        else:

            async def run_dispatcher() -> None:
                self._source_task = asyncio.current_task()
                try:
                    await self._dispatch(start_with_race=start_with_race)
                finally:
                    self._source_task = None

            self._source_future = asyncio.run_coroutine_threadsafe(run_dispatcher(), self._loop)
            self._task = asyncio.wrap_future(self._source_future)
        self._task.add_done_callback(self._on_done)
        self._changed()

    async def _wait_for_input(self, awaitable: Any) -> Any:
        """Mark the dispatcher idle only while it is blocked for new input."""
        with self._lifecycle_lock:
            self._waiting_for_input = True
        self._changed()
        try:
            return await awaitable
        finally:
            with self._lifecycle_lock:
                self._waiting_for_input = False
            self._changed()

    async def _dispatch(self, *, start_with_race: bool) -> None:
        qm = self._queue_manager
        if start_with_race:
            try:
                items = await self._wait_for_input(qm.race())
            except ValueError:
                return
            if not items:
                return
            notification = self._drain(qm, items)
        else:
            item = await self._wait_for_input(self._user_messages.get())
            notification = self._drain(qm, [("user_messages", item)])
        while True:
            if self._on_notification is not None:
                self._on_notification(notification)
            with self._lifecycle_lock:
                self._in_handle = True
            self._set_lifecycle(AgentLifecycle.THINKING)
            self._changed()
            try:
                if self._on_before_handle is not None:
                    value = self._on_before_handle(self._agent)
                    if value is not None:
                        await value
                result = await self._agent.handle(notification)
            except BaseException as exc:
                if self._dispatcher_exit is not None and isinstance(exc, self._dispatcher_exit):
                    return
                raise
            finally:
                with self._lifecycle_lock:
                    self._in_handle = False
                self._set_lifecycle(AgentLifecycle.WAITING)
                self._changed()
            if self._on_after_handle is not None:
                value = self._on_after_handle(self._agent, result)
                if value is not None:
                    await value
            explanation = getattr(result, "explanation", "")
            if explanation and self._on_stop_reason is not None:
                value = self._on_stop_reason(result.kind, explanation)
                if value is not None:
                    await value
            running = qm.running_handles()
            if running:
                now = datetime.datetime.now().strftime("%H:%M:%S")
                lines = "".join(f"  ⠿ {h.label}\n" for h in running)
                self._present(
                    f"\x1b[2m{now} waiting — {len(running)} job(s) running:\n{lines}\x1b[0m"
                )
            try:
                items = await self._wait_for_input(qm.race())
            except ValueError:
                return
            if running and items:
                now = datetime.datetime.now().strftime("%H:%M:%S")
                names = {name for name, _ in items}
                for handle in running:
                    if handle.name in names:
                        self._present(f"\x1b[32m  ✓ {handle.label} — {now}\x1b[0m\n")
            notification = self._drain(qm, items)

    @staticmethod
    def _drain(qm: Any, items: list[tuple[str, Any]]) -> dict[str, list[Any]]:
        pending: dict[str, list[Any]] = {}
        for name, value in items:
            pending.setdefault(name, []).append(value)
        for channel in qm.channels().values():
            if channel.mode == "queue":
                for value in channel.drain():
                    pending.setdefault(channel.name, []).append(value)
        return pending

    def _on_queue_notify(self) -> None:
        # Capture at the transition site. A fast worker-loop job can reach its
        # terminal state before the UI owner drains callbacks; reading mutable
        # handles later would then erase the observable RUNNING transition.
        projection = self._capture_runtime_projection()
        if projection is None:
            return

        def publish_and_dispatch() -> None:
            if not self._publish_runtime_projection(*projection):
                return
            self.ensure_dispatcher(start_with_race=True)

        self._marshal_to_ui_owner(publish_and_dispatch)

    def _on_done(self, task: asyncio.Future[Any]) -> None:
        if task is not self._task:
            return
        notify_cancelled = self._notify_cancelled
        self._cancel_requested = False
        self._notify_cancelled = False
        self._source_task = None
        self._source_future = None
        self._task = None
        self._set_lifecycle(AgentLifecycle.IDLE)
        if task.cancelled() and notify_cancelled and self._on_cancelled is not None:
            self._on_cancelled()
        if not task.cancelled():
            try:
                error = task.exception()
            except asyncio.CancelledError:
                error = None
            if error is not None:
                self._present(f"Agent error: {error}\n")
        self._changed()
        if self._suspend_restart or self._lifecycle_state != "active":
            return
        if self._user_messages.qsize() > 0:
            self.ensure_dispatcher()
        elif self.has_pending_work():
            self.ensure_dispatcher(start_with_race=True)

    def job_snapshots(self) -> tuple[JobSnapshot, ...]:
        """Return immutable projections read on the concrete runtime's owner loop."""
        return self.run(self._job_snapshots_on_owner)

    def _job_snapshots_on_owner(self) -> tuple[JobSnapshot, ...]:
        snapshots: list[JobSnapshot] = []
        channels = self._queue_manager.channels()
        for handle in self._queue_manager.handles():
            channel = channels.get(handle.name)
            snapshots.append(
                JobSnapshot(
                    name=handle.name,
                    label=handle.label,
                    state=handle.state,
                    queued=channel.qsize() if channel is not None else 0,
                    values=tuple(str(value) for value in handle.values),
                    job_id=handle.job_id,
                )
            )
        return tuple(snapshots)

    def pending_user_messages(self) -> tuple[str, ...]:
        """Return the adapter-maintained immutable pending-input projection."""
        with self._lifecycle_lock:
            return self._pending_user_messages

    def set_user_message_accepted_callback(self, callback: Callable[[str], None] | None) -> None:
        """Observe the concrete queue's consumed transition inside the adapter."""
        with self._callback_lock:
            self._user_message_accepted_callback = callback

    def _on_user_message_get(
        self,
        text: str,
        *,
        generation: int,
        user_messages: Any,
        previous: Callable[[str], None] | None,
    ) -> None:
        """Apply a dequeue only while its captured callback lease is current."""
        with self._lifecycle_lock, self._callback_lock:
            if (
                self._lifecycle_state != "active"
                or generation != self._binding_generation
                or user_messages is not self._user_messages
                or not self._bound
            ):
                return
            pending = list(self._pending_user_messages)
            try:
                pending.remove(text)
            except ValueError:
                pass
            self._pending_user_messages = tuple(pending)
            pending_inputs = self._pending_user_messages
            callback = self._user_message_accepted_callback
            self._update_pending_inputs(pending_inputs)
            if previous is not None and previous is not self._owned_user_on_get:
                previous(text)
            if callback is not None:
                callback(text)

    def submit_slash_result(self, result: Any) -> bool:
        """Route a slash result to the concrete agent channel, if present."""
        channel = getattr(self._agent, "_slash_commands_in", None)
        if channel is None:
            return False
        channel.put(result)
        return True

    def seed_and_swap(self, agent: Any, prompt: str) -> Awaitable[None]:
        """Atomically admit a replacement before seeding its private queue."""
        return self.swap_agent(agent, seed_prompt=prompt)

    def has_pending_work(self) -> bool:
        if self._queue_manager.running_handles():
            return True
        return any(
            name != "user_messages" and channel.mode == "queue" and not channel.is_empty()
            for name, channel in self._queue_manager.channels().items()
        )

    def queue_continuation(self, text: str) -> str | None:
        """Queue a host continuation, preferring the system channel."""
        system = getattr(self._agent, "_system_messages_in", None)
        if system is not None:
            system.put(text)
            return "system"
        if self._user_messages is not None:
            self._user_messages.put(text)
            return "user"
        return None

    def _withdraw_pending_input(self) -> tuple[bool, str | None]:
        """Withdraw on the runtime owner, serialized against submit/close/swap."""
        return self.run(self._withdraw_pending_input_on_owner)

    def _withdraw_pending_input_on_owner(self) -> tuple[bool, str | None]:
        with self._lifecycle_lock:
            if self._lifecycle_state != "active":
                return False, None
            item = self._user_messages.pop_last()
            self._pending_user_messages = tuple(
                value for value in self._user_messages.snapshot() if isinstance(value, str)
            )
            pending_inputs = self._pending_user_messages
            # Publish before releasing the lifecycle transaction. Otherwise a
            # concurrent submit can publish a newer snapshot first and then be
            # overwritten by this stale withdrawal snapshot.
            self._update_pending_inputs(pending_inputs)
        return True, None if item is None else str(item)

    def cancel_tasks(self, tasks: list[asyncio.Task[Any]]) -> None:
        """Cancel worker-owned tasks safely from any calling thread."""
        for task in tasks:
            try:
                loop = task.get_loop()
            except RuntimeError:
                loop = None
            if loop is not None and loop.is_running():
                loop.call_soon_threadsafe(task.cancel)
            else:
                task.cancel()

    async def swap_agent(self, agent: Any, *, seed_prompt: str | None = None) -> None:
        """Rebind the concrete-agent callbacks without stopping either agent."""
        with self._lifecycle_lock:
            if self._lifecycle_state != "active":
                raise RuntimeError("local agent runner is not active")
            self._lifecycle_state = "swapping"
            self._binding_generation += 1
            self._suspend_restart += 1
        try:
            await self._cancel_turn_in_lifecycle("swapping", force=True, notify=False)
            # close()/shutdown() may win while cancellation is awaiting. Recheck
            # under the lifecycle lock before installing callbacks on the new
            # agent so a completed close can never be followed by a rebind.
            with self._lifecycle_lock:
                if self._lifecycle_state != "swapping":
                    raise RuntimeError("local agent runner closed during swap")
                with self._callback_lock, _CALLBACK_LEASE_LOCK:
                    was_bound = self._bound
                    old_agent = self._agent
                    old_queue_manager = self._queue_manager
                    old_user_messages = self._user_messages
                    self._restore_callbacks_locked()
                    self._agent = agent
                    self._queue_manager = agent.queue_manager
                    self._user_messages = agent._user_messages_in
                    try:
                        if was_bound:
                            self._acquire_callbacks_locked()
                    except BaseException:
                        # The old lease cannot be stolen while the process-wide
                        # lease lock is held, so failed replacement acquisition
                        # can restore the complete prior binding transactionally.
                        self._agent = old_agent
                        self._queue_manager = old_queue_manager
                        self._user_messages = old_user_messages
                        if was_bound:
                            self._acquire_callbacks_locked()
                        raise
                    if seed_prompt is not None:
                        self._user_messages.put(seed_prompt)
                    self._pending_user_messages = tuple(
                        item for item in self._user_messages.snapshot() if isinstance(item, str)
                    )
                    pending_inputs = self._pending_user_messages
        finally:
            with self._lifecycle_lock:
                self._suspend_restart -= 1
                if self._lifecycle_state == "swapping":
                    self._lifecycle_state = "active"
        self._update_agent_workspace(
            pending_inputs=pending_inputs,
            working_directory=(str(agent.cwd) if getattr(agent, "cwd", None) is not None else None),
        )
        if self._user_messages.qsize() > 0:
            self.ensure_dispatcher()

    def request_cancel(self, *, force: bool = False, notify: bool = True) -> bool:
        """Atomically admit cancellation and cancel the captured turn.

        Asyncio tasks are cancelled through their owning loop, so this method is
        truthful from every calling thread without a deferred UI-loop decision.
        A true result means the cancellation flag was committed before return.
        """
        with self._lifecycle_lock:
            if self._lifecycle_state != "active":
                return False
            task = self._task
            if task is None or task.done() or (not self._in_handle and not force):
                return False
            if self._cancel_requested:
                return True
            self._cancel_requested = True
            self._notify_cancelled = notify
            source = self._source_task
            future = self._source_future
            target = source if source is not None else task
            try:
                if source is not None:
                    source.get_loop().call_soon_threadsafe(source.cancel)
                elif future is not None:
                    if not future.cancel():
                        raise RuntimeError("agent turn is no longer cancellable")
                else:
                    target.get_loop().call_soon_threadsafe(target.cancel)
            except RuntimeError:
                self._cancel_requested = False
                self._notify_cancelled = False
                return False
        self._changed()
        return True

    def _request_cancel_on_owner(
        self,
        *,
        force: bool,
        notify: bool,
        expected_generation: int | None = None,
        allowed_lifecycle_states: tuple[str, ...] = ("active",),
    ) -> bool:
        with self._lifecycle_lock:
            if self._lifecycle_state not in allowed_lifecycle_states or (
                expected_generation is not None and expected_generation != self._binding_generation
            ):
                return False
            task = self._task
            if task is None or task.done() or (not self._in_handle and not force):
                return False
            if self._cancel_requested:
                return True
            self._cancel_requested = True
            self._notify_cancelled = notify
            source = self._source_task
            if source is not None:
                loop = source.get_loop()
                loop.call_soon_threadsafe(source.cancel)
            elif self._source_future is not None:
                self._source_future.cancel()
            else:
                task.cancel()
        self._changed()
        return True

    async def _cancel_turn_in_lifecycle(
        self,
        lifecycle_state: str,
        *,
        force: bool,
        notify: bool,
    ) -> bool:
        """Cancel admitted work while an exclusive lifecycle transition owns it."""
        with self._lifecycle_lock:
            task = self._task
            generation = self._binding_generation
        if (
            task is None
            or task.done()
            or not self._request_cancel_on_owner(
                force=force,
                notify=notify,
                expected_generation=generation,
                allowed_lifecycle_states=(lifecycle_state,),
            )
        ):
            return False
        try:
            if self._source_future is not None:
                await asyncio.wrap_future(self._source_future)
            elif task is not asyncio.current_task():
                await task
        except asyncio.CancelledError:
            pass
        return True

    async def cancel_turn(self, *, force: bool = False, notify: bool = True) -> bool:
        task = self._task
        if task is None or task.done() or not self.request_cancel(force=force, notify=notify):
            return False
        try:
            if self._source_future is not None:
                await asyncio.wrap_future(self._source_future)
            elif task is not asyncio.current_task():
                await task
        except asyncio.CancelledError:
            pass
        return True

    async def cancel_for_transition(self) -> bool:
        """Silently cancel once without restarting work queued for the next state."""
        with self._lifecycle_lock:
            if self._lifecycle_state != "active":
                return False
            self._suspend_restart += 1
        try:
            return await self.cancel_turn(force=True, notify=False)
        finally:
            with self._lifecycle_lock:
                self._suspend_restart -= 1

    def request_stop(self) -> bool:
        """Begin terminal shutdown, including before a UI loop is activated."""
        with self._lifecycle_lock:
            if self._lifecycle_state != "active":
                return False
            self._lifecycle_state = "stopping"

        def start() -> None:
            with self._lifecycle_lock:
                if self._stop_task is None:
                    self._stop_task = asyncio.create_task(
                        self.shutdown(), name="nooa-local-agent-stop"
                    )

        loop = self._ui_loop
        if loop is not None and loop.is_running():
            if self._marshal_to_ui_owner(start) is not None:
                return True
            # The owner loop closed between ``is_running`` and scheduling.
            # Continue below with an independent shutdown coordinator.

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            if self._loop is None:
                # No asynchronous resources exist before either loop is
                # established, so terminal closure can complete synchronously.
                self.close()
                with self._lifecycle_lock:
                    self._resources_shutdown = True
                    self._shutdown_started = True
                self._shutdown_complete.set()
                return True
        else:
            if loop is None or not loop.is_running():
                start()
                return True

        # A worker may exist, or the former UI loop may have just closed.
        # Coordinate from a fresh temporary loop so an admitted stop can never
        # remain stranded in ``stopping``.
        def coordinate_shutdown() -> None:
            try:
                asyncio.run(self.shutdown())
            except BaseException:
                logger.exception("fallback agent shutdown failed")

        coordinator = threading.Thread(
            target=coordinate_shutdown,
            name="nooa-local-agent-shutdown",
            daemon=True,
        )
        self._shutdown_coordinator = coordinator
        coordinator.start()
        return True

    async def wait_stopped(self) -> None:
        """Wait across event loops for an accepted stop request to complete."""
        if self._shutdown_complete.is_set():
            if self._shutdown_error is not None:
                raise self._shutdown_error
            return
        await asyncio.to_thread(self._shutdown_complete.wait)
        if self._shutdown_error is not None:
            raise self._shutdown_error

    def run(self, fn: Callable[[], Any]) -> Any:
        loop = self._loop
        if loop is None or not loop.is_running():
            value = fn()
            if asyncio.iscoroutine(value):
                raise TypeError("run() with a coroutine requires an active worker loop")
            return value
        if getattr(loop, "_thread_id", None) == threading.current_thread().ident:
            value = fn()
            if asyncio.iscoroutine(value):
                raise TypeError("run() cannot block on a coroutine from the worker loop")
            return value

        async def invoke() -> Any:
            value = fn()
            return await value if asyncio.iscoroutine(value) else value

        future = asyncio.run_coroutine_threadsafe(invoke(), loop)
        try:
            return future.result(timeout=30)
        except TimeoutError:
            future.cancel()
            raise

    async def run_async(self, fn: Callable[[], Any]) -> Any:
        loop = self._loop
        if (
            loop is None
            or not loop.is_running()
            or getattr(loop, "_thread_id", None) == threading.current_thread().ident
        ):
            value = fn()
            return await value if asyncio.iscoroutine(value) else value

        async def invoke() -> Any:
            value = fn()
            return await value if asyncio.iscoroutine(value) else value

        future = asyncio.run_coroutine_threadsafe(invoke(), loop)
        try:
            return await asyncio.wait_for(asyncio.wrap_future(future), timeout=30)
        except TimeoutError:
            future.cancel()
            raise

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is not None and self._loop.is_running():
            return self._loop
        loop = asyncio.new_event_loop()
        self._loop = loop
        self._ready.clear()

        def worker() -> None:
            asyncio.set_event_loop(loop)

            def suppress_logging_worker_destroyed(
                owner: asyncio.AbstractEventLoop, context: dict[str, Any]
            ) -> None:
                message = context.get("message", "")
                task = context.get("task")
                if "Task was destroyed" in message and "LoggingWorker" in repr(task):
                    return
                owner.default_exception_handler(context)

            loop.set_exception_handler(suppress_logging_worker_destroyed)
            self._ready.set()
            try:
                loop.run_forever()
            finally:
                pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
                for task in pending:
                    task.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                loop.run_until_complete(loop.shutdown_asyncgens())
                loop.close()

        thread = threading.Thread(target=worker, name="nooa-local-agent-loop", daemon=True)
        self._thread = thread
        try:
            thread.start()
        except BaseException:
            loop.close()
            self._loop = None
            self._thread = None
            raise
        if not self._ready.wait(timeout=5):
            # Preserve live ownership on timeout: the caller can retry teardown
            # via stop_worker_loop rather than losing a possibly running thread.
            if not thread.is_alive():
                loop.close()
                self._loop = None
                self._thread = None
            raise RuntimeError("local agent event loop failed to start")
        return loop

    async def shutdown_queue_manager(
        self, *, agent: Any | None = None, flush: bool = False
    ) -> None:
        queue_manager = self._queue_manager if agent is None else agent.queue_manager

        async def shutdown() -> None:
            await queue_manager.shutdown()
            if flush:
                for channel in queue_manager.channels().values():
                    if channel.mode == "queue":
                        channel.flush()

        await self.run_async(shutdown)

    def ensure_worker_loop(self) -> asyncio.AbstractEventLoop:
        return self._ensure_loop()

    @property
    def worker_loop(self) -> asyncio.AbstractEventLoop | None:
        return self._loop

    async def stop_worker_loop(self) -> None:
        """Stop the owned loop, retaining live handles if termination times out."""
        loop, thread = self._loop, self._thread
        if loop is None:
            return
        if loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread.is_alive():
            await asyncio.to_thread(thread.join, 5)
            if thread.is_alive():
                raise RuntimeError("local agent event-loop thread did not stop")
        self._loop = None
        self._thread = None

    async def shutdown(self) -> None:
        """Stop work once; every concurrent caller waits for completion."""
        with self._lifecycle_lock:
            owner = not self._shutdown_started
            if owner:
                self._shutdown_started = True
        if not owner:
            await asyncio.to_thread(self._shutdown_complete.wait)
            if self._shutdown_error is not None:
                raise self._shutdown_error
            return
        try:
            await self._shutdown_impl()
        except BaseException as exc:
            with self._lifecycle_lock:
                self._shutdown_error = exc
            raise
        finally:
            self._shutdown_complete.set()

    async def _shutdown_impl(self) -> None:
        with self._lifecycle_lock:
            if self._resources_shutdown:
                return
            self._lifecycle_state = "closing"
            self._suspend_restart += 1
        error: BaseException | None = None
        try:
            try:
                await self._cancel_turn_in_lifecycle("closing", force=True, notify=False)
                await self.shutdown_queue_manager()
            except BaseException as exc:
                error = exc
            loop = self._loop
            if loop is not None and loop.is_running():
                future = asyncio.run_coroutine_threadsafe(_stop_litellm_worker(), loop)
                try:
                    await asyncio.wait_for(asyncio.wrap_future(future), timeout=2)
                except Exception:
                    pass
            try:
                await self.stop_worker_loop()
            except BaseException as exc:
                if error is None:
                    error = exc
            else:
                with self._lifecycle_lock:
                    self._resources_shutdown = True
        finally:
            with self._lifecycle_lock:
                self._suspend_restart -= 1
            self.close()
        if error is not None:
            raise error

    def _restore_callbacks(self) -> None:
        with _CALLBACK_LEASE_LOCK:
            self._restore_callbacks_locked()

    def _restore_callbacks_locked(self) -> None:
        if not self._bound:
            return
        # Both hooks are single-subscriber callbacks. Restore only
        # while they are still ours so a later owner is never clobbered.
        current_notify = getattr(self._queue_manager, "_notify_callback", None)
        if getattr(current_notify, "__self__", None) is self:
            self._queue_manager.set_notify_callback(self._previous_notify_callback)
        current_on_get = getattr(self._user_messages, "_on_get", None)
        if current_on_get is self._owned_user_on_get:
            self._user_messages.set_on_get(self._previous_user_on_get)
        self._owned_user_on_get = None
        if getattr(self._queue_manager, _CALLBACK_OWNER_ATTRIBUTE, None) is self:
            delattr(self._queue_manager, _CALLBACK_OWNER_ATTRIBUTE)
        self._bound = False

    def close(self) -> bool:
        with self._lifecycle_lock:
            if self._closed:
                self._lifecycle_state = "closed"
                return False
            self._lifecycle_state = "closing"
            self._binding_generation += 1
            self._closed = True
        with self._callback_lock:
            self._restore_callbacks()
        self._close_state()
        with self._lifecycle_lock:
            self._lifecycle_state = "closed"
        return True

    def _capture_runtime_projection(
        self,
    ) -> tuple[int, CancellationState, tuple[AgentJobSummary, ...]] | None:
        """Freeze mutable runtime state at its transition site."""
        with self._lifecycle_lock:
            if self._lifecycle_state != "active":
                return None
            generation = self._binding_generation
            cancellation = (
                CancellationState.REQUESTED if self._cancel_requested else CancellationState.NONE
            )
            snapshots = self._job_snapshots_on_owner()
        jobs = tuple(
            AgentJobSummary(
                snapshot.name,
                snapshot.label,
                AgentJobState(snapshot.state),
                snapshot.queued,
                tuple(str(value) for value in snapshot.values),
                snapshot.job_id,
            )
            for snapshot in snapshots
        )
        return generation, cancellation, jobs

    def _publish_runtime_projection(
        self,
        generation: int,
        cancellation: CancellationState,
        jobs: tuple[AgentJobSummary, ...],
    ) -> bool:
        # Revalidate after owner-loop marshalling. close()/swap_agent() may
        # retire the captured state while this callback is queued.
        with self._lifecycle_lock:
            if self._lifecycle_state != "active" or generation != self._binding_generation:
                return False
            self._update_runtime_projection(cancellation=cancellation, jobs=jobs)
        callback = self._on_state_change
        if callback is not None:
            callback()
        return True

    def _changed(self) -> None:
        projection = self._capture_runtime_projection()
        if projection is not None:
            self._publish_runtime_projection(*projection)
