# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local-agent adapter and lifecycle-owner tests."""

from __future__ import annotations

import asyncio
import threading
from collections import deque
from collections.abc import Callable
from types import SimpleNamespace

import pytest
from nooa_cli.interactive import (
    AgentJobState,
    AgentJobSummary,
    AgentLifecycle,
    CancellationState,
)
from nooa_cli.interactive.local_agent import LocalAgentRunner
from nooa_cli.tui.agent_controller import AgentController

from nooa.interactive import InteractiveAgent
from nooa.runtime.channels import QueueManager


class GuardedInteractiveAgentStub(InteractiveAgent):
    async def handle(self, notification: dict[str, list[object]]) -> SimpleNamespace:
        return SimpleNamespace(kind="WAIT", explanation="")


class PumpScheduler:
    def __init__(self) -> None:
        self.calls: deque[Callable[[], None]] = deque()

    def schedule(self, callback: Callable[[], None]) -> None:
        self.calls.append(callback)

    def run_all(self) -> None:
        while self.calls:
            self.calls.popleft()()


class AgentStub:
    def __init__(self) -> None:
        self.queue_manager = QueueManager()
        self._user_messages_in = self.queue_manager.queue("user_messages")
        self.emit: Callable[[str], None] | None = lambda _text: None
        self.notifications: list[dict[str, list[object]]] = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def handle(self, notification: dict[str, list[object]]) -> SimpleNamespace:
        self.notifications.append(notification)
        self.entered.set()
        await self.release.wait()
        return SimpleNamespace(kind="WAIT", explanation="")


def test_bind_supports_interactive_agent_without_dynamic_emit_attribute() -> None:
    agent = GuardedInteractiveAgentStub()
    rendered: list[str] = []

    runner = LocalAgentRunner(agent, emit_text=rendered.append, agent_id="guarded")

    assert not hasattr(agent, "emit")
    assert runner.close() is True
    assert not hasattr(agent, "emit")


def test_real_runner_scheduler_failure_disconnects_controller_and_gates_commands() -> None:
    class FailingAfterInitialScheduler(PumpScheduler):
        def __init__(self) -> None:
            super().__init__()
            self.fail = False

        def schedule(self, callback: Callable[[], None]) -> None:
            if self.fail:
                raise RuntimeError("UI scheduler stopped")
            super().schedule(callback)

    scheduler = FailingAfterInitialScheduler()
    runner = LocalAgentRunner(AgentStub(), emit_text=lambda _text: None, agent_id="local-1")
    controller = AgentController(scheduler)
    controller.observe(runner)
    scheduler.run_all()
    scheduler.fail = True

    runner._set_lifecycle(AgentLifecycle.THINKING)

    assert controller.state is None
    assert isinstance(controller.failure, RuntimeError)
    with pytest.raises(RuntimeError, match="not observing an agent"):
        controller.submit("rejected")
    runner.close()


def test_real_runner_listener_failure_disconnects_controller_and_gates_commands() -> None:
    scheduler = PumpScheduler()
    runner = LocalAgentRunner(AgentStub(), emit_text=lambda _text: None, agent_id="local-1")
    should_fail = False

    def changed(_state) -> None:
        if should_fail:
            raise RuntimeError("UI listener failed")

    controller = AgentController(scheduler, changed)
    controller.observe(runner)
    scheduler.run_all()
    should_fail = True

    runner._set_lifecycle(AgentLifecycle.THINKING)
    scheduler.run_all()

    assert controller.state is None
    assert isinstance(controller.failure, RuntimeError)
    with pytest.raises(RuntimeError, match="not observing an agent"):
        controller.interrupt()
    runner.close()


@pytest.mark.asyncio
async def test_runner_chains_and_restores_existing_user_message_hook() -> None:
    agent = AgentStub()
    consumed: list[str] = []
    agent._user_messages_in.set_on_get(consumed.append)
    runner = LocalAgentRunner(agent, emit_text=lambda _text: None, agent_id="local-1")

    agent._user_messages_in.put("hello")
    assert await agent._user_messages_in.get() == "hello"
    assert consumed == ["hello"]

    assert runner.close()
    agent._user_messages_in.put("after")
    assert await agent._user_messages_in.get() == "after"
    assert consumed == ["hello", "after"]


@pytest.mark.asyncio
async def test_swap_cancels_in_flight_old_agent_after_transition_admission() -> None:
    class CancellationTrackingAgent(AgentStub):
        def __init__(self) -> None:
            super().__init__()
            self.cancelled = asyncio.Event()

        async def handle(self, notification):
            self.notifications.append(notification)
            self.entered.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            return SimpleNamespace(kind="WAIT", explanation="")

    old_agent = CancellationTrackingAgent()
    new_agent = AgentStub()
    runner = LocalAgentRunner(old_agent, emit_text=lambda _text: None, agent_id="local-1")
    runner.submit("old turn")
    await asyncio.wait_for(old_agent.entered.wait(), timeout=1)

    await asyncio.wait_for(runner.swap_agent(new_agent), timeout=1)

    assert old_agent.cancelled.is_set()
    assert runner.task is None
    assert runner.cancel_requested is False
    await runner.shutdown()


@pytest.mark.asyncio
async def test_shutdown_cancels_in_flight_turn_after_closing_transition() -> None:
    class CancellationTrackingAgent(AgentStub):
        def __init__(self) -> None:
            super().__init__()
            self.cancelled = asyncio.Event()

        async def handle(self, notification):
            self.notifications.append(notification)
            self.entered.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.set()
                raise
            return SimpleNamespace(kind="WAIT", explanation="")

    agent = CancellationTrackingAgent()
    runner = LocalAgentRunner(agent, emit_text=lambda _text: None, agent_id="local-1")
    runner.submit("active turn")
    await asyncio.wait_for(agent.entered.wait(), timeout=1)

    await asyncio.wait_for(runner.shutdown(), timeout=1)

    assert agent.cancelled.is_set()
    assert runner.closed


@pytest.mark.asyncio
async def test_dedicated_runner_loop_marshals_work_without_renderer_owned_threads() -> None:
    agent = AgentStub()
    runner = LocalAgentRunner(agent, emit_text=lambda _text: None, agent_id="local-1")
    runner.activate(asyncio.get_running_loop())
    try:
        value = await runner.run_async(lambda: asyncio.sleep(0, result=42))
        assert value == 42
        assert runner.run(lambda: "ok") == "ok"
    finally:
        await runner.shutdown()


@pytest.mark.asyncio
async def test_silent_transition_cancel_does_not_notify_or_restart_queued_work() -> None:
    agent = AgentStub()
    notices: list[str] = []
    runner = LocalAgentRunner(
        agent,
        emit_text=lambda _text: None,
        agent_id="local-1",
        on_cancelled=lambda: notices.append("interrupted"),
    )
    runner.submit("first")
    await asyncio.wait_for(agent.entered.wait(), timeout=1)
    runner.submit("second")

    assert await runner.cancel_for_transition()

    assert notices == []
    assert runner.task is None
    assert agent._user_messages_in.qsize() == 1
    await runner.shutdown()


@pytest.mark.asyncio
async def test_shutdown_failure_still_stops_worker_and_restores_callbacks() -> None:
    agent = AgentStub()
    previous_emit = agent.emit

    def previous_notify() -> None:
        pass

    agent.queue_manager.set_notify_callback(previous_notify)
    runner = LocalAgentRunner(agent, emit_text=lambda _text: None, agent_id="local-1")
    runner.activate(asyncio.get_running_loop())
    worker = runner.worker_loop

    async def fail_shutdown() -> None:
        raise RuntimeError("queue shutdown failed")

    agent.queue_manager.shutdown = fail_shutdown
    with pytest.raises(RuntimeError, match="queue shutdown failed"):
        await runner.shutdown()

    assert runner.closed
    assert runner.worker_loop is None
    assert worker is not None and worker.is_closed()
    assert agent.emit is previous_emit
    assert agent.queue_manager._notify_callback is previous_notify


def test_close_serializes_with_in_flight_emit_callback() -> None:
    agent = AgentStub()
    entered = threading.Event()
    release = threading.Event()
    rendered: list[str] = []

    def render(text: str) -> None:
        entered.set()
        assert release.wait(1)
        rendered.append(text)

    runner = LocalAgentRunner(agent, emit_text=render, agent_id="local-1")
    emitter = threading.Thread(target=lambda: runner._present("before"))
    emitter.start()
    assert entered.wait(1)
    closed = threading.Event()
    closer = threading.Thread(target=lambda: (runner.close(), closed.set()))
    closer.start()
    assert not closed.wait(0.05)
    release.set()
    emitter.join(1)
    closer.join(1)

    assert closed.is_set()
    assert rendered == ["before"]
    runner._present("after")
    assert rendered == ["before"]


@pytest.mark.asyncio
async def test_close_wins_against_swap_without_rebinding_callbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_agent = AgentStub()
    new_agent = AgentStub()
    old_emit = old_agent.emit
    new_emit = new_agent.emit
    runner = LocalAgentRunner(old_agent, emit_text=lambda _text: None, agent_id="local-1")
    runner.submit("first")
    await asyncio.wait_for(old_agent.entered.wait(), timeout=1)
    cancel_entered = asyncio.Event()
    cancel_release = asyncio.Event()
    original_cancel = runner._cancel_turn_in_lifecycle

    async def paused_cancel(lifecycle_state: str, *, force: bool, notify: bool) -> bool:
        cancel_entered.set()
        await cancel_release.wait()
        return await original_cancel(lifecycle_state, force=force, notify=notify)

    monkeypatch.setattr(runner, "_cancel_turn_in_lifecycle", paused_cancel)
    swap = asyncio.create_task(runner.swap_agent(new_agent))
    await asyncio.wait_for(cancel_entered.wait(), timeout=1)
    assert runner.close()
    cancel_release.set()
    old_agent.release.set()

    with pytest.raises(RuntimeError, match="closed during swap"):
        await swap
    assert runner.closed
    assert old_agent.emit is old_emit
    assert new_agent.emit is new_emit


def test_worker_start_failure_does_not_retain_half_started_loop(monkeypatch) -> None:
    agent = AgentStub()
    runner = LocalAgentRunner(agent, emit_text=lambda _text: None, agent_id="local-1")

    def fail_start(_self) -> None:
        raise RuntimeError("thread start failed")

    monkeypatch.setattr(threading.Thread, "start", fail_start)
    with pytest.raises(RuntimeError, match="thread start failed"):
        runner.ensure_worker_loop()

    assert runner.worker_loop is None
    assert runner.close()


@pytest.mark.asyncio
async def test_concurrent_shutdown_callers_wait_for_same_completion() -> None:
    agent = AgentStub()
    runner = LocalAgentRunner(agent, emit_text=lambda _text: None, agent_id="local-1")
    entered = asyncio.Event()
    release = asyncio.Event()
    original = runner.shutdown_queue_manager

    async def blocked_shutdown(*, agent=None, flush=False) -> None:
        entered.set()
        await release.wait()
        await original(agent=agent, flush=flush)

    runner.shutdown_queue_manager = blocked_shutdown  # type: ignore[method-assign]
    first = asyncio.create_task(runner.shutdown())
    await entered.wait()
    second = asyncio.create_task(runner.shutdown())
    await asyncio.sleep(0)
    assert not second.done()
    release.set()
    await asyncio.gather(first, second)
    assert runner.closed


def test_runner_can_defer_callback_binding_until_startup_is_transactional() -> None:
    agent = AgentStub()
    previous_emit = agent.emit
    previous_notify = agent.queue_manager._notify_callback
    previous_on_get = agent._user_messages_in._on_get
    runner = LocalAgentRunner(
        agent,
        emit_text=lambda _text: None,
        agent_id="local-1",
        bind_callbacks=False,
    )
    assert agent.emit is previous_emit
    assert agent.queue_manager._notify_callback is previous_notify
    assert agent._user_messages_in._on_get is previous_on_get
    runner.bind()
    assert agent.emit is previous_emit
    assert getattr(agent.queue_manager._notify_callback, "__self__", None) is runner
    assert agent._user_messages_in._on_get is runner._owned_user_on_get
    runner.close()
    assert agent.emit is previous_emit
    assert agent.queue_manager._notify_callback is previous_notify
    assert agent._user_messages_in._on_get is previous_on_get


@pytest.mark.asyncio
async def test_seed_is_not_enqueued_when_close_wins_swap_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_agent = AgentStub()
    new_agent = AgentStub()
    runner = LocalAgentRunner(old_agent, emit_text=lambda _text: None, agent_id="local-1")
    runner.submit("first")
    await asyncio.wait_for(old_agent.entered.wait(), timeout=1)
    cancel_entered = asyncio.Event()
    cancel_release = asyncio.Event()
    original_cancel = runner._cancel_turn_in_lifecycle

    async def paused_cancel(lifecycle_state: str, *, force: bool, notify: bool) -> bool:
        cancel_entered.set()
        await cancel_release.wait()
        return await original_cancel(lifecycle_state, force=force, notify=notify)

    monkeypatch.setattr(runner, "_cancel_turn_in_lifecycle", paused_cancel)
    swap = asyncio.create_task(runner.seed_and_swap(new_agent, "seed"))
    await asyncio.wait_for(cancel_entered.wait(), timeout=1)
    assert runner.close()
    cancel_release.set()
    old_agent.release.set()
    with pytest.raises(RuntimeError, match="closed during swap"):
        await swap
    assert new_agent._user_messages_in.qsize() == 0


@pytest.mark.asyncio
async def test_successful_swap_moves_and_restores_all_callback_bridges() -> None:
    old_agent = AgentStub()
    new_agent = AgentStub()
    old_emit = old_agent.emit
    new_emit = new_agent.emit
    old_notify = old_agent.queue_manager._notify_callback
    new_notify = new_agent.queue_manager._notify_callback
    old_on_get = old_agent._user_messages_in._on_get
    new_on_get = new_agent._user_messages_in._on_get
    rendered: list[str] = []
    runner = LocalAgentRunner(old_agent, emit_text=rendered.append, agent_id="local-1")

    await runner.swap_agent(new_agent)

    assert old_agent.emit is old_emit
    assert old_agent.queue_manager._notify_callback is old_notify
    assert old_agent._user_messages_in._on_get is old_on_get
    assert new_agent.emit is new_emit
    assert getattr(new_agent.queue_manager._notify_callback, "__self__", None) is runner
    assert new_agent._user_messages_in._on_get is runner._owned_user_on_get
    runner._present("new output")
    assert rendered == ["new output"]

    await runner.shutdown()
    assert new_agent.emit is new_emit
    assert new_agent.queue_manager._notify_callback is new_notify
    assert new_agent._user_messages_in._on_get is new_on_get


def test_second_runner_cannot_acquire_an_agents_callback_bridge() -> None:
    agent = AgentStub()
    first = LocalAgentRunner(agent, emit_text=lambda _text: None, agent_id="first")
    second = LocalAgentRunner(
        agent, emit_text=lambda _text: None, agent_id="second", bind_callbacks=False
    )

    with pytest.raises(RuntimeError, match="already has an active runner"):
        second.bind()

    first.close()
    second.bind()
    second.close()


@pytest.mark.asyncio
async def test_shutdown_after_close_still_releases_owned_worker_resources() -> None:
    agent = AgentStub()
    runner = LocalAgentRunner(agent, emit_text=lambda _text: None, agent_id="local-1")
    runner.activate(asyncio.get_running_loop())
    worker = runner.worker_loop

    assert runner.close()
    await runner.shutdown()

    assert runner.worker_loop is None
    assert worker is not None and worker.is_closed()


def test_concurrent_runner_binding_has_exactly_one_callback_lease_owner() -> None:
    agent = AgentStub()
    runners = [
        LocalAgentRunner(
            agent, emit_text=lambda _text: None, agent_id=str(index), bind_callbacks=False
        )
        for index in range(2)
    ]
    barrier = threading.Barrier(3)
    outcomes: list[tuple[str, LocalAgentRunner]] = []

    def bind(runner: LocalAgentRunner) -> None:
        barrier.wait()
        try:
            runner.bind()
        except RuntimeError:
            outcomes.append(("rejected", runner))
        else:
            outcomes.append(("bound", runner))

    threads = [threading.Thread(target=bind, args=(runner,)) for runner in runners]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(1)

    assert [result for result, _runner in outcomes].count("bound") == 1
    assert [result for result, _runner in outcomes].count("rejected") == 1
    for _result, runner in outcomes:
        runner.close()


@pytest.mark.asyncio
async def test_failed_swap_lease_acquisition_restores_old_agent_binding() -> None:
    old_agent = AgentStub()
    occupied_agent = AgentStub()
    runner = LocalAgentRunner(old_agent, emit_text=lambda _text: None, agent_id="moving")
    occupant = LocalAgentRunner(occupied_agent, emit_text=lambda _text: None, agent_id="occupant")

    with pytest.raises(RuntimeError, match="already has an active runner"):
        await runner.swap_agent(occupied_agent)

    assert old_agent.emit is not None
    assert getattr(old_agent.queue_manager._notify_callback, "__self__", None) is runner
    assert old_agent._user_messages_in._on_get is runner._owned_user_on_get
    assert occupied_agent.emit is not None
    assert getattr(occupied_agent.queue_manager._notify_callback, "__self__", None) is occupant
    await runner.shutdown()
    await occupant.shutdown()


@pytest.mark.asyncio
async def test_off_owner_interrupt_rejects_when_no_turn_is_cancellable() -> None:
    runner = LocalAgentRunner(AgentStub(), emit_text=lambda _text: None, agent_id="local-1")
    runner.activate(asyncio.get_running_loop())
    admitted: list[bool] = []

    thread = threading.Thread(target=lambda: admitted.append(runner.request_cancel(force=True)))
    thread.start()
    thread.join()

    assert admitted == [False]
    assert runner.cancel_requested is False
    await runner.shutdown()


@pytest.mark.asyncio
async def test_claimed_old_dequeue_callback_cannot_cross_completed_swap() -> None:
    old_agent = AgentStub()
    new_agent = AgentStub()
    old_previous: list[str] = []
    accepted: list[str] = []
    old_agent._user_messages_in.set_on_get(old_previous.append)
    runner = LocalAgentRunner(old_agent, emit_text=lambda _text: None, agent_id="local-1")
    runner.set_user_message_accepted_callback(accepted.append)
    assert runner.submit("old")
    old_callback = old_agent._user_messages_in._on_get

    # Model a queue implementation that has already claimed its callback but
    # does not invoke it until after the binding transition completes.
    await runner.swap_agent(new_agent)
    assert runner.submit("new")
    old_callback("old")

    assert runner.pending_user_messages() == ("new",)
    assert old_previous == []
    assert accepted == []
    await runner.shutdown()


class _ClosingOwnerLoop:
    def is_running(self) -> bool:
        return True

    def call_soon_threadsafe(self, _callback: Callable[[], None]) -> None:
        raise RuntimeError("event loop is closed")


class _StoppedOwnerLoop:
    def is_running(self) -> bool:
        return False


def test_submit_rejects_and_rolls_back_for_recorded_stopped_owner_loop() -> None:
    agent = AgentStub()
    agent._user_messages_in.put("existing")
    runner = LocalAgentRunner(agent, emit_text=lambda _text: None, agent_id="local-1")
    runner._ui_loop = _StoppedOwnerLoop()  # type: ignore[assignment]

    assert not runner.submit("new")
    assert agent._user_messages_in.snapshot() == ["existing"]
    assert runner.state.pending_inputs == ("existing",)
    runner.close()


def test_submit_state_publication_is_serialized_before_dequeue() -> None:
    agent = AgentStub()
    runner = LocalAgentRunner(agent, emit_text=lambda _text: None, agent_id="local-1")
    # Avoid dispatcher startup; invoke the concrete dequeue callback below.
    runner._task = SimpleNamespace(done=lambda: False)  # type: ignore[assignment]

    submit_publishing = threading.Event()
    release_submit = threading.Event()
    dequeue_finished = threading.Event()
    original_update = runner._update_pending_inputs

    def blocking_update(pending_inputs: tuple[str, ...]) -> None:
        if pending_inputs == ("new",) and not submit_publishing.is_set():
            submit_publishing.set()
            assert release_submit.wait(timeout=1)
        original_update(pending_inputs)

    runner._update_pending_inputs = blocking_update  # type: ignore[method-assign]
    submit_thread = threading.Thread(target=lambda: runner.submit("new"))
    submit_thread.start()
    assert submit_publishing.wait(timeout=1)

    def dequeue() -> None:
        runner._on_user_message_get(
            "new",
            generation=runner._binding_generation,
            user_messages=runner._user_messages,
            previous=None,
        )
        dequeue_finished.set()

    dequeue_thread = threading.Thread(target=dequeue)
    dequeue_thread.start()
    # State publication remains inside the lifecycle transaction, so dequeue
    # cannot publish its newer empty snapshot until submit finishes.
    assert not dequeue_finished.wait(timeout=0.05)

    release_submit.set()
    submit_thread.join(timeout=1)
    dequeue_thread.join(timeout=1)
    assert not submit_thread.is_alive()
    assert not dequeue_thread.is_alive()
    assert runner.state.pending_inputs == runner.pending_user_messages() == ()
    runner.close()


def test_withdraw_state_publication_is_serialized_before_submit() -> None:
    agent = AgentStub()
    runner = LocalAgentRunner(agent, emit_text=lambda _text: None, agent_id="local-1")
    runner._task = SimpleNamespace(done=lambda: False)  # type: ignore[assignment]
    agent._user_messages_in.put("old")

    withdraw_publishing = threading.Event()
    release_withdraw = threading.Event()
    submit_finished = threading.Event()
    original_update = runner._update_pending_inputs

    def blocking_update(pending_inputs: tuple[str, ...]) -> None:
        if pending_inputs == () and not withdraw_publishing.is_set():
            withdraw_publishing.set()
            assert release_withdraw.wait(timeout=1)
        original_update(pending_inputs)

    runner._update_pending_inputs = blocking_update  # type: ignore[method-assign]
    withdraw_thread = threading.Thread(target=runner.withdraw_pending_input)
    withdraw_thread.start()
    assert withdraw_publishing.wait(timeout=1)

    def submit() -> None:
        runner.submit("new")
        submit_finished.set()

    submit_thread = threading.Thread(target=submit)
    submit_thread.start()
    assert not submit_finished.wait(timeout=0.05)

    release_withdraw.set()
    withdraw_thread.join(timeout=1)
    submit_thread.join(timeout=1)
    assert not withdraw_thread.is_alive()
    assert not submit_thread.is_alive()
    assert runner.state.pending_inputs == runner.pending_user_messages() == ("new",)
    runner.close()


def test_withdraw_pending_input_returns_text_and_publishes_state() -> None:
    agent = AgentStub()
    runner = LocalAgentRunner(agent, emit_text=lambda _text: None, agent_id="local-1")

    assert runner.submit("first")
    assert runner.submit("second")
    assert runner.state.pending_inputs == ("first\nsecond",)

    assert runner.withdraw_pending_input() == "first\nsecond"
    assert runner.state.pending_inputs == ()
    assert runner.withdraw_pending_input() is None
    runner.close()


def test_stop_before_activation_is_terminal_and_idempotently_rejected() -> None:
    runner = LocalAgentRunner(AgentStub(), emit_text=lambda _text: None, agent_id="local-1")

    assert runner.stop() is True
    assert runner.state.lifecycle is AgentLifecycle.STOPPED
    assert runner.closed
    assert runner.stop() is False


@pytest.mark.asyncio
async def test_swap_publishes_new_workspace_and_clears_runtime_projection(tmp_path) -> None:
    old_agent = AgentStub()
    old_agent.cwd = tmp_path / "old"
    new_agent = AgentStub()
    new_agent.cwd = tmp_path / "new"
    runner = LocalAgentRunner(old_agent, emit_text=lambda _text: None, agent_id="local-1")
    running = AgentJobSummary("job", "Job", AgentJobState.RUNNING, 1)
    runner._update_runtime_projection(cancellation=CancellationState.REQUESTED, jobs=(running,))

    await runner.swap_agent(new_agent)

    assert runner.state.working_directory == str(new_agent.cwd)
    assert runner.state.workspace.cancellation is CancellationState.NONE
    assert runner.state.workspace.jobs == ()
    await runner.shutdown()


@pytest.mark.asyncio
async def test_stale_runtime_projection_cannot_cross_completed_swap() -> None:
    old_agent = AgentStub()
    new_agent = AgentStub()
    runner = LocalAgentRunner(old_agent, emit_text=lambda _text: None, agent_id="local-1")
    with runner._lifecycle_lock:
        old_generation = runner._binding_generation
    stale_job = AgentJobSummary("old", "Old", AgentJobState.RUNNING, 0)

    await runner.swap_agent(new_agent)

    assert not runner._publish_runtime_projection(
        old_generation, CancellationState.REQUESTED, (stale_job,)
    )
    assert runner.state.workspace.cancellation is CancellationState.NONE
    assert runner.state.workspace.jobs == ()
    await runner.shutdown()


@pytest.mark.asyncio
async def test_submit_rolls_back_when_ui_owner_closes_during_schedule() -> None:
    agent = AgentStub()
    agent._user_messages_in.put("existing")
    runner = LocalAgentRunner(agent, emit_text=lambda _text: None, agent_id="local-1")
    runner._ui_loop = _ClosingOwnerLoop()  # type: ignore[assignment]

    accepted = await asyncio.to_thread(runner.submit, "new")

    assert not accepted
    assert agent._user_messages_in.snapshot() == ["existing"]
    assert runner.state.pending_inputs == ("existing",)
    runner.close()


@pytest.mark.asyncio
async def test_stop_falls_back_when_ui_owner_closes_during_schedule() -> None:
    runner = LocalAgentRunner(AgentStub(), emit_text=lambda _text: None, agent_id="local-1")
    runner.ensure_worker_loop()
    runner._ui_loop = _ClosingOwnerLoop()  # type: ignore[assignment]

    assert await asyncio.to_thread(runner.stop)
    await asyncio.wait_for(runner.wait_stopped(), timeout=2)

    assert runner.closed
    assert runner.worker_loop is None
    assert runner.state.lifecycle is AgentLifecycle.STOPPED


@pytest.mark.asyncio
async def test_quiescent_only_after_active_turn_finishes_naturally() -> None:
    agent = AgentStub()
    runner = LocalAgentRunner(agent, emit_text=lambda _text: None, agent_id="local-1")

    assert runner.is_quiescent
    assert runner.submit("finish this")
    await asyncio.wait_for(agent.entered.wait(), timeout=1)
    assert not runner.is_quiescent

    agent.release.set()
    for _ in range(100):
        if runner.is_quiescent:
            break
        await asyncio.sleep(0.01)

    assert runner.is_quiescent
    assert agent.notifications == [{"user_messages": ["finish this"]}]
    await runner.shutdown()


@pytest.mark.asyncio
async def test_wait_quiescent_does_not_return_during_active_turn() -> None:
    agent = AgentStub()
    runner = LocalAgentRunner(agent, emit_text=lambda _text: None, agent_id="local-1")
    assert runner.submit("finish this")
    await asyncio.wait_for(agent.entered.wait(), timeout=1)

    waiter = asyncio.create_task(runner.wait_quiescent())
    await asyncio.sleep(0.02)
    assert not waiter.done()

    agent.release.set()
    await asyncio.wait_for(waiter, timeout=1)
    await runner.shutdown()
