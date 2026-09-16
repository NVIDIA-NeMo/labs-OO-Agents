from __future__ import annotations

import threading
from collections import deque
from dataclasses import FrozenInstanceError

import pytest
from nooa_cli.interactive import AgentLifecycle, AgentState
from nooa_cli.interactive.local_agent import LocalAgentRunner


class Scheduler:
    def __init__(self):
        self.calls = deque()

    def schedule(self, callback):
        self.calls.append(callback)

    def run(self):
        while self.calls:
            self.calls.popleft()()


class Stub:
    def __init__(self):
        from nooa.runtime.channels import QueueManager

        self.queue_manager = QueueManager()
        self._user_messages_in = self.queue_manager.queue("user_messages")
        self.emit = lambda text: None

    async def handle(self, notification):
        return None


def runner():
    return LocalAgentRunner(Stub(), emit_text=lambda text: None, agent_id="a")


def test_state_is_deeply_immutable():
    state = AgentState("a", "A", None, AgentLifecycle.IDLE)
    with pytest.raises(FrozenInstanceError):
        state.lifecycle = AgentLifecycle.STOPPED
    assert isinstance(state.workspace.pending_inputs, tuple)


def test_observe_captures_atomically_and_schedules_outside_lock():
    r = runner()
    scheduled = []

    class Probe:
        def schedule(self, cb):
            assert r._state_lock.acquire(blocking=False)
            r._state_lock.release()
            scheduled.append(cb)

    observation = r.observe(lambda state: None, Probe())
    assert len(scheduled) == 1
    observation.close()
    r.close()


def test_inline_scheduler_and_close_from_callback_are_safe():
    r = runner()
    holder = []
    seen = []

    class Inline:
        def schedule(self, cb):
            cb()

    def callback(state):
        seen.append(state)
        holder[0].close() if holder else None

    # Initial inline callback necessarily precedes observe's return; a later callback closes itself.
    observation = r.observe(callback, Inline())
    holder.append(observation)
    r._update_pending_inputs(("x",))
    assert len(seen) == 2
    r._update_pending_inputs(("y",))
    assert len(seen) == 2
    r.close()


def test_updates_coalesce_and_callbacks_are_serialized():
    r = runner()
    scheduler = Scheduler()
    seen = []
    observation = r.observe(lambda state: seen.append(state), scheduler)
    r._update_pending_inputs(("one",))
    r._update_pending_inputs(("one", "two"))
    assert len(scheduler.calls) == 1
    scheduler.run()
    assert len(seen) == 1 and seen[0].pending_inputs == ("one", "two")
    observation.close()
    r.close()


def test_scheduler_failure_is_first_wins_and_terminal():
    r = runner()
    first = RuntimeError("scheduler")

    class Failing:
        def schedule(self, cb):
            raise first

    with pytest.raises(RuntimeError, match="scheduler"):
        r.observe(lambda state: None, Failing())
    assert not r._observations
    r.close()


def test_callback_failure_is_first_wins_and_terminal():
    r = runner()
    scheduler = Scheduler()
    failure = RuntimeError("callback")

    def callback(state):
        raise failure

    observation = r.observe(callback, scheduler)
    scheduler.run()
    assert observation not in r._observations
    r._update_pending_inputs(("ignored",))
    assert not scheduler.calls
    r.close()


def test_close_prevents_queued_callback_start():
    r = runner()
    scheduler = Scheduler()
    seen = []
    observation = r.observe(seen.append, scheduler)
    observation.close()
    scheduler.run()
    assert seen == [] and observation not in r._observations
    r.close()


def test_successful_submit_publishes_admitted_state_before_return():
    r = runner()
    assert r.submit("hello")
    assert r.state.pending_inputs == ("hello",)
    r.close()


def test_close_waits_out_new_starts_but_allows_running_callback_to_finish():
    r = runner()
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    class ThreadScheduler:
        def schedule(self, callback):
            threading.Thread(target=callback).start()

    def callback(state):
        entered.set()
        release.wait(1)
        finished.set()

    observation = r.observe(callback, ThreadScheduler())
    assert entered.wait(1)
    observation.close()
    r._update_pending_inputs(("not started",))
    release.set()
    assert finished.wait(1)
    r.close()


def test_concurrent_state_transitions_preserve_orthogonal_fields():
    r = runner()
    barrier = threading.Barrier(3)

    def pending():
        barrier.wait()
        r._update_pending_inputs(("queued",))

    def lifecycle():
        barrier.wait()
        r._set_lifecycle(AgentLifecycle.THINKING)

    threads = [threading.Thread(target=pending), threading.Thread(target=lifecycle)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(1)

    assert all(not thread.is_alive() for thread in threads)
    assert r.state.lifecycle is AgentLifecycle.THINKING
    assert r.state.pending_inputs == ("queued",)
    r.close()
