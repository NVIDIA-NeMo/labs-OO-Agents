from __future__ import annotations

import threading
from collections import deque
from dataclasses import replace

import pytest
from nooa_cli.interactive import AgentLifecycle, AgentState
from nooa_cli.tui.agent_controller import AgentController


class Scheduler:
    def __init__(self):
        self.calls = deque()

    def schedule(self, cb):
        self.calls.append(cb)

    def run(self):
        while self.calls:
            self.calls.popleft()()


class Observation:
    def __init__(self):
        self.closed = False
        self.failure = None

    def close(self):
        self.closed = True


class Agent:
    def __init__(self, name, fail=False):
        self._state = AgentState(name, name, None, AgentLifecycle.IDLE)
        self.fail = fail
        self.callback = None
        self.terminated = None
        self.observation = Observation()
        self.calls = []

    @property
    def state(self):
        return self._state

    def observe(self, callback, scheduler, on_terminated=None):
        if self.fail:
            raise RuntimeError("observe failed")
        self.callback = callback
        self.terminated = on_terminated
        scheduler.schedule(lambda: callback(self._state))
        return self.observation

    def publish(self, state):
        self._state = state
        self.callback(state)

    def submit(self, text):
        self.calls.append(("submit", text))
        return True

    def interrupt(self):
        self.calls.append(("interrupt",))
        return True

    def withdraw_pending_input(self):
        self.calls.append(("withdraw",))
        return "x"

    def stop(self):
        self.calls.append(("stop",))
        return True


def test_observe_is_transactional_when_observe_raises():
    scheduler = Scheduler()
    old = Agent("old")
    controller = AgentController(scheduler)
    controller.observe(old)
    with pytest.raises(RuntimeError, match="observe failed"):
        controller.observe(Agent("new", True))
    assert controller.state.agent_id == "old" and not old.observation.closed


def test_switch_generation_filters_queued_old_callbacks():
    scheduler = Scheduler()
    seen = []
    old = Agent("old")
    new = Agent("new")
    controller = AgentController(scheduler, seen.append)
    controller.observe(old)
    scheduler.run()
    stale = replace(old.state, display_name="stale")
    old.publish(stale)
    controller.observe(new)
    scheduler.run()
    assert (
        old.observation.closed and controller.state.agent_id == "new" and seen[-1].agent_id == "new"
    )


def test_commands_route_directly_and_close_does_not_stop():
    scheduler = Scheduler()
    agent = Agent("a")
    controller = AgentController(scheduler)
    controller.observe(agent)
    assert (
        controller.submit("hello")
        and controller.interrupt()
        and controller.withdraw_pending_input() == "x"
        and controller.stop()
    )
    controller.close()
    assert agent.observation.closed and agent.calls == [
        ("submit", "hello"),
        ("interrupt",),
        ("withdraw",),
        ("stop",),
    ]
    with pytest.raises(RuntimeError, match="not observing an agent"):
        controller.submit("x")


def test_reentrant_publication_is_coalesced_and_delivered():
    scheduler = Scheduler()
    agent = Agent("a")
    seen = []
    controller = None

    def changed(state):
        seen.append(state.display_name if state is not None else None)
        if state is not None and state.display_name == "first":
            agent.publish(replace(state, display_name="second"))

    controller = AgentController(scheduler, changed)
    controller.observe(agent)
    scheduler.run()
    seen.clear()

    agent.publish(replace(agent.state, display_name="first"))

    assert seen == ["first", "second"]
    assert controller.state is not None and controller.state.display_name == "second"
    controller.close()


def test_replacement_waits_for_running_callback_before_publishing_new_agent():
    scheduler = Scheduler()
    old = Agent("old")
    new = Agent("new")
    entered = threading.Event()
    release = threading.Event()
    completed = threading.Event()
    seen = []

    def changed(state):
        if state is not None and state.display_name == "blocking":
            entered.set()
            assert release.wait(1)
            completed.set()
        seen.append(state.agent_id if state is not None else None)

    controller = AgentController(scheduler, changed)
    controller.observe(old)
    scheduler.run()
    stale = replace(old.state, display_name="blocking")
    callback_thread = threading.Thread(target=lambda: old.publish(stale))
    callback_thread.start()
    assert entered.wait(1)

    switched = threading.Event()
    switch_thread = threading.Thread(target=lambda: (controller.observe(new), switched.set()))
    switch_thread.start()
    assert not switched.wait(0.05)
    release.set()
    callback_thread.join(1)
    switch_thread.join(1)

    assert completed.is_set() and switched.is_set()
    scheduler.run()
    assert controller.state is not None and controller.state.agent_id == "new"
    assert seen[-1] == "new"
    controller.close()


def test_replacement_waits_for_running_command_before_switching_agent():
    scheduler = Scheduler()
    entered = threading.Event()
    release = threading.Event()
    command_done = threading.Event()
    switched = threading.Event()

    class BlockingAgent(Agent):
        def submit(self, text):
            self.calls.append(("submit", text))
            entered.set()
            assert release.wait(1)
            return True

    old = BlockingAgent("old")
    new = Agent("new")
    controller = AgentController(scheduler)
    controller.observe(old)
    scheduler.run()

    command_thread = threading.Thread(
        target=lambda: (controller.submit("hello"), command_done.set())
    )
    command_thread.start()
    assert entered.wait(1)

    switch_thread = threading.Thread(target=lambda: (controller.observe(new), switched.set()))
    switch_thread.start()
    assert not switched.wait(0.05)

    release.set()
    command_thread.join(1)
    switch_thread.join(1)

    assert command_done.is_set() and switched.is_set()
    assert old.calls == [("submit", "hello")] and old.observation.closed
    assert controller.state is not None and controller.state.agent_id == "new"
    controller.close()


def test_callback_command_is_rejected_during_replacement_without_deadlock():
    scheduler = Scheduler()
    old = Agent("old")
    entered = threading.Event()
    replacement_observing = threading.Event()
    issue_command = threading.Event()
    callback_done = threading.Event()
    switched = threading.Event()
    outcomes = []

    class Replacement(Agent):
        def observe(self, callback, scheduler, on_terminated=None):
            replacement_observing.set()
            return super().observe(callback, scheduler, on_terminated)

    controller = AgentController(scheduler)

    def changed(state):
        if state is not None and state.display_name == "blocking":
            entered.set()
            assert issue_command.wait(1)
            try:
                controller.submit("from callback")
            except RuntimeError as exc:
                outcomes.append(str(exc))
            callback_done.set()

    controller._on_change = changed
    controller.observe(old)
    scheduler.run()
    thread = threading.Thread(
        target=lambda: old.publish(replace(old.state, display_name="blocking"))
    )
    thread.start()
    assert entered.wait(1)

    replacement = Replacement("new")
    switch = threading.Thread(target=lambda: (controller.observe(replacement), switched.set()))
    switch.start()
    assert replacement_observing.wait(1)
    issue_command.set()

    thread.join(1)
    switch.join(1)
    assert callback_done.is_set() and switched.is_set()
    assert outcomes == ["agent transition is in progress"]
    assert old.calls == []
    controller.close()


def test_replacement_close_failure_leaves_new_installation_usable():
    scheduler = Scheduler()
    old = Agent("old")
    new = Agent("new")
    controller = AgentController(scheduler)
    controller.observe(old)

    def fail_close():
        old.observation.closed = True
        raise RuntimeError("close failed")

    old.observation.close = fail_close
    with pytest.raises(RuntimeError, match="close failed"):
        controller.observe(new)

    assert controller.state is new.state
    assert controller.submit("after failure")
    controller.close()
    assert new.observation.closed


def test_terminal_close_failure_does_not_poison_later_close():
    scheduler = Scheduler()
    agent = Agent("old")
    controller = AgentController(scheduler)
    controller.observe(agent)

    def fail_close():
        agent.observation.closed = True
        raise RuntimeError("close failed")

    agent.observation.close = fail_close
    with pytest.raises(RuntimeError, match="close failed"):
        controller.close()
    assert controller.state is None
    controller.close()


def test_observation_failure_disconnects_and_gates_commands():
    scheduler = Scheduler()
    seen = []
    agent = Agent("old")
    controller = AgentController(scheduler, seen.append)
    controller.observe(agent)
    scheduler.run()
    failure = RuntimeError("scheduler stopped")

    assert agent.terminated is not None
    agent.observation.failure = failure
    agent.terminated(failure)

    assert controller.state is None
    assert controller.failure is failure
    assert seen[-1] is None
    with pytest.raises(RuntimeError, match="not observing an agent"):
        controller.submit("rejected")
