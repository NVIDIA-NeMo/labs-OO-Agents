# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fake-clock admission and strict-fake Session tests; never sleep for the cooldown."""

import asyncio
import math

import pytest
from atom_test_agents import CellLLM, cell
from nooa_atom.agent.agent import AtomAgent
from nooa_atom.agent.factory import create_session_agent
from nooa_atom.agent.reflection_gate import _ReflectionGate
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

from nooa.interactive import Done, NeedInput
from nooa.unifiedllm import LLMResponse


class Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


def record(gate, sequence, kind="done", **kwargs):
    values = {"method": "handle", "committed": True, "ran": True, **kwargs}
    return gate.settled(sequence, kind=kind, **values)


@pytest.mark.parametrize(
    "turns,seconds,expected",
    [
        (49, 3599, False),
        (50, 3599, False),
        (49, 3600, False),
        (50, 3600, True),
    ],
)
def test_default_gate_and_truth_table(turns, seconds, expected):
    clock = Clock()
    gate = _ReflectionGate(clock=clock)
    assert (gate.min_turns, gate.min_seconds) == (50, 3600.0)
    for sequence in range(1, turns + 1):
        assert record(gate, sequence)
    clock.now = seconds
    assert gate.admit(turns) is expected
    assert gate.turns == (0 if expected else turns)
    assert gate.anchor == (seconds if expected else 0)


def test_exact_boundary_skipped_done_preserves_window_but_cannot_be_retried_idle():
    clock, gate = Clock(), None
    gate = _ReflectionGate(min_turns=2, min_seconds=10, clock=clock)
    record(gate, 1)
    assert not gate.admit(1)
    record(gate, 2)
    clock.now = math.nextafter(10.0, 0.0)
    assert not gate.admit(2)
    assert gate.turns == 2 and gate.anchor == 0
    clock.now = 10.0
    assert not gate.admit(2)  # elapsed time alone cannot revive the skipped Done
    record(gate, 3)
    assert gate.admit(3)
    assert gate.turns == 0 and gate.anchor == 10.0
    assert not gate.admit(3)
    assert not record(gate, 3) and not record(gate, 1)
    assert gate.turns == 0


@pytest.mark.parametrize(
    "kind,values,counted",
    [
        ("done", {}, True),
        ("waiting", {}, True),
        ("need_input", {}, True),
        ("error", {}, False),
        ("cancelled", {}, False),
        ("done", {"committed": False}, False),
        ("done", {"ran": False}, False),
        ("done", {"method": "other"}, False),
        ("done", {"method": "_reflect_done_plan"}, False),
    ],
)
def test_only_specified_settled_foreground_handles_count(kind, values, counted):
    gate = _ReflectionGate(min_turns=1, min_seconds=0, clock=Clock())
    opportunity = record(gate, 1, kind, **values)
    assert gate.turns == int(counted)
    assert opportunity is (counted and kind == "done")
    assert gate.admit(1) is (counted and kind == "done")


def test_duplicate_old_ordinals_do_not_count_or_consume_new_done():
    gate = _ReflectionGate(min_turns=2, min_seconds=0, clock=Clock())
    record(gate, 1, "waiting")
    record(gate, 2)
    assert not record(gate, 1, "need_input")
    assert not record(gate, 2)
    assert gate.turns == 2 and gate.admit(2)
    assert not gate.admit(None)


def test_fresh_start_and_independent_gates_never_share_windows():
    clock = Clock(-100)
    root = _ReflectionGate(min_turns=2, min_seconds=10, clock=clock)
    record(root, 1)
    clock.now = -95
    child = _ReflectionGate(min_turns=2, min_seconds=10, clock=clock)
    record(child, 1)
    clock.now = -90
    record(root, 2)
    assert root.admit(2)
    assert child.turns == 1 and child.anchor == -95
    root.start_session()
    assert root.turns == 0 and root.anchor == -90
    assert not root.admit(2)
    record(root, 100)  # restored durable reply ordinal is not historical credit
    assert root.turns == 1


_BAD_TURNS = [0, -1, True, False, 50.0, "50", None, float("nan"), float("inf")]
_BAD_SECONDS = [-1, True, False, "3600", None, float("nan"), float("inf"), -float("inf"), 10**400]


@pytest.mark.parametrize("value", _BAD_TURNS, ids=[str(i) for i in range(len(_BAD_TURNS))])
@pytest.mark.parametrize("entry", ["gate", "options", "agent"])
def test_invalid_min_turns_are_not_coerced(tmp_path, value, entry):
    with pytest.raises(ValueError, match="min_turns"):
        if entry == "gate":
            _ReflectionGate(min_turns=value)
        elif entry == "options":
            SessionOptions(
                workspace=tmp_path,
                agent_spec="nooa_atom.agent:AtomAgent",
                experimental_reflection_min_turns=value,
            )
        else:
            AtomAgent(cwd=tmp_path, llm=CellLLM(), experimental_reflection_min_turns=value)


@pytest.mark.parametrize("value", _BAD_SECONDS, ids=[str(i) for i in range(len(_BAD_SECONDS))])
@pytest.mark.parametrize("entry", ["gate", "options", "agent"])
def test_invalid_min_seconds_are_not_coerced(tmp_path, value, entry):
    with pytest.raises(ValueError, match="min_seconds"):
        if entry == "gate":
            _ReflectionGate(min_seconds=value)
        elif entry == "options":
            SessionOptions(
                workspace=tmp_path,
                agent_spec="nooa_atom.agent:AtomAgent",
                experimental_reflection_min_seconds=value,
            )
        else:
            AtomAgent(cwd=tmp_path, llm=CellLLM(), experimental_reflection_min_seconds=value)


@pytest.mark.parametrize("turns,seconds", [(1, 0), (50, 3600), (2, 0.25)])
def test_valid_configuration_and_child_inheritance(tmp_path, turns, seconds):
    options = SessionOptions(
        workspace=tmp_path,
        agent_spec="nooa_atom.agent:AtomAgent",
        experimental_done_reflection=True,
        experimental_reflection_min_turns=turns,
        experimental_reflection_min_seconds=seconds,
    )
    child = options.inherit(name="child", kept=True)
    assert (child.experimental_reflection_min_turns, child.experimental_reflection_min_seconds) == (
        turns,
        seconds,
    )
    assert child.experimental_done_reflection is True


async def idle(root):
    async def wait():
        while root._agent.turns.running:
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), 10)


def done(text="accepted"):
    return cell(f"return_result(Done(explanation={text!r}))")


def prediction(content="{}"):
    return LLMResponse(content=content, finish_reason="stop")


def prediction_count(llm):
    return sum(call.output_model is not None for call in llm.calls)


async def make_root(tmp_path, responses, clock, *, llm=None, **values):
    llm = CellLLM(responses) if llm is None else llm
    options = SessionOptions(
        workspace=tmp_path,
        agent_spec="nooa_atom.agent:AtomAgent",
        llm=llm,
        experimental_namespace="session",
        experimental_done_reflection=True,
        **values,
    )

    def factory(options, storage):
        agent = create_session_agent(options, storage)
        gate = agent._reflection_gate
        agent._reflection_gate = _ReflectionGate(
            min_turns=gate.min_turns, min_seconds=gate.min_seconds, clock=clock
        )
        return agent

    registry = SessionRegistry(SessionStore(tmp_path / "sessions"), agent_factory=factory)
    root = await registry.create(options)
    return registry, root, llm


async def test_actual_default_model_count_requires_fifty_turns_and_one_hour(tmp_path):
    clock = Clock()
    registry, root, llm = await make_root(
        tmp_path, [*[done() for _ in range(52)], prediction()], clock
    )
    settled = []
    root._agent.event_manager.on("TurnSettled", settled.append)
    try:
        gate = root._agent._reflection_gate
        assert (gate.min_turns, gate.min_seconds) == (50, 3600)
        for _ in range(50):
            result = await root.prompt("rapid request")
            await idle(root)
            assert result is settled[-1].result
        assert prediction_count(llm) == 0 and len(llm.calls) == 50
        assert gate.turns == 50 and gate.anchor == 0
        clock.now = math.nextafter(3600.0, 0.0)
        await root.prompt("just before the time boundary")
        await idle(root)
        assert prediction_count(llm) == 0 and gate.anchor == 0
        clock.now = 3600.0
        for _ in range(10):
            await asyncio.sleep(0)
        assert len(llm.calls) == 51  # time alone never starts maintenance
        result = await root.prompt("at the boundary")
        await idle(root)
        assert result is settled[-1].result
        assert prediction_count(llm) == 1 and len(llm.calls) == 53
        assert gate.turns == 0 and gate.anchor == 3600
        assert root.info.reply_count == 52  # maintenance did not count itself
        await root._after_turn_settled(settled[-1])  # duplicate delivery cannot re-admit
        assert prediction_count(llm) == 1 and gate.turns == 0
    finally:
        await registry.close_all()


async def test_default_time_alone_and_creation_anchor(tmp_path):
    clock = Clock()
    registry, root, llm = await make_root(tmp_path, [done()], clock)
    try:
        clock.now = 3600
        assert await root.prompt("only one turn") == Done(explanation="accepted")
        await idle(root)
        assert prediction_count(llm) == 0 and root._agent._reflection_gate.turns == 1
    finally:
        await registry.close_all()


class AttemptLLM(CellLLM):
    def __init__(self, responses, clock, mode, *, finish_advance=0):
        super().__init__(responses)
        self.clock, self.mode = clock, mode
        self.finish_advance = finish_advance
        self.attempts = []
        self.entered, self.cancelled = asyncio.Event(), asyncio.Event()

    async def acall(self, *args, **kwargs):
        if kwargs.get("output_model") is not None:
            self.attempts.append(self.clock.now)
            self.entered.set()
            if self.mode == "exception":
                raise RuntimeError("prediction failed")
            if self.mode == "timeout":
                try:
                    await asyncio.Event().wait()
                finally:
                    self.cancelled.set()
            result = await super().acall(*args, **kwargs)
            self.clock.now += self.finish_advance
            return result
        return await super().acall(*args, **kwargs)


@pytest.mark.parametrize("mode", ["noop", "invalid", "exception", "timeout"])
async def test_failed_timeout_and_noop_attempts_all_consume_cooldown(tmp_path, monkeypatch, mode):
    from nooa_atom.agent import done_reflection

    monkeypatch.setattr(done_reflection, "REFLECTION_TIMEOUT", 0.03)
    clock = Clock()
    predicted = (
        [prediction("not JSON" if mode == "invalid" else "{}")]
        if mode in {"noop", "invalid"}
        else []
    )
    llm = AttemptLLM(
        [done(), done(), *predicted, *[done() for _ in range(4)], *predicted], clock, mode
    )
    registry, root, _ = await make_root(
        tmp_path,
        [],
        clock,
        llm=llm,
        experimental_reflection_min_turns=2,
        experimental_reflection_min_seconds=10,
    )
    try:
        await root.prompt("one")
        clock.now = 10
        accepted = await root.prompt("two")
        assert accepted == Done(explanation="accepted")
        await idle(root)
        gate = root._agent._reflection_gate
        assert llm.attempts == [10] and gate.turns == 0 and gate.anchor == 10
        if mode == "timeout":
            assert llm.cancelled.is_set()
        for _ in range(3):
            assert await root.prompt("rapid") == Done(explanation="accepted")
            await idle(root)
        assert llm.attempts == [10] and gate.turns == 3 and gate.anchor == 10
        clock.now = 20
        await root.prompt("eligible again")
        await idle(root)
        assert llm.attempts == [10, 20] and gate.turns == 0 and gate.anchor == 20
        assert root.info.reply_count == 6
    finally:
        await registry.close_all()


async def test_completion_time_does_not_move_attempt_anchor(tmp_path):
    clock = Clock()
    llm = AttemptLLM(
        [done(), done(), prediction(), done(), done(), done(), prediction()],
        clock,
        "noop",
        finish_advance=9,
    )
    registry, root, _ = await make_root(
        tmp_path,
        [],
        clock,
        llm=llm,
        experimental_reflection_min_turns=2,
        experimental_reflection_min_seconds=10,
    )
    try:
        await root.prompt("one")
        clock.now = 10
        await root.prompt("two")
        await idle(root)
        assert clock.now == 19 and root._agent._reflection_gate.anchor == 10
        for _ in range(2):
            await root.prompt("rapid")
            await idle(root)
        assert llm.attempts == [10]
        clock.now = 20
        await root.prompt("ten seconds since attempt, not since finish")
        await idle(root)
        assert llm.attempts == [10, 20]
        assert clock.now == 29 and root._agent._reflection_gate.anchor == 20
    finally:
        await registry.close_all()


async def test_waiting_and_need_input_count_but_only_done_admits(tmp_path):
    clock = Clock()
    responses = [cell("return_result(NeedInput(question='q'))") for _ in range(3)]
    responses += [
        cell(
            "job=self.jobs.launch(asyncio.sleep(100), name='pending')\nreturn_result(Waiting(explanation='wait', on=['pending']))"
        ),
        done(),
        prediction(),
    ]
    registry, root, llm = await make_root(
        tmp_path,
        responses,
        clock,
        experimental_reflection_min_turns=4,
        experimental_reflection_min_seconds=10,
    )
    pending = None
    try:
        clock.now = 10
        for _ in range(3):
            assert await root.prompt("ask") == NeedInput(question="q")
            await idle(root)
        pending = asyncio.create_task(root.prompt("wait"))
        while root._agent.jobs.get("pending") is None:
            await asyncio.sleep(0.001)
        await idle(root)
        assert root._agent._reflection_gate.turns == 4 and prediction_count(llm) == 0
        result = await root.prompt("done")
        assert await pending is result
        await idle(root)
        assert prediction_count(llm) == 1 and root.info.reply_count == 5
        assert root._agent._reflection_gate.turns == 0
    finally:
        await registry.close_all()
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)


async def test_cells_rejected_returns_errors_and_cancellation_are_not_turn_credit(tmp_path):
    from nooa_atom.session.session import TurnFailedError

    clock = Clock()
    responses = [
        cell("intermediate=1"),
        cell("return_result(explanation='')"),  # rejected; no settlement
        done(),
        cell("self.started.set()\nawait self.block.wait()"),
        done(),
    ]

    class FailingLLM(CellLLM):
        fail = False
        api_failures = 0

        async def acall(self, *args, **kwargs):
            if self.fail:
                self.api_failures += 1
                raise RuntimeError("model call failed")
            return await super().acall(*args, **kwargs)

    llm = FailingLLM(responses)
    registry, root, _ = await make_root(tmp_path, [], clock, llm=llm)
    pending = None
    try:
        clock.now = 3600
        await root.prompt("several cells, one corrected return")
        await idle(root)
        gate = root._agent._reflection_gate
        assert gate.turns == 1 and prediction_count(llm) == 0
        llm.fail = True
        with pytest.raises(TurnFailedError):
            await root.prompt("error")
        llm.fail = False
        await idle(root)
        assert llm.api_failures == 3
        assert gate.turns == 1 and gate.anchor == 0
        root._agent.started, root._agent.block = asyncio.Event(), asyncio.Event()
        pending = asyncio.create_task(root.prompt("cancel me"))
        await asyncio.wait_for(root._agent.started.wait(), 5)
        assert await root.cancel(by="test")
        await pending
        await idle(root)
        assert gate.turns == 1 and gate.anchor == 0
        await root.prompt("one more settled handle")
        await idle(root)
        assert gate.turns == 2 and prediction_count(llm) == 0
        assert root.info.reply_count == 4  # includes error/cancel, but gate does not
    finally:
        await registry.close_all()
        if pending is not None:
            await asyncio.gather(pending, return_exceptions=True)


async def test_session_creation_not_agent_construction_is_initial_time_anchor(tmp_path):
    clock = Clock()
    llm = CellLLM([done(), done(), prediction()])

    def factory(options, storage):
        agent = create_session_agent(options, storage)
        agent._reflection_gate = _ReflectionGate(min_turns=1, min_seconds=10, clock=clock)
        clock.now = 100  # simulate time spent building/provisioning the agent
        return agent

    registry = SessionRegistry(SessionStore(tmp_path / "sessions"), agent_factory=factory)
    try:
        root = await registry.create(
            SessionOptions(
                workspace=tmp_path,
                agent_spec="nooa_atom.agent:AtomAgent",
                llm=llm,
                experimental_done_reflection=True,
                experimental_reflection_min_turns=1,
                experimental_reflection_min_seconds=10,
            )
        )
        assert root._agent._reflection_gate.anchor == 100
        await root.prompt("not old enough")
        await idle(root)
        assert prediction_count(llm) == 0
        clock.now = 110
        await root.prompt("eligible")
        await idle(root)
        assert prediction_count(llm) == 1
    finally:
        await registry.close_all()


@pytest.mark.parametrize("action", ["cancel", "close"])
async def test_owned_cancel_or_close_while_reflection_running_preserves_done(tmp_path, action):
    clock = Clock()
    llm = AttemptLLM([done()], clock, "timeout")
    registry, root, _ = await make_root(
        tmp_path,
        [],
        clock,
        llm=llm,
        experimental_reflection_min_turns=1,
        experimental_reflection_min_seconds=0,
    )
    try:
        outcome = await root.prompt("go")
        assert outcome == Done(explanation="accepted")
        await asyncio.wait_for(llm.entered.wait(), 5)
        gate = root._agent._reflection_gate
        assert gate.turns == 0 and gate.anchor == 0 and llm.attempts == [0]
        if action == "cancel":
            assert await root.cancel(by="test")
            await idle(root)
        else:
            await asyncio.wait_for(root.close(), 5)
        assert llm.cancelled.is_set()
        assert root.info.reply_count == 1 and gate.turns == 0
    finally:
        await registry.close_all()


async def test_close_before_admission_does_not_consume_eligible_window(tmp_path, monkeypatch):
    from nooa_atom.session.session import Session

    entered, cancelled = asyncio.Event(), asyncio.Event()
    original = Session._after_turn_settled

    async def held(self, event):
        entered.set()
        try:
            await asyncio.Event().wait()
            await original(self, event)
        finally:
            cancelled.set()

    monkeypatch.setattr(Session, "_after_turn_settled", held)
    clock = Clock()
    registry, root, llm = await make_root(
        tmp_path,
        [done()],
        clock,
        experimental_reflection_min_turns=1,
        experimental_reflection_min_seconds=0,
    )
    result = await root.prompt("go")
    await asyncio.wait_for(entered.wait(), 5)
    gate = root._agent._reflection_gate
    assert gate.turns == 1 and gate.anchor == 0
    await asyncio.wait_for(registry.close_all(), 5)
    assert cancelled.is_set() and prediction_count(llm) == 0
    assert result == Done(explanation="accepted") and root.info.reply_count == 1
    assert gate.turns == 1  # no attempt was admitted


def watch_generation(agent, label, seen):
    """Live runtime events, not conversation history (which omits these events)."""

    def capture(event):
        seen.append((label, event.event_type, event.method_name))

    agent.event_manager.on("BeforeAgentCall", capture)
    agent.event_manager.on("BeforeTurn", capture)


def live_reflection_attempts(seen):
    return [entry for entry in seen if entry[2] == "_reflect_done_plan"]


async def test_standalone_enabled_positive_control_proves_runtime_canary_liveness(tmp_path):
    clock = Clock()
    registry, root, llm = await make_root(
        tmp_path,
        [done(), prediction()],
        clock,
        experimental_reflection_min_turns=1,
        experimental_reflection_min_seconds=0,
    )
    seen = []
    watch_generation(root._agent, "standalone-positive-control", seen)
    try:
        assert await root.prompt("go") == Done(explanation="accepted")
        await idle(root)
        attempts = live_reflection_attempts(seen)
        assert {entry[1] for entry in attempts} == {"BeforeAgentCall", "BeforeTurn"}
        assert prediction_count(llm) == 1
        assert root.info.reply_count == 1 and root._agent._reflection_gate.turns == 0
    finally:
        await registry.close_all()


_WORK_DONE = "return_result(Done(explanation='child', result=TaskResult(solution_description='s', evidence='e', how_to_verify='v', report='child')))"


@pytest.mark.parametrize("mode", ["legacy", "call", "session"])
async def test_benchmark_policy_explicit_off_root_child_has_zero_live_reflection_attempts(
    tmp_path, mode
):
    # This is a fake-model canary, not a benchmark run or preset activation.
    clock, seen, agents = Clock(), [], {}
    clients = {
        None: CellLLM(
            [
                cell(
                    "r=await self.jobs.launch(Subagent('child').run('go'), name='kid')\nassert r.report == 'child'\nreturn_result(Done(explanation='root'))"
                )
            ]
        ),
        "kid": CellLLM([cell(_WORK_DONE)]),
    }

    def factory(options, storage):
        assert options.experimental_done_reflection is False
        agent = create_session_agent(
            options.model_copy(update={"llm": clients[options.name]}), storage
        )
        gate = agent._reflection_gate
        agent._reflection_gate = _ReflectionGate(
            min_turns=gate.min_turns, min_seconds=gate.min_seconds, clock=clock
        )
        watch_generation(agent, options.name, seen)
        agents[options.name] = agent
        return agent

    registry = SessionRegistry(SessionStore(tmp_path / "sessions"), agent_factory=factory)
    try:
        options = SessionOptions(
            workspace=tmp_path,
            agent_spec="nooa_atom.agent:AtomAgent",
            experimental_namespace=mode,
            experimental_done_reflection=False,
            experimental_reflection_min_turns=1,
            experimental_reflection_min_seconds=0,
        )
        root = await registry.create(options)
        clock.now = 10000  # fully eligible thresholds if the explicit OFF flag were ignored
        assert await root.prompt("go") == Done(explanation="root")
        await idle(root)
        assert set(agents) == {None, "kid"}
        assert all(agent._done_reflection is False for agent in agents.values())
        assert all(agent._reflection_gate.turns == 0 for agent in agents.values())
        assert all(prediction_count(llm) == 0 for llm in clients.values())
        assert live_reflection_attempts(seen) == []
        assert {label for label, _, method in seen if method == "handle"} == {None, "kid"}
        assert root.info.reply_count == 1
    finally:
        await registry.close_all()


async def test_root_and_kept_child_gate_state_and_options_are_independent(tmp_path):
    clock, agents = Clock(), {}
    clients = {
        None: CellLLM(
            [
                cell(
                    "child=self.jobs.launch(Subagent('child'), name='kid')\nr=await child.send('one')\nreturn_result(Done(explanation='one'))"
                ),
                cell("r=await child.send('two')\nreturn_result(Done(explanation='two'))"),
                prediction(),
                done("three"),
                done("four"),
                prediction(),
            ]
        ),
        "kid": CellLLM([cell(_WORK_DONE), cell(_WORK_DONE), prediction()]),
    }

    def factory(options, storage):
        agent = create_session_agent(
            options.model_copy(update={"llm": clients[options.name]}), storage
        )
        gate = agent._reflection_gate
        agent._reflection_gate = _ReflectionGate(
            min_turns=gate.min_turns, min_seconds=gate.min_seconds, clock=clock
        )
        agents[options.name] = agent
        return agent

    registry = SessionRegistry(SessionStore(tmp_path / "sessions"), agent_factory=factory)
    try:
        root = await registry.create(
            SessionOptions(
                workspace=tmp_path,
                agent_spec="nooa_atom.agent:AtomAgent",
                experimental_namespace="session",
                experimental_done_reflection=True,
                experimental_reflection_min_turns=2,
                experimental_reflection_min_seconds=10,
            )
        )
        assert await root.prompt("one") == Done(explanation="one")
        await idle(root)
        [info] = registry.children(root.id)
        child = registry.sessions[info.id]
        await idle(child)
        root_gate, child_gate = agents[None]._reflection_gate, agents["kid"]._reflection_gate
        assert root_gate is not child_gate
        assert root_gate.turns == child_gate.turns == 1
        assert (
            child.options.experimental_reflection_min_turns,
            child.options.experimental_reflection_min_seconds,
        ) == (2, 10)
        assert child.options.experimental_done_reflection is True
        clock.now = 10
        assert await root.prompt("two") == Done(explanation="two")
        await idle(root)
        await idle(child)
        assert prediction_count(clients[None]) == prediction_count(clients["kid"]) == 1
        assert root_gate.turns == child_gate.turns == 0
        assert root_gate.anchor == child_gate.anchor == 10
        clock.now = 20
        await root.prompt("three")
        await idle(root)
        assert root_gate.turns == 1 and child_gate.turns == 0
        await root.prompt("four")
        await idle(root)
        assert root_gate.anchor == 20 and root_gate.turns == 0
        assert child_gate.anchor == 10 and child_gate.turns == 0
        assert prediction_count(clients[None]) == 2 and prediction_count(clients["kid"]) == 1
        assert root.info.reply_count == 4 and child.info.reply_count == 2
    finally:
        await registry.close_all()


async def test_restart_rearms_without_monotonic_snapshot_and_load_uses_explicit_selection(tmp_path):
    from nooa.storage.json_snapshot import snapshot_to_json

    clock = Clock()

    def factory(options, storage):
        agent = create_session_agent(options, storage)
        gate = agent._reflection_gate
        agent._reflection_gate = _ReflectionGate(
            min_turns=gate.min_turns, min_seconds=gate.min_seconds, clock=clock
        )
        return agent

    store = SessionStore(tmp_path / "sessions")
    registry = SessionRegistry(store, agent_factory=factory)
    try:
        root = await registry.create(
            SessionOptions(
                workspace=tmp_path,
                agent_spec="nooa_atom.agent:AtomAgent",
                llm=CellLLM(
                    [
                        cell("self.v.fact='durable'\nreturn_result(Done(explanation='one'))"),
                        done(),
                        prediction(),
                        done(),
                    ]
                ),
                experimental_namespace="call",
                experimental_done_reflection=True,
                experimental_reflection_min_turns=2,
                experimental_reflection_min_seconds=10,
            )
        )
        await root.prompt("one")
        clock.now = 10
        await root.prompt("two")
        await idle(root)
        clock.now = 11
        await root.prompt("three")
        await idle(root)
        assert root._agent._reflection_gate.turns == 1
        assert root._agent._reflection_gate.anchor == 10
        snapshot = snapshot_to_json(root._agent)
        assert "_reflection_gate" not in snapshot["attributes"]
        await root.wait_for_checkpoint()
        identity = root.id
    finally:
        await registry.close_all()

    clock.now = 1000
    registry = SessionRegistry(store, agent_factory=factory)
    client = CellLLM([done(), done(), done(), prediction()])
    try:
        root = await registry.load(
            identity,
            llm=client,
            experimental_namespace="call",
            experimental_done_reflection=True,
            experimental_reflection_min_turns=3,
            experimental_reflection_min_seconds=20,
        )
        gate = root._agent._reflection_gate
        assert (gate.min_turns, gate.min_seconds, gate.turns, gate.anchor) == (3, 20, 0, 1000)
        assert root._agent.v.fact == "durable" and root.info.reply_count == 3
        clock.now = 1020
        for _ in range(2):
            await root.prompt("new turn")
            await idle(root)
            assert prediction_count(client) == 0
        await root.prompt("third new turn")
        await idle(root)
        assert prediction_count(client) == 1 and gate.anchor == 1020 and gate.turns == 0
    finally:
        await registry.close_all()

    # Experimental selection is explicit on load, not inherited from a snapshot.
    clock.now = 2000
    registry = SessionRegistry(store, agent_factory=factory)
    try:
        root = await registry.load(identity, llm=CellLLM())
        gate = root._agent._reflection_gate
        assert root.options.experimental_done_reflection is False
        assert root._agent._done_reflection is False
        assert (gate.min_turns, gate.min_seconds, gate.turns, gate.anchor) == (50, 3600, 0, 2000)
    finally:
        await registry.close_all()


def test_nonqualifying_settlement_also_advances_dedup_high_water_mark():
    gate = _ReflectionGate(min_turns=2, min_seconds=0, clock=Clock())
    assert not record(gate, 1, "waiting")
    assert not record(gate, 3, "error")
    assert not record(gate, 2, "done")  # stale relative to the observed error
    assert gate.turns == 1 and gate.anchor == 0
    assert not gate.admit(2)
    assert record(gate, 4, "done") and gate.admit(4)
