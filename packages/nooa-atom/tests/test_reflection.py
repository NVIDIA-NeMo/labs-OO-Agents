# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reflection: when the session sends it, where it goes, and who gets it.

The Gym pin (a guarded episode keeps it off) is in test_gym.py.
"""

import asyncio
import logging
from types import SimpleNamespace

import pytest
from atom_test_agents import CellLLM, cell
from nooa_atom.agent.factory import create_session_agent
from nooa_atom.agent.metrics import AtomHarnessMetrics
from nooa_atom.session.items import ItemAdmittedUpdate, TurnStartedUpdate
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.reflection import (
    DEFAULT_INSTRUCTIONS,
    ReflectionSettings,
    ReflectionTrigger,
    context_fraction,
    has_queue,
    reflection_enabled,
)
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore
from nooa_atom.workspace.options import AtomOptions


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


# ---- the trigger ------------------------------------------------------------


def _trigger(clock=None, **values):
    return ReflectionTrigger(ReflectionSettings(**values), clock=clock or Clock())


def _ends(trigger, count, kind="done", fraction=None):
    return [trigger.turn_ended(kind, context_fraction=fraction) for _ in range(count)]


def test_defaults():
    settings = ReflectionSettings()
    assert settings.enabled is None
    assert (settings.every_turns, settings.every_minutes) == (20, 60.0)
    assert (settings.min_gap_turns, settings.context_fraction) == (5, 0.5)
    assert settings.instructions == DEFAULT_INSTRUCTIONS
    for step in ("rename_session", "self.events.collapse", "del", "self.v", "self.todo"):
        assert step in DEFAULT_INSTRUCTIONS
    assert "not a request from the user" in DEFAULT_INSTRUCTIONS
    assert DEFAULT_INSTRUCTIONS.endswith("finish with a short Done.")
    assert AtomOptions().reflection == settings
    with pytest.raises(ValueError, match="extra"):
        ReflectionSettings(children=True)  # root only: there is no setting for children


def test_turns_trigger():
    assert _ends(_trigger(), 20) == [None] * 19 + ["turns"]


def test_minutes_trigger_needs_the_minimum_gap():
    clock = Clock()
    trigger = _trigger(clock)
    clock.now = 3600
    assert _ends(trigger, 5) == [None] * 4 + ["minutes"]


def test_context_trigger_needs_the_minimum_gap():
    assert _ends(_trigger(), 5, "waiting", 0.7) == [None] * 4 + ["context"]
    assert _ends(_trigger(), 6, "done", 0.49) == [None] * 6


def test_only_done_and_waiting_turns_count():
    trigger = _trigger(every_turns=2, min_gap_turns=0)
    for kind in ("need_input", "cancelled", "error"):
        assert trigger.turn_ended(kind, context_fraction=1.0) is None
    assert trigger.turns == 0
    assert _ends(trigger, 2) == [None, "turns"]


def test_a_trigger_set_to_none_is_off():
    clock = Clock()
    trigger = _trigger(clock, every_turns=None, every_minutes=None, context_fraction=None)
    clock.now = 10**6
    assert _ends(trigger, 50, fraction=1.0) == [None] * 50


def test_reflected_resets_the_counters_and_the_gap_applies_again():
    clock = Clock()
    trigger = _trigger(clock, every_turns=3, min_gap_turns=2)
    assert _ends(trigger, 3)[-1] == "turns"
    clock.now = 7200
    trigger.reflected()
    assert (trigger.turns, trigger.minutes()) == (0, 0)
    clock.now = 7200 + 3600 * 5  # long overdue by time, and the context is full
    assert trigger.turn_ended("done", context_fraction=0.9) is None  # 1 turn < min gap 2
    assert trigger.turn_ended("done", context_fraction=0.9) == "minutes"


@pytest.mark.parametrize(
    ("host", "settings", "override", "root", "expected"),
    [
        ("headless", ReflectionSettings(), None, True, False),
        ("acp", ReflectionSettings(), None, True, True),
        ("tui", ReflectionSettings(), None, True, False),  # only ACP is on by default
        ("headless", ReflectionSettings(enabled=True), None, True, True),
        ("acp", ReflectionSettings(enabled=False), None, True, False),
        ("headless", ReflectionSettings(), True, True, True),  # a benchmark arm turns it on
        ("acp", ReflectionSettings(enabled=True), False, True, False),
        ("acp", None, None, True, False),  # not an Atom agent: only when told to
        ("headless", None, True, True, True),
        # a child never reflects, whatever the settings or the override say
        ("acp", ReflectionSettings(), None, False, False),
        ("acp", ReflectionSettings(enabled=True), True, False, False),
    ],
)
def test_enabled_by_host_and_explicit_choice(host, settings, override, root, expected):
    assert reflection_enabled(settings, override=override, host=host, is_root=root) is expected


def test_context_fraction_is_the_context_usage_number():
    agent = SimpleNamespace(context_stats=SimpleNamespace(overall_utilization=0.42))
    assert context_fraction(agent) == 0.42
    assert context_fraction(SimpleNamespace(context_stats=None)) is None


def test_has_queue():
    assert not has_queue(SimpleNamespace(), "system_messages")
    manager = SimpleNamespace(channels=lambda: {"system_messages": SimpleNamespace(mode="queue")})
    assert has_queue(SimpleNamespace(queue_manager=manager), "system_messages")
    assert not has_queue(SimpleNamespace(queue_manager=manager), "other")


def test_workspace_settings_configure_it(tmp_path):
    (tmp_path / ".nooa").mkdir()
    (tmp_path / ".nooa" / "settings.yaml").write_text(
        "atom:\n  reflection:\n    enabled: true\n    every_turns: 3\n    instructions: Tidy.\n"
    )
    settings = AtomOptions.load(tmp_path).reflection
    assert (settings.enabled, settings.every_turns, settings.instructions) == (True, 3, "Tidy.")
    assert settings.min_gap_turns == 5


def test_the_counter_metric():
    metrics = AtomHarnessMetrics()
    metrics.reflection_queued()
    metrics.reflection_queued()
    assert metrics.reflections == 2


# ---- sessions ---------------------------------------------------------------


def done(text):
    return cell(f"return_result(Done(explanation={text!r}))")


REFLECT = "REFLECT NOW"


async def make_root(tmp_path, responses, *, host="acp", reflection=None, edit=None, **settings):
    llm = CellLLM(responses)
    settings = {"every_turns": 2, "min_gap_turns": 1, "instructions": REFLECT, **settings}

    def factory(options, storage):
        agent = create_session_agent(options, storage)
        agent._reflection_settings = ReflectionSettings(**settings)
        if edit is not None:
            edit(agent)
        return agent

    registry = SessionRegistry(SessionStore(tmp_path / "sessions"), agent_factory=factory)
    root = await registry.create(
        SessionOptions(
            workspace=tmp_path,
            agent_spec="nooa_atom.agent:AtomAgent",
            llm=llm,
            host=host,
            reflection=reflection,
        )
    )
    updates = []
    root.subscribe(updates.append)
    return registry, root, llm, updates


async def idle(root):
    async def wait():
        while root._agent.turns.running or root.has_queued_input():
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.02)
        while root._agent.turns.running or root.has_queued_input():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(wait(), 10)


def turn_channels(updates):
    """The channels of each started turn's items."""
    channels = {u.item_id: u.channel for u in updates if isinstance(u, ItemAdmittedUpdate)}
    return [
        [channels.get(i) for i in u.item_ids] for u in updates if isinstance(u, TurnStartedUpdate)
    ]


def reflections(updates):
    return [u for u in updates if getattr(u, "source", None) == "reflection"]


async def test_a_due_reflection_lands_on_system_messages_and_runs_a_normal_turn(tmp_path):
    registry, root, llm, updates = await make_root(
        tmp_path, [done("one"), done("two"), done("tidied"), done("three"), done("tidied again")]
    )
    try:
        assert (await root.prompt("one")).explanation == "one"
        assert (await root.prompt("two")).explanation == "two"
        await idle(root)
        [admitted] = reflections(updates)
        assert (admitted.channel, admitted.text) == ("system_messages", REFLECT)
        assert turn_channels(updates) == [["user_messages"], ["user_messages"], ["system_messages"]]
        assert REFLECT in str(llm.calls[2].messages)
        assert root._reflection.turns == 1  # reset when sent; the reflection turn counted
        assert (await root.prompt("three")).explanation == "three"
        await idle(root)
        # Two more turns (three, then this one's reflection) since the reset: due again.
        assert len(reflections(updates)) == 2 and len(llm.calls) == 5
        assert root._reflection.turns == 1
    finally:
        await registry.close_all()


async def test_the_minimum_gap_holds_back_a_due_trigger(tmp_path):
    registry, root, _llm, updates = await make_root(
        tmp_path, [done("a"), done("b"), done("c")], every_turns=1, min_gap_turns=3
    )
    try:
        for text in "ab":
            await root.prompt(text)
        await idle(root)
        assert reflections(updates) == []
        await root.prompt("c")
        assert len(reflections(updates)) == 1
    finally:
        await registry.close_all()


async def test_headless_sessions_do_not_reflect_unless_told(tmp_path):
    registry, root, _llm, updates = await make_root(
        tmp_path, [done("a"), done("b"), done("c")], host="headless"
    )
    try:
        assert root._reflection is None
        for text in "abc":
            await root.prompt(text)
        await idle(root)
        assert reflections(updates) == []
    finally:
        await registry.close_all()
    registry, root, _llm, updates = await make_root(
        tmp_path / "on", [done("a"), done("b"), done("tidied")], host="headless", reflection=True
    )
    try:
        await root.prompt("a")
        await root.prompt("b")
        await idle(root)
        assert turn_channels(updates)[-1] == ["system_messages"]
    finally:
        await registry.close_all()


async def test_explicit_false_turns_it_off_in_an_acp_host(tmp_path):
    registry, root, _llm, _updates = await make_root(tmp_path, [], reflection=False)
    try:
        assert root._reflection is None
    finally:
        await registry.close_all()


async def test_only_the_root_session_reflects(tmp_path):
    registry, root, _llm, _updates = await make_root(tmp_path, [], reflection=True, enabled=True)
    try:
        assert root._reflection is not None
        child = await registry.create(
            SessionOptions(
                workspace=tmp_path,
                agent_spec="nooa_atom.agent:AtomAgent",
                llm=CellLLM([]),
                host="acp",
                reflection=True,
            ),
            parent_id=root.id,
        )
        assert child.parent_id == root.id
        assert child._reflection is None
    finally:
        await registry.close_all()


async def test_no_system_messages_queue_means_no_reflection(tmp_path):
    def drop_queue(agent):
        agent.queue_manager._channels.pop("system_messages")

    registry, root, _llm, _updates = await make_root(tmp_path, [], reflection=True, edit=drop_queue)
    try:
        assert root._reflection is None
    finally:
        await registry.close_all()


async def test_a_failing_check_is_logged_and_the_turn_is_unaffected(tmp_path, caplog):
    registry, root, _llm, updates = await make_root(tmp_path, [done("one"), done("two")])

    def boom(kind, *, context_fraction):
        raise RuntimeError("boom")

    root._reflection.turn_ended = boom
    try:
        with caplog.at_level(logging.ERROR, logger="nooa_atom.session.session"):
            assert (await root.prompt("one")).explanation == "one"
            assert (await root.prompt("two")).explanation == "two"
        await idle(root)
        assert reflections(updates) == []
        assert root.last_outcome_kind == "done" and root.info.status == "idle"
        assert sum("reflection check failed" in r.getMessage() for r in caplog.records) == 2
    finally:
        await registry.close_all()
