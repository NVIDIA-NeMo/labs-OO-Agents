# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reflection turns: when they are queued, how they are delivered, and what they show."""

import asyncio
from types import SimpleNamespace

import pytest
from atom_test_agents import CellLLM, cell
from nooa_atom.agent.factory import create_session_agent
from nooa_atom.session.items import ItemAdmittedUpdate, TurnEndedUpdate, TurnStartedUpdate
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.reflection import (
    DEFAULT_INSTRUCTIONS,
    ReflectionSettings,
    ReflectionTrigger,
    context_fraction,
    emit_reflection_span,
    reflection_enabled,
)
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore
from nooa_atom.workspace.options import AtomOptions

from nooa.interactive import AgentMessage


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


# ---- the trigger ------------------------------------------------------------


def _trigger(clock=None, **values):
    return ReflectionTrigger(ReflectionSettings(**values), clock=clock or Clock())


def test_defaults():
    settings = ReflectionSettings()
    assert (settings.enabled, settings.children) == (None, False)
    assert (settings.every_turns, settings.every_minutes) == (20, 60.0)
    assert (settings.min_gap_turns, settings.context_fraction) == (5, 0.5)
    assert settings.instructions == DEFAULT_INSTRUCTIONS
    assert (
        "rename_session" in DEFAULT_INSTRUCTIONS and "self.events.collapse" in DEFAULT_INSTRUCTIONS
    )
    assert AtomOptions().reflection == settings


def test_turns_trigger():
    trigger = _trigger()
    reasons = [trigger.turn_ended("done", context_fraction=None) for _ in range(20)]
    assert reasons == [None] * 19 + ["turns"]


def test_minutes_trigger_needs_the_minimum_gap():
    clock = Clock()
    trigger = _trigger(clock)
    clock.now = 3600
    assert [trigger.turn_ended("done", context_fraction=None) for _ in range(5)] == [
        None,
        None,
        None,
        None,
        "minutes",
    ]


def test_context_trigger_needs_the_minimum_gap():
    trigger = _trigger()
    assert [trigger.turn_ended("waiting", context_fraction=0.7) for _ in range(5)] == [
        *[None] * 4,
        "context",
    ]
    fresh = _trigger(min_gap_turns=0)
    assert fresh.turn_ended("done", context_fraction=0.49) is None
    assert fresh.turn_ended("done", context_fraction=0.5) == "context"


def test_only_done_and_waiting_start_one_and_pending_blocks_more():
    trigger = _trigger(every_turns=2, min_gap_turns=0)
    assert trigger.turn_ended("need_input", context_fraction=None) is None
    assert trigger.turn_ended("error", context_fraction=None) is None  # not counted
    assert trigger.turns == 1
    assert trigger.turn_ended("need_input", context_fraction=None) is None  # 2, but asks
    assert trigger.turn_ended("done", context_fraction=None) == "turns"
    trigger.pending = True
    assert trigger.turn_ended("done", context_fraction=1.0) is None  # at most one pending
    trigger.reflected()
    assert (trigger.turns, trigger.pending) == (0, False)
    assert trigger.turn_ended("done", context_fraction=None) is None


def test_a_trigger_set_to_none_is_off():
    clock = Clock()
    trigger = _trigger(clock, every_turns=None, every_minutes=None, context_fraction=None)
    clock.now = 10**6
    assert all(trigger.turn_ended("done", context_fraction=1.0) is None for _ in range(50))


def test_hysteresis_after_a_reflection():
    clock = Clock()
    trigger = _trigger(clock, every_turns=3, min_gap_turns=2)
    assert [trigger.turn_ended("done", context_fraction=None) for _ in range(3)][-1] == "turns"
    clock.now = 7200  # long overdue by time, and the context is full
    trigger.reflected()
    assert trigger.turn_ended("done", context_fraction=0.9) is None  # 1 turn < min gap 2
    assert trigger.turn_ended("done", context_fraction=0.9) == "context"


@pytest.mark.parametrize(
    ("host", "settings", "override", "root", "expected"),
    [
        ("headless", ReflectionSettings(), None, True, False),
        ("acp", ReflectionSettings(), None, True, True),
        ("tui", ReflectionSettings(), None, True, True),
        ("acp", ReflectionSettings(), None, False, False),  # children are off
        ("acp", ReflectionSettings(children=True), None, False, True),
        ("headless", ReflectionSettings(enabled=True), None, True, True),
        ("acp", ReflectionSettings(enabled=False), None, True, False),
        ("headless", ReflectionSettings(), True, True, True),  # a benchmark arm turns it on
        ("acp", ReflectionSettings(enabled=True), False, True, False),
        ("acp", None, None, True, False),  # not an Atom agent: only when told to
        ("headless", None, True, True, True),
    ],
)
def test_enabled_by_host_and_explicit_choice(host, settings, override, root, expected):
    assert reflection_enabled(settings, override=override, host=host, is_root=root) is expected


def test_context_fraction_is_the_context_usage_number():
    agent = SimpleNamespace(context_stats=SimpleNamespace(overall_utilization=0.42))
    assert context_fraction(agent) == 0.42
    assert context_fraction(SimpleNamespace(context_stats=None)) is None


def test_the_span_carries_reason_turns_minutes_and_context(monkeypatch):
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer", lambda name, *a, **k: provider.get_tracer(name))
    emit_reflection_span("context", turns=7, minutes=12.5, fraction=0.61, session_id="s1")
    [span] = exporter.get_finished_spans()
    assert span.name == "atom.reflection"
    assert dict(span.attributes or {}) == {
        "nooa.atom.reflection.reason": "context",
        "nooa.atom.reflection.turns_since_last": 7,
        "nooa.atom.reflection.minutes_since_last": 12.5,
        "nooa.atom.reflection.context_fraction": 0.61,
        "nooa.atom.session.id": "s1",
    }


def test_workspace_settings_configure_it(tmp_path):
    (tmp_path / ".nooa").mkdir()
    (tmp_path / ".nooa" / "settings.yaml").write_text(
        "atom:\n  reflection:\n    enabled: true\n    every_turns: 3\n    instructions: Tidy.\n"
    )
    settings = AtomOptions.load(tmp_path).reflection
    assert (settings.enabled, settings.every_turns, settings.instructions) == (True, 3, "Tidy.")
    assert settings.min_gap_turns == 5


# ---- sessions ---------------------------------------------------------------


def done(text):
    return cell(f"return_result(Done(explanation={text!r}))")


def slow_done(text, seconds=0.3):
    return cell(f"await asyncio.sleep({seconds})\nreturn_result(Done(explanation={text!r}))")


REFLECT = "REFLECT NOW"


async def make_root(tmp_path, responses, *, host="acp", reflection=None, **settings):
    llm = CellLLM(responses)
    settings = {"every_turns": 2, "min_gap_turns": 1, "instructions": REFLECT, **settings}

    def factory(options, storage):
        agent = create_session_agent(options, storage)
        agent._reflection_settings = ReflectionSettings(**settings)
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
        while root._agent.turns.running or root.has_queued_input() or root._reflection_item:
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.02)
        while root._agent.turns.running:
            await asyncio.sleep(0.005)

    await asyncio.wait_for(wait(), 10)


def turns(updates):
    """Per started turn: (reflection?, the channels of its admitted items)."""
    channels = {u.item_id: u.channel for u in updates if isinstance(u, ItemAdmittedUpdate)}
    return [
        (u.reflection, [channels.get(i) for i in u.item_ids])
        for u in updates
        if isinstance(u, TurnStartedUpdate)
    ]


async def test_a_due_reflection_runs_as_a_quiet_turn_with_the_configured_text(tmp_path):
    registry, root, llm, updates = await make_root(
        tmp_path, [done("one"), done("two"), done("renamed and tidied"), done("three")]
    )
    try:
        assert (await root.prompt("one")).explanation == "one"
        assert (await root.prompt("two")).explanation == "two"
        await idle(root)
        [admitted] = [u for u in updates if getattr(u, "source", None) == "reflection"]
        assert (admitted.channel, admitted.text) == ("slash_commands", REFLECT)
        assert turns(updates) == [
            (False, ["user_messages"]),
            (False, ["user_messages"]),
            (True, ["slash_commands"]),
        ]
        ended = [u for u in updates if isinstance(u, TurnEndedUpdate)]
        assert [u.reflection for u in ended] == [False, False, True]
        assert ended[-1].outcome["explanation"] == "renamed and tidied"
        # The user turns replied with their explanation; the reflection did not.
        events = root._agent.events
        messages = [
            event.content
            for event in (events[tag] for tag in events.keys())
            if isinstance(event, AgentMessage)
        ]
        assert messages == ["one", "two"]
        assert REFLECT in str(llm.calls[2].messages)
        trigger = root._reflection
        assert (trigger.turns, trigger.pending, root._reflection_item) == (0, False, None)
        # The reflection turn did not count: one more turn is not due again yet.
        assert (await root.prompt("three")).explanation == "three"
        await idle(root)
        assert trigger.turns == 1 and len(llm.calls) == 4
    finally:
        await registry.close_all()


async def test_headless_sessions_do_not_reflect_unless_told(tmp_path):
    registry, root, llm, updates = await make_root(
        tmp_path, [done("a"), done("b"), done("c")], host="headless"
    )
    try:
        assert root._reflection is None
        for text in "abc":
            await root.prompt(text)
        await idle(root)
        assert all(not reflected for reflected, _ in turns(updates))
    finally:
        await registry.close_all()
    registry, root, llm, updates = await make_root(
        tmp_path / "on", [done("a"), done("b"), done("tidied")], host="headless", reflection=True
    )
    try:
        await root.prompt("a")
        await root.prompt("b")
        await idle(root)
        assert turns(updates)[-1] == (True, ["slash_commands"])
    finally:
        await registry.close_all()


async def test_explicit_false_turns_it_off_in_an_interactive_host(tmp_path):
    registry, root, _llm, _updates = await make_root(tmp_path, [], reflection=False)
    try:
        assert root._reflection is None
    finally:
        await registry.close_all()


async def test_user_input_that_waits_goes_first(tmp_path):
    registry, root, llm, updates = await make_root(
        tmp_path, [done("one"), slow_done("two"), done("three"), done("tidied")]
    )
    try:
        await root.prompt("one")
        second = asyncio.create_task(root.prompt("two"))
        while not root._agent.turns.running:
            await asyncio.sleep(0.005)
        third = await root.submit("three", channel="user_messages", source="user")
        assert (await second).explanation == "two"
        # Due now, but "three" waits: the reflection stays pending, unqueued.
        assert root._reflection.pending
        assert (await root.outcome(third.item_id)).explanation == "three"
        await idle(root)
        assert turns(updates) == [
            (False, ["user_messages"]),
            (False, ["user_messages"]),
            (False, ["user_messages"]),
            (True, ["slash_commands"]),
        ]
    finally:
        await registry.close_all()


async def test_a_queued_reflection_steps_aside_for_new_input_and_is_never_doubled(tmp_path):
    registry, root, llm, updates = await make_root(
        tmp_path,
        [done("one"), slow_done("two"), done("three and four"), done("tidied")],
    )
    try:
        await root.prompt("one")
        second = asyncio.create_task(root.prompt("two"))
        while not root._agent.turns.running:
            await asyncio.sleep(0.005)
        root.pause_dispatch()
        await second
        channel = root._agent.queue_manager.channels()["slash_commands"]
        assert root._reflection_item is not None and channel.qsize() == 1
        assert not root.has_queued_input()  # a queued reflection is not input
        three = await root.submit("three", channel="user_messages", source="user")
        four = await root.submit("four", channel="user_messages", source="user")
        root.resume_dispatch()
        # Both messages share the next turn; the reflection is not part of it.
        assert (await root.outcome(three.item_id)).explanation == "three and four"
        assert (await root.outcome(four.item_id)).explanation == "three and four"
        await idle(root)
        assert channel.qsize() == 0
        started = turns(updates)
        assert started[-1] == (True, ["slash_commands"])
        assert sum(reflected for reflected, _ in started) == 1
        assert started[2] == (False, ["user_messages", "user_messages"])
        assert len(started) == 4
    finally:
        await registry.close_all()


async def test_minutes_and_context_trigger_in_a_session(tmp_path, monkeypatch):
    from nooa_atom.session import session as session_module

    clock = Clock()
    registry, root, llm, updates = await make_root(
        tmp_path,
        [done("a"), done("b"), done("tidied"), done("c"), done("tidied again")],
        every_turns=None,
    )
    spans = []
    monkeypatch.setattr(
        session_module, "emit_reflection_span", lambda reason, **kw: spans.append((reason, kw))
    )
    try:
        root._reflection = ReflectionTrigger(root._reflection.settings, clock=clock)
        await root.prompt("a")
        clock.now = 3600
        await root.prompt("b")
        await idle(root)
        monkeypatch.setattr(session_module, "context_fraction", lambda agent: 0.75)
        await root.prompt("c")
        await idle(root)
        assert [reason for reason, _ in spans] == ["minutes", "context"]
        assert spans[0][1]["turns"] == 2 and spans[0][1]["minutes"] == 60.0
        assert spans[1][1] == {"turns": 1, "minutes": 0.0, "fraction": 0.75, "session_id": root.id}
        assert sum(reflected for reflected, _ in turns(updates)) == 2
    finally:
        await registry.close_all()


async def test_a_reflection_queued_before_a_restart_is_not_replayed(tmp_path):
    registry, root, _llm, _updates = await make_root(tmp_path, [done("one"), slow_done("two")])
    try:
        await root.prompt("one")
        second = asyncio.create_task(root.prompt("two"))
        while not root._agent.turns.running:
            await asyncio.sleep(0.005)
        root.pause_dispatch()
        await second
        item_id = root._reflection_item
        assert item_id is not None
        receipt = root.requeue(REFLECT, channel="slash_commands", source="reflection", item_id="x")
        assert receipt.item_id == "x"
        assert root._agent.queue_manager.channels()["slash_commands"].qsize() == 1
    finally:
        await registry.close_all()
