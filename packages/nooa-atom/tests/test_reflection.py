# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reflection: when the session sends it, what it says, where it goes, and who gets it."""

import asyncio
import logging
from types import SimpleNamespace

import pytest
from atom_test_agents import CellLLM, cell
from nooa_atom.agent.factory import create_session_agent
from nooa_atom.session.items import ItemAdmittedUpdate, TurnStartedUpdate
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.reflection import (
    DEFAULT_BULLETS,
    FOOTER,
    HEADER,
    Reflection,
    ReflectionSettings,
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


# ---- the hysteresis -----------------------------------------------------------


def _reflection(clock=None, posted=None, context=lambda: None, **values):
    return Reflection(
        ReflectionSettings(**values),
        post=(posted if posted is not None else []).append,
        context=context,
        clock=clock or Clock(),
    )


def _ends(reflection, count, kind="done", fraction=None):
    return [reflection.due(kind, context_fraction=fraction) for _ in range(count)]


def test_defaults():
    settings = ReflectionSettings()
    assert settings.enabled is None
    assert (settings.every_turns, settings.every_minutes) == (20, 60.0)
    assert (settings.min_gap_turns, settings.context_fraction) == (5, 0.5)
    assert settings.bullets == {}
    assert AtomOptions().reflection == settings
    with pytest.raises(ValueError, match="extra"):
        ReflectionSettings(children=True)  # root only: there is no setting for children
    with pytest.raises(ValueError, match="extra"):
        ReflectionSettings(instructions="Tidy.")  # replaced by bullets


def test_turns_trigger():
    assert _ends(_reflection(), 20) == [None] * 19 + ["turns"]


def test_minutes_trigger_needs_the_minimum_gap():
    clock = Clock()
    reflection = _reflection(clock)
    clock.now = 3600
    assert _ends(reflection, 5) == [None] * 4 + ["minutes"]


def test_context_trigger_needs_the_minimum_gap():
    assert _ends(_reflection(), 5, "waiting", 0.7) == [None] * 4 + ["context"]
    assert _ends(_reflection(), 6, "done", 0.49) == [None] * 6


def test_only_done_and_waiting_turns_count():
    reflection = _reflection(every_turns=2, min_gap_turns=0)
    for kind in ("need_input", "cancelled", "error"):
        assert reflection.due(kind, context_fraction=1.0) is None
    assert reflection.turns == 0
    assert _ends(reflection, 2) == [None, "turns"]


def test_a_trigger_set_to_none_is_off():
    clock = Clock()
    reflection = _reflection(clock, every_turns=None, every_minutes=None, context_fraction=None)
    clock.now = 10**6
    assert _ends(reflection, 50, fraction=1.0) == [None] * 50


def test_posting_resets_the_counters_and_the_gap_applies_again():
    clock, posted, fraction = Clock(), [], [None]
    reflection = _reflection(
        clock, posted, context=lambda: fraction[0], every_turns=3, min_gap_turns=2
    )
    assert [reflection.turn_ended("done") for _ in range(3)] == [None, None, "turns"]
    assert posted == [reflection.message()]
    assert (reflection.turns, reflection.minutes()) == (0, 0)
    clock.now = 3600 * 5  # long overdue by time, and the context is full
    fraction[0] = 0.9
    assert reflection.turn_ended("done") is None  # 1 turn < min gap 2
    assert reflection.turn_ended("done") == "minutes"
    assert len(posted) == 2 and reflection.turns == 0


def test_the_context_callable_feeds_the_context_trigger():
    posted = []
    reflection = _reflection(posted=posted, context=lambda: 0.8, min_gap_turns=1)
    assert reflection.turn_ended("waiting") == "context"
    assert len(posted) == 1


def test_a_failing_check_is_logged_and_ignored(caplog):
    def boom(text):
        raise RuntimeError("boom")

    reflection = Reflection(
        ReflectionSettings(every_turns=1, min_gap_turns=0), post=boom, clock=Clock()
    )
    with caplog.at_level(logging.ERROR, logger="nooa_atom.session.reflection"):
        assert reflection.turn_ended("done") is None
    assert "reflection check failed" in caplog.text


# ---- the message and its bullets ---------------------------------------------------


def test_the_default_message():
    message = Reflection(ReflectionSettings(), post=print).message()
    lines = message.split("\n")
    assert lines[0] == HEADER
    assert HEADER.startswith("[reflection] A housekeeping message from the session")
    assert "not a request from the user" in HEADER
    assert lines[-1] == FOOTER == "Then finish with a short Done."
    assert list(DEFAULT_BULLETS) == ["rename", "compact", "variables", "todo", "vars"]
    assert lines[1:-1] == [f"{i}. {text}" for i, text in enumerate(DEFAULT_BULLETS.values(), 1)]
    for step in ("rename_session", "self.events.collapse", "del", "self.v", "self.todo"):
        assert step in message
    assert "add new ones for work you've taken on" in DEFAULT_BULLETS["todo"]
    assert "relaunch" not in message and "Cancel jobs" not in message  # old bullet 5 is gone


def test_add_and_remove_bullets_by_name():
    reflection = Reflection(ReflectionSettings(), post=print)
    reflection.add("memory", "Save what you learned.")
    reflection.remove("compact")
    reflection.remove("unknown")  # ignored
    reflection.add("rename", "Rename it.")  # replaces in place
    assert list(reflection.bullets) == ["rename", "variables", "todo", "vars", "memory"]
    lines = reflection.message().split("\n")
    assert lines[1] == "1. Rename it."
    assert lines[-2] == "5. Save what you learned."
    reflection.remove("memory")
    assert "memory" not in reflection.bullets
    with pytest.raises(ValueError):
        reflection.add("empty", "  ")


def test_settings_add_replace_and_remove_bullets():
    settings = ReflectionSettings(bullets={"todo": "Tidy todos.", "vars": "", "notes": "Notes."})
    reflection = Reflection(settings, post=print)
    assert list(reflection.bullets) == ["rename", "compact", "variables", "todo", "notes"]
    assert reflection.bullets["todo"] == "Tidy todos."
    assert reflection.message().split("\n")[-2] == "5. Notes."


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
        "atom:\n  reflection:\n    enabled: true\n    every_turns: 3\n"
        "    bullets:\n      rename: ''\n      notes: Tidy notes.\n"
    )
    settings = AtomOptions.load(tmp_path).reflection
    assert (settings.enabled, settings.every_turns) == (True, 3)
    assert settings.bullets == {"rename": "", "notes": "Tidy notes."}
    assert settings.min_gap_turns == 5


# ---- sessions ---------------------------------------------------------------


def done(text):
    return cell(f"return_result(Done(explanation={text!r}))")


REFLECT = "REFLECT NOW"


async def make_root(tmp_path, responses, *, host="acp", reflection=None, edit=None, **settings):
    llm = CellLLM(responses)
    settings = {"every_turns": 2, "min_gap_turns": 1, **settings}

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
        assert root._agent.reflection is root._reflection  # how skills reach it
        root._agent.reflection.add("skill", REFLECT)
        assert (await root.prompt("one")).explanation == "one"
        assert (await root.prompt("two")).explanation == "two"
        await idle(root)
        [admitted] = reflections(updates)
        assert (admitted.channel, admitted.text) == ("system_messages", root._reflection.message())
        assert admitted.text.split("\n")[-2] == f"6. {REFLECT}"  # the skill's bullet, last
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
        assert root._reflection is None and root._agent.reflection is None
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
        assert child._reflection is None and child._agent.reflection is None
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

    root._reflection.due = boom
    try:
        with caplog.at_level(logging.ERROR, logger="nooa_atom.session.reflection"):
            assert (await root.prompt("one")).explanation == "one"
            assert (await root.prompt("two")).explanation == "two"
        await idle(root)
        assert reflections(updates) == []
        assert root.last_outcome_kind == "done" and root.info.status == "idle"
        assert sum("reflection check failed" in r.getMessage() for r in caplog.records) == 2
    finally:
        await registry.close_all()
