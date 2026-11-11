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
    CORE_BLOCKS,
    FOOTER,
    HEADER,
    Reflection,
    ReflectionBlock,
    ReflectionBlocks,
    ReflectionSettings,
    context_fraction,
    has_queue,
    reflection_enabled,
    runtime_evaluator,
)
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore
from nooa_atom.workspace.options import AtomOptions
from nooa_atom.workspace.settings import settings_path


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


# ---- the hysteresis -----------------------------------------------------------


async def no_snippets(expr):
    raise AssertionError(f"unexpected snippet {expr!r}")


def core_blocks():
    """The blocks the Atom factory registers: the core ones, switched on."""
    blocks = ReflectionBlocks()
    for name, text in CORE_BLOCKS.items():
        blocks.register(name, text=text, on=True)
    return blocks


def _reflection(clock=None, posted=None, context=lambda: None, **values):
    return Reflection(
        ReflectionSettings(**values),
        blocks=core_blocks(),
        post=(posted if posted is not None else []).append,
        evaluate=no_snippets,
        context=context,
        clock=clock or Clock(),
    )


def core_message():
    return "\n".join(
        [HEADER, *(f"{i}. {text}" for i, text in enumerate(CORE_BLOCKS.values(), 1)), FOOTER]
    )


def _ends(reflection, count, kind="done", fraction=None):
    return [reflection.due(kind, context_fraction=fraction) for _ in range(count)]


def test_defaults():
    settings = ReflectionSettings()
    assert settings.enabled is None
    assert (settings.every_turns, settings.every_minutes) == (20, 60.0)
    assert (settings.min_gap_turns, settings.context_fraction) == (5, 0.5)
    assert settings.blocks == {}
    assert AtomOptions().reflection == settings
    with pytest.raises(ValueError, match="extra"):
        ReflectionSettings(children=True)  # root only: there is no setting for children
    with pytest.raises(ValueError, match="extra"):
        ReflectionSettings(bullets={})  # replaced by blocks


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


async def test_a_due_reflection_is_pending_until_sent_and_the_gap_applies_again():
    clock, posted, fraction = Clock(), [], [None]
    reflection = _reflection(
        clock, posted, context=lambda: fraction[0], every_turns=3, min_gap_turns=2
    )
    assert [reflection.turn_ended("done") for _ in range(3)] == [None, None, "turns"]
    assert reflection.pending == "turns" and posted == []
    assert (reflection.turns, reflection.minutes()) == (0, 0)
    assert await reflection.send() == core_message()
    assert posted == [core_message()] and reflection.pending is None
    assert await reflection.send() is None  # nothing pending: nothing sent
    clock.now = 3600 * 5  # long overdue by time, and the context is full
    fraction[0] = 0.9
    assert reflection.turn_ended("done") is None  # 1 turn < min gap 2
    assert reflection.turn_ended("done") == "minutes"
    await reflection.send()
    assert len(posted) == 2 and reflection.turns == 0


async def test_the_context_callable_feeds_the_context_trigger():
    posted = []
    reflection = _reflection(posted=posted, context=lambda: 0.8, min_gap_turns=1)
    assert reflection.turn_ended("waiting") == "context"
    await reflection.send()
    assert len(posted) == 1


def test_while_off_nothing_counts():
    on = [False]
    reflection = _reflection(every_turns=2, min_gap_turns=0)
    reflection.enabled = lambda: on[0]
    assert [reflection.turn_ended("done") for _ in range(5)] == [None] * 5
    assert reflection.turns == 0 and reflection.pending is None
    on[0] = True  # /reflection on: a fresh window
    assert [reflection.turn_ended("done") for _ in range(2)] == [None, "turns"]


async def test_a_failing_send_is_logged_and_ignored(caplog):
    def boom(text):
        raise RuntimeError("boom")

    reflection = Reflection(
        ReflectionSettings(every_turns=1, min_gap_turns=0),
        blocks=core_blocks(),
        post=boom,
        evaluate=no_snippets,
        clock=Clock(),
    )
    assert reflection.turn_ended("done") == "turns"
    with caplog.at_level(logging.ERROR, logger="nooa_atom.session.reflection"):
        assert await reflection.send() is None
    assert "Sending the reflection failed" in caplog.text and reflection.pending is None


# ---- the message and its blocks ---------------------------------------------------


async def test_the_default_message():
    message = await core_blocks().message(ReflectionSettings(), no_snippets)
    lines = message.split("\n")
    assert message == core_message()
    assert HEADER.startswith("[reflection] A housekeeping message from the session")
    assert "not a request from the user" in HEADER
    assert lines[-1] == FOOTER == "Then finish with a short Done."
    assert list(CORE_BLOCKS) == ["rename", "compact", "variables", "todo", "vars"]
    for step in ("rename_session", "self.events.collapse", "del", "self.v", "self.todo"):
        assert step in message
    for block in ("context_usage", "python_cell_state", "todo_status"):
        assert block in message  # points at what the agent already sees
    assert "add new ones for work you've taken on" in CORE_BLOCKS["todo"]


def test_a_bare_registry_is_empty():
    assert ReflectionBlocks().blocks == {}


def test_register_text_and_snippet_blocks_by_name():
    blocks = core_blocks()
    blocks.register("memory", text=" Save what you learned. ")
    blocks.register("notes", expr="self.notes.reflection()")
    blocks.register("rename", text="Rename it.")  # replaces in place
    blocks.unregister("compact")
    blocks.unregister("unknown")  # ignored
    assert list(blocks.blocks) == ["rename", "variables", "todo", "vars", "memory", "notes"]
    assert blocks.blocks["memory"] == ReflectionBlock("memory", text="Save what you learned.")
    assert blocks.blocks["notes"] == ReflectionBlock("notes", expr="self.notes.reflection()")
    assert blocks.blocks["rename"] == ReflectionBlock("rename", text="Rename it.")  # off now
    blocks.register("awaited", expr="await self.notes.reflection()")  # as a context block's expr
    blocks.unregister("notes")
    assert "notes" not in blocks.blocks


@pytest.mark.parametrize(
    "kwargs",
    [
        {},  # neither
        {"text": "a", "expr": "b"},  # both
        {"text": "  "},
        {"expr": "self.notes.reflection("},  # does not parse
        {"expr": "x = 1"},  # a statement, not an expression
    ],
)
def test_a_block_needs_text_or_a_valid_snippet(kwargs):
    blocks = ReflectionBlocks()
    with pytest.raises(ValueError):
        blocks.register("bad", **kwargs)
    assert "bad" not in blocks.blocks


async def test_snippets_are_evaluated_when_the_message_is_built():
    values = {
        "a()": "FROM A",
        "none()": None,
        "empty()": "",
        "boom()": ZeroDivisionError("division by zero"),
        "items()": [1, 2],
    }
    seen = []

    async def evaluate(expr):
        seen.append(expr)
        value = values[expr]
        if isinstance(value, Exception):
            raise value
        return value

    blocks = ReflectionBlocks()
    blocks.register("first", text="Fixed text.")
    for expr in values:
        blocks.register(expr, expr=expr)
    settings = ReflectionSettings(blocks=dict.fromkeys(["first", *values], True))
    assert seen == []  # registering evaluates nothing
    message = await blocks.message(settings, evaluate)
    assert seen == list(values)
    assert message.split("\n")[1:-1] == [
        "1. Fixed text.",
        "2. FROM A",  # None and "" leave their blocks out
        "3. ZeroDivisionError: division by zero",  # inline, as a context block shows it
        "4. [1, 2]",
    ]


async def test_only_switched_on_blocks_are_sent():
    blocks = core_blocks()
    blocks.register("memory", text="SAVE MEMORY")
    blocks.register("notes", expr="self.notes()")

    async def notes(expr):
        assert expr == "self.notes()"
        return "NOTES"

    # Blocks registered with on=True start on, others off; the settings switch either way.
    assert ReflectionSettings().block_on("memory", blocks.blocks["memory"].on) is False
    settings = ReflectionSettings(blocks={"rename": False, "memory": True})
    assert [settings.block_on(name, block.on) for name, block in blocks.blocks.items()] == [
        False,
        True,
        True,
        True,
        True,
        True,
        False,
    ]
    message = await blocks.message(settings, no_snippets)
    assert "rename_session" not in message and "NOTES" not in message
    assert message.split("\n")[-2] == "5. SAVE MEMORY"
    settings.blocks["notes"] = True  # switched on: the snippet is evaluated and sent
    assert (await blocks.message(settings, notes)).split("\n")[-2] == "6. NOTES"
    nothing = ReflectionSettings(blocks=dict.fromkeys(blocks.blocks, False))
    assert await blocks.message(nothing, no_snippets) is None


async def test_nothing_to_say_posts_nothing():
    posted = []
    reflection = _reflection(posted=posted, every_turns=1, min_gap_turns=0)
    reflection.settings.blocks = dict.fromkeys(CORE_BLOCKS, False)
    assert reflection.turn_ended("done") == "turns"
    assert await reflection.send() is None and posted == []


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
        "    blocks:\n      rename: false\n      notes: true\n"
    )
    settings = AtomOptions.load(tmp_path).reflection
    assert (settings.enabled, settings.every_turns) == (True, 3)
    assert settings.blocks == {"rename": False, "notes": True}
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
        tmp_path,
        [done("one"), done("two"), done("tidied"), done("three"), done("tidied again")],
        blocks={"skill": True},
    )
    try:
        assert root._reflection.blocks is root._agent.reflection  # the agent's blocks
        assert root._agent._reflection_timer is root._reflection  # what /reflection reads
        root._agent.reflection.register("skill", text=REFLECT)
        root._agent.reflection.register("off", text="NOT SWITCHED ON")
        assert (await root.prompt("one")).explanation == "one"
        assert (await root.prompt("two")).explanation == "two"
        await idle(root)
        [admitted] = reflections(updates)
        assert admitted.channel == "system_messages"
        assert admitted.text.startswith(core_message().rsplit("\n", 1)[0])
        assert admitted.text.split("\n")[-2] == f"6. {REFLECT}"  # the skill's block, last
        assert "NOT SWITCHED ON" not in admitted.text
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
        assert not root._reflection.enabled()
        assert isinstance(root._agent.reflection, ReflectionBlocks)  # kept, never posted
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
        assert not root._reflection.enabled()
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
        assert isinstance(child._agent.reflection, ReflectionBlocks)
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


_SKILL_AGENT = """\\
from nooa.skill import Skill
from nooa_atom.agent.agent import AtomAgent


class Memory(Skill):
    '''Registers reflection blocks while it is attached.'''

    def attach(self, agent):
        super().attach(agent)
        agent.reflection.register("memory", text="SAVE MEMORY")
        agent.reflection.register("memo", expr="self.memory_note()")
        agent.reflection.register("broken", expr="1 / 0")
        agent.reflection.register("silent", expr="None")

    def detach(self):
        for name in ("memory", "memo", "broken", "silent"):
            self._agent.reflection.unregister(name)
        super().detach()


class SkillAgent(AtomAgent):
    def __init__(self, llm=None, storage=None, **kwargs):
        super().__init__(llm=llm, storage=storage, **kwargs)
        self.notes = 0
        self.skills.register("test.memory", Memory())

    def memory_note(self):
        self.notes += 1
        return f"NOTE {self.notes} FROM {type(self).__name__}"
"""


async def test_a_skill_registers_blocks_and_their_snippets_run_when_sent(tmp_path):
    (tmp_path / "skill_agent.py").write_text(_SKILL_AGENT)
    (tmp_path / ".nooa").mkdir()
    (tmp_path / ".nooa" / "settings.yaml").write_text(
        "atom:\n  active_skills: [test.memory]\n"
        "  reflection:\n    every_turns: 2\n    min_gap_turns: 0\n"
        "    blocks: {rename: false, memory: true, memo: true, broken: true, silent: true}\n"
    )
    llm = CellLLM([done("one"), done("two"), done("tidied")])
    registry = SessionRegistry(
        SessionStore(tmp_path / "sessions"), agent_factory=create_session_agent
    )
    root = await registry.create(
        SessionOptions(
            workspace=tmp_path, agent_spec="./skill_agent.py:SkillAgent", llm=llm, host="acp"
        )
    )
    updates = []
    root.subscribe(updates.append)
    agent = root._agent
    try:
        assert "test.memory" in agent.skills.activated()
        # The skill attached while the agent was built, before the factory
        # registered the core blocks: its blocks come first.
        assert list(agent.reflection.blocks) == [
            "memory",
            "memo",
            "broken",
            "silent",
            *CORE_BLOCKS,
        ]
        assert agent.notes == 0  # registered, not evaluated
        assert (await root.prompt("one")).explanation == "one"
        assert (await root.prompt("two")).explanation == "two"
        await idle(root)
        [admitted] = reflections(updates)
        assert agent.notes == 1  # evaluated once, right before posting
        assert admitted.text.split("\n")[1:-1] == [
            "1. SAVE MEMORY",
            "2. NOTE 1 FROM SkillAgent",  # the context blocks' evaluator: self is the agent
            "3. ZeroDivisionError: division by zero",
            f"4. {CORE_BLOCKS['compact']}",
            f"5. {CORE_BLOCKS['variables']}",
            f"6. {CORE_BLOCKS['todo']}",
            f"7. {CORE_BLOCKS['vars']}",
        ]  # "silent" returned None: left out
        assert "NOTE 1 FROM SkillAgent" in str(llm.calls[2].messages)  # the agent got the text
    finally:
        await registry.close_all()
    assert "memo" not in agent.reflection.blocks  # closing detached the skill


async def test_the_atom_factory_registers_exactly_the_core_blocks_switched_on(tmp_path):
    registry, root, _llm, _updates = await make_root(tmp_path, [])
    try:
        assert root._agent.reflection.blocks == {
            name: ReflectionBlock(name, text=text, on=True) for name, text in CORE_BLOCKS.items()
        }
    finally:
        await registry.close_all()


async def test_the_runtime_evaluator_is_the_context_block_one(tmp_path):
    registry, root, _llm, _updates = await make_root(tmp_path, [])
    try:
        evaluate = runtime_evaluator(root._agent)
        assert await evaluate("self.cwd") == root._agent.cwd
        with pytest.raises(ZeroDivisionError):
            await evaluate("1 / 0")
    finally:
        await registry.close_all()


# ---- /reflection ---------------------------------------------------------------


async def test_reflection_command_on_off_config_and_persistence(tmp_path):
    registry, root, _llm, _updates = await make_root(tmp_path, [], host="headless")

    async def command(args=""):
        return (await root.invoke_command("reflection", args)).text

    path = settings_path("project", workspace=tmp_path)
    try:
        agent = root._agent
        agent.reflection.register("memory", expr="self.memory_note()")
        assert "reflection" in {c.name for c in root.commands()}
        status = await command()
        assert status.startswith(
            "Reflection: off\nBlocks on: rename, compact, variables, todo, vars"
        )
        assert "Triggers: every 2 turns, every 60 minutes, at 50% context" in status
        assert not root._reflection.enabled()

        assert "Reflection is on" in await command("on")
        assert root._reflection.enabled()  # at once, in this session
        assert (await command()).splitlines()[0] == "Reflection: on"
        assert "Next: in 2 turns, or in 60 minutes, or at 50% context" in await command()

        config = await command("config")
        assert "[on ] rename" in config and "[off] memory" in config
        assert "snippet: self.memory_note()" in config
        assert "Reflection block memory is on" in await command("config memory on")
        assert "Reflection block rename is off" in await command("config rename off")
        assert "[on ] memory" in await command("config")
        assert "No reflection block 'nope'" in await command("config nope on")
        assert "Usage: /reflection" in await command("sideways")

        saved = AtomOptions.load(tmp_path).reflection
        assert (saved.enabled, saved.blocks) == (True, {"memory": True, "rename": False})
        assert "every_turns" not in path.read_text()  # triggers stay YAML-only

        assert "Reflection is off" in await command("off")
        assert not root._reflection.enabled()
        assert AtomOptions.load(tmp_path).reflection.enabled is False
    finally:
        await registry.close_all()
