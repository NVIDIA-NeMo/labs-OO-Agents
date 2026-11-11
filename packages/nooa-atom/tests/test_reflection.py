# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The reflection skill: timing, blocks, /reflection and the session hook."""

import asyncio
import logging
from types import SimpleNamespace

import pytest
from atom_test_agents import CellLLM, cell
from nooa_atom.agent.factory import create_session_agent
from nooa_atom.session.items import ItemAdmittedUpdate, TurnStartedUpdate
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore
from nooa_atom.skills import reflection as module
from nooa_atom.skills.reflection import CORE_BLOCKS, FOOTER, HEADER, Reflection, ReflectionConfig
from nooa_atom.tools.todo_list import TodoManager
from nooa_atom.workspace.options import AtomOptions
from nooa_atom.workspace.settings import settings_path

from nooa.context_blocks.models import ContextWindowStats


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def fake_agent(stats=None, names=0, todos=(), keys=0):
    todo = TodoManager()
    for title in todos:
        todo.add(title)
    namespace = SimpleNamespace(
        user_names=[f"x{i}" for i in range(names)], globals={f"x{i}": i for i in range(names)}
    )
    return SimpleNamespace(
        context_stats=stats,
        _handle_namespace=namespace,
        todo=todo,
        vars={f"k{i}": i for i in range(keys)},
    )


def skill(agent=None, clock=None, host="acp", override=None, **config):
    reflection = Reflection(
        ReflectionConfig(**config), host=host, override=override, clock=clock or Clock()
    )
    reflection.blocks["say"] = "Say it."
    reflection.attach(agent or fake_agent())
    return reflection


def run(reflection, kinds, waiting=False):
    """Settle turns of these kinds; return what was posted."""
    posted = []
    for kind in kinds:
        reflection.after_turn(kind, waiting=waiting, post=posted.append)
    return posted


# ---- timing -------------------------------------------------------------------


def test_defaults():
    assert ReflectionConfig().model_dump() == {
        "enabled": None,
        "every_turns": 20,
        "every_minutes": 60.0,
        "min_gap_turns": 5,
        "context_fraction": 0.5,
        "blocks": {},
    }


def test_triggers_and_the_minimum_gap():
    assert len(run(skill(every_turns=3, min_gap_turns=0), ["done"] * 3)) == 1
    clock = Clock()
    timed = skill(clock=clock, every_turns=None, min_gap_turns=2)
    clock.now = 3600
    assert run(timed, ["done"]) == [] and len(run(timed, ["waiting"])) == 1
    stats = ContextWindowStats(
        context_blocks_count=1, events_count=1, prompt_tokens=60, model_context_window=100
    )
    full = skill(fake_agent(stats), every_turns=None, every_minutes=None, min_gap_turns=2)
    assert run(full, ["done"]) == [] and len(run(full, ["done"])) == 1
    # Only Done and Waiting count; a trigger set to None is off.
    assert run(skill(every_turns=1, min_gap_turns=0), ["need_input", "error", "cancelled"]) == []
    assert (
        run(skill(every_turns=None, every_minutes=None, context_fraction=None), ["done"] * 30) == []
    )


def test_while_off_nothing_counts():
    reflection = skill(host="headless", every_turns=2, min_gap_turns=0)
    assert run(reflection, ["done"] * 5) == [] and reflection.turns == 0
    reflection.config.enabled = True
    assert run(reflection, ["done"]) == [] and len(run(reflection, ["done"])) == 1


@pytest.mark.parametrize(
    ("host", "enabled", "override", "expected"),
    [
        ("acp", None, None, True),
        ("headless", None, None, False),
        ("acp", False, None, False),
        ("headless", True, None, True),
        ("acp", True, False, False),
        ("headless", False, True, True),
    ],
)
def test_enabled_by_host_setting_and_session_override(host, enabled, override, expected):
    assert skill(host=host, enabled=enabled, override=override).enabled() is expected


def test_a_the_turn_answering_a_reflection_does_not_count():
    reflection = skill(every_turns=1, min_gap_turns=0)
    assert len(run(reflection, ["done"])) == 1
    assert run(reflection, ["done"]) == [] and reflection.turns == 0  # the reflection's turn
    assert len(run(reflection, ["done"])) == 1  # the next real turn


def test_c_a_due_reflection_waits_while_a_prompt_is_waiting():
    reflection = skill(every_turns=1, min_gap_turns=0)
    assert run(reflection, ["waiting", "waiting"], waiting=True) == []
    assert len(run(reflection, ["done"])) == 1


# ---- blocks -------------------------------------------------------------------


def test_the_core_blocks_report_the_agent_state():
    stats = ContextWindowStats(
        context_blocks_count=12,
        events_count=752,
        prompt_tokens=364_414,
        context_blocks_chars=1_000,
        events_chars=19_000,
        model_context_window=1_000_000,
        reserved_output_tokens=65_536,
    )
    reflection = Reflection()
    reflection.blocks.update(CORE_BLOCKS)
    reflection.attach(fake_agent(stats, names=12, todos=["a", "b"], keys=3))
    text = reflection.message()
    print(text)
    lines = text.splitlines()
    assert (lines[0], lines[-1], len(lines)) == (HEADER, FOOTER, 7)
    assert lines[1].startswith("1. Rename this session")
    assert lines[2].startswith("2. Context is 39% full (364,414 / 934,464 usable tokens): ")
    assert "events ~346,193 tokens in 752 events" in lines[2]
    assert "context blocks ~18,221 in 12 blocks" in lines[2]
    assert "self.events.collapse(" in lines[2] and "standing rules" in lines[2]
    assert lines[3].startswith("3. 12 Python variables are retained")
    assert lines[4].startswith("4. 2 open todos in `self.todo`, the oldest opened ")
    assert lines[5] == "5. `self.v` holds 3 keys: delete those for finished work."
    # Nothing to report: the dynamic blocks step aside; compact keeps its action.
    reflection.attach(fake_agent())
    assert reflection.message().splitlines()[1:-1] == [
        "1. " + CORE_BLOCKS["rename"],
        "2. " + module.compact_block(fake_agent()),
    ]


def test_b_a_failing_block_is_one_line_and_the_rest_go_out(caplog):
    async def later(agent):
        return "never"

    reflection = skill()
    reflection.blocks.update(
        {
            "boom": lambda agent: 1 / 0,
            "async": later,
            "number": lambda agent: 3,
            "none": lambda agent: None,
        }
    )
    reflection.config.blocks["say"] = True
    with caplog.at_level(logging.ERROR, logger=module.__name__):
        lines = reflection.message().splitlines()[1:-1]
    assert lines == [
        "1. Say it.",
        "2. (Block 'boom' failed: ZeroDivisionError: division by zero)",
        "3. (Block 'async' failed: TypeError: blocks must be sync, got a coroutine)",
        "4. (Block 'number' failed: TypeError: returned int, not str)",
    ]
    assert "Traceback" in caplog.text and "ZeroDivisionError" in caplog.text
    reflection.config.blocks = dict.fromkeys(reflection.blocks, False)
    assert reflection.message() is None and run(skill(), []) == []


# ---- /reflection --------------------------------------------------------------


async def test_reflection_command_and_persistence(tmp_path):
    reflection = skill(host="headless")
    reflection.workspace = tmp_path

    async def command(args=""):
        return await reflection.reflection_command(args)

    status = await command()
    assert status.startswith("Reflection: off\nBlocks on: say\n")
    assert "Triggers: every 20 turns, every 60 minutes, at 50% context; at least 5" in status
    assert "Reflection is on (saved to" in await command("on")
    assert reflection.enabled()
    assert "[on ] say  Say it." in await command("config")
    assert "Reflection block say is off" in await command("config say off")
    assert "[off] say" in await command("config")
    assert "No reflection block 'nope'" in await command("config nope on")
    assert (await command("sideways")) == module.USAGE
    saved = AtomOptions.load(tmp_path).reflection
    assert (saved.enabled, saved.blocks) == (True, {"say": False})
    assert "every_turns" not in settings_path("project", workspace=tmp_path).read_text()


async def test_d_a_failed_save_changes_nothing(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("read-only")

    monkeypatch.setattr("nooa_atom.workspace.settings.write_settings_updates", fail)
    reflection = skill(host="headless")
    assert "Could not save" in await reflection.reflection_command("off")
    assert "Could not save" in await reflection.reflection_command("config say off")
    assert reflection.config == ReflectionConfig()  # untouched


async def test_e_on_off_clears_the_session_override(tmp_path):
    reflection = skill(override=True)
    reflection.workspace = tmp_path
    reply = await reflection.reflection_command("off")
    assert "This session's override (reflection=True) is cleared." in reply
    assert not reflection.enabled() and reflection.override is None


# ---- sessions -----------------------------------------------------------------


def done(text):
    return cell(f"return_result(Done(explanation={text!r}))")


async def make_root(tmp_path, responses, *, host="acp", **config):
    def factory(options, storage):
        agent = create_session_agent(options, storage)
        agent.reflection.config = ReflectionConfig(**{"min_gap_turns": 0, **config})
        return agent

    registry = SessionRegistry(SessionStore(tmp_path / "sessions"), agent_factory=factory)
    llm = CellLLM(responses)
    root = await registry.create(
        SessionOptions(
            workspace=tmp_path, agent_spec="nooa_atom.agent:AtomAgent", llm=llm, host=host
        )
    )
    updates = []
    root.subscribe(updates.append)
    return registry, root, llm, updates


async def idle(root):
    async def wait():
        for _ in range(2):
            while root._agent.turns.running or root.has_queued_input():
                await asyncio.sleep(0.005)
            await asyncio.sleep(0.02)

    await asyncio.wait_for(wait(), 10)


def turn_channels(updates):
    channels = {u.item_id: u.channel for u in updates if isinstance(u, ItemAdmittedUpdate)}
    return [
        [channels.get(i) for i in u.item_ids] for u in updates if isinstance(u, TurnStartedUpdate)
    ]


async def test_the_factory_activates_the_skill_with_the_core_blocks(tmp_path):
    registry, root, _llm, _updates = await make_root(tmp_path, [])
    try:
        agent = root._agent
        assert isinstance(agent.reflection, Reflection)
        assert list(agent.reflection.blocks) == ["rename", "compact", "variables", "todo", "vars"]
        assert "nemo.reflection" not in agent.skills.activated()  # not in the model's prompt
        assert "reflection" in {c.name for c in root.commands()}
        reply = (await root.invoke_command("reflection", "")).text
        assert reply.startswith("Reflection: on\n")
    finally:
        await registry.close_all()


async def test_a_due_reflection_is_an_ordinary_item_and_an_idle_session_stays_idle(tmp_path):
    registry, root, llm, updates = await make_root(
        tmp_path,
        [
            cell(
                "=".join(f"v{i}" for i in range(12)) + "=0\nreturn_result(Done(explanation='one'))"
            ),
            done("tidied"),
            done("two"),
            done("tidied"),
        ],
        every_turns=1,
    )
    try:
        assert (await root.prompt("one")).explanation == "one"
        await idle(root)
        [item] = [u for u in updates if getattr(u, "source", None) == "reflection"]
        assert item.channel == "system_messages" and item.text.startswith(HEADER)
        assert "3. 12 Python variables are retained" in item.text  # the live namespace
        assert turn_channels(updates) == [["user_messages"], ["system_messages"]]
        assert len(llm.calls) == 2  # (a): the reflection's own turn triggers nothing
        assert (await root.prompt("two")).explanation == "two"
        await idle(root)
        assert len(llm.calls) == 4
    finally:
        await registry.close_all()


async def test_c_a_waiting_prompt_gets_its_own_result_before_any_reflection(tmp_path):
    registry, root, llm, updates = await make_root(
        tmp_path,
        [
            cell(
                "self.jobs.launch(asyncio.sleep(0.05), name='job')\n"
                "return_result(Waiting(explanation='job running', on=['job']))"
            ),
            done("job finished"),
            done("tidied"),
        ],
        every_turns=1,
    )
    try:
        assert (await root.prompt("start the job")).explanation == "job finished"
        await idle(root)
        assert turn_channels(updates) == [
            ["user_messages"],
            [],
            ["system_messages"],
        ]  # [] is the job
        assert len(llm.calls) == 3
    finally:
        await registry.close_all()


async def test_headless_and_child_sessions_do_not_reflect(tmp_path):
    registry, root, llm, updates = await make_root(
        tmp_path, [done("a"), done("b")], host="headless", every_turns=1
    )
    try:
        for text in ("a", "b"):
            await root.prompt(text)
        await idle(root)
        assert len(llm.calls) == 2
        assert not [u for u in updates if getattr(u, "source", None) == "reflection"]
    finally:
        await registry.close_all()
