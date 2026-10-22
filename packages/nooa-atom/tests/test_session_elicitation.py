# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session.set_asker: forms reach a root session's cells and commands, never a child's."""

import pytest
from atom_test_agents import cell

from nooa import elicit
from nooa.elicit import Elicited

ASK_CELL = """
try:
    answer = await elicit.ask("Your name?", name=elicit.Text())
    found = answer.values["name"]
except elicit.ElicitationUnavailable:
    found = "unavailable"
helper = await ask_name_in_task() if found != "unavailable" else "-"
return_result(Done(explanation=found + "/" + helper))
"""


def _asker(seen):
    async def asker(request):
        seen.append(request.message)
        return Elicited(action="accept", values={"name": "Ada"})

    return asker


async def test_a_root_sessions_cells_and_helpers_see_the_asker(make_session):
    session, _llm = make_session(cell(ASK_CELL))
    seen = []
    session.set_asker(_asker(seen))
    outcome = await session.prompt("hi")
    assert outcome.explanation == "Ada/Ada"
    assert seen == ["Your name?", "Your name?"]


async def test_without_an_asker_ask_is_unavailable_in_a_turn(make_session):
    session, _llm = make_session(cell(ASK_CELL))
    outcome = await session.prompt("hi")
    assert outcome.explanation == "unavailable/-"


async def test_the_asker_in_place_when_asked_is_used(make_session):
    session, _llm = make_session(cell(ASK_CELL), cell(ASK_CELL))
    session.set_asker(_asker([]))
    assert (await session.prompt("one")).explanation == "Ada/Ada"
    session.set_asker(None)
    assert (await session.prompt("two")).explanation == "unavailable/-"


async def test_a_child_session_never_gets_the_asker(registry, models, root_options):
    models.scripts["child"] = [cell(ASK_CELL)]
    root = await registry.create(root_options)
    child = await registry.create(root.options.inherit(name="child"), parent_id=root.id)
    seen = []
    root.set_asker(_asker(seen))
    child.set_asker(_asker(seen))  # even if a host installed one
    outcome = await child.prompt("hi")
    assert outcome.explanation == "unavailable/-"
    assert seen == []


async def test_the_binding_does_not_leak_out_of_a_turn(make_session):
    session, _llm = make_session(cell(ASK_CELL))
    session.set_asker(_asker([]))
    await session.prompt("hi")
    assert elicit.current_asker() is None


class _Commands:
    """A registry whose one command asks a form, as a host-side command does."""

    def commands(self):
        return ()

    def set_on_change(self, _callback):
        pass

    async def invoke(self, name, raw_args):
        try:
            answer = await elicit.ask("Pick", name=elicit.Text())
        except elicit.ElicitationUnavailable:
            return "unavailable"
        return answer.values["name"]


@pytest.mark.parametrize("root", [True, False])
async def test_commands_run_with_the_asker_bound_in_root_sessions(make_session, root):
    session, _llm = make_session(start=False)
    session._agent.slash_commands = _Commands()
    if not root:
        session.parent_id = "parent"
    session.set_asker(_asker([]))
    result = await session.invoke_command("pick", "")
    assert result.text == ("Ada" if root else "unavailable")
    assert elicit.current_asker() is None
