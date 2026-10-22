# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A NeedInput question over ACP: the agent's message, answered by the next prompt."""

import asyncio

import pytest
from acp import text_block
from acp.schema import (
    AgentMessageChunk,
    ClientCapabilities,
    ElicitationCapabilities,
    ElicitationFormCapabilities,
    Implementation,
    UserMessageChunk,
)
from atom_test_agents import ScriptedModels, cell, reply

TIMEOUT = 30
FORMS = ClientCapabilities(elicitation=ElicitationCapabilities(form=ElicitationFormCapabilities()))
POOL = Implementation(name="pool", version="1.0.16")


def _ask(question: str) -> object:
    return cell(f"return_result(NeedInput(question={question!r}))")


async def _run(adapter, workspace, text="push it"):
    session_id = (await adapter.new_session(str(workspace))).session_id
    response = await asyncio.wait_for(adapter.prompt(session_id, [text_block(text)]), TIMEOUT)
    return session_id, response


@pytest.mark.parametrize("capabilities,client_info", [(None, None), (FORMS, None), (FORMS, POOL)])
async def test_a_question_ends_the_turn_as_text_and_the_next_prompt_answers_it(
    make_adapter, workspace, client, capabilities, client_info
):
    models = ScriptedModels({None: [_ask("Which branch: main or dev?"), reply("Using dev.")]})
    adapter = await make_adapter(models, capabilities=capabilities, client_info=client_info)
    session_id, response = await _run(adapter, workspace)
    assert response.stop_reason == "end_turn"
    # Never a dialog: no form, Pool form or permission request, whatever the client.
    assert not [e for e in client.log if e[0] in ("elicitation", "permission", "ext")]
    assert client.texts(AgentMessageChunk, session_id)[-1] == "Which branch: main or dev?\n\n"
    assert len(models.llms[None].calls) == 1

    response = await asyncio.wait_for(adapter.prompt(session_id, [text_block("dev")]), TIMEOUT)
    assert response.stop_reason == "end_turn"
    assert "dev" in str(models.llms[None].calls[1].messages[-2:])
    assert client.texts(AgentMessageChunk, session_id)[-1] == "Using dev.\n\n"
    assert [e.content for e in adapter.session(session_id).transcript() if e.role == "user"] == [
        "push it",
        "dev",
    ]
    assert client.updates(session_id, UserMessageChunk) == []


async def test_a_questions_reason_follows_it(make_adapter, workspace, client):
    models = ScriptedModels(
        {
            None: [
                cell(
                    "return_result(NeedInput(question='Which branch: main or dev?', "
                    "reason='Both have the fix.'))"
                )
            ]
        }
    )
    adapter = await make_adapter(models)
    session_id, _ = await _run(adapter, workspace)
    assert client.texts(AgentMessageChunk, session_id)[-1] == (
        "Which branch: main or dev?\n\nBoth have the fix.\n\n"
    )


async def test_a_done_message_reaches_the_client_before_the_prompt_answers(
    make_adapter, workspace, client
):
    models = ScriptedModels(
        {None: [cell("return_result(Done(explanation='x', message='All set.'))")]}
    )
    adapter = await make_adapter(models)
    session_id, response = await _run(adapter, workspace)
    client.log.append(("response", "prompt", response))
    assert client.texts(AgentMessageChunk, session_id) == ["All set.\n\n"]
    kinds = [
        "message" if entry[0] == "update" and isinstance(entry[2], AgentMessageChunk) else entry[0]
        for entry in client.log
        if entry[0] == "response" or isinstance(entry[2], AgentMessageChunk)
    ]
    assert kinds == ["message", "response"]


async def test_mcp_sign_in_callback_is_not_admitted_or_echoed(
    atom_adapter, workspace, client, monkeypatch
):
    from nooa_atom.session.events import ItemAdmitted
    from nooa_atom.skills.mcp_servers import MCPServers

    completed = []

    async def complete_sign_in(self, name, address):
        completed.append((name, address))
        return "Signed in to remote."

    monkeypatch.setattr(MCPServers, "complete_sign_in", complete_sign_in)
    adapter = await atom_adapter([reply("Hi.")])
    session_id, _ = await _run(adapter, workspace, text="hello")
    session = adapter.session(session_id)
    before = len(session.handle.events.filter(type="ItemAdmitted"))
    client.log.clear()
    address = "https://callback.invalid/?code=UNIQUE_SENTINEL&state=STATE_SENTINEL"
    response = await adapter.prompt(session_id, [text_block("/mcp auth remote " + address)])
    assert response.stop_reason == "end_turn"
    assert completed == [("remote", address)]
    assert len(session.handle.events.filter(type="ItemAdmitted")) == before
    assert client.updates(session_id, UserMessageChunk) == []
    assert (
        len(
            [
                e
                for e in session.handle.events.all_events()
                if isinstance(e, ItemAdmitted) and e.channel == "user_messages"
            ]
        )
        == 1
    )
    await adapter.load_session(str(workspace), session_id)
    visible = str(client.log) + str(session.transcript())
    assert "Signed in to remote." in visible
    assert "UNIQUE_SENTINEL" not in visible
    assert "STATE_SENTINEL" not in visible
    assert address not in visible
