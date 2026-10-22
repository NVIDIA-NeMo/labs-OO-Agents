# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nooa.elicit.ask from a turn, shown as the client's form (standard or Pool)."""

import asyncio

from acp import text_block
from acp.schema import (
    AcceptElicitationResponse,
    AgentMessageChunk,
    ClientCapabilities,
    ConfigOptionUpdate,
    DeclineElicitationResponse,
    ElicitationCapabilities,
    ElicitationFormCapabilities,
    Implementation,
)
from atom_test_agents import ScriptedModels, cell, reply

POOL = Implementation(name="pool", version="1.0.16")
FORMS = ClientCapabilities(elicitation=ElicitationCapabilities(form=ElicitationFormCapabilities()))

ASK = """
try:
    answer = await elicit.ask(
        "Settings?", count=elicit.Integer(minimum=1), ok=elicit.Confirm(), name=elicit.Text()
    )
    found = answer.action + " " + repr(answer.values)
except elicit.ElicitationUnavailable:
    found = "unavailable"
self.message(found)
return_result(Done(explanation=found))
"""


async def _ask(adapter, workspace):
    session_id = (await adapter.new_session(str(workspace))).session_id
    response = await adapter.prompt(session_id, [text_block("go")])
    return session_id, response


async def test_a_form_capable_client_gets_a_standard_form(make_adapter, workspace, client):
    adapter = await make_adapter(ScriptedModels({None: [cell(ASK)]}), capabilities=FORMS)
    client.elicitation_answers = [
        AcceptElicitationResponse(action="accept", content={"count": 3, "ok": True, "name": "x"})
    ]
    session_id, response = await _ask(adapter, workspace)
    assert response.stop_reason == "end_turn"
    [request] = [entry for entry in client.log if entry[0] == "elicitation"]
    assert request[1] == "Settings?"
    mode = request[2]
    assert mode.session_id == session_id
    props = mode.requested_schema.properties
    assert (props["count"].type, props["count"].minimum) == ("integer", 1)
    assert props["ok"].type == "boolean"
    assert mode.requested_schema.required == ["count", "ok", "name"]
    assert "accept {'count': 3, 'ok': True, 'name': 'x'}" in "".join(
        client.texts(AgentMessageChunk, session_id)
    )


async def test_decline_reaches_the_cell(make_adapter, workspace, client):
    adapter = await make_adapter(ScriptedModels({None: [cell(ASK)]}), capabilities=FORMS)
    client.elicitation_answers = [DeclineElicitationResponse(action="decline")]
    session_id, _response = await _ask(adapter, workspace)
    assert "decline {}" in "".join(client.texts(AgentMessageChunk, session_id))


async def test_pool_gets_strings_and_answers_are_coerced(make_adapter, workspace, client):
    adapter = await make_adapter(ScriptedModels({None: [cell(ASK)]}), client_info=POOL)
    client.ext_answers = [
        {"action": "accept", "content": {"count": "4", "ok": "false", "name": "y"}}
    ]
    session_id, _response = await _ask(adapter, workspace)
    [request] = [entry for entry in client.log if entry[0] == "ext"]
    assert request[1] == "poolside/elicitation"
    params = request[2]
    assert params["sessionId"] == session_id and params["mode"] == "form"
    props = params["requestedSchema"]["properties"]
    assert {prop["type"] for prop in props.values()} == {"string"}
    assert "accept {'count': 4, 'ok': False, 'name': 'y'}" in "".join(
        client.texts(AgentMessageChunk, session_id)
    )


async def test_a_failed_pool_form_is_unavailable_and_turns_pool_forms_off(
    make_adapter, workspace, client
):
    adapter = await make_adapter(ScriptedModels({None: [cell(ASK)]}), client_info=POOL)
    client.ext_answers = [RuntimeError("Pool declined the schema")]
    session_id, _response = await _ask(adapter, workspace)
    assert "unavailable" in "".join(client.texts(AgentMessageChunk, session_id))
    assert adapter._pool_forms is False


async def test_a_client_without_forms_is_unavailable(make_adapter, workspace, client):
    adapter = await make_adapter(ScriptedModels({None: [cell(ASK)]}))
    session_id, _response = await _ask(adapter, workspace)
    assert "unavailable" in "".join(client.texts(AgentMessageChunk, session_id))
    assert not [entry for entry in client.log if entry[0] in ("elicitation", "ext")]


async def test_stop_during_a_pending_form_cancels_and_drops_the_late_answer(
    make_adapter, workspace, client
):
    adapter = await make_adapter(
        ScriptedModels({None: [cell(ASK), reply("again")]}), capabilities=FORMS
    )
    session_id = (await adapter.new_session(str(workspace))).session_id
    client.elicitation_gate = asyncio.Event()
    client.elicitation_answers = [
        AcceptElicitationResponse(action="accept", content={"count": 1, "ok": True, "name": "z"})
    ]
    prompt = asyncio.create_task(adapter.prompt(session_id, [text_block("go")]))
    await asyncio.wait_for(client.elicitation_started.wait(), 10)
    await adapter.cancel(session_id)
    response = await asyncio.wait_for(prompt, 10)
    assert response.stop_reason == "cancelled"
    client.elicitation_gate.set()  # the late answer goes nowhere
    assert "accept" not in "".join(client.texts(AgentMessageChunk, session_id))
    # The session goes on normally.
    response = await adapter.prompt(session_id, [text_block("next")])
    assert response.stop_reason == "end_turn"
    assert "again" in "".join(client.texts(AgentMessageChunk, session_id))


async def test_the_asker_answers_cancel_when_stopped(make_adapter, workspace, client):
    """The asker itself: Stop resolves a pending form to ``cancel``."""
    from nooa.elicit import ElicitationRequest, Text

    adapter = await make_adapter(ScriptedModels(), capabilities=FORMS)
    session_id = (await adapter.new_session(str(workspace))).session_id
    session, bridge = adapter.session(session_id), adapter.bridge(session_id)
    client.elicitation_gate = asyncio.Event()
    client.elicitation_answers = [AcceptElicitationResponse(action="accept", content={"a": "b"})]
    pending = asyncio.create_task(
        adapter._elicit(session, bridge, ElicitationRequest("Q", {"a": Text()}))
    )
    await asyncio.wait_for(client.elicitation_started.wait(), 10)
    adapter._cancel_client_calls(session_id)
    answer = await asyncio.wait_for(pending, 10)
    assert answer.action == "cancel" and answer.values == {}


# ---- /connect: a host-side built-in that uses these forms -------------------


async def test_connect_without_forms_replies_with_the_fallback(atom_adapter, workspace, client):
    from nooa_atom.agent.connect_command import CONNECT_FALLBACK

    adapter = await atom_adapter([])
    session_id = (await adapter.new_session(str(workspace))).session_id
    assert "connect" in {c.name for c in adapter.session(session_id).commands()}
    response = await adapter.prompt(session_id, [text_block("/connect")])
    assert response.stop_reason == "end_turn"
    assert CONNECT_FALLBACK in "".join(client.texts(AgentMessageChunk, session_id))
    assert adapter.session(session_id).info.reply_count == 0


async def test_connect_asks_the_client_and_never_the_agent(
    atom_adapter, workspace, client, monkeypatch
):
    from nooa_atom.agent.connect_command import CANCELLED

    adapter = await atom_adapter([], capabilities=FORMS)
    session_id = (await adapter.new_session(str(workspace))).session_id
    client.elicitation_answers = [DeclineElicitationResponse(action="decline")]
    response = await adapter.prompt(session_id, [text_block("/connect")])
    assert response.stop_reason == "end_turn"
    [request] = [entry for entry in client.log if entry[0] == "elicitation"]
    assert list(request[2].requested_schema.properties) == ["provider"]
    assert CANCELLED in "".join(client.texts(AgentMessageChunk, session_id))
    assert adapter.session(session_id).info.reply_count == 0


async def test_stop_ends_connect(atom_adapter, workspace, client):
    adapter = await atom_adapter([], capabilities=FORMS)
    session_id = (await adapter.new_session(str(workspace))).session_id
    client.elicitation_gate = asyncio.Event()
    client.elicitation_answers = [
        AcceptElicitationResponse(action="accept", content={"provider": "nvidia"})
    ]
    prompt = asyncio.create_task(adapter.prompt(session_id, [text_block("/connect")]))
    await asyncio.wait_for(client.elicitation_started.wait(), 10)
    await adapter.cancel(session_id)
    response = await asyncio.wait_for(prompt, 10)
    assert response.stop_reason == "cancelled"
    client.elicitation_gate.set()
    assert len([entry for entry in client.log if entry[0] == "elicitation"]) == 1


async def test_connect_resends_the_config_options(atom_adapter, workspace, client):
    # A saved alias must reach the client's model picker without a reload.
    adapter = await atom_adapter([], capabilities=FORMS)
    session_id = (await adapter.new_session(str(workspace))).session_id
    before = len(client.updates(session_id, ConfigOptionUpdate))
    client.elicitation_answers = [DeclineElicitationResponse(action="decline")]
    await adapter.prompt(session_id, [text_block("/connect")])
    updates = client.updates(session_id, ConfigOptionUpdate)
    assert len(updates) == before + 1
    assert updates[-1].config_options
