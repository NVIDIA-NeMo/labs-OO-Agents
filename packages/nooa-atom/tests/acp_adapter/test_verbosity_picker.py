# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Argument-free /verbosity opens a client form, never an agent turn."""

import asyncio

import pytest
from acp import text_block
from acp.schema import (
    AcceptElicitationResponse,
    AgentMessageChunk,
    ClientCapabilities,
    ConfigOptionUpdate,
    ElicitationCapabilities,
    ElicitationFormCapabilities,
    Implementation,
    UserMessageChunk,
)
from atom_test_agents import ScriptedModels

POOL = Implementation(name="pool", version="1.0.16")


@pytest.mark.parametrize("level", ["silent", "concise", "verbose"])
async def test_pool_command_opens_strict_picker(make_adapter, workspace, client, level):
    adapter = await make_adapter(ScriptedModels(), client_info=POOL)
    session_id = (await adapter.new_session(str(workspace))).session_id
    client.ext_answers = [{"action": "accept", "content": {"verbosity": level}}]
    response = await adapter.prompt(session_id, [text_block("/verbosity")])
    assert response.stop_reason == "end_turn"
    [request] = [entry for entry in client.log if entry[0] == "ext"]
    assert request[1] == "poolside/elicitation"
    params = request[2]
    assert params["sessionId"] == session_id and params["mode"] == "form"
    assert "ACP verbosity: verbose" in params["message"]
    schema = params["requestedSchema"]
    assert schema["required"] == ["verbosity"]
    field = schema["properties"]["verbosity"]
    assert field["type"] == "string" and "anyOf" not in field
    assert [choice["const"] for choice in field["oneOf"]] == ["silent", "concise", "verbose"]
    assert adapter.bridge(session_id).verbosity == level
    assert client.texts(AgentMessageChunk, session_id)[-1] == f"ACP verbosity: {level}.\n\n"
    options = client.updates(session_id, ConfigOptionUpdate)[-1].config_options
    assert next(o for o in options if o.id == "verbosity").current_value == level
    assert client.texts(UserMessageChunk, session_id) == []
    assert adapter.session(session_id).info.reply_count == 0
    assert adapter.session(session_id).transcript() == []


@pytest.mark.parametrize(
    "answer",
    [
        {"action": "decline"},
        {"action": "cancel"},
    ],
)
async def test_dismissing_picker_keeps_level(make_adapter, workspace, client, answer):
    adapter = await make_adapter(ScriptedModels(), client_info=POOL)
    session_id = (await adapter.new_session(str(workspace))).session_id
    adapter.bridge(session_id).set_verbosity("concise")
    client.ext_answers = [answer]
    response = await adapter.prompt(session_id, [text_block("/verbosity")])
    assert response.stop_reason == "end_turn"
    assert adapter.bridge(session_id).verbosity == "concise"
    assert "unchanged" in client.texts(AgentMessageChunk, session_id)[-1]
    assert adapter.session(session_id).info.reply_count == 0
    assert client.texts(UserMessageChunk, session_id) == []


async def test_invalid_picker_answers_do_not_change_level(make_adapter, workspace, client):
    adapter = await make_adapter(ScriptedModels(), client_info=POOL)
    session_id = (await adapter.new_session(str(workspace))).session_id
    client.ext_answers = [
        {"action": "accept", "content": {"verbosity": "invalid"}},
        {"action": "accept", "content": {"verbosity": "invalid"}},
    ]
    await adapter.prompt(session_id, [text_block("/verbosity")])
    assert adapter.bridge(session_id).verbosity == "verbose"
    assert "Usage: /verbosity" in client.texts(AgentMessageChunk, session_id)[-1]
    assert adapter.session(session_id).info.reply_count == 0


async def test_picker_transport_failure_falls_back_to_text(make_adapter, workspace, client):
    adapter = await make_adapter(ScriptedModels(), client_info=POOL)
    session_id = (await adapter.new_session(str(workspace))).session_id
    # FakeClient propagates errors thrown while retrieving an answer.
    client.ext_answers = []
    response = await adapter.prompt(session_id, [text_block("/verbosity")])
    assert response.stop_reason == "end_turn"
    assert "Usage: /verbosity" in client.texts(AgentMessageChunk, session_id)[-1]
    await adapter.prompt(session_id, [text_block("/verbosity silent")])
    assert adapter.bridge(session_id).verbosity == "silent"
    assert adapter.session(session_id).info.reply_count == 0


async def test_stop_cancels_local_picker(make_adapter, workspace, client):
    adapter = await make_adapter(ScriptedModels(), client_info=POOL)
    session_id = (await adapter.new_session(str(workspace))).session_id
    client.ext_gate = asyncio.Event()
    prompt = asyncio.create_task(adapter.prompt(session_id, [text_block("/verbosity")]))
    await client.wait_for(lambda: any(entry[0] == "ext" for entry in client.log))
    await adapter.cancel(session_id)
    response = await asyncio.wait_for(prompt, 5)
    assert response.stop_reason == "cancelled"
    assert adapter.bridge(session_id).verbosity == "verbose"
    assert adapter.session(session_id).info.reply_count == 0


async def test_standard_acp_form_also_stays_local(make_adapter, workspace, client):
    capabilities = ClientCapabilities(
        elicitation=ElicitationCapabilities(form=ElicitationFormCapabilities())
    )
    adapter = await make_adapter(ScriptedModels(), capabilities=capabilities)
    session_id = (await adapter.new_session(str(workspace))).session_id
    client.elicitation_answers = [
        AcceptElicitationResponse(action="accept", content={"verbosity": "concise"})
    ]
    await adapter.prompt(session_id, [text_block("/verbosity")])
    [request] = [entry for entry in client.log if entry[0] == "elicitation"]
    schema = request[2].requested_schema
    assert schema.properties["verbosity"].enum == ["silent", "concise", "verbose"]
    assert adapter.bridge(session_id).verbosity == "concise"
    assert adapter.session(session_id).info.reply_count == 0
    assert client.texts(UserMessageChunk, session_id) == []
