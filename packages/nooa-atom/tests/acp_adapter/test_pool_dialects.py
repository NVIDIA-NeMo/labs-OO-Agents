# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pool 2 (``pool/*`` extensions) next to Pool 1 (``poolside/*``): the dialect per connection."""

import asyncio

import pytest
from acp import PROTOCOL_VERSION, RequestError, text_block
from acp.schema import AgentMessageChunk, Implementation, SessionInfoUpdate
from atom_test_agents import BLOCKING_CELL, ScriptedModels, cell, fresh_events, reply
from nooa_atom.acp.protocol import (
    POOL_DIALECT,
    POOLSIDE_DIALECT,
    initialize_response,
    pool_dialects,
)

TIMEOUT = 30
POOL_1 = Implementation(name="pool", version="1.0.16")
POOL_2 = Implementation(name="pool", version="v2.0.0-alpha")
POOL_UNTAGGED = Implementation(name="pool", version="untagged")


@pytest.mark.parametrize(
    "client_info,expected",
    [
        (None, ()),
        (Implementation(name="zed", version="2.0.0"), ()),
        (POOL_1, (POOLSIDE_DIALECT,)),
        (Implementation(name="pool", version="v1.4.0"), (POOLSIDE_DIALECT,)),
        (POOL_2, (POOL_DIALECT,)),
        (Implementation(name="pool", version="2.1.3"), (POOL_DIALECT,)),
        (Implementation(name="pool", version="v10.0.0"), (POOL_DIALECT,)),
        (POOL_UNTAGGED, (POOLSIDE_DIALECT, POOL_DIALECT)),
    ],
)
def test_the_client_info_picks_the_pool_dialect(client_info, expected):
    assert pool_dialects(client_info) == expected


@pytest.mark.parametrize(
    "client_info,steer_keys",
    [
        (None, ["poolside/session_steer"]),
        (POOL_1, ["poolside/session_steer"]),
        (POOL_2, ["pool/session_steer"]),
        (POOL_UNTAGGED, ["poolside/session_steer", "pool/session_steer"]),
    ],
)
async def test_initialize_advertises_the_steer_of_the_clients_dialect(
    make_adapter, client_info, steer_keys
):
    response = initialize_response(PROTOCOL_VERSION, client_info)
    meta = response.agent_capabilities.field_meta
    assert [key for key in meta if key.endswith("/session_steer")] == steer_keys
    assert all(meta[key] is True for key in steer_keys)
    assert "dev.nooa/inject" in meta
    # The adapter answers as the router does.
    adapter = await make_adapter(ScriptedModels(), client_info=client_info)
    assert adapter._pool_client == (client_info is not None)


def test_pool_1_initialize_is_unchanged():
    dump = {"by_alias": True, "exclude_none": True, "exclude_unset": True, "mode": "json"}
    assert initialize_response(PROTOCOL_VERSION, POOL_1).model_dump(**dump) == (
        initialize_response(PROTOCOL_VERSION).model_dump(**dump)
    )


# ---- helpers ------------------------------------------------------------


async def _new(adapter, workspace):
    return (await adapter.new_session(str(workspace))).session_id


def _input_events(client, session_id):
    return [
        entry[2].field_meta
        for entry in client.log
        if entry[0] == "update"
        and entry[1] == session_id
        and isinstance(entry[2], SessionInfoUpdate)
        and entry[2].field_meta
    ]


def _user_item_ids(adapter, session_id):
    return {
        entry.content: entry.item_id
        for entry in adapter.session(session_id).transcript()
        if entry.role == "user"
    }


async def _steer(adapter, method, session_id, text, input_id):
    """Send a steer on ``method``; return its request as a task once the message is admitted."""
    admitted = asyncio.Event()

    def on_update(update):
        if getattr(update, "kind", "") == "item_admitted" and getattr(update, "text", "") == text:
            admitted.set()

    unsubscribe = adapter.session(session_id).subscribe(on_update)
    params = {
        "sessionId": session_id,
        "inputId": input_id,
        "prompt": [text_block(text).model_dump()],
    }
    try:
        request = asyncio.create_task(adapter.ext_method(method, params))
        await asyncio.wait_for(admitted.wait(), TIMEOUT)
    finally:
        unsubscribe()
    return request


async def _steer_during_a_prompt(make_adapter, workspace, client, client_info, method):
    started, block = fresh_events()
    models = ScriptedModels({None: [cell(BLOCKING_CELL), reply("First."), reply("Second.")]})
    adapter = await make_adapter(models, client_info=client_info)
    session_id = await _new(adapter, workspace)
    prompt = asyncio.create_task(adapter.prompt(session_id, [text_block("first")]))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    steer = await _steer(adapter, method, session_id, "queued second", "steer-7")
    block.set()
    assert await asyncio.wait_for(steer, TIMEOUT) == {"inputId": "steer-7"}
    assert (await asyncio.wait_for(prompt, TIMEOUT)).stop_reason == "end_turn"
    assert "Second.\n\n" in client.texts(AgentMessageChunk, session_id)
    return adapter, session_id


# ---- steering -------------------------------------------------------------


async def test_pool_2_steers_with_pool_session_steer_and_gets_pool_input_events(
    make_adapter, workspace, client
):
    adapter, session_id = await _steer_during_a_prompt(
        make_adapter, workspace, client, POOL_2, "pool/session_steer"
    )
    ids = _user_item_ids(adapter, session_id)
    assert _input_events(client, session_id) == [
        {"pool/inputEventId": ids["first"]},
        {"pool/clientInputId": "steer-7", "pool/inputEventId": ids["queued second"]},
    ]


async def test_either_steer_method_is_accepted_from_any_client(make_adapter, workspace, client):
    adapter, session_id = await _steer_during_a_prompt(
        make_adapter, workspace, client, POOL_1, "pool/session_steer"
    )
    ids = _user_item_ids(adapter, session_id)
    # A known Pool 1 keeps its dialect.
    assert _input_events(client, session_id)[-1] == {
        "poolside/clientInputId": "steer-7",
        "poolside/inputEventId": ids["queued second"],
    }


async def test_pool_2_steer_checks_its_params(make_adapter, workspace):
    adapter = await make_adapter(ScriptedModels(), client_info=POOL_2)
    session_id = await _new(adapter, workspace)
    with pytest.raises(RequestError) as invalid:
        await adapter.ext_method("pool/session_steer", {"sessionId": session_id})
    assert invalid.value.code == RequestError.invalid_params().code


async def test_an_untagged_pool_gets_both_until_it_steers(make_adapter, workspace, client):
    adapter, session_id = await _steer_during_a_prompt(
        make_adapter, workspace, client, POOL_UNTAGGED, "pool/session_steer"
    )
    ids = _user_item_ids(adapter, session_id)
    assert _input_events(client, session_id) == [
        {"poolside/inputEventId": ids["first"], "pool/inputEventId": ids["first"]},
        {"pool/clientInputId": "steer-7", "pool/inputEventId": ids["queued second"]},
    ]
    assert adapter._pool_dialects == (POOL_DIALECT,)


async def test_an_invalid_steer_does_not_pick_the_dialect(make_adapter, workspace):
    adapter = await make_adapter(ScriptedModels(), client_info=POOL_UNTAGGED)
    session_id = await _new(adapter, workspace)
    both = adapter._pool_dialects
    assert len(both) == 2
    for params in (
        {"sessionId": session_id},
        {"sessionId": session_id, "inputId": "x", "prompt": [text_block("/help").model_dump()]},
    ):
        with pytest.raises(RequestError):
            await adapter.ext_method("poolside/session_steer", params)
        assert adapter._pool_dialects == both


async def test_a_replay_to_pool_2_uses_pool_input_events(make_adapter, workspace, client):
    adapter = await make_adapter(ScriptedModels({None: [reply("Hi.")]}), client_info=POOL_2)
    session_id = await _new(adapter, workspace)
    await asyncio.wait_for(adapter.prompt(session_id, [text_block("hello")]), TIMEOUT)
    client.log.clear()
    await adapter.load_session(str(workspace), session_id)
    assert _input_events(client, session_id) == [
        {"pool/inputEventId": _user_item_ids(adapter, session_id)["hello"]}
    ]


# ---- forms ----------------------------------------------------------------


async def test_pool_2_gets_pool_elicitation_with_an_elicitation_id(make_adapter, workspace, client):
    adapter = await make_adapter(ScriptedModels(), client_info=POOL_2)
    session_id = await _new(adapter, workspace)
    client.ext_answers = [
        {"action": "accept", "content": {"verbosity": "silent"}},
        {"action": "accept", "content": {"verbosity": "concise"}},
    ]
    await adapter.prompt(session_id, [text_block("/verbosity")])
    await adapter.prompt(session_id, [text_block("/verbosity")])
    requests = [entry for entry in client.log if entry[0] == "ext"]
    assert [method for _, method, _ in requests] == ["pool/elicitation"] * 2
    first, second = (params for _, _, params in requests)
    assert first["sessionId"] == session_id and first["mode"] == "form"
    assert first["requestedSchema"]["required"] == ["verbosity"]
    assert isinstance(first["elicitationId"], str) and first["elicitationId"]
    assert first["elicitationId"] != second["elicitationId"]
    assert adapter.bridge(session_id).verbosity == "concise"


async def test_pool_1_form_has_no_elicitation_id(make_adapter, workspace, client):
    adapter = await make_adapter(ScriptedModels(), client_info=POOL_1)
    session_id = await _new(adapter, workspace)
    client.ext_answers = [{"action": "accept", "content": {"verbosity": "silent"}}]
    await adapter.prompt(session_id, [text_block("/verbosity")])
    [(_, method, params)] = [entry for entry in client.log if entry[0] == "ext"]
    assert method == "poolside/elicitation"
    assert "elicitationId" not in params


async def test_a_failed_pool_2_form_falls_back_to_text(make_adapter, workspace, client):
    adapter = await make_adapter(ScriptedModels(), client_info=POOL_2)
    session_id = await _new(adapter, workspace)
    client.ext_answers = [RequestError.method_not_found("_pool/elicitation")]
    response = await adapter.prompt(session_id, [text_block("/verbosity")])
    assert response.stop_reason == "end_turn"
    assert "Usage: /verbosity" in client.texts(AgentMessageChunk, session_id)[-1]
    assert not adapter._pool_forms
