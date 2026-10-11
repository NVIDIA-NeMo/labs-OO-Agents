# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pool explicit descriptors: string answers, one-shot failure, actions and Stop."""

import asyncio
import json

import pytest
from acp import RequestError, text_block
from acp.schema import AgentMessageChunk, Implementation, UserMessageChunk
from atom_test_agents import ScriptedModels, cell

from nooa.interactive import FormResponse

POOL = Implementation(name="pool", version="1.0.16")
TIMEOUT = 30


def _ask(typed=True, options=None):
    if typed:
        questions = "[TextQuestion(id='target', label='Target?'), TextQuestion(id='replicas', label='Replicas?'), TextQuestion(id='dry_run', label='Dry run?')]"
    elif options:
        choices = [{"value": value, "title": value} for value in options]
        questions = f"[PickOneQuestion(id='answer', label='Answer?', choices={choices!r})]"
    else:
        questions = "[TextQuestion(id='answer', label='Answer?')]"
    return cell(
        f"return_result(NeedInputForm(heading='Details?', reason='Safety.', questions={questions}))"
    )


def _inspect():
    return cell(
        "[a] = notification['user_messages']\nself.v.answer = a\n"
        "return_result(Done(explanation='received', message='Okay.'))"
    )


def _requests(client):
    return [e[2] for e in client.log if e[0] == "ext"]


async def _run(adapter, workspace):
    sid = (await adapter.new_session(str(workspace))).session_id
    return sid, await asyncio.wait_for(adapter.prompt(sid, [text_block("go")]), TIMEOUT)


@pytest.mark.parametrize(
    "typed,options,content,expected",
    [
        (
            True,
            None,
            {"target": "prod", "replicas": "3", "dry_run": "no"},
            {"target": "prod", "replicas": "3", "dry_run": "no"},
        ),
        (False, None, {"answer": "Aurora"}, {"answer": "Aurora"}),
        (False, ["main", "dev"], {"answer": "dev"}, {"answer": "dev"}),
        (False, ["Yes", "No"], {"answer": "Yes"}, {"answer": "Yes"}),
    ],
)
async def test_pool_accept_has_validated_content_and_order(
    make_adapter, workspace, client, typed, options, content, expected
):
    models = ScriptedModels({None: [_ask(typed, options), _inspect()]})
    client.ext_answers = [
        {
            "action": "accept",
            "content": content,
            "field_meta": {"answer": {"source": "user"}},
            "_meta": {"trace": "reply"},
        }
    ]
    adapter = await make_adapter(models, client_info=POOL)
    sid, response = await _run(adapter, workspace)
    assert response.stop_reason == "end_turn"
    answer = adapter.session(sid)._agent.v.answer
    assert isinstance(answer, FormResponse) and answer.action == "accept"
    assert answer.content == expected
    [params] = _requests(client)
    assert params["mode"] == "form"
    assert params["message"] == "Details?\n\nSafety."
    if typed:
        assert params["_meta"] == {"poolside/field_order": ["target", "replicas", "dry_run"]}
        assert all(p["type"] == "string" for p in params["requestedSchema"]["properties"].values())
    [echo] = client.texts(UserMessageChunk, sid)
    assert json.loads(echo) == {"action": "accept", "content": expected}
    assert len(models.llms[None].calls) == 2
    assert not any(e[0] == "permission" for e in client.log)


@pytest.mark.parametrize("action", ["decline", "cancel"])
async def test_pool_actions_survive_transport_and_replay(make_adapter, workspace, client, action):
    models = ScriptedModels({None: [_ask(), _inspect()]})
    client.ext_answers = [{"action": action, "_meta": {"trace": "reply"}, "field_meta": {}}]
    adapter = await make_adapter(models, client_info=POOL)
    sid, _ = await _run(adapter, workspace)
    assert adapter.session(sid)._agent.v.answer == FormResponse(action=action)
    assert len(_requests(client)) == 1
    visible = "".join(client.texts(AgentMessageChunk, sid))
    assert "Details?" in visible and "Safety." in visible
    for field in ("target", "replicas", "dry_run"):
        assert f'"id": "{field}"' in visible
    assert "Client-reported decline does not tell us" in visible
    assert "questions remain visible as text" in visible
    assert "user declined" not in visible.lower()
    assert len(models.llms[None].calls) == 2
    [echo] = client.texts(UserMessageChunk, sid)
    assert json.loads(echo) == {"action": action, "content": None}
    client.log.clear()
    await adapter.load_session(str(workspace), sid)
    assert json.loads(client.texts(UserMessageChunk, sid)[-1])["action"] == action
    replayed = "".join(client.texts(AgentMessageChunk, sid))
    assert "Details?" in replayed
    for field in ("target", "replicas", "dry_run"):
        assert f'"id": "{field}"' in replayed
    assert not _requests(client)


@pytest.mark.parametrize(
    "invalid", [None, {}, {"answer": " "}, {"answer": 2}, {"answer": "unlisted"}]
)
async def test_pool_invalid_choice_fails_closed_once(make_adapter, workspace, client, invalid):
    models = ScriptedModels({None: [_ask(False, ["main", "dev"])]})
    client.ext_answers = [
        {"action": "accept", "content": invalid},
        {"action": "accept", "content": {"answer": "dev"}},
    ]
    adapter = await make_adapter(models, client_info=POOL)
    sid, _ = await _run(adapter, workspace)
    assert len(_requests(client)) == 1
    assert len(models.llms[None].calls) == 1
    assert not client.texts(UserMessageChunk, sid)
    assert "No automatic retry" in client.texts(AgentMessageChunk, sid)[-1]


@pytest.mark.parametrize("failure", ["bad-content", "unknown-action", "client-error"])
async def test_pool_unsupported_or_invalid_never_manufactures_response(
    make_adapter, workspace, client, failure
):
    ask = _ask()
    bad = {"action": "accept", "content": {"target": "prod", "replicas": 2, "dry_run": "no"}}
    client.ext_answers = [bad, bad]
    if failure == "unknown-action":
        client.ext_answers = [{"action": "unknown"}]
    elif failure == "client-error":
        client.ext_answers = [RequestError(-32601, "Method not found")]
    models = ScriptedModels({None: [ask]})
    adapter = await make_adapter(models, client_info=POOL)
    sid, response = await _run(adapter, workspace)
    assert response.stop_reason == "end_turn"
    assert client.updates(sid, UserMessageChunk) == []
    assert len(models.llms[None].calls) == 1
    assert "Explicit form request" in "".join(client.texts(AgentMessageChunk, sid))


async def test_pool_stop_never_submits_pending_response(make_adapter, workspace, client):
    models = ScriptedModels({None: [_ask()]})
    client.ext_gate = asyncio.Event()
    client.ext_answers = [
        {"action": "accept", "content": {"target": "prod", "replicas": "3", "dry_run": "no"}}
    ]
    adapter = await make_adapter(models, client_info=POOL)
    sid = (await adapter.new_session(str(workspace))).session_id
    pending = asyncio.create_task(adapter.prompt(sid, [text_block("go")]))
    await client.wait_for(lambda: bool(_requests(client)))
    await adapter.cancel(sid)
    assert (await asyncio.wait_for(pending, TIMEOUT)).stop_reason == "cancelled"
    assert client.updates(sid, UserMessageChunk) == []
    assert len(models.llms[None].calls) == 1 and len(client.ext_answers) == 1


async def test_pool_literal_or_text_field_explicitly_accepts_alternative(
    make_adapter, workspace, client
):
    models = ScriptedModels(
        {
            None: [
                cell(
                    "return_result(NeedInputForm(heading='Branch?', questions=[PickOneOrTextQuestion(id='name', label='Branch?', choices=[FormChoice(value='main', title='Main branch')])]))"
                ),
                _inspect(),
            ]
        }
    )
    client.ext_answers = [{"action": "accept", "content": {"name": "release"}}]
    adapter = await make_adapter(models, client_info=POOL)
    sid, _ = await _run(adapter, workspace)
    assert "anyOf" in _requests(client)[0]["requestedSchema"]["properties"]["name"]
    assert adapter.session(sid)._agent.v.answer.content["name"] == "release"
