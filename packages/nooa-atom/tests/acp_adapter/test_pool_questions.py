# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pool explicit descriptors: string answers, one-shot failure, actions and Stop."""

import asyncio

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
        questions = f"[PickOneQuestion(id='answer', label='Answer?', choices={choices!r}, allow_other=False)]"
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
    assert echo == FormResponse(action="accept", content=expected).as_text()
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
    for label in ("Target?", "Replicas?", "Dry run?"):
        assert f"- {label}" in visible
    assert "{" not in visible
    assert "user declined" not in visible.lower()
    assert len(models.llms[None].calls) == 2
    [echo] = client.texts(UserMessageChunk, sid)
    assert echo == FormResponse(action=action).as_text()
    client.log.clear()
    await adapter.load_session(str(workspace), sid)
    assert client.texts(UserMessageChunk, sid)[-1] == FormResponse(action=action).as_text() + "\n"
    replayed = "".join(client.texts(AgentMessageChunk, sid))
    assert "Details?" in replayed
    for label in ("Target?", "Replicas?", "Dry run?"):
        assert f"- {label}" in replayed
    assert not _requests(client)


@pytest.mark.parametrize("invalid", [None, {"answer": [2]}, {"answer": {"x": "y"}}])
async def test_pool_malformed_answer_fails_closed_once(make_adapter, workspace, client, invalid):
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
    assert "form answer wasn't used" in client.texts(AgentMessageChunk, sid)[-1]


@pytest.mark.parametrize("content", [{}, {"answer": " "}, {"answer": 2}, {"answer": "unlisted"}])
async def test_pool_answer_reaches_agent_as_entered(make_adapter, workspace, client, content):
    models = ScriptedModels({None: [_ask(False, ["main", "dev"]), _inspect()]})
    client.ext_answers = [{"action": "accept", "content": content}]
    adapter = await make_adapter(models, client_info=POOL)
    sid, response = await _run(adapter, workspace)
    assert response.stop_reason == "end_turn"
    assert len(_requests(client)) == 1  # never re-asked
    answer = adapter.session(sid)._agent.v.answer
    assert answer == FormResponse(action="accept", content=content)
    assert not any("form answer wasn't used" in t for t in client.texts(AgentMessageChunk, sid))


@pytest.mark.parametrize("failure", ["bad-content", "unknown-action", "client-error"])
async def test_pool_unsupported_or_invalid_never_manufactures_response(
    make_adapter, workspace, client, failure
):
    ask = _ask()
    bad = {"action": "accept", "content": {"target": "prod", "replicas": [2], "dry_run": "no"}}
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
    text = "".join(client.texts(AgentMessageChunk, sid))
    assert "- Replicas?" in text and "Explicit form request" not in text


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
                    "return_result(NeedInputForm(heading='Branch?', questions=[PickOneQuestion(id='name', label='Branch?', choices=[FormChoice(value='main', title='Main branch')])]))"
                ),
                _inspect(),
            ]
        }
    )
    client.ext_answers = [{"action": "accept", "content": {"name": "release"}}]
    adapter = await make_adapter(models, client_info=POOL)
    sid, _ = await _run(adapter, workspace)
    sent = _requests(client)[0]["requestedSchema"]["properties"]["name"]
    # Pool shows anyOf as a bare text box, so the choices go in the description.
    assert "anyOf" not in sent and sent["description"] == "For example: Main branch."
    assert adapter.session(sid)._agent.v.answer.content["name"] == "release"


async def test_pool_string_answers_reach_the_agent_unchanged(make_adapter, workspace, client):
    form = (
        "return_result(NeedInputForm(heading='Deploy', questions=["
        "IntegerQuestion(id='replicas', label='Replicas?', minimum=1, maximum=10), "
        "BooleanQuestion(id='dry_run', label='Dry run?'), "
        "PickManyQuestion(id='regions', label='Regions?', choices=["
        "FormChoice(value='us', title='US'), FormChoice(value='eu', title='EU')])]))"
    )
    models = ScriptedModels({None: [cell(form), _inspect()]})
    client.ext_answers = [
        {"action": "accept", "content": {"replicas": "3", "dry_run": "false", "regions": "us, EU"}}
    ]
    adapter = await make_adapter(models, client_info=POOL)
    sid, _ = await _run(adapter, workspace)
    [params] = _requests(client)
    props = params["requestedSchema"]["properties"]
    assert all(prop["type"] == "string" for prop in props.values())
    answer = adapter.session(sid)._agent.v.answer
    assert answer.content == {"replicas": "3", "dry_run": "false", "regions": "us, EU"}
