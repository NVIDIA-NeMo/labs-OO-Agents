# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A NeedInput question to Pool: its ``_poolside/elicitation`` form, string fields only."""

import asyncio

from acp import RequestError, text_block
from acp.schema import (
    AgentMessageChunk,
    AllowedOutcome,
    Implementation,
    RequestPermissionResponse,
)
from coder_test_agents import ScriptedModels, cell, reply

TIMEOUT = 30
POOL = Implementation(name="pool", version="1.0.16")
POOL_ELICITATION = "poolside/elicitation"


def _show_answer():
    """A cell that sends the answer's repr to the user."""
    return cell(
        "[a] = notification['user_messages']\n"
        "self.message(repr(a))\n"
        "return_result(Done(explanation='answered'))"
    )


async def _run(adapter, workspace, text="push it"):
    session_id = (await adapter.new_session(str(workspace))).session_id
    response = await asyncio.wait_for(adapter.prompt(session_id, [text_block(text)]), TIMEOUT)
    return session_id, response


def _requests(client):
    return [(method, params) for kind, method, params in _ext(client)]


def _ext(client):
    return [entry for entry in client.log if entry[0] == "ext"]


async def test_a_free_text_question_is_a_one_field_pool_form(make_adapter, workspace, client):
    models = ScriptedModels(
        {
            None: [
                cell(
                    "return_result(NeedInput(question='Name the release?', "
                    "reason='The tag needs it.'))"
                ),
                _show_answer(),
            ]
        }
    )
    client.ext_answers = [{"action": "accept", "content": {"answer": "Aurora"}}]
    adapter = await make_adapter(models, client_info=POOL)
    session_id, response = await _run(adapter, workspace)

    assert response.stop_reason == "end_turn"
    assert _requests(client) == [
        (
            POOL_ELICITATION,
            {
                "sessionId": session_id,
                "mode": "form",
                "message": "Name the release?\n\nThe tag needs it.",
                "requestedSchema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string", "title": "Name the release?"}},
                    "required": ["answer"],
                },
            },
        )
    ]
    assert client.texts(AgentMessageChunk, session_id)[-1] == "'Aurora'"


async def test_a_string_answer_type_is_a_pool_form(make_adapter, workspace, client):
    models = ScriptedModels(
        {
            None: [
                cell("return_result(NeedInput(question='Which branch?', answer_type=Answer))"),
                _show_answer(),
            ]
        }
    )
    client.ext_answers = [{"action": "accept", "content": {"branch": "dev"}}]
    adapter = await make_adapter(models, client_info=POOL)
    session_id, response = await _run(adapter, workspace)

    assert response.stop_reason == "end_turn"
    [(_, params)] = _requests(client)
    assert params["message"] == "Which branch?"
    assert params["requestedSchema"] == {
        "type": "object",
        "properties": {"branch": {"type": "string", "title": "Branch"}},
        "required": ["branch"],
    }
    assert client.texts(AgentMessageChunk, session_id)[-1] == "Answer(branch='dev')"


def _ask_rollout():
    return cell("return_result(NeedInput(question='Deploy how?', answer_type=Rollout))")


ROLLOUT_SCHEMA = {
    "type": "object",
    "properties": {
        "target": {"type": "string", "title": "Target", "description": "Where to deploy"},
        "replicas": {"type": "string", "title": "Replicas", "description": "a whole number"},
        "dry_run": {"type": "string", "title": "Dry Run", "description": "yes or no"},
    },
    "required": ["target", "replicas", "dry_run"],
}


async def test_non_string_fields_are_asked_as_strings_and_converted(
    make_adapter, workspace, client
):
    models = ScriptedModels({None: [_ask_rollout(), _show_answer()]})
    client.ext_answers = [
        {"action": "accept", "content": {"target": "prod", "replicas": "3", "dry_run": "no"}}
    ]
    adapter = await make_adapter(models, client_info=POOL)
    session_id, response = await _run(adapter, workspace)

    assert response.stop_reason == "end_turn"
    [(_, params)] = _requests(client)
    assert params["requestedSchema"] == ROLLOUT_SCHEMA
    assert client.texts(AgentMessageChunk, session_id)[-1] == (
        "Rollout(target='prod', replicas=3, dry_run=False)"
    )


async def test_a_value_that_does_not_convert_is_asked_again_with_the_error(
    make_adapter, workspace, client
):
    models = ScriptedModels({None: [_ask_rollout(), _show_answer()]})
    client.ext_answers = [
        {"action": "accept", "content": {"target": "prod", "replicas": "two", "dry_run": "no"}},
        {"action": "accept", "content": {"target": "prod", "replicas": "2", "dry_run": "no"}},
    ]
    adapter = await make_adapter(models, client_info=POOL)
    session_id, response = await _run(adapter, workspace)

    assert response.stop_reason == "end_turn"
    [(_, first), (_, second)] = _requests(client)
    assert first["message"] == "Deploy how?"
    assert second["message"].startswith("Deploy how?\n\nThat answer was not accepted: replicas: ")
    assert "integer" in second["message"]
    assert second["requestedSchema"] == ROLLOUT_SCHEMA
    assert client.texts(AgentMessageChunk, session_id)[-1] == (
        "Rollout(target='prod', replicas=2, dry_run=False)"
    )


async def test_a_second_bad_value_falls_back_to_text(make_adapter, workspace, client):
    models = ScriptedModels({None: [_ask_rollout()]})
    bad = {"action": "accept", "content": {"target": "prod", "replicas": "two", "dry_run": "no"}}
    client.ext_answers = [bad, bad]
    adapter = await make_adapter(models, client_info=POOL)
    session_id, response = await _run(adapter, workspace)

    assert response.stop_reason == "end_turn"
    assert len(_requests(client)) == 2
    assert len(models.llms[None].calls) == 1  # nothing was submitted
    assert client.texts(AgentMessageChunk, session_id)[-1] == "Deploy how?"


async def test_a_declined_pool_form_tells_the_agent(make_adapter, workspace, client):
    models = ScriptedModels({None: [_ask_rollout(), reply("Okay.")]})
    client.ext_answers = [{"action": "decline"}]
    adapter = await make_adapter(models, client_info=POOL)
    _session_id, response = await _run(adapter, workspace)
    assert response.stop_reason == "end_turn"
    assert "(declined to answer)" in str(models.llms[None].calls[1].messages)


async def test_a_failed_pool_form_falls_back_to_text_for_the_connection(
    make_adapter, workspace, client
):
    ask = cell("return_result(NeedInput(question='Name the release?'))")
    models = ScriptedModels({None: [ask, ask]})
    client.ext_answers = [RequestError(-32601, "Method not found")]
    adapter = await make_adapter(models, client_info=POOL)
    session_id, response = await _run(adapter, workspace)
    assert response.stop_reason == "end_turn"
    assert client.texts(AgentMessageChunk, session_id)[-1] == "Name the release?"

    response = await asyncio.wait_for(adapter.prompt(session_id, [text_block("Aurora")]), TIMEOUT)
    assert response.stop_reason == "end_turn"
    assert len(_requests(client)) == 1  # not asked again


async def test_other_clients_do_not_get_pool_forms(make_adapter, workspace, client):
    models = ScriptedModels({None: [cell("return_result(NeedInput(question='Name it?'))")]})
    adapter = await make_adapter(models, client_info=Implementation(name="zed", version="1"))
    session_id, response = await _run(adapter, workspace)
    assert response.stop_reason == "end_turn"
    assert _ext(client) == []
    assert client.texts(AgentMessageChunk, session_id)[-1] == "Name it?"


async def test_a_yes_no_question_to_pool_is_still_a_permission_request(
    make_adapter, workspace, client
):
    models = ScriptedModels(
        {
            None: [
                cell("return_result(NeedInput(question='Delete it?', options=['Yes', 'No']))"),
                reply("Deleted."),
            ]
        }
    )
    client.permission_answers = [
        RequestPermissionResponse(outcome=AllowedOutcome(outcome="selected", option_id="Yes"))
    ]
    adapter = await make_adapter(models, client_info=POOL)
    _session_id, response = await _run(adapter, workspace)
    assert response.stop_reason == "end_turn"
    assert _ext(client) == []
    assert len([e for e in client.log if e[0] == "permission"]) == 1


def _ask_branch():
    return cell("return_result(NeedInput(question='Which branch?', options=['main', 'dev']))")


BRANCH_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {
            "type": "string",
            "title": "Which branch?",
            "description": "One of: main, dev",
            "enum": ["main", "dev"],
        }
    },
    "required": ["answer"],
}


async def test_a_choice_question_is_a_one_field_pool_form(make_adapter, workspace, client):
    models = ScriptedModels({None: [_ask_branch(), _show_answer()]})
    client.ext_answers = [{"action": "accept", "content": {"answer": "dev"}}]
    adapter = await make_adapter(models, client_info=POOL)
    session_id, response = await _run(adapter, workspace)

    assert response.stop_reason == "end_turn"
    assert _requests(client) == [
        (
            POOL_ELICITATION,
            {
                "sessionId": session_id,
                "mode": "form",
                "message": "Which branch?",
                "requestedSchema": BRANCH_SCHEMA,
            },
        )
    ]
    assert client.texts(AgentMessageChunk, session_id)[-1] == "'dev'"


async def test_a_choice_answer_matches_ignoring_case_and_spaces(make_adapter, workspace, client):
    models = ScriptedModels({None: [_ask_branch(), _show_answer()]})
    client.ext_answers = [{"action": "accept", "content": {"answer": "  MAIN "}}]
    adapter = await make_adapter(models, client_info=POOL)
    session_id, response = await _run(adapter, workspace)

    assert response.stop_reason == "end_turn"
    assert client.texts(AgentMessageChunk, session_id)[-1] == "'main'"


async def test_a_choice_answer_that_is_not_a_choice_is_asked_again(make_adapter, workspace, client):
    models = ScriptedModels({None: [_ask_branch(), _show_answer()]})
    client.ext_answers = [
        {"action": "accept", "content": {"answer": "release"}},
        {"action": "accept", "content": {"answer": "Dev"}},
    ]
    adapter = await make_adapter(models, client_info=POOL)
    session_id, response = await _run(adapter, workspace)

    assert response.stop_reason == "end_turn"
    [(_, first), (_, second)] = _requests(client)
    assert first["message"] == "Which branch?"
    assert second["message"] == "Which branch?\n\nThat answer was not one of: main, dev"
    assert second["requestedSchema"] == BRANCH_SCHEMA
    assert client.texts(AgentMessageChunk, session_id)[-1] == "'dev'"


async def test_a_second_answer_that_is_not_a_choice_falls_back_to_text(
    make_adapter, workspace, client
):
    models = ScriptedModels({None: [_ask_branch()]})
    bad = {"action": "accept", "content": {"answer": "release"}}
    client.ext_answers = [bad, bad]
    adapter = await make_adapter(models, client_info=POOL)
    session_id, response = await _run(adapter, workspace)

    assert response.stop_reason == "end_turn"
    assert len(_requests(client)) == 2
    assert len(models.llms[None].calls) == 1  # nothing was submitted
    assert client.texts(AgentMessageChunk, session_id)[-1] == "Which branch?\n\n- main\n- dev"


async def test_other_clients_keep_choice_questions_as_text(make_adapter, workspace, client):
    models = ScriptedModels({None: [_ask_branch()]})
    adapter = await make_adapter(models, client_info=Implementation(name="zed", version="1"))
    session_id, response = await _run(adapter, workspace)
    assert response.stop_reason == "end_turn"
    assert _ext(client) == []
    assert client.texts(AgentMessageChunk, session_id)[-1] == "Which branch?\n\n- main\n- dev"
