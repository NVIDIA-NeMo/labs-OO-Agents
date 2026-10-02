# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A NeedInput question over ACP: a form, a yes/no permission, or text."""

import asyncio

import pytest
from acp import text_block
from acp.schema import (
    AcceptElicitationResponse,
    AgentMessageChunk,
    AllowedOutcome,
    ClientCapabilities,
    DeclineElicitationResponse,
    DeniedOutcome,
    ElicitationCapabilities,
    ElicitationFormCapabilities,
    ElicitationFormSessionMode,
    RequestPermissionResponse,
    ToolCallProgress,
    ToolCallStart,
    UserMessageChunk,
)
from coder_test_agents import ScriptedModels, cell, reply

TIMEOUT = 30
FORMS = ClientCapabilities(elicitation=ElicitationCapabilities(form=ElicitationFormCapabilities()))


def _ask(question: str, options: list[str] | None = None) -> object:
    return cell(f"return_result(NeedInput(question={question!r}, options={options!r}))")


async def _run(adapter, workspace, text="push it"):
    session_id = (await adapter.new_session(str(workspace))).session_id
    response = await asyncio.wait_for(adapter.prompt(session_id, [text_block(text)]), TIMEOUT)
    return session_id, response


async def test_a_form_answer_goes_to_the_agent_and_the_same_prompt_continues(
    make_adapter, workspace, client
):
    models = ScriptedModels({None: [_ask("Which branch?", ["main", "dev"]), reply("Using dev.")]})
    client.elicitation_answers = [
        AcceptElicitationResponse(action="accept", content={"answer": "dev"})
    ]
    adapter = await make_adapter(models, capabilities=FORMS)
    session_id, response = await _run(adapter, workspace)

    assert response.stop_reason == "end_turn"
    [(_, message, mode)] = [entry for entry in client.log if entry[0] == "elicitation"]
    assert message == "Which branch?"
    assert isinstance(mode, ElicitationFormSessionMode)
    assert mode.session_id == session_id
    assert mode.requested_schema.model_dump(mode="json", by_alias=True, exclude_none=True) == {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "title": "Which branch?", "enum": ["main", "dev"]}
        },
        "required": ["answer"],
    }
    # The question is the turn's final message, sent before the form.
    kinds = [
        entry[0] if entry[0] != "update" else type(entry[2]).__name__
        for entry in client.log
        if entry[0] != "update" or isinstance(entry[2], AgentMessageChunk)
    ]
    assert kinds[kinds.index("elicitation") - 1] == "AgentMessageChunk"
    assert client.texts(AgentMessageChunk, session_id)[-2:] == [
        "Which branch?\n\n- main\n- dev\n\n",
        "Using dev.\n\n",
    ]
    assert "dev" in str(models.llms[None].calls[1].messages[-2:])
    assert client.updates(session_id, UserMessageChunk) == []


async def test_a_declined_form_tells_the_agent(make_adapter, workspace, client):
    models = ScriptedModels({None: [_ask("Which branch?", ["main", "dev"]), reply("Okay.")]})
    client.elicitation_answers = [DeclineElicitationResponse(action="decline")]
    adapter = await make_adapter(models, capabilities=FORMS)
    session_id, response = await _run(adapter, workspace)
    assert response.stop_reason == "end_turn"
    assert "(declined to answer)" in str(models.llms[None].calls[1].messages)
    assert client.updates(session_id, UserMessageChunk) == []


async def test_a_free_text_question_uses_a_one_field_form(make_adapter, workspace, client):
    models = ScriptedModels({None: [_ask("Name the release?"), reply("Named.")]})
    client.elicitation_answers = [
        AcceptElicitationResponse(action="accept", content={"answer": "Aurora"})
    ]
    adapter = await make_adapter(models, capabilities=FORMS)
    _session_id, response = await _run(adapter, workspace)
    assert response.stop_reason == "end_turn"
    assert "Aurora" in str(models.llms[None].calls[1].messages)


async def test_without_forms_a_yes_no_question_is_a_permission_request(
    make_adapter, workspace, client
):
    models = ScriptedModels({None: [_ask("Delete the branch?", ["Yes", "No"]), reply("Deleted.")]})
    client.permission_answers = [
        RequestPermissionResponse(outcome=AllowedOutcome(outcome="selected", option_id="Yes"))
    ]
    adapter = await make_adapter(models)
    session_id, response = await _run(adapter, workspace)
    assert response.stop_reason == "end_turn"

    [(_, permission_session, tool_call, options)] = [e for e in client.log if e[0] == "permission"]
    assert permission_session == session_id
    assert [(o.option_id, o.kind) for o in options] == [
        ("Yes", "allow_once"),
        ("No", "reject_once"),
    ]
    [card] = [
        u
        for u in client.updates(session_id, ToolCallStart)
        if u.tool_call_id == tool_call.tool_call_id
    ]
    assert (card.title, card.kind, card.status) == ("Delete the branch?", "other", "pending")
    [done] = [
        u
        for u in client.updates(session_id, ToolCallProgress)
        if u.tool_call_id == card.tool_call_id
    ]
    assert done.status == "completed"
    assert "Yes" in str(models.llms[None].calls[1].messages[-2:])


async def test_a_refused_permission_is_a_declined_answer(make_adapter, workspace, client):
    models = ScriptedModels({None: [_ask("Delete the branch?", ["yes", "no"]), reply("Kept.")]})
    client.permission_answers = [
        RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))
    ]
    adapter = await make_adapter(models)
    _session_id, response = await _run(adapter, workspace)
    assert response.stop_reason == "end_turn"
    assert "(declined to answer)" in str(models.llms[None].calls[1].messages)


async def test_without_forms_other_questions_end_the_turn_as_text(make_adapter, workspace, client):
    models = ScriptedModels({None: [_ask("Which branch?", ["main", "dev"])]})
    adapter = await make_adapter(models)
    session_id, response = await _run(adapter, workspace)
    assert response.stop_reason == "end_turn"
    assert not [e for e in client.log if e[0] in ("elicitation", "permission")]
    assert client.texts(AgentMessageChunk, session_id)[-1] == "Which branch?\n\n- main\n- dev\n\n"


async def test_a_questions_reason_follows_it(make_adapter, workspace, client):
    models = ScriptedModels(
        {
            None: [
                cell(
                    "return_result(NeedInput(question='Which branch?', options=['main', 'dev'], "
                    "reason='Both have the fix.'))"
                )
            ]
        }
    )
    adapter = await make_adapter(models)
    session_id, _ = await _run(adapter, workspace)
    assert client.texts(AgentMessageChunk, session_id)[-1] == (
        "Which branch?\n\n- main\n- dev\n\nBoth have the fix.\n"
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
    assert client.texts(AgentMessageChunk, session_id) == ["All set.\n"]
    kinds = [
        "message" if entry[0] == "update" and isinstance(entry[2], AgentMessageChunk) else entry[0]
        for entry in client.log
        if entry[0] == "response" or isinstance(entry[2], AgentMessageChunk)
    ]
    assert kinds == ["message", "response"]


async def test_cancel_while_a_form_is_open_ends_the_prompt_cancelled(
    make_adapter, workspace, client
):
    models = ScriptedModels({None: [_ask("Which branch?", ["main", "dev"])]})
    client.elicitation_gate = asyncio.Event()  # never answered
    adapter = await make_adapter(models, capabilities=FORMS)
    session_id = (await adapter.new_session(str(workspace))).session_id
    prompt = asyncio.create_task(adapter.prompt(session_id, [text_block("push it")]))
    await asyncio.wait_for(client.elicitation_started.wait(), TIMEOUT)
    await adapter.cancel(session_id)
    response = await asyncio.wait_for(prompt, TIMEOUT)
    assert response.stop_reason == "cancelled"
    assert len(models.llms[None].calls) == 1  # nothing was submitted


@pytest.mark.parametrize("failure", [RuntimeError("client error")])
async def test_a_failing_form_request_falls_back_to_text(make_adapter, workspace, client, failure):
    models = ScriptedModels({None: [_ask("Which branch?", ["main", "dev"])]})

    async def broken(message, mode, **kwargs):
        raise failure

    client.create_elicitation = broken
    adapter = await make_adapter(models, capabilities=FORMS)
    _session_id, response = await _run(adapter, workspace)
    assert response.stop_reason == "end_turn"
