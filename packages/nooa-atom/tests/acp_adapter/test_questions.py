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
    CancelElicitationResponse,
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
from atom_test_agents import ScriptedModels, cell, reply

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
    assert client.texts(UserMessageChunk, session_id) == ["dev"]
    conversation = [
        u
        for u in client.updates(session_id)
        if isinstance(u, (AgentMessageChunk, UserMessageChunk))
    ]
    assert isinstance(conversation[-2], UserMessageChunk)
    assert conversation[-1].content.text == "Using dev.\n\n"
    assert [e.content for e in adapter.session(session_id).transcript() if e.role == "user"] == [
        "push it",
        "dev",
    ]
    assert len(models.llms[None].calls) == 2


@pytest.mark.parametrize(
    "answer",
    [DeclineElicitationResponse(action="decline"), CancelElicitationResponse(action="cancel")],
)
async def test_a_declined_form_tells_the_agent(make_adapter, workspace, client, answer):
    models = ScriptedModels({None: [_ask("Which branch?", ["main", "dev"]), reply("Okay.")]})
    client.elicitation_answers = [answer]
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
    session_id, response = await _run(adapter, workspace)
    assert client.texts(UserMessageChunk, session_id) == ["Aurora"]
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
    assert client.updates(session_id, UserMessageChunk) == []


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
        "Which branch?\n\n- main\n- dev\n\nBoth have the fix.\n\n"
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
    assert client.updates(session_id, UserMessageChunk) == []


@pytest.mark.parametrize("failure", [RuntimeError("client error")])
async def test_a_failing_form_request_falls_back_to_text(make_adapter, workspace, client, failure):
    models = ScriptedModels({None: [_ask("Which branch?", ["main", "dev"])]})

    async def broken(message, mode, **kwargs):
        raise failure

    client.create_elicitation = broken
    adapter = await make_adapter(models, capabilities=FORMS)
    _session_id, response = await _run(adapter, workspace)
    assert response.stop_reason == "end_turn"


async def test_a_typed_standard_form_echoes_serialized_validated_answer(
    make_adapter, workspace, client
):
    import json

    models = ScriptedModels(
        {
            None: [
                cell("return_result(NeedInput(question='Deploy how?', answer_type=Rollout))"),
                reply("Deploying."),
            ]
        }
    )
    client.elicitation_answers = [
        AcceptElicitationResponse(
            action="accept", content={"target": "prod", "replicas": 3, "dry_run": False}
        )
    ]
    adapter = await make_adapter(models, capabilities=FORMS)
    session_id, response = await _run(adapter, workspace)
    assert response.stop_reason == "end_turn"
    [echo] = client.texts(UserMessageChunk, session_id)
    assert json.loads(echo) == {"target": "prod", "replicas": 3, "dry_run": False}
    conversation = [
        u
        for u in client.updates(session_id)
        if isinstance(u, (AgentMessageChunk, UserMessageChunk))
    ]
    assert isinstance(conversation[-2], UserMessageChunk)
    assert conversation[-1].content.text == "Deploying.\n\n"
    assert [e.content for e in adapter.session(session_id).transcript() if e.role == "user"] == [
        "push it",
        echo,
    ]
    assert len(models.llms[None].calls) == 2


@pytest.mark.parametrize(
    "typed,content",
    [
        (False, None),
        (False, {}),
        (False, {"answer": ""}),
        (False, {"answer": " "}),
        (False, {"answer": 2}),
        (False, {"answer": "release"}),
        (True, {"target": "prod"}),
        (True, {"target": "prod", "replicas": "two", "dry_run": False}),
    ],
)
async def test_invalid_standard_form_answers_are_not_admitted_or_echoed(
    make_adapter, workspace, client, typed, content
):
    ask = (
        cell("return_result(NeedInput(question='Deploy how?', answer_type=Rollout))")
        if typed
        else _ask("Which branch?", ["main", "dev"])
    )
    models = ScriptedModels({None: [ask]})
    client.elicitation_answers = [AcceptElicitationResponse(action="accept", content=content)]
    adapter = await make_adapter(models, capabilities=FORMS)
    session_id, response = await _run(adapter, workspace)
    assert response.stop_reason == "end_turn"
    assert client.updates(session_id, UserMessageChunk) == []
    assert len(models.llms[None].calls) == 1
    assert [e.content for e in adapter.session(session_id).transcript() if e.role == "user"] == [
        "push it"
    ]


@pytest.mark.parametrize("pool", [False, True])
@pytest.mark.parametrize("accepted", [False, True])
async def test_form_answer_replay_preserves_admitted_sources_once(
    make_adapter, workspace, client, pool, accepted
):
    from acp.schema import Implementation
    from nooa_atom.session.events import ItemAdmitted

    models = ScriptedModels({None: [_ask("Name?"), reply("Okay.")]})
    if pool:
        client.ext_answers = [
            {"action": "accept", "content": {"answer": "Aurora"}}
            if accepted
            else {"action": "decline"}
        ]
    else:
        client.elicitation_answers = [
            AcceptElicitationResponse(action="accept", content={"answer": "Aurora"})
            if accepted
            else DeclineElicitationResponse(action="decline")
        ]
    adapter = await make_adapter(
        models,
        capabilities=FORMS,
        client_info=Implementation(name="pool", version="1.0.16") if pool else None,
    )
    session_id, _ = await _run(adapter, workspace)
    answer = "Aurora" if accepted else "(declined to answer)"
    assert client.texts(UserMessageChunk, session_id) == ([answer] if accepted else [])
    session = adapter.session(session_id)
    admitted = [
        e
        for e in session.handle.events.all_events()
        if isinstance(e, ItemAdmitted) and e.channel == "user_messages"
    ]
    assert [e.source for e in admitted] == [
        "acp",
        "acp:form-answer" if accepted else "user:declined",
    ]
    assert len(models.llms[None].calls) == 2

    client.log.clear()
    await adapter.load_session(str(workspace), session_id)
    # Replay deliberately includes stored declines; live declines remain silent.
    assert client.texts(UserMessageChunk, session_id) == ["push it\n", answer + "\n"]
    conversation = [
        u
        for u in client.updates(session_id)
        if isinstance(u, (AgentMessageChunk, UserMessageChunk))
    ]
    assert [type(u) for u in conversation] == [
        UserMessageChunk,
        AgentMessageChunk,
        UserMessageChunk,
        AgentMessageChunk,
    ]
    if pool:
        updates = client.updates(session_id)
        for entry in session.transcript():
            if entry.role == "user":
                index = next(
                    i
                    for i, u in enumerate(updates)
                    if isinstance(u, UserMessageChunk) and u.content.text == entry.content + "\n"
                )
                assert updates[index + 1].field_meta == {"poolside/inputEventId": entry.item_id}
    # Loading twice must reuse the bridge; subsequent prompts must not double-echo.
    await adapter.load_session(str(workspace), session_id)
    client.log.clear()
    session.admit("from another host", source="tui")
    await adapter.bridge(session_id).flush()
    assert client.texts(UserMessageChunk, session_id) == ["from another host"]


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
