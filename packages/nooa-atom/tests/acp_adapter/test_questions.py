# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit conversational questions and standard ACP form outcomes."""

import asyncio
import json

import pytest
from acp import text_block
from acp.schema import (
    AcceptElicitationResponse,
    AgentMessageChunk,
    CancelElicitationResponse,
    ClientCapabilities,
    DeclineElicitationResponse,
    ElicitationCapabilities,
    ElicitationFormCapabilities,
    Implementation,
    UserMessageChunk,
)
from atom_test_agents import ScriptedModels, cell, reply

from nooa.interactive import FormResponse

TIMEOUT = 30
FORMS = ClientCapabilities(elicitation=ElicitationCapabilities(form=ElicitationFormCapabilities()))


async def _run(adapter, workspace, text="push it"):
    session_id = (await adapter.new_session(str(workspace))).session_id
    response = await asyncio.wait_for(adapter.prompt(session_id, [text_block(text)]), TIMEOUT)
    return session_id, response


def _form(typed=True, options=None):
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


def _inspect_answer():
    return cell(
        "[a] = notification['user_messages']\n"
        "self.v.answer = a\n"
        "return_result(Done(explanation='received', message='Okay.'))"
    )


@pytest.mark.parametrize("pool", [False, True])
@pytest.mark.parametrize("options", [None, ["main", "dev"], ["Yes", "No"]])
async def test_question_never_opens_a_dialog(make_adapter, workspace, client, pool, options):
    models = ScriptedModels(
        {
            None: [
                cell(
                    f"return_result(NeedInput(question='Which?', reason='Safety.', options={options!r}))"
                ),
                _inspect_answer(),
            ]
        }
    )
    adapter = await make_adapter(
        models,
        capabilities=FORMS,
        client_info=Implementation(name="pool", version="1") if pool else None,
    )
    session_id, response = await _run(adapter, workspace)
    assert response.stop_reason == "end_turn"
    assert not any(e[0] in ("elicitation", "permission", "ext") for e in client.log)
    assert "Safety." in client.texts(AgentMessageChunk, session_id)[-1]
    assert len(models.llms[None].calls) == 1
    await adapter.prompt(session_id, [text_block("unlisted answer")])
    assert adapter.session(session_id)._agent.v.answer == "unlisted answer"


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
async def test_standard_accept_is_structured_validated_and_echoed_once(
    make_adapter, workspace, client, typed, options, content, expected
):
    models = ScriptedModels({None: [_form(typed, options), _inspect_answer()]})
    client.elicitation_answers = [
        AcceptElicitationResponse(
            action="accept",
            content=content,
            field_meta={"target": {"source": "user"}},
            **{"_meta": {"trace": "reply"}},
        )
    ]
    adapter = await make_adapter(models, capabilities=FORMS)
    session_id, response = await _run(adapter, workspace)
    assert response.stop_reason == "end_turn"
    session = adapter.session(session_id)
    answer = session._agent.v.answer
    assert isinstance(answer, FormResponse) and answer.action == "accept"
    assert answer.content == expected
    [echo] = client.texts(UserMessageChunk, session_id)
    assert json.loads(echo) == {"action": "accept", "content": expected}
    [(_, _, mode)] = [e for e in client.log if e[0] == "elicitation"]
    assert list(mode.requested_schema.properties) == (list(content) if typed else ["answer"])
    assert not any(e[0] == "permission" for e in client.log)
    assert len(models.llms[None].calls) == 2
    assert [e.content for e in session.transcript() if e.role == "user"] == ["push it", echo]
    client.log.clear()
    await adapter.load_session(str(workspace), session_id)
    assert client.texts(UserMessageChunk, session_id) == ["push it\n", echo + "\n"]
    assert not any(e[0] == "elicitation" for e in client.log)


@pytest.mark.parametrize(
    "action,response_model",
    [
        ("decline", DeclineElicitationResponse),
        ("cancel", CancelElicitationResponse),
    ],
)
@pytest.mark.parametrize("metadata", [False, True])
async def test_standard_decline_and_cancel_remain_distinct(
    make_adapter, workspace, client, action, response_model, metadata
):
    models = ScriptedModels({None: [_form(), _inspect_answer()]})
    client.elicitation_answers = [
        response_model(action=action, **({"_meta": {"trace": "reply"}} if metadata else {}))
    ]
    adapter = await make_adapter(models, capabilities=FORMS)
    session_id, _ = await _run(adapter, workspace)
    session = adapter.session(session_id)
    assert session._agent.v.answer == FormResponse(action=action)
    [echo] = client.texts(UserMessageChunk, session_id)
    assert json.loads(echo) == {"action": action, "content": None}
    from nooa_atom.session.events import ItemAdmitted

    [event] = [
        e
        for e in session.handle.events.all_events()
        if isinstance(e, ItemAdmitted) and e.source == "acp:form-answer"
    ]
    assert json.loads(event.item_json)["action"] == action
    client.log.clear()
    await adapter.load_session(str(workspace), session_id)
    assert json.loads(client.texts(UserMessageChunk, session_id)[-1])["action"] == action


@pytest.mark.parametrize(
    "typed,content",
    [
        (False, None),
        (False, {}),
        (False, {"answer": ""}),
        (False, {"answer": " "}),
        (False, {"answer": 2}),
        (False, {"answer": "unlisted"}),
        (True, {"target": "prod"}),
        (True, {"target": "prod", "replicas": "bad", "dry_run": False}),
    ],
)
async def test_invalid_standard_content_is_not_admitted(
    make_adapter, workspace, client, typed, content
):
    models = ScriptedModels({None: [_form(typed, ["main", "dev"] if not typed else None)]})
    client.elicitation_answers = [AcceptElicitationResponse(action="accept", content=content)]
    adapter = await make_adapter(models, capabilities=FORMS)
    session_id, response = await _run(adapter, workspace)
    assert response.stop_reason == "end_turn"
    assert client.updates(session_id, UserMessageChunk) == []
    assert len(models.llms[None].calls) == 1
    assert "not accepted" in client.texts(AgentMessageChunk, session_id)[-1]


async def test_stop_open_standard_form_never_injects_an_answer(make_adapter, workspace, client):
    models = ScriptedModels({None: [_form()]})
    client.elicitation_gate = asyncio.Event()
    adapter = await make_adapter(models, capabilities=FORMS)
    session_id = (await adapter.new_session(str(workspace))).session_id
    pending = asyncio.create_task(adapter.prompt(session_id, [text_block("go")]))
    await asyncio.wait_for(client.elicitation_started.wait(), TIMEOUT)
    await adapter.cancel(session_id)
    assert (await asyncio.wait_for(pending, TIMEOUT)).stop_reason == "cancelled"
    assert len(models.llms[None].calls) == 1
    assert client.updates(session_id, UserMessageChunk) == []


@pytest.mark.parametrize("failure", ["unsupported", "client-error"])
async def test_fallback_preserves_form_intent_and_does_not_validate_raw_text(
    make_adapter, workspace, client, failure
):
    ask = _form()
    models = ScriptedModels({None: [ask, _inspect_answer()]})

    async def broken(*args, **kwargs):
        raise RuntimeError("client error")

    if failure == "client-error":
        client.create_elicitation = broken
    adapter = await make_adapter(models, capabilities=None if failure == "unsupported" else FORMS)
    session_id, response = await _run(adapter, workspace)
    assert response.stop_reason == "end_turn"
    [text] = client.texts(AgentMessageChunk, session_id)
    assert "Explicit form request" in text and "unvalidated" in text and '"replicas"' in text
    assert len(models.llms[None].calls) == 1
    client.log.clear()
    await adapter.load_session(str(workspace), session_id)
    assert "Explicit form request" in client.texts(AgentMessageChunk, session_id)[-1]
    assert "unvalidated" in client.texts(AgentMessageChunk, session_id)[-1]
    assert len(models.llms[None].calls) == 1
    await adapter.prompt(session_id, [text_block('{"target":"prod","replicas":3,"dry_run":false}')])
    assert isinstance(adapter.session(session_id)._agent.v.answer, str)


async def test_done_message_is_sent_before_prompt_completion(make_adapter, workspace, client):
    models = ScriptedModels({None: [reply("All set.")]})
    adapter = await make_adapter(models)
    session_id, response = await _run(adapter, workspace)
    assert response.stop_reason == "end_turn"
    assert client.texts(AgentMessageChunk, session_id) == ["All set.\n\n"]


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


@pytest.mark.parametrize("pool", [False, True])
async def test_stop_wins_if_client_swallows_cancellation(make_adapter, workspace, client, pool):
    started = asyncio.Event()

    async def late_accept(*args, **kwargs):
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            content = {
                "target": "prod",
                "replicas": "3",
                "dry_run": "no",
            }
            return (
                {"action": "accept", "content": content}
                if pool
                else AcceptElicitationResponse(action="accept", content=content)
            )

    if pool:
        client.ext_method = late_accept
    else:
        client.create_elicitation = late_accept
    models = ScriptedModels({None: [_form()]})
    adapter = await make_adapter(
        models,
        capabilities=FORMS,
        client_info=Implementation(name="pool", version="1") if pool else None,
    )
    sid = (await adapter.new_session(str(workspace))).session_id
    pending = asyncio.create_task(adapter.prompt(sid, [text_block("go")]))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    await adapter.cancel(sid)
    assert (await asyncio.wait_for(pending, TIMEOUT)).stop_reason == "cancelled"
    assert len(models.llms[None].calls) == 1
    assert client.updates(sid, UserMessageChunk) == []


@pytest.mark.parametrize("pool", [False, True])
async def test_domain_invalid_text_is_delivered_without_form_retry(
    make_adapter, workspace, client, pool
):
    models = ScriptedModels({None: [_form(), _inspect_answer()]})
    content = {"target": "prod", "replicas": "not a number", "dry_run": "anything"}
    if pool:
        client.ext_answers = [{"action": "accept", "content": content}]
    else:
        client.elicitation_answers = [AcceptElicitationResponse(action="accept", content=content)]
    adapter = await make_adapter(
        models,
        capabilities=FORMS,
        client_info=Implementation(name="pool", version="1") if pool else None,
    )
    sid, _ = await _run(adapter, workspace)
    assert adapter.session(sid)._agent.v.answer.content == content
    assert len([e for e in client.log if e[0] in ("ext", "elicitation")]) == 1
    assert len(models.llms[None].calls) == 2


@pytest.mark.parametrize("pool", [False, True])
async def test_superseded_dialog_cannot_admit_stale_accept(make_adapter, workspace, client, pool):
    gate = asyncio.Event()
    started = asyncio.Event()

    async def accept_later(*args, **kwargs):
        started.set()
        await gate.wait()
        content = {
            "target": "prod",
            "replicas": "3",
            "dry_run": "no",
        }
        return (
            {"action": "accept", "content": content}
            if pool
            else AcceptElicitationResponse(action="accept", content=content)
        )

    if pool:
        client.ext_method = accept_later
    else:
        client.create_elicitation = accept_later
    models = ScriptedModels({None: [_form(), reply("Never mind.")]})
    adapter = await make_adapter(
        models,
        capabilities=FORMS,
        client_info=Implementation(name="pool", version="1") if pool else None,
    )
    sid = (await adapter.new_session(str(workspace))).session_id
    pending = asyncio.create_task(adapter.prompt(sid, [text_block("go")]))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    assert (await adapter.prompt(sid, [text_block("never mind")])).stop_reason == "end_turn"
    gate.set()
    assert (await asyncio.wait_for(pending, TIMEOUT)).stop_reason == "end_turn"
    assert len(models.llms[None].calls) == 2
    assert client.updates(sid, UserMessageChunk) == []


@pytest.mark.parametrize("pool", [False, True])
async def test_stop_during_pre_dialog_flush_never_opens_dialog(
    make_adapter, workspace, client, monkeypatch, pool
):
    models = ScriptedModels({None: [_form()]})
    adapter = await make_adapter(
        models,
        capabilities=FORMS,
        client_info=Implementation(name="pool", version="1") if pool else None,
    )
    sid = (await adapter.new_session(str(workspace))).session_id
    bridge = adapter.bridge(sid)
    original = bridge.flush
    entered, release = asyncio.Event(), asyncio.Event()

    async def paused_flush():
        if sid in adapter._asking and not entered.is_set():
            entered.set()
            await release.wait()
        await original()

    monkeypatch.setattr(bridge, "flush", paused_flush)
    pending = asyncio.create_task(adapter.prompt(sid, [text_block("go")]))
    await asyncio.wait_for(entered.wait(), TIMEOUT)
    await adapter.cancel(sid)
    try:
        done, _ = await asyncio.wait({pending}, timeout=1)
        assert pending in done
        assert pending.result().stop_reason == "cancelled"
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
    assert not any(e[0] in ("ext", "elicitation") for e in client.log)
    assert len(models.llms[None].calls) == 1


@pytest.mark.parametrize("pool", [False, True])
@pytest.mark.parametrize(
    "payload", [{}, {"action": "unknown"}, {"action": "decline", "content": {"x": "y"}}]
)
async def test_malformed_transport_envelope_fails_closed_once(
    make_adapter, workspace, client, pool, payload
):
    async def malformed(*args, **kwargs):
        return payload

    if pool:
        client.ext_method = malformed
    else:
        client.create_elicitation = malformed
    models = ScriptedModels({None: [_form()]})
    adapter = await make_adapter(
        models,
        capabilities=FORMS,
        client_info=Implementation(name="pool", version="1") if pool else None,
    )
    sid, response = await _run(adapter, workspace)
    assert response.stop_reason == "end_turn"
    assert not client.texts(UserMessageChunk, sid)
    assert len(models.llms[None].calls) == 1
    assert sum("No automatic retry" in t for t in client.texts(AgentMessageChunk, sid)) == 1


@pytest.mark.parametrize("pool", [False, True])
@pytest.mark.parametrize("operation", ["close", "delete", "teardown", "stop", "session-close"])
@pytest.mark.parametrize("swallow", [False, True])
async def test_teardown_settles_open_form_without_waiting_for_client(
    make_adapter, workspace, client, pool, operation, swallow
):
    entered, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def blocked_client(*args, **kwargs):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            if not swallow:
                raise
            await release.wait()
        return {"action": "accept", "content": {"answer": "late"}}

    if pool:
        client.ext_method = blocked_client
    else:
        client.create_elicitation = blocked_client
    models = ScriptedModels({None: [_form(False)]})
    adapter = await make_adapter(
        models,
        capabilities=FORMS,
        client_info=Implementation(name="pool", version="1") if pool else None,
    )
    sid = (await adapter.new_session(str(workspace))).session_id
    session = adapter.session(sid)
    pending = asyncio.create_task(adapter.prompt(sid, [text_block("go")]))
    await asyncio.wait_for(entered.wait(), TIMEOUT)
    try:
        if operation == "close":
            await adapter.close_session(sid)
        elif operation == "delete":
            await adapter.ext_method("nooa/session/delete", {"sessionId": sid})
        elif operation == "teardown":
            await adapter.close()
        elif operation == "session-close":
            await session.close()
        else:
            await adapter.cancel(sid)
        done, _ = await asyncio.wait({pending}, timeout=1)
        assert pending in done, "teardown left ACP prompt waiting on client"
        assert pending.result().stop_reason == "cancelled"
        assert sid not in adapter._asks and sid not in adapter._asking
        assert cancelled.is_set()
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
    assert len(models.llms[None].calls) == 1
    assert client.updates(sid, UserMessageChunk) == []
    assert not adapter._asks and not adapter._asking and not adapter._ask_stops


async def test_caller_cancel_settles_without_cancellation_swallowing_client(
    make_adapter, workspace, client
):
    entered, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def stuck(*args, **kwargs):
        entered.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.set()
            await release.wait()
        return {"action": "accept", "content": {"answer": "late"}}

    client.create_elicitation = stuck
    models = ScriptedModels({None: [_form(False)]})
    adapter = await make_adapter(models, capabilities=FORMS)
    sid = (await adapter.new_session(str(workspace))).session_id
    pending = asyncio.create_task(adapter.prompt(sid, [text_block("go")]))
    await asyncio.wait_for(entered.wait(), TIMEOUT)
    pending.cancel()
    try:
        done, _ = await asyncio.wait({pending}, timeout=1)
        assert pending in done and pending.cancelled()
        await asyncio.wait_for(cancelled.wait(), TIMEOUT)
        assert not adapter._asks and not adapter._asking
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
    assert len(models.llms[None].calls) == 1
    assert client.updates(sid, UserMessageChunk) == []


@pytest.mark.parametrize("pool", [False, True])
@pytest.mark.parametrize("admission", ["prompt", "steer"])
@pytest.mark.parametrize("operation", ["stop", "detach", "close", "accept"])
async def test_new_dialog_revokes_old_ownership_before_installing_slots(
    make_adapter, workspace, client, pool, admission, operation
):
    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    old_cancelled = asyncio.Event()
    calls = []

    async def dialog(*args, **kwargs):
        index = len(calls)
        calls.append(index)
        entered[index].set()
        try:
            await release[index].wait()
        except asyncio.CancelledError:
            if index == 0:
                old_cancelled.set()
                await release[index].wait()  # deliberately cancellation-resistant
            else:
                raise
        return {"action": "accept", "content": {"answer": "old" if index == 0 else "new"}}

    if pool:
        client.ext_method = dialog
    else:
        client.create_elicitation = dialog
    models = ScriptedModels({None: [_form(False), _form(False), _inspect_answer()]})
    adapter = await make_adapter(
        models,
        capabilities=FORMS,
        client_info=Implementation(name="pool", version="1") if pool else None,
    )
    sid = (await adapter.new_session(str(workspace))).session_id
    session = adapter.session(sid)
    first = asyncio.create_task(adapter.prompt(sid, [text_block("first")]))
    await asyncio.wait_for(entered[0].wait(), TIMEOUT)
    old_need = adapter._asking[sid]
    if admission == "prompt":
        second = asyncio.create_task(adapter.prompt(sid, [text_block("second")]))
    else:
        receipt = await session.steer("second", source="acp")
        second = asyncio.create_task(adapter._finish(session, adapter.bridge(sid), receipt.item_id))
    try:
        await asyncio.wait_for(entered[1].wait(), TIMEOUT)
        await asyncio.wait_for(old_cancelled.wait(), TIMEOUT)
        assert (await asyncio.wait_for(first, 1)).stop_reason == "cancelled"
        new_need = adapter._asking[sid]
        assert new_need is not old_need
        new_task, new_stop = adapter._asks[sid], adapter._ask_stops[sid]
        release[0].set()
        # Await the detached old transport to check its cleanup cannot touch new slots.
        await asyncio.gather(*list(adapter._detached_asks), return_exceptions=True)
        assert adapter._asking[sid] is new_need
        assert adapter._asks[sid] is new_task and adapter._ask_stops[sid] is new_stop
        assert client.updates(sid, UserMessageChunk) == []
        if operation == "stop":
            await adapter.cancel(sid)
        elif operation == "detach":
            adapter._detach(adapter.bridge(sid))
        elif operation == "close":
            await adapter.close_session(sid)
        else:
            release[1].set()
        result = await asyncio.wait_for(second, 1)
        assert result.stop_reason == ("end_turn" if operation == "accept" else "cancelled")
        if operation == "accept":
            assert session._agent.v.answer.content == {"answer": "new"}
        else:
            assert client.updates(sid, UserMessageChunk) == []
        assert sid not in adapter._asks and sid not in adapter._asking
        assert sid not in adapter._ask_stops and sid not in adapter._form_stops
    finally:
        for gate in release:
            gate.set()
        await asyncio.gather(first, second, return_exceptions=True)


@pytest.mark.parametrize("pool", [False, True])
@pytest.mark.parametrize("operation", ["deliver", "fail", "stop", "close"])
async def test_form_failure_uses_ordered_bridge_and_revocable_flush(
    make_adapter, workspace, client, pool, operation
):
    from acp import update_agent_message

    entered, release = asyncio.Event(), asyncio.Event()
    original = client.session_update
    adapter = await make_adapter(
        ScriptedModels({None: [_form(False)]}),
        capabilities=FORMS,
        client_info=Implementation(name="pool", version="1") if pool else None,
    )
    sid = (await adapter.new_session(str(workspace))).session_id
    bridge = adapter.bridge(sid)

    async def ordered_send(session_id, update, **kwargs):
        text = getattr(getattr(update, "content", None), "text", "")
        if text.startswith("before failure"):
            entered.set()
            await release.wait()
        if operation == "fail" and "No automatic retry" in text:
            raise RuntimeError("notice transport failed")
        await original(session_id, update, **kwargs)

    async def malformed(*args, **kwargs):
        bridge.publish(update_agent_message(text_block("before failure")))
        return {"action": "accept", "content": {"answer": 42}}

    client.session_update = ordered_send
    if pool:
        client.ext_method = malformed
    else:
        client.create_elicitation = malformed
    pending = asyncio.create_task(adapter.prompt(sid, [text_block("go")]))
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), TIMEOUT)
        if operation == "stop":
            await adapter.cancel(sid)
        elif operation == "close":
            # Close drains the independent update pump; form ownership must settle
            # before that transport is released, not wait for the close RPC.
            closing = asyncio.create_task(adapter.close_session(sid))
        else:
            # A direct send would overtake the blocked predecessor.
            assert not any("No automatic retry" in t for t in client.texts(AgentMessageChunk, sid))
            release.set()
        if operation == "fail":
            with pytest.raises(RuntimeError, match="notice transport failed"):
                await asyncio.wait_for(pending, 1)
            await bridge.flush()  # error marker resets the pump for subsequent updates
            bridge.publish(update_agent_message(text_block("after failure")))
            await bridge.flush()
            assert client.texts(AgentMessageChunk, sid)[-1] == "after failure\n\n"
        else:
            result = await asyncio.wait_for(pending, 1)
            assert result.stop_reason == ("end_turn" if operation == "deliver" else "cancelled")
        if operation == "deliver":
            texts = client.texts(AgentMessageChunk, sid)
            assert texts[-2] == "before failure\n\n"
            assert "No automatic retry" in texts[-1]
        assert client.updates(sid, UserMessageChunk) == []
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        if closing is not None:
            await asyncio.wait_for(closing, TIMEOUT)
