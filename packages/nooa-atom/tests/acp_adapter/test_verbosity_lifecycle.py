# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local picker cancellation belongs to the session and original ACP attachment."""

import asyncio

import pytest
from acp import text_block
from acp.schema import AgentMessageChunk, Implementation
from atom_test_agents import ScriptedModels, cell
from nooa_atom.workspace.settings import settings_path

POOL = Implementation(name="pool", version="1.0.16")


class GatedForms:
    def __init__(self, *, swallow_cancel=False):
        self.entered = [asyncio.Event(), asyncio.Event()]
        self.release = [asyncio.Event(), asyncio.Event()]
        self.cancelled = [asyncio.Event(), asyncio.Event()]
        self.requests = []
        self.swallow_cancel = swallow_cancel

    async def request(self, method, params):
        index = len(self.requests)
        self.requests.append((method, params))
        self.entered[index].set()
        try:
            await self.release[index].wait()
        except asyncio.CancelledError:
            self.cancelled[index].set()
            if not self.swallow_cancel:
                raise
            await self.release[index].wait()
        # Agent questions have an answer field; the local picker has verbosity.
        key = next(iter(params["requestedSchema"]["properties"]))
        return {"action": "accept", "content": {key: "silent" if key == "verbosity" else "dev"}}


async def wait(event):
    await asyncio.wait_for(event.wait(), 5)


async def test_stop_cancels_all_concurrent_local_pickers(
    make_adapter, workspace, client, monkeypatch
):
    adapter = await make_adapter(ScriptedModels(), client_info=POOL)
    session_id = (await adapter.new_session(str(workspace))).session_id
    forms = GatedForms()
    monkeypatch.setattr(client, "ext_method", forms.request)
    first = asyncio.create_task(adapter.prompt(session_id, [text_block("/verbosity")]))
    await wait(forms.entered[0])
    second = asyncio.create_task(adapter.prompt(session_id, [text_block("/verbosity")]))
    await wait(forms.entered[1])
    await adapter.cancel(session_id)
    results = await asyncio.wait_for(asyncio.gather(first, second), 5)
    assert [r.stop_reason for r in results] == ["cancelled", "cancelled"]
    assert all(e.is_set() for e in forms.cancelled)
    assert not settings_path(workspace=workspace).exists()
    assert adapter.bridge(session_id).verbosity == "verbose"
    assert adapter.session(session_id).info.reply_count == 0
    assert session_id not in adapter._asks


async def test_stop_cancels_agent_question_and_local_picker_together(
    make_adapter, workspace, client, monkeypatch
):
    models = ScriptedModels({None: [cell("return_result(NeedInput(question='Which branch?'))")]})
    adapter = await make_adapter(models, client_info=POOL)
    session_id = (await adapter.new_session(str(workspace))).session_id
    forms = GatedForms()
    monkeypatch.setattr(client, "ext_method", forms.request)
    agent_prompt = asyncio.create_task(adapter.prompt(session_id, [text_block("Ask a question")]))
    await wait(forms.entered[0])
    local_prompt = asyncio.create_task(adapter.prompt(session_id, [text_block("/verbosity")]))
    await wait(forms.entered[1])
    await adapter.cancel(session_id)
    results = await asyncio.wait_for(asyncio.gather(agent_prompt, local_prompt), 5)
    assert [r.stop_reason for r in results] == ["cancelled", "cancelled"]
    assert all(e.is_set() for e in forms.cancelled)
    assert not settings_path(workspace=workspace).exists()
    assert session_id not in adapter._asks


@pytest.mark.parametrize("teardown", ["close_session", "delete", "adapter_close", "session_close"])
async def test_teardown_cancels_picker_without_saving(
    make_adapter, workspace, client, monkeypatch, teardown
):
    adapter = await make_adapter(ScriptedModels(), client_info=POOL)
    session_id = (await adapter.new_session(str(workspace))).session_id
    forms = GatedForms()
    monkeypatch.setattr(client, "ext_method", forms.request)
    request = asyncio.create_task(adapter.prompt(session_id, [text_block("/verbosity")]))
    await wait(forms.entered[0])
    if teardown == "close_session":
        await adapter.close_session(session_id)
    elif teardown == "delete":
        await adapter.ext_method("nooa/session/delete", {"sessionId": session_id})
    elif teardown == "adapter_close":
        await adapter.close()
    else:
        await adapter.session(session_id).close()
    response = await asyncio.wait_for(request, 5)
    assert response.stop_reason == "cancelled"
    assert forms.cancelled[0].is_set()
    assert session_id not in adapter._bridges
    assert session_id not in adapter._asks
    assert not settings_path(workspace=workspace).exists()
    assert not any(
        "ACP verbosity: silent" in text for text in client.texts(AgentMessageChunk, session_id)
    )


@pytest.mark.parametrize("action", ["stop", "close_and_reload"])
async def test_late_answer_ignoring_cancel_cannot_change_original_or_new_attachment(
    make_adapter, workspace, client, monkeypatch, action
):
    adapter = await make_adapter(ScriptedModels(), client_info=POOL)
    session_id = (await adapter.new_session(str(workspace))).session_id
    original_bridge = adapter.bridge(session_id)
    forms = GatedForms(swallow_cancel=True)
    monkeypatch.setattr(client, "ext_method", forms.request)
    request = asyncio.create_task(adapter.prompt(session_id, [text_block("/verbosity")]))
    await wait(forms.entered[0])
    if action == "stop":
        await adapter.cancel(session_id)
    else:
        await adapter.close_session(session_id)
        await adapter.load_session(str(workspace), session_id)
        assert adapter.bridge(session_id) is not original_bridge
    await wait(forms.cancelled[0])
    forms.release[0].set()
    response = await asyncio.wait_for(request, 5)
    assert response.stop_reason == "cancelled"
    assert original_bridge.verbosity == adapter.bridge(session_id).verbosity == "verbose"
    assert not settings_path(workspace=workspace).exists()
    assert session_id not in adapter._asks


async def test_external_prompt_cancel_cleans_only_its_client_request(
    make_adapter, workspace, client, monkeypatch
):
    adapter = await make_adapter(ScriptedModels(), client_info=POOL)
    session_id = (await adapter.new_session(str(workspace))).session_id
    forms = GatedForms()
    monkeypatch.setattr(client, "ext_method", forms.request)
    first = asyncio.create_task(adapter.prompt(session_id, [text_block("/verbosity")]))
    await wait(forms.entered[0])
    second = asyncio.create_task(adapter.prompt(session_id, [text_block("/verbosity")]))
    await wait(forms.entered[1])
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert len(adapter._asks[session_id]) == 1
    forms.release[1].set()
    assert (await asyncio.wait_for(second, 5)).stop_reason == "end_turn"
    assert adapter.bridge(session_id).verbosity == "silent"
    assert session_id not in adapter._asks


async def test_external_prompt_cancellation_survives_client_swallowing_cancel(
    make_adapter, workspace, client, monkeypatch
):
    adapter = await make_adapter(ScriptedModels(), client_info=POOL)
    session_id = (await adapter.new_session(str(workspace))).session_id
    forms = GatedForms(swallow_cancel=True)
    monkeypatch.setattr(client, "ext_method", forms.request)
    request = asyncio.create_task(adapter.prompt(session_id, [text_block("/verbosity")]))
    await wait(forms.entered[0])
    request.cancel()
    await wait(forms.cancelled[0])
    forms.release[0].set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(request, 5)
    assert not settings_path(workspace=workspace).exists()
    assert adapter.bridge(session_id).verbosity == "verbose"
    assert session_id not in adapter._asks


async def test_stop_before_agent_form_registration_does_not_open_or_resume_question(
    make_adapter, workspace, client, monkeypatch
):
    models = ScriptedModels({None: [cell("return_result(NeedInput(question='Which branch?'))")]})
    adapter = await make_adapter(models, client_info=POOL)
    session_id = (await adapter.new_session(str(workspace))).session_id
    bridge = adapter.bridge(session_id)
    entered, release = asyncio.Event(), asyncio.Event()
    original_flush = bridge.flush

    async def gated_flush():
        if session_id in adapter._asking and not entered.is_set():
            entered.set()
            await release.wait()
        await original_flush()

    monkeypatch.setattr(bridge, "flush", gated_flush)
    request = asyncio.create_task(adapter.prompt(session_id, [text_block("Ask a question")]))
    await wait(entered)
    await adapter.cancel(session_id)
    release.set()
    assert (await asyncio.wait_for(request, 5)).stop_reason == "cancelled"
    assert not any(entry[0] == "ext" for entry in client.log)
    assert adapter.session(session_id).info.reply_count == 1
    assert not settings_path(workspace=workspace).exists()


@pytest.mark.parametrize("cancel_source", ["caller", "stop"])
async def test_client_exception_during_cancel_is_not_an_ordinary_form_failure(
    make_adapter, workspace, client, monkeypatch, cancel_source
):
    adapter = await make_adapter(ScriptedModels(), client_info=POOL)
    session_id = (await adapter.new_session(str(workspace))).session_id
    entered = asyncio.Event()

    async def failing_cleanup(method, params):
        entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise RuntimeError("transport error during cancellation") from None

    monkeypatch.setattr(client, "ext_method", failing_cleanup)
    request = asyncio.create_task(adapter.prompt(session_id, [text_block("/verbosity")]))
    await wait(entered)
    if cancel_source == "caller":
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(request, 5)
    else:
        await adapter.cancel(session_id)
        assert (await asyncio.wait_for(request, 5)).stop_reason == "cancelled"
    assert not settings_path(workspace=workspace).exists()
    assert adapter.bridge(session_id).verbosity == "verbose"
    assert adapter._pool_forms  # cancellation is not a failed-capability probe
    assert session_id not in adapter._asks
