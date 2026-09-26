# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ACP adapter over the Session layer, driven in-process through a fake client."""

import asyncio

import pytest
from acp import PROTOCOL_VERSION, RequestError
from acp.schema import (
    AgentMessageChunk,
    AvailableCommandsUpdate,
    ToolCallStart,
    UserMessageChunk,
)
from coder_test_agents import (
    BLOCKING_CELL,
    CODER_SPEC,
    CoderModels,
    ScriptedModels,
    cell,
    fresh_events,
    reply,
)
from nooa_coder.acp.server import initialize_response
from nooa_coder.session.store import SessionStore, sessions_root

TIMEOUT = 30
_RESOURCE_NOT_FOUND = -32002


# ---- initialize --------------------------------------------------------------


def test_initialize_advertises_sessions_mcp_and_the_agent():
    response = initialize_response(PROTOCOL_VERSION)
    capabilities = response.agent_capabilities
    assert response.protocol_version == PROTOCOL_VERSION
    assert capabilities is not None and capabilities.load_session is True
    assert capabilities.mcp_capabilities is not None
    assert capabilities.mcp_capabilities.http and capabilities.mcp_capabilities.sse
    sessions = capabilities.session_capabilities
    assert sessions is not None
    assert sessions.list is not None and sessions.close is not None
    # Not routed by the 0.12 library: advertising them would promise a failure.
    assert sessions.delete is None and sessions.resume is None and sessions.fork is None
    assert response.auth_methods == []
    assert response.agent_info is not None and response.agent_info.name == "nooa-coder"


async def test_initialize_keeps_the_client_capabilities(make_adapter):
    from acp.schema import ClientCapabilities, ElicitationCapabilities, ElicitationFormCapabilities

    capabilities = ClientCapabilities(
        elicitation=ElicitationCapabilities(form=ElicitationFormCapabilities())
    )
    adapter = await make_adapter(ScriptedModels(), capabilities=capabilities)
    assert adapter.client_capabilities == capabilities


# ---- new ---------------------------------------------------------------------


async def test_new_session_returns_the_id_and_auto_mode(make_adapter, workspace, sessions_dir):
    adapter = await make_adapter(ScriptedModels())
    response = await adapter.new_session(str(workspace))
    info = SessionStore(sessions_dir).get(response.session_id)
    assert info.host == "acp"
    assert info.workspace == str(workspace)
    assert info.agent == "coder_test_agents:EchoAgent"
    assert response.modes is not None
    assert response.modes.current_mode_id == "auto"
    assert [mode.id for mode in response.modes.available_modes] == ["auto"]


@pytest.mark.parametrize("cwd", ["relative/path", "/does/not/exist"])
async def test_new_session_rejects_a_workspace_that_is_not_an_absolute_directory(make_adapter, cwd):
    adapter = await make_adapter(ScriptedModels())
    with pytest.raises(RequestError) as caught:
        await adapter.new_session(cwd)
    assert caught.value.code == -32602


async def test_new_session_rejects_additional_directories(make_adapter, workspace, tmp_path):
    adapter = await make_adapter(ScriptedModels())
    with pytest.raises(RequestError):
        await adapter.new_session(str(workspace), additional_directories=[str(tmp_path)])


async def test_commands_are_advertised_after_the_new_session_response(
    make_adapter, workspace, client
):
    adapter = await make_adapter(CoderModels(), agent_spec=CODER_SPEC)
    response = await adapter.new_session(str(workspace))
    # Deferred: nothing may reach the client before it knows the session id.
    assert client.log == []
    await client.wait_for(lambda: client.updates(response.session_id, AvailableCommandsUpdate))
    [commands] = client.updates(response.session_id, AvailableCommandsUpdate)
    names = [command.name for command in commands.available_commands]
    assert {"mcp", "skills", "trace-url", "usage"} <= set(names)


async def test_a_change_to_the_commands_is_advertised_again(
    make_adapter, workspace, client, tmp_path
):
    adapter = await make_adapter(CoderModels(), agent_spec=CODER_SPEC)
    response = await adapter.new_session(str(workspace))
    await client.wait_for(lambda: client.updates(response.session_id, AvailableCommandsUpdate))
    skill = tmp_path / "extra-skills" / "shipit"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: shipit\ndescription: Ship it\n---\nShip.\n")
    session = adapter.session(response.session_id)
    session.agent.slash_commands.add_skills_dir(skill.parent)
    await client.wait_for(
        lambda: len(client.updates(response.session_id, AvailableCommandsUpdate)) == 2
    )
    latest = client.updates(response.session_id, AvailableCommandsUpdate)[-1]
    assert "shipit" in [command.name for command in latest.available_commands]


async def test_startup_warnings_are_sent_as_an_agent_message(make_adapter, workspace, client):
    from acp.schema import AcpMcpServer

    adapter = await make_adapter(CoderModels(), agent_spec=CODER_SPEC)
    response = await adapter.new_session(
        str(workspace), mcp_servers=[AcpMcpServer(name="remote", server_id="x", type="acp")]
    )
    await client.wait_for(lambda: client.updates(response.session_id, AgentMessageChunk))
    [text] = client.texts(AgentMessageChunk, response.session_id)
    assert "started with configuration warnings" in text
    assert "'remote'" in text


# ---- load --------------------------------------------------------------------


async def _finished_session(adapter, workspace, models, text="hello"):
    """A session that ran one turn replying "Hi there.", then was closed."""
    models.scripts[None] = [reply("Hi there.")]
    response = await adapter.new_session(str(workspace))
    session = adapter.session(response.session_id)
    await asyncio.wait_for(session.prompt(text, source="acp"), TIMEOUT)
    await adapter.close_session(response.session_id)
    return response.session_id


async def test_load_replays_the_transcript_before_answering(make_adapter, workspace, client):
    models = ScriptedModels()
    adapter = await make_adapter(models)
    session_id = await _finished_session(adapter, workspace, models)
    client.log.clear()

    response = await adapter.load_session(str(workspace), session_id)
    client.log.append(("response", "load", response))

    replay = [
        (type(entry[2]).__name__, entry[2].content.text)
        for entry in client.log
        if entry[0] == "update" and isinstance(entry[2], (UserMessageChunk, AgentMessageChunk))
    ]
    assert replay == [("UserMessageChunk", "hello\n"), ("AgentMessageChunk", "Hi there.\n\n")]
    assert client.log[-1][0] == "response"
    assert response.modes is not None and response.modes.current_mode_id == "auto"
    assert adapter.session(session_id) is not None
    assert adapter.session(session_id).info.host == "acp"


async def test_load_of_a_live_session_attaches_to_the_same_session(make_adapter, workspace, client):
    models = ScriptedModels({None: [reply("Hi there.")]})
    adapter = await make_adapter(models)
    response = await adapter.new_session(str(workspace))
    session = adapter.session(response.session_id)
    await asyncio.wait_for(session.prompt("hello", source="acp"), TIMEOUT)

    await adapter.load_session(str(workspace), response.session_id)
    assert adapter.session(response.session_id) is session
    assert "hello\n" in client.texts(UserMessageChunk, response.session_id)


async def test_loading_a_session_twice_sends_each_later_update_once(
    make_adapter, workspace, client
):
    models = ScriptedModels({None: [reply("Hi there."), reply("Second answer.")]})
    adapter = await make_adapter(models)
    response = await adapter.new_session(str(workspace))
    session = adapter.session(response.session_id)
    await asyncio.wait_for(session.prompt("hello", source="acp"), TIMEOUT)
    await adapter.load_session(str(workspace), response.session_id)
    await adapter.load_session(str(workspace), response.session_id)
    client.log.clear()

    await asyncio.wait_for(session.prompt("again", source="acp"), TIMEOUT)
    await adapter.bridge(response.session_id).flush()
    assert client.texts(AgentMessageChunk, response.session_id) == ["Second answer.\n\n"]


async def test_load_prepares_before_a_requeued_item_runs(
    make_adapter, workspace, client, sessions_dir
):
    """An item left unhandled at close runs on load, and its tool cards reach the client."""
    started, _block = fresh_events()
    first = await make_adapter(ScriptedModels({None: [cell(BLOCKING_CELL)]}))
    response = await first.new_session(str(workspace))
    session = first.session(response.session_id)
    await session.submit("block", source="acp")
    await asyncio.wait_for(started.wait(), TIMEOUT)
    await session.submit("run this later", source="acp")  # queued behind the turn
    await first.close()
    client.log.clear()

    second = await make_adapter(ScriptedModels({None: [cell("print('later')"), reply("Done.")]}))
    await second.load_session(str(workspace), response.session_id)
    await client.wait_for(lambda: "Done.\n\n" in client.texts(AgentMessageChunk))
    assert client.updates(response.session_id, ToolCallStart)


async def test_load_of_an_unknown_session_is_resource_not_found(make_adapter, workspace):
    adapter = await make_adapter(ScriptedModels())
    for session_id in ("no-such-session", "../escape"):
        with pytest.raises(RequestError) as caught:
            await adapter.load_session(str(workspace), session_id)
        assert caught.value.code == _RESOURCE_NOT_FOUND


async def test_load_of_a_session_open_elsewhere_names_the_owner(
    make_adapter, workspace, sessions_dir
):
    import os

    models = ScriptedModels()
    adapter = await make_adapter(models)
    session_id = await _finished_session(adapter, workspace, models)
    with SessionStore(sessions_dir).open(session_id):
        with pytest.raises(RequestError) as caught:
            await adapter.load_session(str(workspace), session_id)
    assert "already open" in str(caught.value)
    assert str(os.getpid()) in str(caught.value)
    assert caught.value.data["sessionId"] == session_id


# ---- list --------------------------------------------------------------------


async def test_list_shows_roots_with_turns_and_their_status(make_adapter, workspace):
    models = ScriptedModels()
    adapter = await make_adapter(models)
    closed_id = await _finished_session(adapter, workspace, models)
    empty = await adapter.new_session(str(workspace))
    models.scripts[None] = [reply("Live.")]
    live = await adapter.new_session(str(workspace))
    await asyncio.wait_for(adapter.session(live.session_id).prompt("hi", source="acp"), 30)

    listed = await adapter.list_sessions(cwd=str(workspace))
    by_id = {session.session_id: session for session in listed.sessions}
    assert set(by_id) == {closed_id, live.session_id}
    assert empty.session_id not in by_id
    assert by_id[closed_id].field_meta == {"dev.nooa/status": "on_disk"}
    assert by_id[live.session_id].field_meta == {"dev.nooa/status": "idle"}
    assert by_id[closed_id].cwd == str(workspace)
    assert by_id[closed_id].title == f"Untitled session [{closed_id[:8]}]"


async def test_list_leaves_out_sessions_open_in_another_process(
    make_adapter, workspace, sessions_dir
):
    models = ScriptedModels()
    adapter = await make_adapter(models)
    session_id = await _finished_session(adapter, workspace, models)
    with SessionStore(sessions_dir).open(session_id):
        assert (await adapter.list_sessions(cwd=str(workspace))).sessions == []
    assert [s.session_id for s in (await adapter.list_sessions()).sessions] == [session_id]


async def test_list_without_cwd_covers_every_workspace(make_adapter, workspace, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    models = ScriptedModels()
    adapter = await make_adapter(models)
    first = await _finished_session(adapter, workspace, models)
    second = await _finished_session(adapter, other, models)
    assert {s.session_id for s in (await adapter.list_sessions()).sessions} == {first, second}
    assert [s.session_id for s in (await adapter.list_sessions(cwd=str(other))).sessions] == [
        second
    ]


async def test_each_workspace_keeps_its_sessions_in_its_own_directory(
    make_adapter, workspace, tmp_path
):
    other = tmp_path / "other"
    other.mkdir()
    models = ScriptedModels()
    adapter = await make_adapter(models)
    first = await _finished_session(adapter, workspace, models)
    second = await _finished_session(adapter, other, models)
    assert SessionStore(workspace / ".nooa" / "sessions").path_for(first).exists()
    assert SessionStore(other / ".nooa" / "sessions").path_for(second).exists()
    assert [s.session_id for s in (await adapter.list_sessions(cwd=str(workspace))).sessions] == [
        first
    ]
    # Each session loads from the store of the cwd it is loaded with.
    await adapter.load_session(str(other), second)
    with pytest.raises(RequestError) as caught:
        await adapter.load_session(str(other), first)
    assert caught.value.code == _RESOURCE_NOT_FOUND


async def test_list_without_cwd_covers_only_workspaces_this_process_serves(
    make_adapter, workspace, tmp_path
):
    """There is no index of every workspace, so a session no request led here is not listed."""
    unseen = tmp_path / "unseen"
    with SessionStore(sessions_root(unseen)).create(workspace=str(unseen)) as handle:
        from coder_test_agents import SessionUserMessage
        from nooa_coder.session.events import TurnEnded

        handle.events.add(SessionUserMessage(content="hello"))
        handle.events.add(TurnEnded(outcome_kind="done"))
    models = ScriptedModels()
    adapter = await make_adapter(models)
    served = await _finished_session(adapter, workspace, models)
    assert [s.session_id for s in (await adapter.list_sessions()).sessions] == [served]
    unseen.mkdir(exist_ok=True)
    assert [s.session_id for s in (await adapter.list_sessions(cwd=str(unseen))).sessions] == [
        handle.id
    ]


async def test_nooa_sessions_dir_holds_the_sessions_of_every_workspace(
    make_adapter, workspace, tmp_path, monkeypatch
):
    shared = tmp_path / "shared"
    monkeypatch.setenv("NOOA_SESSIONS_DIR", str(shared))
    other = tmp_path / "other"
    other.mkdir()
    models = ScriptedModels()
    adapter = await make_adapter(models)
    first = await _finished_session(adapter, workspace, models)
    second = await _finished_session(adapter, other, models)
    assert {info.id for info in SessionStore(shared).list()} == {first, second}
    assert not (workspace / ".nooa" / "sessions").exists()
    assert [s.session_id for s in (await adapter.list_sessions(cwd=str(other))).sessions] == [
        second
    ]
    assert {s.session_id for s in (await adapter.list_sessions()).sessions} == {first, second}


async def test_list_pages_with_a_cursor(make_adapter, workspace, monkeypatch):
    from nooa_coder.acp import listing

    monkeypatch.setattr(listing, "SESSION_PAGE_SIZE", 2)
    models = ScriptedModels()
    adapter = await make_adapter(models)
    ids = {await _finished_session(adapter, workspace, models) for _ in range(3)}
    first = await adapter.list_sessions(cwd=str(workspace))
    assert len(first.sessions) == 2 and first.next_cursor == "2"
    second = await adapter.list_sessions(cwd=str(workspace), cursor=first.next_cursor)
    assert second.next_cursor is None
    assert {s.session_id for s in first.sessions + second.sessions} == ids
    with pytest.raises(RequestError):
        await adapter.list_sessions(cursor="-1")


# ---- close and delete --------------------------------------------------------


async def test_close_releases_the_session(make_adapter, workspace, sessions_dir):
    adapter = await make_adapter(ScriptedModels())
    response = await adapter.new_session(str(workspace))
    await adapter.close_session(response.session_id)
    assert adapter.session(response.session_id) is None
    assert not SessionStore(sessions_dir).is_active(response.session_id)
    with pytest.raises(RequestError) as caught:
        await adapter.close_session(response.session_id)
    assert caught.value.code == _RESOURCE_NOT_FOUND


async def test_delete_extension_removes_the_session(make_adapter, workspace, sessions_dir):
    models = ScriptedModels()
    adapter = await make_adapter(models)
    session_id = await _finished_session(adapter, workspace, models)
    assert await adapter.ext_method("nooa/session/delete", {"sessionId": session_id}) == {}
    assert not SessionStore(sessions_dir).path_for(session_id).exists()
    assert (await adapter.list_sessions()).sessions == []
    with pytest.raises(RequestError) as caught:
        await adapter.ext_method("nooa/session/delete", {"sessionId": session_id})
    assert caught.value.code == _RESOURCE_NOT_FOUND


async def test_unknown_extension_methods_are_not_found(make_adapter):
    adapter = await make_adapter(ScriptedModels())
    with pytest.raises(RequestError) as caught:
        await adapter.ext_method("nooa/unknown", {})
    assert caught.value.code == -32601


# ---- shutdown ----------------------------------------------------------------


async def test_closing_in_order_runs_every_closer_and_raises_the_last_failure():
    from nooa_coder.acp.server import _close_in_order

    ran: list[str] = []

    async def failing(name):
        ran.append(name)
        raise RuntimeError(name)

    with pytest.raises(RuntimeError, match="second") as caught:
        await _close_in_order(
            lambda: failing("first"), None, lambda: failing("second"), lambda: ran.append("sync")
        )
    assert ran == ["first", "second", "sync"]
    assert str(caught.value.__context__) == "first"


async def test_closing_in_order_lets_a_cancellation_through_at_once():
    """Cancelled part-way, the rest are started but not waited for."""
    from nooa_coder.acp.server import _close_in_order

    first_started, second_started, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def first():
        first_started.set()
        await asyncio.Event().wait()

    async def second():
        second_started.set()
        await release.wait()

    task = asyncio.create_task(_close_in_order(first, second))
    await first_started.wait()
    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=2)
    assert task in done and task.cancelled()
    await asyncio.wait_for(second_started.wait(), 2)
    release.set()


async def test_a_closed_empty_session_can_still_be_loaded_until_shutdown(make_adapter, workspace):
    """session/close keeps the file: a client may load the id again in the same run."""
    adapter = await make_adapter(ScriptedModels())
    session_id = (await adapter.new_session(cwd=str(workspace), mcp_servers=[])).session_id
    path = SessionStore(sessions_root(workspace)).path_for(session_id)
    await adapter.close_session(session_id)
    assert path.exists()
    await adapter.load_session(cwd=str(workspace), session_id=session_id)
    await adapter.close()
    assert not path.exists()


async def test_closing_the_adapter_removes_empty_sessions_and_keeps_used_ones(
    make_adapter, workspace
):
    models = ScriptedModels()
    adapter = await make_adapter(models)
    used = await _finished_session(adapter, workspace, models)
    empty = (await adapter.new_session(cwd=str(workspace), mcp_servers=[])).session_id
    store = SessionStore(sessions_root(workspace))
    await adapter.close()
    assert store.path_for(used).exists()
    assert not store.path_for(empty).exists()
