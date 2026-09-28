# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""/recover over ACP: list the sessions marked in use, and fork one."""

import asyncio

from acp import text_block
from acp.schema import AgentMessageChunk, AvailableCommandsUpdate
from coder_test_agents import ScriptedModels
from nooa_coder.session.store import SessionStore

TIMEOUT = 30


def _stale(store, workspace, title, session_id=None):
    """A session whose lock record names another machine, as a crash there leaves it."""
    with store.create(workspace=str(workspace), session_id=session_id) as handle:
        handle.set_title(title, user_set=True)
    store.path_for(handle.id).with_suffix(".lock").write_text("4242 other-box")
    return handle.id


async def _recover(adapter, client, session_id, args=""):
    text = f"/recover {args}".strip()
    response = await asyncio.wait_for(adapter.prompt(session_id, [text_block(text)]), TIMEOUT)
    assert response.stop_reason == "end_turn"
    return client.texts(AgentMessageChunk, session_id)[-1]


async def test_recover_is_advertised(make_adapter, workspace, client):
    adapter = await make_adapter(ScriptedModels())
    response = await adapter.new_session(str(workspace))
    await client.wait_for(lambda: client.updates(response.session_id, AvailableCommandsUpdate))
    [commands] = client.updates(response.session_id, AvailableCommandsUpdate)
    assert "recover" in [command.name for command in commands.available_commands]


async def test_without_an_argument_it_lists_the_sessions_in_use(
    make_adapter, workspace, sessions_dir, client
):
    store = SessionStore(sessions_dir)
    adapter = await make_adapter(ScriptedModels())
    session_id = (await adapter.new_session(str(workspace))).session_id
    assert await _recover(adapter, client, session_id) == (
        "No session in this workspace is marked in use."
    )

    stale = _stale(store, workspace, "Fix the parser")
    text = await _recover(adapter, client, session_id)
    assert "```text" in text
    assert "Fix the parser" in text and stale[:8] in text and "pid 4242 on other-box" in text
    # This session is open here: it is not listed as in use.
    assert session_id[:8] not in text


async def test_recover_by_id_prefix_forks_the_session(
    make_adapter, workspace, sessions_dir, client
):
    store = SessionStore(sessions_dir)
    stale = _stale(store, workspace, "Fix the parser")
    adapter = await make_adapter(ScriptedModels())
    session_id = (await adapter.new_session(str(workspace))).session_id

    text = await _recover(adapter, client, session_id, stale[:8])

    assert text == (
        'Recovered "Fix the parser" as "Fix the parser (recovered)". Open /resume to continue it.'
    )
    [fork] = [info for info in store.list() if info.forked_from == stale]
    assert fork.title == "Fix the parser (recovered)"


async def test_recover_by_exact_title(make_adapter, workspace, sessions_dir, client):
    store = SessionStore(sessions_dir)
    stale = _stale(store, workspace, "Fix the parser")
    adapter = await make_adapter(ScriptedModels())
    session_id = (await adapter.new_session(str(workspace))).session_id

    text = await _recover(adapter, client, session_id, '"Fix the parser"')

    assert text.startswith('Recovered "Fix the parser"')
    assert [info.forked_from for info in store.list()].count(stale) == 1


async def test_a_session_not_in_use_is_not_forked(make_adapter, workspace, sessions_dir, client):
    store = SessionStore(sessions_dir)
    with store.create(workspace=str(workspace)) as handle:
        handle.set_title("Closed cleanly", user_set=True)
    adapter = await make_adapter(ScriptedModels())
    session_id = (await adapter.new_session(str(workspace))).session_id

    text = await _recover(adapter, client, session_id, handle.id[:8])

    assert text == '"Closed cleanly" is not in use. Open /resume to continue it.'
    assert all(info.forked_from is None for info in store.list())


async def test_an_ambiguous_argument_lists_the_matches(
    make_adapter, workspace, sessions_dir, client
):
    store = SessionStore(sessions_dir)
    first = _stale(store, workspace, "Same title", session_id="aaaa1111-first")
    second = _stale(store, workspace, "Same title", session_id="aaaa2222-second")
    adapter = await make_adapter(ScriptedModels())
    session_id = (await adapter.new_session(str(workspace))).session_id

    for argument in ('"Same title"', "aaaa"):
        text = await _recover(adapter, client, session_id, argument)
        assert "matches 2 sessions" in text
        assert first in text and second in text
    assert all(info.forked_from is None for info in store.list())


async def test_an_argument_that_matches_nothing_says_so(make_adapter, workspace, client):
    adapter = await make_adapter(ScriptedModels())
    session_id = (await adapter.new_session(str(workspace))).session_id
    text = await _recover(adapter, client, session_id, "nothing-like-this")
    assert text == 'No session in this workspace matches "nothing-like-this".'


async def test_a_damaged_file_is_recovered_and_the_reply_says_what_was_lost(
    make_adapter, workspace, sessions_dir, client
):
    store = SessionStore(sessions_dir)
    with store.create(workspace=str(workspace)) as handle:
        handle.set_title("Damaged work", user_set=True)
        for index in range(600):
            handle.set_mode("ask" if index % 2 else "auto")
    path = store.path_for(handle.id)
    with open(path, "r+b") as file:
        file.truncate(path.stat().st_size - 3 * 4096 - 1000)
    path.with_suffix(".lock").write_text("4242 other-box")
    adapter = await make_adapter(ScriptedModels())
    session_id = (await adapter.new_session(str(workspace))).session_id

    text = await _recover(adapter, client, session_id, handle.id[:8])

    assert text.startswith(
        'Recovered "Damaged work" as "Damaged work (recovered)". Open /resume to continue it. '
        "The original file could not be copied whole, so it was copied event by event: "
    )
    assert "could not be read" in text
