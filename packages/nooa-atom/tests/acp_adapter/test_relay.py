# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""With NEMO_RELAY_PLUGIN_CONFIG_PATH set, an ACP session's turns go through NeMo Relay."""

import asyncio
import json
import socket
from contextlib import asynccontextmanager

import pytest
from acp import text_block
from atom_test_agents import ATOM_SPEC, PUBLIC_ATOM_SPEC, AtomModels, CellLLM, cell, reply
from nooa_atom.acp.worker import serve_worker
from nooa_atom.session import relay
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore
from nooa_atom.session.tracing import relay_activated

from nooa.unifiedllm import LLMResponse

nemo_relay = pytest.importorskip("nemo_relay")

TIMEOUT = 30


@pytest.fixture
def events():
    seen: list = []
    nemo_relay.subscribers.register("atom-relay-test", seen.append)
    yield seen
    nemo_relay.subscribers.deregister("atom-relay-test")


@pytest.fixture
def plugins(tmp_path):
    path = tmp_path / "plugins.toml"
    path.write_text("version = 1\n")
    return path


async def _one_turn(adapter, workspace) -> str:
    session_id = (await adapter.new_session(str(workspace))).session_id
    await asyncio.wait_for(adapter.prompt(session_id, [text_block("hi")]), TIMEOUT)
    await adapter.close_session(session_id)
    await nemo_relay.subscribers.flush_async()
    return session_id


async def test_a_turn_is_one_agent_scope_with_its_metadata_on_every_event(
    atom_adapter, workspace, plugins, monkeypatch, events
):
    monkeypatch.setenv(relay.CONFIG_ENV, str(plugins))
    monkeypatch.setenv(relay.METADATA_ENV, json.dumps({"run.name": "r1"}))
    async with relay.activated():
        adapter = await atom_adapter([cell("x = 1"), reply("Hello.\n")])
        session_id = await _one_turn(adapter, workspace)

    expected = {"session_id": session_id, "agent_kind": "atom", "run.name": "r1"}
    start, end = (e for e in events if e.name == "atom")
    assert (start.scope_category, end.scope_category) == ("start", "end")
    assert expected.items() <= start.metadata.items()
    inside = {start.uuid}
    for event in events[events.index(start) + 1 : events.index(end)]:
        assert event.parent_uuid in inside, event.name
        assert expected.items() <= event.metadata.items()
        inside.add(event.uuid)
    assert events[0] is start and events[-1] is end
    assert {"llm", "tool"} <= {e.category for e in events}


async def test_each_turn_is_its_own_agent_scope_of_one_session(
    atom_adapter, workspace, plugins, monkeypatch, events
):
    monkeypatch.setenv(relay.CONFIG_ENV, str(plugins))
    async with relay.activated():
        adapter = await atom_adapter([reply("One.\n"), reply("Two.\n")])
        session_id = (await adapter.new_session(str(workspace))).session_id
        for text in ("first", "second"):
            await asyncio.wait_for(adapter.prompt(session_id, [text_block(text)]), TIMEOUT)
        await adapter.close_session(session_id)
        await nemo_relay.subscribers.flush_async()

    scopes = [e for e in events if e.name == "atom"]
    assert [e.scope_category for e in scopes] == ["start", "end", "start", "end"]
    assert scopes[0].uuid != scopes[2].uuid
    assert not {e.parent_uuid for e in scopes} & {e.uuid for e in scopes}
    assert all(e.metadata["session_id"] == session_id for e in scopes)


async def test_the_hosts_metadata_cannot_rename_the_session_or_its_agent(
    atom_adapter, workspace, plugins, monkeypatch, events
):
    monkeypatch.setenv(relay.CONFIG_ENV, str(plugins))
    metadata = {"session_id": "batch", "agent_kind": "evaluation", "run": "r1"}
    monkeypatch.setenv(relay.METADATA_ENV, json.dumps(metadata))
    async with relay.activated():
        adapter = await atom_adapter([reply("Hello.\n")])
        session_id = await _one_turn(adapter, workspace)

    expected = {"session_id": session_id, "agent_kind": "atom", "run": "r1"}
    assert events
    assert all(expected.items() <= event.metadata.items() for event in events)


async def test_without_the_variable_nothing_goes_through_relay(
    atom_adapter, workspace, monkeypatch, events
):
    monkeypatch.delenv(relay.CONFIG_ENV, raising=False)
    async with relay_activated():
        adapter = await atom_adapter([reply("Hello.\n")])
        await _one_turn(adapter, workspace)

    assert events == []


async def test_a_missing_configuration_stops_the_server(tmp_path, monkeypatch):
    monkeypatch.setenv(relay.CONFIG_ENV, str(tmp_path / "missing.toml"))
    with pytest.raises(FileNotFoundError, match=relay.CONFIG_ENV):
        async with relay.activated():
            pass


async def test_metadata_that_relay_would_drop_stops_the_server(plugins, monkeypatch):
    monkeypatch.setenv(relay.CONFIG_ENV, str(plugins))
    monkeypatch.setenv(relay.METADATA_ENV, json.dumps({"run/name": "r1"}))
    with pytest.raises(ValueError, match=relay.METADATA_ENV):
        async with relay.activated():
            pass


async def test_a_child_that_a_traced_turn_delegates_to_is_in_its_trace(
    workspace, sessions_dir, plugins, monkeypatch, events
):
    monkeypatch.setenv(relay.CONFIG_ENV, str(plugins))
    models = AtomModels()
    models.scripts[None] = [
        cell(
            "await self.delegate('fix', 'fix it', name='fix')\nreturn_result(Done(explanation='ok'))"
        )
    ]
    models.scripts["fix"] = [
        cell(
            "return_result(Done(explanation='fixed', result=TaskResult(solution_description='s',"
            " evidence='e', how_to_verify='h', report='r')))"
        )
    ]

    async def prepare(session):
        relay.trace_session(session)

    async with relay.activated():
        registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=models)
        options = SessionOptions(
            workspace=workspace, agent_spec=ATOM_SPEC, sessions_dir=sessions_dir
        )
        root = await registry.create(options, prepare=prepare)
        await asyncio.wait_for(root.prompt("go"), TIMEOUT)
        await registry.close_all()
        await nemo_relay.subscribers.flush_async()

    assert [(e.name, e.scope_category) for e in events if e.name == "atom"] == [
        ("atom", "start"), ("atom", "start"), ("atom", "end"), ("atom", "end")
    ]  # fmt: skip
    assert events[0].uuid == events[-1].uuid
    inside = {events[0].uuid}
    for event in events[1:-1]:
        assert event.parent_uuid in inside, event.name
        assert event.metadata["session_id"] == root.id
        inside.add(event.uuid)
    assert len([e for e in events if e.category == "llm" and e.scope_category == "start"]) == 2
    child = next(e for e in events[1:] if e.name == "atom" and e.scope_category == "start")
    child_id = child.metadata["subagent_session_id"]
    assert child_id not in {None, root.id}
    in_child = {child.uuid}
    for event in events[events.index(child) + 1 :]:
        if event.parent_uuid in in_child or event.uuid in in_child:
            in_child.add(event.uuid)
            assert event.metadata["subagent_session_id"] == child_id, event.name
        else:
            assert "subagent_session_id" not in event.metadata, event.name


async def test_a_worker_closes_the_plugin_host_before_its_socket(plugins, monkeypatch):
    monkeypatch.setenv(relay.CONFIG_ENV, str(plugins))
    order = []

    @asynccontextmanager
    async def activated():
        yield
        order.append("host closed")

    class Agent:
        async def close(self):
            order.append("agent closed")

    monkeypatch.setattr(relay, "activated", activated)
    ours, theirs = socket.socketpair()
    worker = asyncio.create_task(serve_worker(theirs, id_base=1 << 32, agent=Agent()))
    reader, writer = await asyncio.open_unix_connection(sock=ours)
    writer.write_eof()
    assert await asyncio.wait_for(reader.read(), TIMEOUT) == b""
    order.append("socket closed")
    writer.close()
    await asyncio.wait_for(worker, TIMEOUT)
    assert order == ["agent closed", "host closed", "socket closed"]


async def test_a_child_that_acp_loads_directly_is_a_trace_of_its_own(
    workspace, sessions_dir, plugins, monkeypatch, events
):
    monkeypatch.setenv(relay.CONFIG_ENV, str(plugins))
    models = AtomModels()
    models.scripts["fix"] = [cell("return_result(Done(explanation='fixed'))")]
    options = SessionOptions(workspace=workspace, agent_spec=ATOM_SPEC, sessions_dir=sessions_dir)
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=models)
    root = await registry.create(options)
    child = await registry.create(options.inherit(name="fix", kept=True), parent_id=root.id)
    await registry.close_all()

    async def prepare(session):
        relay.trace_session(session)

    async with relay.activated():
        registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=models)
        loaded = await registry.load(child.id, prepare=prepare)
        await asyncio.wait_for(loaded.prompt("go"), TIMEOUT)
        await registry.close_all()
        await nemo_relay.subscribers.flush_async()

    start, end = (e for e in events if e.name == "atom")
    assert (start.scope_category, end.scope_category) == ("start", "end")
    assert (start.metadata["session_id"], start.metadata["agent_kind"]) == (child.id, "atom")
    assert any(e.category == "llm" for e in events)


async def test_done_reflection_after_a_turn_settles_is_not_in_its_trace(
    workspace, sessions_dir, plugins, monkeypatch, events
):
    monkeypatch.setenv(relay.CONFIG_ENV, str(plugins))
    llm = CellLLM(
        [
            cell("return_result(Done(explanation='ok'))"),
            LLMResponse(content="{}", finish_reason="stop"),
        ]
    )
    options = SessionOptions(
        workspace=workspace,
        agent_spec=PUBLIC_ATOM_SPEC,
        sessions_dir=sessions_dir,
        llm=llm,
        experimental_done_reflection=True,
        experimental_reflection_min_turns=1,
        experimental_reflection_min_seconds=0,
    )

    async def prepare(session):
        relay.trace_session(session)

    async with relay.activated():
        registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=AtomModels())
        root = await registry.create(options, prepare=prepare)
        await asyncio.wait_for(root.prompt("go"), TIMEOUT)
        async with asyncio.timeout(TIMEOUT):
            while root._agent.turns.running:
                await asyncio.sleep(0.01)
        await registry.close_all()
        await nemo_relay.subscribers.flush_async()

    assert [call.output_model is not None for call in llm.calls] == [False, True]
    assert len([e for e in events if e.category == "llm" and e.scope_category == "start"]) == 1
