# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Worker events and owner cleanup share one ordered ACP delivery boundary."""

import asyncio

import pytest
from acp.schema import AgentMessageChunk, ToolCallProgress, ToolCallStart
from nooa_atom.acp.event_bridge import ACPEventBridge
from nooa_atom.agent import AtomAgent
from nooa_atom.session.items import AgentEventUpdate, CancelledUpdate

from nooa.context_blocks.events import ResultStatus, ToolCallEvent
from nooa.events import PythonOutput
from nooa.interactive import AgentMessage
from nooa.unifiedllm import FakeLLMClient

from .test_event_bridge import _FakeSession, _RecordingClient


@pytest.fixture
async def thread_bridge(tmp_path, monkeypatch):
    import threading

    loop = asyncio.get_running_loop()
    old_debug = loop.get_debug()
    loop.set_debug(True)
    agent = AtomAgent(llm=FakeLLMClient(), cwd=tmp_path)
    session = _FakeSession(agent)
    client = _RecordingClient()
    bridge = ACPEventBridge(session, client)  # type: ignore[arg-type]
    owner = threading.get_ident()
    put = bridge._queue.put_nowait

    def owner_put(item):
        # Fail before a wrong-thread asyncio Future wakeup corrupts the waiting
        # pump, so the unfixed regression still has a bounded, clean teardown.
        assert threading.get_ident() == owner
        put(item)

    def owner_handler(handler):
        def call(*args):
            assert threading.get_ident() == owner
            return handler(*args)

        return call

    for handlers in (bridge._card_handlers, bridge._own_handlers):
        for name, handler in handlers.items():
            handlers[name] = owner_handler(handler)
    monkeypatch.setattr(bridge._queue, "put_nowait", owner_put)
    await asyncio.sleep(0)  # The pump is waiting on the queue before the worker emits.
    try:
        yield session, client, bridge
    finally:
        try:
            await asyncio.wait_for(bridge.close(), 3)
        finally:
            loop.set_debug(old_debug)


async def test_worker_events_reach_the_client_in_order(thread_bridge):
    session, client, bridge = thread_bridge

    def emit():
        session._agent.event_manager.add(
            ToolCallEvent(tool_call_id="worker", name="python_cell", arguments={"code": "1 + 1"})
        )
        session._agent.event_manager.add(
            PythonOutput(
                tool_call_id="worker",
                execution_status=ResultStatus.COMPLETE,
                execution_count=1,
                stdout="2",
            )
        )
        session._agent.event_manager.add(AgentMessage(content="worker finished"))

    await asyncio.to_thread(emit)
    await asyncio.wait_for(bridge.flush(), 3)
    updates = [update for _, update in client.updates]
    assert [type(update) for update in updates] == [
        ToolCallStart,
        ToolCallProgress,
        AgentMessageChunk,
    ]
    assert updates[1].status == "completed"
    assert updates[2].content.text.strip() == "worker finished"


@pytest.mark.parametrize("barrier", ["cancel", "close", "verbosity"])
async def test_owner_cleanup_follows_an_admitted_worker_start(thread_bridge, barrier):
    import threading

    session, client, bridge = thread_bridge
    thread = threading.Thread(
        target=lambda: session._agent.event_manager.add(
            ToolCallEvent(tool_call_id="worker", name="python_cell", arguments={"code": "work()"})
        )
    )
    thread.start()
    # Deliberately keep the owner loop occupied until admission has completed;
    # a call_soon callback alone would otherwise let cleanup overtake the start.
    thread.join(3)
    assert not thread.is_alive()
    if barrier == "close":
        await asyncio.wait_for(bridge.close(), 3)
    else:
        if barrier == "cancel":
            session.emit(CancelledUpdate(session_id=session.id, by="user"))
        else:
            bridge.set_verbosity("concise")
        await asyncio.wait_for(bridge.flush(), 3)
    cards = [
        update
        for _, update in client.updates
        if isinstance(update, ToolCallStart | ToolCallProgress)
    ]
    assert len(cards) == 2
    assert cards[0].tool_call_id == cards[1].tool_call_id == "worker"
    assert cards[0].status == "in_progress"
    assert cards[1].status == "failed"
    assert not bridge._open


async def test_worker_child_activity_is_mirrored_on_owner_loop(thread_bridge):
    session, client, bridge = thread_bridge
    child = _FakeSession(session._agent, "child")
    bridge._mirror(child)  # type: ignore[arg-type]

    def emit():
        child.emit(
            AgentEventUpdate(
                session_id=child.id,
                event=ToolCallEvent(
                    tool_call_id="cell", name="python_cell", arguments={"code": "1"}
                ),
            )
        )
        child.emit(
            AgentEventUpdate(
                session_id=child.id,
                event=PythonOutput(
                    tool_call_id="cell", execution_status=ResultStatus.COMPLETE, execution_count=1
                ),
            )
        )

    await asyncio.to_thread(emit)
    await asyncio.wait_for(bridge.flush(), 3)
    cards = [
        update
        for _, update in client.updates
        if isinstance(update, ToolCallStart | ToolCallProgress)
    ]
    assert len(cards) == 2
    assert cards[0].tool_call_id == cards[1].tool_call_id == "child:cell"
    assert cards[1].status == "completed"


async def test_worker_publishes_after_close_are_ignored(thread_bridge):
    from acp import text_block, update_agent_message

    _, client, bridge = thread_bridge
    await bridge.close()
    before = list(client.updates)
    await asyncio.to_thread(bridge.publish, update_agent_message(text_block("late")))
    await asyncio.to_thread(bridge.publish_best_effort, update_agent_message(text_block("late")))
    await asyncio.sleep(0)
    assert client.updates == before
    assert bridge._queue.empty()


@pytest.mark.parametrize("best_effort", [False, True])
async def test_worker_publications_are_included_in_flush(thread_bridge, best_effort):
    import threading

    from acp import text_block, update_agent_message

    _, client, bridge = thread_bridge
    publish = bridge.publish_best_effort if best_effort else bridge.publish
    worker = threading.Thread(target=publish, args=(update_agent_message(text_block("published")),))
    worker.start()
    worker.join(3)
    assert not worker.is_alive()
    await asyncio.wait_for(bridge.flush(), 3)
    assert len(client.updates) == 1
    assert client.updates[0][1].content.text.strip() == "published"
