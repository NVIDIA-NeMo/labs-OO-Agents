# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in execution relationships follow sessions, not ACP attachments."""

import asyncio
import gc
import threading
import weakref
from typing import Any, Literal, cast

import pytest
from acp.interfaces import Client as ACPClient
from acp.schema import ToolCallProgress, ToolCallStart
from nooa_atom.acp.event_bridge import ACPEventBridge
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

from nooa import Agent
from nooa.interactive import Done, InteractiveAgent
from nooa.runtime.hooks import InstrumentationHooks
from nooa.unifiedllm import FakeLLMClient

pytestmark = pytest.mark.timeout(40)

KEY = "nooa.dev/execution"
REFS = []
STREAMS = {}
READY = asyncio.Event()
RELEASE = asyncio.Event()
SHARED: Any = None
THREAD_READY = threading.Event()
THREAD_RELEASE = threading.Event()
THREADS = []


class UnprintableError(Exception):
    def __str__(self):
        raise RuntimeError("broken exception formatter")


class Worker(Agent, llm=FakeLLMClient()):
    def blocked(self):
        THREAD_READY.set()
        THREAD_RELEASE.wait(5)

    def broken(self):
        raise UnprintableError()

    def check(self):
        raise ValueError("expected 11, got 10")

    def work(self):
        try:
            self.check()
        except ValueError:
            pass
        return 10

    async def async_work(self):
        await asyncio.sleep(0)
        return self.work()

    def stream(self):
        yield self.work()
        yield self.work()

    async def wait(self):
        READY.set()
        await RELEASE.wait()
        return self.work()


class Host(InteractiveAgent):
    async def handle(self, notification: dict[str, list[Any]]) -> Done:
        text = notification.get("user_messages", ["work"])[0]
        if text == "resume":
            stream = STREAMS.pop(id(self))
            await asyncio.to_thread(next, stream)
            stream.close()
        else:
            worker = Worker()
            REFS.append(weakref.ref(worker))
            if text == "thread-race":
                from contextvars import copy_context

                context = copy_context()

                def start_thread():
                    thread = threading.Thread(target=lambda: context.run(worker.blocked))
                    THREADS.append(thread)
                    thread.start()
                    assert THREAD_READY.wait(2)

                # This callback runs before the dispatcher settles the failed
                # turn, but its thread's queued start arrives after settlement.
                asyncio.get_running_loop().call_soon(start_thread)
                raise ValueError("turn fails before the loop can deliver the worker start")
            elif text == "shared":
                await SHARED.async_work()
            elif text == "unprintable":
                try:
                    worker.broken()
                except UnprintableError:
                    pass
            elif text == "thread":
                await asyncio.to_thread(worker.work)
            elif text == "wait":
                await worker.wait()
            elif text == "stream":
                stream = worker.stream()
                next(stream)
                STREAMS[id(self)] = stream
            else:
                worker.work()
        return Done(explanation="done")


class Client:
    def __init__(self):
        self.updates = []
        self.threads = []

    async def session_update(self, session_id, update, **kwargs):
        self.updates.append(update)
        self.threads.append(threading.get_ident())


def starts(client):
    return [
        u
        for u in client.updates
        if isinstance(u, ToolCallStart) and u.field_meta and KEY in u.field_meta
    ]


def meta(update):
    return update.field_meta[KEY]


@pytest.fixture
async def setup_tree(tmp_path):
    global READY, RELEASE
    READY, RELEASE = asyncio.Event(), asyncio.Event()
    made = []

    async def make(enabled=True):
        registry = SessionRegistry(
            SessionStore(tmp_path / str(len(made))),
            agent_factory=lambda options, storage: Host(storage=storage, llm=FakeLLMClient()),
            execution_tree=enabled,
        )
        session = await registry.create(
            SessionOptions(workspace=tmp_path, agent_spec=f"{__name__}:Host")
        )
        client = Client()
        bridge = ACPEventBridge(session, cast(ACPClient, client))
        made.append((registry, bridge))
        return session, client, bridge

    yield make
    for registry, bridge in made:
        await registry.close_all()
        await bridge.close()
    for stream in STREAMS.values():
        stream.close()
    STREAMS.clear()
    REFS.clear()


async def test_default_off_and_real_nested_failure(setup_tree):
    session, client, bridge = await setup_tree(False)
    await session.prompt("work")
    await bridge.flush()
    assert not starts(client)
    session, client, bridge = await setup_tree()
    await session.prompt("work")
    await bridge.flush()
    nodes = {meta(u)["spanId"]: meta(u) for u in starts(client)}
    assert {n["name"] for n in nodes.values()} >= {"Host.handle", "Worker.work", "Worker.check"}
    assert all(n["parentSpanId"] is None or n["parentSpanId"] in nodes for n in nodes.values())
    ends = [
        u
        for u in client.updates
        if isinstance(u, ToolCallProgress) and u.field_meta and KEY in u.field_meta
    ]
    assert len(ends) == len(nodes)
    assert next(u for u in ends if meta(u)["name"] == "Worker.check").status == "failed"
    assert next(u for u in ends if meta(u)["name"] == "Host.handle").status == "completed"
    gc.collect()
    assert all(ref() is None for ref in REFS)


async def test_thread_capture_and_two_prompts(setup_tree):
    session, client, bridge = await setup_tree()
    runs = []
    for _ in range(2):
        before = len(client.updates)
        await session.prompt("thread")
        await bridge.flush()
        current = [
            u
            for u in client.updates[before:]
            if isinstance(u, ToolCallStart) and u.field_meta and KEY in u.field_meta
        ]
        runs.append({meta(u)["runId"] for u in current})
        assert len(runs[-1]) == 1
    assert runs[0].isdisjoint(runs[1])
    assert set(client.threads) == {threading.get_ident()}


async def test_later_cancel_preserves_suspended_generator(setup_tree):
    session, client, bridge = await setup_tree()
    await session.prompt("stream")
    await bridge.flush()
    generator = next(meta(u) for u in starts(client) if meta(u)["name"] == "Worker.stream")
    pending = asyncio.create_task(session.prompt("wait"))
    await asyncio.wait_for(READY.wait(), 5)
    await session.cancel()
    await pending
    await session.prompt("resume")
    await bridge.flush()
    children = [meta(u) for u in starts(client) if meta(u)["parentSpanId"] == generator["spanId"]]
    assert len(children) == 2
    assert all(n["runId"] == generator["runId"] for n in children)


async def test_verbosity_and_attach_mid_method(setup_tree):
    session, client, bridge = await setup_tree()
    pending = asyncio.create_task(session.prompt("wait"))
    await asyncio.wait_for(READY.wait(), 5)
    await bridge.flush()
    before = starts(client)
    bridge.set_verbosity("silent")
    await bridge.flush()
    ended = [u.tool_call_id for u in client.updates if isinstance(u, ToolCallProgress)]
    assert all(u.tool_call_id in ended for u in before)
    bridge.set_verbosity("verbose")
    second = Client()
    attached = ACPEventBridge(session, cast(ACPClient, second))
    RELEASE.set()
    await pending
    await bridge.flush()
    await attached.flush()
    for recorded in (client, second):
        nodes = {meta(u)["spanId"]: meta(u) for u in starts(recorded)}
        assert all(n["parentSpanId"] is None or n["parentSpanId"] in nodes for n in nodes.values())
        terminal = [
            u.tool_call_id
            for u in recorded.updates
            if isinstance(u, ToolCallProgress) and u.field_meta and KEY in u.field_meta
        ]
        assert len(terminal) == len(set(terminal))
    assert not session._execution._nodes
    await attached.close(finish_open=False)


async def test_detach_does_not_stop_session_observation(setup_tree):
    session, client, bridge = await setup_tree()
    await bridge.close(finish_open=False)
    await session.prompt("work")
    second = Client()
    attached = ACPEventBridge(session, cast(ACPClient, second))
    await session.prompt("work")
    await attached.flush()
    assert any(meta(u)["name"] == "Worker.work" for u in starts(second))
    await attached.close()


async def test_real_atom_python_tools_and_child_sessions(tmp_path):
    from atom_test_agents import ATOM_SPEC, AtomModels, cell

    models = AtomModels()
    child_code = "return_result(Done(explanation='fixed', result=TaskResult(solution_description='s', evidence='e', how_to_verify='h', report='ok')))"
    models.scripts["helper"] = [cell(child_code)]
    code = (
        "from test_execution_tree import Worker\n"
        "Worker().work()\n"
        "await self.shell.write_file('tree.txt', 'checked')\n"
        "result = await self.shell.run('cat tree.txt')\n"
        "assert result.returncode == 0\n"
        "await self.delegate('verify', 'verify it', name='helper')\n"
        "return_result(Done(explanation='done'))"
    )
    response = cell(code)
    # The same provider id in independent model calls must not merge cards.
    second = cell("return_result(Done(explanation='again'))")
    second = second.replace_parts(
        tuple(
            part.model_copy(update={"id": response.tool_calls[0].id})
            if getattr(part, "kind", None) == "tool_call"
            else part
            for part in second.parts
        )
    )
    models.scripts[None] = [response, second]
    registry = SessionRegistry(
        SessionStore(tmp_path / "sessions"), agent_factory=models, execution_tree=True
    )
    root = await registry.create(SessionOptions(workspace=tmp_path, agent_spec=ATOM_SPEC))
    client = Client()
    bridge = ACPEventBridge(root, cast(ACPClient, client), resolve_child=registry.get)
    try:
        await asyncio.wait_for(root.prompt("go"), 30)
        await bridge.flush()
        nodes = {meta(u)["spanId"]: meta(u) for u in starts(client)}
        assert {n["nodeType"] for n in nodes.values()} == {"method", "python", "file", "terminal"}
        assert len({n["runId"] for n in nodes.values()}) == 1
        assert all(n["parentSpanId"] is None or n["parentSpanId"] in nodes for n in nodes.values())
        assert any(
            n["name"] == "Worker.work" and nodes[n["parentSpanId"]]["nodeType"] == "python"
            for n in nodes.values()
        )
        terminals = [
            u
            for u in client.updates
            if isinstance(u, ToolCallProgress)
            and u.field_meta
            and meta(u)["nodeType"] == "terminal"
            and u.status == "completed"
        ]
        assert any(u.raw_output is not None and u.raw_output["exit_code"] == 0 for u in terminals)
        assert (tmp_path / "tree.txt").read_text() == "checked"
        before = len(client.updates)
        await root.prompt("again")
        await bridge.flush()
        new = [
            meta(u)
            for u in client.updates[before:]
            if isinstance(u, ToolCallStart) and u.field_meta
        ]
        assert {n["spanId"] for n in new}.isdisjoint(nodes)
        assert all(n["runId"] not in {v["runId"] for v in nodes.values()} for n in new)
    finally:
        await registry.close_all()
        await bridge.close()


async def test_execution_observer_composes_with_tracing(setup_tree, monkeypatch):
    from nooa.runtime.hooks import compose_hooks, get_hooks, set_hooks

    calls = []

    class Hooks:
        def __getattr__(self, name):
            def record(**kwargs):
                calls.append(name)

            return record

    session, client, bridge = await setup_tree()
    # Bootstrap hooks are explicitly installed in the fresh session context.
    session2, client2, bridge2 = await setup_tree(False)
    await session2.close()
    registry = SessionRegistry(
        SessionStore(session.options.workspace / "composed"),
        agent_factory=lambda options, storage: Host(storage=storage, llm=FakeLLMClient()),
        execution_tree=True,
    )

    async def prepare(current):
        current.add_loop_context_hook(
            lambda: set_hooks(compose_hooks(get_hooks(), cast(InstrumentationHooks, Hooks())))
        )

    current = await registry.create(session.options, prepare=prepare)
    recording = Client()
    attachment = ACPEventBridge(current, cast(ACPClient, recording))
    try:
        await current.prompt("work")
        await attachment.flush()
        assert "before_agent_call" in calls and "after_agent_call" in calls
        assert any(meta(u)["name"] == "Worker.check" for u in starts(recording))
    finally:
        await registry.close_all()
        await attachment.close()


async def test_cancel_before_commit_cannot_finish_previous_generator(setup_tree, monkeypatch):
    session, client, bridge = await setup_tree()
    await session.prompt("stream")
    await bridge.flush()
    generator = next(meta(u) for u in starts(client) if meta(u)["name"] == "Worker.stream")
    preparing = asyncio.Event()
    original = session._apply_pending_model

    async def block_prepare():
        preparing.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(session, "_apply_pending_model", block_prepare)
    receipt = await session.submit("work")
    pending = asyncio.ensure_future(session.outcome(receipt.item_id))
    await asyncio.wait_for(preparing.wait(), 5)
    await session.cancel()
    await asyncio.wait_for(pending, 5)
    assert generator["spanId"] in session._execution._nodes
    monkeypatch.setattr(session, "_apply_pending_model", original)
    assert session.withdraw(receipt)
    session.resume_dispatch()
    await asyncio.wait_for(session.prompt("resume"), 5)
    await bridge.flush()
    children = [meta(u) for u in starts(client) if meta(u)["parentSpanId"] == generator["spanId"]]
    assert len(children) == 2
    assert {n["runId"] for n in children} == {generator["runId"]}


async def test_long_running_parent_survives_completed_ancestry_eviction(setup_tree):
    from nooa_atom.session.items import ExecutionUpdate

    session, client, bridge = await setup_tree()

    def observation(
        span, parent=None, status: Literal["in_progress", "completed", "failed"] = "in_progress"
    ):
        return ExecutionUpdate(
            session_id=session.id,
            status=status,
            node={
                "version": 1,
                "runId": "long",
                "spanId": span,
                "parentSpanId": parent,
                "_parentSessionId": session.id,
                "nodeType": "method",
                "name": span,
                "startedAtMs": 1,
            },
        )

    bridge._on_execution(observation("root"))
    for index in range(4097):
        bridge._on_execution(observation(str(index), "root"))
        bridge._on_execution(observation(str(index), "root", "completed"))
    bridge._on_execution(observation("last", "root"))
    await bridge.flush()
    last = next(meta(u) for u in starts(client) if meta(u)["name"] == "last")
    assert last["parentSpanId"] == "root"
    assert len(bridge._execution_seen) <= 4096


async def test_cancelling_parent_does_not_finish_running_child(tmp_path):
    from atom_test_agents import ATOM_SPEC, AtomModels, cell, until

    models = AtomModels(
        {
            None: [
                cell(
                    "self.v.child = await self.spawn('helper', 'wait', name='helper')\nreturn_result(Done(explanation='started'))"
                ),
                cell("await asyncio.Event().wait()"),
                cell("return_result(Done(explanation='child outcome handled'))"),
                cell("return_result(Done(explanation='recovered'))"),
            ],
            "helper": [cell("await asyncio.Event().wait()")],
        }
    )
    registry = SessionRegistry(
        SessionStore(tmp_path / "sessions"), agent_factory=models, execution_tree=True
    )
    root = await registry.create(SessionOptions(workspace=tmp_path, agent_spec=ATOM_SPEC))
    client = Client()
    bridge = ACPEventBridge(root, cast(ACPClient, client), resolve_child=registry.get)
    try:
        await asyncio.wait_for(root.prompt("start"), 30)
        child = next(s for s in registry.sessions.values() if s.parent_id == root.id)
        await until(
            lambda: any(
                meta(u)["nodeType"] == "python" and u.tool_call_id.startswith(child.id + ":")
                for u in starts(client)
            )
        )
        await bridge.flush()
        child_starts = {
            u.tool_call_id for u in starts(client) if u.tool_call_id.startswith(child.id + ":")
        }
        pending = asyncio.create_task(root.prompt("wait"))
        await until(lambda: root.info.status == "running")
        await root.cancel()
        await pending
        await bridge.flush()
        assert (
            not {
                u.tool_call_id
                for u in client.updates
                if isinstance(u, ToolCallProgress) and u.status == "failed"
            }
            & child_starts
        )
        await child.cancel()
        await bridge.flush()
        assert {
            u.tool_call_id
            for u in client.updates
            if isinstance(u, ToolCallProgress) and u.status == "failed"
        } & child_starts
        await root.prompt("recover")
        await bridge.flush()
        endings = [
            u.tool_call_id
            for u in client.updates
            if isinstance(u, ToolCallProgress)
            and u.status in {"failed", "completed"}
            and u.field_meta
        ]
        assert len(endings) == len(set(endings))
    finally:
        await registry.close_all()
        await bridge.close()


async def test_unprintable_exception_still_finishes_its_card(setup_tree):
    session, client, bridge = await setup_tree()
    await session.prompt("unprintable")
    await bridge.flush()
    broken = next(u for u in starts(client) if meta(u)["name"] == "Worker.broken")
    endings = [
        u
        for u in client.updates
        if isinstance(u, ToolCallProgress) and u.tool_call_id == broken.tool_call_id
    ]
    assert len(endings) == 1 and endings[0].status == "failed"


async def test_parallel_sessions_do_not_share_child_observations(setup_tree, monkeypatch):
    import sys

    monkeypatch.setattr(sys.modules[__name__], "SHARED", Worker())
    first, left, left_bridge = await setup_tree()
    second, right, right_bridge = await setup_tree()
    await asyncio.wait_for(asyncio.gather(first.prompt("shared"), second.prompt("shared")), 5)
    await asyncio.gather(left_bridge.flush(), right_bridge.flush())
    all_ids = []
    all_runs = []
    for session, client in ((first, left), (second, right)):
        nodes = {meta(u)["spanId"]: meta(u) for u in starts(client)}
        assert sum(n["name"] == "Worker.work" for n in nodes.values()) == 1
        assert all(n["parentSpanId"] is None or n["parentSpanId"] in nodes for n in nodes.values())
        all_ids.append(set(nodes))
        all_runs.append({n["runId"] for n in nodes.values()})
        assert not session._execution._managers
    assert all_ids[0].isdisjoint(all_ids[1])
    assert all_runs[0].isdisjoint(all_runs[1])


async def test_pending_thread_start_cannot_overtake_its_failure(setup_tree):
    from nooa_atom.session.session import TurnFailedError

    THREAD_READY.clear()
    THREAD_RELEASE.clear()
    session, client, bridge = await setup_tree()
    try:
        with pytest.raises(TurnFailedError):
            await asyncio.wait_for(session.prompt("thread-race"), 5)
        await bridge.flush()
        blocked = next(u for u in starts(client) if meta(u)["name"] == "Worker.blocked")
        endings = [
            u
            for u in client.updates
            if isinstance(u, ToolCallProgress) and u.tool_call_id == blocked.tool_call_id
        ]
        assert len(endings) == 1 and endings[0].status == "failed"
    finally:
        THREAD_RELEASE.set()
        for thread in THREADS:
            await asyncio.to_thread(thread.join, 5)
        THREADS.clear()
