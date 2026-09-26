# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The router in-process, with fake workers and a raw JSON-lines client.

Everything talks over socket pairs, as the real router does; no ACP
library on either side, so the tests see the exact frames the router
forwards.
"""

import asyncio
import gc
import json
import logging
import socket
import time
import warnings
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from acp import PROTOCOL_VERSION
from nooa_coder.acp.framing import FRAME_LIMIT, Frame, encode, read_frame
from nooa_coder.acp.router import INIT_REQUEST_ID, Router, WorkerProcess
from nooa_coder.acp.server import initialize_response
from nooa_coder.session.store import SessionStore, sessions_root

TIMEOUT = 10
BASE = 1 << 32

Handler = Callable[["FakeWorker", Frame], Awaitable[None]]


async def _pair() -> tuple[
    tuple[asyncio.StreamReader, asyncio.StreamWriter],
    tuple[asyncio.StreamReader, asyncio.StreamWriter],
]:
    left, right = socket.socketpair()
    a = await asyncio.open_unix_connection(sock=left, limit=FRAME_LIMIT)
    b = await asyncio.open_unix_connection(sock=right, limit=FRAME_LIMIT)
    return a, b


class FakeWorker:
    """A scripted ACP worker on the far end of the router's socket.

    Answers ``initialize``, ``session/new`` (``s<k>``, then ``s<k>-2``...),
    ``session/load``, ``session/close``, ``session/prompt`` and the delete
    extension with success unless ``handlers`` has an entry for the method.
    ``received`` holds every frame read; ``eof`` is set at end of stream.
    ``stall()`` stops reading the socket for good, like a worker whose loop
    is busy in a long cell: the kernel buffers fill and the router's writes
    to it back up.
    """

    def __init__(self, k: int, handlers: dict[str, Handler]) -> None:
        self.k = k
        self.handlers = handlers
        self.received: list[Frame] = []
        self.eof = asyncio.Event()
        self.killed = False
        self.changed = asyncio.Event()
        self._new_count = 0
        self.task: asyncio.Task[None] | None = None
        self.reader: asyncio.StreamReader
        self.writer: asyncio.StreamWriter

    async def start(self) -> WorkerProcess:
        (router_reader, router_writer), (self.reader, self.writer) = await _pair()
        self.task = asyncio.create_task(self._run())
        return WorkerProcess(router_reader, router_writer, self.wait, self.kill, pid=None)

    async def wait(self) -> int:
        if self.task is not None:
            await asyncio.gather(self.task, return_exceptions=True)
        return 0

    def kill(self) -> None:
        self.killed = True
        if self.task is not None:
            self.task.cancel()

    async def send(self, message: dict[str, Any]) -> None:
        self.writer.write(encode(message))
        await self.writer.drain()

    async def reply(self, frame: Frame, result: Any = None, *, error: Any = None) -> None:
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": frame.id}
        if error is not None:
            message["error"] = error
        else:
            message["result"] = result if result is not None else {}
        await self.send(message)

    def stall(self) -> None:
        self.writer.transport.pause_reading()  # pyright: ignore[reportAttributeAccessIssue]

    def methods(self) -> list[str | None]:
        return [frame.method for frame in self.received]

    async def wait_for(self, predicate: Callable[[], bool]) -> None:
        async def poll() -> None:
            while not predicate():
                self.changed.clear()
                await self.changed.wait()

        await asyncio.wait_for(poll(), TIMEOUT)

    async def _run(self) -> None:
        try:
            while True:
                frame = await read_frame(self.reader)
                if frame is None:
                    break
                self.received.append(frame)
                self.changed.set()
                if frame.is_request:
                    asyncio.create_task(self._answer(frame))
        finally:
            self.eof.set()
            self.changed.set()
            self.writer.close()

    async def _answer(self, frame: Frame) -> None:
        handler = self.handlers.get(frame.method or "")
        if handler is not None:
            await handler(self, frame)
            return
        if frame.method == "initialize":
            await self.reply(frame, {"protocolVersion": 1})
        elif frame.method == "session/new":
            self._new_count += 1
            suffix = "" if self._new_count == 1 else f"-{self._new_count}"
            await self.reply(frame, {"sessionId": f"s{self.k}{suffix}"})
        elif frame.method == "session/prompt":
            await self.reply(frame, {"stopReason": "end_turn"})
        else:
            await self.reply(frame, {})


class Harness:
    """A router serving a raw client, spawning ``FakeWorker``s."""

    def __init__(self, workspace: Path) -> None:
        self.cwd = str(workspace)
        self.store = SessionStore(sessions_root(workspace))
        self.handlers: dict[str, Handler] = {}
        self.workers: dict[int, FakeWorker] = {}
        self.spawn_delay: dict[int, float] = {}
        self.spawn_error: dict[int, BaseException] = {}
        self.spawned: list[int] = []
        self.router = Router(spawn=self._spawn)
        self.inbox: list[Frame] = []
        self._next_id = 0

    async def _spawn(self, k: int) -> WorkerProcess:
        self.spawned.append(k)
        if k in self.spawn_delay:
            await asyncio.sleep(self.spawn_delay[k])
        if k in self.spawn_error:
            raise self.spawn_error[k]
        worker = FakeWorker(k, self.handlers)
        self.workers[k] = worker
        return await worker.start()

    async def start(self) -> None:
        (self.reader, self.writer), (router_reader, router_writer) = await _pair()
        self.serving = asyncio.create_task(self.router.serve(router_reader, router_writer))

    async def send(self, message: dict[str, Any]) -> None:
        self.writer.write(encode(message))
        await self.writer.drain()

    async def request(self, method: str, params: Any = None) -> int:
        self._next_id += 1
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        if params is not None:
            message["params"] = params
        await self.send(message)
        return self._next_id

    async def notify(self, method: str, params: Any) -> None:
        await self.send({"jsonrpc": "2.0", "method": method, "params": params})

    async def next_frame(self, timeout: float = TIMEOUT) -> Frame:
        frame = await asyncio.wait_for(read_frame(self.reader), timeout)
        assert frame is not None, "the router closed standard output"
        self.inbox.append(frame)
        return frame

    async def response(self, request_id: Any, timeout: float = TIMEOUT) -> Frame:
        for frame in self.inbox:
            if frame.is_response and frame.id == request_id:
                return frame
        deadline = time.monotonic() + timeout
        while True:
            frame = await self.next_frame(max(0.01, deadline - time.monotonic()))
            if frame.is_response and frame.id == request_id:
                return frame

    async def call(self, method: str, params: Any = None) -> Frame:
        return await self.response(await self.request(method, params))

    async def initialize(self, params: dict[str, Any] | None = None) -> Frame:
        return await self.call("initialize", params or {"protocolVersion": PROTOCOL_VERSION})

    async def initialize_request_only(self) -> None:
        await self.request("initialize", {"protocolVersion": PROTOCOL_VERSION})

    async def new_session(self, cwd: str | None = None) -> str:
        frame = await self.call("session/new", {"cwd": cwd or self.cwd, "mcpServers": []})
        return frame.message["result"]["sessionId"]

    async def close(self) -> None:
        if not self.writer.is_closing():
            self.writer.close()
        await asyncio.wait_for(self.serving, TIMEOUT)


@pytest.fixture
async def harness(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    harness = Harness(workspace)
    await harness.start()
    yield harness
    await harness.close()


def _stored(
    harness: Harness,
    *,
    parent_id: str | None = None,
    turns: int = 0,
    workspace: str | None = None,
    answered: bool = True,
) -> str:
    """A session on disk with ``turns`` user messages, each answered unless told otherwise."""
    handle = harness.store.create(
        agent="a:B", workspace=workspace or harness.cwd, host="acp", parent_id=parent_id
    )
    from coder_test_agents import SessionUserMessage
    from nooa_coder.session.events import TurnEnded

    for _ in range(turns):
        handle.events.add(SessionUserMessage(content="hello"))
        if answered:
            handle.events.add(TurnEnded(outcome_kind="done"))
    handle.close()
    return handle.id


# ---- answered by the router ---------------------------------------------------


async def test_initialize_is_answered_without_a_worker(harness):
    frame = await harness.initialize(
        {"protocolVersion": PROTOCOL_VERSION, "clientCapabilities": {}}
    )
    expected = initialize_response(PROTOCOL_VERSION).model_dump(
        mode="json", by_alias=True, exclude_none=True, exclude_unset=True
    )
    assert frame.message == {"jsonrpc": "2.0", "id": 1, "result": expected}
    assert harness.spawned == []


async def test_authenticate_and_unknown_methods_are_method_not_found(harness):
    for method in ("authenticate", "logout", "_nooa/other"):
        frame = await harness.call(method, {})
        assert frame.message["error"] == {
            "code": -32601,
            "message": "Method not found",
            "data": {"method": method},
        }
    assert harness.spawned == []


async def test_session_list_is_answered_from_the_store_without_a_worker(harness):
    listed = _stored(harness, turns=1)
    _stored(harness)  # no messages: not listed
    frame = await harness.call("session/list", {"cwd": harness.cwd})
    sessions = frame.message["result"]["sessions"]
    assert [entry["sessionId"] for entry in sessions] == [listed]
    assert sessions[0]["_meta"] == {"dev.nooa/status": "on_disk"}
    assert harness.spawned == []


async def test_session_list_reads_the_store_of_each_cwd(harness, tmp_path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    here = _stored(harness, turns=1)
    there = Harness(other)
    elsewhere = _stored(there, turns=1)

    async def listed(params: dict[str, Any]) -> list[str]:
        frame = await harness.call("session/list", params)
        return [entry["sessionId"] for entry in frame.message["result"]["sessions"]]

    # No workspace named yet (Pool lists before session/new, without cwd):
    # the server's own working directory is the workspace.
    monkeypatch.chdir(harness.cwd)
    assert await listed({}) == [here]
    assert await listed({"cwd": str(other)}) == [elsewhere]
    assert await listed({"cwd": harness.cwd}) == [here]
    assert set(await listed({})) == {here, elsewhere}
    assert not (Path(harness.cwd) / ".nooa" / "sessions" / f"{elsewhere}.db").exists()


async def test_a_load_reads_the_store_of_its_cwd(harness, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    session_id = _stored(harness)
    await harness.initialize()
    frame = await harness.call("session/load", {"sessionId": session_id, "cwd": str(other)})
    assert frame.message["error"]["code"] == -32002
    assert harness.spawned == []


async def test_unknown_session_ids_are_resource_not_found_or_dropped(harness):
    frame = await harness.call("session/prompt", {"sessionId": "nope", "prompt": []})
    assert frame.message["error"] == {
        "code": -32002,
        "message": "Resource not found",
        "data": {"uri": "nope"},
    }
    await harness.notify("session/cancel", {"sessionId": "nope"})
    frame = await harness.call("session/list", {})  # the notification produced nothing
    assert "result" in frame.message
    assert [f.id for f in harness.inbox] == [1, 2]


# ---- sessions and workers ----------------------------------------------------


async def test_session_new_spawns_a_worker_that_gets_the_clients_initialize_first(harness):
    params = {
        "protocolVersion": PROTOCOL_VERSION,
        "clientCapabilities": {"fs": {"readTextFile": True}},
    }
    await harness.initialize(params)
    session_id = await harness.new_session()
    assert session_id == "s1"
    assert harness.spawned == [1]
    worker = harness.workers[1]
    first = worker.received[0]
    assert first.method == "initialize" and first.id == INIT_REQUEST_ID
    assert first.params == params
    assert worker.methods()[1] == "session/new"
    # The worker's initialize answer never reaches the client.
    assert all(frame.id != INIT_REQUEST_ID for frame in harness.inbox)


async def test_two_sessions_get_two_workers_and_requests_route_by_session(harness):
    await harness.initialize()
    first, second = await harness.new_session(), await harness.new_session()
    assert (first, second) == ("s1", "s2")
    await harness.call("session/prompt", {"sessionId": second, "prompt": []})
    await harness.call("session/prompt", {"sessionId": first, "prompt": []})
    assert harness.workers[1].methods().count("session/prompt") == 1
    assert harness.workers[2].methods().count("session/prompt") == 1
    prompt = [f for f in harness.workers[2].received if f.method == "session/prompt"][0]
    assert prompt.session_id == "s2"


async def test_worker_requests_reach_the_client_and_replies_route_back_by_id(harness):
    async def ask_then_answer(worker: FakeWorker, frame: Frame) -> None:
        await worker.send(
            {
                "jsonrpc": "2.0",
                "id": worker.k * BASE,
                "method": "session/request_permission",
                "params": {"sessionId": frame.session_id},
            }
        )
        await worker.wait_for(lambda: any(f.is_response for f in worker.received))
        await worker.reply(frame, {"stopReason": "end_turn"})

    harness.handlers["session/prompt"] = ask_then_answer
    await harness.initialize()
    first, second = await harness.new_session(), await harness.new_session()
    prompts = [
        await harness.request("session/prompt", {"sessionId": sid, "prompt": []})
        for sid in (first, second)
    ]
    asked: list[Frame] = []
    while len(asked) < 2:
        frame = await harness.next_frame()
        if frame.method == "session/request_permission":
            asked.append(frame)
    assert sorted(frame.id for frame in asked) == [BASE, 2 * BASE]
    for frame in asked:
        await harness.send({"jsonrpc": "2.0", "id": frame.id, "result": {"outcome": frame.id}})
    for request_id in prompts:
        assert (await harness.response(request_id)).message["result"]["stopReason"] == "end_turn"
    for k in (1, 2):
        [reply] = [f for f in harness.workers[k].received if f.is_response]
        assert reply.message["result"] == {"outcome": k * BASE}


async def test_replies_with_stale_string_or_bool_ids_are_dropped(harness, caplog):
    caplog.set_level(logging.INFO, logger="nooa_coder.acp.router")
    await harness.initialize()
    await harness.new_session()
    worker = harness.workers[1]
    for bad in (99 * BASE, "abc", True, -1):
        await harness.send({"jsonrpc": "2.0", "id": bad, "result": {}})
    await harness.send({"jsonrpc": "2.0", "id": BASE + 3, "result": {"ok": 1}})
    await worker.wait_for(lambda: any(f.is_response for f in worker.received))
    assert [f.id for f in worker.received if f.is_response] == [BASE + 3]
    dropped = [r for r in caplog.records if "dropped" in r.getMessage().lower()]
    assert len(dropped) == 4


async def test_cancel_request_goes_to_the_worker_running_the_request(harness):
    started = asyncio.Event()

    async def hold(worker: FakeWorker, frame: Frame) -> None:
        started.set()
        await worker.wait_for(lambda: "$/cancel_request" in worker.methods())
        await worker.reply(frame, error={"code": -32800, "message": "Request cancelled"})

    harness.handlers["session/prompt"] = hold
    await harness.initialize()
    session_id = await harness.new_session()
    request_id = await harness.request("session/prompt", {"sessionId": session_id, "prompt": []})
    await asyncio.wait_for(started.wait(), TIMEOUT)
    await harness.notify("$/cancel_request", {"requestId": request_id})
    await harness.notify("$/cancel_request", {"requestId": 12345})  # unknown: dropped
    frame = await harness.response(request_id)
    assert frame.message["error"]["code"] == -32800


async def test_frames_from_one_worker_keep_their_order(harness):
    async def chatty(worker: FakeWorker, frame: Frame) -> None:
        note = {"jsonrpc": "2.0", "method": "session/update", "params": {"sessionId": "s1"}}
        await worker.send({**note, "params": {"sessionId": "s1", "n": 0}})
        await worker.reply(frame, {"stopReason": "end_turn"})
        for n in (1, 2, 3):
            await worker.send({**note, "params": {"sessionId": "s1", "n": n}})

    harness.handlers["session/prompt"] = chatty
    await harness.initialize()
    await harness.new_session()
    request_id = await harness.request("session/prompt", {"sessionId": "s1", "prompt": []})
    seen = []
    while len(seen) < 5:
        frame = await harness.next_frame()
        if frame.id == request_id or frame.method == "session/update":
            seen.append("response" if frame.is_response else frame.params["n"])
    assert seen == [0, "response", 1, 2, 3]


async def test_a_five_mib_frame_passes_both_ways(harness):
    text = "z" * (5 * 1024 * 1024)

    async def echo_big(worker: FakeWorker, frame: Frame) -> None:
        await worker.send(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": frame.session_id,
                    "text": frame.params["prompt"][0]["text"],
                },
            }
        )
        await worker.reply(frame, {"stopReason": "end_turn"})

    harness.handlers["session/prompt"] = echo_big
    await harness.initialize()
    session_id = await harness.new_session()
    request = {"sessionId": session_id, "prompt": [{"type": "text", "text": text}]}
    request_id = await harness.request("session/prompt", request)
    await harness.response(request_id)
    [update] = [f for f in harness.inbox if f.method == "session/update"]
    assert update.params["text"] == text
    [prompt] = [f for f in harness.workers[1].received if f.method == "session/prompt"]
    assert prompt.raw == encode(
        {"jsonrpc": "2.0", "id": request_id, "method": "session/prompt", "params": request}
    )


# ---- isolation ----------------------------------------------------------------


async def test_a_stalled_worker_does_not_delay_another_workers_round_trip(harness):
    await harness.initialize()
    stuck, free = await harness.new_session(), await harness.new_session()
    harness.workers[1].stall()
    chunk = "q" * (64 * 1024)
    for _ in range(16):  # 1 MiB for the worker that stopped reading
        await harness.notify("session/cancel", {"sessionId": stuck, "pad": chunk})
    started = time.monotonic()
    frame = await harness.call("session/prompt", {"sessionId": free, "prompt": []})
    assert frame.message["result"]["stopReason"] == "end_turn"
    assert time.monotonic() - started < 2


async def test_a_spawn_in_progress_does_not_delay_another_sessions_cancel(harness):
    await harness.initialize()
    running = await harness.new_session()
    harness.spawn_delay[2] = 3.0
    pending_new = await harness.request("session/new", {"cwd": harness.cwd, "mcpServers": []})
    await asyncio.sleep(0.05)
    started = time.monotonic()
    await harness.notify("session/cancel", {"sessionId": running})
    worker = harness.workers[1]
    await worker.wait_for(lambda: "session/cancel" in worker.methods())
    assert time.monotonic() - started < 1
    assert (await harness.response(pending_new)).message["result"]["sessionId"] == "s2"


# ---- load ---------------------------------------------------------------------


async def test_loading_a_child_goes_to_the_worker_of_its_root(harness):
    root = _stored(harness)
    child = _stored(harness, parent_id=root)
    grandchild = _stored(harness, parent_id=child)
    await harness.initialize()
    for session_id in (root, grandchild):
        frame = await harness.call("session/load", {"sessionId": session_id, "cwd": harness.cwd})
        assert "result" in frame.message
    assert harness.spawned == [1]
    loads = [f.session_id for f in harness.workers[1].received if f.method == "session/load"]
    assert loads == [root, grandchild]
    await harness.call("session/prompt", {"sessionId": grandchild, "prompt": []})
    assert "session/prompt" in harness.workers[1].methods()


async def test_a_child_whose_root_is_not_live_gets_its_own_worker_and_keeps_it(harness):
    root = _stored(harness)
    child = _stored(harness, parent_id=root)
    await harness.initialize()
    await harness.call("session/load", {"sessionId": child, "cwd": harness.cwd})
    await harness.call("session/load", {"sessionId": root, "cwd": harness.cwd})
    assert harness.spawned == [1]  # the tree stays in one worker


async def test_concurrent_loads_of_one_session_spawn_once(harness):
    session_id = _stored(harness)
    await harness.initialize()
    first = await harness.request("session/load", {"sessionId": session_id, "cwd": harness.cwd})
    second = await harness.request("session/load", {"sessionId": session_id, "cwd": harness.cwd})
    for request_id in (first, second):
        assert "result" in (await harness.response(request_id)).message
    assert harness.spawned == [1]
    assert harness.workers[1].methods().count("session/load") == 2


async def test_loading_an_unknown_session_is_resource_not_found_without_a_worker(harness):
    await harness.initialize()
    frame = await harness.call("session/load", {"sessionId": "missing", "cwd": harness.cwd})
    assert frame.message["error"] == {
        "code": -32002,
        "message": "Resource not found",
        "data": {"uri": "missing"},
    }
    assert harness.spawned == []


async def test_a_failed_load_unmaps_the_session_and_stops_the_worker(harness):
    session_id = _stored(harness)

    async def refuse(worker: FakeWorker, frame: Frame) -> None:
        await worker.reply(frame, error={"code": -32600, "message": "already open"})

    harness.handlers["session/load"] = refuse
    await harness.initialize()
    frame = await harness.call("session/load", {"sessionId": session_id, "cwd": harness.cwd})
    assert frame.message["error"]["message"] == "already open"
    await asyncio.wait_for(harness.workers[1].eof.wait(), TIMEOUT)
    frame = await harness.call("session/prompt", {"sessionId": session_id, "prompt": []})
    assert frame.message["error"]["code"] == -32002


# ---- close, delete, failures -----------------------------------------------------


async def test_closing_the_root_stops_its_worker(harness):
    await harness.initialize()
    session_id = await harness.new_session()
    frame = await harness.call("session/close", {"sessionId": session_id})
    assert frame.message["result"] == {}
    await asyncio.wait_for(harness.workers[1].eof.wait(), TIMEOUT)
    frame = await harness.call("session/prompt", {"sessionId": session_id, "prompt": []})
    assert frame.message["error"]["code"] == -32002


async def test_a_failed_new_session_stops_its_worker(harness):
    async def refuse(worker: FakeWorker, frame: Frame) -> None:
        await worker.reply(frame, error={"code": -32602, "message": "Invalid params"})

    harness.handlers["session/new"] = refuse
    await harness.initialize()
    frame = await harness.call("session/new", {"cwd": "relative", "mcpServers": []})
    assert frame.message["error"]["code"] == -32602
    await asyncio.wait_for(harness.workers[1].eof.wait(), TIMEOUT)


async def test_deleting_a_session_that_is_not_live_uses_a_worker_then_stops_it(harness):
    session_id = _stored(harness)
    await harness.initialize()
    frame = await harness.call(
        "_nooa/session/delete", {"sessionId": session_id, "cwd": harness.cwd}
    )
    assert frame.message["result"] == {}
    assert harness.workers[1].methods()[-1] == "_nooa/session/delete"
    await asyncio.wait_for(harness.workers[1].eof.wait(), TIMEOUT)


async def test_a_delete_without_cwd_searches_the_named_workspaces_and_passes_the_cwd_on(harness):
    session_id = _stored(harness)
    await harness.initialize()
    frame = await harness.call("_nooa/session/delete", {"sessionId": session_id})
    assert frame.message["error"]["code"] == -32002  # no workspace named yet
    assert harness.spawned == []

    await harness.call("session/list", {"cwd": harness.cwd})
    frame = await harness.call("_nooa/session/delete", {"sessionId": session_id})
    assert frame.message["result"] == {}
    [delete] = [f for f in harness.workers[1].received if f.method == "_nooa/session/delete"]
    assert delete.params == {"sessionId": session_id, "cwd": harness.cwd}


async def test_a_worker_exiting_mid_prompt_fails_the_prompt_and_answers_close(harness):
    async def hang(worker: FakeWorker, frame: Frame) -> None:
        await asyncio.Event().wait()

    async def die(worker: FakeWorker, frame: Frame) -> None:
        worker.writer.close()
        worker.kill()

    harness.handlers["session/prompt"] = hang
    harness.handlers["session/close"] = die
    await harness.initialize()
    session_id = await harness.new_session()
    prompt = await harness.request("session/prompt", {"sessionId": session_id, "prompt": []})
    await harness.workers[1].wait_for(lambda: "session/prompt" in harness.workers[1].methods())
    close = await harness.request("session/close", {"sessionId": session_id})
    close_frame = await harness.response(close)
    prompt_frame = await harness.response(prompt)
    assert close_frame.message["result"] == {}
    assert prompt_frame.message["error"]["code"] == -32603
    assert session_id in json.dumps(prompt_frame.message["error"])
    [update] = [f for f in harness.inbox if f.method == "session/update"]
    assert update.params["sessionId"] == session_id
    assert update.params["update"] == {
        "sessionUpdate": "session_info_update",
        "_meta": {"status": "worker_exited"},
    }
    assert harness.inbox.index(update) < harness.inbox.index(prompt_frame)
    frame = await harness.call("session/prompt", {"sessionId": session_id, "prompt": []})
    assert frame.message["error"]["code"] == -32002


async def test_a_failed_spawn_answers_the_request_with_an_internal_error(harness):
    harness.spawn_error[1] = OSError("no such interpreter")
    await harness.initialize()
    frame = await harness.call("session/new", {"cwd": harness.cwd, "mcpServers": []})
    assert frame.message["error"]["code"] == -32603
    assert "no such interpreter" in json.dumps(frame.message["error"])


async def test_end_of_input_stops_every_worker(harness):
    await harness.initialize()
    await harness.new_session()
    await harness.new_session()
    await harness.close()
    for worker in harness.workers.values():
        assert worker.eof.is_set()
        assert worker.killed  # the process group is always killed after the wait


async def test_a_load_sent_while_close_is_answered_keeps_the_session(harness):
    await harness.initialize()
    session_id = await harness.new_session()
    close = await harness.request("session/close", {"sessionId": session_id})
    load = await harness.request("session/load", {"sessionId": session_id, "cwd": harness.cwd})
    assert (await harness.response(close)).message["result"] == {}
    assert "result" in (await harness.response(load)).message
    frame = await harness.call("session/prompt", {"sessionId": session_id, "prompt": []})
    assert frame.message["result"] == {"stopReason": "end_turn"}
    assert harness.spawned == [1]
    assert not harness.workers[1].eof.is_set()


async def test_worker_sockets_are_closed_when_workers_end(harness):
    await harness.initialize()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        for _ in range(20):
            session_id = await harness.new_session()
            await harness.call("session/close", {"sessionId": session_id})
            k = harness.spawned[-1]
            await asyncio.wait_for(harness.workers[k].eof.wait(), TIMEOUT)
        await asyncio.sleep(0.2)  # let the router reap the last worker
        harness.workers.clear()
        gc.collect()
    # Only the router's sockets count; other objects collected here (the
    # store's SQLite connections from earlier tests) are not this test's.
    unclosed = [
        str(w.message)
        for w in caught
        if issubclass(w.category, ResourceWarning)
        and any(word in str(w.message) for word in ("StreamWriter", "transport", "socket"))
    ]
    assert unclosed == []


async def test_a_store_error_during_load_routing_is_an_internal_error(harness, monkeypatch):
    def broken(self: Any, path: Any) -> Any:
        raise RuntimeError("store is broken")

    monkeypatch.setattr(SessionStore, "_read_info", broken)
    await harness.initialize()
    request_id = await harness.request("session/load", {"sessionId": "abc", "cwd": harness.cwd})
    frame = await harness.response(request_id, timeout=3)
    assert frame.message["error"]["code"] == -32603
    assert "store is broken" in json.dumps(frame.message["error"])
    assert harness.spawned == []


async def test_a_failure_in_the_input_loop_is_logged_and_ends_the_router(harness, caplog):
    caplog.set_level(logging.ERROR, logger="nooa_coder.acp.router")

    def explode(frame: Frame) -> None:
        raise RuntimeError("handler exploded")

    harness.router._on_client_frame = explode  # type: ignore[method-assign]
    await harness.initialize_request_only()
    await asyncio.wait_for(harness.serving, TIMEOUT)
    assert any(
        record.exc_info and "handler exploded" in str(record.exc_info[1])
        for record in caplog.records
    )


async def test_session_list_gives_a_relative_recorded_workspace_the_store_directory(
    harness, monkeypatch
):
    """The old TUI recorded "../"; the entry's cwd is the directory the store belongs to."""
    import nooa_coder.session.store as store_module

    # create() resolves the workspace now; write the record as the old TUI did.
    monkeypatch.setattr(store_module, "_normalise_workspace", lambda workspace: str(workspace))
    old = _stored(harness, turns=1, workspace="../")
    monkeypatch.undo()
    monkeypatch.chdir(harness.cwd)
    for params in ({}, {"cwd": harness.cwd}):
        frame = await harness.call("session/list", params)
        [entry] = frame.message["result"]["sessions"]
        assert (entry["sessionId"], entry["cwd"]) == (old, str(Path(harness.cwd).resolve()))


async def test_session_list_leaves_out_sessions_the_agent_never_answered(harness):
    answered = _stored(harness, turns=1)
    _stored(harness, turns=1, answered=False)  # a failed first turn: noise
    named = _stored(harness, turns=1, answered=False)
    with harness.store.open(named) as handle:
        handle.set_title("Kept by name", user_set=True)
    frame = await harness.call("session/list", {"cwd": harness.cwd})
    assert {e["sessionId"] for e in frame.message["result"]["sessions"]} == {answered, named}
