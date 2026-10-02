# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The worker role: P3's adapter served on one end of a socket pair.

The client side is the ``acp`` library's own ``ClientSideConnection``, so
these tests check the worker speaks plain ACP.
"""

import asyncio
import os
import signal
import socket
from typing import Any

import pytest
from acp import PROTOCOL_VERSION
from acp.client.connection import ClientSideConnection
from acp.connection import StreamDirection
from acp.schema import AllowedOutcome, PermissionOption, RequestPermissionResponse, ToolCallUpdate
from coder_test_agents import ScriptedModels
from nooa_coder.acp.framing import FRAME_LIMIT
from nooa_coder.acp.server import CoderACPAgent
from nooa_coder.acp.worker import run_worker, serve_worker, start_parent_watchdog
from nooa_coder.session.registry import SessionRegistry
from nooa_coder.session.store import SessionStore, sessions_root

TIMEOUT = 20
ID_BASE = 1 << 32


class AskingAgent(CoderACPAgent):
    """The adapter plus a test extension method that asks the client for permission."""

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method != "test/ask":
            return await super().ext_method(method, params)
        response = await self._require_conn().request_permission(
            session_id=params["sessionId"],
            tool_call=ToolCallUpdate(tool_call_id="t1", title="Run it?"),
            options=[PermissionOption(option_id="allow", name="Allow", kind="allow_once")],
        )
        return {"outcome": response.outcome.outcome}


class Client:
    """A client that allows every permission request."""

    def __init__(self) -> None:
        self.updates: list[Any] = []

    async def request_permission(self, **kwargs: Any) -> RequestPermissionResponse:
        return RequestPermissionResponse(
            outcome=AllowedOutcome(option_id="allow", outcome="selected")
        )

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        self.updates.append((session_id, update))


@pytest.fixture
async def worker(workspace):
    """``(connection, frames_in, worker_task, store)``: a worker on a socket pair."""
    store = SessionStore(sessions_root(workspace))
    models = ScriptedModels()
    agent = AskingAgent(
        lambda root_store: SessionRegistry(root_store, agent_factory=models),
        agent_spec="coder_test_agents:EchoAgent",
    )
    ours, theirs = socket.socketpair()
    task = asyncio.create_task(serve_worker(theirs, id_base=ID_BASE, agent=agent))
    reader, writer = await asyncio.open_unix_connection(sock=ours, limit=FRAME_LIMIT)
    frames_in: list[dict[str, Any]] = []

    def record(event: Any) -> None:
        if event.direction == StreamDirection.INCOMING:
            frames_in.append(event.message)

    conn = ClientSideConnection(
        Client(), writer, reader, use_unstable_protocol=True, observers=[record]
    )
    await asyncio.wait_for(conn.initialize(protocol_version=PROTOCOL_VERSION), TIMEOUT)
    yield conn, frames_in, task, store, writer
    writer.close()
    await asyncio.wait_for(task, TIMEOUT)
    await conn.close()


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "workspace"
    path.mkdir()
    return path


async def test_worker_requests_start_at_the_id_base(worker, workspace):
    conn, frames_in, _task, _store, _writer = worker
    session = await asyncio.wait_for(conn.new_session(cwd=str(workspace)), TIMEOUT)
    for _ in range(2):
        answer = await asyncio.wait_for(
            conn.ext_method("test/ask", {"sessionId": session.session_id}), TIMEOUT
        )
        assert answer == {"outcome": "selected"}
    ids = [
        frame["id"] for frame in frames_in if frame.get("method") == "session/request_permission"
    ]
    assert ids == [ID_BASE, ID_BASE + 1]


async def test_session_close_is_served(worker, workspace):
    conn, _frames, _task, store, _writer = worker
    session = await asyncio.wait_for(conn.new_session(cwd=str(workspace)), TIMEOUT)
    assert store.is_active(session.session_id)
    await asyncio.wait_for(conn.close_session(session_id=session.session_id), TIMEOUT)
    assert not store.is_active(session.session_id)


async def test_end_of_stream_closes_the_sessions_and_ends_the_worker(worker, workspace):
    conn, _frames, task, store, writer = worker
    session = await asyncio.wait_for(conn.new_session(cwd=str(workspace)), TIMEOUT)
    assert store.is_active(session.session_id)
    writer.write_eof()
    await asyncio.wait_for(task, TIMEOUT)
    assert task.exception() is None
    assert not store.is_active(session.session_id)


def test_run_worker_returns_zero_on_end_of_stream():
    ours, theirs = socket.socketpair()
    ours.close()
    models = ScriptedModels()
    code = run_worker(
        theirs.detach(),
        id_base=ID_BASE,
        make_agent=lambda: CoderACPAgent(
            lambda store: SessionRegistry(store, agent_factory=models)
        ),
        watchdog=False,
    )
    assert code == 0


def test_the_watchdog_kills_the_process_group_when_the_parent_changes():
    parents = iter([100, 100, 1])
    killed: list[tuple[int, int]] = []
    thread = start_parent_watchdog(
        interval=0.01,
        getppid=lambda: next(parents, 1),
        killpg=lambda group, sig: killed.append((group, sig)),
    )
    thread.join(TIMEOUT)
    assert not thread.is_alive()
    assert killed == [(0, signal.SIGKILL)]
    assert thread.daemon
    assert os.getpid()  # the test process itself was not touched
