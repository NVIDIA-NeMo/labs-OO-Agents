# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``nooa-coder --http``: ACP over WebSocket, one router per connection.

The end-to-end cases run ``fixtures/fake_agent.py --http`` as a real
process and talk to it with the ``acp`` library's own WebSocket client,
so they check that a standard ACP client can use the server.
"""

import asyncio
import os
import re
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from acp import PROTOCOL_VERSION, connect_to_agent, text_block
from acp.schema import AgentMessageChunk, AllowedOutcome, RequestPermissionResponse
from acp.ws import create_websocket_stream
from nooa_coder.acp.websocket import Gate, is_loopback_host
from websockets.asyncio.client import connect
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus

FAKE_AGENT = Path(__file__).parent / "fixtures" / "fake_agent.py"
TIMEOUT = 60
TOKEN = "test-token"


# ---- the gate ------------------------------------------------------------------


def _headers(**values: str) -> Headers:
    return Headers({name.replace("_", "-"): value for name, value in values.items()})


def test_the_gate_serves_only_the_acp_path():
    gate = Gate(token=None)
    assert gate.check("/acp", _headers()) is None
    assert gate.check("/acp?x=1", _headers()) is None
    status, _reason = gate.check("/", _headers())
    assert status == 404


def test_the_gate_checks_the_token_from_the_header_or_the_query():
    gate = Gate(token=TOKEN)
    assert gate.check("/acp", _headers(Authorization=f"Bearer {TOKEN}")) is None
    assert gate.check("/acp", _headers(Authorization=f"bearer {TOKEN}")) is None
    assert gate.check(f"/acp?token={TOKEN}", _headers()) is None
    for path, headers in [
        ("/acp", _headers()),
        ("/acp", _headers(Authorization="Bearer wrong")),
        ("/acp", _headers(Authorization=TOKEN)),
        ("/acp?token=wrong", _headers()),
    ]:
        status, _reason = gate.check(path, headers)
        assert status == 401, (path, headers)


def test_the_gate_accepts_loopback_and_listed_origins_only():
    gate = Gate(token=None, allowed_origins=["https://app.example/"])
    for origin in [
        "http://localhost:3000",
        "http://127.0.0.1",
        "http://[::1]:8080",
        "https://app.example",
    ]:
        assert gate.check("/acp", _headers(Origin=origin)) is None, origin
    for origin in ["https://evil.example", "null", "https://app.example.evil"]:
        status, _reason = gate.check("/acp", _headers(Origin=origin))
        assert status == 403, origin


def test_loopback_hosts():
    assert all(map(is_loopback_host, ["127.0.0.1", "localhost", "::1", "[::1]", "127.0.0.2"]))
    assert not any(map(is_loopback_host, ["0.0.0.0", "::", "10.0.0.1", "example.com"]))


# ---- end to end ------------------------------------------------------------------


class Client:
    """A client that records session updates and allows every permission request."""

    def __init__(self) -> None:
        self.updates: list[tuple[str, Any]] = []

    async def request_permission(self, **kwargs: Any) -> RequestPermissionResponse:
        return RequestPermissionResponse(
            outcome=AllowedOutcome(option_id="allow", outcome="selected")
        )

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        self.updates.append((session_id, update))

    def texts(self, session_id: str) -> list[str]:
        return [
            update.content.text
            for sid, update in self.updates
            if sid == session_id and isinstance(update, AgentMessageChunk)
        ]


class HttpServer:
    """``fake_agent.py --http --port 0`` and the address it listens on."""

    def __init__(self, process: asyncio.subprocess.Process, url: str) -> None:
        self.process = process
        self.url = url
        self.stderr: list[str] = []
        self._stderr_task = asyncio.create_task(self._read_stderr())

    @classmethod
    async def start(cls, tmp_path: Path, *args: str) -> "HttpServer":
        home = tmp_path / "home"
        home.mkdir(exist_ok=True)
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(FAKE_AGENT),
            "--agent",
            "coder_test_agents:EchoAgent",
            "--model",
            "fake",
            "--sessions-dir",
            str(tmp_path / "sessions"),
            "--http",
            "--port",
            "0",
            *args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=tmp_path,
            env={**os.environ, "HOME": str(home), "NOOA_CODER_TOKEN": TOKEN},
        )
        assert process.stderr is not None
        seen: list[str] = []

        async def listening() -> str:
            assert process.stderr is not None
            while line := (await process.stderr.readline()).decode():
                seen.append(line)
                if found := re.search(r"Listening for ACP connections on (\S+)", line):
                    return found.group(1)
            raise AssertionError("the server exited:\n" + "".join(seen))

        url = await asyncio.wait_for(listening(), TIMEOUT)
        server = cls(process, url)
        server.stderr.extend(seen)
        return server

    async def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        while line := await self.process.stderr.readline():
            self.stderr.append(line.decode())

    async def open(self, client: Client) -> tuple[Any, Any]:
        transport = await create_websocket_stream(
            self.url, headers={"Authorization": f"Bearer {TOKEN}"}
        )
        conn = connect_to_agent(client, transport, use_unstable_protocol=True)
        await asyncio.wait_for(conn.initialize(protocol_version=PROTOCOL_VERSION), TIMEOUT)
        return conn, transport

    async def stop(self) -> int:
        if self.process.returncode is None:
            self.process.send_signal(signal.SIGTERM)
        code = await asyncio.wait_for(self.process.wait(), TIMEOUT)
        await self._stderr_task
        return code


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "workspace"
    path.mkdir()
    return path


@pytest.fixture
async def http_server(tmp_path):
    started: list[HttpServer] = []

    async def start(*args: str) -> HttpServer:
        server = await HttpServer.start(tmp_path, *args)
        started.append(server)
        return server

    yield start
    for server in started:
        if server.process.returncode is None:
            server.process.kill()
            await server.process.wait()


async def test_a_session_survives_its_connection(http_server, workspace):
    """Prompt over one connection; after it closes, a new connection loads the session."""
    server = await http_server()
    client = Client()
    conn, transport = await server.open(client)
    session = await asyncio.wait_for(conn.new_session(cwd=str(workspace)), TIMEOUT)
    response = await asyncio.wait_for(
        conn.prompt(session_id=session.session_id, prompt=[text_block("hello")]), TIMEOUT
    )
    assert response.stop_reason == "end_turn"
    assert "Hi there." in "".join(client.texts(session.session_id))
    await conn.close()
    await transport.close()

    client = Client()
    conn, transport = await server.open(client)
    await asyncio.wait_for(
        conn.load_session(session_id=session.session_id, cwd=str(workspace), mcp_servers=[]),
        TIMEOUT,
    )
    assert "Hi there." in "".join(client.texts(session.session_id))  # the replayed transcript
    await conn.close()
    await transport.close()

    assert await server.stop() == 0, "".join(server.stderr)


async def test_two_connections_at_once(http_server, workspace):
    server = await http_server()
    clients = [Client(), Client()]
    opened = [await server.open(client) for client in clients]
    sessions = [
        await asyncio.wait_for(conn.new_session(cwd=str(workspace)), TIMEOUT)
        for conn, _transport in opened
    ]
    responses = await asyncio.wait_for(
        asyncio.gather(
            *(
                conn.prompt(session_id=session.session_id, prompt=[text_block("hello")])
                for (conn, _transport), session in zip(opened, sessions, strict=True)
            )
        ),
        TIMEOUT,
    )
    assert [response.stop_reason for response in responses] == ["end_turn", "end_turn"]
    for client, session in zip(clients, sessions, strict=True):
        assert "Hi there." in "".join(client.texts(session.session_id))
    for conn, transport in opened:
        await conn.close()
        await transport.close()
    assert await server.stop() == 0, "".join(server.stderr)


async def test_the_upgrade_is_refused_without_the_token_or_off_the_path(http_server):
    server = await http_server()
    base = server.url.removesuffix("/acp")
    refusals = [
        (server.url, {}, 401),
        (server.url, {"Authorization": "Bearer wrong"}, 401),
        (server.url, {"Authorization": f"Bearer {TOKEN}", "Origin": "https://evil.example"}, 403),
        (base + "/other", {"Authorization": f"Bearer {TOKEN}"}, 404),
    ]
    for url, headers, status in refusals:
        with pytest.raises(InvalidStatus) as refused:
            async with connect(url, additional_headers=headers):
                pass
        assert refused.value.response.status_code == status, (url, headers)

    async with connect(f"{server.url}?token={TOKEN}") as websocket:
        assert websocket.response.headers.get("Acp-Connection-Id")
    assert await server.stop() == 0, "".join(server.stderr)


async def test_sigterm_closes_open_connections_and_stops_their_workers(http_server, workspace):
    server = await http_server()
    client = Client()
    conn, transport = await server.open(client)
    await asyncio.wait_for(conn.new_session(cwd=str(workspace)), TIMEOUT)

    async def worker_pid() -> int:
        while True:
            for line in server.stderr:
                if found := re.search(r"worker \d+ pid (\d+) started", line):
                    return int(found.group(1))
            await asyncio.sleep(0.05)

    pid = await asyncio.wait_for(worker_pid(), TIMEOUT)

    assert await server.stop() == 0, "".join(server.stderr)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    await transport.close()
