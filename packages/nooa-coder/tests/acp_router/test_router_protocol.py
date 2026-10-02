# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nooa-coder over real standard input and output, in both server modes.

The same protocol cases run against ``--single-process`` and the router
(the default), with real worker processes started from the test fixture
``fixtures/fake_agent.py``. Router-only cases check process groups and
cleanup when the router is killed.
"""

import asyncio
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from acp import PROTOCOL_VERSION
from nooa_coder.acp.framing import FRAME_LIMIT, Frame, encode
from nooa_coder.session.store import SessionStore

FAKE_AGENT = Path(__file__).parent / "fixtures" / "fake_agent.py"
TIMEOUT = 60
MODES = {"single": ["--single-process"], "router": []}


class Server:
    """A nooa-coder process driven by a raw JSON-lines client."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self.process = process
        self.inbox: list[Frame] = []
        self.stderr: list[str] = []
        self.stray: list[bytes] = []  # stdout lines that are not JSON-RPC objects
        self._next_id = 0
        self._stderr_task = asyncio.create_task(self._read_stderr())

    @classmethod
    async def start(cls, tmp_path: Path, *args: str, env: dict[str, str] | None = None) -> "Server":
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
            *args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=tmp_path,
            env={**os.environ, "HOME": str(home), **(env or {})},
            limit=FRAME_LIMIT,
        )
        return cls(process)

    async def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        async for line in self.process.stderr:
            self.stderr.append(line.decode(errors="replace"))

    def worker_pids(self) -> list[int]:
        return [
            int(match.group(1))
            for line in self.stderr
            if (match := re.search(r"worker \d+ pid (\d+) started", line))
        ]

    async def send(self, message: dict[str, Any]) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(encode(message))
        await self.process.stdin.drain()

    async def request(self, method: str, params: Any) -> int:
        self._next_id += 1
        await self.send({"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params})
        return self._next_id

    async def response(self, request_id: int, timeout: float = TIMEOUT) -> Frame:
        for frame in self.inbox:
            if frame.is_response and frame.id == request_id:
                return frame
        assert self.process.stdout is not None
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            frame = await asyncio.wait_for(self._next_frame(), max(remaining, 0.01))
            assert frame is not None, "".join(self.stderr[-30:])
            self.inbox.append(frame)
            if frame.is_response and frame.id == request_id:
                return frame

    async def _next_frame(self) -> Frame | None:
        """The next JSON-RPC frame; any other stdout line is kept in ``stray``."""
        assert self.process.stdout is not None
        while True:
            try:
                line = await self.process.stdout.readuntil(b"\n")
            except asyncio.IncompleteReadError as exc:
                if exc.partial.strip():
                    self.stray.append(exc.partial)
                return None
            try:
                message = json.loads(line)
            except ValueError:
                message = None
            if not isinstance(message, dict):
                self.stray.append(line)
                continue
            return Frame(line, message)

    async def call(self, method: str, params: Any) -> Frame:
        return await self.response(await self.request(method, params))

    async def initialize(self) -> Frame:
        return await self.call("initialize", {"protocolVersion": PROTOCOL_VERSION})

    async def new_session(self, cwd: Path) -> str:
        frame = await self.call("session/new", {"cwd": str(cwd), "mcpServers": []})
        assert "result" in frame.message, frame.message
        return frame.message["result"]["sessionId"]

    async def prompt(self, session_id: str, text: str = "hello") -> Frame:
        return await self.call(
            "session/prompt",
            {"sessionId": session_id, "prompt": [{"type": "text", "text": text}]},
        )

    def texts(self, session_id: str, kind: str) -> list[str]:
        return [
            frame.params["update"]["content"]["text"]
            for frame in self.inbox
            if frame.method == "session/update"
            and frame.session_id == session_id
            and frame.params["update"].get("sessionUpdate") == kind
        ]

    async def finish(self) -> int:
        assert self.process.stdin is not None
        self.process.stdin.close()
        code = await asyncio.wait_for(self.process.wait(), TIMEOUT)
        await asyncio.wait_for(self._stderr_task, TIMEOUT)
        return code

    async def kill(self) -> None:
        for pid in self.worker_pids():
            _kill_group(pid)
        if self.process.returncode is None:
            self.process.kill()
            await self.process.wait()
        self._stderr_task.cancel()


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _gone(pid: int) -> bool:
    """No such process, or a zombie (PID 1 in a container may not reap orphans)."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return True
    except OSError:
        result = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
        )
        return not result.stdout.strip() or result.stdout.strip().startswith("Z")
    return stat.rsplit(")", 1)[1].split()[0] == "Z"


def _ps(pids: list[int]) -> str:
    if not pids:
        return ""
    result = subprocess.run(
        ["ps", "-o", "pid,ppid,pgid,stat,args", "-p", ",".join(map(str, pids))],
        capture_output=True,
        text=True,
    )
    return result.stdout


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "workspace"
    path.mkdir()
    return path


@pytest.fixture
async def servers(tmp_path):
    started: list[Server] = []

    async def start(mode: str, *args: str, env: dict[str, str] | None = None) -> Server:
        server = await Server.start(tmp_path, *MODES[mode], *args, env=env)
        started.append(server)
        return server

    yield start
    for server in started:
        await server.kill()


# ---- protocol cases, both modes ------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
async def test_new_prompt_list_close_load_delete(mode, servers, workspace, tmp_path):
    server = await servers(mode)
    await server.initialize()
    session_id = await server.new_session(workspace)

    frame = await server.prompt(session_id)
    assert frame.message["result"] == {"stopReason": "end_turn"}
    assert "Hi there.\n\n" in server.texts(session_id, "agent_message_chunk")

    listed = await server.call("session/list", {})
    [entry] = listed.message["result"]["sessions"]
    assert entry["sessionId"] == session_id
    assert entry["_meta"]["dev.nooa/status"] == "idle"

    assert (await server.call("session/close", {"sessionId": session_id})).message["result"] == {}
    frame = await server.prompt(session_id)
    assert frame.message["error"]["code"] == -32002

    server.inbox.clear()
    frame = await server.call(
        "session/load", {"sessionId": session_id, "cwd": str(workspace), "mcpServers": []}
    )
    assert "result" in frame.message, frame.message
    assert server.texts(session_id, "user_message_chunk") == ["hello\n"]
    assert server.texts(session_id, "agent_message_chunk") == ["Hi there.\n\n"]
    assert (await server.call("session/close", {"sessionId": session_id})).message["result"] == {}

    frame = await server.call("_nooa/session/delete", {"sessionId": session_id})
    assert frame.message["result"] == {}
    assert not (tmp_path / "sessions" / f"{session_id}.db").exists()
    assert await server.finish() == 0


@pytest.mark.parametrize("mode", MODES)
async def test_unknown_sessions_and_methods(mode, servers):
    server = await servers(mode)
    await server.initialize()
    frame = await server.call("session/prompt", {"sessionId": "nope", "prompt": []})
    assert frame.message["error"]["code"] == -32002
    frame = await server.call("session/load", {"sessionId": "nope", "cwd": "/", "mcpServers": []})
    assert frame.message["error"] == {
        "code": -32002,
        "message": "Resource not found",
        "data": {"uri": "nope"},
    }
    frame = await server.call("authenticate", {"methodId": "x"})
    assert frame.message["error"]["code"] == -32601
    assert await server.finish() == 0


@pytest.mark.parametrize("mode", MODES)
async def test_two_sessions_prompt_at_the_same_time(mode, servers, workspace):
    server = await servers(mode)
    await server.initialize()
    first, second = await server.new_session(workspace), await server.new_session(workspace)
    requests = [
        await server.request(
            "session/prompt", {"sessionId": sid, "prompt": [{"type": "text", "text": "hi"}]}
        )
        for sid in (first, second)
    ]
    for request_id in requests:
        assert (await server.response(request_id)).message["result"] == {"stopReason": "end_turn"}
    for sid in (first, second):
        assert "Hi there.\n\n" in server.texts(sid, "agent_message_chunk")
    if mode == "router":
        pids = server.worker_pids()
        assert len(set(pids)) == 2
        print(_ps(pids))
    assert await server.finish() == 0


async def test_initialize_answers_are_identical(servers):
    answers = []
    for mode in MODES:
        server = await servers(mode)
        answers.append((await server.initialize()).message)
        assert await server.finish() == 0
    assert answers[0] == answers[1]
    assert json.dumps(answers[0], sort_keys=True) == json.dumps(answers[1], sort_keys=True)


async def test_session_new_latency(servers, workspace):
    timings = {}
    for mode in MODES:
        server = await servers(mode)
        await server.initialize()
        started = time.perf_counter()
        await server.new_session(workspace)
        timings[mode] = time.perf_counter() - started
        assert await server.finish() == 0
        if mode == "router":
            breakdown = [line.strip() for line in server.stderr if "forward_ms" in line]
            assert breakdown, "".join(server.stderr)
            print(breakdown[0])
    print(
        "session/new latency: "
        + ", ".join(f"{mode} {seconds * 1000:.0f} ms" for mode, seconds in timings.items())
    )
    assert max(timings.values()) < 15


# ---- router only -------------------------------------------------------------------


async def test_each_root_has_a_worker_in_its_own_process_group(servers, workspace, tmp_path):
    server = await servers("router")
    await server.initialize()
    await server.new_session(workspace)
    await server.new_session(workspace)
    pids = server.worker_pids()
    assert len(pids) == 2
    router_group = os.getpgid(server.process.pid)
    for pid in pids:
        assert os.getpgid(pid) == pid != router_group
    print(_ps([server.process.pid, *pids]))
    assert await server.finish() == 0
    deadline = time.monotonic() + 10
    while not all(_gone(pid) for pid in pids) and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    assert all(_gone(pid) for pid in pids)


async def test_a_child_session_loads_in_its_roots_worker(servers, workspace, tmp_path):
    """A child id is loaded by the worker that runs its root (attach-by-load)."""
    server = await servers("router")
    await server.initialize()
    root = await server.new_session(workspace)
    await server.prompt(root)
    store = SessionStore(tmp_path / "sessions")
    # A child record, as the registry writes one; the root's worker is its owner.
    handle = store.create(
        agent="coder_test_agents:EchoAgent",
        workspace=str(workspace),
        host="acp",
        parent_id=root,
        depth=1,
    )
    handle.close()
    frame = await server.call(
        "session/load", {"sessionId": handle.id, "cwd": str(workspace), "mcpServers": []}
    )
    assert "result" in frame.message, frame.message
    assert len(server.worker_pids()) == 1
    assert await server.finish() == 0


async def test_killing_the_router_ends_a_busy_worker_within_five_seconds(
    servers, workspace, tmp_path
):
    marker = tmp_path / "hot"
    server = await servers("router", "--fixture-hot", env={"NOOA_FIXTURE_MARKER": str(marker)})
    await server.initialize()
    session_id = await server.new_session(workspace)
    await server.request(
        "session/prompt", {"sessionId": session_id, "prompt": [{"type": "text", "text": "go"}]}
    )
    deadline = time.monotonic() + TIMEOUT
    while not marker.exists() and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    assert marker.exists(), "".join(server.stderr[-30:])
    [worker] = server.worker_pids()
    assert int(marker.read_text()) == worker  # the cell runs in the worker, blocking its loop
    server.process.kill()
    await server.process.wait()
    killed_at = time.monotonic()
    while not _gone(worker) and time.monotonic() - killed_at < 5:
        await asyncio.sleep(0.1)
    assert _gone(worker), _ps([worker])


async def test_sigterm_stops_the_router_and_its_workers(servers, workspace):
    server = await servers("router")
    await server.initialize()
    await server.new_session(workspace)
    [worker] = server.worker_pids()
    server.process.send_signal(signal.SIGTERM)
    assert await asyncio.wait_for(server.process.wait(), TIMEOUT) == 0
    deadline = time.monotonic() + 10
    while not _gone(worker) and time.monotonic() < deadline:
        await asyncio.sleep(0.1)
    assert _gone(worker)


async def test_stray_prints_in_the_router_and_workers_stay_off_the_acp_stream(servers, workspace):
    """The router reserves stdout for ACP, as single-process mode does.

    With ``--fixture-noisy`` the router prints while importing its router
    module (after start-up) and each worker prints while building a model.
    """
    server = await servers("router", "--fixture-noisy")
    await server.initialize()
    session_id = await server.new_session(workspace)
    assert (await server.prompt(session_id)).message["result"] == {"stopReason": "end_turn"}
    assert await server.finish() == 0
    assert server.stray == []
    noise = "".join(server.stderr)
    assert "stray output while importing the router" in noise
    assert "stray output while building a model" in noise
