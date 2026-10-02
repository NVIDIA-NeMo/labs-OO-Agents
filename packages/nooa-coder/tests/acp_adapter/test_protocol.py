# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nooa-coder over real stdio: a subprocess driven by the ACP library's client."""

import asyncio
import json
import signal
import sys
from contextlib import suppress
from pathlib import Path

import pytest
from acp import PROTOCOL_VERSION, RequestError, spawn_agent_process, text_block
from acp.connection import StreamDirection
from acp.schema import (
    AcceptElicitationResponse,
    AgentMessageChunk,
    AvailableCommandsUpdate,
    ClientCapabilities,
    ContentToolCallContent,
    ElicitationCapabilities,
    ElicitationFormCapabilities,
    EnvVariable,
    McpServerStdio,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
)

# Bounds a hang, not the expected duration.
_HANG_TIMEOUT = 60
_FIXTURES = Path(__file__).parent / "fixtures"
_FAKE_AGENT = _FIXTURES / "fake_agent.py"


@pytest.fixture(autouse=True)
def _user_dir_for_subprocesses(tmp_path, monkeypatch):
    """The server's sessions live under this user directory."""
    user = tmp_path / "user"
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user))
    monkeypatch.delenv("NOOA_SESSIONS_DIR", raising=False)
    return user


def _store(user_dir):
    from nooa_coder.session.store import SessionStore

    return SessionStore(user_dir / "sessions")


class _RecordingClient:
    def __init__(self) -> None:
        self.updates: list[tuple[str, object]] = []
        self.tool_started = asyncio.Event()
        self.commands_updated = asyncio.Event()
        self.message_updated = asyncio.Event()
        self.elicitations: list[tuple[str, object]] = []

    async def session_update(self, session_id: str, update: object, **kwargs) -> None:
        self.updates.append((session_id, update))
        if isinstance(update, ToolCallStart):
            self.tool_started.set()
        if isinstance(update, AvailableCommandsUpdate):
            self.commands_updated.set()
        if isinstance(update, AgentMessageChunk):
            self.message_updated.set()

    async def create_elicitation(self, message: str, mode: object, **kwargs):
        self.elicitations.append((message, mode))
        return AcceptElicitationResponse(action="accept", content={"answer": "dev"})

    def texts(self) -> list[str]:
        return [
            update.content.text
            for _, update in self.updates
            if isinstance(update, AgentMessageChunk)
            and isinstance(update.content, TextContentBlock)
        ]


def _spawn(client, *args, cwd, **kwargs):
    return spawn_agent_process(
        client,  # type: ignore[arg-type]
        sys.executable,
        str(_FAKE_AGENT),
        *args,
        cwd=cwd,
        use_unstable_protocol=True,
        **kwargs,
    )


def test_protocol_subprocess_imports_this_checkout(tmp_path):
    import subprocess

    from acp.transports import default_environment

    root = Path(__file__).resolve().parents[4]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, nooa, nooa_coder; print(json.dumps([nooa.__file__, nooa_coder.__file__]))",
        ],
        cwd=tmp_path,
        env=default_environment(),
        capture_output=True,
        text=True,
        check=True,
    )
    assert [Path(path).resolve() for path in json.loads(result.stdout)] == [
        root / "src/nooa/__init__.py",
        root / "packages/nooa-coder/src/nooa_coder/__init__.py",
    ]


async def test_a_session_runs_a_turn_over_stdio(tmp_path):
    client = _RecordingClient()
    incoming: list[dict] = []
    async with _spawn(client, cwd=tmp_path) as (connection, _process):
        connection._conn.add_observer(  # type: ignore[attr-defined]
            lambda event: (
                incoming.append(event.message)
                if event.direction is StreamDirection.INCOMING
                else None
            )
        )
        initialized = await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(tmp_path))
        await asyncio.wait_for(client.commands_updated.wait(), _HANG_TIMEOUT)
        response = await asyncio.wait_for(
            connection.prompt(session.session_id, [text_block("run smoke test")]),
            timeout=_HANG_TIMEOUT,
        )

    assert initialized.agent_info is not None and initialized.agent_info.name == "nooa-coder"
    assert response.stop_reason == "end_turn"
    assert {session_id for session_id, _ in client.updates} == {session.session_id}
    new_session_response = next(
        index for index, message in enumerate(incoming) if message.get("id") == 1
    )
    commands_notification = next(
        index
        for index, message in enumerate(incoming)
        if message.get("params", {}).get("update", {}).get("sessionUpdate")
        == "available_commands_update"
    )
    assert new_session_response < commands_notification
    started = next(update for _, update in client.updates if isinstance(update, ToolCallStart))
    assert (started.kind, started.status) == ("other", "in_progress")
    source = started.content[0] if started.content else None
    assert isinstance(source, ContentToolCallContent)
    assert isinstance(source.content, TextContentBlock)
    assert source.content.text.startswith("```python\n")
    completed = next(u for _, u in client.updates if isinstance(u, ToolCallProgress))
    assert completed.title == "Ran Python"
    assert "NOOA ACP smoke test passed.\n" in client.texts()


async def test_cancellation_finishes_open_tools_and_says_so(tmp_path):
    client = _RecordingClient()
    async with _spawn(client, "--blocking", cwd=tmp_path) as (connection, _process):
        await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(tmp_path))
        prompt = asyncio.create_task(
            connection.prompt(session.session_id, [text_block("wait forever")])
        )
        await asyncio.wait_for(client.tool_started.wait(), _HANG_TIMEOUT)
        await connection.cancel(session.session_id)
        response = await asyncio.wait_for(prompt, _HANG_TIMEOUT)

    assert response.stop_reason == "cancelled"
    started = next(update for _, update in client.updates if isinstance(update, ToolCallStart))
    failed = next(
        update
        for _, update in client.updates
        if isinstance(update, ToolCallProgress) and update.status == "failed"
    )
    assert failed.tool_call_id == started.tool_call_id
    assert failed.title == "Cancelled"
    assert "Stopped at your request.\n" in client.texts()


async def test_cancelling_a_shell_command_reads_as_cancellation(tmp_path):
    client = _RecordingClient()
    async with _spawn(client, "--shell", cwd=tmp_path) as (connection, _process):
        await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(tmp_path))
        prompt = asyncio.create_task(connection.prompt(session.session_id, [text_block("run")]))
        await asyncio.wait_for(client.tool_started.wait(), _HANG_TIMEOUT)
        await asyncio.sleep(0.5)  # let the shell command start
        await connection.cancel(session.session_id)
        response = await asyncio.wait_for(prompt, _HANG_TIMEOUT)

    assert response.stop_reason == "cancelled"
    rendered = "".join(str(update) for _, update in client.updates)
    assert "CancelledError" not in rendered


async def test_session_close_works_over_the_wire(tmp_path):
    client = _RecordingClient()
    async with _spawn(client, cwd=tmp_path) as (connection, _process):
        initialized = await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(tmp_path))
        await connection.close_session(session.session_id)
        with pytest.raises(RequestError, match="(?i)not found"):
            await asyncio.wait_for(
                connection.prompt(session.session_id, [text_block("still there?")]),
                timeout=_HANG_TIMEOUT,
            )
    capabilities = initialized.agent_capabilities.session_capabilities
    assert capabilities is not None and capabilities.close is not None
    assert capabilities.delete is None


async def test_a_question_is_answered_through_a_form_over_the_wire(tmp_path):
    client = _RecordingClient()
    async with _spawn(client, "--question", cwd=tmp_path) as (connection, _process):
        await connection.initialize(
            PROTOCOL_VERSION,
            client_capabilities=ClientCapabilities(
                elicitation=ElicitationCapabilities(form=ElicitationFormCapabilities())
            ),
        )
        session = await connection.new_session(str(tmp_path))
        response = await asyncio.wait_for(
            connection.prompt(session.session_id, [text_block("push it")]), _HANG_TIMEOUT
        )
    assert response.stop_reason == "end_turn"
    [(message, mode)] = client.elicitations
    assert message == "Which branch?"
    assert mode.requested_schema.properties["answer"].enum == ["main", "dev"]
    assert client.texts()[-1] == "Using the answer.\n"


async def test_the_delete_extension_works_over_the_wire(tmp_path, _user_dir_for_subprocesses):
    client = _RecordingClient()
    async with _spawn(client, cwd=tmp_path) as (connection, _process):
        await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(tmp_path))
        await asyncio.wait_for(
            connection.prompt(session.session_id, [text_block("hi")]), _HANG_TIMEOUT
        )
        await connection.close_session(session.session_id)
        assert (
            await connection.ext_method("nooa/session/delete", {"sessionId": session.session_id})
            == {}
        )
        assert (await connection.list_sessions()).sessions == []
    assert not _store(_user_dir_for_subprocesses).path_for(session.session_id).exists()


async def test_the_mcp_handoff_trace_records_names_only(tmp_path):
    client = _RecordingClient()
    trace = tmp_path / "handoff.jsonl"
    secret = "private-mcp-environment-value"
    server = McpServerStdio(
        name="client_probe",
        command=sys.executable,
        args=[str(_FIXTURES / "mcp_probe.py"), "--journal", str(tmp_path / "probe.jsonl")],
        env=[EnvVariable(name="TEST_SECRET", value=secret)],
    )
    async with _spawn(client, cwd=tmp_path, env={"NOOA_ACP_MCP_TRACE": str(trace)}) as (
        connection,
        process,
    ):
        await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(tmp_path), mcp_servers=[server])
        await connection.close_session(session.session_id)
        await connection.load_session(str(tmp_path), session.session_id, mcp_servers=[])
        await connection.close_session(session.session_id)
        content = trace.read_text()
    assert secret not in content
    assert [json.loads(line) for line in content.splitlines()] == [
        {"pid": process.pid, "event": "trace_started"},
        {
            "pid": process.pid,
            "event": "session/new",
            "mcpServersField": "list",
            "servers": [{"name": "client_probe", "transport": "stdio"}],
        },
        {"pid": process.pid, "event": "session/load", "mcpServersField": "list", "servers": []},
    ]
    # The probe server was started for the session.
    assert "start" in (tmp_path / "probe.jsonl").read_text()


async def test_the_tee_records_the_conversation(tmp_path):
    import stat

    client = _RecordingClient()
    log = tmp_path / "tee.jsonl"
    async with _spawn(client, "--tee", str(log), cwd=tmp_path) as (connection, _process):
        await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(tmp_path))
        await asyncio.wait_for(
            connection.prompt(session.session_id, [text_block("hi")]), _HANG_TIMEOUT
        )
        await connection.close_session(session.session_id)
    for _ in range(100):  # the writer thread flushes on exit
        if log.exists() and '"session/close"' in log.read_text():
            break
        await asyncio.sleep(0.05)
    records = [json.loads(line) for line in log.read_text().splitlines()]
    methods_in = [r["frame"].get("method") for r in records if r["dir"] == "in"]
    assert methods_in[:3] == ["initialize", "session/new", "session/prompt"]
    assert any(r["dir"] == "out" and r["frame"].get("method") == "session/update" for r in records)
    assert stat.S_IMODE(log.stat().st_mode) == 0o600


async def test_an_open_session_is_hidden_and_explains_itself(tmp_path, _user_dir_for_subprocesses):
    store = _store(_user_dir_for_subprocesses)
    client = _RecordingClient()
    async with _spawn(client, cwd=tmp_path) as (connection, _process):
        await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(tmp_path))
        await asyncio.wait_for(
            connection.prompt(session.session_id, [text_block("hi")]), _HANG_TIMEOUT
        )
        await connection.close_session(session.session_id)
        listed = await connection.list_sessions(cwd=str(tmp_path))
        assert [s.session_id for s in listed.sessions] == [session.session_id]

        with store.open(session.session_id):  # another owner holds it
            assert (await connection.list_sessions(cwd=str(tmp_path))).sessions == []
            with pytest.raises(RequestError, match="already open") as caught:
                await connection.load_session(session_id=session.session_id, cwd=str(tmp_path))
            assert "Close it in the other client or tab" in str(caught.value)

        await connection.load_session(session_id=session.session_id, cwd=str(tmp_path))
        await connection.close_session(session.session_id)


@pytest.mark.parametrize("shutdown", ["eof", "sigterm"])
@pytest.mark.parametrize("during_turn", [False, True])
async def test_client_shutdown_releases_sessions_for_resume(
    tmp_path, _user_dir_for_subprocesses, shutdown, during_turn
):
    """A client need not call session/close before stopping its agent server."""
    store = _store(_user_dir_for_subprocesses)
    client = _RecordingClient()
    args = ["--blocking"] if during_turn else []
    async with _spawn(client, *args, cwd=tmp_path) as (connection, process):
        await connection.initialize(PROTOCOL_VERSION)
        session = await connection.new_session(str(tmp_path))
        prompt = asyncio.create_task(
            connection.prompt(session.session_id, [text_block("Remember this conversation")])
        )
        if during_turn:
            await asyncio.wait_for(client.tool_started.wait(), _HANG_TIMEOUT)
        else:
            await asyncio.wait_for(prompt, _HANG_TIMEOUT)
        if shutdown == "sigterm":
            process.send_signal(signal.SIGTERM)
        else:
            process.stdin.close()
        await asyncio.wait_for(process.wait(), _HANG_TIMEOUT)
        assert process.returncode == 0
        with suppress(Exception):
            await prompt

    assert not store.is_active(session.session_id)
    async with _spawn(_RecordingClient(), cwd=tmp_path) as (connection, _process):
        await connection.initialize(PROTOCOL_VERSION)
        listed = await connection.list_sessions(cwd=str(tmp_path))
        assert [item.session_id for item in listed.sessions] == [session.session_id]
        await connection.load_session(cwd=str(tmp_path), session_id=session.session_id)
        await connection.close_session(session.session_id)


async def test_stray_prints_do_not_reach_the_acp_stream(tmp_path):
    """Only JSON-RPC frames go to stdout; anything else the process prints goes to stderr.

    Likewise only the transport reads standard input.
    """
    import os

    from acp.transports import default_environment

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(_FAKE_AGENT),
        "--noisy",
        cwd=tmp_path,
        env={
            **default_environment(),
            "NEMO_OO_USER_DIR": os.environ["NEMO_OO_USER_DIR"],
            "NOISY_STDIN_LOG": str(tmp_path / "stdin-read.bin"),
        },
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": 1}},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/new",
            "params": {"cwd": str(tmp_path), "mcpServers": []},
        },
    ]
    lines: list[bytes] = []
    for request in requests:
        process.stdin.write(json.dumps(request).encode() + b"\n")
        await process.stdin.drain()
        while True:
            line = await asyncio.wait_for(process.stdout.readline(), _HANG_TIMEOUT)
            assert line, "the server closed its output"
            lines.append(line)
            if json.loads(line).get("id") == request["id"]:
                break
    process.stdin.close()
    rest, stderr = await asyncio.wait_for(process.communicate(), _HANG_TIMEOUT)
    assert process.returncode == 0
    assert all(isinstance(json.loads(line), dict) for line in lines + rest.splitlines())
    assert b"noise on stdout" in stderr
    # Standard input is the client's: the process's own readers see nothing.
    assert (tmp_path / "stdin-read.bin").read_bytes() == b"<eof>"
