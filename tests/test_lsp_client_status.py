# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for LSPClient health-status tracking.

ValidationPolicy gates must be able to distinguish "the code has an error"
from "the language server crashed". LSPClient therefore exposes a ``status``
attribute with a four-state lifecycle:

    UNKNOWN   - constructed, initialize handshake not yet completed
    COMPLETE  - initialize succeeded; the client is healthy
    DEGRADED  - the server sent undecodable data; results may be unreliable
    FAILED    - connection lost or server process gone; no query can be trusted

These tests exercise the transitions without a real language server, except
for one integration test that uses pyright-langserver when available.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import Callable

import pytest

from nooa.lsp.client import LSPClient, LSPClientError, LSPClientStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeStdin:
    """Minimal stdin stand-in: accepts writes, lets us simulate a broken pipe."""

    def __init__(self):
        self.writes: list[bytes] = []
        self.broken = False
        # optional callback(data) for reactive servers
        self.on_write: Callable[[bytes], None] | None = None

    def write(self, data: bytes) -> None:
        if self.broken:
            raise ConnectionResetError("broken pipe")
        self.writes.append(data)
        if self.on_write is not None:
            self.on_write(data)


class _FakeStdout:
    """Byte pipe whose reads block until data arrives or the pipe closes.

    Nothing is buffered ahead of time: the owning ``_FakeProcess`` pushes
    response frames *reactively* when the matching request is written, so a
    scripted response can never be consumed before its request exists (the bug
    that made an eagerly-buffered fake hang). Reads poll on a short sleep,
    which is deterministic — there is no condition/event wakeup to lose.
    """

    def __init__(self):
        self._buffer = bytearray()
        self._pos = 0
        self._closed = False

    def close(self) -> None:
        self._closed = True

    def push(self, body: bytes) -> None:
        """Append one Content-Length framed message body."""
        self._buffer += f"Content-Length: {len(body)}\r\n\r\n".encode() + body

    def at_eof(self) -> bool:
        return self._closed and self._pos >= len(self._buffer)

    async def readline(self) -> bytes:
        while True:
            if self._pos < len(self._buffer):
                try:
                    end = self._buffer.index(b"\n", self._pos) + 1
                except ValueError:
                    end = len(self._buffer)
                line = bytes(self._buffer[self._pos:end])
                self._pos = end
                return line
            if self._closed:
                return b""
            await asyncio.sleep(0.001)

    async def readexactly(self, n: int) -> bytes:
        while self._pos + n > len(self._buffer):
            if self._closed:
                raise asyncio.IncompleteReadError(bytes(self._buffer[self._pos:]), n)
            await asyncio.sleep(0.001)
        data = bytes(self._buffer[self._pos: self._pos + n])
        self._pos += n
        return data


class _FakeProcess:
    """Reactive subprocess stand-in for lifecycle tests.

    ``responses`` maps a request id to the raw frame bodies to push when a
    request with that id is written to stdin (e.g. ``{1: [init_response]}``).
    Because responses are pushed only in reply to their request, the read loop
    can never consume one early. Any request without a scripted response is
    auto-acked with a null result, and the LSP ``exit`` notification triggers a
    clean exit — close enough to a real server to drive the whole lifecycle.

    ``wait()`` blocks until the process "dies"; ``kill()`` and
    ``exit_cleanly()`` simulate abrupt and cooperative termination.
    """

    def __init__(self, responses: dict[int, list[bytes]] | None = None):
        self.stdin = _FakeStdin()
        self.stdout = _FakeStdout()
        self.returncode: int | None = None
        self._died = asyncio.Event()
        self._responses = responses or {}
        self.stdin.on_write = self._on_write

    def _on_write(self, data: bytes) -> None:
        if self._died.is_set():
            return
        try:
            body = data.split(b"\r\n\r\n", 1)[1]
            msg = json.loads(body)
        except (IndexError, json.JSONDecodeError):
            return
        # A well-behaved LSP server exits on the `exit` notification.
        if msg.get("method") == "exit":
            self.exit_cleanly()
            return
        if "id" not in msg or "method" not in msg:
            return  # notification — no response needed
        msg_id = msg["id"]
        if msg_id in self._responses:
            for frame in self._responses[msg_id]:
                self.stdout.push(frame)
        else:
            self.stdout.push(
                json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": None}).encode()
            )

    async def wait(self) -> int:
        await self._died.wait()
        return self.returncode or 0

    def kill(self) -> None:
        self.returncode = -9
        self._died.set()
        self.stdout.close()

    def exit_cleanly(self) -> None:
        self.returncode = 0
        self._died.set()
        self.stdout.close()


def _initialize_response(msg_id: int = 1) -> bytes:
    return json.dumps(
        {"jsonrpc": "2.0", "id": msg_id, "result": {"capabilities": {"renameProvider": True}}}
    ).encode()


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


class TestInitialStatus:
    def test_status_is_unknown_after_construction(self):
        client = LSPClient(command=["fake-server"], root_uri="file:///tmp")
        assert client.status == LSPClientStatus.UNKNOWN

    def test_status_constant_values_are_stable_strings(self):
        # Gates compare against these strings; changing them is a breaking change.
        assert LSPClientStatus.UNKNOWN == "UNKNOWN"
        assert LSPClientStatus.COMPLETE == "COMPLETE"
        assert LSPClientStatus.DEGRADED == "DEGRADED"
        assert LSPClientStatus.FAILED == "FAILED"

    async def test_send_before_start_raises_with_status_context(self):
        client = LSPClient(command=["fake-server"], root_uri="file:///tmp")
        with pytest.raises(LSPClientError, match="UNKNOWN"):
            client._send({"jsonrpc": "2.0", "method": "initialized"})


# ---------------------------------------------------------------------------
# Successful startup
# ---------------------------------------------------------------------------


class TestStartupTransition:
    async def test_status_complete_after_successful_initialize(self, monkeypatch):
        client = LSPClient(command=["fake-server"], root_uri="file:///tmp")
        process = _FakeProcess(responses={1: [_initialize_response()]})

        async def fake_exec(*args, **kwargs):
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        result = await client.start()

        assert client.status == LSPClientStatus.COMPLETE
        assert result.capabilities == {"renameProvider": True}
        await client.stop()

    async def test_queries_work_when_complete(self, monkeypatch):
        client = LSPClient(command=["fake-server"], root_uri="file:///tmp")
        definition_result = json.dumps(
            {"jsonrpc": "2.0", "id": 2, "result": [{"uri": "file:///tmp/x.py"}]}
        ).encode()
        process = _FakeProcess(responses={1: [_initialize_response()], 2: [definition_result]})

        async def fake_exec(*args, **kwargs):
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        await client.start()

        result = await client.definition("file:///tmp/x.py", {"line": 1, "character": 2})
        assert result == [{"uri": "file:///tmp/x.py"}]
        assert client.status == LSPClientStatus.COMPLETE
        await client.stop()


# ---------------------------------------------------------------------------
# DEGRADED on undecodable server output
# ---------------------------------------------------------------------------


class TestDegradedTransition:
    async def test_json_parse_error_marks_degraded(self, monkeypatch):
        client = LSPClient(command=["fake-server"], root_uri="file:///tmp")
        # The bad frame is pushed alongside the initialize response.
        process = _FakeProcess(responses={1: [_initialize_response(), b"this is not json"]})

        async def fake_exec(*args, **kwargs):
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        await client.start()

        # The read loop races the test: with a fully-buffered fake stream it may
        # consume the bad frame before or shortly after start() returns. Poll
        # until degradation is observed rather than asserting an intermediate
        # COMPLETE that may never be visible.
        for _ in range(100):
            await asyncio.sleep(0.01)
            if client.status == LSPClientStatus.DEGRADED:
                break
        assert client.status == LSPClientStatus.DEGRADED
        assert client._decode_errors == 1
        await client.stop()

    async def test_degraded_client_can_still_answer_queries(self, monkeypatch):
        client = LSPClient(command=["fake-server"], root_uri="file:///tmp")
        definition_result = json.dumps({"jsonrpc": "2.0", "id": 2, "result": []}).encode()
        # Bad frame arrives with initialize; the valid response answers the later query.
        process = _FakeProcess(
            responses={1: [_initialize_response(), b"garbage"], 2: [definition_result]}
        )

        async def fake_exec(*args, **kwargs):
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        await client.start()
        for _ in range(100):
            await asyncio.sleep(0.01)
            if client.status == LSPClientStatus.DEGRADED:
                break
        assert client.status == LSPClientStatus.DEGRADED

        result = await client.definition("file:///tmp/x.py", {"line": 0, "character": 0})
        assert result == []
        assert client.status == LSPClientStatus.DEGRADED
        await client.stop()


# ---------------------------------------------------------------------------
# FAILED on connection loss
# ---------------------------------------------------------------------------


class TestFailedTransition:
    async def test_unexpected_process_death_marks_failed(self, monkeypatch):
        client = LSPClient(command=["fake-server"], root_uri="file:///tmp")
        process = _FakeProcess(responses={1: [_initialize_response()]})

        async def fake_exec(*args, **kwargs):
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        await client.start()
        assert client.status == LSPClientStatus.COMPLETE

        process.kill()
        for _ in range(100):
            await asyncio.sleep(0.01)
            if client.status == LSPClientStatus.FAILED:
                break
        assert client.status == LSPClientStatus.FAILED

    async def test_clean_stop_marks_failed(self, monkeypatch):
        client = LSPClient(command=["fake-server"], root_uri="file:///tmp")
        # shutdown (id=2) is auto-acked; the fake exits when `exit` is written.
        process = _FakeProcess(responses={1: [_initialize_response()]})

        async def fake_exec(*args, **kwargs):
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        await client.start()
        # The fake server exits when stop()'s `exit` notification reaches stdin.
        await client.stop()
        assert client.status == LSPClientStatus.FAILED

    async def test_query_after_failure_raises_instead_of_hanging(self, monkeypatch):
        """The critical gate property: a dead server must not look like clean code."""
        client = LSPClient(command=["fake-server"], root_uri="file:///tmp")
        process = _FakeProcess(responses={1: [_initialize_response()]})

        async def fake_exec(*args, **kwargs):
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        await client.start()
        process.kill()
        for _ in range(100):
            await asyncio.sleep(0.01)
            if client.status == LSPClientStatus.FAILED:
                break

        with pytest.raises(LSPClientError, match="FAILED"):
            await client.definition("file:///tmp/x.py", {"line": 0, "character": 0})


# ---------------------------------------------------------------------------
# Integration: real pyright server, real kill
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    shutil.which("pyright-langserver") is None, reason="pyright-langserver not installed"
)
class TestRealServerLifecycle:
    async def test_full_lifecycle_with_pyright(self, tmp_path):
        client = LSPClient(
            command=["pyright-langserver", "--stdio"], root_uri=tmp_path.as_uri()
        )
        assert client.status == LSPClientStatus.UNKNOWN

        await client.start()
        try:
            assert client.status == LSPClientStatus.COMPLETE

            assert client.process is not None
            client.process.kill()
            for _ in range(200):
                await asyncio.sleep(0.01)
                if client.status == LSPClientStatus.FAILED:
                    break
            assert client.status == LSPClientStatus.FAILED

            with pytest.raises(LSPClientError, match="FAILED"):
                await client.definition(
                    f"{tmp_path.as_uri()}/x.py", {"line": 0, "character": 0}
                )
        finally:
            # Cancel the read loop, which is otherwise wedged on readline():
            # pyright's node launcher spawns a child that keeps the stdout pipe
            # open, so killing the parent never delivers EOF.
            await client.stop()
