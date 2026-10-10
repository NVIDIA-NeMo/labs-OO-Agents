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
import importlib
import json
import shutil
from collections.abc import Callable

import pytest

from nooa.lsp.client import LSPClient, LSPClientError, LSPClientStatus
from nooa.lsp.facade import LSPDocumentFacade
from nooa.lsp.protocol import (
    Diagnostic,
    DocumentSymbol,
    Location,
    Position,
    Range,
    SymbolInformation,
    TextDocumentEdit,
    TextEdit,
    WorkspaceEdit,
    WorkspaceSymbol,
)
from nooa.lsp.registry import LSPServerConfig, LSPServerRegistry
from nooa.lsp.skill import LSPSkill

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
        exec_kwargs = {}

        async def fake_exec(*args, **kwargs):
            exec_kwargs.update(kwargs)
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        result = await client.start()

        assert client.status == LSPClientStatus.COMPLETE
        assert result.capabilities == {"renameProvider": True}
        assert exec_kwargs["stdin"] == asyncio.subprocess.PIPE
        assert exec_kwargs["stdout"] == asyncio.subprocess.PIPE
        assert exec_kwargs["stderr"] == asyncio.subprocess.DEVNULL
        await client.stop()

    async def test_queries_work_when_complete(self, monkeypatch):
        client = LSPClient(command=["fake-server"], root_uri="file:///tmp")
        definition_result = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "result": [
                    {
                        "uri": "file:///tmp/x.py",
                        "range": {
                            "start": {"line": 0, "character": 0},
                            "end": {"line": 0, "character": 4},
                        },
                    }
                ],
            }
        ).encode()
        process = _FakeProcess(responses={1: [_initialize_response()], 2: [definition_result]})

        async def fake_exec(*args, **kwargs):
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        await client.start()

        result = await client.definition("file:///tmp/x.py", {"line": 1, "character": 2})
        assert result == [
            Location(
                uri="file:///tmp/x.py",
                range={
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 4},
                },
            )
        ]
        assert client.status == LSPClientStatus.COMPLETE
        await client.stop()


class TestDefinitionResult:
    @pytest.mark.parametrize(
        ("raw_result", "expected"),
        [
            (None, []),
            (
                {
                    "uri": "file:///tmp/x.py",
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 4},
                    },
                },
                [
                    Location(
                        uri="file:///tmp/x.py",
                        range={
                            "start": {"line": 0, "character": 0},
                            "end": {"line": 0, "character": 4},
                        },
                    )
                ],
            ),
            (
                [
                    {
                        "uri": "file:///tmp/x.py",
                        "range": {
                            "start": {"line": 0, "character": 0},
                            "end": {"line": 0, "character": 4},
                        },
                    }
                ],
                [
                    Location(
                        uri="file:///tmp/x.py",
                        range={
                            "start": {"line": 0, "character": 0},
                            "end": {"line": 0, "character": 4},
                        },
                    )
                ],
            ),
        ],
    )
    async def test_definition_always_returns_a_list(
        self, monkeypatch, raw_result, expected
    ):
        client = LSPClient(command=["fake-server"], root_uri="file:///tmp")

        async def fake_send_request(method, params=None):
            return raw_result

        monkeypatch.setattr(client, "send_request", fake_send_request)

        result = await client.definition("file:///tmp/x.py", {"line": 0, "character": 0})

        assert result == expected
        assert isinstance(result, list)

    async def test_definition_rejects_malformed_result(self, monkeypatch):
        client = LSPClient(command=["fake-server"], root_uri="file:///tmp")

        async def fake_send_request(method, params=None):
            return "not a location"

        monkeypatch.setattr(client, "send_request", fake_send_request)

        with pytest.raises(LSPClientError, match="textDocument/definition"):
            await client.definition("file:///tmp/x.py", {"line": 0, "character": 0})


class TestLSPPackageInterface:
    def test_skill_and_protocol_models_are_importable(self):
        package = importlib.import_module("nooa.lsp")
        skill_module = importlib.import_module("nooa.lsp.skill")

        assert skill_module.LSPSkill is package.LSPSkill
        assert package.WorkspaceSymbol is WorkspaceSymbol

    def test_registry_resolves_and_registers_extensions(self):
        registry = LSPServerRegistry()

        assert registry.get_server_for_extension(".py").command == [
            "pyright-langserver",
            "--stdio",
        ]
        assert registry.get_server_for_extension(".unknown") is None

        custom = LSPServerConfig(
            command=["custom-lsp"], extensions=[".custom"]
        )
        registry.register(custom)
        assert registry.get_server_for_extension(".custom") is custom

    async def test_references_are_parsed_as_locations(self, monkeypatch):
        client = LSPClient(command=["fake-server"], root_uri="file:///tmp")
        raw_location = {
            "uri": "file:///tmp/x.py",
            "range": {
                "start": {"line": 1, "character": 2},
                "end": {"line": 1, "character": 5},
            },
        }
        calls = []

        async def fake_send_request(method, params=None):
            calls.append((method, params))
            return [raw_location]

        monkeypatch.setattr(client, "send_request", fake_send_request)
        result = await client.references(
            "file:///tmp/x.py", {"line": 1, "character": 2}, True
        )

        assert result == [Location.model_validate(raw_location)]
        assert calls == [
            (
                "textDocument/references",
                {
                    "textDocument": {"uri": "file:///tmp/x.py"},
                    "position": {"line": 1, "character": 2},
                    "context": {"includeDeclaration": True},
                },
            )
        ]

    async def test_document_symbols_are_typed_recursively(self, monkeypatch):
        client = LSPClient(command=["fake-server"], root_uri="file:///tmp")
        raw_symbol = {
            "name": "Outer",
            "kind": 5,
            "range": {
                "start": {"line": 0, "character": 0},
                "end": {"line": 4, "character": 0},
            },
            "selectionRange": {
                "start": {"line": 0, "character": 6},
                "end": {"line": 0, "character": 11},
            },
            "children": [
                {
                    "name": "Inner",
                    "kind": 6,
                    "range": {
                        "start": {"line": 1, "character": 4},
                        "end": {"line": 2, "character": 0},
                    },
                    "selectionRange": {
                        "start": {"line": 1, "character": 8},
                        "end": {"line": 1, "character": 13},
                    },
                }
            ],
        }

        async def fake_send_request(method, params=None):
            assert method == "textDocument/documentSymbol"
            return [raw_symbol]

        monkeypatch.setattr(client, "send_request", fake_send_request)
        symbols = await client.document_symbol("file:///tmp/x.py")

        assert isinstance(symbols[0], DocumentSymbol)
        assert isinstance(symbols[0].children[0], DocumentSymbol)
        assert symbols[0].children[0].name == "Inner"

    async def test_skill_resolves_nested_typed_document_symbol(
        self, monkeypatch
    ):
        skill = LSPSkill(root_uri="file:///tmp")
        facade = LSPDocumentFacade(
            LSPClient(command=["fake-server"], root_uri="file:///tmp"),
            "file:///tmp/x.py",
        )
        nested_symbol = DocumentSymbol(
            name="Target",
            kind=12,
            range=Range(
                start=Position(line=2, character=0),
                end=Position(line=2, character=12),
            ),
            selectionRange=Range(
                start=Position(line=2, character=4),
                end=Position(line=2, character=10),
            ),
        )
        parent_symbol = DocumentSymbol(
            name="Parent",
            kind=5,
            range=Range(
                start=Position(line=0, character=0),
                end=Position(line=4, character=0),
            ),
            selectionRange=Range(
                start=Position(line=0, character=6),
                end=Position(line=0, character=12),
            ),
            children=[nested_symbol],
        )
        captured = []

        async def fake_for_file(filepath):
            return facade

        async def fake_document_symbols():
            return [parent_symbol]

        async def fake_references(line, character, include_declaration=True):
            captured.append((line, character))
            return []

        monkeypatch.setattr(skill, "for_file", fake_for_file)
        monkeypatch.setattr(facade, "document_symbols", fake_document_symbols)
        monkeypatch.setattr(facade, "references", fake_references)

        assert await skill.find_references("src/x.py", "Target") == []
        assert captured == [(2, 4)]

    async def test_document_symbol_information_is_typed(self, monkeypatch):
        client = LSPClient(command=["fake-server"], root_uri="file:///tmp")
        raw_symbol = {
            "name": "Target",
            "kind": 12,
            "containerName": "module",
            "location": {
                "uri": "file:///tmp/x.py",
                "range": {
                    "start": {"line": 0, "character": 4},
                    "end": {"line": 0, "character": 10},
                },
            },
        }

        async def fake_send_request(method, params=None):
            return [raw_symbol]

        monkeypatch.setattr(client, "send_request", fake_send_request)
        symbols = await client.document_symbol("file:///tmp/x.py")

        assert isinstance(symbols[0], SymbolInformation)
        assert symbols[0].location.range.start == Position(line=0, character=4)

    async def test_workspace_symbols_are_typed_and_exposed_by_facade(
        self, monkeypatch
    ):
        client = LSPClient(command=["fake-server"], root_uri="file:///tmp")
        raw_symbol = {
            "name": "Target",
            "kind": 12,
            "containerName": "module",
            "location": {"uri": "file:///tmp/x.py"},
        }
        calls = []

        async def fake_send_request(method, params=None):
            calls.append((method, params))
            return [raw_symbol]

        monkeypatch.setattr(client, "send_request", fake_send_request)
        facade = LSPDocumentFacade(client, "file:///tmp/x.py")
        symbols = await facade.workspace_symbols("Target")

        assert symbols == [WorkspaceSymbol.model_validate(raw_symbol)]
        assert calls == [("workspace/symbol", {"query": "Target"})]

    async def test_rename_returns_a_typed_workspace_edit(self, monkeypatch):
        client = LSPClient(command=["fake-server"], root_uri="file:///tmp")
        raw_edit = {
            "changes": {
                "file:///tmp/x.py": [
                    {
                        "range": {
                            "start": {"line": 0, "character": 0},
                            "end": {"line": 0, "character": 3},
                        },
                        "newText": "New",
                    }
                ]
            },
            "documentChanges": [
                {
                    "textDocument": {
                        "uri": "file:///tmp/x.py",
                        "version": 2,
                    },
                    "edits": [
                        {
                            "range": {
                                "start": {"line": 1, "character": 0},
                                "end": {"line": 1, "character": 3},
                            },
                            "newText": "New",
                        }
                    ],
                }
            ],
        }

        async def fake_send_request(method, params=None):
            assert method == "textDocument/rename"
            return raw_edit

        monkeypatch.setattr(client, "send_request", fake_send_request)
        edit = await client.rename(
            "file:///tmp/x.py", {"line": 0, "character": 0}, "New"
        )

        assert isinstance(edit, WorkspaceEdit)
        assert edit.changes["file:///tmp/x.py"][0] == TextEdit.model_validate(
            raw_edit["changes"]["file:///tmp/x.py"][0]
        )
        assert isinstance(edit.documentChanges[0], TextDocumentEdit)

    def test_diagnostics_are_typed_and_malformed_data_degrades_client(self):
        client = LSPClient(command=["fake-server"], root_uri="file:///tmp")
        raw_diagnostic = {
            "range": {
                "start": {"line": 1, "character": 0},
                "end": {"line": 1, "character": 4},
            },
            "severity": 1,
            "message": "problem",
        }
        message = {
            "method": "textDocument/publishDiagnostics",
            "params": {
                "uri": "file:///tmp/x.py",
                "diagnostics": [raw_diagnostic],
            },
        }

        client._handle_message(message)
        diagnostics = client.get_diagnostics("file:///tmp/x.py")
        assert diagnostics == [Diagnostic.model_validate(raw_diagnostic)]
        assert isinstance(diagnostics[0], Diagnostic)

        client._handle_message(
            {
                **message,
                "params": {
                    "uri": "file:///tmp/x.py",
                    "diagnostics": ["invalid"],
                },
            }
        )
        assert client.status == LSPClientStatus.DEGRADED
        assert client.get_diagnostics("file:///tmp/x.py") == diagnostics

    def test_server_requests_receive_method_specific_responses(self):
        client = LSPClient(command=["fake-server"], root_uri="file:///tmp")
        process = _FakeProcess()
        client.process = process
        client.status = LSPClientStatus.COMPLETE

        client._handle_message(
            {
                "jsonrpc": "2.0",
                "id": 7,
                "method": "workspace/configuration",
                "params": {"items": [{}, {}, {}]},
            }
        )
        configuration_response = json.loads(
            process.stdin.writes[-1].split(b"\r\n\r\n", 1)[1]
        )
        assert configuration_response == {
            "jsonrpc": "2.0",
            "id": 7,
            "result": [None, None, None],
        }

        client._handle_message(
            {
                "jsonrpc": "2.0",
                "id": 8,
                "method": "client/registerCapability",
                "params": {},
            }
        )
        known_response = json.loads(
            process.stdin.writes[-1].split(b"\r\n\r\n", 1)[1]
        )
        assert known_response == {"jsonrpc": "2.0", "id": 8, "result": None}

        client._handle_message(
            {
                "jsonrpc": "2.0",
                "id": 9,
                "method": "server/unsupportedRequest",
                "params": {},
            }
        )
        unknown_response = json.loads(
            process.stdin.writes[-1].split(b"\r\n\r\n", 1)[1]
        )
        assert unknown_response == {
            "jsonrpc": "2.0",
            "id": 9,
            "error": {"code": -32601, "message": "Method not found"},
        }

    @pytest.mark.parametrize("exit_path", ["eof", "read_error", "cancel"])
    async def test_read_loop_exit_fails_pending_requests(
        self, monkeypatch, exit_path
    ):
        client = LSPClient(command=["fake-server"], root_uri="file:///tmp")
        process = _FakeProcess()
        client.process = process
        client.status = LSPClientStatus.COMPLETE
        pending = asyncio.get_running_loop().create_future()
        client._pending_requests[99] = pending

        if exit_path == "eof":
            process.stdout.close()
        elif exit_path == "read_error":
            async def fail_readline():
                raise OSError("read failed")

            process.stdout.readline = fail_readline

        client._run_task = asyncio.create_task(client._read_loop())
        if exit_path == "cancel":
            await asyncio.sleep(0)
            client._run_task.cancel()
        await client._run_task

        assert client.status == LSPClientStatus.FAILED
        assert client._pending_requests == {}
        with pytest.raises(LSPClientError, match="connection closed"):
            await pending


    async def test_json_rpc_error_is_raised_as_client_error(self, monkeypatch):
        client = LSPClient(command=["fake-server"], root_uri="file:///tmp")
        response = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "error": {"code": -32601, "message": "method not found"},
            }
        ).encode()
        process = _FakeProcess(
            responses={1: [_initialize_response()], 2: [response]}
        )

        async def fake_exec(*args, **kwargs):
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        await client.start()
        try:
            with pytest.raises(LSPClientError):
                await client.workspace_symbol("Target")
        finally:
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
