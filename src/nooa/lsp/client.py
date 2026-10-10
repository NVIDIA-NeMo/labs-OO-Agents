# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LSP Client implementation over asyncio subprocess."""

import asyncio
import json
import logging
from typing import Any, Dict

from pydantic import TypeAdapter, ValidationError

from .protocol import (
    Diagnostic,
    DocumentSymbol,
    InitializeResult,
    Location,
    LocationLink,
    SymbolInformation,
    WorkspaceEdit,
    WorkspaceSymbol,
)

logger = logging.getLogger(__name__)

_CLIENT_REQUEST_METHODS = {
    "client/registerCapability",
    "client/unregisterCapability",
    "window/showDocument",
    "window/showMessageRequest",
    "window/workDoneProgress/create",
    "workspace/applyEdit",
    "workspace/workspaceFolders",
}
_SUPPORTED_POSITION_ENCODINGS = ("utf-8", "utf-16", "utf-32")


class LSPClientError(Exception):
    """Exception raised for LSP client errors."""


class LSPClientStatus:
    """Lifecycle/health states for an LSPClient.

    Lets validation gates distinguish "the code has an error" from
    "the language server crashed": queries against a non-COMPLETE
    client cannot be trusted as evidence about the code.
    """

    UNKNOWN = "UNKNOWN"      # constructed, initialize not yet completed
    COMPLETE = "COMPLETE"    # initialize handshake succeeded; healthy
    DEGRADED = "DEGRADED"    # server sent undecodable data; may be unreliable
    FAILED = "FAILED"        # connection lost / server process gone


class LSPClient:
    """A generic LSP JSON-RPC client."""

    def __init__(self, command: list[str], root_uri: str):
        """Create a client for the server command and workspace URI."""
        self.command = command
        self.root_uri = root_uri
        self.process: asyncio.subprocess.Process | None = None
        self._next_id = 1
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._diagnostics: Dict[str, list[Any]] = {}
        self._run_task: asyncio.Task | None = None
        self._watch_task: asyncio.Task | None = None
        self.capabilities: dict[str, Any] = {}
        self.position_encoding = "utf-16"
        self.status: str = LSPClientStatus.UNKNOWN
        self._decode_errors: int = 0

    async def start(self) -> InitializeResult:
        """Start the LSP server process and initialize the connection."""
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._run_task = asyncio.create_task(self._read_loop())
        self._watch_task = asyncio.create_task(self._watch_process())
        # Process is up; the handshake itself happens while DEGRADED (not yet
        # COMPLETE) so _send's guard doesn't reject the initialize request.
        self.status = LSPClientStatus.DEGRADED

        # Send initialize
        init_res = await self.send_request(
            "initialize",
            {
                "processId": None,
                "rootUri": self.root_uri,
                "capabilities": {
                    "general": {
                        "positionEncodings": list(
                            _SUPPORTED_POSITION_ENCODINGS
                        )
                    },
                    "workspace": {},
                    "textDocument": {
                        "publishDiagnostics": {},
                        "rename": {"prepareSupport": True},
                    },
                },
            },
        )
        initialize_result = self._parse_response(
            "initialize", init_res, InitializeResult
        )
        self.capabilities = initialize_result.capabilities
        self.position_encoding = self.capabilities.get(
            "positionEncoding", "utf-16"
        )
        if self.position_encoding not in _SUPPORTED_POSITION_ENCODINGS:
            raise LSPClientError(
                "Unsupported LSP position encoding: "
                f"{self.position_encoding}"
            )

        # Send initialized
        await self.send_notification("initialized", {})
        # Only promote to COMPLETE from the handshake state. The read loop runs
        # concurrently and may already have flagged DEGRADED (undecodable frame)
        # or FAILED (server died) while we awaited — don't clobber either.
        if self.status not in (LSPClientStatus.FAILED, LSPClientStatus.DEGRADED):
            self.status = LSPClientStatus.COMPLETE
        elif self.status == LSPClientStatus.DEGRADED and not self._decode_errors:
            # DEGRADED here is just the handshake placeholder, not a decode error.
            self.status = LSPClientStatus.COMPLETE
        return initialize_result

    async def stop(self):
        """Shutdown the LSP server and clean up resources."""
        if self.process:
            if self.process.returncode is None:
                try:
                    await asyncio.wait_for(
                        self.send_request("shutdown", {}), timeout=1.0
                    )
                    await self.send_notification("exit", {})
                except (LSPClientError, OSError, asyncio.TimeoutError):
                    pass

                # Give it a short moment to exit gracefully
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    try:
                        self.process.kill()
                    except ProcessLookupError:
                        pass

            if self.process.returncode is None:
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    logger.warning("LSP server did not exit after kill")
        # Server is gone either way; no query result can be trusted past here.
        self.status = LSPClientStatus.FAILED

        if self._run_task and not self._run_task.done():
            self._run_task.cancel()
        if self._watch_task and not self._watch_task.done():
            self._watch_task.cancel()

    async def _read_loop(self):
        """Read and dispatch framed JSON-RPC messages from the server."""
        try:
            while self.process and self.process.stdout and not self.process.stdout.at_eof():
                content_length = None
                while True:
                    line_bytes = await self.process.stdout.readline()
                    if not line_bytes:
                        return
                    line_str = line_bytes.decode("utf-8").strip()
                    if not line_str:
                        break
                    if line_str.lower().startswith("content-length:"):
                        content_length = int(line_str.split(":")[1].strip())

                if content_length is None:
                    continue

                try:
                    content = await self.process.stdout.readexactly(content_length)
                except asyncio.IncompleteReadError:
                    return
                try:
                    message = json.loads(content)
                    self._handle_message(message)
                except json.JSONDecodeError:
                    self._decode_errors += 1
                    self.status = LSPClientStatus.DEGRADED
                    logger.error("Failed to decode LSP message: %s", content)
            # The loop only ends when the stream reaches EOF: the pipe closed
            # under us, so the server is gone. asyncio's process.wait() can stay
            # pending even after the child dies (returncode set, waiter never
            # woken), so EOF here — not _watch_process — is the reliable death
            # signal for a real server.
            self._mark_connection_lost()
        except asyncio.CancelledError:
            pass
        except (OSError, ValueError, TypeError, AttributeError) as e:
            logger.error("LSP Read Error: %s", e)
        finally:
            self._mark_connection_lost()

    def _mark_connection_lost(self):
        """Flag the client FAILED and fail any in-flight requests.

        A dead server must never look like healthy code to a validation gate,
        and a request awaiting a reply that will never come must raise rather
        than hang. Called from the read loop on EOF/read error and from
        _watch_process when the server process exits.
        """
        self.status = LSPClientStatus.FAILED
        for future in self._pending_requests.values():
            if not future.done():
                future.set_exception(LSPClientError("LSP connection closed"))
        self._pending_requests.clear()

    async def _watch_process(self):
        """Mark the client FAILED if the server process dies unexpectedly.

        ValidationPolicy gates key off ``status``: a dead server must never
        look like clean code. A clean stop() flips status to FAILED itself
        before awaiting process exit, so this only fires on unexpected death.

        Poll ``returncode`` rather than ``await process.wait()``: a wrapper
        server (e.g. pyright's node launcher) can spawn a child that inherits
        the stdout pipe, so killing the parent leaves the pipe open — the read
        loop never sees EOF and asyncio's wait() can stay pending even though
        the parent was reaped and ``returncode`` is set. The reaped exit code
        is the one reliable death signal in that case.
        """
        try:
            while self.process is not None and self.process.returncode is None:
                await asyncio.sleep(0.05)
            self._mark_connection_lost()
        except asyncio.CancelledError:
            pass

    def _handle_message(self, message: dict[str, Any]):
        """Dispatch a server response or notification to its handler."""
        if "id" in message and "method" in message:
            self._handle_server_request(message)
        elif "id" in message:
            # Response
            msg_id = message["id"]
            if msg_id in self._pending_requests:
                future = self._pending_requests.pop(msg_id)
                if not future.done():
                    if "error" in message:
                        future.set_exception(LSPClientError(message["error"]))
                    else:
                        future.set_result(message.get("result"))
        elif "method" in message:
            # Notification or Request from server
            if message["method"] == "textDocument/publishDiagnostics":
                params = message.get("params", {})
                uri = params.get("uri")
                if uri:
                    try:
                        self._diagnostics[uri] = self._parse_list_response(
                            "textDocument/publishDiagnostics",
                            params.get("diagnostics", []),
                            Diagnostic,
                        )
                    except LSPClientError as error:
                        self.status = LSPClientStatus.DEGRADED
                        logger.error("Invalid LSP diagnostics: %s", error)

    def _handle_server_request(self, request: dict[str, Any]) -> None:
        """Reply to requests initiated by the language server."""
        method = request["method"]
        request_id = request["id"]
        if method == "workspace/configuration":
            params = request.get("params", {})
            items = params.get("items", []) if isinstance(params, dict) else []
            result = [None] * len(items) if isinstance(items, list) else []
            response = {"jsonrpc": "2.0", "id": request_id, "result": result}
        elif method in _CLIENT_REQUEST_METHODS:
            response = {"jsonrpc": "2.0", "id": request_id, "result": None}
        else:
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": -32601, "message": "Method not found"},
            }
        self._send(response)

    @staticmethod
    def _parse_response(method: str, result: Any, result_type: Any) -> Any:
        try:
            return TypeAdapter(result_type).validate_python(result)
        except ValidationError as error:
            raise LSPClientError(f"Invalid {method} response") from error

    @classmethod
    def _parse_list_response(
        cls, method: str, result: Any, item_type: Any
    ) -> list[Any]:
        if result is None:
            return []
        if not isinstance(result, list):
            raise LSPClientError(
                f"Invalid {method} response: expected a list or null"
            )
        return cls._parse_response(method, result, list[item_type])

    async def send_request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send a JSON-RPC request and return its response result."""
        if self.status == LSPClientStatus.FAILED:
            raise LSPClientError(
                f"LSP server not available (status={self.status})"
            )
        msg_id = self._next_id
        self._next_id += 1
        msg = {
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": method,
        }
        if params is not None:
            msg["params"] = params

        future = asyncio.get_running_loop().create_future()
        self._pending_requests[msg_id] = future

        self._send(msg)
        return await future

    async def send_notification(self, method: str, params: dict[str, Any] | None = None):
        """Send a JSON-RPC notification without waiting for a response."""
        msg: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            msg["params"] = params
        self._send(msg)

    def _send(self, message: dict[str, Any]):
        """Encode and write a JSON-RPC message to the server process."""
        if self.status in (LSPClientStatus.FAILED, LSPClientStatus.UNKNOWN):
            raise LSPClientError(f"LSP server not available (status={self.status})")
        if not self.process or not self.process.stdin:
            self.status = LSPClientStatus.FAILED
            raise LSPClientError("Not connected")
        content = json.dumps(message).encode("utf-8")
        header = f"Content-Length: {len(content)}\r\n\r\n".encode("utf-8")
        self.process.stdin.write(header + content)

    # High-level LSP methods
    async def definition(
        self, uri: str, position: dict[str, int]
    ) -> list[Location | LocationLink]:
        """Return definition locations for a document position.

        A single location is wrapped in a list, and a null result becomes an
        empty list. Raises LSPClientError if the server returns an invalid
        result.
        """
        result = await self.send_request(
            "textDocument/definition", {"textDocument": {"uri": uri}, "position": position}
        )
        if result is None:
            return []
        if isinstance(result, dict):
            result = [result]
        return self._parse_list_response(
            "textDocument/definition", result, Location | LocationLink
        )

    async def references(
        self, uri: str, position: dict[str, int], include_declaration: bool = False
    ) -> list[Location]:
        """Return references to the symbol at a document position.

        Args:
            uri: URI of the document to query.
            position: Zero-based line and character of the symbol.
            include_declaration: Whether to include the symbol declaration.

        Returns:
            Typed LSP Location models; an empty list if none are found.

        Raises:
            LSPClientError: If the server response is not a list of locations
                or null.
        """
        result = await self.send_request(
            "textDocument/references",
            {
                "textDocument": {"uri": uri},
                "position": position,
                "context": {"includeDeclaration": include_declaration},
            },
        )
        return self._parse_list_response(
            "textDocument/references", result, Location
        )

    async def document_symbol(
        self, uri: str
    ) -> list[DocumentSymbol | SymbolInformation]:
        """Return typed symbols declared in a document."""
        result = await self.send_request(
            "textDocument/documentSymbol", {"textDocument": {"uri": uri}}
        )
        return self._parse_list_response(
            "textDocument/documentSymbol",
            result,
            DocumentSymbol | SymbolInformation,
        )

    async def workspace_symbol(self, query: str) -> list[WorkspaceSymbol]:
        """Return typed workspace symbols matching a query string."""
        result = await self.send_request("workspace/symbol", {"query": query})
        return self._parse_list_response(
            "workspace/symbol", result, WorkspaceSymbol
        )

    async def rename(
        self, uri: str, position: dict[str, int], new_name: str
    ) -> WorkspaceEdit | None:
        """Return a typed workspace edit for renaming a symbol."""
        result = await self.send_request(
            "textDocument/rename",
            {"textDocument": {"uri": uri}, "position": position, "newName": new_name},
        )
        if result is None:
            return None
        return self._parse_response(
            "textDocument/rename", result, WorkspaceEdit
        )

    def get_diagnostics(self, uri: str) -> list[Diagnostic]:
        """Return the latest diagnostics published for a document URI."""
        return self._diagnostics.get(uri, [])
