# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LSP Client implementation over asyncio subprocess."""

import asyncio
import json
import logging
from typing import Any, Dict

from .protocol import InitializeResult

logger = logging.getLogger(__name__)


class LSPClientError(Exception):
    """Exception raised for LSP client errors."""


class LSPClient:
    """A generic LSP JSON-RPC client."""

    def __init__(self, command: list[str], root_uri: str):
        self.command = command
        self.root_uri = root_uri
        self.process: asyncio.subprocess.Process | None = None
        self._next_id = 1
        self._pending_requests: Dict[int, asyncio.Future] = {}
        self._diagnostics: Dict[str, list[Any]] = {}
        self._run_task: asyncio.Task | None = None
        self.capabilities: dict[str, Any] = {}

    async def start(self) -> InitializeResult:
        """Start the LSP server process and initialize the connection."""
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._run_task = asyncio.create_task(self._read_loop())

        # Send initialize
        init_res = await self.send_request(
            "initialize",
            {
                "processId": None,
                "rootUri": self.root_uri,
                "capabilities": {
                    "workspace": {},
                    "textDocument": {
                        "publishDiagnostics": {},
                        "rename": {"prepareSupport": True},
                    },
                },
            },
        )
        self.capabilities = init_res.get("capabilities", {})

        # Send initialized
        await self.send_notification("initialized", {})
        return InitializeResult(capabilities=self.capabilities)

    async def stop(self):
        """Shutdown the LSP server and clean up resources."""
        if self.process:
            if self.process.returncode is None:
                try:
                    await asyncio.wait_for(self.send_request("shutdown", {}), timeout=2.0)
                    await self.send_notification("exit", {})
                except Exception:
                    pass
                
                # Give it a short moment to exit gracefully
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    try:
                        self.process.kill()
                    except ProcessLookupError:
                        pass
                    
            await self.process.wait()
            
        if self._run_task and not self._run_task.done():
            self._run_task.cancel()

    async def _read_loop(self):
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
                    
                content = await self.process.stdout.readexactly(content_length)
                try:
                    message = json.loads(content)
                    self._handle_message(message)
                except json.JSONDecodeError:
                    logger.error("Failed to decode LSP message: %s", content)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("LSP Read Error: %s", e)
        finally:
            for fut in self._pending_requests.values():
                if not fut.done():
                    fut.set_exception(LSPClientError("LSP server connection closed"))
            self._pending_requests.clear()

    def _handle_message(self, message: dict[str, Any]):
        if "id" in message and "method" not in message:
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
                    self._diagnostics[uri] = params.get("diagnostics", [])

    async def send_request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        if self._run_task and self._run_task.done():
            raise LSPClientError("LSP server connection closed")
            
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
        msg = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params is not None:
            msg["params"] = params
        self._send(msg)

    def _send(self, message: dict[str, Any]):
        if not self.process or not self.process.stdin:
            raise LSPClientError("Not connected")
        content = json.dumps(message).encode("utf-8")
        header = f"Content-Length: {len(content)}\r\n\r\n".encode("utf-8")
        self.process.stdin.write(header + content)

    # High-level LSP methods
    async def definition(self, uri: str, position: dict[str, int]) -> Any:
        return await self.send_request(
            "textDocument/definition", {"textDocument": {"uri": uri}, "position": position}
        )

    async def references(
        self, uri: str, position: dict[str, int], include_declaration: bool = False
    ) -> Any:
        return await self.send_request(
            "textDocument/references",
            {
                "textDocument": {"uri": uri},
                "position": position,
                "context": {"includeDeclaration": include_declaration},
            },
        )

    async def document_symbol(self, uri: str) -> Any:
        return await self.send_request(
            "textDocument/documentSymbol", {"textDocument": {"uri": uri}}
        )

    async def rename(self, uri: str, position: dict[str, int], new_name: str) -> Any:
        return await self.send_request(
            "textDocument/rename",
            {"textDocument": {"uri": uri}, "position": position, "newName": new_name},
        )

    def get_diagnostics(self, uri: str) -> list[Any]:
        return self._diagnostics.get(uri, [])
