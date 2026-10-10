# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LSP Skill module for NOOA agents."""

import asyncio
import pathlib
from collections.abc import Sequence
from typing import Dict

from nooa.skill import Skill

from .client import LSPClient
from .facade import LSPDocumentFacade
from .protocol import DocumentSymbol, Location, Position, SymbolInformation
from .registry import LSPServerRegistry


def _character_units(character: str, encoding: str) -> int:
    if encoding == "utf-8":
        return len(character.encode("utf-8"))
    if encoding == "utf-16":
        return len(character.encode("utf-16-le")) // 2
    if encoding == "utf-32":
        return 1
    raise ValueError(f"Unsupported LSP position encoding: {encoding}")


def _lsp_character_to_index(
    text: str, character: int, encoding: str
) -> int | None:
    if character < 0:
        return None
    units = 0
    for index, value in enumerate(text):
        if units == character:
            return index
        units += _character_units(value, encoding)
        if units > character:
            return None
    return len(text) if units == character else None


def _index_to_lsp_character(text: str, index: int, encoding: str) -> int:
    return sum(_character_units(value, encoding) for value in text[:index])


class LSPSkill(Skill):
    """Provides Language Server Protocol (LSP) capabilities for code intelligence.

    Document requests are async coroutines and must be awaited; calling one
    without `await` returns a coroutine object, not a facade or result. Symbol
    results are typed models with fields such as `symbol.name`,
    `symbol.selectionRange`, or `symbol.location`.
    `LSPDocumentFacade.diagnostics()` is synchronous and must not be awaited.

    Use this skill to perform repository-aware semantic code navigation:
        lsp = await self.lsp.for_file("src/orders.py")
        defs = await lsp.definition(line=10, character=5)
        refs = await lsp.references(line=10, character=5)
        
    For a name-based one-shot lookup (compiler-accurate alternative to text search):
        refs = await self.lsp.find_references("src/orders.py", "OrderService")
    """

    def __init__(self, root_uri: str | None = None):
        """Initialize language server clients for the given workspace URI."""
        super().__init__()
        if root_uri is None:
            self._root_uri = pathlib.Path.cwd().as_uri()
        else:
            if not root_uri.startswith("file://"):
                self._root_uri = pathlib.Path(root_uri).absolute().as_uri()
            else:
                self._root_uri = root_uri

        self.registry = LSPServerRegistry()
        self._clients: Dict[tuple[str, ...], LSPClient] = {}
        self._client_locks: dict[tuple[str, ...], asyncio.Lock] = {}
        self._opened_documents: dict[str, tuple[int, str]] = {}
        self._opened_document_clients: dict[str, tuple[str, ...]] = {}

    async def for_file(self, filepath: str) -> LSPDocumentFacade | None:
        """Get an LSP facade for a given file. Must be awaited.

        Args:
            filepath: Path to the source file (e.g., 'src/main.py').

        Returns:
            An LSPDocumentFacade, or None if no language server is available for the file extension.
        """
        path = pathlib.Path(filepath).absolute()
        ext = path.suffix

        server_config = self.registry.get_server_for_extension(ext)
        if not server_config:
            return None

        server_cmd_key = tuple(server_config.command)
        lock = self._client_locks.setdefault(server_cmd_key, asyncio.Lock())
        async with lock:
            client = await self._get_or_start_client(
                server_cmd_key, server_config.command
            )

        uri = path.as_uri()
        content = path.read_text(encoding="utf-8")
        state = self._opened_documents.get(uri)
        if state is None:
            lang_id = ext.lstrip(".")

            # Standardize common language IDs
            if lang_id == "py":
                lang_id = "python"
            elif lang_id == "rs":
                lang_id = "rust"
            elif lang_id == "js":
                lang_id = "javascript"
            elif lang_id == "ts":
                lang_id = "typescript"
            elif lang_id == "tsx":
                lang_id = "typescriptreact"
            elif lang_id == "jsx":
                lang_id = "javascriptreact"

            await client.send_notification(
                "textDocument/didOpen",
                {
                    "textDocument": {
                        "uri": uri,
                        "languageId": lang_id,
                        "version": 1,
                        "text": content,
                    }
                },
            )
            self._opened_documents[uri] = (1, content)
        elif state[1] != content:
            version = state[0] + 1
            await client.send_notification(
                "textDocument/didChange",
                {
                    "textDocument": {"uri": uri, "version": version},
                    "contentChanges": [{"text": content}],
                },
            )
            self._opened_documents[uri] = (version, content)

        self._opened_document_clients[uri] = server_cmd_key
        return LSPDocumentFacade(client, uri)

    async def _get_or_start_client(
        self, server_cmd_key: tuple[str, ...], command: list[str]
    ) -> LSPClient:
        client = self._clients.get(server_cmd_key)
        if client is not None and client.status == "FAILED":
            stale_uris = [
                uri
                for uri, owner in self._opened_document_clients.items()
                if owner == server_cmd_key
            ]
            for stale_uri in stale_uris:
                self._opened_documents.pop(stale_uri, None)
                self._opened_document_clients.pop(stale_uri, None)
            self._clients.pop(server_cmd_key)
            client = None

        if client is None:
            client = LSPClient(command=command, root_uri=self._root_uri)
            try:
                await client.start()
            except BaseException:
                await client.stop()
                raise
            self._clients[server_cmd_key] = client
        return client

    async def find_references(
        self, filepath: str, symbol: str
    ) -> list[Location]:
        """Find references to a named symbol using the language server. Must be awaited.

        Compiler-accurate alternative to text-based search: resolves the
        symbol's declaration via document_symbols, then queries references
        at that exact position (declaration included in results).

        Prefer this to text search when precision matters. The symbol must be
        present in the document's symbols. For a known position, use
        ``lsp.for_file`` and then ``references(line, character)`` directly.

        Args:
            filepath: Path to the file where the symbol is defined.
            symbol: The symbol name to find references for (e.g. 'get_llm_client').

        Returns:
            Typed LSP Location models. Empty if no server or symbol is
            available.

        Raises:
            LSPClientError: If the language server returns a malformed result.

        Example:
            refs = await self.lsp.find_references("src/nooa/unifiedllm/registry.py", "get_llm_client")
        """
        facade = await self.for_file(filepath)
        if not facade:
            return []
            
        symbols = await facade.document_symbols()
        
        def _find_symbol_pos(
            syms: Sequence[DocumentSymbol | SymbolInformation], target: str
        ) -> Position | None:
            """Find a symbol's LSP position, searching nested symbols too."""
            for item in syms:
                if item.name == target:
                    if isinstance(item, DocumentSymbol):
                        return item.selectionRange.start

                    symbol_range = item.location.range
                    start = symbol_range.start
                    sym_uri = item.location.uri
                    client_key = self._opened_document_clients.get(sym_uri)
                    if (
                        sym_uri not in self._opened_documents
                        or client_key not in self._clients
                    ):
                        return None

                    lines = self._opened_documents[sym_uri][1].splitlines()
                    end = symbol_range.end
                    encoding = self._clients[client_key].position_encoding
                    if (
                        start.line < 0
                        or end.line >= len(lines)
                        or start.line > end.line
                        or start.character < 0
                        or end.character < 0
                    ):
                        return None

                    for line_number in range(start.line, end.line + 1):
                        line = lines[line_number]
                        line_start = 0
                        if line_number == start.line:
                            line_start = _lsp_character_to_index(
                                line, start.character, encoding
                            )
                            if line_start is None:
                                return None
                        line_end = len(line)
                        if line_number == end.line:
                            line_end = _lsp_character_to_index(
                                line, end.character, encoding
                            )
                            if line_end is None:
                                return None
                        offset = line.find(target, line_start, line_end)
                        if offset != -1:
                            return Position(
                                line=line_number,
                                character=_index_to_lsp_character(
                                    line, offset, encoding
                                ),
                            )
                    return None

                if isinstance(item, DocumentSymbol) and item.children:
                    result = _find_symbol_pos(
                        item.children, target
                    )
                    if result:
                        return result
            return None
            
        pos = _find_symbol_pos(symbols, symbol)
        if pos:
            return await facade.references(pos.line, pos.character)
        return []

    async def shutdown(self):
        """Shutdown all running LSP clients. Must be awaited."""
        for client in self._clients.values():
            await client.stop()
        self._clients.clear()
        self._opened_documents.clear()
        self._opened_document_clients.clear()
