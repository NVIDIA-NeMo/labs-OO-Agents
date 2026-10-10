# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LSP Skill module for NOOA agents."""

import pathlib
from collections.abc import Sequence
from typing import Dict

from nooa.skill import Skill

from .client import LSPClient
from .facade import LSPDocumentFacade
from .protocol import DocumentSymbol, Location, Position, SymbolInformation
from .registry import LSPServerRegistry


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
        self._opened_documents: dict[str, tuple[int, str]] = {}

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
        if server_cmd_key not in self._clients or self._clients[server_cmd_key].status == "FAILED":
            client = LSPClient(command=server_config.command, root_uri=self._root_uri)
            try:
                await client.start()
            except BaseException:
                await client.stop()
                raise
            self._clients[server_cmd_key] = client

        client = self._clients[server_cmd_key]
        uri = path.as_uri()

        # Ensure the document is "open" from the LSP's perspective.
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

        return LSPDocumentFacade(client, uri)

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
                    if sym_uri not in self._opened_documents:
                        return None

                    lines = self._opened_documents[sym_uri][1].splitlines()
                    end = symbol_range.end
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
                        line_start = start.character if line_number == start.line else 0
                        line_end = end.character if line_number == end.line else len(line)
                        offset = line.find(target, line_start, line_end)
                        if offset != -1:
                            return Position(line=line_number, character=offset)
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
