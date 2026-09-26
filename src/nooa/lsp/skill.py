# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LSP Skill module for NOOA agents."""

import pathlib
from typing import Dict

from nooa.skill import Skill

from .client import LSPClient
from .facade import LSPDocumentFacade
from .registry import LSPServerRegistry


class LSPSkill(Skill):
    """Provides Language Server Protocol (LSP) capabilities for code intelligence.

    Use this skill to perform repository-aware semantic code navigation:
        lsp = await self.lsp.for_file("src/orders.py")
        defs = await lsp.definition(line=10, character=5)
        refs = await lsp.references(line=10, character=5)
        
    Note: All methods in this skill and the returned facade document requests are async coroutines
    and must be awaited. However, `LSPDocumentFacade.diagnostics()` is synchronous and must not be awaited.
    """

    def __init__(self, root_uri: str | None = None):
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
        """Get an LSP facade for a given file.

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

    async def find_references(self, filepath: str, symbol: str) -> Any:
        """Find all references to a named symbol - compiler-accurate via LSP.
        
        Prefer this over text search when precision matters. 
        Note that this returns LSP Locations and requires the symbol to be present
        in the document's document_symbols first to find its position. If you know
        the position, use `lsp.for_file` and then `references(line, character)` directly.
        """
        facade = await self.for_file(filepath)
        if not facade:
            return []
            
        symbols = await facade.document_symbols()
        
        def _find_symbol_pos(syms: list[Any], target: str) -> dict[str, int] | None:
            for s in syms:
                if s.get("name") == target:
                    if "selectionRange" in s:
                        return s["selectionRange"]["start"]
                    elif "location" in s and "range" in s["location"]:
                        start = s["location"]["range"]["start"]
                        sym_uri = s["location"]["uri"]
                        if sym_uri in self._opened_documents:
                            content = self._opened_documents[sym_uri][1]
                            lines = content.splitlines()
                            if 0 <= start["line"] < len(lines):
                                offset = lines[start["line"]].find(target, start["character"])
                                if offset != -1:
                                    return {"line": start["line"], "character": offset}
                        return start
                if "children" in s and s["children"]:
                    res = _find_symbol_pos(s["children"], target)
                    if res:
                        return res
            return None
            
        pos = _find_symbol_pos(symbols if isinstance(symbols, list) else [], symbol)
        if pos:
            return await facade.references(pos["line"], pos["character"])
        return []

    async def shutdown(self):
        """Shutdown all running LSP clients."""
        for client in self._clients.values():
            await client.stop()
        self._clients.clear()
        self._opened_documents.clear()
