# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-facing facade for interacting with LSP documents."""

from .client import LSPClient
from .protocol import (
    Diagnostic,
    DocumentSymbol,
    Location,
    LocationLink,
    SymbolInformation,
    WorkspaceEdit,
    WorkspaceSymbol,
)


class LSPDocumentFacade:
    """Provides a simplified, document-centric view of LSP capabilities.
    
    This is the object returned to the agent when calling `lsp.for_file("...")`.
    """

    def __init__(self, client: LSPClient, uri: str):
        """Create a document facade backed by a client and document URI."""
        self._client = client
        self._uri = uri

    async def definition(
        self, line: int, character: int
    ) -> list[Location | LocationLink]:
        """Find the definition of the symbol at the given position. Must be awaited.
        
        Args:
            line: 0-indexed line number.
            character: 0-indexed character offset.
            
        Returns:
            Typed LSP Location models, or an empty list if none exist.
        """
        return await self._client.definition(
            self._uri, {"line": line, "character": character}
        )

    async def references(
        self, line: int, character: int, include_declaration: bool = True
    ) -> list[Location]:
        """Find all references to the symbol at the given position. Must be awaited.
        
        Args:
            line: 0-indexed line number.
            character: 0-indexed character offset.
            include_declaration: Whether to include the declaration itself in the results.
            
        Returns:
            A list of typed LSP Location models, or an empty list.
        """
        return await self._client.references(
            self._uri, {"line": line, "character": character}, include_declaration
        )

    async def document_symbols(self) -> list[DocumentSymbol | SymbolInformation]:
        """Get all symbols defined in this document. Must be awaited.
        
        Returns:
            Typed LSP SymbolInformation or DocumentSymbol models. A null
            server response is normalized to an empty list.
        """
        return await self._client.document_symbol(self._uri)

    async def workspace_symbols(self, query: str) -> list[WorkspaceSymbol]:
        """Find typed symbols across the workspace matching a query."""
        return await self._client.workspace_symbol(query)

    async def rename(
        self, line: int, character: int, new_name: str
    ) -> WorkspaceEdit | None:
        """Rename the symbol at the given position.
        
        Args:
            line: 0-indexed line number.
            character: 0-indexed character offset.
            new_name: The new name to apply.
            
        Returns:
            A typed WorkspaceEdit model, or None if the rename is unavailable.
        """
        return await self._client.rename(
            self._uri, {"line": line, "character": character}, new_name
        )

    def diagnostics(self) -> list[Diagnostic]:
        """Get the latest diagnostics (errors, warnings) for this document.
        
        Returns:
            A list of LSP Diagnostic objects.
        """
        return self._client.get_diagnostics(self._uri)
