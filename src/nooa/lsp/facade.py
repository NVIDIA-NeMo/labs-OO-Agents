# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-facing facade for interacting with LSP documents."""

from typing import Any

from .client import LSPClient


class LSPDocumentFacade:
    """Provides a simplified, document-centric view of LSP capabilities.
    
    This is the object returned to the agent when calling `lsp.for_file("...")`.
    """

    def __init__(self, client: LSPClient, uri: str):
        self._client = client
        self._uri = uri

    async def definition(self, line: int, character: int) -> Any:
        """Find the definition of the symbol at the given position.
        
        Args:
            line: 0-indexed line number.
            character: 0-indexed character offset.
            
        Returns:
            The raw LSP location result (either a single Location or a list).
        """
        return await self._client.definition(
            self._uri, {"line": line, "character": character}
        )

    async def references(
        self, line: int, character: int, include_declaration: bool = False
    ) -> Any:
        """Find all references to the symbol at the given position.
        
        Args:
            line: 0-indexed line number.
            character: 0-indexed character offset.
            include_declaration: Whether to include the declaration itself in the results.
            
        Returns:
            A list of LSP Locations.
        """
        return await self._client.references(
            self._uri, {"line": line, "character": character}, include_declaration
        )

    async def document_symbols(self) -> Any:
        """Get all symbols defined in this document.
        
        Returns:
            A list of LSP SymbolInformation or DocumentSymbol objects.
        """
        return await self._client.document_symbol(self._uri)

    async def rename(self, line: int, character: int, new_name: str) -> Any:
        """Rename the symbol at the given position.
        
        Args:
            line: 0-indexed line number.
            character: 0-indexed character offset.
            new_name: The new name to apply.
            
        Returns:
            A WorkspaceEdit object describing the required file modifications.
        """
        return await self._client.rename(
            self._uri, {"line": line, "character": character}, new_name
        )

    def diagnostics(self) -> list[Any]:
        """Get the latest diagnostics (errors, warnings) for this document.
        
        Returns:
            A list of LSP Diagnostic objects.
        """
        return self._client.get_diagnostics(self._uri)
