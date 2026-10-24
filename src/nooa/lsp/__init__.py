# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Language Server Protocol (LSP) capabilities for NOOA."""

from .client import LSPClient, LSPClientError
from .facade import LSPDocumentFacade
from .protocol import (
    CreateFile,
    DeleteFile,
    Diagnostic,
    DocumentSymbol,
    InitializeResult,
    Location,
    LocationLink,
    Position,
    Range,
    RenameFile,
    SymbolInformation,
    TextDocumentEdit,
    TextEdit,
    VersionedTextDocumentIdentifier,
    WorkspaceEdit,
    WorkspaceSymbol,
    WorkspaceSymbolLocation,
)
from .registry import LSPServerConfig, LSPServerRegistry
from .skill import LSPSkill

__all__ = [
    "LSPSkill",
    "LSPServerRegistry",
    "LSPServerConfig",
    "LSPDocumentFacade",
    "LSPClient",
    "LSPClientError",
    "InitializeResult",
    "Position",
    "Range",
    "Location",
    "LocationLink",
    "Diagnostic",
    "CreateFile",
    "DeleteFile",
    "DocumentSymbol",
    "RenameFile",
    "SymbolInformation",
    "TextEdit",
    "TextDocumentEdit",
    "VersionedTextDocumentIdentifier",
    "WorkspaceEdit",
    "WorkspaceSymbol",
    "WorkspaceSymbolLocation",
]
