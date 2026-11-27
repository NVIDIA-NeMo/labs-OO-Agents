# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed models for Language Server Protocol payloads."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Position(BaseModel):
    line: int
    character: int

class Range(BaseModel):
    start: Position
    end: Position

class Location(BaseModel):
    uri: str
    range: Range


class LocationLink(BaseModel):
    originSelectionRange: Range | None = None
    targetUri: str
    targetRange: Range
    targetSelectionRange: Range


class Diagnostic(BaseModel):
    range: Range
    severity: int | None = None
    code: str | int | None = None
    source: str | None = None
    message: str

class SymbolInformation(BaseModel):
    name: str
    kind: int
    location: Location
    containerName: str | None = None


class DocumentSymbol(BaseModel):
    name: str
    kind: int
    tags: list[int] | None = None
    detail: str | None = None
    deprecated: bool | None = None
    range: Range
    selectionRange: Range
    children: list[DocumentSymbol] = Field(default_factory=list)


class WorkspaceSymbolLocation(BaseModel):
    uri: str
    range: Range | None = None


class WorkspaceSymbol(BaseModel):
    name: str
    kind: int
    tags: list[int] | None = None
    containerName: str | None = None
    deprecated: bool | None = None
    location: Location | WorkspaceSymbolLocation


class TextEdit(BaseModel):
    range: Range
    newText: str


class VersionedTextDocumentIdentifier(BaseModel):
    uri: str
    version: int | None


class TextDocumentEdit(BaseModel):
    textDocument: VersionedTextDocumentIdentifier
    edits: list[TextEdit]


class CreateFile(BaseModel):
    kind: Literal["create"]
    uri: str
    options: dict[str, bool] | None = None
    annotationId: str | None = None


class RenameFile(BaseModel):
    kind: Literal["rename"]
    oldUri: str
    newUri: str
    options: dict[str, bool] | None = None
    annotationId: str | None = None


class DeleteFile(BaseModel):
    kind: Literal["delete"]
    uri: str
    options: dict[str, bool] | None = None
    annotationId: str | None = None


class WorkspaceEdit(BaseModel):
    changes: dict[str, list[TextEdit]] | None = None
    documentChanges: list[TextDocumentEdit | CreateFile | RenameFile | DeleteFile] | None = None

class InitializeResult(BaseModel):
    capabilities: dict[str, Any]
