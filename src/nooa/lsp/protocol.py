# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LSP Protocol Types."""

from typing import Any
from pydantic import BaseModel

class Position(BaseModel):
    line: int
    character: int

class Range(BaseModel):
    start: Position
    end: Position

class Location(BaseModel):
    uri: str
    range: Range

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

class WorkspaceEdit(BaseModel):
    changes: dict[str, list[Any]] | None = None
    documentChanges: list[Any] | None = None

class InitializeResult(BaseModel):
    capabilities: dict[str, Any]
