# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""python_cell's optional raw ``text`` argument.

A call may carry ``text``: an object mapping names to raw strings (file
contents, scripts, patches). The cell gets it verbatim as the dict variable
``text``, so the model never writes those strings as Python literals inside a
JSON argument (the top SyntaxError class in traces: quoting and escaping).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

CELL_TEXT_NAME = "text"

CELL_TEXT_DESCRIPTION = (
    "Optional raw texts for this cell, by name (file contents, scripts, patches): "
    '{"app.py": "...", "test_app.py": "..."}. The cell gets them verbatim as the dict '
    "`text`; no quoting or escaping needed. Example code: "
    "`for path, content in text.items(): await self.files.write_file(path, content)` or "
    '`await sh.run("python3 -", stdin=text["script"])`.'
)

BARE_STRING_NOTE = (
    "Note: `text` arrived as a bare string and is bound as "
    '{"text": <string>}; pass an object of name -> string next time.'
)


class PythonCellArguments(BaseModel):
    """python_cell's parameters: Python source plus optional raw texts."""

    code: str = Field(description="Python code to execute.")
    # A plain object schema (no anyOf/null union): some providers handle unions badly.
    text: dict[str, str] = Field(default_factory=dict, description=CELL_TEXT_DESCRIPTION)


@dataclass(frozen=True)
class CellText:
    """A validated ``text`` argument: ``value`` to bind, or the ``error`` to report."""

    value: dict[str, str] | None = None
    error: str | None = None
    bare_string: bool = False

    @property
    def chars(self) -> int:
        return sum(len(v) for v in (self.value or {}).values())


def parse_cell_text(raw: Any) -> CellText:
    """Validate a call's ``text`` argument.

    ``None`` and ``{}`` mean no text (nothing is bound). A bare string is
    accepted as ``{"text": raw}``. Keys must be non-empty strings and values
    strings; anything else is an error naming the bad entry. Values are kept
    exactly as received.
    """
    if raw is None:
        return CellText()
    if isinstance(raw, str):
        return CellText(value={CELL_TEXT_NAME: raw}, bare_string=True)
    if not isinstance(raw, dict):
        return CellText(
            error=(
                "Execution error: `text` must be an object mapping names to strings, "
                f'like {{"app.py": "..."}}; got {type(raw).__name__}. Nothing ran.'
            )
        )
    if not raw:
        return CellText()
    for key, value in raw.items():
        shown = repr(key) if len(repr(key)) <= 80 else repr(key)[:77] + "..."
        if not isinstance(key, str) or not key:
            return CellText(
                error=(
                    "Execution error: every `text` key must be a non-empty string "
                    f"name; got {shown}. Nothing ran."
                )
            )
        if not isinstance(value, str):
            return CellText(
                error=(
                    f"Execution error: `text[{shown}]` must be a string; got "
                    f"{type(value).__name__}. Pass raw contents as strings. Nothing ran."
                )
            )
    return CellText(value=dict(raw))
