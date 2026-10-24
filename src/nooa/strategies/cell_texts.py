# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Named text variables for the ``python_cell`` tool (its ``texts`` argument).

Models that write long text (reports, file contents, scripts) as Python string
literals inside a JSON tool argument often break the quoting. ``texts`` carries
each text as a plain JSON string; the strategy binds it as a ``str`` variable
before the cell's code runs. Binding is data-only: values go straight into the
namespace and are never interpolated into source code.
"""

import builtins as _builtins
import keyword
from collections.abc import Iterable
from typing import Any, TypedDict


class CellText(TypedDict):
    """A named text bound as a ``str`` variable before the cell's code runs."""

    name: str
    text: str


class CellTextsError(ValueError):
    """Invalid ``texts`` argument; the message is one line for the model."""


# Python's own builtins resolve in every cell (``print``, ``len``, ``open``...).
PYTHON_BUILTIN_NAMES: frozenset[str] = frozenset(dir(_builtins))


def parse_cell_texts(raw: Any, reserved: Iterable[str]) -> dict[str, str]:
    """Validate a ``texts`` argument and return ``{name: text}`` in order.

    ``None`` or ``[]`` gives ``{}``. Raises :class:`CellTextsError` with a
    single-line message on the first problem; nothing should be bound or run then.
    Rebinding an earlier text or an ordinary cell local is allowed.
    """
    if raw is None:
        return {}
    if not isinstance(raw, list):
        raise CellTextsError(
            f'`texts` must be a list of {{"name": ..., "text": ...}} objects, '
            f"got {type(raw).__name__}."
        )
    reserved_names = set(reserved)
    bound: dict[str, str] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) != {"name", "text"}:
            raise CellTextsError(
                f'`texts[{index}]` must be an object with exactly the keys "name" and "text".'
            )
        name, text = item["name"], item["text"]
        if not isinstance(name, str) or not isinstance(text, str):
            raise CellTextsError(f'`texts[{index}]`: "name" and "text" must be strings.')
        if not name.isidentifier() or keyword.iskeyword(name):
            raise CellTextsError(
                f"`texts[{index}]`: {name!r} is not a valid variable name "
                "(use a Python identifier that is not a keyword)."
            )
        if name.startswith("__") and name.endswith("__"):
            raise CellTextsError(f"`texts[{index}]`: dunder name {name!r} is reserved.")
        if name in reserved_names:
            raise CellTextsError(
                f"`texts[{index}]`: {name!r} is a runtime-provided name; pick another name."
            )
        if name in bound:
            raise CellTextsError(f"`texts[{index}]`: duplicate name {name!r}.")
        bound[name] = text
    return bound


# SyntaxError messages from text written as Python string literals.
_TEXTS_HINT_TRIGGERS = (
    "unterminated string literal",
    "unterminated triple-quoted string literal",
    "unexpected character after line continuation character",
)

TEXTS_HINT = (
    "Hint: for long text, pass it in `texts` as "
    '[{"name": "report", "text": "..."}] and use `report` in code; no escaping needed.'
)


def is_string_literal_syntax_error(error: BaseException | None) -> bool:
    """True for a SyntaxError (raw, validator-wrapped or sandboxed) from a broken literal."""
    if error is None:
        return False
    from nooa.runtime.sandbox.errors import SandboxExecutionError

    message: str | None = None
    if isinstance(error, SyntaxError):
        message = error.msg
    elif isinstance(getattr(error, "original_exception", None), SyntaxError):
        message = error.original_exception.msg  # type: ignore[attr-defined]
    elif isinstance(error, SandboxExecutionError) and error.original_type in {
        "SyntaxError",
        "CodeSyntaxError",
    }:
        message = error.diagnostic or str(error)
    return bool(message) and any(trigger in message for trigger in _TEXTS_HINT_TRIGGERS)
