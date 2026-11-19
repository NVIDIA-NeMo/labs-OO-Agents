# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Condense numpydoc-style docstrings (NumPy, SciPy) into short doc() entries.

``condensed_callable_info(func)`` keeps the library's own text: the real signature
(``inspect.signature``, else the docstring's signature line), the first summary
paragraph, one line per parameter and return value, and the See Also names. Notes,
Examples, References and version directives are dropped. The function is never called.
"""

from __future__ import annotations

import inspect
import re
from typing import Any

from nooa.agentdoc.ext import CallableInfo

_MAX_ITEM_CHARS = 110
_SECTION_RULE = re.compile(r"^-{3,}$")
_SIMPLE_TYPE = re.compile(r"^[A-Za-z_][\w.]*$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")


def _sections(text: str) -> tuple[list[str], dict[str, list[str]]]:
    """Split a numpydoc docstring into its preamble lines and named sections."""
    lines = text.splitlines()
    preamble: list[str] = []
    sections: dict[str, list[str]] = {}
    current: list[str] = preamble
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if line.strip() and not line.startswith(" ") and _SECTION_RULE.match(nxt):
            current = sections.setdefault(line.strip().lower(), [])
            i += 2
            continue
        current.append(line)
        i += 1
    return preamble, sections


def _paragraphs(lines: list[str]) -> list[str]:
    paras: list[str] = []
    buf: list[str] = []
    for line in [*lines, ""]:
        if line.strip():
            buf.append(line.strip())
        elif buf:
            paras.append(" ".join(buf))
            buf = []
    return paras


def _shorten(text: str, limit: int = _MAX_ITEM_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _first_sentence(text: str) -> str:
    return _SENTENCE_END.split(" ".join(text.split()), maxsplit=1)[0]


def _entries(lines: list[str]) -> list[tuple[str, str]]:
    """``name : type`` headers (unindented) with their first description sentence."""
    entries: list[tuple[str, list[str]]] = []
    for line in lines:
        if not line.strip():
            if entries:
                entries[-1][1].append("")
            continue
        if not line.startswith(" "):
            entries.append((line.strip(), []))
        elif entries:
            entries[-1][1].append(line.strip())
    result = []
    for header, body in entries:
        # Only the first paragraph of the description; directives start later ones.
        first = []
        for line in body:
            if not line or line.startswith(".."):
                break
            first.append(line)
        result.append((header, _first_sentence(" ".join(first))))
    return result


def _entry_line(header: str, desc: str) -> str:
    name, _, typ = header.partition(" : ")
    head = f"{name.strip()}: {typ.strip()}" if typ else name.strip()
    return _shorten(f"{head} — {desc}" if desc else head)


def _see_also(lines: list[str], limit: int = 8) -> list[str]:
    names: list[str] = []
    for line in lines:
        if not line.strip() or line.startswith(" "):
            continue
        for part in line.split(":", 1)[0].split(","):
            name = part.strip().strip("`")
            if name and name not in names:
                names.append(name)
    return names[:limit]


def _signature_from_doc(name: str, preamble: list[str]) -> str | None:
    """The ``name(...)`` signature line that compiled numpy functions start with."""
    paras = _paragraphs(preamble)
    if not paras or not re.match(rf"^(\w+\.)*{re.escape(name)}\(", paras[0]):
        return None
    text = paras[0]
    start = text.index("(")
    depth = 0
    for i, char in enumerate(text[start:], start):
        depth += {"(": 1, ")": -1}.get(char, 0)
        if depth == 0:
            return text[start : i + 1]
    return text[start:]


def condensed_callable_info(func: Any, name: str | None = None) -> CallableInfo:
    """Return a short CallableInfo for a numpydoc-documented function (never calls it)."""
    name = name or getattr(func, "__name__", None) or "function"
    text = inspect.getdoc(func) or ""
    preamble, sections = _sections(text)
    doc_signature = _signature_from_doc(name, preamble)
    try:
        signature = str(inspect.signature(func)).replace("<no value>", "...")
    except (TypeError, ValueError):
        signature = doc_signature or "(...)"

    paras = _paragraphs(preamble)
    if doc_signature is not None:
        paras = paras[1:]
    paras = [p for p in paras if not p.startswith("..")]

    out: list[str] = [paras[0]] if paras else []
    params = [_entry_line(h, d) for h, d in _entries(sections.get("parameters", []))]
    if params:
        out += ["", "Parameters:", *(f"    {p}" for p in params)]
    returns = _entries(sections.get("returns", []) or sections.get("yields", []))
    if returns:
        out += ["Returns:", *(f"    {_entry_line(h, d)}" for h, d in returns[:4])]
    see_also = _see_also(sections.get("see also", []))
    if see_also:
        out.append(_shorten("See also: " + ", ".join(see_also)))

    return_type = ""
    if len(returns) == 1:
        _, _, typ = returns[0][0].partition(" : ")
        typ = typ.split(",")[0].strip()
        # Only simple type names (ndarray, float, bool); descriptive types stay in the text.
        return_type = typ if _SIMPLE_TYPE.match(typ) else ""
    return CallableInfo(
        name=name,
        signature=signature,
        return_type=return_type,
        docstring="\n".join(out) or None,
    )


def doc_signature(func: Any, name: str | None = None) -> str | None:
    """The signature line a numpydoc docstring starts with (e.g. ``a.sum(axis=None, ...)``)."""
    name = name or getattr(func, "__name__", "") or ""
    preamble, _ = _sections(inspect.getdoc(func) or "")
    return _signature_from_doc(name, preamble)


def summary_line(func: Any) -> str:
    """The first summary sentence of a numpydoc docstring (after any signature line)."""
    name = getattr(func, "__name__", "") or ""
    preamble, _ = _sections(inspect.getdoc(func) or "")
    paras = _paragraphs(preamble)
    if paras and _signature_from_doc(name, preamble) is not None:
        paras = paras[1:]
    return _shorten(_first_sentence(paras[0])) if paras else ""
