# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Renderers: one snapshot, the notation each consumer shows.

- ``render_output(text)``: Shell-style output, ``HEAD\n[... N bytes omitted; ...]\nTAIL``.
- ``render_python(snapshot)``: agentdoc's ``pformat`` / prefill notation.
- ``render_json(snapshot, max_chars=...)``: plain, valid JSON within ``max_chars``
  (#342's trace preview semantics: nothing is embedded in user data).
- ``preview_attributes(direction, snapshot)``: the trace span attributes that
  say whether and where a JSON preview is incomplete.
"""

from __future__ import annotations

import sys
from typing import Any

from nooa.snapshot._capture import (
    dumps,
    fallback_candidates,
    fallback_json,
    fit_string,
    opaque_json,
    pointer_child,
)
from nooa.snapshot._model import (
    Mapping,
    Object,
    Opaque,
    Ref,
    Scalar,
    Sequence,
    Snapshot,
    Text,
    Unavailable,
)

__all__ = ["preview_attributes", "render_json", "render_output", "render_python"]

PREVIEW_VERSION = 1
_MAX_INCOMPLETE_PATHS = 32
_MAX_INCOMPLETE_PATH_CHARS = 2_048
_MAX_KEY_CHARS = 50


# --- output notation ---------------------------------------------------------------


def render_output(text: Text) -> str:
    """``text`` as the model sees command output: whole, or head and tail around one line.

    The line says how much was left out and where the whole output is::

        [... 1,234 bytes omitted; full output: /path]
        [... 1,234 bytes omitted; /path holds the first 65,536 of 99,999 bytes]
        [... 1,234 bytes omitted; full output not kept]
    """
    if text.complete:
        return text.head + text.tail
    omitted = f"[... {text.omitted:,} {text.unit} omitted; "
    full = text.full
    if full is None:
        line = omitted + "full output not kept]"
    elif full.kept < full.total:
        line = omitted + f"{full.path} holds the first {full.kept:,} of {full.total:,} bytes]"
    else:
        line = omitted + f"full output: {full.path}]"
    return (text.head + "\n" if text.head else "") + line + ("\n" + text.tail if text.tail else "")


# --- Python notation (agentdoc pformat / prefill) ------------------------------------


def _multiline(s: str) -> str:
    """agentdoc's triple-quoted form for strings with newlines or both quote kinds."""
    if "'''" in s and '"""' in s:
        return repr(s)
    quote = '"""' if "'''" in s else "'''"
    return f"{quote}{s}{quote}"


def _python_str(s: str) -> str:
    if "\n" in s or ("'" in s and '"' in s):
        return _multiline(s)
    return repr(s)


def _python_text(text: Text) -> str:
    if text.complete:
        return _python_str(text.head + text.tail)
    if text.unit == "bytes":
        head_b, tail_b = text.head.encode("utf-8"), text.tail.encode("utf-8")
        parts = [f"bytes(len={text.length}"]
        if head_b or not tail_b:
            parts.append(f"[:{len(head_b)}]={head_b!r}")
        if tail_b:
            parts.append(f"[-{len(tail_b)}:]={tail_b!r}")
        return ", ".join(parts) + ")"
    parts = [f"str(len={text.length}"]
    if text.head or not text.tail:
        parts.append(f"[:{len(text.head)}]={text.head!r}")
    if text.tail:
        parts.append(f"[-{len(text.tail)}:]={text.tail!r}")
    return ", ".join(parts) + ")"


def _python_key(key: Any) -> str:
    """A mapping key as agentdoc shows it (string keys cut to 50 characters)."""
    if type(key) is str and len(key) > _MAX_KEY_CHARS:
        head_n = (_MAX_KEY_CHARS + 1) // 2
        tail_n = _MAX_KEY_CHARS - head_n
        return f"str(len={len(key)}, [:{head_n}]={key[:head_n]!r}, [-{tail_n}:]={key[-tail_n:]!r})"
    return _python_str(key) if type(key) is str else repr(key)


def _brackets(type_name: str) -> tuple[str, str]:
    return ("(", ")") if type_name == "tuple" else ("[", "]")


def _python_items(type_name: str, items: tuple[Snapshot, ...]) -> str:
    left, right = _brackets(type_name)
    inner = ", ".join(render_python(item) for item in items)
    if type_name == "tuple" and len(items) == 1:
        inner += ","
    return f"{left}{inner}{right}"


def _length(length: int | None) -> str:
    return "?" if length is None else str(length)


def _python_sequence(node: Sequence) -> str:
    left, right = _brackets(node.type_name)
    if node.complete:
        return _python_items(node.type_name, node.items)
    if node.reason == "max_depth" and not node.items:
        return f"{left}{node.type_name}: {_length(node.length)} items{right}"
    parts = [f"{node.type_name}(len={_length(node.length)}"]
    parts.append(f"[:{len(node.items)}]={_python_items(node.type_name, node.items)}")
    if node.tail:
        parts.append(f"[-{len(node.tail)}:]={_python_items(node.type_name, node.tail)}")
    return ", ".join(parts) + ")"


def _python_entries(items: tuple[tuple[Any, Snapshot], ...]) -> list[str]:
    return [f"{_python_key(key)}: {render_python(value)}" for key, value in items]


def _python_mapping(node: Mapping) -> str:
    if node.complete:
        return "{" + ", ".join(_python_entries(node.items)) + "}"
    if node.reason == "max_depth" and not node.items:
        return f"{{{node.type_name}: {_length(node.length)} items}}"
    parts = _python_entries(node.items)
    if node.tail:
        parts.append("...")
        parts.extend(_python_entries(node.tail))
    return f"{node.type_name}(len={_length(node.length)}, items={{" + ", ".join(parts) + "})"


def _python_object(node: Object) -> str:
    parts = [f"{name}={render_python(value)}" for name, value in node.fields]
    if not node.complete:
        parts.append("...")
    return f"{node.type_name}(" + ", ".join(parts) + ")"


def render_python(snapshot: Snapshot) -> str:
    """``snapshot`` in agentdoc's ``pformat`` / CodeAct prefill notation, on one line.

    Complete values are Python literals; cut ones use the slice-keys markers
    ``str(len=N, [:h]='...', [-t:]='...')``, ``list(len=N, [:h]=[...], [-t:]=[...])``
    and ``dict(len=N, items={..., ...})``; depth cuts ``[list: N items]``;
    cycles ``<cycle>``; uninspected values ``<opaque: Type>``.
    """
    if type(snapshot) is Scalar:
        return repr(snapshot.value)
    if type(snapshot) is Text:
        return _python_text(snapshot)
    if type(snapshot) is Sequence:
        return _python_sequence(snapshot)
    if type(snapshot) is Mapping:
        return _python_mapping(snapshot)
    if type(snapshot) is Object:
        return _python_object(snapshot)
    if type(snapshot) is Opaque:
        return f"<opaque: {snapshot.type_name}>"
    if type(snapshot) is Ref:
        return "<cycle>"
    if type(snapshot) is Unavailable:
        return f"<unavailable: {snapshot.reason}>"
    raise TypeError(f"not a snapshot node: {type(snapshot).__name__}")


# --- JSON notation (trace previews) ---------------------------------------------------


class _Report:
    """Incomplete JSON Pointer paths (bounded, collapsing to the root) and reasons."""

    def __init__(self) -> None:
        self.paths: list[str] = []
        self.path_chars = 0
        self.collapsed = False
        self.reasons: set[str] = set()

    def incomplete(self, path: str, reason: str | None) -> None:
        if reason is not None:
            self.reasons.add(reason)
        if self.collapsed or path in self.paths:
            return
        if (
            len(self.paths) >= _MAX_INCOMPLETE_PATHS
            or self.path_chars + len(path) > _MAX_INCOMPLETE_PATH_CHARS
        ):
            self.paths = [""]
            self.path_chars = 0
            self.collapsed = True
            return
        self.paths.append(path)
        self.path_chars += len(path)


def _json_key(key: Any) -> str | None:
    """The JSON object key a snapshot key becomes, as ``json.dumps`` converts it."""
    if type(key) is str:
        return key
    if key is None:
        return "null"
    if type(key) is bool:
        return "true" if key else "false"
    if type(key) is int or type(key) is float:
        return repr(key)
    return None


def _placeholder(node: Snapshot) -> Any:
    """The minimal value of a wrapper entry's own type (#342's ``trace_fields``)."""
    node_type = type(node)
    if node_type is Text:
        return ""
    if node_type is Sequence:
        return []
    if node_type is Mapping or node_type is Object:
        return {}
    if node_type is Ref:
        assert isinstance(node, Ref)
        return [] if node.kind == "sequence" else {}
    if node_type is Scalar:
        assert isinstance(node, Scalar)
        value = node.value
        if type(value) is bytes:
            return ""
        return fallback_candidates(type(value).__name__)[0]
    if node_type is Unavailable:
        assert isinstance(node, Unavailable)
        return fallback_candidates(node.type_name)[0] if node.type_name else None
    return None


def _json_wrapper(
    entries: tuple[tuple[Any, Snapshot], ...], max_size: int, path: str, report: _Report
) -> tuple[dict[str, Any], int] | None:
    """A ``capture_fields`` wrapper: each entry reserves its placeholder first."""
    shapes: dict[str, Any] = {}
    for key, child in entries:
        if type(key) is not str or key in shapes:
            return None
        shapes[key] = _placeholder(child)
    total = len(dumps(shapes))
    if total > max_size:
        return None
    output = dict(shapes)
    for key, child in entries:
        reserved = len(dumps(shapes[key]))
        available = reserved + (max_size - total)
        value, child_size = _json(child, available, pointer_child(path, key), report)
        output[key] = value
        total += child_size - reserved
    return output, total


def _json_entries(
    entries: tuple[tuple[Any, Snapshot], ...], max_size: int, path: str, report: _Report
) -> tuple[dict[str, Any], int, str | None]:
    output: dict[str, Any] = {}
    size = 2
    for key, child in entries:
        json_key = _json_key(key)
        if json_key is None or json_key in output:
            return output, size, "opaque"
        separator = 2 if output else 0
        remaining = max_size - size - separator
        if len(json_key) + 2 > remaining:
            return output, size, "max_length"
        overhead = len(dumps(json_key)) + 2
        available = remaining - overhead
        if available < 1:
            return output, size, "max_length"
        value, child_size = _json(child, available, pointer_child(path, json_key), report)
        output[json_key] = value
        size += separator + overhead + child_size
    return output, size, None


def _json(node: Snapshot, max_size: int, path: str, report: _Report) -> tuple[Any, int]:
    """One node as a JSON-compatible value fitted to ``max_size`` (#342's accounting)."""
    node_type = type(node)
    if node_type is Text or (node_type is Scalar and type(node.value) is bytes):  # type: ignore[union-attr]
        if node_type is Text:
            text: str = node.head  # type: ignore[union-attr]
        else:
            text = bytes(node.value).decode("utf-8", errors="replace")  # type: ignore[union-attr,arg-type]
        value, size = fit_string(text, max_size)
        if type(value) is not str or len(value) < len(text):
            report.incomplete(path, "max_length")
        if not node.complete:
            report.incomplete(path, node.reason)
        return value, size
    if node_type is Scalar:
        scalar = node.value  # type: ignore[union-attr]
        encoded = dumps(scalar)
        if len(encoded) <= max_size:
            return scalar, len(encoded)
        report.incomplete(path, "max_length")
        return fallback_json(type(scalar).__name__, max_size)
    if node_type is Sequence or node_type is Mapping or node_type is Object or node_type is Ref:
        if max_size < 2:
            report.incomplete(path, getattr(node, "reason", None) or "max_length")
            name = "list" if node_type is Sequence else "dict"
            return fallback_json(name, max_size)
    if node_type is Sequence:
        assert isinstance(node, Sequence)
        output: list[Any] = []
        size = 2
        cut = False
        for index, child in enumerate(node.items):
            separator = 2 if output else 0
            available = max_size - size - separator
            if available < 1:
                cut = True
                break
            value, child_size = _json(child, available, pointer_child(path, str(index)), report)
            output.append(value)
            size += separator + child_size
        if cut:
            report.incomplete(path, "max_length")
        if not node.complete:
            report.incomplete(path, node.reason)
        return output, size
    if node_type is Mapping or node_type is Object:
        assert isinstance(node, (Mapping, Object))
        entries = node.items if isinstance(node, Mapping) else node.fields
        if isinstance(node, Mapping) and node.wrapper:
            wrapped = _json_wrapper(entries, max_size, path, report)
            if wrapped is not None:
                return wrapped
        mapping, size, stop = _json_entries(entries, max_size, path, report)
        if stop is not None:
            report.incomplete(path, stop)
        if not node.complete:
            report.incomplete(path, node.reason)
        return mapping, size
    if node_type is Ref:
        assert isinstance(node, Ref)
        report.incomplete(path, "cycle")
        return ([], 2) if node.kind == "sequence" else ({}, 2)
    if node_type is Opaque:
        assert isinstance(node, Opaque)
        report.incomplete(path, "opaque")
        fallback_type = node.type_name if node.module == "builtins" else ""
        return opaque_json(node.type_name, fallback_type, max_size)
    if node_type is Unavailable:
        assert isinstance(node, Unavailable)
        report.incomplete(path, node.reason)
        return fallback_json(node.type_name, max_size)
    raise TypeError(f"not a snapshot node: {node_type.__name__}")


def _render(snapshot: Snapshot, max_chars: int) -> tuple[str, _Report]:
    if max_chars < 1:
        raise ValueError("render_json max_chars must be at least 1")
    report = _Report()
    value, size = _json(snapshot, max_chars, "", report)
    text = dumps(value)
    if len(text) != size or len(text) > max_chars:
        raise AssertionError("snapshot JSON accounting disagrees with the encoded size")
    return text, report


def render_json(snapshot: Snapshot, *, max_chars: int) -> str:
    """``snapshot`` as plain, valid JSON of at most ``max_chars`` characters.

    Strings keep a prefix; containers their first items; uninspected values
    become ``"<opaque: Type>"``; cycles ``[]`` / ``{}``. Nothing marks a cut
    inside the data: ``preview_attributes`` reports where the preview is
    incomplete.
    """
    return _render(snapshot, max_chars)[0]


def _spill_paths(node: Snapshot, found: list[str]) -> None:
    node_type = type(node)
    if node_type is Text:
        assert isinstance(node, Text)
        if node.full is not None and node.full.path not in found:
            found.append(node.full.path)
    elif node_type is Sequence:
        assert isinstance(node, Sequence)
        for child in node.items + node.tail:
            _spill_paths(child, found)
    elif node_type is Mapping:
        assert isinstance(node, Mapping)
        for _key, child in node.items + node.tail:
            _spill_paths(child, found)
    elif node_type is Object:
        assert isinstance(node, Object)
        for _name, child in node.fields:
            _spill_paths(child, found)


def preview_attributes(
    direction: str, snapshot: Snapshot, *, max_chars: int | None = None
) -> dict[str, Any]:
    """Span attributes for a JSON preview of ``snapshot`` (``direction``: input/output).

    Always ``nooa.{direction}.preview.version`` and ``.incomplete``; when
    incomplete, ``.paths`` (RFC 6901 JSON Pointers, at most 32 and 2,048
    characters, else the root ``""``) and ``.reasons`` (sorted, unique);
    ``.full_output`` lists spill files, when any. Pass the ``max_chars`` given
    to ``render_json`` so that cuts made while rendering are reported too.
    """
    _text, report = _render(snapshot, sys.maxsize if max_chars is None else max_chars)
    prefix = f"nooa.{direction}.preview"
    attributes: dict[str, Any] = {
        f"{prefix}.version": PREVIEW_VERSION,
        f"{prefix}.incomplete": bool(report.paths),
    }
    if report.paths:
        attributes[f"{prefix}.paths"] = tuple(report.paths)
    if report.reasons:
        attributes[f"{prefix}.reasons"] = tuple(sorted(report.reasons))
    spills: list[str] = []
    _spill_paths(snapshot, spills)
    if spills:
        attributes[f"{prefix}.full_output"] = tuple(spills)
    return attributes
