# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Renderers: one snapshot, the notation each consumer shows.

- ``render_output(snapshot)``: Shell-style output, ``HEAD\n[... N bytes omitted; ...]\nTAIL``.
- ``render_python(snapshot)``: agentdoc's ``pformat`` / prefill notation.
- ``render_json(snapshot, max_chars=...)``: plain, valid JSON within ``max_chars``
  (#342's trace preview semantics: nothing is embedded in user data).
- ``preview_attributes(direction, snapshot)``: the trace span attributes that
  say whether and where a JSON preview is incomplete.

``render_python``, ``render_output`` and ``render_json`` each have a per-kind
hook registry (``render_python.register("numpy.ndarray")``): an ``Object``
whose ``kind`` has a hook renders through it, its children through the
renderer ``r`` the hook is given. A missing or failing hook means generic
rendering; rendering never fails because of a hook.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
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

__all__ = [
    "JsonRenderer",
    "OutputRenderer",
    "PythonRenderer",
    "preview_attributes",
    "render_json",
    "render_output",
    "render_python",
]

PREVIEW_VERSION = 1
_MAX_INCOMPLETE_PATHS = 32
_MAX_INCOMPLETE_PATH_CHARS = 2_048
_MAX_KEY_CHARS = 50


# --- output notation ---------------------------------------------------------------


def _output_text(text: Text) -> str:
    """``text`` in the output notation: whole, or head and tail around one line."""
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


def _length(length: int | None) -> str:
    return "?" if length is None else str(length)


TextHook = Callable[[Object, Any], str]
"""A per-kind render hook: ``hook(node, r) -> str``; it renders children only through ``r``."""


class _Hooks:
    """One renderer's per-kind hook registry (``kind`` is a plain string)."""

    def __init__(self, renderer: str) -> None:
        self.renderer = renderer
        self.functions: dict[str, Callable[..., str]] = {}

    def register(self, kind: str, function: Callable[..., str] | None = None) -> Any:
        if type(kind) is not str or not kind:
            raise TypeError(f"{self.renderer}.register needs a kind (a non-empty str)")

        def add(function: Callable[..., str]) -> Callable[..., str]:
            self.functions[kind] = function
            return function

        return add(function) if function is not None else add

    def unregister(self, kind: str) -> None:
        self.functions.pop(kind, None)

    def find(self, node: Snapshot) -> Callable[..., str] | None:
        if type(node) is not Object:
            return None
        kind = node.kind
        return self.functions.get(kind) if type(kind) is str else None


class PythonRenderer:
    """What a ``render_python`` hook renders children with: ``r.render(child)``.

    Children rendered through ``r`` use the renderer's own notation and cut
    markers (and their own hooks). ``r.generic(node)`` renders a node without
    its hook.
    """

    def __init__(self, hooks: _Hooks) -> None:
        self._hooks = hooks
        self._active: list[int] = []  # ids of the nodes whose hook is running

    def render(self, node: Snapshot) -> str:
        """``node`` in this renderer's notation, through its hook when one is registered."""
        hook = self._hooks.find(node)
        if hook is not None and id(node) not in self._active:
            self._active.append(id(node))
            try:
                rendered = hook(node, self)
            except Exception:
                rendered = None
            finally:
                self._active.pop()
            if type(rendered) is str:
                return rendered
        return self.generic(node)

    def generic(self, node: Snapshot) -> str:
        """``node`` rendered generically (no hook for ``node`` itself; children use theirs)."""
        if type(node) is Scalar:
            return repr(node.value)
        if type(node) is Text:
            return self.text(node)
        if type(node) is Sequence:
            return self._sequence(node)
        if type(node) is Mapping:
            return self._mapping(node)
        if type(node) is Object:
            return self._object(node)
        if type(node) is Opaque:
            return f"<opaque: {node.type_name}>"
        if type(node) is Ref:
            return "<cycle>"
        if type(node) is Unavailable:
            return f"<unavailable: {node.reason}>"
        raise TypeError(f"not a snapshot node: {type(node).__name__}")

    def text(self, text: Text) -> str:
        """A ``Text`` node: a literal when complete, else the slice-keys marker."""
        return _python_text(text)

    def _items(self, type_name: str, items: tuple[Snapshot, ...]) -> str:
        left, right = _brackets(type_name)
        inner = ", ".join(self.render(item) for item in items)
        if type_name == "tuple" and len(items) == 1:
            inner += ","
        return f"{left}{inner}{right}"

    def _sequence(self, node: Sequence) -> str:
        left, right = _brackets(node.type_name)
        if node.complete:
            return self._items(node.type_name, node.items)
        if node.reason == "max_depth" and not node.items:
            return f"{left}{node.type_name}: {_length(node.length)} items{right}"
        parts = [f"{node.type_name}(len={_length(node.length)}"]
        parts.append(f"[:{len(node.items)}]={self._items(node.type_name, node.items)}")
        if node.tail:
            parts.append(f"[-{len(node.tail)}:]={self._items(node.type_name, node.tail)}")
        return ", ".join(parts) + ")"

    def _entries(self, items: tuple[tuple[Any, Snapshot], ...]) -> list[str]:
        return [f"{_python_key(key)}: {self.render(value)}" for key, value in items]

    def _mapping(self, node: Mapping) -> str:
        if node.complete:
            return "{" + ", ".join(self._entries(node.items)) + "}"
        if node.reason == "max_depth" and not node.items:
            return f"{{{node.type_name}: {_length(node.length)} items}}"
        parts = self._entries(node.items)
        if node.tail:
            parts.append("...")
            parts.extend(self._entries(node.tail))
        return f"{node.type_name}(len={_length(node.length)}, items={{" + ", ".join(parts) + "})"

    def _object(self, node: Object) -> str:
        parts = [f"{name}={self.render(value)}" for name, value in node.fields]
        if not node.complete:
            parts.append("...")
        return f"{node.type_name}(" + ", ".join(parts) + ")"


class OutputRenderer(PythonRenderer):
    """What a ``render_output`` hook renders children with: ``Text`` in the output notation."""

    def text(self, text: Text) -> str:
        """A ``Text`` node: whole, or head and tail around the omitted-output line."""
        return _output_text(text)


class _TextRender:
    """A text renderer entry point: ``render(snapshot)`` plus its own hook registry.

    ``render_python`` and ``render_output`` are instances of the two tiny
    subclasses below; they differ only in the renderer class a hook gets (and
    their docstrings). Each has an independent registry. Hooks format nodes
    that ``capture`` already bounded: there is no output-length cap here (see
    *Extension contract* in ``docs/snapshot-model.md``).
    """

    _renderer: type[PythonRenderer] = PythonRenderer

    def __init__(self, name: str) -> None:
        self._hooks = _Hooks(name)

    def __call__(self, snapshot: Snapshot) -> str:
        return self._renderer(self._hooks).render(snapshot)

    def register(self, kind: str, function: TextHook | None = None) -> Any:
        """Register ``hook(node: Object, r) -> str`` for objects of ``kind``.

        ``r`` is this renderer's ``PythonRenderer`` / ``OutputRenderer``. Use as
        ``@render_python.register("numpy.ndarray")`` or ``register(kind, hook)``;
        registering again replaces it. A hook that raises or returns a non-``str``
        falls back to the generic notation.
        """
        return self._hooks.register(kind, function)

    def unregister(self, kind: str) -> None:
        """Remove the hook for ``kind`` (no-op when absent)."""
        self._hooks.unregister(kind)


class _RenderPython(_TextRender):
    """``render_python(snapshot)``, with ``render_python.register(kind)`` for per-kind hooks."""

    def __call__(self, snapshot: Snapshot) -> str:
        """``snapshot`` in agentdoc's ``pformat`` / CodeAct prefill notation, on one line.

        Complete values are Python literals; cut ones use the slice-keys markers
        ``str(len=N, [:h]='...', [-t:]='...')``, ``list(len=N, [:h]=[...], [-t:]=[...])``
        and ``dict(len=N, items={..., ...})``; depth cuts ``[list: N items]``;
        cycles ``<cycle>``; uninspected values ``<opaque: Type>``. An ``Object``
        whose ``kind`` has a hook renders through it; a hook that raises or
        returns a non-``str`` falls back to the generic notation.
        """
        return super().__call__(snapshot)


class _RenderOutput(_TextRender):
    """``render_output(snapshot)``, with ``render_output.register(kind)`` for per-kind hooks."""

    _renderer = OutputRenderer

    def __call__(self, snapshot: Snapshot) -> str:
        """``snapshot`` as the model sees command output.

        A ``Text`` is whole, or head and tail around one line that says how
        much was left out and where the whole output is::

            [... 1,234 bytes omitted; full output: /path]
            [... 1,234 bytes omitted; /path holds the first 65,536 of 99,999 bytes]
            [... 1,234 bytes omitted; full output not kept]

        Other nodes use ``render_python``'s notation, with nested ``Text`` in
        the output notation. An ``Object`` whose ``kind`` has a hook renders
        through it; a hook that raises falls back to the generic notation.
        """
        return super().__call__(snapshot)


render_python = _RenderPython("render_python")
render_output = _RenderOutput("render_output")


# --- JSON notation (trace previews) ---------------------------------------------------


class _Report:
    """Incomplete JSON Pointer paths (bounded, collapsing to the root) and reasons.

    It also carries the render hooks of the walk (``None``: generic only).
    """

    def __init__(self, hooks: _Hooks | None = None) -> None:
        self.hooks = hooks
        self.active: list[int] = []  # ids of the nodes whose hook is running
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

    def child(self) -> _Report:
        """A fresh report for a hook's attempt, merged only if the hook succeeds."""
        child = _Report(self.hooks)
        child.active = self.active
        return child

    def merge(self, other: _Report) -> None:
        for path in other.paths:
            self.incomplete(path, None)
        self.reasons |= other.reasons


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


class JsonRenderer:
    """What a ``render_json`` hook renders children with: ``r.render(child)``.

    ``r.max_chars`` is the budget of the node being rendered: the hook's
    output must be valid JSON whose canonical encoding fits it, otherwise the
    node is rendered generically. ``r.render`` returns a child's JSON text
    fitted to ``max_chars`` (at most ``r.max_chars``), with the usual cuts.
    """

    def __init__(self, max_chars: int, path: str, report: _Report) -> None:
        self.max_chars = max_chars
        self._path = path
        self._report = report

    def render(
        self, node: Snapshot, *, max_chars: int | None = None, key: str | None = None
    ) -> str:
        """``node`` as JSON text of at most ``max_chars`` characters.

        ``key``: the child's name in the hook's output, for the JSON Pointer
        paths that ``preview_attributes`` reports.
        """
        budget = self.max_chars if max_chars is None else min(max_chars, self.max_chars)
        if budget < 1:
            raise ValueError("JsonRenderer.render needs max_chars of at least 1")
        path = self._path if key is None else pointer_child(self._path, key)
        value, _size = _json(node, budget, path, self._report)
        return dumps(value)


def _reject_constant(name: str) -> Any:
    raise ValueError(f"not valid JSON: {name}")


def _json_hooked(
    node: Object, hook: Callable[..., str], max_size: int, path: str, report: _Report
) -> tuple[Any, int] | None:
    """A hook's JSON for ``node``, or ``None`` when it raises, is invalid or too long."""
    attempt = report.child()
    report.active.append(id(node))
    try:
        text = hook(node, JsonRenderer(max_size, path, attempt))
        if type(text) is not str or len(text) > 8 * max_size + 64:
            return None
        value = json.loads(text, parse_constant=_reject_constant)
        encoded = dumps(value)
    except Exception:
        return None
    finally:
        report.active.pop()
    if len(encoded) > max_size:
        return None
    report.merge(attempt)
    if not node.complete:
        report.incomplete(path, node.reason)
    return value, len(encoded)


def _json(node: Snapshot, max_size: int, path: str, report: _Report) -> tuple[Any, int]:
    """One node as a JSON-compatible value fitted to ``max_size`` (#342's accounting)."""
    node_type = type(node)
    if report.hooks is not None and node_type is Object and id(node) not in report.active:
        hook = report.hooks.find(node)
        if hook is not None:
            hooked = _json_hooked(node, hook, max_size, path, report)  # type: ignore[arg-type]
            if hooked is not None:
                return hooked
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
    hooks = render_json._hooks
    report = _Report(hooks if hooks.functions else None)
    value, size = _json(snapshot, max_chars, "", report)
    text = dumps(value)
    if len(text) != size or len(text) > max_chars:
        raise AssertionError("snapshot JSON accounting disagrees with the encoded size")
    return text, report


class _RenderJson:
    """``render_json(snapshot, max_chars=...)``, with ``render_json.register(kind)``."""

    def __init__(self) -> None:
        self._hooks = _Hooks("render_json")

    def __call__(self, snapshot: Snapshot, *, max_chars: int) -> str:
        """``snapshot`` as plain, valid JSON of at most ``max_chars`` characters.

        Strings keep a prefix; containers their first items; uninspected values
        become ``"<opaque: Type>"``; cycles ``[]`` / ``{}``. Nothing marks a cut
        inside the data: ``preview_attributes`` reports where the preview is
        incomplete. An ``Object`` whose ``kind`` has a hook renders through it
        when the hook returns valid JSON that fits the node's budget; otherwise
        (or when it raises) generically, so the result stays valid and bounded.
        """
        return _render(snapshot, max_chars)[0]

    def register(self, kind: str, function: Callable[..., str] | None = None) -> Any:
        """Register ``hook(node: Object, r: JsonRenderer) -> str`` (JSON text) for ``kind``.

        Trace previews stay generic unless a hook is registered: keep hook
        output plain data that tools can read.
        """
        return self._hooks.register(kind, function)

    def unregister(self, kind: str) -> None:
        """Remove the hook for ``kind`` (no-op when absent)."""
        self._hooks.unregister(kind)


render_json = _RenderJson()


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
