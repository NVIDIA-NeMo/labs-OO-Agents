# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Renderers: one snapshot, the notation each consumer shows.

- ``render_output(snapshot)``: Shell-style output, ``HEAD\n[... N bytes omitted; ...]\nTAIL``.
- ``render_python(snapshot)``: agentdoc's ``pformat`` / prefill notation.
- ``render_json(snapshot, max_chars=...)``: plain, valid JSON within ``max_chars``
  (#342's trace preview semantics: nothing is embedded in user data).
- ``preview_attributes(direction, snapshot)``: the trace span attributes that
  say whether and where a JSON preview is incomplete.
- ``render_trace(snapshot, direction=..., max_chars=...)``: both, for a trace span.

Only ``render_python`` has hooks, one per kind (``render_python.register(
"numpy.ndarray")``): an ``Object`` whose ``kind`` has a hook renders through
it, and the hook renders its children through the renderer ``r`` it is given.
A missing or failing hook, or one nested more than 64 deep, means generic
rendering; rendering never fails because of a hook. ``render_output`` and
``render_json`` are generic. Only ``render_json`` bounds its output
(``max_chars``); the text renderers format what ``capture`` kept
(*Extension contract* in ``docs/snapshot-model.md``).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any, cast

from nooa.snapshot._codec import (
    decode,
    dumps,
    fallback_candidates,
    fallback_json,
    fit_string,
    json_key,
    opaque_json,
    pointer_child,
)
from nooa.snapshot._model import (
    Mapping,
    Object,
    Ref,
    Scalar,
    Sequence,
    Snapshot,
    Text,
    Unavailable,
)

__all__ = [
    "PythonRenderer",
    "preview_attributes",
    "render_json",
    "render_output",
    "render_python",
    "render_trace",
]

PREVIEW_VERSION = 1
_MAX_INCOMPLETE_PATHS = 32
_MAX_INCOMPLETE_PATH_CHARS = 2_048
_MAX_KEY_CHARS = 50


# --- output notation ---------------------------------------------------------------


def _shown(part: str | bytes) -> str:
    """A kept part as text to print: bytes decode as UTF-8, with replacement."""
    return decode(part) if isinstance(part, bytes) else part


def _output_text(text: Text) -> str:
    """``text`` in the output notation: whole, or head and tail around one line."""
    head, tail = _shown(text.head), _shown(text.tail)
    if text.complete:
        return head + tail
    omitted = f"[... {text.omitted:,} {text.unit} omitted; "
    full = text.full
    if full is None:
        line = omitted + "full output not kept]"
    elif full.cut:
        line = omitted + f"{full.path} holds the first {full.kept:,} of {full.total:,} bytes]"
    else:
        line = omitted + f"full output: {full.path}]"
    return (head + "\n" if head else "") + line + ("\n" + tail if tail else "")


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
    """A literal when complete, else ``str(len=N, [:h]=..., [-t:]=...)`` (``bytes(...)``).

    A bytes ``Text`` shows the bytes it kept, as ``bytes`` literals.
    """
    head, tail = text.head, text.tail
    if text.complete:
        whole = head + tail  # type: ignore[operator]
        return repr(whole) if isinstance(whole, bytes) else _python_str(whole)
    parts = [f"{'bytes' if text.unit == 'bytes' else 'str'}(len={text.length}"]
    if head or not tail:
        parts.append(f"[:{len(head)}]={head!r}")
    if tail:
        parts.append(f"[-{len(tail)}:]={tail!r}")
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


TextHook = Callable[[Object, "PythonRenderer"], str]
"""A ``render_python`` hook: ``hook(node, r) -> str``; it renders children only through ``r``."""

_MAX_HOOK_NESTING = 64  # hooks running inside hooks: deeper nodes render generically


class PythonRenderer:
    """What a ``render_python`` hook renders children with: ``r.render(child)``.

    Children rendered through ``r`` use the renderer's own notation and cut
    markers (and their own hooks). ``r.generic(node)`` renders a node without
    its hook. Hooks nest at most 64 deep: a node met while 64 hooks are running
    renders generically, so a hook that renders its own node, or builds new
    nodes of its kind, still ends. There is no output-length limit: the nodes
    were bounded by ``capture``, and the string a hook returns is not checked
    for length (*Extension contract* in ``docs/snapshot-model.md``).
    """

    def __init__(self, hooks: dict[str, TextHook]) -> None:
        self._hooks = hooks
        self._depth = 0  # hooks running

    def render(self, node: Snapshot) -> str:
        """``node`` in this renderer's notation, through its hook when one is registered."""
        kind = node.kind if type(node) is Object else None
        hook = self._hooks.get(kind) if kind is not None else None
        if hook is not None and self._depth < _MAX_HOOK_NESTING:
            self._depth += 1
            try:
                rendered = hook(node, self)  # type: ignore[arg-type]
            except Exception:
                rendered = None
            finally:
                self._depth -= 1
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
        if type(node) is Ref:
            return "<cycle>"
        if type(node) is Unavailable:
            if node.reason == "opaque":
                return f"<opaque: {node.type_name}>"
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


class _OutputNotation(PythonRenderer):
    """``render_output``'s walk: ``Text`` in the output notation, no hooks."""

    def text(self, text: Text) -> str:
        return _output_text(text)


class _RenderPython:
    """``render_python(snapshot)``, with ``render_python.register(kind)`` for per-kind hooks."""

    def __init__(self) -> None:
        self._hooks: dict[str, TextHook] = {}

    def __call__(self, snapshot: Snapshot) -> str:
        """``snapshot`` in agentdoc's ``pformat`` / CodeAct prefill notation, not column-wrapped.

        Complete values are Python literals; cut ones use the slice-keys markers
        ``str(len=N, [:h]='...', [-t:]='...')``, ``list(len=N, [:h]=[...], [-t:]=[...])``
        and ``dict(len=N, items={..., ...})``; depth cuts ``[list: N items]``;
        cycles ``<cycle>``; uninspected values ``<opaque: Type>``. An ``Object``
        whose ``kind`` has a hook renders through it; a hook that raises or
        returns a non-``str``, or one nested more than 64 deep, falls back to
        the generic notation.
        """
        return PythonRenderer(self._hooks).render(snapshot)

    def register(self, kind: str, function: TextHook | None = None) -> Any:
        """Register ``hook(node: Object, r: PythonRenderer) -> str`` for objects of ``kind``.

        Use as ``@render_python.register("numpy.ndarray")`` or
        ``register(kind, hook)``; registering again replaces it. A hook that
        raises or returns a non-``str`` falls back to the generic notation.
        """
        if type(kind) is not str or not kind:
            raise TypeError("render_python.register needs a kind (a non-empty str)")

        def add(function: TextHook) -> TextHook:
            self._hooks[kind] = function
            return function

        return add(function) if function is not None else add

    def unregister(self, kind: str) -> None:
        """Remove the hook for ``kind`` (no-op when absent)."""
        self._hooks.pop(kind, None)


render_python = _RenderPython()


def render_output(snapshot: Snapshot) -> str:
    """``snapshot`` as the model sees command output.

    A ``Text`` is whole, or head and tail around one line that says how
    much was left out and where the whole output is::

        [... 1,234 bytes omitted; full output: /path]
        [... 1,234 bytes omitted; /path holds the first 65,536 of 99,999 bytes]
        [... 1,234 bytes omitted; full output not kept]

    Other nodes use ``render_python``'s generic notation (no hooks), with
    nested ``Text`` in the output notation.
    """
    return _OutputNotation({}).render(snapshot)


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


def _spill_paths(node: Snapshot, found: list[str]) -> list[str]:
    """Every spill file of ``node``, depth first (heads and tails, fields), once each."""
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
    return found


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
        if node.reason == "opaque" or not node.type_name:
            return None
        return fallback_candidates(node.type_name)[0]
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
        key_text = json_key(key, live=False)  # a detached snapshot key
        separator = 2 if output else 0
        remaining = max_size - size - separator
        if key_text is None or key_text in output:
            stop = "key"
        elif len(key_text) + 2 > remaining:
            stop = "max_length"
        else:
            overhead = len(dumps(key_text)) + 2
            available = remaining - overhead
            if available >= 1:
                value, child_size = _json(child, available, pointer_child(path, key_text), report)
                output[key_text] = value
                size += separator + overhead + child_size
                continue
            stop = "max_length"
        return output, size, stop
    return output, size, None


def _json(node: Snapshot, max_size: int, path: str, report: _Report) -> tuple[Any, int]:
    """One node as a JSON-compatible value fitted to ``max_size`` (#342's accounting)."""
    node_type = type(node)
    if node_type is Text or (node_type is Scalar and type(node.value) is bytes):  # type: ignore[union-attr]
        head: Any = node.head if node_type is Text else node.value  # type: ignore[union-attr]
        chars = node_type is Text and node.unit == "chars"  # type: ignore[union-attr]
        if type(head) is not (str if chars else bytes):
            raise TypeError("a Text node's head is a str (chars) or bytes (bytes)")
        text = cast(str, head) if chars else decode(cast(bytes, head))  # bytes: decoded
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
    if node_type is Unavailable:
        assert isinstance(node, Unavailable)
        report.incomplete(path, node.reason)
        if node.reason == "opaque":
            fallback_type = node.type_name if node.module == "builtins" else ""
            return opaque_json(node.type_name, fallback_type, max_size)
        return fallback_json(node.type_name, max_size)
    raise TypeError(f"not a snapshot node: {node_type.__name__}")


def _render(snapshot: Snapshot, max_chars: int) -> tuple[str, _Report]:
    """The JSON text and its report (incomplete paths and reasons), from one walk."""
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
    incomplete. There are no hooks: every node renders generically.
    """
    return _render(snapshot, max_chars)[0]


def _attributes(direction: str, report: _Report, snapshot: Snapshot) -> dict[str, Any]:
    """The span attributes of a report, plus the spill files of ``snapshot``."""
    prefix = f"nooa.{direction}.preview"
    attributes: dict[str, Any] = {
        f"{prefix}.version": PREVIEW_VERSION,
        f"{prefix}.incomplete": bool(report.paths),
    }
    if report.paths:
        attributes[f"{prefix}.paths"] = tuple(report.paths)
    if report.reasons:
        attributes[f"{prefix}.reasons"] = tuple(sorted(report.reasons))
    spills = _spill_paths(snapshot, [])
    if spills:
        attributes[f"{prefix}.full_output"] = tuple(spills)
    return attributes


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
    budget = sys.maxsize if max_chars is None else max_chars
    return _attributes(direction, _render(snapshot, budget)[1], snapshot)


def render_trace(
    snapshot: Snapshot, *, direction: str, max_chars: int
) -> tuple[str, dict[str, Any]]:
    """A trace span's preview of ``snapshot``: the JSON text and its span attributes.

    Equal to ``(render_json(snapshot, max_chars=m), preview_attributes(direction,
    snapshot, max_chars=m))``, from one JSON walk (``direction``: ``"input"`` or
    ``"output"``).
    """
    text, report = _render(snapshot, max_chars)
    return text, _attributes(direction, report, snapshot)
