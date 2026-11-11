# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The renderers: ``render_output``, ``render_python`` (with its hooks), ``render_json``, ``render_trace``.

Rendering is the last step: it reads detached nodes only, never a live
value, and does not import the capture layers. It shares the JSON
accounting with capture through ``_codec``, so ``render_json`` of a
captured snapshot keeps exactly what capture kept.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Callable
from typing import Any, cast

from nooa.snapshot._codec import (
    Container,
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
    if type_name == "tuple":
        return "(", ")"
    if type_name == "set":
        return "{", "}"
    if type_name == "frozenset":
        return "frozenset({", "})"
    return "[", "]"


_UNORDERED = ("set", "frozenset")
_SEQUENCE_TYPES = ("list", "tuple", "set", "frozenset")


TextHook = Callable[[Object, "PythonRenderer"], str]
"""A ``render_python`` hook: ``hook(node, r) -> str``; it renders children only through ``r``."""

_MAX_HOOK_NESTING = 64  # hooks running inside hooks: deeper nodes render generically


class PythonRenderer:
    """The renderer ``r`` a ``render_python`` hook receives: ``hook(node, r) -> str``.

    ``render_python`` makes one per call; a hook never builds one. Methods:

    - ``r.render(child)``: ``child`` in the renderer's notation, with its cut
      markers, through the child's own hook when it has one. A hook renders
      every child this way.
    - ``r.generic(node)``: ``node`` without its own hook (its children still
      use theirs); a hook calls it to extend the generic notation.

    Hooks nest at most 64 deep: a node met while 64 hooks are running
    renders generically. So a hook that renders its own node through ``r``,
    or builds new nodes of its kind, still ends, well inside Python's
    recursion limit. There is no length limit: the nodes were bounded by
    ``capture``, and the string a hook returns is not checked for length.
    """

    def __init__(self, hooks: dict[str, TextHook], width: int | None = None) -> None:
        self._hooks = hooks
        self._depth = 0  # hooks running
        self._width = width  # None: one line
        self._level = 0  # indentation of the container being expanded

    def render(self, node: Snapshot) -> str:
        """``node`` in this renderer's notation, through its kind's hook when one is registered.

        A hook that raises an ``Exception`` or returns a non-``str``, and any
        node met while 64 hooks are running, renders as ``generic(node)``.
        Raises ``TypeError`` when ``node`` is not a snapshot node.
        """
        kind = node.kind if type(node) is Object else None
        hook = None
        if kind is not None:
            hook = self._hooks.get(kind) or _STANDARD_HOOKS.get(kind)
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
        """``node`` in the generic notation, without its own hook; its children use theirs.

        Raises ``TypeError`` when ``node`` is not a snapshot node.
        """
        if type(node) is Scalar:
            return repr(node.value)
        if type(node) is Text:
            return self._text(node)
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

    def _text(self, text: Text) -> str:
        """A ``Text`` node: a literal when complete, else the slice-keys marker."""
        return _python_text(text)

    def _layout(self, flat: Callable[[], str], expanded: Callable[[str, str], str]) -> str:
        """``flat()`` when it is shorter than the width (or no width is set), else ``expanded(ind, inner)``.

        ``expanded`` gets this level's indentation and the next one's; the
        children it renders are one level deeper. This is agentdoc's layout:
        a container goes on several lines when its one-line form is not
        shorter than the width.
        """
        width = self._width
        if width is None:
            return flat()
        self._width = None
        try:
            one_line = flat()
        finally:
            self._width = width
        if len(one_line) < width:
            return one_line
        indent = "    " * self._level
        self._level += 1
        try:
            return expanded(indent, indent + "    ")
        finally:
            self._level -= 1

    def _lines(self, inner: str, rendered: list[str]) -> str:
        return "".join(f"{inner}{item},\n" for item in rendered)

    def _items(self, type_name: str, items: tuple[Snapshot, ...]) -> str:
        left, right = _brackets(type_name)
        inner = ", ".join(self.render(item) for item in items)
        if type_name == "tuple" and len(items) == 1:
            inner += ","
        return f"{left}{inner}{right}"

    def _whole_sequence(self, node: Sequence) -> str:
        """A complete sequence: a literal, prefixed with its type unless it is a built-in one."""
        if not node.head:
            if node.type_name in _UNORDERED:
                return f"{node.type_name}()"
            left, right = _brackets(node.type_name)
            empty = left + right
            return empty if node.type_name in _SEQUENCE_TYPES else f"{node.type_name}({empty})"
        prefix, suffix = (
            ("", "") if node.type_name in _SEQUENCE_TYPES else (f"{node.type_name}(", ")")
        )
        left, right = _brackets(node.type_name)

        def expanded(indent: str, inner: str) -> str:
            rendered = [self.render(item) for item in node.head]
            return f"{prefix}{left}\n{self._lines(inner, rendered)}{indent}{right}{suffix}"

        return self._layout(
            lambda: prefix + self._items(node.type_name, node.head) + suffix, expanded
        )

    def _sequence(self, node: Sequence) -> str:
        left, right = _brackets(node.type_name)
        unordered = node.type_name in _UNORDERED
        if node.complete:
            return self._whole_sequence(node)
        if node.reason == "max_depth" and not node.head:
            return f"{left}{node.type_name}: {node.length} items{right}"
        start = f"{node.type_name}(len={node.length}"
        if unordered:

            def flat_set() -> str:
                inner = ", ".join([*(self.render(item) for item in node.head), "..."])
                return f"{start}, items={left}{inner}{right})"

            def expanded_set(indent: str, inner: str) -> str:
                rendered = [self.render(item) for item in node.head]
                body = self._lines(inner, rendered) + f"{inner}...\n"
                return f"{start}, items={left}\n{body}{indent}{right})"

            return self._layout(flat_set, expanded_set)

        def flat() -> str:
            parts = [start, f"[:{len(node.head)}]={self._items(node.type_name, node.head)}"]
            if node.tail:
                parts.append(f"[-{len(node.tail)}:]={self._items(node.type_name, node.tail)}")
            return ", ".join(parts) + ")"

        def expanded(indent: str, inner: str) -> str:
            self._level += 1
            try:
                head = [self.render(item) for item in node.head]
                tail = [self.render(item) for item in node.tail]
            finally:
                self._level -= 1
            deeper = inner + "    "
            text = f"{start},\n{inner}[:{len(head)}]={left}\n{self._lines(deeper, head)}{inner}{right},\n"
            if tail:
                text += (
                    f"{inner}[-{len(tail)}:]={left}\n{self._lines(deeper, tail)}{inner}{right},\n"
                )
            return text + f"{indent})"

        return self._layout(flat, expanded)

    def _entries(self, items: tuple[tuple[Any, Snapshot], ...]) -> list[str]:
        return [f"{_python_key(key)}: {self.render(value)}" for key, value in items]

    def _mapping(self, node: Mapping) -> str:
        if node.complete:
            if not node.head:
                return "{}" if node.type_name == "dict" else f"{node.type_name}({{}})"
            prefix, suffix = ("", "") if node.type_name == "dict" else (f"{node.type_name}(", ")")

            def expanded(indent: str, inner: str) -> str:
                body = self._lines(inner, self._entries(node.head))
                return f"{prefix}{{\n{body}{indent}}}{suffix}"

            return self._layout(
                lambda: prefix + "{" + ", ".join(self._entries(node.head)) + "}" + suffix,
                expanded,
            )
        if node.reason == "max_depth" and not node.head:
            return f"{{{node.type_name}: {node.length} items}}"
        start = f"{node.type_name}(len={node.length}, items={{"

        def flat_cut() -> str:
            parts = self._entries(node.head)
            if node.tail:
                parts.append("...")
                parts.extend(self._entries(node.tail))
            return start + ", ".join(parts) + "})"

        def expanded_cut(indent: str, inner: str) -> str:
            body = self._lines(inner, self._entries(node.head))
            if node.tail:
                body += f"{inner}...\n" + self._lines(inner, self._entries(node.tail))
            return f"{start}\n{body}{indent}}})"

        return self._layout(flat_cut, expanded_cut)

    def _object(self, node: Object) -> str:
        parts = [f"{name}={self.render(value)}" for name, value in node.fields]
        if not node.complete:
            parts.append("...")
        return f"{node.type_name}(" + ", ".join(parts) + ")"


# --- the standard types' notation (``_std``) ----------------------------------------


def _whole(node: Snapshot | None) -> str | bytes:
    """A complete ``Text`` / bytes ``Scalar``'s content; a hook that meets anything else falls back."""
    if type(node) is Text and node.complete:
        return node.head
    if type(node) is Scalar and type(node.value) is bytes:
        return node.value
    raise TypeError("not a complete text")


def _scalar(node: Snapshot | None) -> Any:
    if type(node) is not Scalar:
        raise TypeError("not a scalar")
    return node.value


def _repr_hook(node: Object, r: PythonRenderer) -> str:
    text = node.field("repr")
    if type(text) is not Text:
        raise TypeError("no repr")
    if text.complete:
        return cast(str, text.head)
    parts = [f"{node.type_name}(repr_len={text.length}", f"[:{len(text.head)}]={text.head!r}"]
    if text.tail:
        parts.append(f"[-{len(text.tail)}:]={text.tail!r}")
    return ", ".join(parts) + ")"


def _iso_hook(parse: Callable[[str], Any]) -> TextHook:
    def hook(node: Object, r: PythonRenderer) -> str:
        return repr(parse(cast(str, _whole(node.field("iso")))))

    return hook


def _timedelta_hook(node: Object, r: PythonRenderer) -> str:
    days, seconds, micros = (
        _scalar(node.field(name)) for name in ("days", "seconds", "microseconds")
    )
    return repr(datetime.timedelta(days=days, seconds=seconds, microseconds=micros))


def _decimal_hook(node: Object, r: PythonRenderer) -> str:
    return f"Decimal({_whole(node.field('value'))!r})"


def _uuid_hook(node: Object, r: PythonRenderer) -> str:
    return f"UUID({_whole(node.field('hex'))!r})"


def _path_hook(node: Object, r: PythonRenderer) -> str:
    return f"{node.type_name}({_whole(node.field('path'))!r})"


def _pattern_hook(node: Object, r: PythonRenderer) -> str:
    pattern = _whole(node.field("pattern"))
    flags = _scalar(node.field("flags"))
    if type(flags) is not int:
        raise TypeError("flags")
    if isinstance(pattern, str):
        flags &= ~re.UNICODE
    if not flags:
        return f"re.compile({pattern!r})"
    return f"re.compile({pattern!r}, {re.RegexFlag(flags)!r})"


def _range_hook(node: Object, r: PythonRenderer) -> str:
    start, stop, step = (_scalar(node.field(name)) for name in ("start", "stop", "step"))
    return repr(range(start, stop, step))


def _complex_hook(node: Object, r: PythonRenderer) -> str:
    return repr(complex(_scalar(node.field("real")), _scalar(node.field("imag"))))


def _function_hook(prefix: str) -> TextHook:
    def hook(node: Object, r: PythonRenderer) -> str:
        name = _whole(node.field("name"))
        signature = node.field("signature")
        tail = "" if signature is None else _whole(signature)
        return f"<{prefix} {name}{tail}>"

    return hook


def _named_hook(prefix: str) -> TextHook:
    def hook(node: Object, r: PythonRenderer) -> str:
        return f"<{prefix} {_whole(node.field('name'))!r}>"

    return hook


def _enum_hook(node: Object, r: PythonRenderer) -> str:
    value = node.field("value")
    if value is None:
        raise TypeError("no value")
    name_node = node.field("name")
    if type(name_node) is Scalar and name_node.value is None:
        return f"<{node.type_name}: {r.render(value)}>"
    return f"<{node.type_name}.{_whole(name_node)}: {r.render(value)}>"


def _exception_hook(node: Object, r: PythonRenderer) -> str:
    args = node.field("args")
    if type(args) is Sequence and args.complete:
        return f"{node.type_name}(" + ", ".join(r.render(item) for item in args.head) + ")"
    if args is None:
        raise TypeError("no args")
    return f"{node.type_name}(args={r.render(args)})"


_STANDARD_HOOKS: dict[str, TextHook] = {
    "python.repr": _repr_hook,
    "python.datetime": _iso_hook(datetime.datetime.fromisoformat),
    "python.date": _iso_hook(datetime.date.fromisoformat),
    "python.time": _iso_hook(datetime.time.fromisoformat),
    "python.timedelta": _timedelta_hook,
    "python.decimal": _decimal_hook,
    "python.uuid": _uuid_hook,
    "python.path": _path_hook,
    "python.pattern": _pattern_hook,
    "python.range": _range_hook,
    "python.complex": _complex_hook,
    "python.function": _function_hook("function"),
    "python.method": _function_hook("bound method"),
    "python.builtin": _function_hook("built-in function"),
    "python.module": _named_hook("module"),
    "python.class": _named_hook("class"),
    "python.enum": _enum_hook,
    "python.exception": _exception_hook,
}
"""The notation of the standard types capture reads natively; a registered hook of the same kind wins."""


class _OutputNotation(PythonRenderer):
    """``render_output``'s walk: ``Text`` in the output notation, no hooks."""

    def _text(self, text: Text) -> str:
        return _output_text(text)


class _RenderPython:
    """``render_python(snapshot) -> str``: a snapshot in agentdoc's ``pformat`` / prefill notation.

    Complete values are Python literals (a string with newlines in triple
    quotes). Cut values use the slice-key markers:

    - ``str(len=N, [:h]='...', [-t:]='...')`` and ``bytes(len=N, ...)``
      with ``bytes`` literals of the kept bytes;
    - ``list(len=N, [:h]=[...], [-t:]=[...])`` (and ``tuple``);
    - ``dict(len=N, items={..., ..., ...})``;
    - ``[list: N items]`` / ``{dict: N items}`` at ``max_depth``;
    - ``Type(a=1, ...)`` for an incomplete ``Object``;
    - ``<cycle>`` for a ``Ref``, ``<opaque: Type>`` for an uninspected
      value, ``<unavailable: reason>`` for another ``Unavailable``.

    The standard types capture reads natively render in their familiar
    notation: ``{1, 2}``, ``range(0, 10)``, ``(1+2j)``,
    ``datetime.datetime(2026, 11, 11, 9, 30)``, ``Decimal('1.50')``,
    ``UUID('...')``, ``PosixPath('/tmp/x')``, ``re.compile('a+b')``,
    ``ValueError('boom')``, ``<Color.RED: 1>``, ``Pt(x=1, y=2)``,
    ``Counter({'a': 2})``, ``<function mod.f(a, b=2)>``, ``<class 'mod.C'>``,
    ``<module 'json'>``; a ``repr`` fallback (``Limits.fallback="repr"``)
    as the repr itself, or ``T(repr_len=N, [:h]='...', [-t:]='...')`` when
    cut. A registered hook for one of their kinds (``"python.datetime"``...)
    replaces the standard one.

    String keys longer than 50 characters are cut the same way. The output
    has no length limit: it formats what ``capture`` bounded. With
    ``width=`` it is laid out on several lines as agentdoc does.

    Hooks (the one extension point for rendering): an ``Object`` whose
    ``kind`` has a hook renders as ``hook(node, r)``, with ``r`` a
    ``PythonRenderer``. Register with ``@render_python.register(kind)`` or
    ``render_python.register(kind, hook)``; remove with
    ``render_python.unregister(kind)``. The registry is process-wide.
    Rendering never fails because of a hook: one that raises or returns a
    non-``str``, or one nested more than 64 deep, falls back to the generic
    notation. Raises ``TypeError`` when ``snapshot`` is not a snapshot node.

    >>> from nooa.snapshot import Limits, Object, capture, render_python
    >>> render_python(capture({"a": "x" * 100}, Limits(max_string=6, tail_share=0.5)))
    "{'a': str(len=100, [:3]='xxx', [-3:]='xxx')}"
    >>> @render_python.register("point")
    ... def point(node: Object, r) -> str:
    ...     return f"<{r.render(node.field('x'))}, {r.render(node.field('y'))}>"
    >>> node = Object("P", (("x", capture(1)), ("y", capture(2))), kind="point")
    >>> render_python(node)
    '<1, 2>'
    >>> render_python.unregister("point")
    """

    def __init__(self) -> None:
        self._hooks: dict[str, TextHook] = {}

    def __call__(self, snapshot: Snapshot, *, width: int | None = None) -> str:
        """``snapshot`` in agentdoc's ``pformat`` / prefill notation; see ``render_python``.

        ``width`` (keyword-only, default ``None``: one line): agentdoc's
        layout. A container whose one-line form is not shorter than
        ``width`` characters goes on several lines, one item per line,
        indented by four spaces per level (``0`` expands every non-empty
        container). Objects stay on one line; their fields' containers
        follow the same rule.
        """
        return PythonRenderer(self._hooks, width).render(snapshot)

    def register(self, kind: str, function: TextHook | None = None) -> Any:
        """Register ``hook(node: Object, r: PythonRenderer) -> str`` for objects of ``kind``.

        - ``kind``: a non-empty ``str``, matched against ``Object.kind``. The
          library need not be imported to register a hook.
        - ``function`` (default ``None``): the hook. ``None``: return a
          decorator, ``@render_python.register("numpy.ndarray")``.

        Returns the hook (or the decorator). Registering a kind again
        replaces its hook. A hook renders children only through
        ``r.render(child)``; one that raises an ``Exception`` or returns a
        non-``str`` falls back to the generic notation. Raises ``TypeError``
        when ``kind`` is not a non-empty ``str``.
        """
        if type(kind) is not str or not kind:
            raise TypeError("render_python.register needs a kind (a non-empty str)")

        def add(function: TextHook) -> TextHook:
            self._hooks[kind] = function
            return function

        return add(function) if function is not None else add

    def unregister(self, kind: str) -> None:
        """Remove the hook for ``kind``; a no-op when there is none."""
        self._hooks.pop(kind, None)


render_python = _RenderPython()


def render_output(snapshot: Snapshot) -> str:
    """``snapshot`` as the model sees command output: ``HEAD``, one notice line, ``TAIL``.

    A complete ``Text`` is its content. A cut ``Text`` is its head, a line
    that says how much was left out (in its unit) and where the whole output
    is, and its tail, each on its own line when not empty::

        [... 1,234 bytes omitted; full output: /path]
        [... 1,234 bytes omitted; /path holds the first 65,536 of 99,999 bytes]
        [... 1,234 bytes omitted; full output not kept]

    Bytes are decoded as UTF-8 with replacement. Other nodes use
    ``render_python``'s generic notation without hooks, with nested ``Text``
    in the output notation. No length limit: it prints what the node kept.
    Raises ``TypeError`` when ``snapshot`` is not a snapshot node.

    >>> from nooa.snapshot import Spill, Text, render_output
    >>> text = Text(head="ab", tail="yz", length=10, unit="chars", reason="max_length",
    ...             full=Spill("/tmp/out.txt", 10, 10))
    >>> print(render_output(text))
    ab
    [... 6 chars omitted; full output: /tmp/out.txt]
    yz
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
    if type(node) is Text:
        if node.full is not None and node.full.path not in found:
            found.append(node.full.path)
    elif type(node) is Sequence:
        for child in node.head + node.tail:
            _spill_paths(child, found)
    elif type(node) is Mapping:
        for _key, child in node.head + node.tail:
            _spill_paths(child, found)
    elif type(node) is Object:
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
        if node.builtin:  # a non-finite float: capture reserved its 0.0
            return fallback_candidates("float")[0]
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
    entries: tuple[tuple[Any, Snapshot], ...],
    reason: str | None,
    max_size: int,
    path: str,
    report: _Report,
) -> tuple[Any, int]:
    """A ``Mapping``'s items or an ``Object``'s fields, cut with ``reason`` (``None``: whole)."""
    output: dict[str, Any] = {}
    box = Container(max_size)
    for key, child in entries:
        key_text = json_key(key, live=False)  # a detached snapshot key
        if key_text is None or key_text in output:
            box.stop("key")
            break
        slot = box.slot(key_text)
        if slot is None:
            break
        overhead, available = slot
        value, child_size = _json(child, available, pointer_child(path, key_text), report)
        output[key_text] = value
        box.commit(overhead, child_size)
    if box.reason is not None:
        report.incomplete(path, box.reason)
    if reason is not None:
        report.incomplete(path, reason)
    return output, box.size


def _json_string(text: str, max_size: int, path: str, report: _Report) -> tuple[Any, int]:
    """A string's longest prefix that fits; a shorter one is reported as cut."""
    value, size = fit_string(text, max_size)
    if type(value) is not str or len(value) < len(text):
        report.incomplete(path, "max_length")
    return value, size


def _json_text(node: Text, max_size: int, path: str, report: _Report) -> tuple[Any, int]:
    chars = node.unit == "chars"
    head = node.head
    if type(head) is not (str if chars else bytes):
        raise TypeError("a Text node's head is a str (chars) or bytes (bytes)")
    text = cast(str, head) if chars else decode(cast(bytes, head))  # bytes: decoded
    built = _json_string(text, max_size, path, report)
    if not node.complete:
        report.incomplete(path, node.reason)
    return built


def _json_scalar(node: Scalar, max_size: int, path: str, report: _Report) -> tuple[Any, int]:
    scalar = node.value
    if type(scalar) is bytes:  # short bytes: shown decoded, like a bytes ``Text``
        return _json_string(decode(scalar), max_size, path, report)
    encoded = dumps(scalar)
    if len(encoded) <= max_size:
        return scalar, len(encoded)
    report.incomplete(path, "max_length")
    return fallback_json(type(scalar).__name__, max_size)


def _json_sequence(node: Sequence, max_size: int, path: str, report: _Report) -> tuple[Any, int]:
    output: list[Any] = []
    box = Container(max_size)
    for index, child in enumerate(node.head):
        slot = box.slot()
        if slot is None:
            report.incomplete(path, box.reason)
            break
        value, child_size = _json(child, slot[1], pointer_child(path, str(index)), report)
        output.append(value)
        box.commit(0, child_size)
    if not node.complete:
        report.incomplete(path, node.reason)
    return output, box.size


def _json_unavailable(
    node: Unavailable, max_size: int, path: str, report: _Report
) -> tuple[Any, int]:
    report.incomplete(path, node.reason)
    if node.reason == "opaque":
        fallback_type = "float" if node.builtin else ""  # builtin: a non-finite float
        return opaque_json(node.type_name, fallback_type, max_size)
    return fallback_json(node.type_name, max_size)


def _json(node: Snapshot, max_size: int, path: str, report: _Report) -> tuple[Any, int]:
    """One node as a JSON-compatible value fitted to ``max_size`` (#342's accounting).

    One small rule per node type, by exact type. A container with less than
    2 characters left has no room for its brackets: a fallback.
    """
    if type(node) is Scalar:
        return _json_scalar(node, max_size, path, report)
    if type(node) is Text:
        return _json_text(node, max_size, path, report)
    if type(node) is Unavailable:
        return _json_unavailable(node, max_size, path, report)
    if type(node) is Ref:
        reason, empty = "cycle", ([], 2) if node.kind == "sequence" else ({}, 2)
    elif type(node) is Sequence or type(node) is Mapping or type(node) is Object:
        reason, empty = node.reason, None
    else:
        raise TypeError(f"not a snapshot node: {type(node).__name__}")
    if max_size < 2:
        report.incomplete(path, reason or "max_length")
        return fallback_json("list" if type(node) is Sequence else "dict", max_size)
    if empty is not None:  # a Ref
        report.incomplete(path, reason)
        return empty
    if type(node) is Sequence:
        return _json_sequence(node, max_size, path, report)
    if type(node) is Mapping:
        wrapped = _json_wrapper(node.head, max_size, path, report) if node.wrapper else None
        return wrapped or _json_entries(node.head, reason, max_size, path, report)
    return _json_entries(cast(Object, node).fields, reason, max_size, path, report)


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
    """``snapshot`` as plain, valid JSON of at most ``max_chars`` characters (keyword-only).

    The encoding is canonical: ASCII-only, ``", "`` and ``": "``
    separators, keys in snapshot order. Strings keep a prefix (bytes are
    decoded as UTF-8 with replacement); containers keep their first items
    (a kept ``tail`` is not shown); uninspected values become
    ``"<opaque: Type>"``; cycles ``[]`` / ``{}``; other ``Unavailable``
    nodes, and anything that does not fit, the smallest value of their
    type (``""``, ``0``, ``[]``, ``{}``, ...). Mapping keys convert as
    ``json.dumps`` converts them, and a mapping stops at a key that would
    collide. A ``capture_fields`` wrapper keeps every field. Nothing marks a
    cut inside the data: ``render_trace`` reports where the text is
    incomplete. No hooks.

    Exact accounting: for ``snapshot = capture(v, limits)``,
    ``render_json(snapshot, max_chars=limits.max_chars)`` cuts nothing
    further, and on #342's domain it is #342's text. A smaller
    ``max_chars`` cuts further and stays valid.

    Raises ``ValueError`` when ``max_chars`` is less than 1, ``TypeError``
    for a node that is not a snapshot node.

    >>> from nooa.snapshot import capture, render_json
    >>> render_json(capture({"a": "x" * 30, "b": [1, 2]}), max_chars=20)
    '{"a": "xxxxxxxxxxx"}'
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


def render_trace(
    snapshot: Snapshot, *, direction: str, max_chars: int
) -> tuple[str, dict[str, Any]]:
    """A trace span's preview of ``snapshot``: ``(json_text, attributes)``, from one JSON walk.

    - ``snapshot``: the node to preview.
    - ``direction`` (keyword-only): ``"input"`` or ``"output"``, the middle
      of the attribute names.
    - ``max_chars`` (keyword-only): the bound of the text, as for
      ``render_json``.

    ``json_text`` is ``render_json(snapshot, max_chars=max_chars)``. The
    attributes say whether and where it is incomplete, including the cuts
    rendering made (with ``prefix = f"nooa.{direction}.preview"``):

    - ``{prefix}.version``: always, ``1``;
    - ``{prefix}.incomplete``: always, a ``bool``;
    - ``{prefix}.paths``: when incomplete, the JSON Pointers (RFC 6901) of
      the incomplete nodes, a tuple of at most 32 paths and 2,048
      characters, else ``("",)`` (the root);
    - ``{prefix}.reasons``: when incomplete, the sorted, unique reasons;
    - ``{prefix}.full_output``: when any ``Text`` in the snapshot spilled,
      the spill file paths, depth first, whatever the text shows.

    The version, incomplete and paths attributes are #342's. Raises as
    ``render_json`` does.

    >>> from nooa.snapshot import capture, render_trace
    >>> text, attributes = render_trace(capture({"a": "x" * 30}), direction="output", max_chars=20)
    >>> text
    '{"a": "xxxxxxxxxxx"}'
    >>> attributes["nooa.output.preview.paths"], attributes["nooa.output.preview.reasons"]
    (('/a',), ('max_length',))
    """
    text, report = _render(snapshot, max_chars)
    return text, _attributes(direction, report, snapshot)
