# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The canonical key encoder: the JSON object key of a mapping key that has no JSON form.

A mapping key that is an exact ``str``, ``int``, ``float``, ``bool`` or
``None`` becomes its ``json.dumps`` key (``_codec.json_key``). Any other key
(a tuple, a frozenset, an enum member, a datetime...) is captured as a small
node, and ``key_text(node)`` is its key: capture uses it to size the entry and
to find colliding keys, ``render_json`` / ``render_trace`` print it as the
object key, and ``render_python`` shows it.

The encoding is its own and stable: Python notation on one line, as
``render_python`` printed keys when it was introduced (complete values as
literals, cut ones with the slice-key markers, the standard types in their
familiar notation, no registered hook). It deliberately does not call the
renderers, so a change to the display notation cannot change which keys
capture keeps, where it cuts or how large a snapshot is. Golden tests pin it
(``tests/snapshot/test_key_text.py``).
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Callable
from typing import Any, cast

from nooa.snapshot._model import Mapping, Object, Ref, Scalar, Sequence, Snapshot, Text, Unavailable

__all__ = ["key_text"]

_MAX_KEY_CHARS = 50  # a nested string key longer than this is cut


def key_text(node: Snapshot) -> str:
    """The canonical JSON object key of node key ``node`` (see the module).

    Raises ``TypeError`` when ``node`` is not a snapshot node.

    >>> from nooa.snapshot import capture
    >>> key_text(capture((1, "a"))), key_text(capture(frozenset({2})))
    ("(1, 'a')", 'frozenset({2})')
    """
    if type(node) is Object:
        form = _STANDARD.get(node.kind) if node.kind is not None else None
        if form is not None:
            try:
                text = form(node)
            except Exception:
                text = None
            if type(text) is str:
                return text
        parts = [f"{name}={key_text(value)}" for name, value in node.fields]
        if not node.complete:
            parts.append("..." if node.length is None else f"... of {node.length:,} attributes")
        return f"{node.type_name}(" + ", ".join(parts) + ")"
    if type(node) is Scalar:
        return repr(node.value)
    if type(node) is Text:
        return _text(node)
    if type(node) is Sequence:
        return _sequence(node)
    if type(node) is Mapping:
        return _mapping(node)
    if type(node) is Ref:
        return "<cycle>"
    if type(node) is Unavailable:
        if node.reason == "opaque":
            return f"<opaque: {node.type_name}>"
        return f"<unavailable: {node.reason}>"
    raise TypeError(f"not a snapshot node: {type(node).__name__}")


# --- literals ------------------------------------------------------------------------


def _string(s: str) -> str:
    """``s`` as a literal: ``repr``, or in triple quotes when it has a newline or both quote kinds."""
    if "\n" not in s and not ("'" in s and '"' in s):
        return repr(s)
    if "'''" in s and '"""' in s:
        return repr(s)
    quote = '"""' if "'''" in s else "'''"
    return f"{quote}{s}{quote}"


def _text(text: Text) -> str:
    """A literal when complete, else ``str(len=N, [:h]=..., [-t:]=...)`` (``bytes(...)``)."""
    head, tail = text.head, text.tail
    if text.complete:
        whole = head + tail  # type: ignore[operator]
        return repr(whole) if isinstance(whole, bytes) else _string(whole)
    parts = [f"{'bytes' if text.unit == 'bytes' else 'str'}(len={text.length}"]
    if head or not tail:
        parts.append(f"[:{len(head)}]={head!r}")
    if tail:
        parts.append(f"[-{len(tail)}:]={tail!r}")
    return ", ".join(parts) + ")"


def _brackets(type_name: str) -> tuple[str, str]:
    """The brackets of a sequence named ``type_name``: ``()``, ``{}``, ``frozenset({})`` or ``[]``."""
    if type_name == "tuple":
        return "(", ")"
    if type_name == "set":
        return "{", "}"
    if type_name == "frozenset":
        return "frozenset({", "})"
    return "[", "]"


_UNORDERED = ("set", "frozenset")
_LITERAL_SEQUENCES = ("list", "tuple", "set", "frozenset")


def _items(type_name: str, items: tuple[Snapshot, ...]) -> str:
    """``items`` between the brackets of ``type_name`` (a one-item tuple keeps its comma)."""
    left, right = _brackets(type_name)
    inner = ", ".join(key_text(item) for item in items)
    if type_name == "tuple" and len(items) == 1:
        inner += ","
    return f"{left}{inner}{right}"


def _sequence(node: Sequence) -> str:
    """A literal when complete (``T([...])`` for a named one), else a slice-key or ``items=`` marker."""
    left, right = _brackets(node.type_name)
    literal = node.type_name in _LITERAL_SEQUENCES
    if node.complete:
        if not node.head and node.type_name in _UNORDERED:
            return f"{node.type_name}()"
        body = _items(node.type_name, node.head) if node.head else left + right
        return body if literal else f"{node.type_name}({body})"
    if node.reason == "max_depth" and not node.head:
        return f"{left}{node.type_name}: {node.length} items{right}"
    start = f"{node.type_name}(len={node.length}"
    if node.type_name in _UNORDERED:
        inner = ", ".join([*(key_text(item) for item in node.head), "..."])
        return f"{start}, items={left}{inner}{right})"
    parts = [start, f"[:{len(node.head)}]={_items(node.type_name, node.head)}"]
    if node.tail:
        parts.append(f"[-{len(node.tail)}:]={_items(node.type_name, node.tail)}")
    return ", ".join(parts) + ")"


def _key(key: Any) -> str:
    """A key inside a key: a literal, a node key's text, or a string key over 50 characters cut."""
    if any(type(key) is kind for kind in _NODES):
        return key_text(key)
    if type(key) is str and len(key) > _MAX_KEY_CHARS:
        head_n = (_MAX_KEY_CHARS + 1) // 2
        tail_n = _MAX_KEY_CHARS - head_n
        return f"str(len={len(key)}, [:{head_n}]={key[:head_n]!r}, [-{tail_n}:]={key[-tail_n:]!r})"
    return _string(key) if type(key) is str else repr(key)


_NODES = (Scalar, Text, Sequence, Mapping, Object, Unavailable, Ref)


def _mapping(node: Mapping) -> str:
    """A literal when complete (``T({...})`` for a named one), else ``{T: N items}`` or ``T(len=N, items={...})``."""
    entries = [f"{_key(key)}: {key_text(value)}" for key, value in node.head]
    if node.complete:
        if not node.head:
            return "{}" if node.type_name == "dict" else f"{node.type_name}({{}})"
        body = "{" + ", ".join(entries) + "}"
        return body if node.type_name == "dict" else f"{node.type_name}({body})"
    if node.reason == "max_depth" and not node.head:
        return f"{{{node.type_name}: {node.length} items}}"
    if node.tail:
        entries.append("...")
        entries.extend(f"{_key(key)}: {key_text(value)}" for key, value in node.tail)
    return f"{node.type_name}(len={node.length}, items={{" + ", ".join(entries) + "})"


# --- the standard types ----------------------------------------------------------------
# Each form raises (``TypeError``...) when the node is not as capture builds it;
# ``key_text`` then uses the generic ``Type(field=...)`` notation.


def _whole(node: Snapshot | None) -> str | bytes:
    """The content of a complete ``Text`` or a bytes ``Scalar``."""
    if type(node) is Text and node.complete:
        return node.head
    if type(node) is Scalar and type(node.value) is bytes:
        return node.value
    raise TypeError("not a complete text")


def _scalar(node: Snapshot | None) -> Any:
    """The value of a ``Scalar``."""
    if type(node) is not Scalar:
        raise TypeError("not a scalar")
    return node.value


def _repr(node: Object) -> str:
    text = node.field("repr")
    if type(text) is not Text:
        raise TypeError("no repr")
    if text.complete:
        return cast(str, text.head)
    parts = [f"{node.type_name}(repr_len={text.length}", f"[:{len(text.head)}]={text.head!r}"]
    if text.tail:
        parts.append(f"[-{len(text.tail)}:]={text.tail!r}")
    return ", ".join(parts) + ")"


def _iso(parse: Callable[[str], Any]) -> Callable[[Object], str]:
    return lambda node: repr(parse(cast(str, _whole(node.field("iso")))))


def _timedelta(node: Object) -> str:
    days, seconds, micros = (_scalar(node.field(n)) for n in ("days", "seconds", "microseconds"))
    return repr(datetime.timedelta(days=days, seconds=seconds, microseconds=micros))


def _pattern(node: Object) -> str:
    pattern = _whole(node.field("pattern"))
    flags = _scalar(node.field("flags"))
    if type(flags) is not int:
        raise TypeError("flags")
    if isinstance(pattern, str):
        flags &= ~re.UNICODE
    if not flags:
        return f"re.compile({pattern!r})"
    return f"re.compile({pattern!r}, {re.RegexFlag(flags)!r})"


def _range(node: Object) -> str:
    start, stop, step = (_scalar(node.field(name)) for name in ("start", "stop", "step"))
    return repr(range(start, stop, step))


def _int(node: Object) -> str:
    digits = _scalar(node.field("digits"))
    if type(digits) is not int:
        raise TypeError("no digit count")
    sign = ", negative" if node.field("negative") is not None else ""
    return f"int({digits:,} digits{sign})"


def _function(prefix: str) -> Callable[[Object], str]:
    def form(node: Object) -> str:
        signature = node.field("signature")
        tail = "" if signature is None else _whole(signature)
        return f"<{prefix} {_whole(node.field('name'))}{tail}>"

    return form


def _named(prefix: str) -> Callable[[Object], str]:
    return lambda node: f"<{prefix} {_whole(node.field('name'))!r}>"


def _enum(node: Object) -> str:
    value = node.field("value")
    if value is None:
        raise TypeError("no value")
    name_node = node.field("name")
    if type(name_node) is Scalar and name_node.value is None:
        return f"<{node.type_name}: {key_text(value)}>"
    return f"<{node.type_name}.{_whole(name_node)}: {key_text(value)}>"


def _exception(node: Object) -> str:
    args = node.field("args")
    if type(args) is Sequence and args.complete:
        return f"{node.type_name}(" + ", ".join(key_text(item) for item in args.head) + ")"
    if args is None:
        raise TypeError("no args")
    return f"{node.type_name}(args={key_text(args)})"


_STANDARD: dict[str, Callable[[Object], str]] = {
    "python.repr": _repr,
    "python.datetime": _iso(datetime.datetime.fromisoformat),
    "python.date": _iso(datetime.date.fromisoformat),
    "python.time": _iso(datetime.time.fromisoformat),
    "python.timedelta": _timedelta,
    "python.decimal": lambda node: f"Decimal({_whole(node.field('value'))!r})",
    "python.uuid": lambda node: f"UUID({_whole(node.field('hex'))!r})",
    "python.path": lambda node: f"{node.type_name}({_whole(node.field('path'))!r})",
    "python.pattern": _pattern,
    "python.range": _range,
    "python.complex": lambda node: repr(
        complex(_scalar(node.field("real")), _scalar(node.field("imag")))
    ),
    "python.int": _int,
    "python.function": _function("function"),
    "python.method": _function("bound method"),
    "python.builtin": _function("built-in function"),
    "python.module": _named("module"),
    "python.class": _named("class"),
    "python.enum": _enum,
    "python.exception": _exception,
}
"""The standard kinds' key notation, by ``Object.kind``."""
