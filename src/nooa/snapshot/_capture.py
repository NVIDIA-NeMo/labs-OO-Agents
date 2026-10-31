# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``capture()``: a bounded snapshot of a finished Python value, without running user code.

Based on the bounded trace JSON builder of PR #342 (Severin Klingler): exact
built-in types are read natively, Pydantic v2 models and dataclasses through
their declared fields in raw instance storage, and everything else is
``Opaque``. No ``repr()``, ``str()``, truthiness, properties, descriptors,
custom iteration, hashing or equality runs on an inspected value.

One shared budget (``Limits``) bounds the whole capture:

- ``max_chars``: the encoded size of the snapshot as canonical JSON (the
  notation ``render_json`` emits), accounted exactly as in #342, so that
  ``render_json(capture(v, limits), max_chars=limits.max_chars)`` never cuts
  anything further;
- ``max_nodes``: inspection work (values, container entries, MRO entries,
  field and visibility metadata);
- ``max_depth``: nesting;
- ``max_items`` / ``max_string``: items per container and characters per
  string, kept from the start and (``tail_share``) the end.

``capture.register(T)`` extends it with nearest-MRO dispatch; a registered
function builds its children only through the ``Out`` it is given, so the
shared budget holds.
"""

from __future__ import annotations

import dataclasses
import json
import math
import types
import typing
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, NamedTuple, cast

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from nooa.agentdoc._docs import SpecAnnotation
from nooa.agentdoc._visibility import hidden
from nooa.snapshot._model import (
    Key,
    Mapping,
    Object,
    Opaque,
    Reason,
    Ref,
    Scalar,
    Sequence,
    Snapshot,
    Text,
    Unavailable,
)

__all__ = ["DEFAULT_LIMITS", "Limits", "Out", "capture", "capture_fields"]

_JSON_KWARGS: dict[str, Any] = {
    "ensure_ascii": True,
    "allow_nan": False,
    "separators": (", ", ": "),
    "sort_keys": False,
}
_FIELD_METADATA_ATTR = "_agentdoc_fields_docs"
_OPAQUE_PREFIX = "<opaque: "
_OPAQUE_SUFFIX = ">"
_MAX_TYPE_NAME = 200
_TYPE_DICT = type.__dict__["__dict__"]
_TYPE_MRO = type.__dict__["__mro__"]
_TYPE_NAME = type.__dict__["__name__"]
_BASE_MODEL_DICT = BaseModel.__dict__["__dict__"]
_DATACLASS_FIELD_TYPE = cast(Any, dataclasses)._FIELD
_ANNOTATED_ALIAS = type(typing.Annotated[int, "x"])
_BUILTIN_PAIRS: tuple[tuple[type[Any], str], ...] = (
    (str, "str"),
    (bytes, "bytes"),
    (int, "int"),
    (float, "float"),
    (bool, "bool"),
    (type(None), "NoneType"),
    (list, "list"),
    (tuple, "tuple"),
    (dict, "dict"),
)


def dumps(value: Any) -> str:
    """Encode with the canonical options used by accounting and by ``render_json``."""
    return json.dumps(value, **_JSON_KWARGS)


@dataclass(frozen=True)
class Limits:
    """One shared budget for a capture.

    ``max_chars`` counts the snapshot's canonical JSON encoding; ``max_nodes``
    inspection work; ``max_depth`` nesting (the root is depth 0).
    ``max_items`` caps the items kept per container and ``max_string`` the
    characters kept per string (``None``: only the shared budget). When a
    string or container is cut, ``tail_share`` of what is kept comes from its
    end (0: a prefix, as in trace previews; 0.5: agentdoc's head/tail split,
    the head taking the odd one).
    """

    max_chars: int = 50_000
    max_nodes: int = 2_000
    max_depth: int = 16
    max_items: int | None = None
    max_string: int | None = None
    tail_share: float = 0.0

    def split(self, kept: int) -> tuple[int, int]:
        """``kept`` units as (head, tail) under ``tail_share``."""
        tail = int(kept * self.tail_share)
        return kept - tail, tail


DEFAULT_LIMITS = Limits()


def _validate_limits(limits: Limits) -> None:
    """Reject configurations that cannot produce a useful valid root value."""
    if limits.max_chars < 4:
        raise ValueError("snapshot max_chars must be at least 4")
    if limits.max_nodes <= 0:
        raise ValueError("snapshot max_nodes must be greater than 0")
    if limits.max_depth < 0:
        raise ValueError("snapshot max_depth must be non-negative")
    if limits.max_items is not None and limits.max_items < 0:
        raise ValueError("snapshot max_items must be non-negative or None")
    if limits.max_string is not None and limits.max_string < 0:
        raise ValueError("snapshot max_string must be non-negative or None")
    if not 0.0 <= limits.tail_share <= 1.0:
        raise ValueError("snapshot tail_share must be between 0 and 1")


class _Built(NamedTuple):
    node: Snapshot
    size: int


class _State:
    """Mutable bounded traversal state shared by one capture."""

    def __init__(self, limits: Limits) -> None:
        self.limits = limits
        self.nodes_left = limits.max_nodes
        self.max_depth = limits.max_depth
        # id(container) -> (JSON Pointer, kind) of the containers being built.
        self.active: dict[int, tuple[str, Literal["sequence", "mapping", "object"]]] = {}

    def take_node(self) -> bool:
        """Consume one inspection unit, returning false when exhausted."""
        if self.nodes_left <= 0:
            return False
        self.nodes_left -= 1
        return True


def pointer_child(path: str, component: str) -> str:
    """Append one RFC 6901-escaped component to a JSON Pointer path."""
    escaped = component.replace("~", "~0").replace("/", "~1")
    return f"{path}/{escaped}"


# --- fallbacks -------------------------------------------------------------


def fallback_candidates(type_name: str) -> tuple[Any, ...]:
    """#342's type-oriented smallest JSON fallbacks, by built-in type name."""
    if type_name == "str":
        return ("",)
    if type_name in ("list", "tuple"):
        return ([],)
    if type_name == "dict":
        return ({},)
    if type_name == "bool":
        return (False, 0)
    if type_name == "int":
        return (0,)
    if type_name == "float":
        return (0.0, 0)
    if type_name == "NoneType":
        return (None, 0)
    return ("", 0)


def fallback_json(type_name: str, max_size: int) -> tuple[Any, int]:
    """The first fallback for ``type_name`` that fits ``max_size`` (else ``0``)."""
    for candidate in fallback_candidates(type_name):
        encoded = dumps(candidate)
        if len(encoded) <= max_size:
            return candidate, len(encoded)
    # Callers only descend when at least one JSON character is available.
    return 0, 1


def _builtin_name(value: Any) -> str:
    """The name of an exact built-in type, else "" (identity checks: no metaclass hooks)."""
    value_type = type(value)
    for builtin, name in _BUILTIN_PAIRS:
        if value_type is builtin:
            return name
    return ""


def _fallback(value: Any, reason: Reason, max_size: int) -> _Built:
    """A valid, type-oriented fallback node that fits ``max_size``."""
    value_type = type(value)
    _candidate, size = fallback_json(_builtin_name(value), max_size)
    if size == 2 and value_type is str:
        length = str.__len__(value)
        return _Built(Text("", "", length, "chars", False, reason, omitted=length), 2)
    if size == 2 and (value_type is list or value_type is tuple):
        length = len(value)  # exact built-in: no user hook
        return _Built(Sequence(value_type.__name__, length, (), False, reason), 2)
    if size == 2 and value_type is dict:
        return _Built(Mapping("dict", dict.__len__(value), (), False, reason), 2)
    return _Built(Unavailable(reason, _builtin_name(value)), size)


# --- type inspection without hooks ------------------------------------------


def _bounded_type_name(value: Any, max_chars: int) -> str:
    """Read a bounded native type name without invoking instance/metaclass hooks."""
    if max_chars <= 0:
        return ""
    cls = type(value)
    try:
        name = _TYPE_NAME.__get__(cls, type(cls))
    except Exception:
        return "object"[:max_chars]
    if type(name) is not str:
        return "object"[:max_chars]
    return name[:max_chars]


def _raw_namespace(cls: type[Any]) -> MappingProxyType | None:
    """Return a class's native mappingproxy without custom metaclass lookup."""
    try:
        namespace = _TYPE_DICT.__get__(cls, type(cls))
    except Exception:
        return None
    return namespace if type(namespace) is MappingProxyType else None


def _module_name(cls: type[Any]) -> str:
    namespace = _raw_namespace(cls)
    if namespace is None:
        return ""
    module = namespace.get("__module__")
    return module[:_MAX_TYPE_NAME] if type(module) is str else ""


def _safe_mro(value: Any, state: _State) -> tuple[type[Any], ...] | None:
    """The value's MRO, each entry inspection-budgeted and with a raw namespace."""
    cls = type(value)
    try:
        mro = _TYPE_MRO.__get__(cls, type(cls))
    except Exception:
        return None
    if type(mro) is not tuple:
        return None
    checked: list[type[Any]] = []
    for base in mro:
        if not state.take_node():
            return None
        if base is not object and _raw_namespace(base) is None:
            return None
        checked.append(base)
    return tuple(checked)


def _index_of(mro: tuple[type[Any], ...], target: type[Any]) -> int | None:
    for index, base in enumerate(mro):
        if base is target:
            return index
    return None


def _dataclass_index(mro: tuple[type[Any], ...]) -> int | None:
    """Index of the nearest class whose own namespace holds dataclass fields."""
    for index, base in enumerate(mro):
        if base is object:
            continue
        namespace = _raw_namespace(base)
        if namespace is not None and type(namespace.get("__dataclass_fields__")) is dict:
            return index
    return None


# --- scalars and strings -----------------------------------------------------


def _string_cost(head: str, tail: str) -> int:
    """JSON cost of a kept string: the encoded head plus the tail's characters."""
    return len(dumps(head)) + len(tail)


def _cut_string(value: str, max_size: int, max_keep: int | None, limits: Limits) -> _Built:
    """Keep only a fitting head (and tail) of an exact string, copying a bounded part."""
    length = str.__len__(value)
    if max_size < 2:
        return _Built(Unavailable("max_length", "str"), 1)

    raw_limit = max_size - 2
    keep_cap = raw_limit if max_keep is None else min(raw_limit, max_keep)
    if length <= keep_cap:
        encoded_size = len(dumps(value))
        if encoded_size <= max_size:
            return _Built(Text(value, "", length, "chars", True, None), encoded_size)
    kept_cap = min(length, keep_cap)
    reason: Reason = "max_length"

    def parts(kept: int) -> tuple[str, str]:
        if kept >= length:
            return value, ""
        head_n, tail_n = limits.split(kept)
        return value[:head_n], value[length - tail_n :] if tail_n else ""

    head, tail = parts(kept_cap)
    if _string_cost(head, tail) > max_size:
        low, high = 0, kept_cap
        while low < high:
            midpoint = (low + high + 1) // 2
            if _string_cost(*parts(midpoint)) <= max_size:
                low = midpoint
            else:
                high = midpoint - 1
        head, tail = parts(low)
    kept = len(head) + len(tail)
    if kept >= length:
        return _Built(Text(value, "", length, "chars", True, None), len(dumps(value)))
    node = Text(head, tail, length, "chars", False, reason, omitted=length - kept)
    return _Built(node, _string_cost(head, tail))


def safe_end(data: bytes | bytearray, end: int) -> int:
    """``end`` moved back to a UTF-8 code point boundary (forward if that leaves nothing)."""
    end = min(end, len(data))
    cut = end
    while 0 < cut < len(data) and data[cut] & 0xC0 == 0x80:
        cut -= 1
    if cut > 0 or end == len(data) or end == 0:
        return cut
    cut = end  # a code point longer than the kept part: keep it whole
    while cut < len(data) and data[cut] & 0xC0 == 0x80:
        cut += 1
    return cut


def safe_start(data: bytes | bytearray, start: int) -> int:
    """``start`` moved forward past UTF-8 continuation bytes."""
    start = max(0, start)
    while start < len(data) and data[start] & 0xC0 == 0x80:
        start += 1
    return start


_REPLACEMENT = "\ufffd"
_REPLACEMENT_BYTES = _REPLACEMENT.encode("utf-8")


def decode_counted(data: bytes | bytearray) -> tuple[str, int]:
    """``data`` decoded as UTF-8 with replacement, and the replacements introduced."""
    text = bytes(data).decode("utf-8", errors="replace")
    introduced = text.count(_REPLACEMENT) - bytes(data).count(_REPLACEMENT_BYTES)
    return text, max(0, introduced)


def _cut_bytes(value: bytes, max_size: int, max_keep: int | None, limits: Limits) -> _Built:
    """A complete ``Scalar`` for fitting bytes, else a ``Text`` in bytes, cut on code points."""
    length = bytes.__len__(value)
    if max_size < 2:
        return _Built(Unavailable("max_length", "bytes"), 1)
    raw_limit = max_size - 2  # a byte decodes to at most one character
    keep_cap = raw_limit if max_keep is None else min(raw_limit, max_keep)
    if length <= keep_cap:
        decoded, _replaced = decode_counted(value)
        encoded_size = len(dumps(decoded))
        if encoded_size <= max_size:
            return _Built(Scalar(value), encoded_size)

    def parts(kept: int) -> tuple[bytes, bytes]:
        head_n, tail_n = limits.split(kept)
        head_end = safe_end(value, head_n) if head_n else 0
        tail_start = safe_start(value, length - tail_n) if tail_n else length
        tail_start = max(tail_start, head_end)
        return value[:head_end], value[tail_start:]

    def cost(kept: int) -> int:
        head, tail = parts(kept)
        return _string_cost(decode_counted(head)[0], decode_counted(tail)[0])

    kept_cap = min(length - 1, keep_cap)
    low = kept_cap
    if cost(kept_cap) > max_size:
        low, high = 0, kept_cap
        while low < high:
            midpoint = (low + high + 1) // 2
            if cost(midpoint) <= max_size:
                low = midpoint
            else:
                high = midpoint - 1
    head_b, tail_b = parts(low)
    if cost(low) > max_size:  # a whole code point did not fit
        head_b, tail_b = b"", b""
    head, replaced_head = decode_counted(head_b)
    tail, replaced_tail = decode_counted(tail_b)
    node = Text(
        head,
        tail,
        length,
        "bytes",
        False,
        "max_length",
        replaced=replaced_head + replaced_tail,
        omitted=length - len(head_b) - len(tail_b),
    )
    return _Built(node, _string_cost(head, tail))


def _build_integer(value: int, max_size: int) -> _Built:
    """Encode an integer only when decimal conversion is itself bounded."""
    bits = int.bit_length(value)
    # Decimal digits are approximately bits * log10(2). The upper bound keeps
    # conversion below both the output budget and Python's default digit guard.
    estimated_digits = (bits * 30_103) // 100_000 + 1 + (1 if value < 0 else 0)
    if estimated_digits > max_size or estimated_digits > 4_000:
        return _fallback(value, "max_length", max_size)
    try:
        encoded = dumps(value)
    except (ValueError, OverflowError):
        return _fallback(value, "max_length", max_size)
    if len(encoded) > max_size:
        return _fallback(value, "max_length", max_size)
    return _Built(Scalar(value), len(encoded))


def opaque_json(type_name: str, fallback_type: str, max_size: int) -> tuple[Any, int]:
    """#342's ``"<opaque: Type>"`` label fitted to ``max_size``, else a smallest fallback."""
    fallback = fallback_json(fallback_type, max_size)
    if max_size < 2:
        return fallback
    name_budget = max(0, max_size - len(_OPAQUE_PREFIX) - len(_OPAQUE_SUFFIX) - 2)
    label = f"{_OPAQUE_PREFIX}{type_name[:name_budget]}{_OPAQUE_SUFFIX}"
    built = fit_string(label, max_size)
    return built if built[1] <= max_size else fallback


def fit_string(value: str, max_size: int) -> tuple[Any, int]:
    """The longest prefix of ``value`` whose JSON encoding fits ``max_size`` (#342)."""
    if max_size < 2:
        return 0, 1
    raw_limit = max_size - 2
    candidate = value if len(value) <= raw_limit else value[:raw_limit]
    encoded = dumps(candidate)
    if len(encoded) <= max_size:
        return candidate, len(encoded)
    low, high = 0, len(candidate)
    while low < high:
        midpoint = (low + high + 1) // 2
        if len(dumps(candidate[:midpoint])) <= max_size:
            low = midpoint
        else:
            high = midpoint - 1
    prefix = candidate[:low]
    return prefix, len(dumps(prefix))


def _opaque(value: Any, max_size: int) -> _Built:
    """An ``Opaque`` node for an unsupported value, sized as its bounded label."""
    cls = type(value)
    name_budget = max(0, max_size - len(_OPAQUE_PREFIX) - len(_OPAQUE_SUFFIX) - 2)
    name = _bounded_type_name(value, name_budget)
    module = _module_name(cls)
    _label, size = opaque_json(name, _builtin_name(value), max_size)
    return _Built(Opaque(module, name), size)


# --- containers ---------------------------------------------------------------


def _enter(value: Any, path: str, kind: Any, state: _State) -> Ref | None:
    """Mark a container active, or return a ``Ref`` to its active ancestor."""
    marker = id(value)
    found = state.active.get(marker)
    if found is not None:
        return Ref(found[0], found[1])
    state.active[marker] = (path, kind)
    return None


def _key_json(key: Any) -> str | None:
    """The JSON object key a mapping key becomes, or None when it is not supported."""
    key_type = type(key)
    if key_type is str:
        return key
    if key is None:
        return "null"
    if key_type is bool:
        return "true" if key else "false"
    if key_type is int:
        if int.bit_length(key) > 128:
            return None
        return int.__repr__(key)
    if key_type is float:
        return float.__repr__(key) if math.isfinite(key) else None
    return None


class _Container:
    """Accounting for one container being built: its size, kept children and reason."""

    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self.size = 2
        self.count = 0
        self.reason: Reason | None = None

    def stop(self, reason: Reason) -> None:
        if self.reason is None:
            self.reason = reason

    def room(self, overhead: int = 0) -> int:
        separator = 2 if self.count else 0
        return self.max_size - self.size - separator - overhead

    def commit(self, overhead: int, size: int) -> None:
        separator = 2 if self.count else 0
        self.size += separator + overhead + size
        self.count += 1


def _key_overhead(key: str, box: _Container) -> int | None:
    """The encoded size of ``"key": `` if it can fit, without encoding an oversized key."""
    # JSON escaping never makes a key shorter: reject an obviously oversized
    # key before encoding it, then verify the exact size.
    if len(key) + 2 > box.room():
        return None
    return len(dumps(key)) + 2


def _build_sequence(
    value: list[Any] | tuple[Any, ...], max_size: int, depth: int, path: str, state: _State
) -> _Built:
    """Build a bounded sequence node from an exact list or tuple."""
    type_name = "list" if type(value) is list else "tuple"
    if type(value) is list:
        length = list.__len__(cast(list[Any], value))
    else:
        length = tuple.__len__(cast(tuple[Any, ...], value))
    if depth >= state.max_depth:
        return _Built(Sequence(type_name, length, (), False, "max_depth"), 2)
    ref = _enter(value, path, "sequence", state)
    if ref is not None:
        return _Built(ref, 2)
    try:
        get = list.__getitem__ if type(value) is list else tuple.__getitem__
        return _sequence_from(
            (get(cast(Any, value), index) for index in range(length)),
            length,
            type_name,
            max_size,
            depth,
            path,
            state,
            tail_source=lambda count: [
                get(cast(Any, value), index) for index in range(length - count, length)
            ],
        )
    finally:
        state.active.pop(id(value), None)


def _sequence_from(
    items: Iterable[Any],
    length: int | None,
    type_name: str,
    max_size: int,
    depth: int,
    path: str,
    state: _State,
    *,
    tail_source: Callable[[int], list[Any]] | None = None,
) -> _Built:
    """Kept items of a sequence: a head (and a tail when known), under the shared budget."""
    limits = state.limits
    box = _Container(max_size)
    head_n: int | None = None
    tail_n = 0
    if limits.max_items is not None and (length is None or length > limits.max_items):
        head_n, tail_n = limits.split(limits.max_items)
        if tail_source is None or length is None:
            head_n, tail_n = limits.max_items, 0

    def add(child: Any, index: int, out: list[Snapshot]) -> bool:
        if state.nodes_left <= 0:
            box.stop("max_nodes")
            return False
        available = box.room()
        if available < 1:
            box.stop("max_length")
            return False
        built = _build(child, available, depth + 1, pointer_child(path, str(index)), state)
        out.append(built.node)
        box.commit(0, built.size)
        return True

    head: list[Snapshot] = []
    exhausted = True
    iterator = iter(items)
    index = 0
    while True:
        if head_n is not None and index >= head_n:
            exhausted = False
            break
        try:
            child = next(iterator)
        except StopIteration:
            break
        if not add(child, index, head):
            exhausted = False
            break
        index += 1
    if length is None and exhausted:
        length = index
    tail: list[Snapshot] = []
    if box.reason is None and tail_n and length is not None and tail_source is not None:
        for offset, child in enumerate(tail_source(tail_n)):
            if not add(child, length - tail_n + offset, tail):
                break
    kept = len(head) + len(tail)
    complete = length is not None and kept >= length and box.reason is None
    if not complete and box.reason is None:
        box.stop("max_items")
    node = Sequence(
        type_name, length, tuple(head), complete, None if complete else box.reason, tuple(tail)
    )
    return _Built(node, box.size)


def _build_dict(
    value: dict[Any, Any], max_size: int, depth: int, path: str, state: _State
) -> _Built:
    """Build a bounded mapping node without visiting rejected tail entries."""
    length = dict.__len__(value)
    if depth >= state.max_depth:
        return _Built(Mapping("dict", length, (), False, "max_depth"), 2)
    ref = _enter(value, path, "mapping", state)
    if ref is not None:
        return _Built(ref, 2)
    try:
        limits = state.limits
        tail_n = 0
        head_n = length
        if limits.max_items is not None and length > limits.max_items:
            head_n, tail_n = limits.split(limits.max_items)
        tail_items: list[tuple[Any, Any]] = []
        if tail_n:
            backwards = reversed(dict.items(value))
            for _ in range(tail_n):
                tail_items.append(next(backwards))
            tail_items.reverse()
        return _mapping_from(
            dict.items(value), length, "dict", max_size, depth, path, state, head_n, tail_items
        )
    finally:
        state.active.pop(id(value), None)


def _mapping_from(
    items: Iterable[tuple[Any, Any]],
    length: int | None,
    type_name: str,
    max_size: int,
    depth: int,
    path: str,
    state: _State,
    head_n: int | None,
    tail_items: list[tuple[Any, Any]] | None = None,
) -> _Built:
    box = _Container(max_size)

    def add(key: Any, child: Any, out: list[tuple[Key, Snapshot]]) -> bool:
        if state.nodes_left <= 0:
            box.stop("max_nodes")
            return False
        if not state.take_node():
            box.stop("max_nodes")
            return False
        json_key = _key_json(key)
        if json_key is None:
            box.stop("opaque")
            return False
        overhead = _key_overhead(json_key, box)
        if overhead is None:
            box.stop("max_length")
            return False
        available = box.room(overhead)
        if available < 1:
            box.stop("max_length")
            return False
        built = _build(child, available, depth + 1, pointer_child(path, json_key), state)
        out.append((cast(Key, key), built.node))
        box.commit(overhead, built.size)
        return True

    head: list[tuple[Key, Snapshot]] = []
    iterator = iter(items)
    exhausted = True
    count = 0
    while True:
        if length is not None and count >= length:
            break
        if head_n is not None and count >= head_n:
            exhausted = False
            break
        if state.nodes_left <= 0:
            exhausted = False
            box.stop("max_nodes")
            break
        try:
            key, child = next(iterator)
        except StopIteration:
            break
        if not add(key, child, head):
            exhausted = False
            break
        count += 1
    if length is None and exhausted:
        length = count
    tail: list[tuple[Key, Snapshot]] = []
    if box.reason is None and tail_items:
        for key, child in tail_items:
            if not add(key, child, tail):
                break
    kept = len(head) + len(tail)
    complete = length is not None and kept >= length and box.reason is None
    if not complete and box.reason is None:
        box.stop("max_items")
    node = Mapping(
        type_name, length, tuple(head), complete, None if complete else box.reason, tuple(tail)
    )
    return _Built(node, box.size)


# --- visibility (agentdoc ``hidden`` / ``spec(hidden=...)``) ------------------------


def _visibility_overrides(
    obj_dict: dict[str, Any], mro: tuple[type[Any], ...], state: _State
) -> dict[str, bool] | None:
    """Read complete raw hidden overrides before exposing fields (None: fail closed)."""
    class_overrides: dict[str, bool] = {}

    def read_map(raw: Any, *, overwrite: bool) -> bool:
        if raw is None:
            return True
        if type(raw) is not dict:
            return False
        for field_name, metadata in dict.items(raw):
            if not state.take_node():
                return False
            if type(field_name) is not str or type(metadata) is not dict:
                return False
            hidden_value = None
            for metadata_name, metadata_value in dict.items(metadata):
                if not state.take_node():
                    return False
                if type(metadata_name) is not str:
                    return False
                if metadata_name == "hidden":
                    hidden_value = metadata_value
            if hidden_value is not None:
                if type(hidden_value) is not bool:
                    return False
                if overwrite or field_name not in class_overrides:
                    class_overrides[field_name] = hidden_value
        return True

    for base in mro:
        if not state.take_node():
            return None
        namespace = _raw_namespace(base)
        if namespace is None:
            return None
        if not read_map(namespace.get(_FIELD_METADATA_ATTR), overwrite=False):
            return None

    if not read_map(dict.get(obj_dict, _FIELD_METADATA_ATTR), overwrite=True):
        return None
    return class_overrides


def _metadata_hidden(metadata: Any, state: _State) -> bool | None:
    """Whether recognized marker metadata hides a field (None: malformed, fail closed)."""
    if type(metadata) is not list and type(metadata) is not tuple:
        return None
    annotated_hidden = False
    for item in metadata:
        if not state.take_node():
            return None
        if item is hidden:
            annotated_hidden = True
        elif type(item) is SpecAnnotation:
            try:
                kwargs = object.__getattribute__(item, "kwargs")
            except Exception:
                return None
            if type(kwargs) is not dict:
                return None
            hidden_value = None
            for metadata_name, metadata_value in dict.items(kwargs):
                if not state.take_node():
                    return None
                if type(metadata_name) is not str:
                    return None
                if metadata_name == "hidden":
                    hidden_value = metadata_value
            if hidden_value is not None:
                if type(hidden_value) is not bool:
                    return None
                if hidden_value:
                    annotated_hidden = True
        # Foreign constraint metadata is deliberately ignored.
    return annotated_hidden


def _field_hidden(field_info: FieldInfo, state: _State) -> bool | None:
    """Read only recognized resolved visibility metadata of a Pydantic field."""
    try:
        metadata = object.__getattribute__(field_info, "metadata")
    except Exception:
        return None
    if type(metadata) is not list:
        return None
    return _metadata_hidden(metadata, state)


def _annotation_hidden(annotation: Any, state: _State) -> bool | None:
    """Whether a dataclass field annotation carries ``hidden`` (None: fail closed).

    String annotations (PEP 563) are not evaluated: one that mentions
    ``hidden`` or ``spec`` inside ``Annotated`` fails closed.
    """
    if type(annotation) is str:
        if "Annotated" in annotation and ("hidden" in annotation or "spec" in annotation):
            return None
        return False
    if type(annotation) is not _ANNOTATED_ALIAS:
        return False
    try:
        metadata = object.__getattribute__(annotation, "__metadata__")
    except Exception:
        return None
    return _metadata_hidden(metadata, state)


# --- Pydantic models and dataclasses ---------------------------------------------


def _pydantic_fields(mro: tuple[type[Any], ...], state: _State) -> dict[str, FieldInfo] | None:
    """Find an already-built exact Pydantic field dictionary in raw namespaces."""
    for base in mro:
        if not state.take_node():
            return None
        namespace = _raw_namespace(base)
        if namespace is None:
            return None
        # Pydantic stored the completed map directly as ``model_fields`` before
        # 2.10, then moved it to ``__pydantic_fields__``. Read raw namespaces so
        # neither layout invokes the metaclass descriptor exposed by newer
        # releases.
        for key in ("__pydantic_fields__", "model_fields"):
            fields = namespace.get(key)
            if fields is not None:
                return fields if type(fields) is dict else None
    return None


def _closed(type_name: str, module: str) -> _Built:
    """A model whose fields cannot be read safely: no field at all (fail closed)."""
    return _Built(Object(type_name, (), False, "opaque", module), 2)


def _copy_storage(obj_dict: Any, state: _State) -> dict[str, Any] | None:
    """A copy of raw instance storage after checking every key (None: fail closed)."""
    if type(obj_dict) is not dict:
        return None
    # Storage can be mutated directly. Copy only after validating every key so
    # later fixed-string lookups cannot collide with hostile key objects.
    safe: dict[str, Any] = {}
    for stored_name, stored_value in dict.items(obj_dict):
        if not state.take_node() or type(stored_name) is not str:
            return None
        safe[stored_name] = stored_value
    return safe


def _object_fields(
    value: Any,
    type_name: str,
    module: str,
    selected: Iterable[tuple[str, Any] | None],
    max_size: int,
    depth: int,
    path: str,
    state: _State,
) -> _Built:
    """Emit the selected (name, value) fields of an object; ``None`` entries fail closed."""
    box = _Container(max_size)
    fields: list[tuple[str, Snapshot]] = []
    for entry in selected:
        if entry is None:
            return _closed(type_name, module)
        name, child = entry
        overhead = _key_overhead(name, box)
        if overhead is None:
            box.stop("max_length")
            break
        available = box.room(overhead)
        if available < 1:
            box.stop("max_length")
            break
        built = _build(child, available, depth + 1, pointer_child(path, name), state)
        fields.append((name, built.node))
        box.commit(overhead, built.size)
    complete = box.reason is None
    return _Built(Object(type_name, tuple(fields), complete, box.reason, module), box.size)


_STOP = object()


def _build_pydantic(
    value: BaseModel,
    mro: tuple[type[Any], ...],
    max_size: int,
    depth: int,
    path: str,
    state: _State,
) -> _Built:
    """Build a bounded declared/stored-field view of a Pydantic v2 model."""
    type_name = _bounded_type_name(value, _MAX_TYPE_NAME)
    module = _module_name(type(value))
    if depth >= state.max_depth:
        return _Built(Object(type_name, (), False, "max_depth", module), 2)
    ref = _enter(value, path, "object", state)
    if ref is not None:
        return _Built(ref, 2)
    try:
        try:
            obj_dict = _BASE_MODEL_DICT.__get__(value, type(value))
        except Exception:
            obj_dict = None
        storage = _copy_storage(obj_dict, state)
        if storage is None:
            return _closed(type_name, module)
        overrides = _visibility_overrides(storage, mro, state)
        fields = _pydantic_fields(mro, state)
        if overrides is None or fields is None:
            return _closed(type_name, module)
        stopped: list[Reason] = []

        def selected() -> typing.Iterator[tuple[str, Any] | None]:
            for name, field_info in dict.items(fields):
                if not state.take_node():
                    stopped.append("max_nodes")
                    return
                if type(name) is not str or type(field_info) is not FieldInfo:
                    yield None
                    return
                try:
                    excluded = object.__getattribute__(field_info, "exclude")
                    represented = object.__getattribute__(field_info, "repr")
                except Exception:
                    yield None
                    return
                if excluded is not None and type(excluded) is not bool:
                    yield None
                    return
                if type(represented) is not bool:
                    yield None
                    return
                if excluded is True or represented is False:
                    continue
                metadata_hidden = _field_hidden(field_info, state)
                if metadata_hidden is None:
                    yield None
                    return
                if overrides.get(name, metadata_hidden) or name not in storage:
                    continue
                yield name, dict.__getitem__(storage, name)

        built = _object_fields(value, type_name, module, selected(), max_size, depth, path, state)
        return _with_stop(built, stopped)
    finally:
        state.active.pop(id(value), None)


def _with_stop(built: _Built, stopped: list[Reason]) -> _Built:
    node = built.node
    if stopped and type(node) is Object and node.complete:
        node = Object(node.type_name, node.fields, False, stopped[0], node.module)
    return _Built(node, built.size)


def _instance_storage(value: Any, mro: tuple[type[Any], ...]) -> dict[str, Any] | None | object:
    """Raw ``__dict__`` through the native getset descriptor; ``{}`` for slotted classes."""
    for base in mro:
        namespace = _raw_namespace(base) if base is not object else None
        if namespace is None:
            continue
        descriptor = namespace.get("__dict__")
        if descriptor is None:
            continue
        if type(descriptor) is not types.GetSetDescriptorType:
            return _STOP  # a custom __dict__: refuse
        try:
            return descriptor.__get__(value, type(value))
        except Exception:
            return _STOP
    return {}


def _slot_value(name: str, mro: tuple[type[Any], ...], value: Any) -> Any:
    """A slotted field's value through its native member descriptor, else ``_STOP``."""
    for base in mro:
        namespace = _raw_namespace(base) if base is not object else None
        if namespace is None:
            continue
        descriptor = namespace.get(name)
        if descriptor is None:
            continue
        if type(descriptor) is not types.MemberDescriptorType:
            return _STOP
        try:
            return descriptor.__get__(value, type(value))
        except Exception:
            return _STOP
    return _STOP


def _build_dataclass(
    value: Any,
    mro: tuple[type[Any], ...],
    carrier: type[Any],
    max_size: int,
    depth: int,
    path: str,
    state: _State,
) -> _Built:
    """Declared fields of a dataclass (from the class), values from raw instance storage."""
    type_name = _bounded_type_name(value, _MAX_TYPE_NAME)
    module = _module_name(type(value))
    if depth >= state.max_depth:
        return _Built(Object(type_name, (), False, "max_depth", module), 2)
    ref = _enter(value, path, "object", state)
    if ref is not None:
        return _Built(ref, 2)
    try:
        raw = _instance_storage(value, mro)
        storage = _copy_storage(raw, state) if raw is not _STOP else None
        namespace = _raw_namespace(carrier)
        fields = namespace.get("__dataclass_fields__") if namespace is not None else None
        if storage is None or type(fields) is not dict:
            return _closed(type_name, module)
        overrides = _visibility_overrides(storage, mro, state)
        if overrides is None:
            return _closed(type_name, module)
        stopped: list[Reason] = []

        def selected() -> typing.Iterator[tuple[str, Any] | None]:
            for name, field in dict.items(fields):
                if not state.take_node():
                    stopped.append("max_nodes")
                    return
                if type(name) is not str or type(field) is not dataclasses.Field:
                    yield None
                    return
                try:
                    field_type = object.__getattribute__(field, "_field_type")
                    represented = object.__getattribute__(field, "repr")
                    annotation = object.__getattribute__(field, "type")
                except Exception:
                    yield None
                    return
                if field_type is not _DATACLASS_FIELD_TYPE:
                    continue  # ClassVar / InitVar pseudo-fields
                if type(represented) is not bool:
                    yield None
                    return
                if represented is False or name.startswith("_"):
                    continue
                annotation_hidden = _annotation_hidden(annotation, state)
                if annotation_hidden is None:
                    yield None
                    return
                if overrides.get(name, annotation_hidden):
                    continue
                if name in storage:
                    yield name, dict.__getitem__(storage, name)
                    continue
                slot = _slot_value(name, mro, value)
                if slot is not _STOP:
                    yield name, slot

        built = _object_fields(value, type_name, module, selected(), max_size, depth, path, state)
        return _with_stop(built, stopped)
    finally:
        state.active.pop(id(value), None)


# --- registry --------------------------------------------------------------------

CaptureFunction = Callable[[Any, "Out"], Snapshot]


class Out:
    """What a registered capture function builds its snapshot with.

    Every method charges the capture's shared budget and returns a node; the
    function returns one node built by its own ``Out``. ``capture(child)``
    delegates a value; ``sequence``, ``mapping`` and ``object`` pull children
    lazily from an iterable and stop at the budget without consuming the rest.
    """

    def __init__(self, value: Any, max_size: int, depth: int, path: str, state: _State) -> None:
        self._value = value
        self._remaining = max_size
        self._depth = depth
        self._path = path
        self._state = state
        self._built: list[_Built] = []
        self._index = 0

    def _keep(self, built: _Built) -> Snapshot:
        self._remaining = max(0, self._remaining - built.size)
        self._built.append(built)
        return built.node

    def _size_of(self, node: Any) -> int | None:
        for built in self._built:
            if built.node is node:
                return built.size
        return None

    def capture(self, child: Any, key: str | int | None = None) -> Snapshot:
        """Capture ``child`` as this value's snapshot (or a part of it), at the same depth."""
        if key is None:
            key = self._index
            self._index += 1
        path = pointer_child(self._path, str(key))
        if self._remaining < 1:
            return self._keep(_Built(Unavailable("max_length", _builtin_name(child)), 1))
        return self._keep(_build(child, self._remaining, self._depth, path, self._state))

    def sequence(
        self, items: Iterable[Any], *, length: int | None = None, type_name: str = "list"
    ) -> Sequence:
        """A sequence of ``items`` (pulled one at a time), cut like a list."""
        state = self._state
        if self._depth >= state.max_depth or self._remaining < 2:
            reason: Reason = "max_depth" if self._depth >= state.max_depth else "max_length"
            return cast(
                Sequence, self._keep(_Built(Sequence(type_name, length, (), False, reason), 2))
            )
        built = _sequence_from(
            items, length, type_name, self._remaining, self._depth, self._path, state
        )
        return cast(Sequence, self._keep(built))

    def mapping(
        self,
        items: Iterable[tuple[Any, Any]],
        *,
        length: int | None = None,
        type_name: str = "dict",
    ) -> Mapping:
        """A mapping of ``(key, value)`` pairs; keys must be exact built-in scalars."""
        state = self._state
        if self._depth >= state.max_depth or self._remaining < 2:
            reason: Reason = "max_depth" if self._depth >= state.max_depth else "max_length"
            return cast(
                Mapping, self._keep(_Built(Mapping(type_name, length, (), False, reason), 2))
            )
        head_n = state.limits.max_items
        built = _mapping_from(
            items, length, type_name, self._remaining, self._depth, self._path, state, head_n
        )
        return cast(Mapping, self._keep(built))

    def object(self, fields: Iterable[tuple[str, Any]], *, type_name: str | None = None) -> Object:
        """An object with these ``(name, value)`` fields (names must be ``str``)."""
        state = self._state
        name = (
            type_name if type_name is not None else _bounded_type_name(self._value, _MAX_TYPE_NAME)
        )
        module = _module_name(type(self._value))
        if self._depth >= state.max_depth or self._remaining < 2:
            reason: Reason = "max_depth" if self._depth >= state.max_depth else "max_length"
            return cast(Object, self._keep(_Built(Object(name, (), False, reason, module), 2)))
        stopped: list[Reason] = []

        def selected() -> typing.Iterator[tuple[str, Any] | None]:
            for field_name, child in fields:
                if not state.take_node():
                    stopped.append("max_nodes")
                    return
                if type(field_name) is not str:
                    yield None
                    return
                yield field_name, child

        built = _object_fields(
            self._value, name, module, selected(), self._remaining, self._depth, self._path, state
        )
        return cast(Object, self._keep(_with_stop(built, stopped)))

    def opaque(self) -> Opaque:
        """This value as ``Opaque`` (what an unregistered type becomes)."""
        return cast(Opaque, self._keep(_opaque(self._value, max(self._remaining, 1))))


class _Registry:
    def __init__(self) -> None:
        self.entries: list[tuple[type[Any], CaptureFunction]] = []

    def find(self, mro: tuple[type[Any], ...]) -> tuple[int, CaptureFunction] | None:
        """The registration nearest in ``mro`` (identity comparisons only)."""
        for index, base in enumerate(mro):
            for registered, function in self.entries:
                if registered is base:
                    return index, function
        return None


_REGISTRY = _Registry()


def _build_registered(
    function: CaptureFunction, value: Any, max_size: int, depth: int, path: str, state: _State
) -> _Built:
    ref = _enter(value, path, "object", state)
    if ref is not None:
        return _Built(ref, 2)
    out = Out(value, max_size, depth, path, state)
    node: Any = None
    size: int | None = None
    try:
        node = function(value, out)
        size = out._size_of(node)
    except Exception:
        size = None
    finally:
        state.active.pop(id(value), None)
    if node is None or size is None or size > max_size:
        failed = Unavailable("error", _bounded_type_name(value, _MAX_TYPE_NAME))
        return _Built(failed, fallback_json("", max_size)[1])
    return _Built(node, size)


# --- the walk --------------------------------------------------------------------


def _with_reason(built: _Built, reason: Reason) -> _Built:
    """A committed fallback node, saying why it was used."""
    node = built.node
    if type(node) in (Text, Sequence, Mapping, Object, Unavailable):
        node = dataclasses.replace(cast(Any, node), reason=reason)
    return _Built(node, built.size)


def _build(
    value: Any,
    max_size: int,
    depth: int,
    path: str,
    state: _State,
    *,
    committed_fallback: _Built | None = None,
) -> _Built:
    """Recursively build one bounded snapshot node (sized as its JSON encoding)."""
    if not state.take_node():
        if committed_fallback is not None:
            return _with_reason(committed_fallback, "max_nodes")
        return _fallback(value, "max_nodes", max_size)

    limits = state.limits
    value_type = type(value)
    if value_type is str:
        return _cut_string(value, max_size, limits.max_string, limits)
    if value is None or value_type is bool:
        encoded = "null" if value is None else ("true" if value else "false")
        if len(encoded) <= max_size:
            return _Built(Scalar(value), len(encoded))
        return committed_fallback or _fallback(value, "max_length", max_size)
    if value_type is int:
        return _build_integer(value, max_size)
    if value_type is float:
        if not math.isfinite(value):
            name_budget = max(0, max_size - len(_OPAQUE_PREFIX) - len(_OPAQUE_SUFFIX) - 2)
            name = "float"[:name_budget]
            return _Built(Opaque("builtins", name), opaque_json(name, "float", max_size)[1])
        encoded = dumps(value)
        if len(encoded) <= max_size:
            return _Built(Scalar(value), len(encoded))
        return committed_fallback or _fallback(value, "max_length", max_size)
    if value_type is bytes:
        return _cut_bytes(value, max_size, limits.max_string, limits)
    if value_type is list or value_type is tuple:
        if max_size < 2:
            return committed_fallback or _fallback(value, "max_length", max_size)
        return _build_sequence(value, max_size, depth, path, state)
    if value_type is dict:
        if max_size < 2:
            return committed_fallback or _fallback(value, "max_length", max_size)
        return _build_dict(value, max_size, depth, path, state)

    mro = _safe_mro(value, state)
    if mro is None:
        return _opaque(value, max_size)
    found = _REGISTRY.find(mro)
    model_index = _index_of(mro, BaseModel)
    dataclass_index = _dataclass_index(mro)
    candidates = [
        (found[0], 0) if found is not None else None,
        (model_index, 1) if model_index is not None else None,
        (dataclass_index, 2) if dataclass_index is not None else None,
    ]
    present = [candidate for candidate in candidates if candidate is not None]
    if not present:
        return _opaque(value, max_size)
    _index, kind = min(present)
    if max_size < 2:
        return committed_fallback or _fallback(value, "max_length", max_size)
    if kind == 0:
        assert found is not None
        return _build_registered(found[1], value, max_size, depth, path, state)
    if kind == 1:
        return _build_pydantic(value, mro, max_size, depth, path, state)
    assert dataclass_index is not None
    return _build_dataclass(value, mro, mro[dataclass_index], max_size, depth, path, state)


class _CaptureFunction:
    """``capture(value, limits)``, with ``capture.register(T)`` to extend it."""

    def __call__(self, value: object, limits: Limits = DEFAULT_LIMITS) -> Snapshot:
        """A bounded snapshot of ``value``; no user code runs on unregistered types."""
        _validate_limits(limits)
        state = _State(limits)
        return _build(value, limits.max_chars, 0, "", state).node

    def register(self, cls: type[Any], function: CaptureFunction | None = None) -> Any:
        """Register ``function(value, out) -> Snapshot`` for ``cls`` and its subclasses.

        Use as ``@capture.register(T)`` or ``capture.register(T, function)``.
        The registration nearest in a value's MRO wins (over the built-in
        Pydantic and dataclass views too); registering ``cls`` again replaces
        it. Exact built-in types (``str``, ``list``, ...) are always native.
        An exception in ``function``, or a node not built by its ``out``,
        becomes ``Unavailable("error")``.
        """
        if not isinstance(cls, type):
            raise TypeError(f"capture.register needs a type, got {type(cls).__name__}")

        def add(function: CaptureFunction) -> CaptureFunction:
            entries = _REGISTRY.entries
            entries[:] = [entry for entry in entries if entry[0] is not cls]
            entries.append((cls, function))
            return function

        return add(function) if function is not None else add

    def unregister(self, cls: type[Any]) -> None:
        """Remove the registration for exactly ``cls`` (no-op when absent)."""
        entries = _REGISTRY.entries
        entries[:] = [entry for entry in entries if entry[0] is not cls]


capture = _CaptureFunction()


def _field_placeholder(value: Any, state: _State) -> tuple[Any, Snapshot]:
    """Classify one framework wrapper field into a minimal JSON root shape and node."""
    if not state.take_node():
        raise ValueError("snapshot max_nodes cannot classify wrapper fields")
    value_type = type(value)
    if value_type is str:
        length = str.__len__(value)
        return "", Text("", "", length, "chars", False, "max_length", omitted=length)
    if value_type is list or value_type is tuple:
        return [], Sequence(value_type.__name__, len(value), (), False, "max_length")
    if value_type is dict:
        return {}, Mapping("dict", dict.__len__(value), (), False, "max_length")
    if value is None:
        return None, Unavailable("max_length", "NoneType")
    if value_type is bool:
        return False, Unavailable("max_length", "bool")
    if value_type is int:
        return 0, Unavailable("max_length", "int")
    if value_type is float:
        return 0.0, Unavailable("max_length", "float")
    mro = _safe_mro(value, state)
    if mro is not None and _index_of(mro, BaseModel) is not None:
        name = _bounded_type_name(value, _MAX_TYPE_NAME)
        return {}, Object(name, (), False, "max_length", _module_name(type(value)))
    return None, Unavailable("max_length", "NoneType")


def capture_fields(*, limits: Limits = DEFAULT_LIMITS, **fields: object) -> Mapping:
    """Capture a framework-owned wrapper so that every root field survives, with its type.

    Each field first reserves a minimal placeholder of its own type (``[]``,
    ``{}``, ``""``, ...); raises ``ValueError`` when ``limits`` cannot hold the
    complete minimal wrapper (#342's ``trace_fields``).
    """
    _validate_limits(limits)
    state = _State(limits)
    placeholders = {name: _field_placeholder(value, state) for name, value in fields.items()}
    skeleton = dumps({name: shape for name, (shape, _node) in placeholders.items()})
    if len(skeleton) > limits.max_chars:
        raise ValueError("snapshot max_chars cannot hold required wrapper fields")

    items: list[tuple[Key, Snapshot]] = []
    total_size = len(skeleton)
    for name, value in fields.items():
        shape, fallback_node = placeholders[name]
        placeholder = _Built(fallback_node, len(dumps(shape)))
        available = placeholder.size + (limits.max_chars - total_size)
        built = _build(
            value, available, 1, pointer_child("", name), state, committed_fallback=placeholder
        )
        items.append((name, built.node))
        total_size += built.size - placeholder.size
    return Mapping("dict", len(items), tuple(items), True, None, wrapper=True)
