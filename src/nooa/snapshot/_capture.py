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

``capture.register(T, shape)`` extends it with nearest-MRO dispatch: a
shape (``array_like``, ``ndarray_like``, ``mapping_like``, ``object_like``,
``text_like``) maps out the structure of a type through accessors, and capture
builds the snapshot with the same builders as the built-in types, so the
shared budget and the notation hold. Pydantic models and dataclasses are
internal ``object_like`` shapes over their privileged field readers.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import types
import typing
import weakref
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, NamedTuple, cast

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from nooa.agentdoc._docs import SpecAnnotation
from nooa.agentdoc._visibility import hidden
from nooa.snapshot._codec import (
    OPAQUE_PREFIX,
    OPAQUE_SUFFIX,
    decode_counted,
    dumps,
    fallback_candidates,
    fallback_json,
    json_key,
    opaque_json,
    pointer_child,
    replaced_in,
    safe_end,
    safe_start,
)
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

__all__ = [
    "DEFAULT_LIMITS",
    "Limits",
    "array_like",
    "capture",
    "capture_fields",
    "mapping_like",
    "ndarray_like",
    "object_like",
    "text_like",
]

_FIELD_METADATA_ATTR = "_agentdoc_fields_docs"
_MAX_TYPE_NAME = 200
_TYPE_DICT = type.__dict__["__dict__"]
_TYPE_MRO = type.__dict__["__mro__"]
_TYPE_NAME = type.__dict__["__name__"]
_BASE_MODEL_DICT = BaseModel.__dict__["__dict__"]
_DATACLASS_FIELD_TYPE = cast(Any, dataclasses)._FIELD
_ANNOTATED_ALIAS = type(typing.Annotated[int, "x"])
"""Longer string annotations are not parsed (the textual fail-closed rule applies)."""
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
        # id(owner) -> (owner, checked copy of its namespace or None); see ``_safe_dict``.
        self.namespaces: dict[int, tuple[Any, dict[str, Any] | None]] = {}

    def take_node(self) -> bool:
        """Consume one inspection unit, returning false when exhausted."""
        if self.nodes_left <= 0:
            return False
        self.nodes_left -= 1
        return True


# --- fallbacks -------------------------------------------------------------


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


def _safe_dict(owner: Any, raw: Any, state: _State) -> dict[str, Any] | None:
    """A copy of the native namespace ``raw`` of ``owner`` if every key is an exact ``str``.

    A dict lookup compares a stored key with ``==`` when the hashes match, so
    a fixed-name lookup in a namespace holding a foreign key object (a class
    built with ``type(name, bases, {key: ...})``, a module ``__dict__`` written
    directly) would run that key's ``__eq__``. The keys are checked by type
    first, without hashing or comparing any of them; one with another key
    type gives ``None`` (fail closed). Later lookups read the checked copy,
    which is kept for the rest of this capture (by ``owner`` identity).
    """
    marker = id(owner)
    cached = state.namespaces.get(marker)
    if cached is not None and cached[0] is owner:
        return cached[1]
    # The pairs are taken in one C-level step (no key is hashed or compared)
    # before the loop: a live namespace such as ``sys.modules`` may change
    # while it runs (an import in another thread). ``dict.copy`` is avoided.
    if type(raw) is MappingProxyType:
        items: Iterable[tuple[Any, Any]] = list(raw.items())  # a class namespace: a native dict
    elif type(raw) is dict:
        items = list(dict.items(raw))
    else:
        return None
    safe: dict[str, Any] | None = {}
    for key, value in items:
        if type(key) is not str:
            safe = None
            break
        safe[key] = value
    state.namespaces[marker] = (owner, safe)
    return safe


def _raw_namespace(cls: type[Any], state: _State) -> dict[str, Any] | None:
    """A class's own namespace (no metaclass lookup), checked by ``_safe_dict``."""
    cached = state.namespaces.get(id(cls))
    if cached is not None and cached[0] is cls:
        return cached[1]
    try:
        namespace = _TYPE_DICT.__get__(cls, type(cls))
    except Exception:
        return None
    if type(namespace) is not MappingProxyType:
        return None
    return _safe_dict(cls, namespace, state)


def _module_name(cls: type[Any], state: _State) -> str:
    namespace = _raw_namespace(cls, state)
    if namespace is None:
        return ""
    module = dict.get(namespace, "__module__")
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
        if base is not object and _raw_namespace(base, state) is None:
            return None
        checked.append(base)
    return tuple(checked)


def _index_of(mro: tuple[type[Any], ...], target: type[Any]) -> int | None:
    for index, base in enumerate(mro):
        if base is target:
            return index
    return None


def _dataclass_index(mro: tuple[type[Any], ...], state: _State) -> int | None:
    """Index of the nearest class whose own namespace holds dataclass fields."""
    for index, base in enumerate(mro):
        if base is object:
            continue
        namespace = _raw_namespace(base, state)
        if namespace is not None and type(dict.get(namespace, "__dataclass_fields__")) is dict:
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
        return _cut_chars(length, value, value, length, max_size, limits)
    head_n, tail_n = limits.split(keep_cap)
    tail = value[length - tail_n :] if tail_n else ""
    return _cut_chars(length, value[:head_n], tail, keep_cap, max_size, limits)


def _cut_chars(
    length: int, head_src: str, tail_src: str, kept_cap: int, max_size: int, limits: Limits
) -> _Built:
    """The cut ``Text`` of a string of ``length`` chars, keeping at most ``kept_cap`` of them.

    Shared by exact ``str`` and ``text_like``. ``head_src`` / ``tail_src`` are
    the original's first / last chars that ``limits.split(kept_cap)`` keeps
    (the whole string for both when ``kept_cap >= length``); every smaller cut
    is a part of them, so the original is never read again.
    """

    def parts(kept: int) -> tuple[str, str]:
        if kept >= length:
            return head_src, ""
        head_n, tail_n = limits.split(kept)
        return head_src[:head_n], tail_src[len(tail_src) - tail_n :] if tail_n else ""

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
        return _Built(Text(head, "", length, "chars", True, None), len(dumps(head)))
    node = Text(head, tail, length, "chars", False, "max_length", omitted=length - kept)
    return _Built(node, _string_cost(head, tail))


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
    return _cut_byte_parts(length, value, value, 0, min(length - 1, keep_cap), max_size, limits)


def _cut_byte_parts(
    length: int,
    head_src: bytes,
    tail_src: bytes,
    tail_offset: int,
    kept_cap: int,
    max_size: int,
    limits: Limits,
) -> _Built:
    """The cut bytes ``Text`` of ``length`` bytes, keeping at most ``kept_cap`` (< ``length``).

    Shared by exact ``bytes`` and ``text_like``. With ``(H, T) =
    limits.split(kept_cap)``: ``head_src`` holds the original's first ``H + 1``
    bytes (one more, to see whether ``H`` is a code point boundary) and
    ``tail_src`` the bytes from ``tail_offset = length - T``; both may be the
    whole value (offset 0). Every smaller cut is a part of them.
    """

    def parts(kept: int) -> tuple[bytes, bytes]:
        head_n, tail_n = limits.split(kept)
        head_end = safe_end(head_src, head_n) if head_n else 0
        if tail_n:
            tail_start = safe_start(tail_src, length - tail_n - tail_offset) + tail_offset
        else:
            tail_start = length
        tail_start = max(tail_start, head_end)
        return head_src[:head_end], tail_src[tail_start - tail_offset :]

    def cost(kept: int) -> int:
        head, tail = parts(kept)
        return _string_cost(decode_counted(head)[0], decode_counted(tail)[0])

    low = kept_cap
    if cost(kept_cap) > max_size:
        low, high = 0, kept_cap
        while low < high:
            midpoint = (low + high + 1) // 2
            if cost(midpoint) <= max_size:
                low = midpoint
            else:
                high = midpoint - 1
    head, tail = parts(low)
    if cost(low) > max_size:  # a whole code point did not fit
        head, tail = b"", b""
    node = Text(
        head,
        tail,
        length,
        "bytes",
        False,
        "max_length",
        replaced=replaced_in(head, tail),
        omitted=length - len(head) - len(tail),
    )
    return _Built(node, _string_cost(decode_counted(head)[0], decode_counted(tail)[0]))


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


def _opaque(value: Any, max_size: int, state: _State) -> _Built:
    """An ``Opaque`` node for an unsupported value, sized as its bounded label."""
    cls = type(value)
    name_budget = max(0, max_size - len(OPAQUE_PREFIX) - len(OPAQUE_SUFFIX) - 2)
    name = _bounded_type_name(value, name_budget)
    module = _module_name(cls, state)
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
    split: tuple[int, int] | None = None,
) -> _Built:
    """Kept items of a sequence: a head (and a tail when known), under the shared budget.

    Items may be private ``_Part`` builders (see ``_build_slot``). ``split``:
    the (head, tail) counts to keep instead of the ``max_items`` rule
    (``ndarray_like``'s per-axis selection; it needs ``length`` and ``tail_source``).
    """
    limits = state.limits
    box = _Container(max_size)
    head_n: int | None = None
    tail_n = 0
    if split is not None:
        if length is not None and split[0] + split[1] < length:
            head_n, tail_n = split
    elif limits.max_items is not None and (length is None or length > limits.max_items):
        head_n, tail_n = limits.split(limits.max_items)
        if tail_source is None or length is None:
            head_n, tail_n = limits.max_items, 0

    def add(child: Any, index: int, kept: list[Snapshot]) -> bool:
        if state.nodes_left <= 0:
            box.stop("max_nodes")
            return False
        available = box.room()
        if available < 1:
            box.stop("max_length")
            return False
        built = _build_slot(child, available, depth + 1, pointer_child(path, str(index)), state)
        kept.append(built.node)
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

    def add(key: Any, child: Any, kept: list[tuple[Key, Snapshot]]) -> bool:
        if state.nodes_left <= 0:
            box.stop("max_nodes")
            return False
        if not state.take_node():
            box.stop("max_nodes")
            return False
        key_text = json_key(key, live=True)
        if key_text is None:
            box.stop("opaque")
            return False
        overhead = _key_overhead(key_text, box)
        if overhead is None:
            box.stop("max_length")
            return False
        available = box.room(overhead)
        if available < 1:
            box.stop("max_length")
            return False
        built = _build_slot(child, available, depth + 1, pointer_child(path, key_text), state)
        kept.append((cast(Key, key), built.node))
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
        namespace = _raw_namespace(base, state)
        if namespace is None:
            return None
        if not read_map(dict.get(namespace, _FIELD_METADATA_ATTR), overwrite=False):
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


def _alias_hidden(annotation: Any, state: _State) -> bool | None:
    """Whether a resolved annotation is an ``Annotated`` whose raw metadata hides (None: unreadable).

    Only an exact ``typing`` ``Annotated`` alias (type identity) is inspected,
    through its raw ``__metadata__``; anything else is visible.
    """
    if type(annotation) is not _ANNOTATED_ALIAS:
        return False
    try:
        metadata = object.__getattribute__(annotation, "__metadata__")
    except Exception:
        return None
    return _metadata_hidden(metadata, state)


_UNREAD = object()

_HINTS: dict[int, tuple[weakref.ref[type[Any]], dict[str, tuple[Any, ...]] | None]] = {}
"""Per dataclass, keyed by ``id`` and re-checked by identity: its resolved hint metadata."""


def _resolved_metadata(cls: type[Any]) -> dict[str, tuple[Any, ...]] | None:
    """The outermost ``Annotated`` metadata of each resolved type hint of ``cls`` (None: unreadable).

    String annotations (PEP 563) are read the way Python reads them:
    ``typing.get_type_hints(cls, include_extras=True)``, evaluated once per
    class. The cache is keyed by ``id(cls)`` and re-checked with ``is``, so the
    class is never hashed or compared; it keeps only a weak reference to the
    class and the marker metadata, and ``weakref.finalize`` drops the entry
    with the class. A hint that is not an exact ``Annotated`` alias has no
    metadata (``()``). If evaluation raises, the class's hints are unreadable.
    """
    key = id(cls)
    entry = _HINTS.get(key)
    if entry is not None and entry[0]() is cls:
        return entry[1]
    metadata: dict[str, tuple[Any, ...]] | None = {}
    try:
        hints = typing.get_type_hints(cls, include_extras=True)
        for name, hint in dict.items(hints):
            if type(name) is not str:
                continue
            found: Any = ()
            if type(hint) is _ANNOTATED_ALIAS:
                found = object.__getattribute__(hint, "__metadata__")
                if type(found) is not tuple:
                    raise TypeError("unreadable Annotated metadata")
            metadata[name] = found
    except Exception:
        metadata = None
    try:
        reference = weakref.ref(cls)
        weakref.finalize(cls, _HINTS.pop, key, None)
    except Exception:
        return metadata  # not cacheable: evaluated again next time
    _HINTS[key] = (reference, metadata)
    return metadata


class _Part:
    """A slot builder of a shape: a field, an item or a level built only once its slot is known.

    Private: only the shape builders of this module make parts (users register
    shapes, never parts). ``build(max_size, depth, path)`` builds it in its
    slot; ``min_size`` / ``min_nodes`` are what a later sibling reserves for it.
    ``exhausted(max_size)``: what a structural part (an items sequence, an
    ndarray level) becomes when no node work is left for it; ``None`` for a
    part that captures a value, whose ``_build`` charges itself.
    """

    __slots__ = ("build", "exhausted", "min_nodes", "min_size")

    def __init__(
        self,
        build: Callable[[int, int, str], _Built],
        min_size: int = 2,
        min_nodes: int = 1,
        exhausted: Callable[[int], _Built] | None = None,
    ) -> None:
        self.build = build
        self.min_size = min_size
        self.min_nodes = min_nodes
        self.exhausted = exhausted


def _build_slot(child: Any, max_size: int, depth: int, path: str, state: _State) -> _Built:
    """``child`` in a container slot: built when it is a shape's ``_Part``, else captured.

    A structural part is one node of the shared work; a part that captures a
    value is charged by that capture.
    """
    if type(child) is _Part:
        if child.exhausted is not None and not state.take_node():
            return child.exhausted(max_size)
        return child.build(max_size, depth, path)
    return _build(child, max_size, depth, path, state)


# --- Pydantic models and dataclasses ---------------------------------------------


def _pydantic_fields(mro: tuple[type[Any], ...], state: _State) -> dict[str, FieldInfo] | None:
    """Find an already-built exact Pydantic field dictionary in raw namespaces."""
    for base in mro:
        if not state.take_node():
            return None
        namespace = _raw_namespace(base, state)
        if namespace is None:
            return None
        # Pydantic stored the completed map directly as ``model_fields`` before
        # 2.10, then moved it to ``__pydantic_fields__``. Read raw namespaces so
        # neither layout invokes the metaclass descriptor exposed by newer
        # releases.
        for key in ("__pydantic_fields__", "model_fields"):
            fields = dict.get(namespace, key)
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
    type_name: str,
    module: str,
    selected: Iterable[tuple[str, Any] | None],
    max_size: int,
    depth: int,
    path: str,
    state: _State,
    kind: str | None = None,
    reserves: list[tuple[int, int]] | None = None,
    capped: bool = True,
) -> _Built:
    """Emit the selected (name, value) fields of an object; ``None`` entries fail closed.

    A field value may be a shape's ``_Part``, built in its field's slot (see
    ``_build_slot``). ``reserves[i]``: (chars, nodes) held back while field
    ``i`` is built so that every later field still fits its minimal fallback.

    ``Limits.max_items`` caps the fields kept, like the items of any container:
    the first ones are kept (a field order is not a sequence end) and the
    object is marked ``max_items`` when another field was left out.
    ``capped=False``: the caller already applied it (a shape's structural
    fields, such as ``items`` or ``data``, are never cut by it).
    """
    box = _Container(max_size)
    fields: list[tuple[str, Snapshot]] = []
    max_items = state.limits.max_items if capped else None
    for index, entry in enumerate(selected):
        if entry is None:
            return _closed(type_name, module)
        if max_items is not None and index >= max_items:
            box.stop("max_items")
            break
        name, child = entry
        overhead = _key_overhead(name, box)
        if overhead is None:
            box.stop("max_length")
            break
        held_chars, held_nodes = (
            reserves[index] if reserves is not None and index < len(reserves) else (0, 0)
        )
        room = box.room(overhead)
        if room < 1:
            box.stop("max_length")
            break
        # Never hold back so much that this field loses its own minimal value.
        available = max(min(room, _min_size(child)), room - held_chars)
        held_nodes = min(held_nodes, max(0, state.nodes_left - 1))
        state.nodes_left -= held_nodes
        try:
            built = _build_slot(child, available, depth + 1, pointer_child(path, name), state)
        finally:
            state.nodes_left += held_nodes
        fields.append((name, built.node))
        box.commit(overhead, built.size)
    complete = box.reason is None
    node = Object(type_name, tuple(fields), complete, box.reason, module, kind)
    return _Built(node, box.size)


def _min_size(child: Any) -> int:
    """The size of the minimal type-oriented fallback a captured value can always keep.

    Only identity checks against exact built-in types: the type of an unknown
    value is never compared, hashed or asked for attributes (a metaclass
    ``__eq__`` must not run).
    """
    if type(child) is _Part:
        return child.min_size
    name = _builtin_name(child)
    return len(dumps(fallback_candidates(name)[0])) if name else 2


_STOP = object()


_Entries = tuple[typing.Iterator[tuple[str, Any] | None] | None, list[Reason]]
"""A field reader's result: the (name, value) fields, ``None`` (fail closed), and why it stopped."""


def _pydantic_entries(
    value: BaseModel, mro: tuple[type[Any], ...], index: int, state: _State
) -> _Entries:
    """The privileged field reader of a Pydantic v2 model: declared fields, raw stored values."""
    try:
        obj_dict = _BASE_MODEL_DICT.__get__(value, type(value))
    except Exception:
        obj_dict = None
    storage = _copy_storage(obj_dict, state)
    if storage is None:
        return None, []
    overrides = _visibility_overrides(storage, mro, state)
    fields = _pydantic_fields(mro, state)
    if overrides is None or fields is None:
        return None, []
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

    return selected(), stopped


def _with_stop(built: _Built, stopped: list[Reason]) -> _Built:
    node = built.node
    if stopped and type(node) is Object and node.complete:
        node = dataclasses.replace(node, complete=False, reason=stopped[0])
    return _Built(node, built.size)


def _instance_storage(
    value: Any, mro: tuple[type[Any], ...], state: _State
) -> dict[str, Any] | None | object:
    """Raw ``__dict__`` through the native getset descriptor; ``{}`` for slotted classes."""
    for base in mro:
        namespace = _raw_namespace(base, state) if base is not object else None
        if namespace is None:
            continue
        descriptor = dict.get(namespace, "__dict__")
        if descriptor is None:
            continue
        if type(descriptor) is not types.GetSetDescriptorType:
            return _STOP  # a custom __dict__: refuse
        try:
            return descriptor.__get__(value, type(value))
        except Exception:
            return _STOP
    return {}


def _slot_value(name: str, mro: tuple[type[Any], ...], value: Any, state: _State) -> Any:
    """A slotted field's value through its native member descriptor, else ``_STOP``."""
    for base in mro:
        namespace = _raw_namespace(base, state) if base is not object else None
        if namespace is None:
            continue
        descriptor = dict.get(namespace, name)
        if descriptor is None:
            continue
        if type(descriptor) is not types.MemberDescriptorType:
            return _STOP
        try:
            return descriptor.__get__(value, type(value))
        except Exception:
            return _STOP
    return _STOP


def _dataclass_entries(
    value: Any, mro: tuple[type[Any], ...], index: int, state: _State
) -> _Entries:
    """The privileged field reader of a dataclass: declared fields of ``mro[index]``, raw values."""
    carrier = mro[index]
    raw = _instance_storage(value, mro, state)
    storage = _copy_storage(raw, state) if raw is not _STOP else None
    namespace = _raw_namespace(carrier, state)
    fields = dict.get(namespace, "__dataclass_fields__") if namespace is not None else None
    if storage is None or type(fields) is not dict:
        return None, []
    overrides = _visibility_overrides(storage, mro, state)
    if overrides is None:
        return None, []
    stopped: list[Reason] = []

    def selected() -> typing.Iterator[tuple[str, Any] | None]:
        resolved: Any = _UNREAD  # the class's resolved hints, read at the first string annotation
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
            if type(annotation) is str:
                if resolved is _UNREAD:
                    resolved = _resolved_metadata(carrier)
                metadata = dict.get(resolved, name) if resolved is not None else None
                annotation_hidden = None if metadata is None else _metadata_hidden(metadata, state)
            else:
                annotation_hidden = _alias_hidden(annotation, state)
            if annotation_hidden is None:
                # This field can't be shown safely: omit it, the object is incomplete.
                stopped.append("opaque")
                continue
            if overrides.get(name, annotation_hidden):
                continue
            if name in storage:
                yield name, dict.__getitem__(storage, name)
                continue
            slot = _slot_value(name, mro, value, state)
            if slot is not _STOP:
                yield name, slot

    return selected(), stopped


# --- shapes: the extension API -------------------------------------------------------
#
# A shape maps out the structure of a type (its length, items, fields, text
# slices) through accessors; capture builds the snapshot itself, with the same
# builders and size rules as the built-in types. Accessors are trusted
# extension code: they run only when capture decides it needs their result.


_OBJECT_MIN_NODES = 16
"""Nodes held back for a later field whose value is not known yet (its MRO and a few fields)."""

_NDARRAY_ELEMENT_CHARS = 6
"""Characters ``ndarray_like`` assumes per shown element (a short number and its separator)."""

_STRUCTURE_FIELDS = {
    "array": ("items",),
    "mapping": ("items",),
    "ndarray": ("shape", "labels", "data"),
}


def _metadata(value: Any, what: str) -> str:
    """``value`` as an exact built-in ``str`` (a ``str`` subclass is copied), else ``TypeError``.

    Names and kinds a shape declares are kept in the snapshot, which must hold
    no live object: a ``str`` subclass could carry state and run its own
    ``__str__`` / ``__eq__`` when a renderer reads it.
    """
    value_type = type(value)
    if value_type is str:
        return value
    try:
        mro = _TYPE_MRO.__get__(value_type, type(value_type))
    except Exception:
        mro = ()
    if type(mro) is tuple and any(base is str for base in mro):
        return str.__str__(value)  # an exact copy; no method of the subclass runs
    raise TypeError(f"{what} must be a str, got {_bounded_type_name(value, _MAX_TYPE_NAME)}")


def _accessor(function: Any, what: str) -> Callable[..., Any]:
    if not callable(function):
        raise TypeError(f"{what} must be callable")
    return function


def _kind(kind: Any) -> str | None:
    return None if kind is None else _metadata(kind, "kind")


def _accessor_fields(fields: Any, shape: str) -> tuple[tuple[str, Callable[[Any], Any]], ...]:
    """A shape's ``fields``: an ordered ``dict`` of ``str`` name -> accessor, copied."""
    if fields is None:
        return ()
    if not isinstance(fields, dict):
        raise TypeError("fields must be a dict of name -> accessor")
    reserved = _STRUCTURE_FIELDS.get(shape, ())
    copied: list[tuple[str, Callable[[Any], Any]]] = []
    for name, function in dict.items(cast(dict[Any, Any], fields)):
        name = _metadata(name, "a field name")
        if name in reserved:
            raise TypeError(f"field name {name!r} is taken by the shape itself")
        copied.append((name, _accessor(function, f"field {name!r}")))
    return tuple(copied)


@dataclass(frozen=True)
class _ArrayLike:
    length: Callable[[Any], int]
    items: Callable[[Any], Iterable[Any]]
    fields: tuple[tuple[str, Callable[[Any], Any]], ...]
    kind: str | None


@dataclass(frozen=True)
class _MappingLike:
    length: Callable[[Any], int]
    items: Callable[[Any], Iterable[tuple[Any, Any]]]
    fields: tuple[tuple[str, Callable[[Any], Any]], ...]
    kind: str | None


@dataclass(frozen=True)
class _NdarrayLike:
    shape: Callable[[Any], tuple[int, ...]]
    element: Callable[[Any, tuple[int, ...]], Any]
    fields: tuple[tuple[str, Callable[[Any], Any]], ...]
    labels: tuple[tuple[int, Callable[[Any], Any]], ...]
    kind: str | None


@dataclass(frozen=True)
class _ObjectLike:
    """Ordered named accessors, or (internal) a privileged field ``reader`` (Pydantic, dataclasses)."""

    fields: tuple[tuple[str, Callable[[Any], Any]], ...]
    kind: str | None
    reader: Callable[[Any, tuple[type[Any], ...], int, _State], _Entries] | None = None


@dataclass(frozen=True)
class _TextLike:
    length: Callable[[Any], int]
    slice: Callable[[Any, int, int], str | bytes]
    unit: Literal["chars", "bytes"]
    kind: str | None


_Shape = _ArrayLike | _MappingLike | _NdarrayLike | _ObjectLike | _TextLike
_SHAPE_TYPES: tuple[type[Any], ...] = (
    _ArrayLike,
    _MappingLike,
    _NdarrayLike,
    _ObjectLike,
    _TextLike,
)


def array_like(
    length: Callable[[Any], int],
    items: Callable[[Any], Iterable[Any]],
    *,
    fields: dict[str, Callable[[Any], Any]] | None = None,
    kind: str | None = None,
) -> _ArrayLike:
    """A shape for sequence-like values: ``length(v)`` items, pulled lazily from ``items(v)``.

    ``length(v)`` is called once; ``items(v)`` is iterated only as far as the
    kept items (``max_items``, the budget), plus at most one. ``fields``: name
    -> ``accessor(v)``, each called once, only when its field is kept. Without
    fields the value becomes a ``Sequence``; with fields an ``Object`` whose
    fields are followed by ``items`` (that ``Sequence``).
    """
    return _ArrayLike(
        _accessor(length, "length"),
        _accessor(items, "items"),
        _accessor_fields(fields, "array"),
        _kind(kind),
    )


def mapping_like(
    length: Callable[[Any], int],
    items: Callable[[Any], Iterable[tuple[Any, Any]]],
    *,
    fields: dict[str, Callable[[Any], Any]] | None = None,
    kind: str | None = None,
) -> _MappingLike:
    """A shape for mapping-like values: ``items(v)`` yields ``(key, value)`` pairs lazily.

    As ``array_like``: a ``Mapping``, or with ``fields`` an ``Object`` whose
    fields are followed by ``items`` (that ``Mapping``). Keys must be exact
    ``str``, ``int``, ``float``, ``bool`` or ``None``, as for a ``dict``.
    """
    return _MappingLike(
        _accessor(length, "length"),
        _accessor(items, "items"),
        _accessor_fields(fields, "mapping"),
        _kind(kind),
    )


def ndarray_like(
    shape: Callable[[Any], tuple[int, ...]],
    element: Callable[[Any, tuple[int, ...]], Any],
    *,
    fields: dict[str, Callable[[Any], Any]] | None = None,
    labels: dict[int, Callable[[Any], Any]] | None = None,
    kind: str | None = None,
) -> _NdarrayLike:
    """A shape for n-dimensional arrays: ``shape(v)``, and ``element(v, index)`` per shown element.

    The value becomes an ``Object`` with its ``fields``, then ``shape`` (the
    tuple), ``labels`` (when given) and ``data``: nested ``Sequence`` nodes, one
    level per axis, keeping the first and last entries along every axis like
    numpy's summarised print (the rule: *Shapes* in ``docs/snapshot-model.md``).
    ``element`` is called only for the shown elements. ``labels``: axis ->
    ``accessor(v)`` returning that axis's labels (e.g. DataFrame columns);
    only the shown ones are read (``labels[i]``) and kept.
    """
    label_accessors: list[tuple[int, Callable[[Any], Any]]] = []
    if labels is not None:
        if not isinstance(labels, dict):
            raise TypeError("labels must be a dict of axis -> accessor")
        for axis, function in dict.items(cast(dict[Any, Any], labels)):
            if type(axis) is not int or axis < 0:
                raise TypeError("a labels axis must be a non-negative int")
            label_accessors.append((axis, _accessor(function, f"labels[{axis}]")))
    return _NdarrayLike(
        _accessor(shape, "shape"),
        _accessor(element, "element"),
        _accessor_fields(fields, "ndarray"),
        tuple(label_accessors),
        _kind(kind),
    )


def object_like(fields: dict[str, Callable[[Any], Any]], *, kind: str | None = None) -> _ObjectLike:
    """A shape for record-like values: ordered name -> ``accessor(v)``, an ``Object``.

    Fields are built in order; each leaves room for the minimal value of every
    later field, and an accessor runs only when its field is kept.
    """
    if not isinstance(fields, dict):
        raise TypeError("fields must be a dict of name -> accessor")
    return _ObjectLike(_accessor_fields(fields, "object"), _kind(kind))


def text_like(
    length: Callable[[Any], int],
    slice: Callable[[Any, int, int], str | bytes],
    *,
    unit: Literal["chars", "bytes"] = "chars",
    kind: str | None = None,
) -> _TextLike:
    """A shape for text too large to copy: ``length(v)`` units, read with ``slice(v, start, stop)``.

    ``slice`` returns ``str`` (``unit="chars"``) or ``bytes`` (``"bytes"``). It
    is called only for the head and tail capture keeps, at most twice, never
    for the whole of a text that is cut. The value becomes a ``Text``.
    """
    if unit != "chars" and unit != "bytes":
        raise TypeError('unit must be "chars" or "bytes"')
    return _TextLike(_accessor(length, "length"), _accessor(slice, "slice"), unit, _kind(kind))


_PYDANTIC_SHAPE = _ObjectLike((), None, _pydantic_entries)
_DATACLASS_SHAPE = _ObjectLike((), None, _dataclass_entries)


class _ShapeError(Exception):
    """An accessor failed in a way that leaves no sensible part: the whole value is an error."""


def _call(function: Callable[..., Any], *args: Any) -> Any:
    """Call an accessor; any exception becomes ``_ShapeError``."""
    try:
        return function(*args)
    except Exception as error:
        raise _ShapeError from error


def _shape_length(function: Callable[[Any], Any], value: Any) -> int:
    length = _call(function, value)
    if type(length) is not int or length < 0:
        raise _ShapeError("length must be a non-negative int")
    return length


def _error(type_name: str, max_size: int) -> _Built:
    """An accessor failed: ``Unavailable("error")``, sized as its fallback."""
    return _Built(Unavailable("error", type_name), fallback_json("", max_size)[1])


def _guarded(
    iterator: typing.Iterator[Any], failed: list[bool], pairs: bool
) -> typing.Iterator[Any]:
    """``iterator`` until it ends or raises (``failed``); ``pairs``: each item an exact 2-tuple."""
    while True:
        try:
            item = next(iterator)
        except StopIteration:
            return
        except Exception:
            failed.append(True)
            return
        if pairs and (type(item) is not tuple or tuple.__len__(item) != 2):
            failed.append(True)
            return
        yield item


def _failed(built: _Built, failed: list[bool]) -> _Built:
    """A sequence or mapping whose items raised: incomplete, reason ``error``."""
    node = built.node
    if failed and (type(node) is Sequence or type(node) is Mapping):
        node = dataclasses.replace(cast(Any, node), complete=False, reason="error")
    return _Built(node, built.size)


def _empty(
    node_type: Any, type_name: str, length: int | None, max_size: int, reason: Reason
) -> _Built:
    if max_size < 2:
        return _Built(Unavailable(reason, "list" if node_type is Sequence else "dict"), 1)
    return _Built(node_type(type_name, length, (), False, reason), 2)


def _collection(
    shape: _ArrayLike | _MappingLike,
    value: Any,
    type_name: str,
    length: int,
    max_size: int,
    depth: int,
    path: str,
    state: _State,
) -> _Built:
    """The items of an ``array_like`` / ``mapping_like`` value, with the list / dict builders."""
    try:
        source = iter(shape.items(value))
    except Exception:
        return _error(type_name, max_size)
    failed: list[bool] = []
    if type(shape) is _ArrayLike:
        items = _guarded(itertools.islice(source, length), failed, pairs=False)
        built = _sequence_from(items, length, type_name, max_size, depth, path, state)
    else:
        pairs = _guarded(source, failed, pairs=True)
        head_n = state.limits.max_items
        built = _mapping_from(pairs, length, type_name, max_size, depth, path, state, head_n)
    return _failed(built, failed)


def _field_part(function: Callable[[Any], Any], value: Any, state: _State) -> _Part:
    """A field whose accessor runs only when the field is built (kept), captured normally."""

    def build(max_size: int, depth: int, path: str) -> _Built:
        try:
            child = function(value)
        except Exception:
            return _error("", max_size)
        return _build(child, max_size, depth, path, state)

    return _Part(build, 2, _OBJECT_MIN_NODES)


def _structure_part(
    node_type: Any, length: int | None, build: Callable[[int, int, str], _Built], state: _State
) -> _Part:
    """A ``Sequence`` / ``Mapping`` field of a shape (``items``, ``data``, ``labels``): one node."""

    def built(max_size: int, depth: int, path: str) -> _Built:
        if depth >= state.max_depth or max_size < 2:
            reason: Reason = "max_depth" if depth >= state.max_depth else "max_length"
            return _empty(
                node_type, "list" if node_type is Sequence else "dict", length, max_size, reason
            )
        return build(max_size, depth, path)

    def exhausted(max_size: int) -> _Built:
        name = "list" if node_type is Sequence else "dict"
        return _empty(node_type, name, length, max_size, "max_nodes")

    return _Part(built, 2, 1, exhausted)


def _build_object(
    shape: _ObjectLike | _ArrayLike | _MappingLike | _NdarrayLike,
    value: Any,
    mro: tuple[type[Any], ...],
    index: int,
    max_size: int,
    depth: int,
    path: str,
    state: _State,
) -> _Built:
    """An ``Object`` of a shape: a field reader's fields, or accessor fields and structure parts.

    Pydantic models and dataclasses are ``_ObjectLike`` shapes whose ``reader``
    is their privileged field reader; registered shapes add their accessor
    fields (built only when kept, each leaving room for the later ones) and
    their structure (``items``, ``shape`` / ``labels`` / ``data``), never cut
    by ``max_items``.
    """
    type_name = _bounded_type_name(value, _MAX_TYPE_NAME)
    module = _module_name(type(value), state)
    kind = shape.kind
    if depth >= state.max_depth:
        return _Built(Object(type_name, (), False, "max_depth", module, kind), 2)
    ref = _enter(value, path, "object", state)
    if ref is not None:
        return _Built(ref, 2)
    try:
        if type(shape) is _ObjectLike and shape.reader is not None:
            selected, stopped = shape.reader(value, mro, index, state)
            if selected is None:
                return _closed(type_name, module)
            built = _object_fields(type_name, module, selected, max_size, depth, path, state)
            return _with_stop(built, stopped)
        stopped = []
        entries: list[tuple[str, Any]] = [
            (name, _field_part(function, value, state)) for name, function in shape.fields
        ]
        max_items = state.limits.max_items
        if max_items is not None and len(entries) > max_items:
            del entries[max_items:]
            stopped.append("max_items")
        try:
            entries += _structure(shape, value, type_name, max_size, state)
        except _ShapeError:
            return _error(type_name, max_size)
        reserves, _nodes = _field_reserves(entries)

        def fields() -> typing.Iterator[tuple[str, Any] | None]:
            for entry in entries:
                if not state.take_node():
                    stopped.append("max_nodes")
                    return
                yield entry

        built = _object_fields(
            type_name,
            module,
            fields(),
            max_size,
            depth,
            path,
            state,
            kind,
            reserves=reserves,
            capped=False,
        )
        return _with_stop(built, stopped)
    finally:
        state.active.pop(id(value), None)


def _structure(
    shape: _ObjectLike | _ArrayLike | _MappingLike | _NdarrayLike,
    value: Any,
    type_name: str,
    max_size: int,
    state: _State,
) -> list[tuple[str, Any]]:
    """The structure fields that follow a shape's accessor fields (``_ShapeError``: whole value)."""
    if type(shape) is _ArrayLike or type(shape) is _MappingLike:
        collection = shape
        length = _shape_length(collection.length, value)
        node_type = Sequence if type(collection) is _ArrayLike else Mapping
        name = "list" if node_type is Sequence else "dict"

        def items(slot: int, depth: int, path: str) -> _Built:
            return _collection(collection, value, name, length, slot, depth, path, state)

        return [("items", _structure_part(node_type, length, items, state))]
    if type(shape) is _NdarrayLike:
        return _ndarray_structure(shape, value, max_size, state)
    return []


# --- ndarray_like ------------------------------------------------------------------


def _ndarray_dims(shape: _NdarrayLike, value: Any) -> tuple[int, ...]:
    dims = _call(shape.shape, value)
    if type(dims) is not tuple or any(type(n) is not int or n < 0 for n in dims):
        raise _ShapeError("shape must be a tuple of non-negative ints")
    return dims


def _ndarray_selection(
    dims: tuple[int, ...], labelled: int, max_size: int, state: _State
) -> list[tuple[int, int]]:
    """Per axis, the (head, tail) entries shown: numpy's summary, sized by the limits.

    ``K`` entries per axis at most (all of a shorter axis), split between the
    start and the end (the start takes the odd one). ``K`` is ``max_items``
    when set, and never more than the budget affords: the largest ``K`` whose
    shown elements and labels (``_NDARRAY_ELEMENT_CHARS`` each) and nested
    brackets fit three quarters of the value's characters, and whose elements
    and levels fit three quarters of the node work left. The rest of the
    budget is left for the fields, the shape and the keys; capture's usual
    budget still cuts elements larger than assumed.
    """
    limits = state.limits
    longest = max(dims, default=0)
    cap = longest if limits.max_items is None else min(longest, limits.max_items)
    chars = max_size * 3 // 4
    nodes = state.nodes_left * 3 // 4

    def fits(k: int) -> bool:
        kept = [min(n, k) for n in dims]
        elements = math.prod(kept)
        levels = 0
        rows = 1
        for count in kept:
            levels += rows
            rows *= count
        shown_labels = labelled * k
        cost = (elements + shown_labels) * _NDARRAY_ELEMENT_CHARS + 2 * levels
        return cost <= chars and elements + levels <= nodes

    low, high = 0, cap
    while low < high:
        midpoint = (low + high + 1) // 2
        if fits(midpoint):
            low = midpoint
        else:
            high = midpoint - 1
    selection: list[tuple[int, int]] = []
    for n in dims:
        if n <= low:
            selection.append((n, 0))
        else:
            selection.append(((low + 1) // 2, low // 2))
    return selection


def _indexed(
    length: int,
    split: tuple[int, int],
    child: Callable[[int], Any],
    type_name: str,
    max_size: int,
    depth: int,
    path: str,
    state: _State,
) -> _Built:
    """A ``Sequence`` of the entries ``child(i)`` at the head and tail indices of ``split``."""
    head_n, tail_n = split

    def tail(_count: int) -> list[Any]:
        return [child(i) for i in range(length - tail_n, length)]

    return _sequence_from(
        (child(i) for i in range(head_n)),
        length,
        type_name,
        max_size,
        depth,
        path,
        state,
        tail_source=tail,
        split=split,
    )


def _ndarray_structure(
    shape: _NdarrayLike, value: Any, max_size: int, state: _State
) -> list[tuple[str, Any]]:
    """``shape``, ``labels`` and ``data`` of an ``ndarray_like`` value."""
    dims = _ndarray_dims(shape, value)
    ndim = len(dims)
    selection = _ndarray_selection(dims, len(shape.labels), max_size, state)

    def element(index: tuple[int, ...]) -> _Part:
        def build(slot: int, depth: int, path: str) -> _Built:
            try:
                child = shape.element(value, index)
            except Exception:
                return _error("", slot)
            return _build(child, slot, depth, path, state)

        return _Part(build, 2, _OBJECT_MIN_NODES)

    def level(axis: int, prefix: tuple[int, ...], slot: int, depth: int, path: str) -> _Built:
        def child(i: int) -> _Part:
            index = (*prefix, i)
            if axis + 1 == ndim:
                return element(index)
            return _structure_part(
                Sequence,
                dims[axis + 1],
                lambda s, d, p: level(axis + 1, index, s, d, p),
                state,
            )

        return _indexed(dims[axis], selection[axis], child, "list", slot, depth, path, state)

    entries: list[tuple[str, Any]] = [("shape", dims)]
    if shape.labels:

        def labels(slot: int, depth: int, path: str) -> _Built:
            parts = [(axis, label_part(axis, function)) for axis, function in shape.labels]
            return _mapping_from(parts, len(parts), "dict", slot, depth, path, state, None)

        def label_part(axis: int, function: Callable[[Any], Any]) -> _Part:
            def build(slot: int, depth: int, path: str) -> _Built:
                if axis >= ndim:
                    return _error("", slot)
                try:
                    found = function(value)
                except Exception:
                    return _error("", slot)

                def label(i: int) -> _Part:
                    def read(s: int, d: int, p: str) -> _Built:
                        try:
                            child = found[i]
                        except Exception:
                            return _error("", s)
                        return _build(child, s, d, p, state)

                    return _Part(read, 2, _OBJECT_MIN_NODES)

                split = selection[axis]
                return _indexed(dims[axis], split, label, "list", slot, depth, path, state)

            return _structure_part(Sequence, dims[axis] if axis < ndim else None, build, state)

        entries.append(("labels", _structure_part(Mapping, len(shape.labels), labels, state)))
    if ndim == 0:
        entries.append(("data", element(())))
    else:
        data = _structure_part(Sequence, dims[0], lambda s, d, p: level(0, (), s, d, p), state)
        entries.append(("data", data))
    return entries


# --- text_like ---------------------------------------------------------------------


def _build_text(shape: _TextLike, value: Any, max_size: int, state: _State) -> _Built:
    """A ``Text`` of a ``text_like`` value, reading only the head and tail it keeps.

    Any accessor error (``length``, ``slice``, a slice of the wrong type or
    size) leaves no sensible part: the whole value is ``Unavailable("error")``.
    """
    type_name = _bounded_type_name(value, _MAX_TYPE_NAME)
    try:
        return _text_parts(shape, value, max_size, state.limits)
    except _ShapeError:
        return _error(type_name, max_size)


def _text_parts(shape: _TextLike, value: Any, max_size: int, limits: Limits) -> _Built:
    length = _shape_length(shape.length, value)
    chars = shape.unit == "chars"

    def fetch(start: int, stop: int) -> Any:
        part = _call(shape.slice, value, start, stop)
        if chars:
            if type(part) is not str:
                try:
                    part = _metadata(part, "a chars slice")
                except TypeError as error:
                    raise _ShapeError from error
            size = str.__len__(part)
        else:
            if type(part) is bytearray or type(part) is memoryview:
                part = bytes(part)
            if type(part) is not bytes:
                raise _ShapeError("a bytes slice must be bytes")
            size = bytes.__len__(part)
        if size != stop - start:
            raise _ShapeError("a slice must hold exactly the units asked for")
        return part

    if max_size < 2:
        return _Built(Unavailable("max_length", "str" if chars else "bytes"), 1)
    raw_limit = max_size - 2
    keep_cap = raw_limit if limits.max_string is None else min(raw_limit, limits.max_string)
    if length <= keep_cap:  # small enough to be kept whole: one read
        whole = fetch(0, length)
        if chars:
            encoded_size = len(dumps(whole))
            if encoded_size <= max_size:
                return _Built(Text(whole, "", length, "chars", True, None), encoded_size)
            return _cut_chars(length, whole, whole, length, max_size, limits)
        decoded, replaced = decode_counted(whole)
        encoded_size = len(dumps(decoded))
        if encoded_size <= max_size:
            return _Built(
                Text(whole, b"", length, "bytes", True, None, None, replaced), encoded_size
            )
        return _cut_byte_parts(length, whole, whole, 0, length - 1, max_size, limits)
    head_n, tail_n = limits.split(keep_cap)
    empty: Any = "" if chars else b""
    tail = fetch(length - tail_n, length) if tail_n else empty
    if chars:
        head = fetch(0, head_n) if head_n else empty
        return _cut_chars(length, head, tail, keep_cap, max_size, limits)
    # One byte past the head shows whether the head ends on a code point boundary.
    if not head_n:
        head = empty
    elif head_n + 1 < length:
        head = fetch(0, head_n + 1)
    else:  # no tail, and the head is one byte short of the whole: read that byte apart
        head = fetch(0, head_n) + fetch(head_n, length)
    return _cut_byte_parts(length, head, tail, length - tail_n, keep_cap, max_size, limits)


# --- dispatch ----------------------------------------------------------------------


def _build_shape(
    shape: _Shape,
    value: Any,
    mro: tuple[type[Any], ...],
    index: int,
    max_size: int,
    depth: int,
    path: str,
    state: _State,
) -> _Built:
    """Build ``value`` (whose MRO matched at ``index``) as ``shape`` describes it."""
    shape_type = type(shape)
    if shape_type is _TextLike:
        return _build_text(cast(_TextLike, shape), value, max_size, state)
    if shape_type is _ObjectLike or shape_type is _NdarrayLike or cast(Any, shape).fields:
        return _build_object(cast(Any, shape), value, mro, index, max_size, depth, path, state)
    collection = cast(_ArrayLike | _MappingLike, shape)
    type_name = _bounded_type_name(value, _MAX_TYPE_NAME)
    node_type = Sequence if shape_type is _ArrayLike else Mapping
    try:
        length = _shape_length(collection.length, value)
    except _ShapeError:
        return _error(type_name, max_size)
    if depth >= state.max_depth:
        return _Built(node_type(type_name, length, (), False, "max_depth"), 2)
    ref = _enter(value, path, "sequence" if node_type is Sequence else "mapping", state)
    if ref is not None:
        return _Built(ref, 2)
    try:
        return _collection(collection, value, type_name, length, max_size, depth, path, state)
    finally:
        state.active.pop(id(value), None)


def _field_reserves(pairs: list[tuple[Any, Any]]) -> tuple[list[tuple[int, int]], int]:
    """Per field: the (chars, nodes) every later field needs for its minimal fallback.

    Also the node work all the fields need together.
    """
    needs: list[tuple[int, int]] = []
    for name, child in pairs:
        # separator + "name": + minimal value; one node for the field, some for its value.
        nodes = 1 + (child.min_nodes if type(child) is _Part else 1)
        needs.append((2 + len(dumps(name)) + 2 + _min_size(child), nodes))
    reserves: list[tuple[int, int]] = []
    chars = nodes = 0
    for need_chars, need_nodes in reversed(needs):
        reserves.append((chars, nodes))
        chars += need_chars
        nodes += need_nodes
    reserves.reverse()
    return reserves, nodes


class _Registry:
    """The process-wide ``capture.register(T, shape)`` registrations, matched by class identity.

    A list of ``(cls, shape)`` pairs scanned with ``is``: a class is never
    hashed or compared with ``==``, so a metaclass ``__hash__``/``__eq__``
    cannot run during dispatch.
    """

    def __init__(self) -> None:
        self.entries: list[tuple[type[Any], _Shape]] = []

    def find(self, mro: tuple[type[Any], ...]) -> tuple[int, _Shape] | None:
        """The registration nearest in ``mro`` (identity comparisons only)."""
        if not self.entries:
            return None
        for index, base in enumerate(mro):
            for registered, shape in self.entries:
                if registered is base:
                    return index, shape
        return None


_REGISTRY = _Registry()


# --- the walk --------------------------------------------------------------------


def _with_reason(built: _Built, reason: Reason) -> _Built:
    """A committed fallback node, saying why it was used."""
    node = built.node
    node_type = type(node)
    if (
        node_type is Text
        or node_type is Sequence
        or node_type is Mapping
        or node_type is Object
        or node_type is Unavailable
    ):
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
            name_budget = max(0, max_size - len(OPAQUE_PREFIX) - len(OPAQUE_SUFFIX) - 2)
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
        return _opaque(value, max_size, state)
    found = _REGISTRY.find(mro)
    model_index = _index_of(mro, BaseModel)
    dataclass_index = _dataclass_index(mro, state)
    candidates = [
        (found[0], 0) if found is not None else None,
        (model_index, 1) if model_index is not None else None,
        (dataclass_index, 2) if dataclass_index is not None else None,
    ]
    present = [candidate for candidate in candidates if candidate is not None]
    if not present:
        return _opaque(value, max_size, state)
    _index, kind = min(present)
    if max_size < 2:
        return committed_fallback or _fallback(value, "max_length", max_size)
    if kind == 0:
        assert found is not None
        return _build_shape(found[1], value, mro, found[0], max_size, depth, path, state)
    if kind == 1:
        assert model_index is not None
        return _build_object(_PYDANTIC_SHAPE, value, mro, model_index, max_size, depth, path, state)
    assert dataclass_index is not None
    return _build_object(
        _DATACLASS_SHAPE, value, mro, dataclass_index, max_size, depth, path, state
    )


class _CaptureFunction:
    """``capture(value, limits)``, with ``capture.register(T, shape)`` to extend it."""

    def __call__(self, value: object, limits: Limits = DEFAULT_LIMITS) -> Snapshot:
        """A bounded snapshot of ``value``; no user code runs on unregistered types."""
        _validate_limits(limits)
        state = _State(limits)
        return _build(value, limits.max_chars, 0, "", state).node

    def register(self, cls: type[Any], shape: _Shape) -> None:
        """Capture ``cls`` and its subclasses as ``shape`` describes them.

        ``shape`` is one of ``array_like``, ``ndarray_like``, ``mapping_like``,
        ``object_like`` or ``text_like``; anything else raises ``TypeError``.
        Registering needs the imported type, so supporting a library is opt-in
        by whoever sets up the agent; the registry is process-wide. Classes
        are matched by identity (``is``) along the value's raw MRO, so no user
        code runs during lookup. The registration nearest in a value's MRO
        wins (over the built-in Pydantic and dataclass views too); registering
        ``cls`` again replaces it. Exact built-in types (``str``, ``list``, ...)
        are always native.
        """
        if not isinstance(cls, type):
            raise TypeError(f"capture.register needs a type, got {type(cls).__name__}")
        shape_type = type(shape)
        if not any(shape_type is known for known in _SHAPE_TYPES) or (
            shape_type is _ObjectLike and cast(_ObjectLike, shape).reader is not None
        ):
            raise TypeError(
                "capture.register needs a shape (array_like, ndarray_like, mapping_like,"
                f" object_like or text_like), got {_bounded_type_name(shape, _MAX_TYPE_NAME)}"
            )
        entries = _REGISTRY.entries
        entries[:] = [entry for entry in entries if entry[0] is not cls]
        entries.append((cls, shape))

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
        return {}, Object(name, (), False, "max_length", _module_name(type(value), state))
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
