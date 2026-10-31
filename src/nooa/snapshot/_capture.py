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

import ast
import builtins
import dataclasses
import math
import sys
import types
import typing
import weakref
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, NamedTuple, cast

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from nooa.agentdoc._docs import SpecAnnotation, spec
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

__all__ = ["DEFAULT_LIMITS", "Limits", "Out", "capture", "capture_fields"]

_FIELD_METADATA_ATTR = "_agentdoc_fields_docs"
_MAX_TYPE_NAME = 200
_TYPE_DICT = type.__dict__["__dict__"]
_TYPE_MRO = type.__dict__["__mro__"]
_TYPE_NAME = type.__dict__["__name__"]
_BASE_MODEL_DICT = BaseModel.__dict__["__dict__"]
_DATACLASS_FIELD_TYPE = cast(Any, dataclasses)._FIELD
_ANNOTATED_ALIAS = type(typing.Annotated[int, "x"])
_MODULE_DICT = cast(Any, types.ModuleType).__dict__["__dict__"]
_MAX_ANNOTATION_CHARS = 2000
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
        head_units=len(head_b),
        tail_units=len(tail_b),
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
    out: Out | None = None,
) -> _Built:
    """Kept items of a sequence: a head (and a tail when known), under the shared budget.

    ``out``: the ``Out`` whose parts may appear as items (see ``_build_slot``).
    """
    limits = state.limits
    box = _Container(max_size)
    head_n: int | None = None
    tail_n = 0
    if limits.max_items is not None and (length is None or length > limits.max_items):
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
        built = _build_slot(
            child, available, depth + 1, pointer_child(path, str(index)), state, out
        )
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
    out: Out | None = None,
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
        built = _build_slot(child, available, depth + 1, pointer_child(path, key_text), state, out)
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


def _module_namespace(name: Any, state: _State) -> dict[str, Any] | None:
    """A loaded module's raw ``__dict__`` (no import, no attribute hooks; None: unknown)."""
    if type(name) is not str:
        return None
    modules = _safe_dict(sys.modules, sys.modules, state)
    module = dict.get(modules, name) if modules is not None else None
    return None if module is None else _module_namespace_of(module, state)


_UNRESOLVED = object()


def _resolve_name(expr: ast.expr, namespace: dict[str, Any], state: _State) -> Any:
    """What a ``Name`` or dotted ``Attribute`` names in ``namespace`` (raw reads only)."""
    if not state.take_node():
        return _UNRESOLVED
    if type(expr) is ast.Name:
        found = dict.get(namespace, expr.id, _UNRESOLVED)
        if found is _UNRESOLVED:
            builtin_names = _safe_dict(_BUILTINS, _BUILTINS, state)
            if builtin_names is not None:
                found = dict.get(builtin_names, expr.id, _UNRESOLVED)
        return found
    if type(expr) is ast.Attribute:
        base = _resolve_name(expr.value, namespace, state)
        if base is _UNRESOLVED:
            return _UNRESOLVED
        inner = _module_namespace_of(base, state)
        if inner is None:
            return _UNRESOLVED  # only module attributes are read
        return dict.get(inner, expr.attr, _UNRESOLVED)
    return _UNRESOLVED


def _module_namespace_of(value: Any, state: _State) -> dict[str, Any] | None:
    """A module's raw ``__dict__``, checked by ``_safe_dict`` (None: not a module)."""
    try:
        namespace = _MODULE_DICT.__get__(value)
    except Exception:
        return None
    return _safe_dict(namespace, namespace, state) if type(namespace) is dict else None


_BUILTINS: dict[str, Any] = builtins.__dict__


def _is_annotated(value: Any, state: _State) -> bool:
    if value is typing.Annotated:
        return True
    extensions = _module_namespace("typing_extensions", state)
    return extensions is not None and value is dict.get(extensions, "Annotated", _UNRESOLVED)


def _marker_expr_hidden(expr: ast.expr, namespace: dict[str, Any], state: _State) -> bool | None:
    """Whether one ``Annotated`` metadata expression hides the field (None: can't tell)."""
    if type(expr) is ast.Constant:
        return False
    if type(expr) is ast.Name or type(expr) is ast.Attribute:
        found = _resolve_name(expr, namespace, state)
        if found is _UNRESOLVED:
            return None
        return found is hidden
    if type(expr) is ast.Call:
        function = _resolve_name(expr.func, namespace, state)
        if function is _UNRESOLVED:
            return None
        if function is not spec and function is not SpecAnnotation:
            return False  # foreign constraint metadata (pydantic Field, ...)
        if expr.args:
            return None
        hidden_value: Any = None
        for keyword in expr.keywords:
            if not state.take_node():
                return None
            if keyword.arg is None:
                return None  # ``**kwargs``: unreadable
            if keyword.arg == "hidden":
                if type(keyword.value) is not ast.Constant:
                    return None
                hidden_value = keyword.value.value
        if hidden_value is None:
            return False
        if type(hidden_value) is not bool:
            return None
        return hidden_value
    return None  # any other expression could evaluate to a marker


def _parsed_annotation_hidden(
    expr: ast.expr, namespace: dict[str, Any], state: _State
) -> bool | None:
    """Mirror of the runtime rule: only an outermost ``Annotated`` (flattened) hides."""
    if not state.take_node():
        return None
    if type(expr) is not ast.Subscript:
        return False
    head = _resolve_name(expr.value, namespace, state)
    annotated: bool | None = None if head is _UNRESOLVED else _is_annotated(head, state)
    if annotated is False:
        return False
    elements = expr.slice.elts if type(expr.slice) is ast.Tuple else [expr.slice]
    if len(elements) < 2:
        return False
    found = False
    unknown = False
    if type(elements[0]) is ast.Subscript:
        inner = _parsed_annotation_hidden(elements[0], namespace, state)
        unknown = inner is None
        found = inner is True
    for element in elements[1:]:
        marker = _marker_expr_hidden(element, namespace, state)
        if marker is None:
            unknown = True
        elif marker:
            found = True
    if annotated:
        return None if unknown else found
    # An unresolved head (a local alias?) with marker-like metadata: can't tell.
    return None if found or unknown else False


def _textual_annotation_hidden(annotation: str) -> bool | None:
    if "Annotated" in annotation and ("hidden" in annotation or "spec" in annotation):
        return None
    return False


def _annotation_hidden(
    annotation: Any, state: _State, namespace: dict[str, Any] | None = None
) -> bool | None:
    """Whether a dataclass field annotation carries ``hidden`` (None: fail closed).

    String annotations (PEP 563) are never evaluated: they are parsed with
    ``ast`` and their names (``Name`` or dotted module ``Attribute``) are looked
    up in ``namespace``, the raw ``__dict__`` of the module that declared the
    field, then in builtins. Resolved objects are compared by identity with
    ``Annotated`` and the agentdoc ``hidden`` / ``spec`` markers. Inside an
    ``Annotated`` subscript, metadata that can't be read safely (an unknown
    name, a ``spec(...)`` call without a literal ``hidden``) fails closed;
    unknown names elsewhere (``list[Foo]``, forward references) don't matter.
    Without a namespace (the declaring module is unloaded, renamed or
    unreadable) the string is still parsed, its names resolved against
    builtins only: an unresolved head with name or call metadata (an alias
    such as ``A[str, H]``) is unknown and fails closed, while ``list[int]`` or
    ``dict[str, Foo]`` stay visible. For a string that is too long or doesn't
    parse, one that mentions ``hidden`` or ``spec`` inside ``Annotated``
    fails closed.
    """
    if type(annotation) is str:
        if len(annotation) > _MAX_ANNOTATION_CHARS:
            return _textual_annotation_hidden(annotation)
        if namespace is None:
            namespace = {}
        try:
            tree = ast.parse(annotation, mode="eval")
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            return _textual_annotation_hidden(annotation)
        try:
            return _parsed_annotation_hidden(tree.body, namespace, state)
        except RecursionError:
            return None
    if type(annotation) is not _ANNOTATED_ALIAS:
        return False
    try:
        metadata = object.__getattribute__(annotation, "__metadata__")
    except Exception:
        return None
    return _metadata_hidden(metadata, state)


def _declaring_module(
    name: str, field: Any, mro: tuple[type[Any], ...], state: _State
) -> dict[str, Any] | None:
    """The raw namespace of the module whose dataclass declared ``field``.

    Inherited fields keep their ``Field`` object, so the declaring class is
    the most basic class whose own ``__dataclass_fields__`` holds it.
    """
    for base in reversed(mro):
        if not state.take_node():
            return None
        namespace = _raw_namespace(base, state)
        if namespace is None:
            continue
        fields = dict.get(namespace, "__dataclass_fields__")
        if type(fields) is not dict:
            continue
        # ``fields`` is class data too: check its keys before a lookup by name.
        checked = _safe_dict(fields, fields, state)
        if checked is not None and dict.get(checked, name) is field:
            qualified = _raw_qualified_name(base, state)
            return _module_namespace(qualified[0], state) if qualified is not None else None
    return None


def _part_of(child: Any, out: Out | None) -> _Deferred | None:
    """``child`` when it is a part of ``out`` (identity checks only), else ``None``."""
    if out is not None and type(child) is _Deferred and child._out is out:
        return child
    return None


def _build_slot(
    child: Any, max_size: int, depth: int, path: str, state: _State, out: Out | None
) -> _Built:
    """``child`` in a container slot: built when it is a part of ``out``, else captured."""
    part = _part_of(child, out)
    if part is not None:
        return part.build(max_size, depth, path)
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
    value: Any,
    type_name: str,
    module: str,
    selected: Iterable[tuple[str, Any] | None],
    max_size: int,
    depth: int,
    path: str,
    state: _State,
    kind: str | None = None,
    out: Out | None = None,
    reserves: list[tuple[int, int]] | None = None,
) -> _Built:
    """Emit the selected (name, value) fields of an object; ``None`` entries fail closed.

    ``out``: the ``Out`` whose parts may appear as field values; each is
    built in its field's slot (see ``_build_slot``).

    ``reserves[i]``: (chars, nodes) held back while field ``i`` is built so
    that every later field still fits its minimal fallback.

    ``Limits.max_items`` caps the fields kept, like the items of any container:
    the first ones are kept (a field order is not a sequence end) and the
    object is marked ``max_items`` when another field was left out.
    """
    box = _Container(max_size)
    fields: list[tuple[str, Snapshot]] = []
    max_items = state.limits.max_items
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
            built = _build_slot(child, available, depth + 1, pointer_child(path, name), state, out)
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
    if type(child) is _Deferred:
        return child.min_size
    name = _builtin_name(child)
    return len(dumps(fallback_candidates(name)[0])) if name else 2


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
    module = _module_name(type(value), state)
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
    module = _module_name(type(value), state)
    if depth >= state.max_depth:
        return _Built(Object(type_name, (), False, "max_depth", module), 2)
    ref = _enter(value, path, "object", state)
    if ref is not None:
        return _Built(ref, 2)
    try:
        raw = _instance_storage(value, mro, state)
        storage = _copy_storage(raw, state) if raw is not _STOP else None
        namespace = _raw_namespace(carrier, state)
        fields = dict.get(namespace, "__dataclass_fields__") if namespace is not None else None
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
                namespace = (
                    _declaring_module(name, field, mro, state) if type(annotation) is str else None
                )
                annotation_hidden = _annotation_hidden(annotation, state, namespace)
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

        built = _object_fields(value, type_name, module, selected(), max_size, depth, path, state)
        return _with_stop(built, stopped)
    finally:
        state.active.pop(id(value), None)


# --- registry --------------------------------------------------------------------

CaptureFunction = Callable[[Any, "Out"], Snapshot]


_OBJECT_MIN_NODES = 16
"""Nodes held back for a later field that captures a non-built-in value."""


class _Deferred:
    """A part an ``Out`` builds only when its slot is known (see ``Out``).

    Private: extension code only passes the part an ``out`` method returned on
    (as a field, item or value of the same ``out``, or as its return value);
    ``build``, ``min_size`` and ``min_nodes`` are this module's accounting,
    not an API. A part is single-use: it is built in at most one slot.
    """

    __slots__ = ("_build", "_out", "_used", "min_nodes", "min_size")

    def __init__(
        self,
        out: Out,
        build: Callable[[int, int, str], _Built],
        min_size: int,
        min_nodes: int = 1,
    ) -> None:
        self._out = out
        self._build = build
        self._used = False
        self.min_size = min_size
        self.min_nodes = min_nodes

    def build(self, max_size: int, depth: int, path: str) -> _Built:
        """Build in a slot at ``depth`` / ``path``, within ``max_size``; a second use raises."""
        if self._used:
            raise ValueError(
                "an out part takes one slot only: call the out method again for another"
            )
        self._used = True
        return self._build(max_size, depth, path)


def _metadata(value: Any, what: str) -> str:
    """``value`` as an exact built-in ``str`` (a ``str`` subclass is copied), else ``TypeError``.

    Names, kinds and type names a registered function passes are kept in the
    snapshot, which must hold no live object: a ``str`` subclass could carry
    state and run its own ``__str__`` / ``__eq__`` when a renderer reads it.
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


def _length(value: Any) -> int | None:
    """A declared length as an exact non-negative ``int`` (or ``None``), else ``TypeError``."""
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise TypeError("length must be a non-negative int or None")
    return value


class Out:
    """What a registered capture function builds its snapshot with.

    Every method returns a deferred part: nothing is built (and no item is
    pulled) until the part takes its slot, as a field value of
    ``object(...)``, an item of ``sequence(...)``, a value of ``mapping(...)``
    or the function's return value. So the function returns one tree of
    parts, built top-down, each part in its own slot (its share of the
    budget, its depth and path). Parts are opaque: pass them on unchanged.
    A part is single-use: a second use raises ``ValueError`` (the value then
    becomes ``Unavailable("error")``); call the method again for a second
    placement. ``object`` builds its fields in order: its own wrapper and the
    minimal fallback of every later field are reserved first (for a ``dict``,
    ``list`` or ``tuple`` of fields, and only while the slot fits them: a
    field is never held below its own minimum), so a greedy field (a long
    ``sequence``) gets what is left. The function returns one part of its
    own ``Out``; it always fits the value's slot of the shared budget (node
    work is charged to the shared ``max_nodes`` as it happens). When each part
    is built and which budget it uses: *Extension contract* in
    ``docs/snapshot-model.md``.
    """

    def __init__(self, value: Any, max_size: int, depth: int, path: str, state: _State) -> None:
        self._value = value
        self._max_size = max_size
        self._depth = depth
        self._path = path
        self._state = state

    def _finish(self, result: Any) -> _Built | None:
        """The function's result built in the value's own slot (None: not a part of ours)."""
        part = _part_of(result, self)
        return None if part is None else part.build(self._max_size, self._depth, self._path)

    def capture(self, child: Any) -> _Deferred:
        """``child`` captured as this value's snapshot or a part of it.

        It is built in the slot it takes: returned, at this value's depth and
        path; as a field, item or value, one level deeper, under that name.
        """

        def build(max_size: int, depth: int, path: str) -> _Built:
            if max_size < 1:
                return _Built(Unavailable("max_length", _builtin_name(child)), 1)
            return _build(child, max_size, depth, path, self._state)

        # A non-built-in value needs node work for its MRO and a few fields.
        min_nodes = 1 if _builtin_name(child) else _OBJECT_MIN_NODES
        return _Deferred(self, build, _min_size(child), min_nodes)

    def _empty(
        self, node_type: Any, type_name: str, length: int | None, max_size: int, depth: int
    ) -> _Built:  # type_name / length: checked by ``_metadata`` / ``_length``
        reason: Reason = "max_depth" if depth >= self._state.max_depth else "max_length"
        if max_size < 2:
            name = "list" if node_type is Sequence else "dict"
            return _Built(Unavailable(reason, name), 1)
        return _Built(node_type(type_name, length, (), False, reason), 2)

    def sequence(
        self, items: Iterable[Any], *, length: int | None = None, type_name: str = "list"
    ) -> _Deferred:
        """A sequence of ``items`` (pulled one at a time when built), cut like a list."""
        type_name = _metadata(type_name, "Sequence type_name")
        length = _length(length)
        state = self._state

        def build(max_size: int, depth: int, path: str) -> _Built:
            if depth >= state.max_depth or max_size < 2:
                return self._empty(Sequence, type_name, length, max_size, depth)
            return _sequence_from(items, length, type_name, max_size, depth, path, state, out=self)

        return _Deferred(self, build, 2)

    def mapping(
        self,
        items: Iterable[tuple[Any, Any]],
        *,
        length: int | None = None,
        type_name: str = "dict",
    ) -> _Deferred:
        """A mapping of ``(key, value)`` pairs; keys must be exact built-in scalars."""
        type_name = _metadata(type_name, "Mapping type_name")
        length = _length(length)
        state = self._state

        def build(max_size: int, depth: int, path: str) -> _Built:
            if depth >= state.max_depth or max_size < 2:
                return self._empty(Mapping, type_name, length, max_size, depth)
            head_n = state.limits.max_items
            return _mapping_from(
                items,
                length,
                type_name,
                max_size,
                depth,
                path,
                state,
                head_n,
                out=self,
            )

        return _Deferred(self, build, 2)

    def object(
        self,
        fields: Iterable[tuple[str, Any]] | dict[str, Any],
        *,
        type_name: str | None = None,
        kind: str | None = None,
    ) -> _Deferred:
        """An object with these ``(name, value)`` fields (names must be ``str``).

        ``fields`` may also be a plain ``dict`` (its items, in order). A value
        may be a part of this ``out`` (``out.capture(...)``, a nested
        ``out.object(...)``, ...): fields are built in order, each in its own
        slot, leaving room for the minimal fallback of every later field (for
        a ``dict``, ``list`` or ``tuple``; other iterables are pulled lazily,
        without reservation; see *Extension contract* in
        ``docs/snapshot-model.md`` for where reservation stops). ``kind`` is
        the rendering key stored on the node (``Object.kind``) for per-kind
        render hooks.
        """
        if kind is not None:
            kind = _metadata(kind, "Object kind")
        if type_name is not None:
            type_name = _metadata(type_name, "Object type_name")
        known: list[Any] | None = None
        if isinstance(fields, dict):
            # A dict subclass's own ``items`` is not trusted: read it as a plain dict.
            known = list(dict.items(cast(dict[str, Any], fields)))
        elif type(fields) is list or type(fields) is tuple:
            known = list(cast(Any, fields))
        pairs: Iterable[Any] = known if known is not None else fields
        state = self._state
        name = (
            type_name if type_name is not None else _bounded_type_name(self._value, _MAX_TYPE_NAME)
        )
        module = _module_name(type(self._value), state)
        reserves: list[tuple[int, int]] | None = None
        min_nodes = 1
        if known is not None:
            max_items = state.limits.max_items
            # Later fields are cut anyway.
            reserves, needed = _field_reserves(known if max_items is None else known[:max_items])
            min_nodes += needed

        def build(max_size: int, depth: int, path: str) -> _Built:
            if depth >= state.max_depth or max_size < 2:
                reason: Reason = "max_depth" if depth >= state.max_depth else "max_length"
                if max_size < 2:
                    return _Built(Unavailable(reason, "dict"), 1)
                return _Built(Object(name, (), False, reason, module, kind), 2)
            stopped: list[Reason] = []

            def selected() -> typing.Iterator[tuple[str, Any] | None]:
                for field_name, child in pairs:
                    if not state.take_node():
                        stopped.append("max_nodes")
                        return
                    if type(field_name) is not str:
                        yield None
                        return
                    yield field_name, child

            built = _object_fields(
                self._value,
                name,
                module,
                selected(),
                max_size,
                depth,
                path,
                state,
                kind,
                out=self,
                reserves=reserves,
            )
            return _with_stop(built, stopped)

        return _Deferred(self, build, 2, min_nodes)

    def opaque(self) -> _Deferred:
        """This value as ``Opaque`` (what an unregistered type becomes)."""
        state = self._state

        def build(max_size: int, depth: int, path: str) -> _Built:
            return _opaque(self._value, max(max_size, 1), state)

        return _Deferred(self, build, 2)


def _field_reserves(pairs: list[tuple[Any, Any]]) -> tuple[list[tuple[int, int]], int]:
    """Per field: the (chars, nodes) every later field needs for its minimal fallback.

    Also the node work all the fields need together (an object part's ``min_nodes``).
    """
    needs: list[tuple[int, int]] = []
    for entry in pairs:
        if type(entry) is not tuple or len(entry) != 2 or type(entry[0]) is not str:
            break  # the object fails closed or stops here
        name, child = entry
        # separator + "name": + minimal value; one node for the field, some for its value.
        nodes = 1 + (child.min_nodes if type(child) is _Deferred else 1)
        needs.append((2 + len(dumps(name)) + 2 + _min_size(child), nodes))
    reserves: list[tuple[int, int]] = []
    chars = nodes = 0
    for need_chars, need_nodes in reversed(needs):
        reserves.append((chars, nodes))
        chars += need_chars
        nodes += need_nodes
    reserves.reverse()
    return reserves, nodes


_TYPE_MODULE = type.__dict__["__module__"]
_TYPE_QUALNAME = type.__dict__["__qualname__"]


def _raw_qualified_name(cls: type[Any], state: _State) -> tuple[str, str] | None:
    """``(__module__, __qualname__)`` read through ``type``'s own descriptors.

    The getters of ``type`` read the class's native storage, so neither a
    metaclass ``__getattribute__`` nor a class-level attribute can intercept
    them. Non-``str`` values (possible for ``__module__``) give ``None``. The
    ``__module__`` getter looks the name up in the class namespace, so that
    namespace is checked first (``_raw_namespace``: exact ``str`` keys only).
    """
    if _raw_namespace(cls, state) is None:
        return None
    try:
        module = _TYPE_MODULE.__get__(cls, type(cls))
        qualname = _TYPE_QUALNAME.__get__(cls, type(cls))
    except Exception:
        return None
    if type(module) is not str or type(qualname) is not str:
        return None
    return module, qualname


class _Registry:
    """Eager registrations (by class identity) and lazy ones (by qualified name)."""

    def __init__(self) -> None:
        self.entries: list[tuple[type[Any], CaptureFunction]] = []
        # (module, qualname) -> function; keys are exact ``str`` pairs only.
        self.lazy: dict[tuple[str, str], CaptureFunction] = {}
        # id(cls) -> (weak reference to cls, its raw (module, qualname) then,
        # lazy function or None). Keyed by ``id`` and checked by identity, so
        # a class is never hashed or compared; the name is compared too, as a
        # class can legally be renamed (``__module__`` / ``__qualname__``).
        self._resolved: dict[
            int,
            tuple[weakref.ref[type[Any]], tuple[str, str] | None, CaptureFunction | None],
        ] = {}

    def changed(self) -> None:
        """Forget cached lazy resolutions (a registration was added or removed)."""
        self._resolved.clear()

    def _lazy_for(self, cls: type[Any], state: _State) -> CaptureFunction | None:
        if not self.lazy:
            return None
        marker = id(cls)
        name = _raw_qualified_name(cls, state)
        cached = self._resolved.get(marker)
        # Exact ``str`` pairs (or None): comparing them runs no user code.
        if cached is not None and cached[0]() is cls and cached[1] == name:
            return cached[2]
        function = self.lazy.get(name) if name is not None else None
        resolved = self._resolved

        def forget(_ref: Any, marker: int = marker) -> None:
            entry = resolved.get(marker)
            if entry is not None and entry[0] is _ref:
                del resolved[marker]

        try:
            reference = weakref.ref(cls, forget)
        except TypeError:
            return function  # not weakly referenceable: resolve without caching
        resolved[marker] = (reference, name, function)
        return function

    def find(self, mro: tuple[type[Any], ...], state: _State) -> tuple[int, CaptureFunction] | None:
        """The registration nearest in ``mro`` (identity comparisons only).

        At one class an eager registration wins over a lazy one.
        """
        for index, base in enumerate(mro):
            for registered, function in self.entries:
                if registered is base:
                    return index, function
            lazy = self._lazy_for(base, state)
            if lazy is not None:
                return index, lazy
        return None


_REGISTRY = _Registry()


def _build_registered(
    function: CaptureFunction, value: Any, max_size: int, depth: int, path: str, state: _State
) -> _Built:
    ref = _enter(value, path, "object", state)
    if ref is not None:
        return _Built(ref, 2)
    out = Out(value, max_size, depth, path, state)
    result: _Built | None = None
    try:
        result = out._finish(function(value, out))
    except Exception:
        result = None
    finally:
        state.active.pop(id(value), None)
    if result is None or result.size > max_size:
        failed = Unavailable("error", _bounded_type_name(value, _MAX_TYPE_NAME))
        return _Built(failed, fallback_json("", max_size)[1])
    return result


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
    found = _REGISTRY.find(mro, state)
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
        An exception in ``function`` (a part used twice too), or a result
        that is not a part of its ``out``, becomes ``Unavailable("error")``.
        """
        if not isinstance(cls, type):
            raise TypeError(f"capture.register needs a type, got {type(cls).__name__}")

        def add(function: CaptureFunction) -> CaptureFunction:
            entries = _REGISTRY.entries
            entries[:] = [entry for entry in entries if entry[0] is not cls]
            entries.append((cls, function))
            _REGISTRY.changed()
            return function

        return add(function) if function is not None else add

    def unregister(self, cls: type[Any]) -> None:
        """Remove the registration for exactly ``cls`` (no-op when absent)."""
        entries = _REGISTRY.entries
        entries[:] = [entry for entry in entries if entry[0] is not cls]
        _REGISTRY.changed()

    def register_lazy(
        self, module: str, qualname: str, function: CaptureFunction | None = None
    ) -> Any:
        """Register ``function(value, out)`` for the class named ``module`` / ``qualname``.

        Use as ``@capture.register_lazy("numpy", "ndarray")`` or
        ``capture.register_lazy("numpy", "ndarray", function)``. Only the name
        is recorded: the module is never imported, so a library NOOA does not
        depend on can be supported. At capture time each class of the value's
        raw MRO is matched by its native ``__module__`` and ``__qualname__``
        (no metaclass hooks); the nearest class matching an eager
        (``register``) or a lazy registration wins, and at the same class the
        eager one. Registering the same name again replaces it.
        """
        if type(module) is not str or type(qualname) is not str or not module or not qualname:
            raise TypeError("capture.register_lazy needs a module and a qualname (non-empty str)")
        key = (module, qualname)

        def add(function: CaptureFunction) -> CaptureFunction:
            _REGISTRY.lazy[key] = function
            _REGISTRY.changed()
            return function

        return add(function) if function is not None else add

    def unregister_lazy(self, module: str, qualname: str) -> None:
        """Remove the lazy registration for ``module`` / ``qualname`` (no-op when absent)."""
        _REGISTRY.lazy.pop((module, qualname), None)
        _REGISTRY.changed()


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
