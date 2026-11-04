# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``capture()``: a bounded snapshot of a finished Python value, without running user code.

Based on the bounded trace JSON builder of PR #342 (Severin Klingler): exact
built-in types are read natively, Pydantic v2 models and dataclasses through
their declared fields in raw instance storage, and everything else is
``Unavailable("opaque")``. No ``repr()``, ``str()``, truthiness, properties, descriptors,
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
import functools
import itertools
import math
import operator
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
    decode,
    dumps,
    fallback_candidates,
    fallback_json,
    json_key,
    opaque_json,
    opaque_name_budget,
    pointer_child,
    safe_end,
    safe_start,
)
from nooa.snapshot._model import (
    Key,
    Mapping,
    Object,
    Reason,
    Ref,
    Scalar,
    Sequence,
    Snapshot,
    Text,
    Unavailable,
)

__all__ = [
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


def _split(limits: Limits, kept: int) -> tuple[int, int]:
    """``kept`` units as (head, tail) under ``limits.tail_share``."""
    tail = int(kept * limits.tail_share)
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
        return _Built(Text(head="", length=length, unit="chars", reason=reason), 2)
    if size == 2 and (value_type is list or value_type is tuple):
        length = len(value)  # exact built-in: no user hook
        return _Built(Sequence(value_type.__name__, length, (), reason), 2)
    if size == 2 and value_type is dict:
        return _Built(Mapping("dict", dict.__len__(value), (), reason), 2)
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
    try:
        namespace = _TYPE_DICT.__get__(cls, type(cls))
    except Exception:
        return None
    if type(namespace) is not MappingProxyType:
        return None
    return _safe_dict(cls, namespace, state)


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


# --- scalars and strings -----------------------------------------------------


def _string_cost(head: str, tail: str) -> int:
    """JSON cost of a kept string: the encoded head plus the tail's characters."""
    return len(dumps(head)) + len(tail)


def _slicer(value: str | bytes) -> Callable[[int, int], Any]:
    """``read(start, stop)`` of an exact ``str`` or ``bytes``: a native slice."""
    return lambda start, stop: value[start:stop]


def _largest(fits: Callable[[int], bool], high: int) -> int:
    """The largest ``k`` in ``0..high`` with ``fits(k)``, for a ``fits`` that turns false as ``k`` grows."""
    if fits(high):
        return high
    low = 0
    while low < high:
        midpoint = (low + high + 1) // 2
        if fits(midpoint):
            low = midpoint
        else:
            high = midpoint - 1
    return low


def _cut_text(
    length: int,
    read: Callable[[int, int], Any],
    unit: Literal["chars", "bytes"],
    max_size: int,
    limits: Limits,
    whole_scalar: bool = False,
) -> _Built:
    """A ``Text`` of the ``length`` units that ``read(start, stop)`` returns, cut to fit.

    Shared by exact ``str`` / ``bytes`` (``read`` slices them) and ``text_like``
    (its checked ``slice``). ``read`` runs at most twice, only for what may be
    kept: the whole text when it is short enough, else the tail and the head
    (one byte longer for bytes, to see whether the head ends on a code point
    boundary). ``whole_scalar``: complete bytes are a ``Scalar`` (exact ``bytes``).
    """
    chars = unit == "chars"
    if max_size < 2:
        return _Built(Unavailable("max_length", "str" if chars else "bytes"), 1)
    raw_limit = max_size - 2  # a byte decodes to at most one character
    keep_cap = raw_limit if limits.max_string is None else min(raw_limit, limits.max_string)
    if length <= keep_cap:  # small enough to be kept whole: one read
        whole = read(0, length)
        encoded_size = len(dumps(whole if chars else decode(whole)))
        if encoded_size <= max_size:
            if whole_scalar:
                return _Built(Scalar(whole), encoded_size)
            return _Built(Text(head=whole, length=length, unit=unit), encoded_size)
        if chars:
            return _cut_chars(length, whole, whole, length, max_size, limits)
        return _cut_byte_parts(length, whole, whole, 0, length - 1, max_size, limits)
    head_n, tail_n = _split(limits, keep_cap)
    empty: Any = "" if chars else b""
    tail = read(length - tail_n, length) if tail_n else empty
    if chars:
        head = read(0, head_n) if head_n else empty
        return _cut_chars(length, head, tail, keep_cap, max_size, limits)
    if not head_n:
        head = empty
    elif head_n + 1 < length:
        head = read(0, head_n + 1)
    else:  # no tail, and the head is one byte short of the whole: read that byte apart
        head = read(0, head_n) + read(head_n, length)
    return _cut_byte_parts(length, head, tail, length - tail_n, keep_cap, max_size, limits)


def _cut_chars(
    length: int, head_src: str, tail_src: str, kept_cap: int, max_size: int, limits: Limits
) -> _Built:
    """The cut ``Text`` of a string of ``length`` chars, keeping at most ``kept_cap`` of them.

    ``head_src`` / ``tail_src`` are the original's first / last chars that
    ``_split(limits, kept_cap)`` keeps (the whole string for both when
    ``kept_cap >= length``); every smaller cut is a part of them, so the
    original is never read again.
    """

    def parts(kept: int) -> tuple[str, str]:
        if kept >= length:
            return head_src, ""
        head_n, tail_n = _split(limits, kept)
        return head_src[:head_n], tail_src[len(tail_src) - tail_n :] if tail_n else ""

    head, tail = parts(_largest(lambda kept: _string_cost(*parts(kept)) <= max_size, kept_cap))
    kept = len(head) + len(tail)
    if kept >= length:
        return _Built(Text(head=head, length=length, unit="chars"), len(dumps(head)))
    node = Text(head=head, tail=tail, length=length, unit="chars", reason="max_length")
    return _Built(node, _string_cost(head, tail))


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

    With ``(H, T) = _split(limits, kept_cap)``: ``head_src`` holds the
    original's first ``H + 1`` bytes (one more, to see whether ``H`` is a code
    point boundary) and ``tail_src`` the bytes from ``tail_offset = length - T``;
    both may be the whole value (offset 0). Every smaller cut is a part of them.
    """

    def parts(kept: int) -> tuple[bytes, bytes]:
        head_n, tail_n = _split(limits, kept)
        head_end = safe_end(head_src, head_n) if head_n else 0
        if tail_n:
            tail_start = safe_start(tail_src, length - tail_n - tail_offset) + tail_offset
        else:
            tail_start = length
        tail_start = max(tail_start, head_end)
        return head_src[:head_end], tail_src[tail_start - tail_offset :]

    def cost(kept: int) -> int:
        head, tail = parts(kept)
        return _string_cost(decode(head), decode(tail))

    # parts(0) is empty (cost 2 <= max_size), so the largest fitting cut always fits.
    head, tail = parts(_largest(lambda kept: cost(kept) <= max_size, kept_cap))
    node = Text(head=head, tail=tail, length=length, unit="bytes", reason="max_length")
    return _Built(node, _string_cost(decode(head), decode(tail)))


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


def _opaque(value: Any, max_size: int) -> _Built:
    """An ``Unavailable("opaque")`` node for an unsupported value, sized as its bounded label.

    ``builtin``: the value's type is an exact built-in one (a non-finite
    ``float``), so its typed #342 fallback applies when the label does not fit.
    """
    name = _bounded_type_name(value, opaque_name_budget(max_size))
    fallback_type = _builtin_name(value)
    _label, size = opaque_json(name, fallback_type, max_size)
    return _Built(Unavailable("opaque", name, builtin=bool(fallback_type)), size)


# --- containers ---------------------------------------------------------------


def _entered(
    value: Any,
    path: str,
    kind: Literal["sequence", "mapping", "object"],
    state: _State,
    build: Callable[[], _Built],
) -> _Built:
    """``build()`` the container ``value`` while it is active, or a ``Ref`` to its active ancestor."""
    marker = id(value)
    found = state.active.get(marker)
    if found is not None:
        return _Built(Ref(found[0], found[1]), 2)
    state.active[marker] = (path, kind)
    try:
        return build()
    finally:
        state.active.pop(marker, None)


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


def _plan(length: int, limits: Limits, has_tail: bool) -> tuple[int, int]:
    """The (head, tail) entries of a container of ``length`` that ``max_items`` keeps.

    A source that can only be read forwards (``has_tail=False``) keeps a head only.
    """
    max_items = limits.max_items
    if max_items is None or length <= max_items:
        return length, 0
    return _split(limits, max_items) if has_tail else (max_items, 0)


def _build_sequence(
    value: list[Any] | tuple[Any, ...], max_size: int, depth: int, path: str, state: _State
) -> _Built:
    """Build a bounded sequence node from an exact list or tuple."""
    type_name = "list" if type(value) is list else "tuple"
    length = len(value)  # an exact built-in: no user hook
    if depth >= state.limits.max_depth:
        return _Built(Sequence(type_name, length, (), "max_depth"), 2)
    get = list.__getitem__ if type(value) is list else tuple.__getitem__
    head_n, tail_n = _plan(length, state.limits, has_tail=True)
    return _entered(
        value,
        path,
        "sequence",
        state,
        lambda: _container(
            (get(cast(Any, value), index) for index in range(head_n)),
            length,
            type_name,
            False,
            max_size,
            depth,
            path,
            state,
            head_n,
            tail_n,
            lambda: (get(cast(Any, value), index) for index in range(length - tail_n, length)),
        ),
    )


def _build_dict(
    value: dict[Any, Any], max_size: int, depth: int, path: str, state: _State
) -> _Built:
    """Build a bounded mapping node without visiting rejected tail entries."""
    length = dict.__len__(value)
    if depth >= state.limits.max_depth:
        return _Built(Mapping("dict", length, (), "max_depth"), 2)
    head_n, tail_n = _plan(length, state.limits, has_tail=True)
    return _entered(
        value,
        path,
        "mapping",
        state,
        lambda: _container(
            dict.items(value),
            length,
            "dict",
            True,
            max_size,
            depth,
            path,
            state,
            head_n,
            tail_n,
            lambda: list(itertools.islice(reversed(dict.items(value)), tail_n))[::-1],
        ),
    )


def _container(
    entries: Iterable[Any],
    length: int,
    type_name: str,
    keyed: bool,
    max_size: int,
    depth: int,
    path: str,
    state: _State,
    head_n: int,
    tail_n: int = 0,
    tail: Callable[[], Iterable[Any]] = tuple,
) -> _Built:
    """A ``Sequence`` of ``length`` items, or (``keyed``) a ``Mapping`` of ``(key, value)`` entries.

    Keeps the first ``head_n`` (at most ``length``) entries pulled from
    ``entries`` and, when every one of them fit, the last ``tail_n`` from
    ``tail()``, under the shared budget. Entries may be private ``_Part``
    builders (see ``_build_slot``), or ``_Omitted`` notes. A mapping entry
    also costs one node for its key, checked before the pair is pulled.
    """
    box = _Container(max_size)

    def add(entry: Any, index: int, kept: list[Any]) -> bool:
        if state.nodes_left <= 0:
            box.stop("max_nodes")
            return False
        key: Any = None
        if keyed:
            key, child = entry
            state.take_node()
            segment = json_key(key, live=True)
            if segment is None:
                box.stop("key")
                return False
            overhead = _key_overhead(segment, box)
            if overhead is None:
                box.stop("max_length")
                return False
        else:
            child, segment, overhead = entry, str(index), 0
        available = box.room(overhead)
        if available < 1:
            box.stop("max_length")
            return False
        built = _build_slot(child, available, depth + 1, pointer_child(path, segment), state)
        kept.append((cast(Key, key), built.node) if keyed else built.node)
        box.commit(overhead, built.size)
        return True

    head: list[Any] = []
    omitted: Reason | None = None
    iterator = iter(entries)
    while len(head) < head_n:
        if keyed and state.nodes_left <= 0:
            box.stop("max_nodes")
            break
        try:
            entry = next(iterator)
        except StopIteration:
            break
        if type(entry) is _Omitted:
            omitted = omitted or entry.reason
            continue
        if not add(entry, len(head), head):
            break
    kept_tail: list[Any] = []
    if box.reason is None and tail_n:
        for offset, entry in enumerate(tail()):
            if not add(entry, length - tail_n + offset, kept_tail):
                break
    reason = box.reason or omitted
    if reason is None and len(head) + len(kept_tail) < length:
        reason = "max_items"
    if keyed:
        node: Snapshot = Mapping(type_name, length, tuple(head), reason, tuple(kept_tail))
    else:
        node = Sequence(type_name, length, tuple(head), reason, tuple(kept_tail))
    return _Built(node, box.size)


# --- visibility (agentdoc ``hidden`` / ``spec(hidden=...)``) ------------------------


_MALFORMED = object()


def _hidden_flag(raw: Any, state: _State) -> Any:
    """The ``hidden`` entry of raw metadata keywords: a ``bool``, ``None`` when absent.

    ``_MALFORMED`` (fail closed) unless ``raw`` is an exact ``dict`` with exact
    ``str`` keys, each charged one node, whose ``hidden`` is absent or a ``bool``.
    """
    if type(raw) is not dict:
        return _MALFORMED
    found = None
    for name, value in dict.items(raw):
        if not state.take_node() or type(name) is not str:
            return _MALFORMED
        if name == "hidden":
            found = value
    return found if found is None or type(found) is bool else _MALFORMED


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
            if not state.take_node() or type(field_name) is not str:
                return False
            flag = _hidden_flag(metadata, state)
            if flag is _MALFORMED:
                return False
            if flag is not None and (overwrite or field_name not in class_overrides):
                class_overrides[field_name] = flag
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
            flag = _hidden_flag(kwargs, state)
            if flag is _MALFORMED:
                return None
            if flag:
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
    slot; a later sibling reserves ``min_nodes`` and 2 characters for it.
    ``exhausted(max_size)``: what a structural part (an items sequence, an
    ndarray level) becomes when no node work is left for it; ``None`` for a
    part that captures a value (``_lazy``), whose ``_build`` charges itself.
    """

    __slots__ = ("build", "exhausted", "min_nodes")

    def __init__(
        self,
        build: Callable[[int, int, str], _Built],
        min_nodes: int,
        exhausted: Callable[[int], _Built] | None = None,
    ) -> None:
        self.build = build
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


def _closed(type_name: str) -> _Built:
    """A model whose fields cannot be read safely: no field at all (fail closed)."""
    return _Built(Object(type_name, (), "unreadable"), 2)


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


class _Omitted:
    """An entry stream's note that something was left out here, and why.

    The one way a field reader or a shape's items say "incomplete because":
    the builder keeps reading, and the first such reason is the node's when
    the builder's own limits left it complete.
    """

    __slots__ = ("reason",)
    reason: Reason

    def __init__(self, reason: Reason) -> None:
        self.reason = reason


class _FailClosed(Exception):
    """Raised by a field reader whose object can't be read safely: no field is kept."""


def _object_fields(
    type_name: str,
    entries: Iterable[tuple[str, Any] | _Omitted],
    max_size: int,
    depth: int,
    path: str,
    state: _State,
    kind: str | None = None,
    reserves: list[tuple[int, int]] | None = None,
    capped: bool = True,
) -> _Built:
    """Emit the (name, value) fields of an object; a reader's ``_FailClosed`` keeps none.

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
    omitted: Reason | None = None
    fields: list[tuple[str, Snapshot]] = []
    max_items = state.limits.max_items if capped else None
    iterator = iter(entries)
    while True:
        try:
            entry = next(iterator)
        except StopIteration:
            break
        except _FailClosed:
            return _closed(type_name)
        if type(entry) is _Omitted:
            omitted = omitted or entry.reason
            continue
        index = len(fields)
        if max_items is not None and index >= max_items:
            box.stop("max_items")
            break
        name, child = cast(tuple[str, Any], entry)
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
    reason = box.reason or omitted
    node = Object(type_name, tuple(fields), reason, kind)
    return _Built(node, box.size)


def _min_size(child: Any) -> int:
    """The size of the minimal type-oriented fallback a captured value can always keep.

    Only identity checks against exact built-in types: the type of an unknown
    value is never compared, hashed or asked for attributes (a metaclass
    ``__eq__`` must not run).
    """
    if type(child) is _Part:
        return 2
    name = _builtin_name(child)
    return len(dumps(fallback_candidates(name)[0])) if name else 2


_Fields = typing.Iterator[tuple[str, Any] | _Omitted]
"""A field reader's (name, value) fields and ``_Omitted`` notes; it raises ``_FailClosed``."""


def _pydantic_entries(
    value: BaseModel, mro: tuple[type[Any], ...], carrier: type[Any], state: _State
) -> _Fields:
    """The privileged field reader of a Pydantic v2 model: declared fields, raw stored values."""
    try:
        obj_dict = _BASE_MODEL_DICT.__get__(value, type(value))
    except Exception:
        obj_dict = None
    storage = _copy_storage(obj_dict, state)
    if storage is None:
        raise _FailClosed
    overrides = _visibility_overrides(storage, mro, state)
    fields = _pydantic_fields(mro, state)
    if overrides is None or fields is None:
        raise _FailClosed
    for name, field_info in dict.items(fields):
        if not state.take_node():
            yield _Omitted("max_nodes")
            return
        if type(name) is not str or type(field_info) is not FieldInfo:
            raise _FailClosed
        try:
            excluded = object.__getattribute__(field_info, "exclude")
            represented = object.__getattribute__(field_info, "repr")
        except Exception:
            raise _FailClosed from None
        if excluded is not None and type(excluded) is not bool:
            raise _FailClosed
        if type(represented) is not bool:
            raise _FailClosed
        if excluded is True or represented is False:
            continue
        metadata_hidden = _field_hidden(field_info, state)
        if metadata_hidden is None:
            raise _FailClosed
        if overrides.get(name, metadata_hidden) or name not in storage:
            continue
        yield name, dict.__getitem__(storage, name)


def _instance_storage(value: Any, mro: tuple[type[Any], ...], state: _State) -> Any:
    """Raw ``__dict__`` through the native getset descriptor; ``{}`` for slotted classes.

    ``None`` when it can't be read safely (a custom ``__dict__``: refuse).
    """
    for base in mro:
        namespace = _raw_namespace(base, state) if base is not object else None
        if namespace is None:
            continue
        descriptor = dict.get(namespace, "__dict__")
        if descriptor is None:
            continue
        if type(descriptor) is not types.GetSetDescriptorType:
            return None
        try:
            return descriptor.__get__(value, type(value))
        except Exception:
            return None
    return {}


_NO_SLOT = object()


def _slot_value(name: str, mro: tuple[type[Any], ...], value: Any, state: _State) -> Any:
    """A slotted field's value through its native member descriptor, else ``_NO_SLOT``."""
    for base in mro:
        namespace = _raw_namespace(base, state) if base is not object else None
        if namespace is None:
            continue
        descriptor = dict.get(namespace, name)
        if descriptor is None:
            continue
        if type(descriptor) is not types.MemberDescriptorType:
            return _NO_SLOT
        try:
            return descriptor.__get__(value, type(value))
        except Exception:
            return _NO_SLOT
    return _NO_SLOT


def _dataclass_entries(
    value: Any, mro: tuple[type[Any], ...], carrier: type[Any], state: _State
) -> _Fields:
    """The privileged field reader of a dataclass: declared fields of ``carrier``, raw values."""
    storage = _copy_storage(_instance_storage(value, mro, state), state)
    namespace = _raw_namespace(carrier, state)
    fields = dict.get(namespace, "__dataclass_fields__") if namespace is not None else None
    if storage is None or type(fields) is not dict:
        raise _FailClosed
    overrides = _visibility_overrides(storage, mro, state)
    if overrides is None:
        raise _FailClosed
    resolved: Any = _UNREAD  # the class's resolved hints, read at the first string annotation
    for name, field in dict.items(fields):
        if not state.take_node():
            yield _Omitted("max_nodes")
            return
        if type(name) is not str or type(field) is not dataclasses.Field:
            raise _FailClosed
        try:
            field_type = object.__getattribute__(field, "_field_type")
            represented = object.__getattribute__(field, "repr")
            annotation = object.__getattribute__(field, "type")
        except Exception:
            raise _FailClosed from None
        if field_type is not _DATACLASS_FIELD_TYPE:
            continue  # ClassVar / InitVar pseudo-fields
        if type(represented) is not bool:
            raise _FailClosed
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
            yield _Omitted("unreadable")
            continue
        if overrides.get(name, annotation_hidden):
            continue
        if name in storage:
            yield name, dict.__getitem__(storage, name)
            continue
        slot = _slot_value(name, mro, value, state)
        if slot is not _NO_SLOT:
            yield name, slot


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
    fields: tuple[tuple[str, Callable[[Any], Any]], ...]
    kind: str | None


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


@dataclass(frozen=True)
class _FieldReader:
    """The internal shape of Pydantic models and dataclasses: a privileged field reader.

    ``read(value, mro, carrier, state)`` yields the fields of the class
    ``carrier`` (the MRO entry that matched). Never registered by users.
    """

    read: Callable[[Any, tuple[type[Any], ...], type[Any], _State], _Fields]


_PYDANTIC = _FieldReader(_pydantic_entries)
_DATACLASS = _FieldReader(_dataclass_entries)


class _ShapeError(Exception):
    """An accessor failed in a way that leaves no sensible part: the whole value is an error.

    Raised for the whole-value accessors (``length``, ``shape``, ``slice``) and
    the value checks on their results. A part's accessor (a field, an element,
    a label) that raises becomes that part's ``Unavailable("error")``; a shape's
    ``items`` that raise end the items with an ``error`` omission (``_guarded``).
    """


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


def _guarded(iterator: typing.Iterator[Any], pairs: bool) -> typing.Iterator[Any]:
    """A shape's items until they end, or raise (``pairs``: or one is not an exact 2-tuple).

    A failure ends them with an ``error`` omission.
    """
    while True:
        try:
            item = next(iterator)
        except StopIteration:
            return
        except Exception:
            yield _Omitted("error")
            return
        if pairs and (type(item) is not tuple or tuple.__len__(item) != 2):
            yield _Omitted("error")
            return
        yield item


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
    keyed = type(shape) is _MappingLike
    head_n, _tail_n = _plan(length, state.limits, has_tail=False)
    items = _guarded(source, pairs=keyed)
    return _container(items, length, type_name, keyed, max_size, depth, path, state, head_n)


def _lazy(read: Callable[[], Any], state: _State) -> _Part:
    """A part that captures ``read()``, called only when its slot is kept.

    ``read`` is a part's accessor (a field, an element, a label): when it
    raises, that part is ``Unavailable("error")``.
    """

    def build(max_size: int, depth: int, path: str) -> _Built:
        try:
            child = read()
        except Exception:
            return _error("", max_size)
        return _build(child, max_size, depth, path, state)

    return _Part(build, _OBJECT_MIN_NODES)


def _structure_part(
    node_type: Any, length: int, build: Callable[[int, int, str], _Built], state: _State
) -> _Part:
    """A ``Sequence`` / ``Mapping`` field of a shape (``items``, ``data``, ``labels``): one node."""
    name = "list" if node_type is Sequence else "dict"

    def empty(max_size: int, reason: Reason) -> _Built:
        if max_size < 2:
            return _Built(Unavailable(reason, name), 1)
        return _Built(node_type(name, length, (), reason), 2)

    def built(max_size: int, depth: int, path: str) -> _Built:
        if depth >= state.limits.max_depth:
            return empty(max_size, "max_depth")
        if max_size < 2:
            return empty(max_size, "max_length")
        return build(max_size, depth, path)

    return _Part(built, 1, lambda max_size: empty(max_size, "max_nodes"))


def _build_object(
    shape: _ObjectLike | _ArrayLike | _MappingLike | _NdarrayLike | _FieldReader,
    value: Any,
    mro: tuple[type[Any], ...],
    carrier: type[Any],
    max_size: int,
    depth: int,
    path: str,
    state: _State,
) -> _Built:
    """An ``Object`` of a shape: a field reader's fields, or accessor fields and structure parts.

    Pydantic models and dataclasses are ``_FieldReader`` shapes: their
    privileged reader yields the fields. Registered shapes add their accessor
    fields (built only when kept, each leaving room for the later ones) and
    their structure (``items``, ``shape`` / ``labels`` / ``data``), never cut
    by ``max_items``.
    """
    type_name = _bounded_type_name(value, _MAX_TYPE_NAME)
    if isinstance(shape, _FieldReader):  # one of this module's shapes: no user hook
        if depth >= state.limits.max_depth:
            return _Built(Object(type_name, (), "max_depth"), 2)
        reader = shape.read
        return _entered(
            value,
            path,
            "object",
            state,
            lambda: _object_fields(
                type_name, reader(value, mro, carrier, state), max_size, depth, path, state
            ),
        )
    if depth >= state.limits.max_depth:
        return _Built(Object(type_name, (), "max_depth", shape.kind), 2)
    accessors = shape
    return _entered(
        value,
        path,
        "object",
        state,
        lambda: _shape_fields(accessors, value, type_name, max_size, depth, path, state),
    )


def _shape_fields(
    shape: _ObjectLike | _ArrayLike | _MappingLike | _NdarrayLike,
    value: Any,
    type_name: str,
    max_size: int,
    depth: int,
    path: str,
    state: _State,
) -> _Built:
    """A registered shape's ``Object``: its accessor fields, then its structure."""
    entries: list[tuple[str, Any]] = [
        (name, _lazy(functools.partial(function, value), state)) for name, function in shape.fields
    ]
    max_items = state.limits.max_items
    cut = max_items is not None and len(entries) > max_items
    del entries[max_items if cut else len(entries) :]
    try:
        entries += _structure(shape, value, type_name, max_size, state)
    except _ShapeError:
        return _error(type_name, max_size)
    reserves = _field_reserves(entries)

    def fields() -> _Fields:
        if cut:
            yield _Omitted("max_items")
        for entry in entries:
            if not state.take_node():
                yield _Omitted("max_nodes")
                return
            yield entry

    return _object_fields(
        type_name,
        fields(),
        max_size,
        depth,
        path,
        state,
        shape.kind,
        reserves=reserves,
        capped=False,
    )


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

    low = _largest(fits, cap)
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
    max_size: int,
    depth: int,
    path: str,
    state: _State,
) -> _Built:
    """A ``list`` ``Sequence`` of the entries ``child(i)`` at the head and tail indices of ``split``."""
    head_n, tail_n = split
    return _container(
        (child(i) for i in range(head_n)),
        length,
        "list",
        False,
        max_size,
        depth,
        path,
        state,
        head_n,
        tail_n,
        lambda: (child(i) for i in range(length - tail_n, length)),
    )


def _ndarray_structure(
    shape: _NdarrayLike, value: Any, max_size: int, state: _State
) -> list[tuple[str, Any]]:
    """``shape``, ``labels`` and ``data`` of an ``ndarray_like`` value."""
    dims = _ndarray_dims(shape, value)
    selection = _ndarray_selection(dims, len(shape.labels), max_size, state)
    entries: list[tuple[str, Any]] = [("shape", dims)]
    if shape.labels:
        labels = [
            (axis, _ndarray_labels(function, value, dims, selection, axis, state))
            for axis, function in shape.labels
        ]
        count = len(labels)

        def build(slot: int, depth: int, path: str) -> _Built:
            return _container(labels, count, "dict", True, slot, depth, path, state, count)

        entries.append(("labels", _structure_part(Mapping, count, build, state)))
    entries.append(("data", _ndarray_part(shape, value, dims, selection, (), state)))
    return entries


def _ndarray_part(
    shape: _NdarrayLike,
    value: Any,
    dims: tuple[int, ...],
    selection: list[tuple[int, int]],
    index: tuple[int, ...],
    state: _State,
) -> _Part:
    """The part at ``index``: an element once every axis is indexed, else the next axis's level."""
    axis = len(index)
    if axis == len(dims):
        return _lazy(functools.partial(shape.element, value, index), state)

    def level(slot: int, depth: int, path: str) -> _Built:
        def child(i: int) -> _Part:
            return _ndarray_part(shape, value, dims, selection, (*index, i), state)

        return _indexed(dims[axis], selection[axis], child, slot, depth, path, state)

    return _structure_part(Sequence, dims[axis], level, state)


def _ndarray_labels(
    function: Callable[[Any], Any],
    value: Any,
    dims: tuple[int, ...],
    selection: list[tuple[int, int]],
    axis: int,
    state: _State,
) -> _Part:
    """The labels of ``axis``: ``function(value)[i]`` for each shown ``i``.

    An axis the dims don't have is ``Unavailable("error")`` in every slot,
    also when a limit leaves it no room (sized like the sequence would be).
    """
    if axis >= len(dims):

        def error(slot: int, *_where: Any) -> _Built:
            return _error("", slot)

        return _Part(error, 1, error)

    def build(slot: int, depth: int, path: str) -> _Built:
        try:
            found = function(value)
        except Exception:
            return _error("", slot)

        def label(i: int) -> _Part:
            return _lazy(functools.partial(operator.getitem, found, i), state)

        return _indexed(dims[axis], selection[axis], label, slot, depth, path, state)

    return _structure_part(Sequence, dims[axis], build, state)


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

    return _cut_text(length, fetch, shape.unit, max_size, limits)


# --- dispatch ----------------------------------------------------------------------


def _build_shape(
    shape: _Shape | _FieldReader,
    value: Any,
    mro: tuple[type[Any], ...],
    carrier: type[Any],
    max_size: int,
    depth: int,
    path: str,
    state: _State,
) -> _Built:
    """Build ``value`` (whose MRO entry ``carrier`` matched) as ``shape`` describes it."""
    shape_type = type(shape)
    if shape_type is _TextLike:
        return _build_text(cast(_TextLike, shape), value, max_size, state)
    if (
        shape_type is _FieldReader
        or shape_type is _ObjectLike
        or shape_type is _NdarrayLike
        or cast(Any, shape).fields
    ):
        return _build_object(cast(Any, shape), value, mro, carrier, max_size, depth, path, state)
    collection = cast(_ArrayLike | _MappingLike, shape)
    type_name = _bounded_type_name(value, _MAX_TYPE_NAME)
    node_type = Sequence if shape_type is _ArrayLike else Mapping
    try:
        length = _shape_length(collection.length, value)
    except _ShapeError:
        return _error(type_name, max_size)
    if depth >= state.limits.max_depth:
        return _Built(node_type(type_name, length, (), "max_depth"), 2)
    return _entered(
        value,
        path,
        "sequence" if node_type is Sequence else "mapping",
        state,
        lambda: _collection(collection, value, type_name, length, max_size, depth, path, state),
    )


def _field_reserves(pairs: list[tuple[Any, Any]]) -> list[tuple[int, int]]:
    """Per field: the (chars, nodes) every later field needs for its minimal fallback."""
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
    return reserves


_REGISTRY: list[tuple[type[Any], _Shape]] = []
"""The process-wide ``capture.register(T, shape)`` registrations, matched by class identity.

Scanned with ``is``: a class is never hashed or compared with ``==``, so a
metaclass ``__hash__`` / ``__eq__`` cannot run during dispatch.
"""


def _find_shape(
    mro: tuple[type[Any], ...], state: _State
) -> tuple[type[Any], _Shape | _FieldReader] | None:
    """The shape of the nearest MRO entry that has one (identity comparisons only).

    At one entry a registration comes first, then Pydantic's ``BaseModel``,
    then a class whose own namespace declares dataclass fields.
    """
    for base in mro:
        for registered, shape in _REGISTRY:
            if registered is base:
                return base, shape
        if base is BaseModel:
            return base, _PYDANTIC
        namespace = _raw_namespace(base, state) if base is not object else None
        if namespace is not None and type(dict.get(namespace, "__dataclass_fields__")) is dict:
            return base, _DATACLASS
    return None


# --- the walk --------------------------------------------------------------------


def _build(value: Any, max_size: int, depth: int, path: str, state: _State) -> _Built:
    """Recursively build one bounded snapshot node (sized as its JSON encoding)."""
    if not state.take_node():
        return _fallback(value, "max_nodes", max_size)

    limits = state.limits
    value_type = type(value)
    if value_type is str:
        return _cut_text(str.__len__(value), _slicer(value), "chars", max_size, limits)
    if value is None or value_type is bool:
        encoded = "null" if value is None else ("true" if value else "false")
        if len(encoded) <= max_size:
            return _Built(Scalar(value), len(encoded))
        return _fallback(value, "max_length", max_size)
    if value_type is int:
        return _build_integer(value, max_size)
    if value_type is float:
        if not math.isfinite(value):
            return _opaque(value, max_size)
        encoded = dumps(value)
        if len(encoded) <= max_size:
            return _Built(Scalar(value), len(encoded))
        return _fallback(value, "max_length", max_size)
    if value_type is bytes:
        length = bytes.__len__(value)
        return _cut_text(length, _slicer(value), "bytes", max_size, limits, whole_scalar=True)
    if value_type is list or value_type is tuple or value_type is dict:
        if max_size < 2:
            return _fallback(value, "max_length", max_size)
        if value_type is dict:
            return _build_dict(value, max_size, depth, path, state)
        return _build_sequence(value, max_size, depth, path, state)

    mro = _safe_mro(value, state)
    found = None if mro is None else _find_shape(mro, state)
    if mro is None or found is None:
        return _opaque(value, max_size)
    if max_size < 2:
        return _fallback(value, "max_length", max_size)
    carrier, shape = found
    return _build_shape(shape, value, mro, carrier, max_size, depth, path, state)


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
        if not any(shape_type is known for known in _SHAPE_TYPES):
            raise TypeError(
                "capture.register needs a shape (array_like, ndarray_like, mapping_like,"
                f" object_like or text_like), got {_bounded_type_name(shape, _MAX_TYPE_NAME)}"
            )
        _REGISTRY[:] = [entry for entry in _REGISTRY if entry[0] is not cls]
        _REGISTRY.append((cls, shape))

    def unregister(self, cls: type[Any]) -> None:
        """Remove the registration for exactly ``cls`` (no-op when absent)."""
        _REGISTRY[:] = [entry for entry in _REGISTRY if entry[0] is not cls]


capture = _CaptureFunction()


def _field_placeholder(value: Any, state: _State) -> tuple[Any, Snapshot]:
    """One framework wrapper field's minimal JSON root value, and its node when no work is left.

    An exact built-in type other than ``bytes`` reserves its smallest
    fallback (``""``, ``[]``, ``{}``, ``0``, ...), a Pydantic model ``{}``,
    anything else ``null`` (#342's ``trace_fields``).
    """
    if not state.take_node():
        raise ValueError("snapshot max_nodes cannot classify wrapper fields")
    name = _builtin_name(value)
    if name and name != "bytes":
        shape = fallback_candidates(name)[0]
        return shape, _fallback(value, "max_nodes", len(dumps(shape))).node
    mro = _safe_mro(value, state)
    if mro is not None and any(base is BaseModel for base in mro):
        name = _bounded_type_name(value, _MAX_TYPE_NAME)
        return {}, Object(name, (), "max_nodes")
    return None, Unavailable("max_nodes", "NoneType")


def capture_fields(*, limits: Limits = DEFAULT_LIMITS, **fields: object) -> Mapping:
    """Capture a framework-owned wrapper so that every root field survives, with its type.

    Each field first reserves a minimal placeholder of its own type (``[]``,
    ``{}``, ``""``, ...); raises ``ValueError`` when ``limits`` cannot hold the
    complete minimal wrapper (#342's ``trace_fields``). A field reached with
    no node work left keeps its placeholder.
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
        shape, starved = placeholders[name]
        reserved = len(dumps(shape))
        if state.nodes_left > 0:
            available = reserved + (limits.max_chars - total_size)
            built = _build(value, available, 1, pointer_child("", name), state)
        else:
            built = _Built(starved, reserved)
        items.append((name, built.node))
        total_size += built.size - reserved
    return Mapping("dict", len(items), tuple(items), wrapper=True)
