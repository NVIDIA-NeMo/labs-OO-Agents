# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The capture builder: one bounded walk over a live value, each node sized as canonical JSON.

What lives here: dispatch (exact built-in types natively, then the shape of
the nearest MRO entry: a ``capture.register`` registration, Pydantic's
``BaseModel``, a dataclass), the container, text and object builders, the
builders of the registered shapes and the typed fallbacks. Every node's
size is its exact ``render_json`` size, so the walk stays within
``Limits.max_chars``; every step charges ``Limits.max_nodes``.

Live classes and instances are read through ``_raw``. Shape accessors are
the only extension code that runs, and only here (see ``_raw``).
"""

from __future__ import annotations

import functools
import itertools
import math
import operator
import typing
from collections.abc import Callable, Iterable
from typing import Any, Literal, NamedTuple, cast

from pydantic import BaseModel

from nooa.snapshot._codec import (
    Container,
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
from nooa.snapshot._limits import Limits, _split, _State
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
from nooa.snapshot._raw import (
    _MAX_TYPE_NAME,
    _bounded_type_name,
    _builtin_name,
    _FailClosed,
    _raw_namespace,
    _safe_mro,
)
from nooa.snapshot._shapes import (
    _DATACLASS,
    _PYDANTIC,
    _Collection,
    _FieldReader,
    _metadata,
    _NdarrayLike,
    _ObjectLike,
    _Shape,
    _TextLike,
)
from nooa.snapshot._visibility import _Fields, _Omitted


class _Built(NamedTuple):
    node: Snapshot
    size: int


# --- fallbacks -------------------------------------------------------------


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


def _at(_data: Any, index: int) -> int:
    """Any index of a ``str`` is a cut point (``safe_end`` / ``safe_start`` for bytes)."""
    return index


def _same(text: str) -> str:
    return text


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
    boundary). Every smaller cut is a part of what was read, so the text is
    never read again. ``whole_scalar``: complete bytes are a ``Scalar``.
    """
    chars = unit == "chars"
    if max_size < 2:
        return _Built(Unavailable("max_length", "str" if chars else "bytes"), 1)
    raw_limit = max_size - 2  # a byte decodes to at most one character
    cap = raw_limit if limits.max_string is None else min(raw_limit, limits.max_string)
    if length <= cap:  # small enough to be kept whole: one read
        whole = read(0, length)
        encoded_size = len(dumps(whole if chars else decode(whole)))
        if encoded_size <= max_size:
            if whole_scalar:
                return _Built(Scalar(whole), encoded_size)
            return _Built(Text(head=whole, length=length, unit=unit), encoded_size)
        head_src = tail_src = whole
        tail_offset, cap = 0, length - 1
    else:
        head_n, tail_n = _split(limits, cap)
        tail_offset = length - tail_n
        empty: Any = "" if chars else b""
        tail_src = read(tail_offset, length) if tail_n else empty
        stop = head_n if chars else head_n + 1
        if not head_n:
            head_src = empty
        elif stop < length:
            head_src = read(0, stop)
        else:  # no tail, and the head is one byte short of the whole: read that byte apart
            head_src = read(0, head_n) + read(head_n, length)
    end, start, show = (_at, _at, _same) if chars else (safe_end, safe_start, decode)

    def parts(kept: int) -> tuple[Any, Any]:
        head_n, tail_n = _split(limits, kept)
        head_end = end(head_src, head_n)
        tail_start = start(tail_src, length - tail_n - tail_offset) + tail_offset
        tail_start = max(tail_start, head_end) if tail_n else length
        return head_src[:head_end], tail_src[tail_start - tail_offset :]

    def cost(kept: int) -> int:
        head, tail = parts(kept)
        return _string_cost(show(head), show(tail))

    # Nothing is kept at 0 (cost 2 <= max_size), so the largest fitting cut always fits.
    kept = _largest(lambda kept: cost(kept) <= max_size, cap)
    head, tail = parts(kept)
    node = Text(head=head, tail=tail, length=length, unit=unit, reason="max_length")
    return _Built(node, cost(kept))


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


def _enter(
    value: Any,
    kind: Literal["sequence", "mapping", "object"],
    empty: Snapshot,
    depth: int,
    path: str,
    state: _State,
    build: Callable[[], _Built],
) -> _Built:
    """``build()`` the container ``value`` while it is active.

    At ``max_depth`` it is ``empty`` (a node without entries, cut ``max_depth``);
    inside itself, a ``Ref`` to its active ancestor.
    """
    if depth >= state.limits.max_depth:
        return _Built(empty, 2)
    marker = id(value)
    found = state.active.get(marker)
    if found is not None:
        return _Built(Ref(found[0], found[1]), 2)
    state.active[marker] = (path, kind)
    try:
        return build()
    finally:
        state.active.pop(marker, None)


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
    get = list.__getitem__ if type(value) is list else tuple.__getitem__
    head_n, tail_n = _plan(length, state.limits, has_tail=True)
    empty = Sequence(type_name, length, (), "max_depth")
    return _enter(
        value,
        "sequence",
        empty,
        depth,
        path,
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
    head_n, tail_n = _plan(length, state.limits, has_tail=True)
    empty = Mapping("dict", length, (), "max_depth")
    return _enter(
        value,
        "mapping",
        empty,
        depth,
        path,
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
    also costs one node for its key, checked before the pair is pulled. A
    mapping stops with ``"key"`` at a key with no JSON form or one whose
    JSON form an earlier kept key has (``1`` and ``"1"``), so the snapshot
    renders as it was sized.
    """
    box = Container(max_size)
    seen: set[str] = set()  # the JSON keys kept so far (our own exact ``str``)

    def add(entry: Any, index: int, kept: list[Any]) -> bool:
        if state.nodes_left <= 0:
            box.stop("max_nodes")
            return False
        key: Any = None
        if keyed:
            key, child = entry
            state.take_node()
            segment = json_key(key, live=True)
            if segment is None or segment in seen:
                # No JSON key, or one that collides in JSON (``1`` and ``"1"``):
                # stop here, as ``render_json`` would, keeping the first.
                box.stop("key")
                return False
            seen.add(segment)
        else:
            child, segment = entry, str(index)
        slot = box.slot(segment if keyed else None)
        if slot is None:
            return False
        overhead, available = slot
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


def _closed(type_name: str) -> _Built:
    """A model whose fields cannot be read safely: no field at all (fail closed)."""
    return _Built(Object(type_name, (), "unreadable"), 2)


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
    box = Container(max_size)
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
        slot = box.slot(name)
        if slot is None:
            break
        overhead, room = slot
        held_chars, held_nodes = (
            reserves[index] if reserves is not None and index < len(reserves) else (0, 0)
        )
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


_OBJECT_MIN_NODES = 16
"""Nodes held back for a later field whose value is not known yet (its MRO and a few fields)."""

_NDARRAY_ELEMENT_CHARS = 6
"""Characters ``ndarray_like`` assumes per shown element (a short number and its separator)."""


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
    shape: _Collection,
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
    head_n, _tail_n = _plan(length, state.limits, has_tail=False)
    items = _guarded(source, pairs=shape.keyed)
    return _container(items, length, type_name, shape.keyed, max_size, depth, path, state, head_n)


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


def _shape_fields(
    shape: _ObjectLike | _Collection | _NdarrayLike,
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
        entries += _structure(shape, value, max_size, state)
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
    shape: _ObjectLike | _Collection | _NdarrayLike, value: Any, max_size: int, state: _State
) -> list[tuple[str, Any]]:
    """The structure fields that follow a shape's accessor fields (``_ShapeError``: whole value)."""
    match shape:
        case _Collection():
            collection = shape
            length = _shape_length(collection.length, value)
            node_type, name = (Mapping, "dict") if collection.keyed else (Sequence, "list")

            def items(slot: int, depth: int, path: str) -> _Built:
                return _collection(collection, value, name, length, slot, depth, path, state)

            return [("items", _structure_part(node_type, length, items, state))]
        case _NdarrayLike():
            return _ndarray_structure(shape, value, max_size, state)
        case _ObjectLike():
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


def _build_text(
    shape: _TextLike, value: Any, type_name: str, max_size: int, state: _State
) -> _Built:
    """A ``Text`` of a ``text_like`` value, reading only the head and tail it keeps.

    Any accessor error (``length``, ``slice``, a slice of the wrong type or
    size) leaves no sensible part: the whole value is ``Unavailable("error")``.
    """
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
    """Build ``value`` (whose MRO entry ``carrier`` matched) as ``shape`` describes it.

    Pydantic models and dataclasses are ``_FieldReader`` shapes: their
    privileged reader yields the fields of an ``Object``. A registered shape
    with fields is an ``Object`` of its accessor fields (built only when kept,
    each leaving room for the later ones) and its structure (``items``,
    ``shape`` / ``labels`` / ``data``), never cut by ``max_items``.
    """
    type_name = _bounded_type_name(value, _MAX_TYPE_NAME)
    match shape:
        case _TextLike():
            return _build_text(shape, value, type_name, max_size, state)
        case _FieldReader(read=read):
            return _enter(
                value,
                "object",
                Object(type_name, (), "max_depth"),
                depth,
                path,
                state,
                lambda: _object_fields(
                    type_name, read(value, mro, carrier, state), max_size, depth, path, state
                ),
            )
        case _Collection(fields=()):
            collection = shape
            try:
                length = _shape_length(collection.length, value)
            except _ShapeError:
                return _error(type_name, max_size)
            node_type = Mapping if collection.keyed else Sequence
            return _enter(
                value,
                "mapping" if collection.keyed else "sequence",
                node_type(type_name, length, (), "max_depth"),
                depth,
                path,
                state,
                lambda: _collection(
                    collection, value, type_name, length, max_size, depth, path, state
                ),
            )
        case _:
            accessors = shape
            return _enter(
                value,
                "object",
                Object(type_name, (), "max_depth", accessors.kind),
                depth,
                path,
                state,
                lambda: _shape_fields(accessors, value, type_name, max_size, depth, path, state),
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
