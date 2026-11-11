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

import dataclasses
import functools
import logging
import math
import operator
import time
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
from nooa.snapshot._fields import (
    _DATACLASS,
    _PYDANTIC,
    _FieldReader,
    _plain_entries,
    _Scoped,
    _visibility_overrides,
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
    _Fields,
    _Omitted,
    _raw_namespace,
    _safe_mro,
)
from nooa.snapshot._render import key_text
from nooa.snapshot._shapes import (
    _REGISTRY,
    _Collection,
    _metadata,
    _NdarrayLike,
    _ObjectLike,
    _Shape,
    _TextLike,
)
from nooa.snapshot._std import (
    _CollectionView,
    _exact_view,
    _mro_view,
    _ObjectView,
    _qualified_type_name,
    _TextView,
    _View,
)


class _Built(NamedTuple):
    node: Snapshot
    size: int


# --- fallbacks -------------------------------------------------------------


_log = logging.getLogger(__name__)

SLOW_REPR_SECONDS = 0.1
"""A ``repr`` the fallback runs that takes longer than this logs a developer warning."""
LARGE_REPR_CHARS = 1_000_000
"""A ``repr`` the fallback runs that returns more characters than this logs a developer warning."""


def _run_repr(value: Any, state: _State) -> str | None:
    """``repr(value)`` for the ``"repr"`` fallback; ``None`` when it raises or is not a ``str``.

    The value's own code runs here, unbounded before the cut: one slower than
    ``SLOW_REPR_SECONDS`` or longer than ``LARGE_REPR_CHARS`` logs a warning
    naming the type, the time and the size, so a developer can give the type
    a cheaper reading.
    """
    start = time.perf_counter()
    try:
        text = repr(value)
    except Exception:
        return None
    elapsed = time.perf_counter() - start
    if type(text) is not str:
        return None
    if elapsed > SLOW_REPR_SECONDS or str.__len__(text) > LARGE_REPR_CHARS:
        _log.warning(
            "snapshot: repr() of %s took %.3f s and returned %s characters",
            _qualified_type_name(value, state),
            elapsed,
            f"{str.__len__(text):,}",
        )
    return text


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


def _build_integer(value: int, max_size: int, depth: int, path: str, state: _State) -> _Built:
    """Encode an integer only when decimal conversion is itself bounded; else its digit count."""
    bits = int.bit_length(value)
    # Decimal digits are approximately bits * log10(2). The upper bound keeps
    # conversion below both the output budget and Python's default digit guard.
    estimated_digits = (bits * 30_103) // 100_000 + 1 + (1 if value < 0 else 0)
    if estimated_digits > max_size or estimated_digits > 4_000:
        return _big_integer(value, max_size, depth, path, state)
    try:
        encoded = dumps(value)
    except (ValueError, OverflowError):
        return _big_integer(value, max_size, depth, path, state)
    if len(encoded) > max_size:
        return _big_integer(value, max_size, depth, path, state)
    return _Built(Scalar(value), len(encoded))


def _big_integer(value: int, max_size: int, depth: int, path: str, state: _State) -> _Built:
    """An int too long to keep: an ``Object`` (``kind="python.int"``) of its digit count, cut.

    The count comes from ``math.log10`` (no decimal conversion): exact but
    within float precision of a power of ten, where it may be one off.
    """
    negative = value < 0
    entries: list[tuple[str, Any]] = [
        ("digits", int(math.log10(-value if negative else value)) + 1)
    ]
    if negative:
        entries.append(("negative", True))
    built = _object_fields("int", iter(entries), max_size, depth, path, state, "python.int")
    node = cast(Object, built.node)
    if len(node.fields) < len(entries):
        return _fallback(value, "max_length", max_size)
    return _Built(dataclasses.replace(node, reason="max_length"), built.size)


def _repr_object(value: Any, max_size: int, depth: int, path: str, state: _State) -> _Built | None:
    """``repr(value)`` (bounded, see ``_run_repr``) as an ``Object(kind="python.repr")``; ``None``: none."""
    if max_size < 2:
        return None
    text = _run_repr(value, state)
    if text is None:
        return None
    type_name = _bounded_type_name(value, _MAX_TYPE_NAME)
    entries = iter([("repr", text)])
    return _object_fields(
        type_name, entries, max_size, depth, path, state, "python.repr", capped=False
    )


def _defines_repr(mro: tuple[type[Any], ...], state: _State) -> bool:
    """Whether a class in ``mro`` other than ``object`` defines ``__repr__`` (raw namespaces)."""
    for base in mro:
        if base is object:
            continue
        namespace = _raw_namespace(base, state)
        if namespace is not None and "__repr__" in namespace:
            return True
    return False


def _opaque(value: Any, max_size: int, depth: int, path: str, state: _State) -> _Built:
    """What a value capture cannot read becomes, as ``Limits.fallback`` asks.

    ``"opaque"``: an ``Unavailable("opaque")`` node sized as its bounded
    label, named ``module.QualName`` (``builtins`` left out). ``builtin``:
    the value's type is an exact built-in one (a non-finite ``float``), so
    its typed #342 fallback applies when the label does not fit.
    ``"repr"``: ``repr(value)``, kept as an ``Object`` (``kind="python.repr"``)
    whose ``repr`` field is cut like a string; a raising ``repr`` gives the
    opaque node.
    """
    if state.limits.fallback == "repr":
        built = _repr_object(value, max_size, depth, path, state)
        if built is not None:
            return built
    name = _qualified_type_name(value, state)[: opaque_name_budget(max_size)]
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

    At ``max_depth`` it is ``empty`` (a node without entries, cut ``max_depth``),
    unless it has no entries to cut; inside itself, a ``Ref`` to its active ancestor.
    """
    empty_container = (type(empty) is Sequence or type(empty) is Mapping) and empty.length == 0
    if depth >= state.limits.max_depth and not empty_container:  # an empty one is whole anyway
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


def _builtin_collection(value: list[Any] | tuple[Any, ...] | dict[Any, Any]) -> _CollectionView:
    """The ``_CollectionView`` of an exact ``list``, ``tuple`` or ``dict``.

    Entries are read with the built-in type's own methods (``list.__getitem__``,
    ``tuple.__getitem__``, ``dict.items``), so no user code runs. The view's
    ``forward`` and ``backward`` are lazy: ``_container`` pulls only the
    entries it keeps.

    Args:
        value: a value whose type is exactly ``list``, ``tuple`` or ``dict``.

    Returns:
        A keyed view (``type_name`` ``"dict"``) for a dict, else an unkeyed
        one named ``"list"`` or ``"tuple"``, both readable from either end.
    """
    if type(value) is dict:
        mapping = value
        return _CollectionView(
            "dict",
            dict.__len__(mapping),
            True,
            lambda: iter(dict.items(mapping)),
            lambda: reversed(dict.items(mapping)),
        )
    items = cast(Any, value)
    get = list.__getitem__ if type(value) is list else tuple.__getitem__
    length = len(items)
    return _CollectionView(
        type(value).__name__,
        length,
        False,
        lambda: (get(items, index) for index in range(length)),
        lambda: (get(items, index) for index in range(length - 1, -1, -1)),
    )


def _container(
    entries: Iterable[Any],
    *,
    length: int,
    type_name: str,
    keyed: bool,
    max_size: int,
    depth: int,
    path: str,
    state: _State,
    head_n: int,
    tail_n: int = 0,
    backwards: Callable[[], Iterable[Any]] | None = None,
    budget_tail: bool = True,
) -> _Built:
    """A ``Sequence`` of ``length`` items, or (``keyed``) a ``Mapping`` of ``(key, value)`` entries.

    Keeps the first ``head_n`` (at most ``length``) entries pulled from
    ``entries`` and, when every one of them fit, the last ``tail_n`` from
    ``backwards()`` (the entries from the last one back), under the shared
    budget. Entries may be private ``_Part`` builders (see ``_build_slot``),
    or ``_Omitted`` notes. A mapping entry also costs one node for its key,
    checked before the pair is pulled. A mapping stops with ``"key"`` at a
    key with no JSON form or one whose JSON form an earlier kept key has
    (``1`` and ``"1"``), so the snapshot renders as it was sized.

    With ``Limits.tail_share`` above 0, a ``backwards`` source and
    ``budget_tail`` (not for an ndarray's planned selection), a budget cut
    keeps both ends too. The head is filled as without it; when the budget
    stops it, its last entries are dropped until it fits the budget less
    ``tail_share`` of it, and the tail is filled from the end with what is
    left (``tail_n``, when ``max_items`` cut, also bounds it). So a
    container that fits is kept as it is, and every entry is built once.
    ``tail_share`` 0 keeps a head only, as #342 does.
    """
    fill = _Fill(Container(max_size), keyed, depth, path, state)
    head, omitted = fill.head(entries, head_n)
    share = state.limits.tail_share
    if budget_tail and backwards is not None and share > 0 and fill.box.reason == "max_length":
        # The budget cut the head: give the end its share of the budget.
        fill.shorten_head(head, max_size - int((max_size - 2) * share))
        tail_n = length - len(head) if tail_n == 0 else tail_n
    tail_n = min(tail_n, length - len(head))
    tail: list[Any] = []
    if fill.box.reason is None and tail_n and backwards is not None:
        tail = fill.tail(backwards(), tail_n, length)
    if omitted is None and len(head) + len(tail) == length:
        head, tail = head + tail, []  # the two ends met: everything is kept
    reason = fill.box.reason or omitted
    if reason is None and len(head) + len(tail) < length:
        reason = "max_items"
    node_type = Mapping if keyed else Sequence
    return _Built(node_type(type_name, length, tuple(head), reason, tuple(tail)), fill.box.size)


class _Fill:
    """One container being filled: its budget (``box``), the JSON keys kept, the head's costs.

    Each entry is built once, by ``add``. ``head`` fills from the start;
    ``shorten_head`` gives back the last head entries to make room for a
    tail, and ``tail`` fills from the end.
    """

    def __init__(self, box: Container, keyed: bool, depth: int, path: str, state: _State) -> None:
        self.box = box
        self.keyed = keyed
        self.depth = depth
        self.path = path
        self.state = state
        self.seen: set[str] = set()  # the JSON keys kept so far (our own exact ``str``)
        # Each head entry's share of ``box.size``, and its JSON key (a mapping's).
        self.head_costs: list[tuple[int, str | None]] = []

    def head(self, entries: Iterable[Any], head_n: int) -> tuple[list[Any], Reason | None]:
        """Up to ``head_n`` entries from the start, and the reason of the first ``_Omitted`` note."""
        head: list[Any] = []
        omitted: Reason | None = None
        iterator = iter(entries)
        while len(head) < head_n:
            if self.keyed and self.state.nodes_left <= 0:
                self.box.stop("max_nodes")
                break
            try:
                entry = next(iterator)
            except StopIteration:
                break
            if type(entry) is _Omitted:
                omitted = omitted or entry.reason
                continue
            if not self.add(entry, len(head), head, in_head=True):
                break
        return head, omitted

    def shorten_head(self, head: list[Any], limit: int) -> None:
        """Drop the last head entries until the box is at most ``limit``; the box may go on."""
        while head and self.box.size > limit:
            head.pop()
            cost, segment = self.head_costs.pop()
            self.box.pop(cost)
            if segment is not None:
                self.seen.discard(segment)
        self.box.resume()

    def tail(self, backwards: Iterable[Any], tail_n: int, length: int) -> list[Any]:
        """Up to ``tail_n`` entries from the end (``backwards``: the last one first), in order."""
        tail: list[Any] = []
        for offset, entry in enumerate(backwards):
            if offset >= tail_n or not self.add(entry, length - 1 - offset, tail, in_head=False):
                break
        tail.reverse()
        return tail

    def add(self, entry: Any, index: int, kept: list[Any], *, in_head: bool) -> bool:
        """Build ``entry`` into the room left and append it to ``kept``; ``False``: the box stopped."""
        box, state = self.box, self.state
        if state.nodes_left <= 0:
            box.stop("max_nodes")
            return False
        key: Any = None
        if self.keyed:
            key, child = entry
            state.take_node()
            segment = json_key(key, live=True)
            if segment is None:
                key = _key_node(key, state)  # a tuple, an enum member...: a node key
                segment = key_text(key) if key is not None else None
            if segment is None or segment in self.seen:
                # No JSON key, or one that collides in JSON (``1`` and ``"1"``):
                # stop here, as ``render_json`` would, keeping the first.
                box.stop("key")
                return False
            self.seen.add(segment)
        else:
            child, segment = entry, str(index)
        slot = box.slot(segment if self.keyed else None)
        if slot is None:
            return False
        overhead, available = slot
        built = _build_slot(
            child, available, self.depth + 1, pointer_child(self.path, segment), state
        )
        kept.append((cast(Key, key), built.node) if self.keyed else built.node)
        cost = (2 if box.count else 0) + overhead + built.size
        box.commit(overhead, built.size)
        if in_head:
            self.head_costs.append((cost, segment if self.keyed else None))
        return True


_KEY_LIMITS = Limits(max_chars=400, max_nodes=50, max_depth=2, max_items=8, max_string=50)
"""The bounds of a node key: a key is a label, so it is kept small."""


def _key_node(key: Any, state: _State) -> Snapshot | None:
    """A key that has no JSON form, captured small (``Limits.fallback`` as the mapping's).

    ``None`` (the mapping stops with ``"key"``) when no inspection work is
    left, for an ``int`` or ``float`` key, and for a key capture cannot read (an
    ``Unavailable`` label says nothing). The key's own capture is not
    charged to the mapping's budget beyond the one node of the key.
    """
    if state.nodes_left <= 0 or type(key) is int or type(key) is float:
        return None  # a number json_key refused (a huge int, nan) stays refused
    limits = dataclasses.replace(_KEY_LIMITS, fallback=state.limits.fallback)
    node = _build(key, limits.max_chars, 0, "", _State(limits)).node
    return None if type(node) is Unavailable else node


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
    if type(child) is _Scoped:
        return _build_scoped(child, max_size, depth, path, state)
    return _build(child, max_size, depth, path, state)


def _scoped_limits(outer: Limits, limits: dict[str, Any]) -> Limits:
    """``outer`` with a field's ``spec()`` ``max_string`` / ``max_length`` (as ``max_items``)."""
    changes: dict[str, Any] = {}
    if "max_string" in limits:
        changes["max_string"] = limits["max_string"]
    if "max_length" in limits:
        changes["max_items"] = limits["max_length"]
    return dataclasses.replace(outer, **changes)


def _build_scoped(child: _Scoped, max_size: int, depth: int, path: str, state: _State) -> _Built:
    """A field with its own ``spec(max_string=..., max_length=...)``: built under them, subtree-wide."""
    outer = state.limits
    state.limits = _scoped_limits(outer, child.limits)
    try:
        return _build(child.value, max_size, depth, path, state)
    finally:
        state.limits = outer


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
    length: int | None = None,
) -> _Built:
    """Emit the (name, value) fields of an object; a reader's ``_FailClosed`` keeps none.

    ``length``: a plain object's attribute count, kept on the node when it is cut.

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
    node = Object(type_name, tuple(fields), reason, kind, length if reason is not None else None)
    return _Built(node, box.size)


def _min_size(child: Any) -> int:
    """The size of the minimal type-oriented fallback a captured value can always keep.

    Only identity checks against exact built-in types: the type of an unknown
    value is never compared, hashed or asked for attributes (a metaclass
    ``__eq__`` must not run).
    """
    if type(child) is _Part:
        return 2
    if type(child) is _Scoped:
        child = child.value
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
    return _container(
        items,
        length=length,
        type_name=type_name,
        keyed=shape.keyed,
        max_size=max_size,
        depth=depth,
        path=path,
        state=state,
        head_n=head_n,
    )


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
    mro: tuple[type[Any], ...],
    type_name: str,
    max_size: int,
    depth: int,
    path: str,
    state: _State,
) -> _Built:
    """A registered shape's ``Object``: its accessor fields, then its structure.

    A ``spec(cls, name, ...)`` override on the class applies to the accessor
    field ``name``: ``hidden=True`` leaves it out (its accessor never runs),
    ``max_string`` / ``max_length`` bound its subtree.
    """
    overrides = _class_overrides(mro, state) if shape.fields else {}
    entries: list[tuple[str, Any]] = []
    for name, function in shape.fields:
        flags = overrides.get(name, {})
        if flags.get("hidden"):
            continue
        part = _lazy(functools.partial(function, value), state)
        entries.append((name, _scoped_part(part, flags, state)))
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


def _class_overrides(mro: tuple[type[Any], ...], state: _State) -> dict[str, dict[str, Any]]:
    """The ``spec(cls, name, ...)`` overrides of the classes of ``mro``; none when unreadable."""
    try:
        return _visibility_overrides({}, mro, state)
    except _FailClosed:
        return {}


def _scoped_part(part: _Part, flags: dict[str, Any], state: _State) -> _Part:
    """``part`` built under a field's own ``max_string`` / ``max_length`` (when it has them)."""
    limits = {key: flags[key] for key in ("max_string", "max_length") if key in flags}
    if not limits:
        return part

    def build(max_size: int, depth: int, path: str) -> _Built:
        outer = state.limits
        state.limits = _scoped_limits(outer, limits)
        try:
            return part.build(max_size, depth, path)
        finally:
            state.limits = outer

    return _Part(build, part.min_nodes, part.exhausted)


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
        length=length,
        type_name="list",
        keyed=False,
        max_size=max_size,
        depth=depth,
        path=path,
        state=state,
        head_n=head_n,
        tail_n=tail_n,
        backwards=lambda: (child(i) for i in range(length - 1, length - tail_n - 1, -1)),
        budget_tail=False,  # the selection already fits both ends to the budget
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
            return _container(
                labels,
                length=count,
                type_name="dict",
                keyed=True,
                max_size=slot,
                depth=depth,
                path=path,
                state=state,
                head_n=count,
            )

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
                lambda: _shape_fields(
                    accessors, value, mro, type_name, max_size, depth, path, state
                ),
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
        return _build_integer(value, max_size, depth, path, state)
    if value_type is float:
        if not math.isfinite(value):
            return _opaque(value, max_size, depth, path, state)
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
        return _build_view(_builtin_collection(value), value, max_size, depth, path, state)

    if max_size < 2:
        return _fallback(value, "max_length", max_size)
    view = _exact_view(value, state)
    if view is not None:
        return _build_view(view, value, max_size, depth, path, state)
    mro = _safe_mro(value, state)
    if mro is None:
        return _opaque(value, max_size, depth, path, state)
    found = _find_shape(mro, state)
    if found is not None:
        carrier, shape = found
        return _build_shape(shape, value, mro, carrier, max_size, depth, path, state)
    view = _mro_view(value, mro, state)
    if view is not None:
        return _build_view(view, value, max_size, depth, path, state)
    if limits.fallback == "repr" and _defines_repr(mro, state):
        # A deliberate ``__repr__`` beats the attribute reader in model-facing
        # previews (``<Response [200]>``); traces never run it.
        built = _repr_object(value, max_size, depth, path, state)
        if built is not None:
            return built
    plain = _plain_entries(value, mro, state)
    if plain is not None:
        entries, count = plain
        type_name = _bounded_type_name(value, _MAX_TYPE_NAME)
        return _enter(
            value,
            "object",
            Object(type_name, (), "max_depth"),
            depth,
            path,
            state,
            lambda: _object_fields(
                type_name, iter(entries), max_size, depth, path, state, length=count
            ),
        )
    return _opaque(value, max_size, depth, path, state)


def _build_view(
    view: _View, value: Any, max_size: int, depth: int, path: str, state: _State
) -> _Built:
    """Build a standard type's view (``_std``) with the usual builders."""
    match view:
        case _TextView():
            return _cut_text(view.length, view.read, "chars", max_size, state.limits)
        case _CollectionView():
            collection = view
            has_tail = collection.backward is not None
            head_n, tail_n = _plan(collection.length, state.limits, has_tail=has_tail)
            node_type = Mapping if collection.keyed else Sequence
            return _enter(
                value,
                "mapping" if collection.keyed else "sequence",
                node_type(collection.type_name, collection.length, (), "max_depth"),
                depth,
                path,
                state,
                lambda: _container(
                    collection.forward(),
                    length=collection.length,
                    type_name=collection.type_name,
                    keyed=collection.keyed,
                    max_size=max_size,
                    depth=depth,
                    path=path,
                    state=state,
                    head_n=head_n,
                    tail_n=tail_n,
                    backwards=collection.backward,
                ),
            )
        case _ObjectView():
            record = view
            capped = record.kind is None  # a namedtuple's fields are its items

            def build() -> _Built:
                return _object_fields(
                    record.type_name,
                    iter(record.entries),
                    max_size,
                    depth,
                    path,
                    state,
                    record.kind,
                    capped=capped,
                )

            if not record.enter:
                return build()
            empty = Object(record.type_name, (), "max_depth", record.kind)
            return _enter(value, "object", empty, depth, path, state, build)
