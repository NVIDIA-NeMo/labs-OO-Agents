# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The five public shape constructors and the private shape classes they return.

``array_like``, ``mapping_like``, ``ndarray_like``, ``object_like`` and
``text_like`` check and copy their arguments into frozen shape objects, the
only things ``capture.register`` accepts. ``_FieldReader`` is the internal
shape of Pydantic models and dataclasses. Nothing here reads a captured
value: ``_walk`` builds the snapshot nodes from these shapes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal, cast

from nooa.snapshot._limits import _State
from nooa.snapshot._raw import _MAX_TYPE_NAME, _TYPE_MRO, _bounded_type_name
from nooa.snapshot._visibility import _dataclass_entries, _Fields, _pydantic_entries

# --- shapes: the extension API -------------------------------------------------------
#
# A shape maps out the structure of a type (its length, items, fields, text
# slices) through accessors; capture builds the snapshot itself, with the same
# builders and size rules as the built-in types. Accessors are trusted
# extension code: they run only when capture decides it needs their result.


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
class _Collection:
    """The shape ``array_like`` or (``keyed``) ``mapping_like`` returns; pass it to ``capture.register``."""

    length: Callable[[Any], int]
    items: Callable[[Any], Iterable[Any]]
    fields: tuple[tuple[str, Callable[[Any], Any]], ...]
    kind: str | None
    keyed: bool


@dataclass(frozen=True)
class _NdarrayLike:
    """The shape ``ndarray_like`` returns; pass it to ``capture.register``."""

    shape: Callable[[Any], tuple[int, ...]]
    element: Callable[[Any, tuple[int, ...]], Any]
    fields: tuple[tuple[str, Callable[[Any], Any]], ...]
    labels: tuple[tuple[int, Callable[[Any], Any]], ...]
    kind: str | None


@dataclass(frozen=True)
class _ObjectLike:
    """The shape ``object_like`` returns; pass it to ``capture.register``."""

    fields: tuple[tuple[str, Callable[[Any], Any]], ...]
    kind: str | None


@dataclass(frozen=True)
class _TextLike:
    """The shape ``text_like`` returns; pass it to ``capture.register``."""

    length: Callable[[Any], int]
    slice: Callable[[Any, int, int], str | bytes]
    unit: Literal["chars", "bytes"]
    kind: str | None


_Shape = _Collection | _NdarrayLike | _ObjectLike | _TextLike
_SHAPE_TYPES: tuple[type[Any], ...] = (_Collection, _NdarrayLike, _ObjectLike, _TextLike)


def array_like(
    length: Callable[[Any], int],
    items: Callable[[Any], Iterable[Any]],
    *,
    fields: dict[str, Callable[[Any], Any]] | None = None,
    kind: str | None = None,
) -> _Collection:
    """A shape for sequence-like values, for ``capture.register``: a length and lazy items.

    Arguments:

    - ``length``: ``length(v) -> int``, called once; it must return a
      non-negative exact ``int``.
    - ``items``: ``items(v) -> iterable``, iterated only as far as the kept
      items (``Limits.max_items`` and the budget) plus at most one. Only a
      head is kept: the iterable is read forwards.
    - ``fields`` (default ``None``): an ordered ``dict`` of field name ->
      ``accessor(v)``. Names must be ``str`` (a subclass is copied) and
      not taken by the shape's own parts. Each accessor is called at most
      once, only when its field is kept. The first ``Limits.max_items``
      fields are kept; each field leaves room for the minimal value of
      every later one.
    - ``kind`` (default ``None``): a ``str`` stored as ``Object.kind``; it
      selects a ``render_python`` hook. Only an ``Object`` (a shape with
      fields, or ``ndarray_like`` / ``object_like``) keeps it.

    The value becomes a ``Sequence`` (``type_name``: the class name) or,
    with ``fields``, an ``Object`` whose fields are followed by ``items``,
    that ``Sequence`` (``type_name="list"``).

    Errors: when ``length`` raises or returns something else, the whole
    value is ``Unavailable("error")``. When ``items`` raises on the call,
    the ``items`` part is ``Unavailable("error")`` with fields, the whole
    value without; when it raises while being iterated, the items end there
    and the ``Sequence`` gets reason ``"error"``. A field accessor that
    raises makes its field ``Unavailable("error")``.

    Accessors are trusted code; they run only when capture keeps their
    result. What an accessor returns is captured like any value, under the
    same budget: a list is a ``Sequence``, a registered type uses its own
    shape, an unknown type is ``Unavailable("opaque")``, and ``v`` itself is
    a ``Ref`` (a cycle).

    Raises ``TypeError`` when an accessor is not callable, ``fields`` is not
    a ``dict``, or a name or ``kind`` is not a ``str``.

    >>> from nooa.snapshot import Limits, array_like, capture, render_python
    >>> class Bag:
    ...     def __init__(self, items):
    ...         self.items = items
    >>> capture.register(Bag, array_like(lambda b: len(b.items), lambda b: iter(b.items)))
    >>> render_python(capture(Bag(list(range(50))), Limits(max_items=3)))
    'Bag(len=50, [:3]=[0, 1, 2])'
    >>> capture.unregister(Bag)
    """
    return _Collection(
        _accessor(length, "length"),
        _accessor(items, "items"),
        _accessor_fields(fields, "array"),
        _kind(kind),
        keyed=False,
    )


def mapping_like(
    length: Callable[[Any], int],
    items: Callable[[Any], Iterable[tuple[Any, Any]]],
    *,
    fields: dict[str, Callable[[Any], Any]] | None = None,
    kind: str | None = None,
) -> _Collection:
    """A shape for mapping-like values, for ``capture.register``: a length and lazy pairs.

    Arguments:

    - ``length``: ``length(v) -> int``, called once (a non-negative exact
      ``int``).
    - ``items``: ``items(v)`` yields ``(key, value)`` pairs (exact 2-tuples),
      read forwards and only as far as the kept entries plus at most one.
      Keys are kept as for a ``dict``: exact ``str``, ``int`` (at most 128
      bits), finite ``float``, ``bool`` or ``None``; another key, or one
      whose JSON form an earlier kept key has, stops the mapping with
      reason ``"key"``.
    - ``fields`` (default ``None``): an ordered ``dict`` of field name ->
      ``accessor(v)``. Names must be ``str`` (a subclass is copied) and
      not taken by the shape's own parts. Each accessor is called at most
      once, only when its field is kept. The first ``Limits.max_items``
      fields are kept; each field leaves room for the minimal value of
      every later one.
    - ``kind`` (default ``None``): a ``str`` stored as ``Object.kind``; it
      selects a ``render_python`` hook. Only an ``Object`` (a shape with
      fields, or ``ndarray_like`` / ``object_like``) keeps it.

    The value becomes a ``Mapping`` (``type_name``: the class name) or,
    with ``fields``, an ``Object`` whose fields are followed by ``items``,
    that ``Mapping`` (``type_name="dict"``). Errors are handled as for
    ``array_like``; a yielded item that is not a pair also ends the items
    with reason ``"error"``.

    Accessors are trusted code; they run only when capture keeps their
    result. What an accessor returns is captured like any value, under the
    same budget: a list is a ``Sequence``, a registered type uses its own
    shape, an unknown type is ``Unavailable("opaque")``, and ``v`` itself is
    a ``Ref`` (a cycle).

    Raises ``TypeError`` as ``array_like`` does.
    """
    return _Collection(
        _accessor(length, "length"),
        _accessor(items, "items"),
        _accessor_fields(fields, "mapping"),
        _kind(kind),
        keyed=True,
    )


def ndarray_like(
    shape: Callable[[Any], tuple[int, ...]],
    element: Callable[[Any, tuple[int, ...]], Any],
    *,
    fields: dict[str, Callable[[Any], Any]] | None = None,
    labels: dict[int, Callable[[Any], Any]] | None = None,
    kind: str | None = None,
) -> _NdarrayLike:
    """A shape for n-dimensional arrays, for ``capture.register``: a shape tuple and elements.

    Arguments:

    - ``shape``: ``shape(v) -> tuple[int, ...]``, called once (a tuple of
      non-negative exact ``int``).
    - ``element``: ``element(v, index)``, with ``index`` a tuple of one
      ``int`` per axis; called only for the elements kept.
    - ``labels`` (default ``None``): a ``dict`` of axis (a non-negative
      ``int``) -> ``accessor(v)`` returning that axis's labels (DataFrame
      columns, an index). Only the shown labels are read, as
      ``accessor(v)[i]``.
    - ``fields`` (default ``None``): an ordered ``dict`` of field name ->
      ``accessor(v)``. Names must be ``str`` (a subclass is copied) and
      not taken by the shape's own parts. Each accessor is called at most
      once, only when its field is kept. The first ``Limits.max_items``
      fields are kept; each field leaves room for the minimal value of
      every later one.
    - ``kind`` (default ``None``): a ``str`` stored as ``Object.kind``; it
      selects a ``render_python`` hook. Only an ``Object`` (a shape with
      fields, or ``ndarray_like`` / ``object_like``) keeps it.

    The value becomes an ``Object``: its ``fields``, then ``shape`` (the
    tuple), ``labels`` (a ``Mapping`` axis -> ``Sequence`` of the shown
    labels, when given) and ``data``: nested ``Sequence`` nodes, one level
    per axis, each with its axis's length. A 0-d array's ``data`` is its one
    element.

    Selection, like numpy's summarised print: along every axis, the first
    and last entries, ``K`` at most (all of a shorter axis), the start
    taking the odd one; ``K = 4`` shows indices ``0, 1, n-2, n-1``. ``K``
    is ``Limits.max_items`` when set, and never more than the budget
    affords: the shown elements and labels (6 characters each) and the
    brackets fit three quarters of the value's characters, and the elements
    and levels three quarters of the inspection work left. ``tail_share`` is
    ignored: both ends are always shown. Elements larger than assumed are
    still cut by the budget, from the end.

    Errors: when ``shape`` raises or returns something else, the whole
    value is ``Unavailable("error")``. An ``element``, a label or a field
    accessor that raises makes only that part ``Unavailable("error")``; so
    does a ``labels`` axis the shape does not have.

    Accessors are trusted code; they run only when capture keeps their
    result. What an accessor returns is captured like any value, under the
    same budget: a list is a ``Sequence``, a registered type uses its own
    shape, an unknown type is ``Unavailable("opaque")``, and ``v`` itself is
    a ``Ref`` (a cycle).

    Raises ``TypeError`` when an accessor is not callable, ``fields`` or
    ``labels`` is not a ``dict``, an axis is not a non-negative ``int``, or
    a name or ``kind`` is not a ``str``.
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
    """A shape for record-like values, for ``capture.register``: named fields.

    Arguments:

    - ``fields``: an ordered ``dict`` of field name -> ``accessor(v)``.
      Names must be ``str`` (a subclass is copied). Each accessor is called
      at most once, only when its field is kept; the first
      ``Limits.max_items`` fields are kept, and each leaves room for the
      minimal value of every later one.
    - ``kind`` (default ``None``): a ``str`` stored as ``Object.kind``; it
      selects a ``render_python`` hook.

    The value becomes an ``Object`` (``type_name``: the class name). A field
    accessor that raises makes its field ``Unavailable("error")``.

    Accessors are trusted code; they run only when capture keeps their
    result. What an accessor returns is captured like any value, under the
    same budget: a list is a ``Sequence``, a registered type uses its own
    shape, an unknown type is ``Unavailable("opaque")``, and ``v`` itself is
    a ``Ref`` (a cycle).

    Raises ``TypeError`` when ``fields`` is not a ``dict``, an accessor is
    not callable, or a name or ``kind`` is not a ``str``.
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
    """A shape for text too large to copy, for ``capture.register``: a length and slices.

    Arguments:

    - ``length``: ``length(v) -> int``, the size in ``unit``, called once (a
      non-negative exact ``int``).
    - ``slice``: ``slice(v, start, stop)``, returning exactly ``stop -
      start`` units: a ``str`` for ``"chars"`` (a subclass is copied), or
      ``bytes`` for ``"bytes"`` (a ``bytearray`` or ``memoryview`` is
      copied).
    - ``unit`` (default ``"chars"``): ``"chars"`` or ``"bytes"``.
    - ``kind`` (default ``None``): checked and stored on the shape; a
      ``Text`` has no ``kind``, so no hook uses it.

    The value becomes a ``Text`` with the right ``length``, ``omitted`` and
    ``unit``, cut exactly as a ``str`` or ``bytes`` of that content would
    be (UTF-8-safe for bytes). ``slice`` is called at most twice, only for
    the head and the tail kept, never for the whole of a text that is cut;
    for bytes, the head read takes one byte more, to see whether the head
    ends on a code point boundary. A text short enough to be kept whole is
    read in one call.

    Errors: there is no partial result. When ``length`` or ``slice`` raises,
    or a slice has the wrong type or size, the whole value is
    ``Unavailable("error")``.

    Raises ``TypeError`` when an accessor is not callable, ``unit`` is not
    ``"chars"`` or ``"bytes"``, or ``kind`` is not a ``str``.

    >>> from nooa.snapshot import Limits, capture, render_python, text_like
    >>> class Rope:
    ...     def __init__(self, text):
    ...         self.text = text
    >>> capture.register(Rope, text_like(lambda r: len(r.text), lambda r, a, b: r.text[a:b]))
    >>> render_python(capture(Rope("y" * 100), Limits(max_string=6, tail_share=0.5)))
    "str(len=100, [:3]='yyy', [-3:]='yyy')"
    >>> capture.unregister(Rope)
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
