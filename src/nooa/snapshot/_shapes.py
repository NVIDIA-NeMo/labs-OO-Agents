# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The five public shape constructors and the shape classes they make.

``array_like``, ``mapping_like``, ``ndarray_like``, ``object_like`` and
``text_like`` validate and copy their arguments into frozen shape classes;
``_FieldReader`` is the internal shape of Pydantic models and dataclasses.
No walking: ``_walk`` builds snapshots from these classes.
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
    """``array_like``, or (``keyed``) ``mapping_like``, whose items are (key, value) pairs."""

    length: Callable[[Any], int]
    items: Callable[[Any], Iterable[Any]]
    fields: tuple[tuple[str, Callable[[Any], Any]], ...]
    kind: str | None
    keyed: bool


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


_Shape = _Collection | _NdarrayLike | _ObjectLike | _TextLike
_SHAPE_TYPES: tuple[type[Any], ...] = (_Collection, _NdarrayLike, _ObjectLike, _TextLike)


def array_like(
    length: Callable[[Any], int],
    items: Callable[[Any], Iterable[Any]],
    *,
    fields: dict[str, Callable[[Any], Any]] | None = None,
    kind: str | None = None,
) -> _Collection:
    """A shape for sequence-like values: ``length(v)`` items, pulled lazily from ``items(v)``.

    ``length(v)`` is called once; ``items(v)`` is iterated only as far as the
    kept items (``max_items``, the budget), plus at most one. ``fields``: name
    -> ``accessor(v)``, each called once, only when its field is kept. Without
    fields the value becomes a ``Sequence``; with fields an ``Object`` whose
    fields are followed by ``items`` (that ``Sequence``).
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
    """A shape for mapping-like values: ``items(v)`` yields ``(key, value)`` pairs lazily.

    As ``array_like``: a ``Mapping``, or with ``fields`` an ``Object`` whose
    fields are followed by ``items`` (that ``Mapping``). Keys must be exact
    ``str``, ``int``, ``float``, ``bool`` or ``None``, as for a ``dict``.
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
