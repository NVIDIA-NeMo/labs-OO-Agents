# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The standard types capture reads natively, and how it reads them: no user code runs.

What lives here: ``_exact_view`` (values whose exact type is one of a fixed
table of standard types) and ``_mro_view`` (values whose raw MRO contains
``enum.Enum``, ``BaseException``, ``type``, a namedtuple's ``tuple``, or a
built-in container that no class above it overrides). Each returns a *view*:
what the value holds, as built-in values and children to capture, or
``None``. ``_walk`` builds the nodes from a view with its usual builders.

Every read is one of:

- a C-level descriptor of the standard type itself, taken from that type's
  ``__dict__`` and called on a value whose type was checked by identity
  (``range.start``, ``datetime.tzinfo``, ``BaseException.args``,
  ``FunctionType.__code__``, ``re.Pattern.pattern``...);
- a C-level method of an exact standard type (``datetime.isoformat`` with
  no tzinfo or a ``datetime.timezone``, ``Decimal.__str__``,
  ``dict.items``, ``set.__iter__``);
- for exact ``pathlib`` paths only, ``PurePath.__str__``: standard-library
  Python code on a value whose type is exactly a ``pathlib`` class;
- raw class namespaces and instance storage, through ``_raw``.

A value that does not meet a view's checks gets no view (``None``); the
caller then tries the next reader.
"""

from __future__ import annotations

import array
import collections
import datetime
import decimal
import enum
import pathlib
import re
import types
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, cast

from nooa.snapshot._limits import _State
from nooa.snapshot._raw import (
    _MAX_TYPE_NAME,
    _TYPE_NAME,
    _bounded_type_name,
    _copy_storage,
    _FailClosed,
    _instance_storage,
    _raw_namespace,
    _safe_dict,
)

_TYPE_QUALNAME = type.__dict__["__qualname__"]
_FUNCTION_CODE = types.FunctionType.__dict__["__code__"]
_FUNCTION_DEFAULTS = types.FunctionType.__dict__["__defaults__"]
_FUNCTION_KWDEFAULTS = types.FunctionType.__dict__["__kwdefaults__"]
_FUNCTION_QUALNAME = types.FunctionType.__dict__["__qualname__"]
_FUNCTION_MODULE = types.FunctionType.__dict__["__module__"]
_BUILTIN_QUALNAME = types.BuiltinFunctionType.__dict__["__qualname__"]
_METHOD_FUNC = types.MethodType.__dict__["__func__"]
_MODULE_DICT: Any = cast(Any, types.ModuleType).__dict__["__dict__"]
_EXCEPTION_ARGS = BaseException.__dict__["args"]
_PATTERN_PATTERN: Any = re.Pattern.__dict__["pattern"]
_PATTERN_FLAGS = re.Pattern.__dict__["flags"]
_UUID_INT = uuid.UUID.__dict__["int"]
_DATETIME_TZINFO = datetime.datetime.__dict__["tzinfo"]
_TIME_TZINFO = datetime.time.__dict__["tzinfo"]
_PATH_TYPES = (
    pathlib.PurePosixPath,
    pathlib.PureWindowsPath,
    pathlib.PosixPath,
    pathlib.WindowsPath,
)
_VARARGS = 0x04
_VARKEYWORDS = 0x08


@dataclass(frozen=True)
class _ObjectView:
    """An ``Object`` of ``entries`` (name -> value to capture), with ``kind``.

    ``enter``: the entries hold user values, so the object counts as a
    container for ``max_depth`` and cycles.
    """

    type_name: str
    kind: str | None
    entries: list[tuple[str, Any]]
    enter: bool = False


@dataclass(frozen=True)
class _CollectionView:
    """A ``Sequence`` (or, ``keyed``, a ``Mapping``) read natively.

    ``forward()`` yields the items (pairs when keyed) from the start;
    ``backward()`` from the end, or ``None`` for an unordered source.
    """

    type_name: str
    length: int
    keyed: bool
    forward: Callable[[], Iterator[Any]]
    backward: Callable[[], Iterator[Any]] | None


@dataclass(frozen=True)
class _TextView:
    """The characters of a ``str`` subclass, read by native slices."""

    length: int
    read: Callable[[int, int], str]


_View = _ObjectView | _CollectionView | _TextView


def _guard(iterator: Iterator[Any]) -> Iterator[Any]:
    """Native iteration that stops quietly when the container changes size under it."""
    try:
        yield from iterator
    except RuntimeError:
        return


# --- names --------------------------------------------------------------------------


def _class_name(cls: Any, state: _State) -> str:
    """``module.QualName`` of a class from raw reads (``builtins`` is left out)."""
    try:
        qualname = _TYPE_QUALNAME.__get__(cls, type(cls))
    except Exception:
        qualname = None
    if type(qualname) is not str:
        try:
            qualname = _TYPE_NAME.__get__(cls, type(cls))
        except Exception:
            qualname = "object"
        if type(qualname) is not str:
            qualname = "object"
    namespace = _raw_namespace(cls, state)
    module = dict.get(namespace, "__module__") if namespace is not None else None
    if type(module) is not str or module == "builtins":
        return qualname[:_MAX_TYPE_NAME]
    return f"{module}.{qualname}"[:_MAX_TYPE_NAME]


def _qualified_type_name(value: Any, state: _State) -> str:
    """The module-qualified name of ``type(value)``: the opaque label's name."""
    return _class_name(type(value), state)


# --- functions ----------------------------------------------------------------------


def _default_text(value: Any, state: _State) -> str:
    """A parameter default, captured small and rendered (no user code)."""
    from nooa.snapshot._capture import capture
    from nooa.snapshot._limits import Limits
    from nooa.snapshot._render import render_python

    limits = Limits(max_chars=400, max_nodes=50, max_depth=1, max_items=3, max_string=40)
    return render_python(capture(value, limits))


def _signature(function: types.FunctionType, state: _State, skip_first: bool) -> str:
    """``(a, b=2, *args, c, **kw)`` from the code object and the defaults: no ``inspect``."""
    code = _FUNCTION_CODE.__get__(function, types.FunctionType)
    if type(code) is not types.CodeType:
        return "(...)"
    names = code.co_varnames
    positional = code.co_argcount
    positional_only = code.co_posonlyargcount
    keyword_only = code.co_kwonlyargcount
    flags = code.co_flags
    defaults = _FUNCTION_DEFAULTS.__get__(function, types.FunctionType) or ()
    kwdefaults = _FUNCTION_KWDEFAULTS.__get__(function, types.FunctionType) or {}
    if type(defaults) is not tuple or type(kwdefaults) is not dict:
        return "(...)"
    params: list[str] = []
    first_default = positional - len(defaults)
    for index in range(positional):
        name = names[index]
        if index >= first_default:
            name += "=" + _default_text(defaults[index - first_default], state)
        params.append(name)
        if positional_only and index == positional_only - 1:
            params.append("/")
    cursor = positional + keyword_only
    if flags & _VARARGS:
        params.append("*" + names[cursor])
        cursor += 1
    elif keyword_only:
        params.append("*")
    for index in range(positional, positional + keyword_only):
        name = names[index]
        if type(name) is str and name in kwdefaults:
            name += "=" + _default_text(dict.__getitem__(kwdefaults, name), state)
        params.append(name)
    if flags & _VARKEYWORDS:
        params.append("**" + names[cursor])
    if skip_first and params and params[0] not in ("/", "*") and not params[0].startswith("*"):
        params = params[1:]
        if params and params[0] == "/":
            params = params[1:]
    return "(" + ", ".join(params) + ")"


def _function_name(function: types.FunctionType) -> str:
    qualname = _FUNCTION_QUALNAME.__get__(function, types.FunctionType)
    module = _FUNCTION_MODULE.__get__(function, types.FunctionType)
    if type(qualname) is not str:
        qualname = "<function>"
    if type(module) is str and module not in ("builtins", "__main__"):
        return f"{module}.{qualname}"[:_MAX_TYPE_NAME]
    return qualname[:_MAX_TYPE_NAME]


def _function_view(value: Any, state: _State) -> _ObjectView:
    signature = _signature(value, state, skip_first=False)
    entries = [("name", _function_name(value)), ("signature", signature)]
    return _ObjectView("function", "python.function", entries)


def _method_view(value: Any, state: _State) -> _ObjectView | None:
    function = _METHOD_FUNC.__get__(value, types.MethodType)
    if type(function) is not types.FunctionType:
        return None
    signature = _signature(function, state, skip_first=True)
    entries = [("name", _function_name(function)), ("signature", signature)]
    return _ObjectView("method", "python.method", entries)


def _builtin_view(value: Any, state: _State) -> _ObjectView:
    name = _BUILTIN_QUALNAME.__get__(value, types.BuiltinFunctionType)
    if type(name) is not str:
        name = "<built-in>"
    return _ObjectView("builtin_function_or_method", "python.builtin", [("name", name)])


def _module_view(value: Any, state: _State) -> _ObjectView | None:
    try:
        raw = _MODULE_DICT.__get__(value, type(value))
    except Exception:
        return None
    namespace = _safe_dict(value, raw, state)
    name = dict.get(namespace, "__name__") if namespace is not None else None
    if type(name) is not str:
        return None
    return _ObjectView("module", "python.module", [("name", name[:_MAX_TYPE_NAME])])


# --- the exact standard types -------------------------------------------------------


def _datetime_view(value: Any, state: _State) -> _ObjectView | None:
    tzinfo = _DATETIME_TZINFO.__get__(value, datetime.datetime)
    if tzinfo is not None and type(tzinfo) is not datetime.timezone:
        return None  # another tzinfo's utcoffset() is its own code
    return _ObjectView("datetime", "python.datetime", [("iso", datetime.datetime.isoformat(value))])


def _time_view(value: Any, state: _State) -> _ObjectView | None:
    tzinfo = _TIME_TZINFO.__get__(value, datetime.time)
    if tzinfo is not None and type(tzinfo) is not datetime.timezone:
        return None
    return _ObjectView("time", "python.time", [("iso", datetime.time.isoformat(value))])


def _date_view(value: Any, state: _State) -> _ObjectView:
    return _ObjectView("date", "python.date", [("iso", datetime.date.isoformat(value))])


def _timedelta_view(value: Any, state: _State) -> _ObjectView:
    entries = [
        ("days", datetime.timedelta.__dict__["days"].__get__(value)),
        ("seconds", datetime.timedelta.__dict__["seconds"].__get__(value)),
        ("microseconds", datetime.timedelta.__dict__["microseconds"].__get__(value)),
    ]
    return _ObjectView("timedelta", "python.timedelta", entries)


def _decimal_view(value: Any, state: _State) -> _ObjectView:
    return _ObjectView("Decimal", "python.decimal", [("value", decimal.Decimal.__str__(value))])


def _uuid_view(value: Any, state: _State) -> _ObjectView | None:
    number = _UUID_INT.__get__(value, uuid.UUID)
    if type(number) is not int or not 0 <= number < 1 << 128:
        return None
    text = f"{number:032x}"
    hex_text = f"{text[:8]}-{text[8:12]}-{text[12:16]}-{text[16:20]}-{text[20:]}"
    return _ObjectView("UUID", "python.uuid", [("hex", hex_text)])


def _path_view(value: Any, state: _State) -> _ObjectView | None:
    text = pathlib.PurePath.__str__(value)
    if type(text) is not str:
        return None
    return _ObjectView(_bounded_type_name(value, _MAX_TYPE_NAME), "python.path", [("path", text)])


def _pattern_view(value: Any, state: _State) -> _ObjectView:
    entries = [
        ("pattern", _PATTERN_PATTERN.__get__(value, re.Pattern)),
        ("flags", _PATTERN_FLAGS.__get__(value, re.Pattern)),
    ]
    return _ObjectView("Pattern", "python.pattern", entries)


def _range_view(value: Any, state: _State) -> _ObjectView:
    entries = [("start", value.start), ("stop", value.stop), ("step", value.step)]
    return _ObjectView("range", "python.range", entries)


def _complex_view(value: Any, state: _State) -> _ObjectView:
    entries = [
        ("real", complex.__dict__["real"].__get__(value)),
        ("imag", complex.__dict__["imag"].__get__(value)),
    ]
    return _ObjectView("complex", "python.complex", entries)


def _set_view(value: Any, state: _State) -> _CollectionView:
    cls = type(value)
    length = cls.__len__(value)
    return _CollectionView(cls.__name__, length, False, lambda: _guard(cls.__iter__(value)), None)


_EXACT: tuple[tuple[type[Any], Callable[[Any, _State], _View | None]], ...] = (
    (set, _set_view),
    (frozenset, _set_view),
    (range, _range_view),
    (complex, _complex_view),
    (datetime.datetime, _datetime_view),
    (datetime.date, _date_view),
    (datetime.time, _time_view),
    (datetime.timedelta, _timedelta_view),
    (decimal.Decimal, _decimal_view),
    (uuid.UUID, _uuid_view),
    (re.Pattern, _pattern_view),
    (types.FunctionType, _function_view),
    (types.MethodType, _method_view),
    (types.BuiltinFunctionType, _builtin_view),
    (types.ModuleType, _module_view),
    *((path_type, _path_view) for path_type in _PATH_TYPES),
)


def _exact_view(value: Any, state: _State) -> _View | None:
    """The view of a value whose exact type is a standard type in the table, else ``None``."""
    value_type = type(value)
    for known, read in _EXACT:
        if value_type is known:
            try:
                return read(value, state)
            except Exception:
                return None
    return None


# --- standard bases in the MRO -------------------------------------------------------

_ITERATION_NAMES = frozenset(
    ("__iter__", "__reversed__", "__getitem__", "__len__", "items", "keys", "values")
)


def _ordered_dict_items(value: Any) -> Iterator[Any]:
    return _guard(iter(collections.OrderedDict.items(value)))


def _ordered_dict_backward(value: Any) -> Iterator[Any]:
    return _guard(reversed(collections.OrderedDict.items(value)))


def _container_view(
    value: Any, mro: tuple[type[Any], ...], native: type[Any], index: int, state: _State
) -> _View | None:
    """A container subclass read through its built-in base, when nothing above overrides it."""
    for base in mro[:index]:
        namespace = _raw_namespace(base, state)
        if namespace is None:
            return None
        if any(name in namespace for name in _ITERATION_NAMES):
            return None
    type_name = _bounded_type_name(value, _MAX_TYPE_NAME)
    if native is collections.OrderedDict:
        length = dict.__len__(value)
        return _CollectionView(
            type_name,
            length,
            True,
            lambda: _ordered_dict_items(value),
            lambda: _ordered_dict_backward(value),
        )
    if native is dict:
        return _CollectionView(
            type_name,
            dict.__len__(value),
            True,
            lambda: _guard(iter(dict.items(value))),
            lambda: _guard(reversed(dict.items(value))),
        )
    if native is list or native is tuple:
        length = native.__len__(value)
        get = native.__getitem__
        return _CollectionView(
            type_name,
            length,
            False,
            lambda: (get(value, i) for i in range(length)),
            lambda: (get(value, i) for i in range(length - 1, -1, -1)),
        )
    if native is set or native is frozenset:
        return _CollectionView(
            type_name, native.__len__(value), False, lambda: _guard(native.__iter__(value)), None
        )
    if native is collections.deque:
        deque = collections.deque
        return _CollectionView(
            type_name,
            deque.__len__(value),
            False,
            lambda: _guard(deque.__iter__(value)),
            lambda: _guard(deque.__reversed__(value)),
        )
    if native is array.array:
        length = array.array.__len__(value)
        get = array.array.__getitem__
        return _CollectionView(
            type_name,
            length,
            False,
            lambda: (get(value, i) for i in range(min(length, array.array.__len__(value)))),
            lambda: (
                get(value, i) for i in range(min(length, array.array.__len__(value)) - 1, -1, -1)
            ),
        )
    if native is str:
        return _TextView(
            str.__len__(value), lambda start, stop: str.__getitem__(value, slice(start, stop))
        )
    return None


_CONTAINER_BASES = (
    collections.OrderedDict,
    dict,
    list,
    tuple,
    set,
    frozenset,
    str,
    collections.deque,
    array.array,
)


def _namedtuple_fields(
    mro: tuple[type[Any], ...], index: int, state: _State
) -> tuple[str, ...] | None:
    for base in mro[:index]:
        namespace = _raw_namespace(base, state)
        if namespace is None:
            return None
        fields = dict.get(namespace, "_fields")
        if fields is not None:
            if type(fields) is tuple and all(type(name) is str for name in fields):
                return fields
            return None
    return None


def _enum_view(value: Any, mro: tuple[type[Any], ...], state: _State) -> _View | None:
    try:
        storage = _copy_storage(_instance_storage(value, mro, state), state)
    except _FailClosed:
        return None
    name = dict.get(storage, "_name_")
    if name is not None and type(name) is not str:
        return None
    if "_value_" not in storage:
        return None
    entries = [("name", name), ("value", dict.__getitem__(storage, "_value_"))]
    return _ObjectView(
        _bounded_type_name(value, _MAX_TYPE_NAME), "python.enum", entries, enter=True
    )


def _mro_view(value: Any, mro: tuple[type[Any], ...], state: _State) -> _View | None:
    """The view of a value with a standard base in its raw MRO (identity checks), else ``None``.

    ``enum.Enum`` anywhere in the MRO first; then in MRO order, the first of: ``BaseException`` (its
    ``args``); ``type`` (the value is a class); a namedtuple (a ``tuple``
    whose class declares ``_fields``); a built-in container (``dict``,
    ``OrderedDict``, ``list``, ``tuple``, ``set``, ``frozenset``, ``str``,
    ``collections.deque``, ``array.array``) that no class above it in the
    MRO overrides an iteration method of.
    """
    if any(base is enum.Enum for base in mro):  # before ``str`` / ``int`` (a StrEnum, an IntEnum)
        try:
            return _enum_view(value, mro, state)
        except Exception:
            return None
    for index, base in enumerate(mro):
        try:
            if base is BaseException:
                args = _EXCEPTION_ARGS.__get__(value, BaseException)
                if type(args) is not tuple:
                    return None
                type_name = _bounded_type_name(value, _MAX_TYPE_NAME)
                return _ObjectView(type_name, "python.exception", [("args", args)], enter=True)
            if base is type:
                return _ObjectView("type", "python.class", [("name", _class_name(value, state))])
            if base is tuple:
                fields = _namedtuple_fields(mro, index, state)
                if fields is not None and len(fields) == tuple.__len__(value):
                    type_name = _bounded_type_name(value, _MAX_TYPE_NAME)
                    entries = [(name, tuple.__getitem__(value, i)) for i, name in enumerate(fields)]
                    return _ObjectView(type_name, None, entries, enter=True)
            for native in _CONTAINER_BASES:
                if base is native:
                    return _container_view(value, mro, native, index, state)
        except Exception:
            return None
    return None
