# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The reads of live classes and instances of unknown type: none of them runs user code.

This is the module to audit for principle 1 ("never run code on values").
Its helpers read a class's name, namespace and MRO through ``type``'s own
descriptors (no metaclass lookup), and instance storage through the native
``__dict__`` getset descriptor or ``__slots__`` member descriptors. A
namespace or storage dict is used only after every key was checked to be an
exact ``str``, without hashing or comparing a key. Exact built-in types are
named by identity (``_BUILTIN_PAIRS``). A read that cannot be made safely
gives ``None`` / ``_NO_SLOT`` or raises ``_FailClosed``; nothing here falls
back to ``getattr``, ``repr``, ``==``, ``hash`` or iteration of a user
object.

Outside this module, the capture layers touch a live value or class only in
these ways:

- ``type(v)``, ``id(v)`` and ``is``, which run no user code;
- native operations on a value whose exact type was checked first
  (``type(v) is ...``): the built-in ``str``, ``bytes``, ``int``,
  ``float``, ``bool``, ``list``, ``tuple`` and ``dict`` in ``_walk`` and
  ``_codec``, and, in ``_fields``, ``object.__getattribute__`` on a
  Pydantic ``FieldInfo``, a ``dataclasses.Field``, an ``Annotated`` alias
  and agentdoc's ``SpecAnnotation``;
- class metadata, the way Python reads it: ``_fields`` calls
  ``typing.get_type_hints(cls, include_extras=True)`` for a dataclass
  that has string annotations (a success is cached by class identity; an
  exception fails closed for that capture and is retried by the next one).
  This evaluates annotation code the class's author wrote;
- registered shapes, in ``_walk`` only: their accessors (trusted extension
  code), and what is done with their results: iterating ``items(v)``,
  indexing ``labels(v)[i]``, and copying a ``slice`` result (``bytes()`` of
  an exact ``bytearray`` / ``memoryview``, ``str.__str__`` of a ``str``
  subclass).

``_shapes`` also checks the arguments of the shape constructors
(``callable``, ``isinstance(fields, dict)``); those come from the code that
registers a shape, not from a captured value.
"""

from __future__ import annotations

import types
from collections.abc import Iterable, Iterator
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel

from nooa.snapshot._limits import _State
from nooa.snapshot._model import Reason

_MAX_TYPE_NAME = 200  # the most characters of a type name kept in a node
# ``type``'s own descriptors: reading a class's namespace, MRO and name through
# them skips any metaclass attribute of the same name.
_TYPE_DICT = type.__dict__["__dict__"]
_TYPE_MRO = type.__dict__["__mro__"]
_TYPE_NAME = type.__dict__["__name__"]
_BASE_MODEL_DICT = BaseModel.__dict__["__dict__"]  # a Pydantic model's raw instance ``__dict__``
# The exact built-in types and their names, matched by identity.
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


# --- type inspection without hooks ------------------------------------------


def _builtin_name(value: Any) -> str:
    """The name of ``value``'s type when it is exactly one of ``_BUILTIN_PAIRS``, else ``""``.

    Identity checks only, so no metaclass hook runs.

    >>> _builtin_name([]), _builtin_name(True), _builtin_name(object())
    ('list', 'bool', '')
    """
    value_type = type(value)
    for builtin, name in _BUILTIN_PAIRS:
        if value_type is builtin:
            return name
    return ""


def _bounded_type_name(value: Any, max_chars: int) -> str:
    """The ``__name__`` of ``value``'s class, at most ``max_chars`` characters.

    Read through ``type``'s own descriptor, so no instance or metaclass hook
    runs. ``"object"`` (cut to ``max_chars``) when the name cannot be read or
    is not an exact ``str``; ``""`` when ``max_chars`` is 0 or less.
    """
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

    ``raw`` is a class's ``mappingproxy`` or an instance or module ``dict``;
    anything else gives ``None``. A dict lookup compares a stored key with ``==`` when the hashes match, so
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
    """``cls``'s own namespace, read through ``type``'s descriptor and checked by ``_safe_dict``.

    ``None`` when it cannot be read, is not a ``mappingproxy``, or holds a key
    that is not an exact ``str``.
    """
    try:
        namespace = _TYPE_DICT.__get__(cls, type(cls))
    except Exception:
        return None
    if type(namespace) is not MappingProxyType:
        return None
    return _safe_dict(cls, namespace, state)


def _safe_mro(value: Any, state: _State) -> tuple[type[Any], ...] | None:
    """The MRO of ``value``'s class, read through ``type``'s own descriptor.

    Each entry costs one unit of work and must have a readable namespace
    (``_raw_namespace``; ``object`` is not checked). ``None`` when the MRO
    cannot be read, is not a tuple, the work runs out, or an entry's
    namespace is unreadable: the caller then treats the value as opaque.
    """
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


class _FailClosed(Exception):
    """What can't be read safely: raised by a field reader and the helpers it calls.

    A field reader's ``_FailClosed`` keeps no field of its object
    (``_object_fields``); a dataclass field whose annotation raises it is
    left out alone (``_dataclass_entries``).
    """


def _namespace(cls: type[Any], state: _State) -> dict[str, Any]:
    """``_raw_namespace(cls)``; raises ``_FailClosed`` when it is ``None``."""
    namespace = _raw_namespace(cls, state)
    if namespace is None:
        raise _FailClosed
    return namespace


def _copy_storage(obj_dict: Any, state: _State, limit: int | None = None) -> dict[str, Any]:
    """A copy of raw instance storage ``obj_dict``, every key checked to be an exact ``str``.

    Each entry copied costs one unit of work.

    Args:
        obj_dict: the instance's raw ``__dict__``.
        state: the capture's state, charged per entry.
        limit: copy only the first ``limit`` entries (insertion order) and
            stop quietly there; ``None``: all of them.

    Raises:
        _FailClosed: ``obj_dict`` is not an exact ``dict``, a key is not an
            exact ``str``, or the work runs out before the copy is done.
    """
    if type(obj_dict) is not dict:
        raise _FailClosed
    # Storage can be mutated directly. Copy only after validating every key so
    # later fixed-string lookups cannot collide with hostile key objects.
    safe: dict[str, Any] = {}
    for stored_name, stored_value in dict.items(obj_dict):
        if limit is not None and len(safe) >= limit:
            break
        if not state.take_node() or type(stored_name) is not str:
            raise _FailClosed
        safe[stored_name] = stored_value
    return safe


def _instance_storage(value: Any, mro: tuple[type[Any], ...], state: _State) -> Any:
    """``value``'s raw ``__dict__``, read through the native getset descriptor.

    The nearest ``__dict__`` entry in ``mro`` (by raw namespaces) is used;
    ``{}`` when no class has one (a slotted class). The result is not copied
    or checked: pass it to ``_copy_storage``.

    Raises:
        _FailClosed: the ``__dict__`` entry is not the native descriptor (a
            custom ``__dict__``), or reading it raised.
    """
    for base in mro:
        namespace = _raw_namespace(base, state) if base is not object else None
        if namespace is None:
            continue
        descriptor = dict.get(namespace, "__dict__")
        if descriptor is None:
            continue
        if type(descriptor) is not types.GetSetDescriptorType:
            raise _FailClosed
        try:
            return descriptor.__get__(value, type(value))
        except Exception:
            raise _FailClosed from None
    return {}


_NO_SLOT = object()
"""What ``_slot_value`` returns for a slot it cannot read (a sentinel, never a field value)."""


def _slot_value(name: str, mro: tuple[type[Any], ...], value: Any, state: _State) -> Any:
    """The value of slot ``name`` of ``value``, read through its native member descriptor.

    The nearest class in ``mro`` whose raw namespace holds ``name`` decides.
    ``_NO_SLOT`` when no class does, the entry is not a member descriptor,
    or the slot is unset (reading it raised).
    """
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


class _Omitted:
    """A note in an entry stream that something was left out at this point, and why.

    Field readers and shape builders yield one instead of an entry. The
    builder skips it and keeps reading; the first note's ``reason`` becomes
    the node's reason when the builder's own limits left the node complete.

    Args:
        reason: why the entry was left out.
    """

    __slots__ = ("reason",)
    reason: Reason

    def __init__(self, reason: Reason) -> None:
        self.reason = reason


_Fields = Iterator[tuple[str, Any] | _Omitted]
"""A field reader's (name, value) fields and ``_Omitted`` notes; it raises ``_FailClosed``."""
