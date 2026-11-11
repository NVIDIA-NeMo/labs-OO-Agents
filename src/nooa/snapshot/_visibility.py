# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which fields capture may show, and the field readers of Pydantic models and dataclasses.

What lives here:

- the hidden-field rules: agentdoc ``hidden`` / ``spec(hidden=...)``
  markers and the ``_agentdoc_fields_docs`` overrides of a class or an
  instance;
- the annotation metadata of a dataclass with string annotations:
  ``typing.get_type_hints(cls, include_extras=True)``; a success is cached
  by class identity in ``_HINTS``. On a failure each annotation is resolved
  on its own, only the unresolvable ones fail closed, and that result holds
  for the current capture only: the next one tries the whole class again;
- ``_pydantic_entries`` and ``_dataclass_entries``, the readers that yield
  the visible declared fields of a Pydantic v2 model or a dataclass from
  raw instance storage.

Live classes and instances are read through ``_raw``; the only other
reads are ``object.__getattribute__`` on metadata objects of an exact,
checked type (see ``_raw``). A field whose visibility cannot be read is
left out (``_Omitted("unreadable")``) or, when the field list itself cannot
be read, the reader raises ``_FailClosed`` and the object keeps no field.
"""

from __future__ import annotations

import dataclasses
import functools
import sys
import typing
import weakref
from collections.abc import Callable
from typing import Any, cast

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from nooa.agentdoc._docs import SpecAnnotation
from nooa.agentdoc._visibility import hidden
from nooa.snapshot._limits import _State
from nooa.snapshot._model import Reason
from nooa.snapshot._raw import (
    _BASE_MODEL_DICT,
    _NO_SLOT,
    _TYPE_MRO,
    _copy_storage,
    _FailClosed,
    _instance_storage,
    _namespace,
    _raw_namespace,
    _slot_value,
)

_FIELD_METADATA_ATTR = "_agentdoc_fields_docs"
_CLASS_METADATA_ATTR = "_agentdoc_docs"
_DATACLASS_FIELD_TYPE = cast(Any, dataclasses)._FIELD
_ANNOTATED_ALIAS = type(typing.Annotated[int, "x"])
"""The type of an ``Annotated[...]`` hint, compared by identity (``type(hint) is ...``)."""


# --- visibility (agentdoc ``hidden`` / ``spec(hidden=...)``) ------------------------


_SPEC_KEYS = ("hidden", "max_string", "max_length", "hide_empty")
"""The ``spec()`` keywords capture honours: visibility, per-field limits, ``hide_empty``."""


def _spec_flags(raw: Any, state: _State) -> dict[str, Any]:
    """The ``spec()`` keywords capture honours, from raw metadata keywords.

    ``_FailClosed`` unless ``raw`` is an exact ``dict`` with exact ``str``
    keys, each charged one node, whose ``hidden`` is absent or a ``bool``.
    ``hide_empty`` must be a ``bool`` and ``max_string`` / ``max_length``
    ``None`` or a non-negative exact ``int``; another value is ignored.
    """
    if type(raw) is not dict:
        raise _FailClosed
    found: dict[str, Any] = {}
    for name, value in dict.items(raw):
        if not state.take_node() or type(name) is not str:
            raise _FailClosed
        if name == "hidden":
            if value is not None and type(value) is not bool:
                raise _FailClosed
            if value is not None:
                found[name] = value
        elif name == "hide_empty":
            if type(value) is bool:
                found[name] = value
        elif name == "max_string" or name == "max_length":
            if value is None or (type(value) is int and value >= 0):
                found[name] = value
    return found


def _hidden_flag(raw: Any, state: _State) -> bool | None:
    """The ``hidden`` entry of raw metadata keywords: a ``bool``, ``None`` when absent."""
    return _spec_flags(raw, state).get("hidden")


def _visibility_overrides(
    obj_dict: dict[str, Any], mro: tuple[type[Any], ...], state: _State
) -> dict[str, dict[str, Any]]:
    """Read the complete raw ``spec(cls, name, ...)`` overrides before exposing fields (``_FailClosed``).

    Field name -> the ``spec()`` keywords capture honours (``_SPEC_KEYS``);
    the instance's own overrides win, then the nearest class's.
    """
    overrides: dict[str, dict[str, Any]] = {}

    def read_map(raw: Any, *, overwrite: bool) -> None:
        if raw is None:
            return
        if type(raw) is not dict:
            raise _FailClosed
        for field_name, metadata in dict.items(raw):
            if not state.take_node() or type(field_name) is not str:
                raise _FailClosed
            flags = _spec_flags(metadata, state)
            merged = overrides.setdefault(field_name, {})
            for key, value in flags.items():
                if overwrite or key not in merged:
                    merged[key] = value

    for base in mro:
        if not state.take_node():
            raise _FailClosed
        read_map(dict.get(_namespace(base, state), _FIELD_METADATA_ATTR), overwrite=False)
    read_map(dict.get(obj_dict, _FIELD_METADATA_ATTR), overwrite=True)
    return overrides


def _metadata_spec(metadata: Any, state: _State) -> dict[str, Any]:
    """The ``spec()`` keywords of recognized marker metadata (``_FailClosed``: malformed).

    ``hidden`` is the agentdoc marker; a ``SpecAnnotation`` gives its
    keywords (a later one wins).
    """
    if type(metadata) is not list and type(metadata) is not tuple:
        raise _FailClosed
    found: dict[str, Any] = {}
    for item in metadata:
        if not state.take_node():
            raise _FailClosed
        if item is hidden:
            found["hidden"] = True
        elif type(item) is SpecAnnotation:
            try:
                kwargs = object.__getattribute__(item, "kwargs")
            except Exception:
                raise _FailClosed from None
            flags = _spec_flags(kwargs, state)
            if flags.get("hidden") is False and found.get("hidden"):
                flags.pop("hidden")  # an earlier ``hidden`` marker is not undone
            found.update(flags)
        # Foreign constraint metadata is deliberately ignored.
    return found


def _metadata_hidden(metadata: Any, state: _State) -> bool:
    """Whether recognized marker metadata hides a field (``_FailClosed``: malformed)."""
    return bool(_metadata_spec(metadata, state).get("hidden"))


def _field_spec(field_info: FieldInfo, state: _State) -> dict[str, Any]:
    """Read only the recognized resolved ``spec()`` metadata of a Pydantic field."""
    try:
        metadata = object.__getattribute__(field_info, "metadata")
    except Exception:
        raise _FailClosed from None
    if type(metadata) is not list:
        raise _FailClosed
    return _metadata_spec(metadata, state)


def _annotation_spec(
    name: str,
    annotation: Any,
    hints: Callable[[], dict[str, tuple[Any, ...]]],
    state: _State,
) -> dict[str, Any]:
    """The ``spec()`` keywords a field's annotation carries (``_FailClosed``: unreadable).

    A string annotation reads the class's resolved hint metadata, ``hints()``.
    Otherwise only an exact ``typing`` ``Annotated`` alias (type identity) is
    inspected, through its raw ``__metadata__``; anything else is visible.
    """
    if type(annotation) is str:
        metadata = dict.get(hints(), name)
        if metadata is None:
            raise _FailClosed
        return _metadata_spec(metadata, state)
    if type(annotation) is not _ANNOTATED_ALIAS:
        return {}
    try:
        metadata = object.__getattribute__(annotation, "__metadata__")
    except Exception:
        raise _FailClosed from None
    return _metadata_spec(metadata, state)


_HINTS: dict[int, tuple[weakref.ref[type[Any]], dict[str, tuple[Any, ...]]]] = {}
"""Per dataclass, keyed by ``id`` and re-checked by identity: its resolved hint metadata."""


def _hint_metadata(hint: Any) -> tuple[Any, ...]:
    """The outermost ``Annotated`` metadata of one resolved hint (``()``: none); raises if malformed."""
    if type(hint) is not _ANNOTATED_ALIAS:
        return ()
    found = object.__getattribute__(hint, "__metadata__")
    if type(found) is not tuple:
        raise TypeError("unreadable Annotated metadata")
    return found


def _resolved_metadata(cls: type[Any], state: _State) -> dict[str, tuple[Any, ...]]:
    """The outermost ``Annotated`` metadata of each resolved type hint of ``cls``.

    String annotations (PEP 563) are read the way Python reads them:
    ``typing.get_type_hints(cls, include_extras=True)``. A successful result
    is cached for the process, keyed by ``id(cls)`` and re-checked with
    ``is``, so the class is never hashed or compared; the cache keeps only a
    weak reference to the class and the marker metadata, and
    ``weakref.finalize`` drops the entry with the class. A hint that is not an
    exact ``Annotated`` alias has no metadata (``()``).

    If evaluation raises (for example, a name imported only under
    ``TYPE_CHECKING``, or a forward reference to a name not defined yet),
    each annotation is resolved on its own instead (``_single_hints``), as
    agentdoc does: only the fields whose annotation can't be resolved are
    missing from the result, so they alone fail closed. That result is
    recorded on ``state`` (evaluated once per capture, not once per
    instance), never cached for the process: the next capture tries the
    whole class again.
    """
    key = id(cls)
    entry = _HINTS.get(key)
    if entry is not None and entry[0]() is cls:
        return entry[1]
    failed = state.unreadable_hints.get(key)
    if failed is not None and failed[0] is cls:
        return failed[1]
    metadata: dict[str, tuple[Any, ...]] = {}
    try:
        hints = typing.get_type_hints(cls, include_extras=True)
        for name, hint in dict.items(hints):
            if type(name) is str:
                metadata[name] = _hint_metadata(hint)
    except Exception:
        partial = _single_hints(cls, state)
        state.unreadable_hints[key] = (cls, partial)  # retried by the next capture
        return partial
    try:
        reference = weakref.ref(cls)
        weakref.finalize(cls, _HINTS.pop, key, None)
    except Exception:
        return metadata  # not cacheable: evaluated again next time
    _HINTS[key] = (reference, metadata)
    return metadata


def _single_hints(cls: type[Any], state: _State) -> dict[str, tuple[Any, ...]]:
    """The metadata of each annotation of ``cls`` that resolves on its own; the rest are missing.

    Resolves every annotation as ``typing.get_type_hints`` would, one at a
    time: walking the MRO from ``object`` down (a subclass's annotation
    wins), each through ``get_type_hints`` of a one-annotation probe class
    with the declaring class's namespaces and type parameters. An
    annotation that raises, or a namespace that can't be read, is left out.
    """
    metadata: dict[str, tuple[Any, ...]] = {}
    try:
        mro = _TYPE_MRO.__get__(cls, type(cls))
    except Exception:
        return metadata
    if type(mro) is not tuple:
        return metadata
    for base in reversed(mro):
        namespace = _raw_namespace(base, state)
        if namespace is None:
            continue
        annotations = dict.get(namespace, "__annotations__")
        if type(annotations) is not dict:
            continue
        module_name = dict.get(namespace, "__module__")
        module = sys.modules.get(module_name) if type(module_name) is str else None
        module_globals = getattr(module, "__dict__", None) if module is not None else None
        if type(module_globals) is not dict:
            module_globals = {}
        type_params = dict.get(namespace, "__type_params__", ())
        for name, raw in list(dict.items(annotations)):
            if not state.take_node():
                return metadata
            if type(name) is not str:
                continue
            metadata.pop(name, None)  # this class declares it: an inherited hint no longer applies
            try:
                probe = type("_Probe", (), {"__annotations__": {name: raw}})
                if type(type_params) is tuple and type_params:
                    probe.__type_params__ = type_params
                # The arguments are ``get_type_hints``'s own for a class: the
                # class namespace as globals and the module's as locals, so
                # module names win, exactly as for the whole class.
                hint = typing.get_type_hints(
                    probe, globalns=dict(namespace), localns=module_globals, include_extras=True
                )[name]
                metadata[name] = _hint_metadata(hint)
            except Exception:
                continue  # unresolvable: this field fails closed
    return metadata


# --- Pydantic models and dataclasses ---------------------------------------------


def _pydantic_fields(mro: tuple[type[Any], ...], state: _State) -> dict[str, FieldInfo]:
    """Find an already-built exact Pydantic field dictionary in raw namespaces (``_FailClosed``)."""
    for base in mro:
        if not state.take_node():
            raise _FailClosed
        namespace = _namespace(base, state)
        # Pydantic stored the completed map directly as ``model_fields`` before
        # 2.10, then moved it to ``__pydantic_fields__``. Read raw namespaces so
        # neither layout invokes the metaclass descriptor exposed by newer
        # releases.
        for key in ("__pydantic_fields__", "model_fields"):
            fields = dict.get(namespace, key)
            if fields is not None:
                if type(fields) is not dict:
                    raise _FailClosed
                return fields
    raise _FailClosed


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


_Fields = typing.Iterator[tuple[str, Any] | _Omitted]
"""A field reader's (name, value) fields and ``_Omitted`` notes; it raises ``_FailClosed``."""


class _Scoped:
    """A field value with its own ``spec(max_string=..., max_length=...)`` limits.

    A field reader yields ``(name, _Scoped(value, limits))``; ``_walk`` builds
    ``value`` under ``limits`` (``max_string`` / ``max_items`` replaced) for
    its whole subtree. Private: only the readers here make it.
    """

    __slots__ = ("limits", "value")

    def __init__(self, value: Any, limits: dict[str, int | None]) -> None:
        self.value = value
        self.limits = limits


def _is_empty(value: Any) -> bool:
    """``None`` or an empty exact ``str``, ``bytes``, ``list``, ``tuple``, ``dict``, ``set``, ``frozenset``."""
    if value is None:
        return True
    value_type = type(value)
    if value_type in _EMPTY_TYPES:  # exact built-in types: their own ``__len__``
        return len(value) == 0
    return False


_EMPTY_TYPES = (str, bytes, list, tuple, dict, set, frozenset)


def _class_hide_empty(mro: tuple[type[Any], ...], state: _State) -> bool:
    """A class-level ``@spec(hide_empty=True)``: the nearest ``_agentdoc_docs`` in the MRO that sets it."""
    for base in mro:
        if base is object:
            continue
        raw = dict.get(_namespace(base, state), _CLASS_METADATA_ATTR)
        if raw is None:
            continue
        flag = _spec_flags(raw, state).get("hide_empty")
        if flag is not None:
            return flag
    return False


def _field_entry(
    name: str, value: Any, flags: dict[str, Any], hide_empty: bool
) -> tuple[str, Any] | None:
    """A visible field as its reader yields it: ``None`` when ``hide_empty`` leaves it out."""
    if flags.get("hide_empty", hide_empty) and _is_empty(value):
        return None
    limits = {key: flags[key] for key in ("max_string", "max_length") if key in flags}
    return (name, _Scoped(value, limits) if limits else value)


def _merged(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    if not override:
        return base
    merged = dict(base)
    merged.update(override)
    return merged


def _pydantic_entries(
    value: BaseModel, mro: tuple[type[Any], ...], carrier: type[Any], state: _State
) -> _Fields:
    """The privileged field reader of a Pydantic v2 model: declared fields, raw stored values."""
    try:
        obj_dict = _BASE_MODEL_DICT.__get__(value, type(value))
    except Exception:
        obj_dict = None
    storage = _copy_storage(obj_dict, state)
    try:
        overrides = _visibility_overrides(storage, mro, state)
    finally:  # the field map is read (and charged) even when the overrides fail closed
        fields = _pydantic_fields(mro, state)
    hide_empty = _class_hide_empty(mro, state)
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
        flags = _merged(_field_spec(field_info, state), overrides.get(name))
        if flags.get("hidden") or name not in storage:
            continue
        entry = _field_entry(name, dict.__getitem__(storage, name), flags, hide_empty)
        if entry is not None:
            yield entry


def _dataclass_entries(
    value: Any, mro: tuple[type[Any], ...], carrier: type[Any], state: _State
) -> _Fields:
    """The privileged field reader of a dataclass: declared fields of ``carrier``, raw values."""
    storage = _copy_storage(_instance_storage(value, mro, state), state)
    fields = dict.get(_namespace(carrier, state), "__dataclass_fields__")
    if type(fields) is not dict:
        raise _FailClosed
    overrides = _visibility_overrides(storage, mro, state)
    hide_empty = _class_hide_empty(mro, state)
    # The class's resolved hints, read once, at the first string annotation.
    hints = functools.cache(functools.partial(_resolved_metadata, carrier, state))
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
        try:
            annotation_flags = _annotation_spec(name, annotation, hints, state)
        except _FailClosed:
            # This field can't be shown safely: omit it, the object is incomplete.
            yield _Omitted("unreadable")
            continue
        flags = _merged(annotation_flags, overrides.get(name))
        if flags.get("hidden"):
            continue
        if name in storage:
            field_value = dict.__getitem__(storage, name)
        else:
            field_value = _slot_value(name, mro, value, state)
            if field_value is _NO_SLOT:
                continue
        entry = _field_entry(name, field_value, flags, hide_empty)
        if entry is not None:
            yield entry


# --- plain objects ----------------------------------------------------------------


def _slot_names(mro: tuple[type[Any], ...], state: _State) -> list[str]:
    """The ``__slots__`` names the classes of ``mro`` declare, in MRO order (``_FailClosed``)."""
    names: list[str] = []
    for base in mro:
        if base is object:
            continue
        declared = dict.get(_namespace(base, state), "__slots__")
        if declared is None:
            continue
        if type(declared) is str:
            declared = (declared,)
        if type(declared) is not tuple and type(declared) is not list:
            raise _FailClosed
        for name in declared:
            if type(name) is not str:
                raise _FailClosed
            if name not in names:
                names.append(name)
    return names


def _plain_annotation_spec(name: str, mro: tuple[type[Any], ...], state: _State) -> dict[str, Any]:
    """The ``spec()`` keywords of the nearest class annotation of ``name`` in the MRO."""
    for base in mro:
        if base is object:
            continue
        annotations = dict.get(_namespace(base, state), "__annotations__")
        if type(annotations) is not dict:
            continue
        for key, annotation in dict.items(annotations):
            if type(key) is str and key == name:
                hints = functools.partial(_resolved_metadata, base, state)
                return _annotation_spec(name, annotation, hints, state)
    return {}


def _class_defaults(mro: tuple[type[Any], ...], state: _State) -> list[tuple[str, Any]]:
    """The annotated class attributes of ``mro`` with a plain (non-descriptor) value, bases first.

    ``label: str = "ok"`` declared on a class is a field of its instances
    until they set their own. A value whose type defines ``__get__`` (a
    function, a property) is not a field: reading it through the instance
    would run it.
    """
    found: dict[str, Any] = {}
    for base in reversed(mro):
        if base is object:
            continue
        namespace = _namespace(base, state)
        annotations = dict.get(namespace, "__annotations__")
        if type(annotations) is not dict:
            continue
        for name in list(dict.keys(annotations)):
            if not state.take_node():
                raise _FailClosed
            if type(name) is not str or name not in namespace:
                continue
            default = dict.__getitem__(namespace, name)
            if not _is_descriptor(default, state):
                found.pop(name, None)  # a subclass's declaration comes later
                found[name] = default
    return list(found.items())


def _is_descriptor(value: Any, state: _State) -> bool:
    """Whether ``type(value)`` defines ``__get__`` (raw namespaces; ``True`` when unreadable)."""
    cls = type(value)
    try:
        mro = _TYPE_MRO.__get__(cls, type(cls))
    except Exception:
        return True
    if type(mro) is not tuple:
        return True
    for base in mro:
        if base is object:
            continue
        namespace = _raw_namespace(base, state)
        if namespace is None or "__get__" in namespace:
            return True
    return False


def _is_class(value: Any, state: _State) -> bool:
    """Whether ``value`` is a class: ``type`` in its type's raw MRO (``isinstance`` reads ``__class__``)."""
    cls = type(value)
    try:
        mro = _TYPE_MRO.__get__(cls, type(cls))
    except Exception:
        return False
    return type(mro) is tuple and any(base is type for base in mro)


def _plain_entries(
    value: Any, mro: tuple[type[Any], ...], state: _State
) -> list[tuple[str, Any] | _Omitted] | None:
    """The public instance attributes of a plain object, read raw; ``None``: not a plain object.

    A plain object is an instance of a class whose MRO defines no
    ``__getattribute__`` (no proxies), with the standard ``__dict__`` slot
    (a native getset descriptor) or declared ``__slots__``. Its fields are
    the instance ``__dict__`` entries, in insertion order, then the filled
    slots: never a ``_private`` name, a callable other than a class, or a
    field agentdoc hides (``hidden`` / ``spec(hidden=True)`` in a class
    annotation or a ``spec(cls, name, hidden=True)`` override). The other
    ``spec()`` keywords apply as for dataclasses (``hide_empty``,
    ``max_string``, ``max_length``). An object
    with no field left whose class defines ``__repr__`` is not a plain
    object either: its repr is all it shows.
    """
    try:
        has_dict = False
        defines_repr = False
        for base in mro:
            if base is object:
                continue
            namespace = _namespace(base, state)
            if "__getattribute__" in namespace:
                return None
            if "__dict__" in namespace:
                has_dict = True
            if "__repr__" in namespace:
                defines_repr = True
        slots = _slot_names(mro, state)
        if not has_dict and not [name for name in slots if name not in ("__dict__", "__weakref__")]:
            return None
        storage = _copy_storage(_instance_storage(value, mro, state), state) if has_dict else {}
        overrides = _visibility_overrides(storage, mro, state)
        hide_empty = _class_hide_empty(mro, state)
        found: list[tuple[str, Any]] = []
        for name, default in _class_defaults(mro, state):
            if name in storage:
                found.append((name, dict.__getitem__(storage, name)))
            elif name not in slots:
                found.append((name, default))
        declared = {name for name, _value in found}
        found += [(name, item) for name, item in dict.items(storage) if name not in declared]
        for name in slots:
            if name in storage or name in ("__dict__", "__weakref__"):
                continue
            slot = _slot_value(name, mro, value, state)
            if slot is not _NO_SLOT:
                found.append((name, slot))
        entries: list[tuple[str, Any] | _Omitted] = []
        for name, field_value in found:
            if not state.take_node():
                entries.append(_Omitted("max_nodes"))
                break
            if name.startswith("_"):
                continue
            if callable(field_value) and not _is_class(field_value, state):
                continue
            try:
                annotation_flags = _plain_annotation_spec(name, mro, state)
            except _FailClosed:
                entries.append(_Omitted("unreadable"))
                continue
            flags = _merged(annotation_flags, overrides.get(name))
            if flags.get("hidden"):
                continue
            entry = _field_entry(name, field_value, flags, hide_empty)
            if entry is not None:
                entries.append(entry)
    except _FailClosed:
        return None
    if not entries and defines_repr:
        return None
    return entries
