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
_DATACLASS_FIELD_TYPE = cast(Any, dataclasses)._FIELD
_ANNOTATED_ALIAS = type(typing.Annotated[int, "x"])
"""The type of an ``Annotated[...]`` hint, compared by identity (``type(hint) is ...``)."""


# --- visibility (agentdoc ``hidden`` / ``spec(hidden=...)``) ------------------------


def _hidden_flag(raw: Any, state: _State) -> bool | None:
    """The ``hidden`` entry of raw metadata keywords: a ``bool``, ``None`` when absent.

    ``_FailClosed`` unless ``raw`` is an exact ``dict`` with exact ``str``
    keys, each charged one node, whose ``hidden`` is absent or a ``bool``.
    """
    if type(raw) is not dict:
        raise _FailClosed
    found = None
    for name, value in dict.items(raw):
        if not state.take_node() or type(name) is not str:
            raise _FailClosed
        if name == "hidden":
            found = value
    if found is not None and type(found) is not bool:
        raise _FailClosed
    return found


def _visibility_overrides(
    obj_dict: dict[str, Any], mro: tuple[type[Any], ...], state: _State
) -> dict[str, bool]:
    """Read complete raw hidden overrides before exposing fields (``_FailClosed``)."""
    class_overrides: dict[str, bool] = {}

    def read_map(raw: Any, *, overwrite: bool) -> None:
        if raw is None:
            return
        if type(raw) is not dict:
            raise _FailClosed
        for field_name, metadata in dict.items(raw):
            if not state.take_node() or type(field_name) is not str:
                raise _FailClosed
            flag = _hidden_flag(metadata, state)
            if flag is not None and (overwrite or field_name not in class_overrides):
                class_overrides[field_name] = flag

    for base in mro:
        if not state.take_node():
            raise _FailClosed
        read_map(dict.get(_namespace(base, state), _FIELD_METADATA_ATTR), overwrite=False)
    read_map(dict.get(obj_dict, _FIELD_METADATA_ATTR), overwrite=True)
    return class_overrides


def _metadata_hidden(metadata: Any, state: _State) -> bool:
    """Whether recognized marker metadata hides a field (``_FailClosed``: malformed)."""
    if type(metadata) is not list and type(metadata) is not tuple:
        raise _FailClosed
    annotated_hidden = False
    for item in metadata:
        if not state.take_node():
            raise _FailClosed
        if item is hidden:
            annotated_hidden = True
        elif type(item) is SpecAnnotation:
            try:
                kwargs = object.__getattribute__(item, "kwargs")
            except Exception:
                raise _FailClosed from None
            if _hidden_flag(kwargs, state):
                annotated_hidden = True
        # Foreign constraint metadata is deliberately ignored.
    return annotated_hidden


def _field_hidden(field_info: FieldInfo, state: _State) -> bool:
    """Read only recognized resolved visibility metadata of a Pydantic field."""
    try:
        metadata = object.__getattribute__(field_info, "metadata")
    except Exception:
        raise _FailClosed from None
    if type(metadata) is not list:
        raise _FailClosed
    return _metadata_hidden(metadata, state)


def _annotation_hidden(
    name: str,
    annotation: Any,
    hints: Callable[[], dict[str, tuple[Any, ...]]],
    state: _State,
) -> bool:
    """Whether a dataclass field's annotation hides it (``_FailClosed``: unreadable).

    A string annotation reads the class's resolved hint metadata, ``hints()``.
    Otherwise only an exact ``typing`` ``Annotated`` alias (type identity) is
    inspected, through its raw ``__metadata__``; anything else is visible.
    """
    if type(annotation) is str:
        metadata = dict.get(hints(), name)
        if metadata is None:
            raise _FailClosed
        return _metadata_hidden(metadata, state)
    if type(annotation) is not _ANNOTATED_ALIAS:
        return False
    try:
        metadata = object.__getattribute__(annotation, "__metadata__")
    except Exception:
        raise _FailClosed from None
    return _metadata_hidden(metadata, state)


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
        if overrides.get(name, _field_hidden(field_info, state)) or name not in storage:
            continue
        yield name, dict.__getitem__(storage, name)


def _dataclass_entries(
    value: Any, mro: tuple[type[Any], ...], carrier: type[Any], state: _State
) -> _Fields:
    """The privileged field reader of a dataclass: declared fields of ``carrier``, raw values."""
    storage = _copy_storage(_instance_storage(value, mro, state), state)
    fields = dict.get(_namespace(carrier, state), "__dataclass_fields__")
    if type(fields) is not dict:
        raise _FailClosed
    overrides = _visibility_overrides(storage, mro, state)
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
            annotation_hidden = _annotation_hidden(name, annotation, hints, state)
        except _FailClosed:
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
