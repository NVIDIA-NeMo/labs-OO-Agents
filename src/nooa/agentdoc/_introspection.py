# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Safe instance classification and value extraction for agent documentation."""

import inspect
from typing import Any

from nooa.agentdoc._info import TypeInfo
from nooa.agentdoc._structured import _ClassRef, _InstanceRef

_MISSING = object()


def _instance_dict(obj: Any) -> dict[str, Any] | None:
    """Return an instance ``__dict__`` without invoking ``__getattr__``."""
    try:
        value = object.__getattribute__(obj, "__dict__")
    except AttributeError:
        return None
    return value if isinstance(value, dict) else None


def _is_structured_instance(obj: Any, *, respect_custom_repr: bool = True) -> bool:
    """Check if object is an instance that should be formatted with type info.

    Returns True for:
    - Pydantic models
    - dataclasses
    - NamedTuples
    - attrs classes
    - Any custom class instance with __dict__ (not built-in types)

    Args:
        obj: Candidate instance.
        respect_custom_repr: If true, plain classes with custom ``__repr__``
            are treated as values. ``doc()`` disables this to retain type docs.

    Returns False for:
    - Types (classes themselves)
    - Built-in types (str, int, list, dict, etc.)
    - None
    """
    if isinstance(obj, type):
        return False

    # Skip built-in types and None
    if obj is None:
        return False

    if isinstance(obj, (str, int, float, bool, bytes, bytearray)):
        return False

    if isinstance(obj, (_ClassRef, _InstanceRef)):
        return False

    obj_type = type(obj)

    # Check for NamedTuple BEFORE checking for regular tuples
    # NamedTuples have a _fields attribute
    if (
        hasattr(obj_type, "_fields")
        and isinstance(getattr(obj_type, "_fields", None), tuple)
        and isinstance(obj, tuple)
    ):
        return True

    # Now safe to exclude regular tuples, lists, sets, dicts
    if isinstance(obj, (list, tuple, set, frozenset, dict)):
        return False

    # Pydantic (check before builtins guard — classes defined in exec()/REPL
    # get __module__='builtins' but are still structured types)
    if hasattr(obj_type, "model_fields"):
        return True

    # dataclass
    import dataclasses

    if dataclasses.is_dataclass(obj_type):
        return True

    # attrs
    if hasattr(obj_type, "__attrs_attrs__"):
        return True

    # Skip if it's a built-in type (range, slice, memoryview, etc.)
    if obj_type.__module__ == "builtins":
        return False

    # pformat() respects a custom __repr__, while doc() always renders the
    # type-level API contract and augments it with runtime instance fields.
    if respect_custom_repr:
        for klass in obj_type.__mro__:
            if klass is object:
                break
            if "__repr__" in klass.__dict__:
                return False

    # In doc mode, every non-builtin instance is documentable even when an
    # empty/private-only __slots__ leaves it with no runtime values. pformat()
    # still requires a public slot before choosing structured value rendering.
    if _instance_dict(obj) is None:
        if not respect_custom_repr:
            return True
        return any(
            slot
            for klass in obj_type.__mro__
            if klass is not object
            for slot in getattr(klass, "__slots__", ())
            if not slot.startswith("_")
        )

    # Any other custom class instance with __dict__
    return True


def _extract_instance_values(obj: Any, type_info: TypeInfo) -> dict[str, Any]:
    """Extract current field values from an instance.

    Args:
        obj: Instance to extract values from
        type_info: TypeInfo describing the type

    Returns:
        Dictionary mapping field names to current values
    """
    values = {}

    # First, get values for type fields
    obj_type = type(obj)

    # Respect Pydantic's exclude=True — those fields often contain large
    # internal state (e.g. captured_locals with arbitrary user objects that
    # can trigger expensive I/O when formatted, blocking the event loop).
    _excluded_fields: set[str] = set()
    if hasattr(obj_type, "model_fields"):
        _excluded_fields = {
            name
            for name, field_info in obj_type.model_fields.items()
            if getattr(field_info, "exclude", False)
        }

    obj_dict = _instance_dict(obj) or {}

    # Collect slot names for __slots__-based classes (no __dict__)
    _slots: set[str] = set()
    for cls in obj_type.__mro__:
        slots = getattr(cls, "__slots__", ())
        if isinstance(slots, str):
            _slots.add(slots)
        else:
            _slots.update(slots)

    for field in type_info.fields:
        if field.name in _excluded_fields:
            continue
        # Read from __dict__ first (instance data, always safe).
        if field.name in obj_dict:
            values[field.name] = obj_dict[field.name]
        elif field.name in _slots:
            # A subclass may replace an inherited slot with an arbitrary
            # descriptor. Only invoke the concrete member descriptor found by
            # static lookup; properties and custom descriptors are not safe.
            slot_descriptor = inspect.getattr_static(obj_type, field.name, _MISSING)
            if inspect.ismemberdescriptor(slot_descriptor):
                try:
                    values[field.name] = slot_descriptor.__get__(obj, obj_type)
                except AttributeError:
                    pass
        elif isinstance(obj, dict) and field.name in obj:
            # TypedDict instances are dicts
            values[field.name] = obj[field.name]
        else:
            # Fall back to class-level plain values (non-descriptors).
            # This handles annotated class defaults like `x: int = 5`.
            # Descriptors (property, classmethod, etc.) are skipped — they
            # can trigger arbitrary I/O.
            class_val = inspect.getattr_static(obj_type, field.name, _MISSING)
            if (
                class_val is not _MISSING
                and inspect.getattr_static(type(class_val), "__get__", None) is None
            ):
                values[field.name] = class_val

    # Also include runtime-only attributes that are not declared type fields.
    # Most objects store them in __dict__; Pydantic models with extra="allow"
    # store them separately in __pydantic_extra__.
    from nooa.agentdoc._visibility import is_hidden_field as _is_hidden_field

    def _include_dynamic(name: str, value: Any) -> bool:
        return (
            name not in values
            and not name.startswith("_")
            and not callable(value)
            and not _is_hidden_field(obj, name)
            and name not in _excluded_fields
        )

    for name, value in obj_dict.items():
        if _include_dynamic(name, value):
            values[name] = value

    pydantic_extra = None
    if hasattr(obj_type, "model_fields"):
        try:
            pydantic_extra = object.__getattribute__(obj, "__pydantic_extra__")
        except AttributeError:
            pass
    if isinstance(pydantic_extra, dict):
        for name, value in pydantic_extra.items():
            if _include_dynamic(name, value):
                values[name] = value

    return values