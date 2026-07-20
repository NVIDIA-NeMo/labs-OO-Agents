# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified type serialization for agent snapshots.

Provides ``serialize()`` and ``deserialize()`` — an 8-step isinstance dispatch
that handles JSON primitives, enums, collections, nosnapshot values, Pydantic
models, dataclasses, and ``@snapshotable``-decorated classes.

Envelope format for non-primitive types::

    {"__type__": "pydantic"|"dataclass"|"dict_class",
     "__class__": "module.ClassName",
     "data": {...}}

    {"__type__": "tuple",
     "data": [...]}
"""

from __future__ import annotations

import dataclasses
import enum
import importlib
import inspect
from typing import Any

from pydantic import BaseModel

from nooa.errors.storage import DeserializationError, SerializationError
from nooa.storage.markers import is_nosnapshot_value, is_snapshotable_class

# ---------------------------------------------------------------------------
# Sentinel for "skip this value" (nosnapshot)
# ---------------------------------------------------------------------------


class _Skip:
    """Sentinel returned by serialize() for nosnapshot values."""

    _instance: _Skip | None = None

    def __new__(cls) -> _Skip:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "SKIP"

    def __bool__(self) -> bool:
        return False


SKIP = _Skip()

# Envelope type tags
_PYDANTIC = "pydantic"
_DATACLASS = "dataclass"
_DICT_CLASS = "dict_class"
_TUPLE = "tuple"
_ENVELOPE_TYPES = frozenset({_PYDANTIC, _DATACLASS, _DICT_CLASS, _TUPLE})
_LEGACY_CLASS_PREFIXES = {
    "nemo_oo_agents.": "nooa.",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def serialize(value: Any) -> tuple[Any, set[str]]:
    """Serialize a value into a JSON-compatible structure.

    Returns:
        A tuple of (serialized_blob, allowlist) where allowlist is the set
        of fully-qualified class names encountered during serialization.
        The allowlist must be passed to ``deserialize()`` for security.

    Raises:
        SerializationError: If the value contains an unsupported type.
    """
    allowlist: set[str] = set()
    blob = _serialize(value, allowlist)
    return blob, allowlist


def deserialize(blob: Any, allowlist: set[str]) -> Any:
    """Deserialize a blob produced by ``serialize()``.

    Args:
        blob: The serialized data.
        allowlist: Set of FQN strings that are permitted for class
            reconstruction. Populated by ``serialize()``.

    Returns:
        The restored Python object.

    Raises:
        DeserializationError: If the blob references a class not in the
            allowlist or the class cannot be imported.
    """
    return _deserialize(blob, allowlist)


# ---------------------------------------------------------------------------
# Internal dispatch — serialize
# ---------------------------------------------------------------------------


def _serialize(value: Any, allowlist: set[str]) -> Any:
    # 1. Enums — unwrap to value (must be before primitives since StrEnum/IntEnum are isinstance str/int)
    if isinstance(value, enum.Enum):
        return _serialize(value.value, allowlist)

    # 2. JSON primitives — pass through
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    # 3. Collections — recurse (filtering out SKIP sentinels)
    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise SerializationError(
                    f"Dict key {k!r} (type: {type(k).__name__}) is not a string. "
                    f"JSON requires string keys."
                )
            serialized = _serialize(v, allowlist)
            if serialized is not SKIP:
                result[k] = serialized
        return result
    if isinstance(value, tuple):
        return {
            "__type__": _TUPLE,
            "data": [s for item in value if (s := _serialize(item, allowlist)) is not SKIP],
        }
    if isinstance(value, list):
        return [s for item in value if (s := _serialize(item, allowlist)) is not SKIP]

    # 4. nosnapshot — return SKIP sentinel
    if is_nosnapshot_value(value):
        return SKIP

    # 5. Pydantic models
    if isinstance(value, BaseModel):
        fqn = _fqn(type(value))
        allowlist.add(fqn)
        # Iterate over model_fields directly so nested non-Pydantic objects
        # (dataclasses, @snapshotable) are serialized with their type envelopes
        # instead of being flattened to plain dicts by model_dump().
        data = {}
        for field_name in type(value).model_fields:
            serialized = _serialize(getattr(value, field_name), allowlist)
            if serialized is not SKIP:
                data[field_name] = serialized
        return {"__type__": _PYDANTIC, "__class__": fqn, "data": data}

    # 6. Dataclasses
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fqn = _fqn(type(value))
        allowlist.add(fqn)
        fields = dataclasses.fields(value)
        data = {}
        for f in fields:
            serialized = _serialize(getattr(value, f.name), allowlist)
            if serialized is not SKIP:
                data[f.name] = serialized
        return {"__type__": _DATACLASS, "__class__": fqn, "data": data}

    # 7. @snapshotable classes
    if is_snapshotable_class(type(value)):
        fqn = _fqn(type(value))
        allowlist.add(fqn)
        data = {}
        for k, v in vars(value).items():
            serialized = _serialize(v, allowlist)
            if serialized is not SKIP:
                data[k] = serialized
        return {"__type__": _DICT_CLASS, "__class__": fqn, "data": data}

    # 8. Unknown type — error with helpful message
    type_name = type(value).__qualname__
    raise SerializationError(
        f"Cannot serialize value of type {type_name!r}.\n"
        f"To fix, either:\n"
        f"  1. Decorate {type_name} with @snapshotable to serialize via vars()\n"
        f"  2. Mark the field as Annotated[{type_name}, nosnapshot] to skip it\n"
        f"  3. Wrap the value in a Pydantic BaseModel for automatic serialization"
    )


# ---------------------------------------------------------------------------
# Internal dispatch — deserialize
# ---------------------------------------------------------------------------


def _deserialize(blob: Any, allowlist: set[str]) -> Any:
    # Primitives
    if blob is None or isinstance(blob, (bool, int, float, str)):
        return blob

    # Lists
    if isinstance(blob, list):
        return [_deserialize(item, allowlist) for item in blob]

    # Dicts — check if envelope
    if isinstance(blob, dict):
        envelope_type = blob.get("__type__")
        if envelope_type == _TUPLE and "data" in blob:
            return tuple(_deserialize(item, allowlist) for item in blob["data"])
        if envelope_type in _ENVELOPE_TYPES and "__class__" in blob and "data" in blob:
            return _deserialize_envelope(blob, allowlist)
        # Regular dict — recurse values
        return {k: _deserialize(v, allowlist) for k, v in blob.items()}

    return blob


def _deserialize_envelope(blob: dict[str, Any], allowlist: set[str]) -> Any:
    """Reconstruct a typed object from an envelope dict."""
    fqn = blob["__class__"]
    envelope_type = blob["__type__"]
    data = blob["data"]

    if fqn not in allowlist:
        raise DeserializationError(
            f"Class {fqn!r} is not in the allowlist. "
            f"This blob may have been tampered with or produced by a different serialize() call."
        )

    cls = _import_class(fqn)

    if envelope_type == _PYDANTIC:
        # Deserialize nested envelopes first (e.g. dataclass/snapshotable
        # fields), then let Pydantic validate the reconstructed objects.
        deserialized_data = {k: _deserialize(v, allowlist) for k, v in data.items()}
        # Filter to known fields for lenient restoration (handles extra="forbid"
        # models that would reject extra fields from schema drift).
        known_fields = set(cls.model_fields)
        filtered = {k: v for k, v in deserialized_data.items() if k in known_fields}
        return cls.model_validate(filtered)

    if envelope_type == _DATACLASS:
        # Lenient: filter to accepted kwargs, let defaults fill gaps
        deserialized_data = {k: _deserialize(v, allowlist) for k, v in data.items()}
        accepted = {p.name for p in inspect.signature(cls).parameters.values()}
        filtered = {k: v for k, v in deserialized_data.items() if k in accepted}
        return cls(**filtered)

    if envelope_type == _DICT_CLASS:
        # @snapshotable: deserialize data, filter extra keys to __init__ params
        deserialized_data = {k: _deserialize(v, allowlist) for k, v in data.items()}
        accepted = {p.name for p in inspect.signature(cls.__init__).parameters.values()} - {"self"}
        filtered = {k: v for k, v in deserialized_data.items() if k in accepted}
        obj = cls(**filtered)
        # Set any remaining attrs that aren't __init__ params but were in vars()
        for k, v in deserialized_data.items():
            if k not in accepted:
                setattr(obj, k, v)
        return obj

    raise DeserializationError(f"Unknown envelope type: {envelope_type!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fqn(cls: type) -> str:
    """Fully qualified name: ``module.ClassName``."""
    return f"{cls.__module__}.{cls.__qualname__}"


def _import_class(fqn: str) -> type:
    """Import a class by its fully qualified name.

    Supports nested/inner classes whose ``__qualname__`` contains dots
    (e.g. ``module.Outer.Inner``).  We try progressively shorter module
    paths so that ``getattr`` can traverse the class hierarchy.

    Raises:
        DeserializationError: If the module or class cannot be found.
    """
    candidates = [fqn]
    for old_prefix, new_prefix in _LEGACY_CLASS_PREFIXES.items():
        if fqn.startswith(old_prefix):
            candidates.append(new_prefix + fqn.removeprefix(old_prefix))

    for candidate in candidates:
        parts = candidate.split(".")
        if len(parts) < 2:
            raise DeserializationError(f"Invalid fully qualified name: {fqn!r}")
        for i in range(len(parts) - 1, 0, -1):
            module_path = ".".join(parts[:i])
            try:
                obj = importlib.import_module(module_path)
                for attr in parts[i:]:
                    obj = getattr(obj, attr)
                return obj  # type: ignore[return-value]
            except (ImportError, AttributeError):
                continue
    raise DeserializationError(f"Cannot import class {fqn!r}: module or attribute not found")
