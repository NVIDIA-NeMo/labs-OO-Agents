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


# Maximum nesting depth of a snapshot value, enforced by both ``serialize()``
# and ``deserialize()`` so the limit no longer depends on how deep the caller's
# stack already is. Depth counts containers on the path from the root: a dict,
# list, tuple, Pydantic model, dataclass or @snapshotable object is one level
# (on restore a type envelope is one level, not two), so whatever serializes
# also restores.
#
# Measured cost per level on CPython 3.13: 2 frames on save (``_serialize`` ->
# ``_child``), 2 on restore for plain lists/dicts/tuples and 3 for a type
# envelope (``_deserialize`` -> ``_deserialize_envelope`` ->
# ``_deserialize_child``). 200 levels therefore need at most ~610 frames, so the
# limit is reached before Python's default recursion limit (1000) even when the
# caller is already ~200 (up to ~390) frames deep, as Atom's checkpoint is.
# Real snapshot data is far shallower than this.
MAX_SNAPSHOT_DEPTH = 200

# Abbreviate error paths longer than this many segments.
_MAX_PATH_SEGMENTS = 16


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
        SerializationError: If the value contains an unsupported type, a
            cycle (a container that contains itself, directly or through
            other objects), or nests deeper than ``MAX_SNAPSHOT_DEPTH``. The
            message names the path to the offending value. Shared references
            that do not form a cycle are fine (each is serialized as a copy).
    """
    state = _SerializeState()
    try:
        blob = _serialize(value, state, 0)
    except RecursionError as exc:
        # Cycles and over-deep values are detected above, so this is something
        # else, e.g. a recursive ``__snapshot_data__`` or a caller that is
        # already near the recursion limit.
        raise SerializationError(
            "Snapshot serialization hit Python's recursion limit within the "
            f"snapshot depth limit ({MAX_SNAPSHOT_DEPTH}); this is not a "
            "detected cycle (see the chained RecursionError)"
        ) from exc
    return blob, state.allowlist


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
            allowlist, the class cannot be imported, the blob nests deeper
            than ``MAX_SNAPSHOT_DEPTH``, or restoring hits Python's
            recursion limit.
    """
    path: list[str] = []
    try:
        return _deserialize(blob, allowlist, 0, path)
    except RecursionError as exc:
        raise DeserializationError(
            "Snapshot restore hit Python's recursion limit within the snapshot "
            f"depth limit ({MAX_SNAPSHOT_DEPTH}) (see the chained RecursionError)"
        ) from exc


# ---------------------------------------------------------------------------
# Internal dispatch — serialize
# ---------------------------------------------------------------------------


class _SerializeState:
    """Per-call state: the allowlist and the containers on the current path.

    ``active`` maps ``id()`` of each container being serialized to its index
    in ``path``. Entries are removed when the container is finished, so a
    value reachable twice without a cycle (a diamond) is not an error.
    """

    __slots__ = ("active", "allowlist", "path")

    def __init__(self) -> None:
        self.allowlist: set[str] = set()
        self.active: dict[int, int] = {}
        self.path: list[str] = []


def _format_path(segments: list[str]) -> str:
    """Render path segments (``['k']``, ``[0]``, ``.attr``) as ``<root>...``."""
    if len(segments) > _MAX_PATH_SEGMENTS:
        half = _MAX_PATH_SEGMENTS // 2
        skipped = len(segments) - 2 * half
        segments = [*segments[:half], f"...({skipped} more)...", *segments[-half:]]
    return "<root>" + "".join(segments)


def _enter(value: Any, state: _SerializeState, depth: int) -> None:
    """Check ``value`` for a cycle and the depth limit, then mark it active."""
    key = id(value)
    if (start := state.active.get(key)) is not None:
        raise SerializationError(
            f"Snapshot value contains a cycle: {_format_path(state.path)} "
            f"refers back to {_format_path(state.path[:start])} "
            f"({type(value).__name__})"
        )
    if depth >= MAX_SNAPSHOT_DEPTH:
        raise SerializationError(
            f"Snapshot value nests deeper than the limit of {MAX_SNAPSHOT_DEPTH} "
            f"levels at {_format_path(state.path)}"
        )
    state.active[key] = len(state.path)


def _child(value: Any, segment: str, state: _SerializeState, depth: int) -> Any:
    """Serialize one child under ``segment`` of the current path."""
    state.path.append(segment)
    try:
        return _serialize(value, state, depth)
    finally:
        state.path.pop()


def _serialize(value: Any, state: _SerializeState, depth: int) -> Any:
    """Encode supported values recursively, omitting values marked nosnapshot.

    ``depth`` is the number of containers above ``value``.
    """
    # 1. Enums — unwrap to value (must be before primitives since StrEnum/IntEnum are isinstance str/int)
    if isinstance(value, enum.Enum):
        return _serialize(value.value, state, depth)

    # 2. JSON primitives — pass through
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    # 3. nosnapshot — check before collections so marked subclasses are skipped
    if is_nosnapshot_value(value):
        return SKIP

    _enter(value, state, depth)
    depth += 1  # children of this container
    try:
        allowlist = state.allowlist

        # 4. Collections — recurse (filtering out SKIP sentinels)
        if isinstance(value, dict):
            result = {}
            for k, v in value.items():
                if not isinstance(k, str):
                    raise SerializationError(
                        f"Dict key {k!r} (type: {type(k).__name__}) is not a string. "
                        f"JSON requires string keys."
                    )
                serialized = _child(v, f"[{k!r}]", state, depth)
                if serialized is not SKIP:
                    result[k] = serialized
            return result
        if isinstance(value, (tuple, list)):
            items = [
                s
                for i, item in enumerate(value)
                if (s := _child(item, f"[{i}]", state, depth)) is not SKIP
            ]
            return {"__type__": _TUPLE, "data": items} if isinstance(value, tuple) else items

        # 5. Pydantic models
        if isinstance(value, BaseModel):
            fqn = _fqn(type(value))
            allowlist.add(fqn)
            if snapshot_data := getattr(value, "__snapshot_data__", None):
                # Durable JSON types can own their schema and migration. Other
                # models still need recursive envelopes for arbitrary live fields.
                return {"__type__": _PYDANTIC, "__class__": fqn, "data": snapshot_data()}
            # Iterate over model_fields directly so nested non-Pydantic objects
            # (dataclasses, @snapshotable) are serialized with their type envelopes
            # instead of being flattened to plain dicts by model_dump().
            data = {}
            for field_name in type(value).model_fields:
                serialized = _child(getattr(value, field_name), f".{field_name}", state, depth)
                if serialized is not SKIP:
                    data[field_name] = serialized
            for field_name, extra in (value.model_extra or {}).items():
                serialized = _child(extra, f".{field_name}", state, depth)
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
                serialized = _child(getattr(value, f.name), f".{f.name}", state, depth)
                if serialized is not SKIP:
                    data[f.name] = serialized
            return {"__type__": _DATACLASS, "__class__": fqn, "data": data}

        # 7. @snapshotable classes
        if is_snapshotable_class(type(value)):
            fqn = _fqn(type(value))
            allowlist.add(fqn)
            data = {}
            for k, v in vars(value).items():
                serialized = _child(v, f".{k}", state, depth)
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
    finally:
        del state.active[id(value)]


# ---------------------------------------------------------------------------
# Internal dispatch — deserialize
# ---------------------------------------------------------------------------


def _check_depth(depth: int, path: list[str]) -> None:
    """Reject a blob container nested deeper than ``MAX_SNAPSHOT_DEPTH``."""
    if depth >= MAX_SNAPSHOT_DEPTH:
        raise DeserializationError(
            f"Snapshot blob nests deeper than the limit of {MAX_SNAPSHOT_DEPTH} "
            f"levels at {_format_path(path)}"
        )


def _deserialize_child(
    blob: Any, segment: str, allowlist: set[str], depth: int, path: list[str]
) -> Any:
    """Restore one child under ``segment``; primitives skip the path bookkeeping."""
    if blob is None or isinstance(blob, (bool, int, float, str)):
        return blob
    path.append(segment)
    try:
        return _deserialize(blob, allowlist, depth, path)
    finally:
        path.pop()


def _deserialize(blob: Any, allowlist: set[str], depth: int, path: list[str]) -> Any:
    """Restore ``blob``; ``depth`` is the number of containers above it."""
    # Primitives
    if blob is None or isinstance(blob, (bool, int, float, str)):
        return blob

    # Lists
    if isinstance(blob, list):
        _check_depth(depth, path)
        return [
            _deserialize_child(item, f"[{i}]", allowlist, depth + 1, path)
            for i, item in enumerate(blob)
        ]

    # Dicts — check if envelope
    if isinstance(blob, dict):
        _check_depth(depth, path)
        envelope_type = blob.get("__type__")
        if envelope_type == _TUPLE and "data" in blob:
            return tuple(
                [
                    _deserialize_child(item, f"[{i}]", allowlist, depth + 1, path)
                    for i, item in enumerate(blob["data"])
                ]
            )
        if envelope_type in _ENVELOPE_TYPES and "__class__" in blob and "data" in blob:
            return _deserialize_envelope(blob, allowlist, depth + 1, path)
        # Regular dict — recurse values
        return {
            k: _deserialize_child(v, f"[{k!r}]", allowlist, depth + 1, path)
            for k, v in blob.items()
        }

    return blob


def _deserialize_envelope(
    blob: dict[str, Any], allowlist: set[str], depth: int, path: list[str]
) -> Any:
    """Reconstruct a typed object from an envelope dict; its fields are at ``depth``."""
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
        if restore_snapshot := getattr(cls, "__restore_snapshot__", None):
            deserialized_data = {
                k: _deserialize_child(v, f".{k}", allowlist, depth, path) for k, v in data.items()
            }
            return restore_snapshot(deserialized_data)
        if cls.model_config.get("extra") != "allow":
            # Discard obsolete fields before restoring their values: removed
            # types need not be importable or allowlisted for schema drift.
            data = {k: v for k, v in data.items() if k in cls.model_fields}
        # Restore retained nested envelopes before Pydantic validates them.
        deserialized_data = {
            k: _deserialize_child(v, f".{k}", allowlist, depth, path) for k, v in data.items()
        }
        return cls.model_validate(deserialized_data)

    if envelope_type == _DATACLASS:
        # Lenient: filter to accepted kwargs, let defaults fill gaps
        deserialized_data = {
            k: _deserialize_child(v, f".{k}", allowlist, depth, path) for k, v in data.items()
        }
        accepted = {p.name for p in inspect.signature(cls).parameters.values()}
        filtered = {k: v for k, v in deserialized_data.items() if k in accepted}
        return cls(**filtered)

    if envelope_type == _DICT_CLASS:
        # @snapshotable: deserialize data, filter extra keys to __init__ params
        deserialized_data = {
            k: _deserialize_child(v, f".{k}", allowlist, depth, path) for k, v in data.items()
        }
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
