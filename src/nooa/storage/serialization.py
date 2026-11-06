# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified type serialization for agent snapshots.

Provides ``serialize()`` and ``deserialize()`` — an 8-step isinstance dispatch
that handles JSON primitives, enums, collections, nosnapshot values, Pydantic
models, dataclasses, and ``@snapshotable``-decorated classes.

Pydantic fields use canonical Python names for both storage and restoration,
independently of external input aliases. Explicit restoration hooks own their format.
Models that override ``model_validate`` retain their own input-alias behavior,
even if the override only delegates to ``super()``. With ``extra="allow"``, old
keys matching renamed fields only through ``AliasChoices`` become extras rather
than migrating; use ``__restore_snapshot__`` for such migrations.

Envelope format for non-primitive types::

    {"__type__": "pydantic"|"dataclass"|"dict_class",
     "__class__": "module.ClassName",
     "data": {...}}

    {"__type__": "tuple",
     "data": [...]}

A plain dict that itself carries an envelope key (``__type__`` or
``__class__``) is escaped as ``{"__type__": "plain_dict", "data": {...}}`` so
it comes back as the same plain dict instead of being read as an envelope:
data must not choose what ``deserialize()`` builds. Blobs written before the
escape existed contain no ``plain_dict`` envelopes and load unchanged.
"""

from __future__ import annotations

import dataclasses
import enum
import importlib
import inspect
import logging
from typing import Any

from pydantic import BaseModel

from nooa.errors.storage import DeserializationError, SerializationError
from nooa.storage.markers import is_nosnapshot_value, is_snapshotable_class

logger = logging.getLogger(__name__)

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
_PLAIN_DICT = "plain_dict"
# Keys that make a dict look like an envelope; plain dicts carrying them are escaped.
_ENVELOPE_KEYS = ("__type__", "__class__")
_ENVELOPE_TYPES = frozenset({_PYDANTIC, _DATACLASS, _DICT_CLASS, _TUPLE})
_LEGACY_CLASS_PREFIXES = {
    "nemo_oo_agents.": "nooa.",
    # Sessions saved by nooa-coder, which became nooa-atom (its coding/
    # module became agent/).
    "nooa_coder.coding.": "nooa_atom.agent.",
    "nooa_coder.": "nooa_atom.",
}


# Maximum nesting depth of a snapshot value, enforced by both ``serialize()``
# and ``deserialize()`` so the limit no longer depends on how deep the caller's
# stack already is. Depth counts containers on the path from the root: a dict,
# list, tuple, Pydantic model, dataclass or @snapshotable object is one level
# (on restore a type envelope is one level, not two), so whatever serializes
# also restores. Data returned by ``__snapshot_data__`` is checked on save with
# the same accounting restore uses.
#
# Measured frames per level on CPython 3.13 (minimum recursion limit for 200
# levels): save 1 for dicts, lists and tuples, 2 for models, dataclasses and
# @snapshotable objects (``_serialize_fields``); restore 1 for dicts, 2 for
# lists and tuples (``_deserialize_items``), ~3 for a type envelope
# (``_deserialize_envelope`` -> ``_deserialize_fields``, a dataclass slightly
# more for ``inspect.signature``). 200 levels therefore need at most ~615
# frames, below Python's default recursion limit (1000) even when the caller is
# already ~200 (up to ~380) frames deep. Real snapshot data is far shallower.
MAX_SNAPSHOT_DEPTH = 200

# Abbreviate error paths longer than this many segments.
_MAX_PATH_SEGMENTS = 16

# Exact types handled inline without a recursive call. Subclasses take the
# full dispatch, where enums are unwrapped (step 1) and any other str, int,
# float or bool subclass passes through as a plain value (step 2).
_PRIMITIVE_TYPES = frozenset({str, int, float, bool, type(None)})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def serialize(value: Any, *, base_depth: int = 0) -> tuple[Any, set[str]]:
    """Serialize a value into a JSON-compatible structure.

    Args:
        value: The value to serialize.
        base_depth: Number of containers that will sit above ``value`` once it
            is embedded in a larger snapshot. The depth limit is applied as if
            ``value`` were that deep, so a check made ahead of time (e.g. by
            ``SnapshotVars``) agrees with the later full save. Paths in error
            messages stay relative to ``value``.

    Returns:
        A tuple of (serialized_blob, allowlist) where allowlist is the set
        of fully-qualified class names encountered during serialization.
        The allowlist must be passed to ``deserialize()`` for security.

    Raises:
        SerializationError: If the value contains an unsupported type, a
            non-string dict key, a cycle (a container that contains itself,
            directly or through other objects), or nests deeper than
            ``MAX_SNAPSHOT_DEPTH``. The message names the path to the offending
            value and ``reason`` is ``"type"``, ``"key"``, ``"cycle"`` or
            ``"depth"``. Shared references that do not form a cycle are fine
            (each is serialized as a copy).
    """
    state = _SerializeState()
    try:
        blob = _serialize(value, state, base_depth)
    except _PathError as exc:
        error = SerializationError(exc.render("Snapshot value", base_depth))
        error.reason = exc.reason
        raise error from None
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
            than ``MAX_SNAPSHOT_DEPTH`` (``reason`` is ``"depth"``), or
            restoring hits Python's recursion limit.
    """
    try:
        return _deserialize(blob, allowlist, 0)
    except _PathError as exc:
        error = DeserializationError(exc.render("Snapshot blob", 0))
        error.reason = exc.reason
        raise error from None
    except RecursionError as exc:
        raise DeserializationError(
            "Snapshot restore hit Python's recursion limit within the snapshot "
            f"depth limit ({MAX_SNAPSHOT_DEPTH}) (see the chained RecursionError)"
        ) from exc


# ---------------------------------------------------------------------------
# Error paths — built only while an error unwinds
# ---------------------------------------------------------------------------


class _PathError(Exception):
    """Internal failure with a path, converted to the public error at the top.

    The happy path keeps no path at all. While this exception unwinds, each
    container frame appends the segment (``['k']``, ``[0]``, ``.attr``) of the
    child it was working on, so ``segments`` lists the path innermost first.
    ``reason`` is ``"depth"``, ``"cycle"``, ``"type"`` (unsupported value) or
    ``"key"`` (non-string dict key); the last two carry their message in
    ``detail``, with ``{path}`` where the path goes.
    """

    def __init__(
        self,
        reason: str = "depth",
        *,
        cycle_start: int = 0,
        type_name: str = "",
        detail: str = "",
    ) -> None:
        super().__init__()
        self.segments: list[str] = []
        self.reason = reason
        # Depth of the container a cycle refers back to.
        self.cycle_start = cycle_start
        self.type_name = type_name
        self.detail = detail

    def render(self, subject: str, base_depth: int) -> str:
        path = self.segments[::-1]
        if self.reason == "depth":
            return (
                f"{subject} nests deeper than the limit of {MAX_SNAPSHOT_DEPTH} "
                f"levels at {_format_path(path)}"
            )
        if self.reason != "cycle":
            return self.detail.replace("{path}", _format_path(path))
        # A cycle can refer back above ``base_depth``; name the root then.
        back = path[: max(self.cycle_start - base_depth, 0)]
        return (
            f"{subject} contains a cycle: {_format_path(path)} "
            f"refers back to {_format_path(back)} ({self.type_name})"
        )


def _format_path(segments: list[str]) -> str:
    """Render path segments (``['k']``, ``[0]``, ``.attr``) as ``<root>...``."""
    if len(segments) > _MAX_PATH_SEGMENTS:
        half = _MAX_PATH_SEGMENTS // 2
        skipped = len(segments) - 2 * half
        segments = [*segments[:half], f"...({skipped} more)...", *segments[-half:]]
    return "<root>" + "".join(segments)


# ---------------------------------------------------------------------------
# Internal dispatch — serialize
# ---------------------------------------------------------------------------


class _SerializeState:
    """Per-call state: the allowlist and the containers on the current path.

    ``active`` maps ``id()`` of each container being serialized to its depth.
    Entries are removed when the container is finished, so a value reachable
    twice without a cycle (a diamond) is not an error.
    """

    __slots__ = ("active", "allowlist")

    def __init__(self) -> None:
        self.allowlist: set[str] = set()
        self.active: dict[int, int] = {}


def _serialize_fields(items: Any, state: _SerializeState, depth: int, data: dict) -> dict:
    """Serialize ``(name, value)`` attribute pairs into ``data`` at ``depth``."""
    name = None
    try:
        for name, v in items:
            if type(v) in _PRIMITIVE_TYPES:
                data[name] = v
            elif (s := _serialize(v, state, depth)) is not SKIP:
                data[name] = s
    except _PathError as exc:
        exc.segments.append(f".{name}")
        raise
    return data


def _serialize(value: Any, state: _SerializeState, depth: int) -> Any:
    """Encode supported values recursively, omitting values marked nosnapshot.

    ``depth`` is the number of containers above ``value``.
    """
    # 1. Enums — unwrap to value (must be before primitives since StrEnum/IntEnum are isinstance str/int)
    if isinstance(value, enum.Enum):
        return _serialize(value.value, state, depth)

    # 2. JSON primitives — pass through. This includes str/int/float/bool
    # subclasses, which are stored as plain values (restored as the base type);
    # a ``__nosnapshot__`` marker on such a subclass is not honoured.
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    # 3. nosnapshot — check before collections so marked dict/list/tuple
    # subclasses are skipped
    if is_nosnapshot_value(value):
        return SKIP

    key = id(value)
    active = state.active
    if (start := active.get(key)) is not None:
        raise _PathError("cycle", cycle_start=start, type_name=type(value).__name__)
    if depth >= MAX_SNAPSHOT_DEPTH:
        raise _PathError()
    active[key] = depth
    depth += 1  # children of this container
    try:
        allowlist = state.allowlist

        # 4. Collections — recurse (filtering out SKIP sentinels)
        if isinstance(value, dict):
            result = {}
            k = None
            try:
                for k, v in value.items():
                    if not isinstance(k, str):
                        break
                    if type(v) in _PRIMITIVE_TYPES:
                        result[k] = v
                    elif (serialized := _serialize(v, state, depth)) is not SKIP:
                        result[k] = serialized
                else:
                    if any(key in value for key in _ENVELOPE_KEYS):
                        return {"__type__": _PLAIN_DICT, "data": result}
                    return result
            except _PathError as exc:
                exc.segments.append(f"[{k!r}]")
                raise
            # Raised outside the try: the path names this dict, not the key.
            raise _PathError("key", detail=_bad_key_message(k))
        if isinstance(value, (tuple, list)):
            items = []
            i = -1
            try:
                for i, item in enumerate(value):  # noqa: B007 - read by the except handler
                    if type(item) in _PRIMITIVE_TYPES:
                        items.append(item)
                    elif (s := _serialize(item, state, depth)) is not SKIP:
                        items.append(s)
            except _PathError as exc:
                exc.segments.append(f"[{i}]")
                raise
            return {"__type__": _TUPLE, "data": items} if isinstance(value, tuple) else items

        # 5. Pydantic models
        if isinstance(value, BaseModel):
            fqn = _fqn(type(value))
            allowlist.add(fqn)
            if snapshot_data := getattr(value, "__snapshot_data__", None):
                # Durable JSON types can own their schema and migration. Other
                # models still need recursive envelopes for arbitrary live fields.
                data = snapshot_data()
                if isinstance(data, dict):
                    # Serialize each value too, so a nested plain dict that looks
                    # like an envelope is escaped and comes back as a dict, not a
                    # typed object. This also applies the cycle and depth checks.
                    data = _serialize_fields(data.items(), state, depth, {})
                else:
                    # Restore walks this data with the depth limit, so save must too.
                    _check_snapshot_data_depth(data, depth)
                return {"__type__": _PYDANTIC, "__class__": fqn, "data": data}
            # Iterate over model_fields directly so nested non-Pydantic objects
            # (dataclasses, @snapshotable) are serialized with their type envelopes
            # instead of being flattened to plain dicts by model_dump().
            data = _serialize_fields(
                ((name, getattr(value, name)) for name in type(value).model_fields),
                state,
                depth,
                {},
            )
            if extra := value.model_extra:
                _serialize_fields(extra.items(), state, depth, data)
            return {"__type__": _PYDANTIC, "__class__": fqn, "data": data}

        # 6. Dataclasses
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            fqn = _fqn(type(value))
            allowlist.add(fqn)
            data = _serialize_fields(
                ((f.name, getattr(value, f.name)) for f in dataclasses.fields(value)),
                state,
                depth,
                {},
            )
            return {"__type__": _DATACLASS, "__class__": fqn, "data": data}

        # 7. @snapshotable classes
        if is_snapshotable_class(type(value)):
            fqn = _fqn(type(value))
            allowlist.add(fqn)
            fields = vars(value)
            lenient = getattr(type(value), "__snapshot_lenient_dict__", None)
            if isinstance(fields.get(lenient), dict):
                fields = dict(fields)
                lenient_dict = fields.pop(lenient)
                data = _serialize_fields(fields.items(), state, depth, {})
                data[lenient] = _serialize_lenient_dict(
                    type(value).__name__, lenient, lenient_dict, state, depth
                )
            else:
                data = _serialize_fields(fields.items(), state, depth, {})
            return {"__type__": _DICT_CLASS, "__class__": fqn, "data": data}

        # 8. Unknown type — error with helpful message
        type_name = type(value).__qualname__
        raise _PathError(
            "type",
            detail=f"Cannot serialize value of type {type_name!r} at {{path}}.\n"
            f"To fix, either:\n"
            f"  1. Decorate {type_name} with @snapshotable to serialize via vars()\n"
            f"  2. Mark the field as Annotated[{type_name}, nosnapshot] to skip it\n"
            f"  3. Wrap the value in a Pydantic BaseModel for automatic serialization",
        )
    finally:
        del active[key]


def _bad_key_message(k: Any) -> str:
    return (
        f"Dict key {k!r} (type: {type(k).__name__}) at {{path}} is not a string. "
        f"JSON requires string keys."
    )


def _serialize_lenient_dict(
    owner: str, name: str, value: dict, state: _SerializeState, depth: int
) -> dict:
    """Serialize the ``__snapshot_lenient_dict__`` attribute ``name`` at ``depth``.

    ``@snapshotable`` classes such as ``SnapshotVars`` name a dict attribute
    whose entries are independent: an entry that fails to serialize (a cycle,
    too deep, an unsupported type or key) is skipped with a warning naming its
    key, so the other entries still persist. The dict itself is checked like
    any other container.
    """
    key = id(value)
    active = state.active
    try:
        if (start := active.get(key)) is not None:
            raise _PathError("cycle", cycle_start=start, type_name="dict")
        if depth >= MAX_SNAPSHOT_DEPTH:
            raise _PathError()
    except _PathError as exc:
        exc.segments.append(f".{name}")
        raise
    active[key] = depth
    depth += 1  # the entries
    try:
        result = {}
        for k, v in value.items():
            try:
                if not isinstance(k, str):
                    raise _PathError("key", detail=_bad_key_message(k))
                if type(v) in _PRIMITIVE_TYPES:
                    result[k] = v
                elif (serialized := _serialize(v, state, depth)) is not SKIP:
                    result[k] = serialized
            except (_PathError, SerializationError, TypeError, ValueError, RecursionError) as exc:
                reason = (
                    exc.render(f"Value for {k!r}", depth)
                    if isinstance(exc, _PathError)
                    else str(exc)
                )
                logger.warning(
                    "Snapshot: skipping non-serializable %s key %r (%s): %s",
                    owner,
                    k,
                    type(v).__name__,
                    reason,
                )
        # Same escape as the plain-dict branch: keys must not choose what
        # deserialize() builds.
        if any(k in value for k in _ENVELOPE_KEYS):
            return {"__type__": _PLAIN_DICT, "data": result}
        return result
    finally:
        del active[key]


def _check_snapshot_data_depth(data: Any, depth: int) -> None:
    """Depth-check ``__snapshot_data__`` output the way restore will walk it.

    Restore treats the returned dict as the envelope's fields, so its values
    sit at ``depth`` (the envelope's own level is already counted).
    """
    if not isinstance(data, dict):
        _check_blob_depth(data, depth)
        return
    name = None
    try:
        for name, v in data.items():  # noqa: B007 - read by the except handler
            _check_blob_depth(v, depth)
    except _PathError as exc:
        exc.segments.append(f".{name}")
        raise


def _check_blob_depth(blob: Any, depth: int) -> None:
    """Raise ``_PathError`` where ``_deserialize`` would hit the depth limit.

    Mirrors ``_deserialize``'s accounting (a type envelope is one level). A
    Python tuple is walked like a list, which is never more lenient.
    """
    if isinstance(blob, (list, tuple)):
        if depth >= MAX_SNAPSHOT_DEPTH:
            raise _PathError()
        items: Any = blob
    elif isinstance(blob, dict):
        if depth >= MAX_SNAPSHOT_DEPTH:
            raise _PathError()
        envelope_type = blob.get("__type__")
        if envelope_type == _PLAIN_DICT and isinstance(blob.get("data"), dict):
            blob = blob["data"]  # an escaped plain dict counts as one level
            envelope_type = None
        if envelope_type == _TUPLE and "data" in blob:
            items = blob["data"]
        elif envelope_type in _ENVELOPE_TYPES and "__class__" in blob and "data" in blob:
            _check_snapshot_data_depth(blob["data"], depth + 1)
            return
        else:
            k = None
            try:
                for k, v in blob.items():  # noqa: B007 - read by the except handler
                    if type(v) not in _PRIMITIVE_TYPES:
                        _check_blob_depth(v, depth + 1)
            except _PathError as exc:
                exc.segments.append(f"[{k!r}]")
                raise
            return
    else:
        return
    i = -1
    try:
        for i, item in enumerate(items):  # noqa: B007 - read by the except handler
            if type(item) not in _PRIMITIVE_TYPES:
                _check_blob_depth(item, depth + 1)
    except _PathError as exc:
        exc.segments.append(f"[{i}]")
        raise


# ---------------------------------------------------------------------------
# Internal dispatch — deserialize
# ---------------------------------------------------------------------------


def _deserialize_items(items: Any, allowlist: set[str], depth: int) -> list[Any]:
    """Restore list (or tuple envelope) items, which sit at ``depth``."""
    out = []
    i = -1
    try:
        for i, item in enumerate(items):  # noqa: B007 - read by the except handler
            out.append(
                item if type(item) in _PRIMITIVE_TYPES else _deserialize(item, allowlist, depth)
            )
    except _PathError as exc:
        exc.segments.append(f"[{i}]")
        raise
    return out


def _deserialize_fields(data: dict[str, Any], allowlist: set[str], depth: int) -> dict[str, Any]:
    """Restore an envelope's fields, which sit at ``depth``."""
    out = {}
    k = None
    try:
        for k, v in data.items():
            out[k] = v if type(v) in _PRIMITIVE_TYPES else _deserialize(v, allowlist, depth)
    except _PathError as exc:
        exc.segments.append(f".{k}")
        raise
    return out


def _deserialize(blob: Any, allowlist: set[str], depth: int) -> Any:
    """Restore ``blob``; ``depth`` is the number of containers above it."""
    # Primitives
    if blob is None or isinstance(blob, (bool, int, float, str)):
        return blob

    # Lists
    if isinstance(blob, list):
        if depth >= MAX_SNAPSHOT_DEPTH:
            raise _PathError()
        return _deserialize_items(blob, allowlist, depth + 1)

    # Dicts — check if envelope
    if isinstance(blob, dict):
        if depth >= MAX_SNAPSHOT_DEPTH:
            raise _PathError()
        envelope_type = blob.get("__type__")
        if envelope_type == _PLAIN_DICT and isinstance(blob.get("data"), dict):
            # An escaped plain dict: restore its values, never the dict itself.
            # Its values sit one level below the escaped dict, as for a regular dict.
            result = {}
            k = None
            try:
                for k, v in blob["data"].items():
                    result[k] = (
                        v if type(v) in _PRIMITIVE_TYPES else _deserialize(v, allowlist, depth + 1)
                    )
            except _PathError as exc:
                exc.segments.append(f"[{k!r}]")
                raise
            return result
        if envelope_type == _TUPLE and "data" in blob:
            return tuple(_deserialize_items(blob["data"], allowlist, depth + 1))
        if envelope_type in _ENVELOPE_TYPES and "__class__" in blob and "data" in blob:
            return _deserialize_envelope(blob, allowlist, depth + 1)
        # Regular dict — recurse values
        result = {}
        k = None
        try:
            for k, v in blob.items():
                result[k] = (
                    v if type(v) in _PRIMITIVE_TYPES else _deserialize(v, allowlist, depth + 1)
                )
        except _PathError as exc:
            exc.segments.append(f"[{k!r}]")
            raise
        return result

    return blob


def _deserialize_envelope(blob: dict[str, Any], allowlist: set[str], depth: int) -> Any:
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
            return restore_snapshot(_deserialize_fields(data, allowlist, depth))
        if cls.model_config.get("extra") != "allow":
            # Discard obsolete fields before restoring their values: removed
            # types need not be importable or allowlisted for schema drift.
            data = {k: v for k, v in data.items() if k in cls.model_fields}
        # Restore retained nested envelopes before Pydantic validates them.
        return _validate_model_snapshot(cls, _deserialize_fields(data, allowlist, depth))

    if envelope_type == _DATACLASS:
        # Lenient: filter to accepted kwargs, let defaults fill gaps
        deserialized_data = _deserialize_fields(data, allowlist, depth)
        accepted = {p.name for p in inspect.signature(cls).parameters.values()}
        filtered = {k: v for k, v in deserialized_data.items() if k in accepted}
        return cls(**filtered)

    if envelope_type == _DICT_CLASS:
        # @snapshotable: deserialize data, filter extra keys to __init__ params
        deserialized_data = _deserialize_fields(data, allowlist, depth)
        accepted = {p.name for p in inspect.signature(cls.__init__).parameters.values()} - {"self"}
        filtered = {k: v for k, v in deserialized_data.items() if k in accepted}
        obj = cls(**filtered)
        # Set any remaining attrs that aren't __init__ params but were in vars()
        for k, v in deserialized_data.items():
            if k not in accepted:
                setattr(obj, k, v)
        return obj

    raise DeserializationError(f"Unknown envelope type: {envelope_type!r}")


def _validate_model_snapshot(cls: type[BaseModel], data: dict[str, Any]) -> BaseModel:
    """Validate canonical snapshot field names without changing external input aliases."""
    if getattr(cls.model_validate, "__func__", None) is not getattr(
        BaseModel.model_validate, "__func__", None
    ):
        # An application-supplied validation entry point retains its existing
        # input contract, just like an explicit __restore_snapshot__ hook.
        return cls.model_validate(data)
    return cls.model_validate(data, by_name=True, by_alias=False)


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
