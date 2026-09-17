# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, no-user-code JSON previews for trace inputs and outputs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from nooa.agentdoc._docs import SpecAnnotation
from nooa.agentdoc._visibility import hidden

_JSON_KWARGS = {
    "ensure_ascii": True,
    "allow_nan": False,
    "separators": (", ", ": "),
    "sort_keys": False,
}
_FIELD_METADATA_ATTR = "_agentdoc_fields_docs"
_MAX_INCOMPLETE_PATHS = 32
_MAX_INCOMPLETE_PATH_CHARS = 2_048
_OPAQUE_PREFIX = "<opaque: "
_OPAQUE_SUFFIX = ">"
_TYPE_DICT = type.__dict__["__dict__"]
_TYPE_MRO = type.__dict__["__mro__"]
_TYPE_NAME = type.__dict__["__name__"]
_BASE_MODEL_DICT = BaseModel.__dict__["__dict__"]


@dataclass(frozen=True)
class TraceJSON:
    """A JSON preview plus locations whose source values are incomplete."""

    text: str
    incomplete_paths: tuple[str, ...]


@dataclass(frozen=True)
class Limits:
    """Hard output, inspection-work, and nesting limits for a trace preview."""

    max_chars: int = 50_000
    max_nodes: int = 2_000
    max_depth: int = 16


DEFAULT_LIMITS = Limits()


@dataclass(frozen=True)
class _Built:
    value: Any
    size: int


class _State:
    """Mutable bounded traversal state shared by one preview operation."""

    def __init__(self, limits: Limits) -> None:
        self.nodes_left = limits.max_nodes
        self.max_depth = limits.max_depth
        self.active_ids: set[int] = set()
        self.paths: list[str] = []
        self.path_chars = 0
        self.paths_collapsed = False

    def take_node(self) -> bool:
        """Consume one inspection unit, returning false when exhausted."""
        if self.nodes_left <= 0:
            return False
        self.nodes_left -= 1
        return True

    def add_incomplete(self, path: str) -> None:
        """Record a bounded incomplete path, collapsing overflow to the root."""
        if self.paths_collapsed or path in self.paths:
            return
        if (
            len(self.paths) >= _MAX_INCOMPLETE_PATHS
            or self.path_chars + len(path) > _MAX_INCOMPLETE_PATH_CHARS
        ):
            self.paths = [""]
            self.path_chars = 0
            self.paths_collapsed = True
            return
        self.paths.append(path)
        self.path_chars += len(path)


def _dumps(value: Any) -> str:
    """Encode with the canonical options used by accounting and final output."""
    return json.dumps(value, **_JSON_KWARGS)


def _validate_limits(limits: Limits) -> None:
    """Reject configurations that cannot produce a useful valid root value."""
    if limits.max_chars < 4:
        raise ValueError("trace JSON max_chars must be at least 4")
    if limits.max_nodes <= 0:
        raise ValueError("trace JSON max_nodes must be greater than 0")
    if limits.max_depth < 0:
        raise ValueError("trace JSON max_depth must be non-negative")


def _pointer_child(path: str, component: str) -> str:
    """Append one RFC 6901-escaped component to a JSON Pointer path."""
    escaped = component.replace("~", "~0").replace("/", "~1")
    return f"{path}/{escaped}"


def _smallest_fallback(value: Any, max_size: int) -> _Built:
    """Return a valid, type-oriented fallback that fits ``max_size``."""
    value_type = type(value)
    candidates: list[Any]
    if value_type is str:
        candidates = [""]
    elif value_type is list or value_type is tuple:
        candidates = [[]]
    elif value_type is dict:
        candidates = [{}]
    elif value_type is bool:
        candidates = [False, 0]
    elif value_type is int:
        candidates = [0]
    elif value_type is float:
        candidates = [0.0, 0]
    elif value is None:
        candidates = [None, 0]
    else:
        candidates = ["", 0]

    for candidate in candidates:
        encoded = _dumps(candidate)
        if len(encoded) <= max_size:
            return _Built(candidate, len(encoded))
    # Callers only descend when at least one JSON character is available.
    return _Built(0, 1)


def _bounded_type_name(value: Any, max_chars: int) -> str:
    """Read a bounded native type name without invoking instance/metaclass hooks."""
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


def _opaque(value: Any, max_size: int, path: str, state: _State) -> _Built:
    """Return a bounded descriptive string for an unsupported value."""
    state.add_incomplete(path)
    fallback = _smallest_fallback(value, max_size)
    if max_size < 2:
        return fallback

    # The raw label is bounded before concatenation and JSON escaping.
    name_budget = max(0, max_size - len(_OPAQUE_PREFIX) - len(_OPAQUE_SUFFIX) - 2)
    name = _bounded_type_name(value, name_budget)
    label = f"{_OPAQUE_PREFIX}{name}{_OPAQUE_SUFFIX}"
    built = _build_string(label, max_size, path, state, mark_truncated=False)
    return built if built.size <= max_size else fallback


def _build_string(
    value: str,
    max_size: int,
    path: str,
    state: _State,
    *,
    mark_truncated: bool = True,
) -> _Built:
    """Encode only a fitting prefix of an exact string."""
    if max_size < 2:
        if mark_truncated:
            state.add_incomplete(path)
        return _Built(0, 1)

    raw_limit = max_size - 2
    candidate = value if len(value) <= raw_limit else value[:raw_limit]
    encoded = _dumps(candidate)
    if len(encoded) <= max_size:
        if mark_truncated and len(candidate) < len(value):
            state.add_incomplete(path)
        return _Built(candidate, len(encoded))

    low = 0
    high = len(candidate)
    while low < high:
        midpoint = (low + high + 1) // 2
        if len(_dumps(candidate[:midpoint])) <= max_size:
            low = midpoint
        else:
            high = midpoint - 1
    prefix = candidate[:low]
    if mark_truncated and len(prefix) < len(value):
        state.add_incomplete(path)
    return _Built(prefix, len(_dumps(prefix)))


def _build_integer(value: int, max_size: int, path: str, state: _State) -> _Built:
    """Encode an integer only when decimal conversion is itself bounded."""
    bits = int.bit_length(value)
    # Decimal digits are approximately bits * log10(2). The upper bound keeps
    # conversion below both the output budget and Python's default digit guard.
    estimated_digits = (bits * 30_103) // 100_000 + 1 + (1 if value < 0 else 0)
    if estimated_digits > max_size or estimated_digits > 4_000:
        state.add_incomplete(path)
        return _smallest_fallback(value, max_size)
    try:
        encoded = _dumps(value)
    except (ValueError, OverflowError):
        state.add_incomplete(path)
        return _smallest_fallback(value, max_size)
    if len(encoded) > max_size:
        state.add_incomplete(path)
        return _smallest_fallback(value, max_size)
    return _Built(value, len(encoded))


def _enter_container(value: Any, path: str, state: _State) -> bool:
    """Mark a container active, reporting an active-ancestor cycle."""
    marker = id(value)
    if marker in state.active_ids:
        state.add_incomplete(path)
        return False
    state.active_ids.add(marker)
    return True


def _build_sequence(
    value: list[Any] | tuple[Any, ...],
    max_size: int,
    depth: int,
    path: str,
    state: _State,
) -> _Built:
    """Build a bounded list preview from an exact list or tuple."""
    if depth >= state.max_depth or not _enter_container(value, path, state):
        state.add_incomplete(path)
        return _Built([], 2)

    output: list[Any] = []
    size = 2
    try:
        list_value = cast(list[Any], value)
        tuple_value = cast(tuple[Any, ...], value)
        if type(value) is list:
            length = list.__len__(list_value)
        else:
            length = tuple.__len__(tuple_value)
        for index in range(length):
            if state.nodes_left <= 0:
                state.add_incomplete(path)
                break
            separator = 2 if output else 0
            available = max_size - size - separator
            if available < 1:
                state.add_incomplete(path)
                break
            if type(value) is list:
                child = list.__getitem__(list_value, index)
            else:
                child = tuple.__getitem__(tuple_value, index)
            child_path = _pointer_child(path, str(index))
            built = _build(child, available, depth + 1, child_path, state)
            output.append(built.value)
            size += separator + built.size
        if len(output) < length:
            state.add_incomplete(path)
        return _Built(output, size)
    finally:
        state.active_ids.discard(id(value))


def _build_dict(
    value: dict[str, Any],
    max_size: int,
    depth: int,
    path: str,
    state: _State,
) -> _Built:
    """Build a bounded dictionary preview without visiting rejected tail entries."""
    if depth >= state.max_depth or not _enter_container(value, path, state):
        state.add_incomplete(path)
        return _Built({}, 2)

    output: dict[str, Any] = {}
    size = 2
    try:
        iterator = iter(dict.items(value))
        while True:
            if state.nodes_left <= 0:
                if len(output) < dict.__len__(value):
                    state.add_incomplete(path)
                break
            try:
                key, child = next(iterator)
            except StopIteration:
                break
            if not state.take_node():
                state.add_incomplete(path)
                break
            if type(key) is not str:
                state.add_incomplete(path)
                break

            separator = 2 if output else 0
            remaining = max_size - size - separator
            # JSON escaping never makes a key shorter. Reject an obviously
            # oversized key before encoding it, then verify the exact size.
            if len(key) + 2 > remaining:
                state.add_incomplete(path)
                break
            encoded_key = _dumps(key)
            overhead = len(encoded_key) + 2
            available = remaining - overhead
            if available < 1:
                state.add_incomplete(path)
                break

            child_path = _pointer_child(path, key)
            built = _build(child, available, depth + 1, child_path, state)
            output[key] = built.value
            size += separator + overhead + built.size
        return _Built(output, size)
    finally:
        state.active_ids.discard(id(value))


def _raw_namespace(cls: type[Any]) -> MappingProxyType | None:
    """Return a class's native mappingproxy without custom metaclass lookup."""
    try:
        namespace = _TYPE_DICT.__get__(cls, type(cls))
    except Exception:
        return None
    return namespace if type(namespace) is MappingProxyType else None


def _pydantic_mro(value: Any, state: _State) -> tuple[type[Any], ...] | None:
    """Return a safely inspected BaseModel MRO, or ``None`` for unknown values."""
    cls = type(value)
    try:
        mro = _TYPE_MRO.__get__(cls, type(cls))
    except Exception:
        return None
    if type(mro) is not tuple:
        return None
    found = False
    checked: list[type[Any]] = []
    for base in mro:
        if not state.take_node():
            return None
        if base is not object and _raw_namespace(base) is None:
            return None
        checked.append(base)
        if base is BaseModel:
            found = True
    return tuple(checked) if found else None


def _visibility_overrides(
    obj_dict: dict[str, Any],
    mro: tuple[type[Any], ...],
    state: _State,
) -> dict[str, bool] | None:
    """Read complete raw NeMo hidden overrides before exposing model fields."""
    class_overrides: dict[str, bool] = {}

    def read_map(raw: Any, *, overwrite: bool) -> bool:
        if raw is None:
            return True
        if type(raw) is not dict:
            return False
        for field_name, metadata in dict.items(raw):
            if not state.take_node():
                return False
            if type(field_name) is not str or type(metadata) is not dict:
                return False
            hidden_value = None
            for metadata_name, metadata_value in dict.items(metadata):
                if not state.take_node():
                    return False
                if type(metadata_name) is not str:
                    return False
                if metadata_name == "hidden":
                    hidden_value = metadata_value
            if hidden_value is not None:
                if type(hidden_value) is not bool:
                    return False
                if overwrite or field_name not in class_overrides:
                    class_overrides[field_name] = hidden_value
        return True

    for base in mro:
        if not state.take_node():
            return None
        namespace = _raw_namespace(base)
        if namespace is None:
            return None
        if not read_map(namespace.get(_FIELD_METADATA_ATTR), overwrite=False):
            return None

    if not read_map(dict.get(obj_dict, _FIELD_METADATA_ATTR), overwrite=True):
        return None
    return class_overrides


def _field_hidden(field_info: FieldInfo, state: _State) -> bool | None:
    """Read only recognized resolved NeMo visibility metadata."""
    try:
        metadata = object.__getattribute__(field_info, "metadata")
    except Exception:
        return None
    if type(metadata) is not list:
        return None

    annotated_hidden = False
    for item in metadata:
        if not state.take_node():
            return None
        if item is hidden:
            annotated_hidden = True
        elif type(item) is SpecAnnotation:
            try:
                kwargs = object.__getattribute__(item, "kwargs")
            except Exception:
                return None
            if type(kwargs) is not dict:
                return None
            hidden_value = None
            for metadata_name, metadata_value in dict.items(kwargs):
                if not state.take_node():
                    return None
                if type(metadata_name) is not str:
                    return None
                if metadata_name == "hidden":
                    hidden_value = metadata_value
            if hidden_value is not None:
                if type(hidden_value) is not bool:
                    return None
                if hidden_value:
                    annotated_hidden = True
        # Foreign Pydantic constraint metadata is deliberately ignored.
    return annotated_hidden


def _pydantic_fields(mro: tuple[type[Any], ...], state: _State) -> dict[str, FieldInfo] | None:
    """Find an already-built exact Pydantic field dictionary in raw namespaces."""
    for base in mro:
        if not state.take_node():
            return None
        namespace = _raw_namespace(base)
        if namespace is None:
            return None
        fields = namespace.get("__pydantic_fields__")
        if fields is not None:
            return fields if type(fields) is dict else None
    return None


def _build_pydantic(
    value: BaseModel,
    mro: tuple[type[Any], ...],
    max_size: int,
    depth: int,
    path: str,
    state: _State,
) -> _Built:
    """Build a bounded declared/stored-field view of a Pydantic v2 model."""
    if depth >= state.max_depth or not _enter_container(value, path, state):
        state.add_incomplete(path)
        return _Built({}, 2)

    try:
        obj_dict = _BASE_MODEL_DICT.__get__(value, type(value))
    except Exception:
        obj_dict = None
    if type(obj_dict) is not dict:
        state.active_ids.discard(id(value))
        state.add_incomplete(path)
        return _Built({}, 2)

    # Pydantic storage can be mutated directly. Copy only after validating every
    # key so later fixed-string lookups cannot collide with hostile key objects.
    safe_obj_dict: dict[str, Any] = {}
    for stored_name, stored_value in dict.items(obj_dict):
        if not state.take_node() or type(stored_name) is not str:
            state.active_ids.discard(id(value))
            state.add_incomplete(path)
            return _Built({}, 2)
        safe_obj_dict[stored_name] = stored_value

    overrides = _visibility_overrides(safe_obj_dict, mro, state)
    fields = _pydantic_fields(mro, state)
    if overrides is None or fields is None:
        state.active_ids.discard(id(value))
        state.add_incomplete(path)
        return _Built({}, 2)

    output: dict[str, Any] = {}
    size = 2
    try:
        for name, field_info in dict.items(fields):
            if not state.take_node():
                state.add_incomplete(path)
                break
            if type(name) is not str or type(field_info) is not FieldInfo:
                state.add_incomplete(path)
                return _Built({}, 2)
            try:
                excluded = object.__getattribute__(field_info, "exclude")
                represented = object.__getattribute__(field_info, "repr")
            except Exception:
                state.add_incomplete(path)
                return _Built({}, 2)
            if excluded is not None and type(excluded) is not bool:
                state.add_incomplete(path)
                return _Built({}, 2)
            if type(represented) is not bool:
                state.add_incomplete(path)
                return _Built({}, 2)
            if excluded is True or represented is False:
                continue

            metadata_hidden = _field_hidden(field_info, state)
            if metadata_hidden is None:
                state.add_incomplete(path)
                return _Built({}, 2)
            hidden_value = overrides.get(name, metadata_hidden)
            if hidden_value or name not in safe_obj_dict:
                continue

            separator = 2 if output else 0
            remaining = max_size - size - separator
            if len(name) + 2 > remaining:
                state.add_incomplete(path)
                break
            encoded_key = _dumps(name)
            overhead = len(encoded_key) + 2
            available = remaining - overhead
            if available < 1:
                state.add_incomplete(path)
                break

            child_path = _pointer_child(path, name)
            built = _build(
                dict.__getitem__(safe_obj_dict, name), available, depth + 1, child_path, state
            )
            output[name] = built.value
            size += separator + overhead + built.size
        return _Built(output, size)
    finally:
        state.active_ids.discard(id(value))


def _build(
    value: Any,
    max_size: int,
    depth: int,
    path: str,
    state: _State,
    *,
    committed_fallback: _Built | None = None,
) -> _Built:
    """Recursively build one bounded JSON-compatible preview node."""
    fallback = committed_fallback or _smallest_fallback(value, max_size)
    if not state.take_node():
        state.add_incomplete(path)
        return fallback

    value_type = type(value)
    if value_type is str:
        return _build_string(value, max_size, path, state)
    if value is None:
        encoded = "null"
        if len(encoded) <= max_size:
            return _Built(None, len(encoded))
        state.add_incomplete(path)
        return fallback
    if value_type is bool:
        encoded = "true" if value else "false"
        if len(encoded) <= max_size:
            return _Built(value, len(encoded))
        state.add_incomplete(path)
        return fallback
    if value_type is int:
        return _build_integer(value, max_size, path, state)
    if value_type is float:
        if not math.isfinite(value):
            return _opaque(value, max_size, path, state)
        encoded = _dumps(value)
        if len(encoded) <= max_size:
            return _Built(value, len(encoded))
        state.add_incomplete(path)
        return fallback
    if value_type is list or value_type is tuple:
        if max_size < 2:
            state.add_incomplete(path)
            return fallback
        return _build_sequence(value, max_size, depth, path, state)
    if value_type is dict:
        if max_size < 2:
            state.add_incomplete(path)
            return fallback
        return _build_dict(value, max_size, depth, path, state)

    mro = _pydantic_mro(value, state)
    if mro is not None:
        if max_size < 2:
            state.add_incomplete(path)
            return fallback
        return _build_pydantic(value, mro, max_size, depth, path, state)
    return _opaque(value, max_size, path, state)


def trace_json(value: object, *, limits: Limits = DEFAULT_LIMITS) -> TraceJSON:
    """Return a bounded JSON preview of one arbitrary Python value."""
    _validate_limits(limits)
    state = _State(limits)
    built = _build(value, limits.max_chars, 0, "", state)
    text = _dumps(built.value)
    if len(text) > limits.max_chars:
        raise AssertionError("trace JSON accounting exceeded max_chars")
    return TraceJSON(text=text, incomplete_paths=tuple(state.paths))


def _field_placeholder(value: Any, state: _State) -> Any:
    """Classify one framework wrapper field into a minimal JSON root shape."""
    if not state.take_node():
        raise ValueError("trace JSON max_nodes cannot classify wrapper fields")
    value_type = type(value)
    if value_type is str:
        return ""
    if value_type is list or value_type is tuple:
        return []
    if value_type is dict:
        return {}
    if value is None:
        return None
    if value_type is bool:
        return False
    if value_type is int:
        return 0
    if value_type is float:
        return 0.0

    mro = _pydantic_mro(value, state)
    return {} if mro is not None else None


def trace_fields(*, limits: Limits = DEFAULT_LIMITS, **fields: object) -> TraceJSON:
    """Capture a framework-owned object while preserving every root field."""
    _validate_limits(limits)
    state = _State(limits)
    placeholders = {name: _field_placeholder(value, state) for name, value in fields.items()}
    skeleton = _dumps(placeholders)
    if len(skeleton) > limits.max_chars:
        raise ValueError("trace JSON max_chars cannot hold required wrapper fields")

    output = dict(placeholders)
    total_size = len(skeleton)
    for name, value in fields.items():
        placeholder = _Built(placeholders[name], len(_dumps(placeholders[name])))
        available = placeholder.size + (limits.max_chars - total_size)
        child_path = _pointer_child("", name)
        built = _build(
            value,
            available,
            1,
            child_path,
            state,
            committed_fallback=placeholder,
        )
        output[name] = built.value
        total_size += built.size - placeholder.size

    text = _dumps(output)
    if len(text) > limits.max_chars:
        raise AssertionError("trace JSON field accounting exceeded max_chars")
    return TraceJSON(text=text, incomplete_paths=tuple(state.paths))


__all__ = ["DEFAULT_LIMITS", "Limits", "TraceJSON", "trace_fields", "trace_json"]
