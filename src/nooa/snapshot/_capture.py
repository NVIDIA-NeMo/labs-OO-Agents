# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``capture()``: a bounded snapshot of a finished Python value, without running user code.

Based on the bounded trace JSON builder of PR #342 (Severin Klingler): exact
built-in types are read natively, Pydantic v2 models and dataclasses through
their declared fields in raw instance storage, and everything else is
``Unavailable("opaque")``. No ``repr()``, ``str()``, truthiness, properties, descriptors,
custom iteration, hashing or equality runs on an inspected value.

One shared budget (``Limits``) bounds the whole capture:

- ``max_chars``: the encoded size of the snapshot as canonical JSON (the
  notation ``render_json`` emits), accounted exactly as in #342, so that
  ``render_json(capture(v, limits), max_chars=limits.max_chars)`` never cuts
  anything further;
- ``max_nodes``: inspection work (values, container entries, MRO entries,
  field and visibility metadata);
- ``max_depth``: nesting;
- ``max_items`` / ``max_string``: items per container and characters per
  string, kept from the start and (``tail_share``) the end.

``capture.register(T, shape)`` extends it with nearest-MRO dispatch: a
shape (``array_like``, ``ndarray_like``, ``mapping_like``, ``object_like``,
``text_like``) maps out the structure of a type through accessors, and capture
builds the snapshot with the same builders as the built-in types, so the
shared budget and the notation hold. Pydantic models and dataclasses are
internal ``object_like`` shapes over their privileged field readers.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from nooa.snapshot._codec import dumps, fallback_candidates, pointer_child
from nooa.snapshot._limits import DEFAULT_LIMITS, Limits, _State, _validate_limits
from nooa.snapshot._model import Key, Mapping, Object, Snapshot, Unavailable
from nooa.snapshot._raw import _MAX_TYPE_NAME, _bounded_type_name, _builtin_name, _safe_mro
from nooa.snapshot._shapes import (
    _SHAPE_TYPES,
    _Shape,
    array_like,
    mapping_like,
    ndarray_like,
    object_like,
    text_like,
)
from nooa.snapshot._walk import _REGISTRY, _build, _Built, _fallback

__all__ = [
    "Limits",
    "array_like",
    "capture",
    "capture_fields",
    "mapping_like",
    "ndarray_like",
    "object_like",
    "text_like",
]


class _CaptureFunction:
    """``capture(value, limits)``, with ``capture.register(T, shape)`` to extend it."""

    def __call__(self, value: object, limits: Limits = DEFAULT_LIMITS) -> Snapshot:
        """A bounded snapshot of ``value``; no user code runs on unregistered types."""
        _validate_limits(limits)
        state = _State(limits)
        return _build(value, limits.max_chars, 0, "", state).node

    def register(self, cls: type[Any], shape: _Shape) -> None:
        """Capture ``cls`` and its subclasses as ``shape`` describes them.

        ``shape`` is one of ``array_like``, ``ndarray_like``, ``mapping_like``,
        ``object_like`` or ``text_like``; anything else raises ``TypeError``.
        Registering needs the imported type, so supporting a library is opt-in
        by whoever sets up the agent; the registry is process-wide. Classes
        are matched by identity (``is``) along the value's raw MRO, so no user
        code runs during lookup. The registration nearest in a value's MRO
        wins (over the built-in Pydantic and dataclass views too); registering
        ``cls`` again replaces it. Exact built-in types (``str``, ``list``, ...)
        are always native.
        """
        if not isinstance(cls, type):
            raise TypeError(f"capture.register needs a type, got {type(cls).__name__}")
        shape_type = type(shape)
        if not any(shape_type is known for known in _SHAPE_TYPES):
            raise TypeError(
                "capture.register needs a shape (array_like, ndarray_like, mapping_like,"
                f" object_like or text_like), got {_bounded_type_name(shape, _MAX_TYPE_NAME)}"
            )
        _REGISTRY[:] = [entry for entry in _REGISTRY if entry[0] is not cls]
        _REGISTRY.append((cls, shape))

    def unregister(self, cls: type[Any]) -> None:
        """Remove the registration for exactly ``cls`` (no-op when absent)."""
        _REGISTRY[:] = [entry for entry in _REGISTRY if entry[0] is not cls]


capture = _CaptureFunction()


def _field_placeholder(value: Any, state: _State) -> tuple[Any, Snapshot]:
    """One framework wrapper field's minimal JSON root value, and its node when no work is left.

    An exact built-in type other than ``bytes`` reserves its smallest
    fallback (``""``, ``[]``, ``{}``, ``0``, ...), a Pydantic model ``{}``,
    anything else ``null`` (#342's ``trace_fields``).
    """
    if not state.take_node():
        raise ValueError("snapshot max_nodes cannot classify wrapper fields")
    name = _builtin_name(value)
    if name and name != "bytes":
        shape = fallback_candidates(name)[0]
        return shape, _fallback(value, "max_nodes", len(dumps(shape))).node
    mro = _safe_mro(value, state)
    if mro is not None and any(base is BaseModel for base in mro):
        name = _bounded_type_name(value, _MAX_TYPE_NAME)
        return {}, Object(name, (), "max_nodes")
    return None, Unavailable("max_nodes", "NoneType")


def capture_fields(*, limits: Limits = DEFAULT_LIMITS, **fields: object) -> Mapping:
    """Capture a framework-owned wrapper so that every root field survives, with its type.

    Each field first reserves a minimal placeholder of its own type (``[]``,
    ``{}``, ``""``, ...); raises ``ValueError`` when ``limits`` cannot hold the
    complete minimal wrapper (#342's ``trace_fields``). A field reached with
    no node work left keeps its placeholder.
    """
    _validate_limits(limits)
    state = _State(limits)
    placeholders = {name: _field_placeholder(value, state) for name, value in fields.items()}
    skeleton = dumps({name: shape for name, (shape, _node) in placeholders.items()})
    if len(skeleton) > limits.max_chars:
        raise ValueError("snapshot max_chars cannot hold required wrapper fields")

    items: list[tuple[Key, Snapshot]] = []
    total_size = len(skeleton)
    for name, value in fields.items():
        shape, starved = placeholders[name]
        reserved = len(dumps(shape))
        if state.nodes_left > 0:
            available = reserved + (limits.max_chars - total_size)
            built = _build(value, available, 1, pointer_child("", name), state)
        else:
            built = _Built(starved, reserved)
        items.append((name, built.node))
        total_size += built.size - reserved
    return Mapping("dict", len(items), tuple(items), wrapper=True)
