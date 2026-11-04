# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The snapshot model: small, detached, frozen trees describing a value or an output.

A snapshot holds no live objects. Every node says whether it is complete and,
if not, why: ``complete`` is ``reason is None`` on every node. Renderers (``nooa.snapshot.render_*``) turn a
snapshot into the notation a consumer shows; see ``docs/snapshot-model.md``.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Literal

__all__ = [
    "REASONS",
    "Key",
    "Mapping",
    "Object",
    "Reason",
    "Ref",
    "Scalar",
    "Sequence",
    "Snapshot",
    "Spill",
    "Text",
    "Unavailable",
]

Reason = Literal[
    "max_length",
    "max_items",
    "max_depth",
    "max_nodes",
    "cycle",
    "opaque",
    "unreadable",
    "key",
    "error",
]
"""Why a node is incomplete.

``max_length`` / ``max_items`` / ``max_depth`` / ``max_nodes``: a limit cut it;
``cycle``: a reference to an enclosing container; ``opaque``: a value of a
type NOOA does not inspect; ``unreadable``: an object's fields, or a field's
metadata (``spec`` hidden marker, annotation), could not be read safely, so
they were left out (fail closed); ``key``: a mapping stopped at a key with no
JSON form, or one that collides with an earlier key; ``error``: an accessor
of a registered shape raised. Whether a spill file holds the whole content is
not a reason: ``Text.full`` says it.
"""

REASONS: tuple[str, ...] = typing.get_args(Reason)
"""Every ``Reason``, in declaration order (derived, so the two cannot drift)."""

Key = str | int | float | bool | None
"""A mapping key kept in a snapshot (exact built-in scalar types only)."""


@dataclass(frozen=True)
class Spill:
    """Where the whole content of a ``Text`` was written.

    ``kept < total``: the file is itself cut (it holds the first ``kept`` bytes).
    """

    path: str
    kept: int  # bytes written to the file
    total: int  # bytes produced

    @property
    def cut(self) -> bool:
        """Whether the file holds less than everything that was produced."""
        return self.kept < self.total


@dataclass(frozen=True)
class Scalar:
    """A complete ``None``, ``bool``, ``int``, ``float`` or (short) ``bytes`` value."""

    value: None | bool | int | float | bytes

    @property
    def complete(self) -> bool:
        return True

    @property
    def reason(self) -> None:
        return None


@dataclass(frozen=True, kw_only=True)
class Text:
    """A string or an output stream: kept head and tail, and how much there was.

    ``length`` is the size of the original in ``unit``. A chars ``Text`` keeps
    ``str``; a bytes ``Text`` keeps the original ``bytes``, so ``len(head)``
    and ``len(tail)`` are its kept units and ``omitted`` (derived) is the rest:
    ``length - len(head) - len(tail)``. Renderers decode bytes only when they
    print.

    Fields are keyword-only. ``complete`` is derived too (``reason is None``),
    so a ``Text`` is consistent by construction: building one whose parts do
    not match its unit, that keeps more than ``length``, or that is complete
    with a ``tail`` or omitted units raises ``ValueError``. An empty ``tail``
    may be left out (it becomes ``head[:0]``).

    ``full``: where the whole content was written, if anywhere; ``None`` when
    nothing was spilled (no spill asked for, or the file could not be
    written). ``full.cut``: the file itself holds only the first ``full.kept``
    bytes. ``reason`` says why the kept parts are cut, never whether the
    file is whole.
    """

    head: str | bytes
    tail: str | bytes = ""
    length: int
    unit: Literal["chars", "bytes"]
    reason: Reason | None = None
    full: Spill | None = None

    def __post_init__(self) -> None:
        kept = str if self.unit == "chars" else bytes
        if not self.tail:
            object.__setattr__(self, "tail", self.head[:0])
        if type(self.head) is not kept or type(self.tail) is not kept:
            raise ValueError(f"a {self.unit} Text keeps {kept.__name__} parts")
        if self.omitted < 0:
            raise ValueError("a Text keeps at most length units")
        if self.reason is None and (self.tail or self.omitted):
            raise ValueError("a complete Text keeps everything in head")

    @property
    def complete(self) -> bool:
        """Whether nothing was cut (no ``reason``)."""
        return self.reason is None

    @property
    def omitted(self) -> int:
        """The units left out between ``head`` and ``tail`` (0 when complete)."""
        return self.length - len(self.head) - len(self.tail)


@dataclass(frozen=True)
class Sequence:
    """A list or tuple: ``length`` items originally, ``items`` kept from the start.

    ``tail`` holds items kept from the end when the head/tail split keeps both.
    """

    type_name: str
    length: int | None
    items: tuple[Snapshot, ...]
    reason: Reason | None = None
    tail: tuple[Snapshot, ...] = ()

    @property
    def complete(self) -> bool:
        """Whether nothing was cut (no ``reason``)."""
        return self.reason is None


@dataclass(frozen=True)
class Mapping:
    """A dict: ``length`` entries originally, ``items`` kept from the start.

    ``wrapper``: a framework wrapper from ``capture_fields`` whose every entry
    keeps at least a minimal value of its own type when rendered.
    """

    type_name: str
    length: int | None
    items: tuple[tuple[Key, Snapshot], ...]
    reason: Reason | None = None
    tail: tuple[tuple[Key, Snapshot], ...] = ()
    wrapper: bool = False

    @property
    def complete(self) -> bool:
        """Whether nothing was cut (no ``reason``)."""
        return self.reason is None


@dataclass(frozen=True)
class Object:
    """A Pydantic model, a dataclass or a registered type: its kept fields, in order.

    ``kind``: an optional rendering key set by a registered shape
    (``ndarray_like(..., kind="numpy.ndarray")``); renderers look up per-kind
    hooks by it (``render_python.register(kind)``).
    """

    type_name: str
    fields: tuple[tuple[str, Snapshot], ...]
    reason: Reason | None = None
    module: str = ""
    kind: str | None = None

    @property
    def complete(self) -> bool:
        """Whether nothing was cut (no ``reason``)."""
        return self.reason is None

    def field(self, name: str) -> Snapshot | None:
        """The kept field called ``name``, or ``None`` when it was not kept."""
        for field_name, node in self.fields:
            if field_name == name:
                return node
        return None


@dataclass(frozen=True)
class Unavailable:
    """A value with no content kept, and why (``type_name``: its type, if known).

    ``reason="opaque"``: a value of a type NOOA does not inspect (``module``:
    its type's module, ``""`` when unknown); it renders as
    ``<opaque: type_name>``. Any other reason: a read that failed or was
    refused, or a limit that left no room (``module`` is ``None``).
    """

    reason: Reason
    type_name: str = ""
    module: str | None = None

    @property
    def complete(self) -> bool:
        return False


@dataclass(frozen=True)
class Ref:
    """A cyclic reference to an enclosing container.

    ``id`` is the JSON Pointer of that container in the snapshot; ``kind`` is
    the kind of node it points at.
    """

    id: str
    kind: Literal["sequence", "mapping", "object"]

    @property
    def complete(self) -> bool:
        return False

    @property
    def reason(self) -> Reason:
        return "cycle"


Snapshot = Scalar | Text | Sequence | Mapping | Object | Unavailable | Ref
"""One node of a snapshot tree."""
