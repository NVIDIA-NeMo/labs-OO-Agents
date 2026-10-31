# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The snapshot model: small, detached, frozen trees describing a value or an output.

A snapshot holds no live objects. Every node says whether it is complete and,
if not, why (``reason``). Renderers (``nooa.snapshot.render_*``) turn a
snapshot into the notation a consumer shows; see ``docs/snapshot-model.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "REASONS",
    "Key",
    "Mapping",
    "Object",
    "Opaque",
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
    "file_limit",
    "not_kept",
    "error",
]
"""Why a node is incomplete. ``error``: a registered capture function raised."""

REASONS: tuple[str, ...] = (
    "max_length",
    "max_items",
    "max_depth",
    "max_nodes",
    "cycle",
    "opaque",
    "file_limit",
    "not_kept",
    "error",
)

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


@dataclass(frozen=True)
class Text:
    """A string or an output stream: kept head and tail, and how much there was.

    ``length`` is the size of the original in ``unit``; ``omitted`` the units
    left out between ``head`` and ``tail`` (0 when complete). ``replaced``
    counts replacement characters that decoding introduced (``unit == "bytes"``).

    ``head_units`` / ``tail_units``: the original units ``head`` and ``tail``
    stand for. A bytes ``Text`` needs them: a U+FFFD that decoding introduced
    stands for 1 to 3 original bytes, a genuine one for 3, and both re-encode
    to 3. ``TextStream``, ``capture`` and ``merge`` set them on every bytes
    ``Text``; ``None`` (chars, or built by hand) derives them, see ``units()``.
    For a ``Text`` they produce, ``length == head_units + tail_units + omitted``.
    """

    head: str
    tail: str
    length: int
    unit: Literal["chars", "bytes"]
    complete: bool
    reason: Reason | None
    full: Spill | None = None
    replaced: int = 0
    omitted: int = 0
    head_units: int | None = None
    tail_units: int | None = None

    def units(self) -> tuple[int, int]:
        """The original units of ``head`` and ``tail``: the recorded counts, else derived.

        Derived: ``len`` for chars, the UTF-8 length for bytes; a bytes
        ``Text`` without a tail derives its head as everything kept
        (``length - omitted``), which is exact.
        """
        head, tail = self.head_units, self.tail_units
        if head is not None and tail is not None:
            return head, tail
        if self.unit == "chars":
            derived_head, derived_tail = len(self.head), len(self.tail)
        else:
            derived_tail = len(self.tail.encode("utf-8", errors="surrogateescape"))
            if self.tail or tail is not None:
                derived_head = len(self.head.encode("utf-8", errors="surrogateescape"))
            else:
                derived_head = max(0, self.length - (0 if self.complete else self.omitted))
        return (derived_head if head is None else head), (derived_tail if tail is None else tail)


@dataclass(frozen=True)
class Sequence:
    """A list or tuple: ``length`` items originally, ``items`` kept from the start.

    ``tail`` holds items kept from the end when the head/tail split keeps both.
    """

    type_name: str
    length: int | None
    items: tuple[Snapshot, ...]
    complete: bool
    reason: Reason | None
    tail: tuple[Snapshot, ...] = ()


@dataclass(frozen=True)
class Mapping:
    """A dict: ``length`` entries originally, ``items`` kept from the start.

    ``wrapper``: a framework wrapper from ``capture_fields`` whose every entry
    keeps at least a minimal value of its own type when rendered.
    """

    type_name: str
    length: int | None
    items: tuple[tuple[Key, Snapshot], ...]
    complete: bool
    reason: Reason | None
    tail: tuple[tuple[Key, Snapshot], ...] = ()
    wrapper: bool = False


@dataclass(frozen=True)
class Object:
    """A Pydantic model, a dataclass or a registered type: its kept fields, in order.

    ``kind``: an optional rendering key set by a registered capture function
    (``out.object(..., kind="numpy.ndarray")``); renderers look up per-kind
    hooks by it (``render_python.register(kind)``).
    """

    type_name: str
    fields: tuple[tuple[str, Snapshot], ...]
    complete: bool
    reason: Reason | None
    module: str = ""
    kind: str | None = None

    def field(self, name: str) -> Snapshot | None:
        """The kept field called ``name``, or ``None`` when it was not kept."""
        for field_name, node in self.fields:
            if field_name == name:
                return node
        return None


@dataclass(frozen=True)
class Opaque:
    """A value of a type that NOOA does not inspect."""

    module: str
    type_name: str

    @property
    def complete(self) -> bool:
        return False

    @property
    def reason(self) -> Reason:
        return "opaque"


@dataclass(frozen=True)
class Unavailable:
    """A read that failed or was refused (``type_name``: the value's type, if known)."""

    reason: Reason
    type_name: str = ""

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


Snapshot = Scalar | Text | Sequence | Mapping | Object | Opaque | Unavailable | Ref
"""One node of a snapshot tree."""
