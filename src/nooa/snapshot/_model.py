# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The snapshot nodes: small, detached, frozen trees that describe a value or an output.

The public node classes (``Scalar``, ``Text``, ``Sequence``, ``Mapping``,
``Object``, ``Unavailable``, ``Ref``), their union ``Snapshot``, the
``Reason`` names and ``Overflow``. This module imports nothing else from the
package, so producers (``_walk``, ``_stream``) and renderers (``_render``)
share it without depending on each other.

A snapshot holds no live objects: only built-in values and other nodes.
Every node has ``complete`` and ``reason``; ``complete`` is
``reason is None``. Both describe the node itself, not its descendants: a
complete ``Sequence`` keeps every item, but an item may itself be cut.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "Mapping",
    "Object",
    "Reason",
    "Ref",
    "Scalar",
    "Sequence",
    "Snapshot",
    "Overflow",
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
"""Why a node is incomplete (``node.reason``; ``None`` on a complete node).

- ``"max_length"``: ``Limits.max_chars`` or ``Limits.max_string`` cut it
  (for a ``Text``: it keeps a head and a tail).
- ``"max_items"``: ``Limits.max_items`` left out items or fields.
- ``"max_depth"``: the node is at ``Limits.max_depth``; it keeps no entries.
- ``"max_nodes"``: ``Limits.max_nodes`` (the inspection work) ran out.
- ``"cycle"``: a reference to an enclosing container (only on ``Ref``).
- ``"opaque"``: a value of a type capture does not inspect (only on
  ``Unavailable``).
- ``"unreadable"``: an object's fields, or a field's hidden marker or
  annotation, could not be read without running user code, or Python could
  not evaluate the class's annotations. The affected fields are left out
  (fail closed).
- ``"key"``: a mapping stopped at a key with no JSON object key form, or one
  that collides with an earlier key.
- ``"error"``: a shape accessor raised, or returned something its shape
  cannot use.

Whether an overflow file holds the whole content is not a reason: ``Text.full``
says it. A reason is local to its node: a parent of a cut node may have none.
"""


@dataclass(frozen=True)
class Overflow:
    """Where the whole content of a ``Text`` was written: ``Text.full``.

    Fields:

    - ``path``: the file.
    - ``kept``: the bytes written to the file, from the start.
    - ``total``: the bytes the stream produced.

    ``cut`` (``kept < total``): the file is itself cut; it holds only the
    first ``kept`` bytes. The file is UTF-8 for a chars stream (except that
    a ``surrogateescape`` character is written as its original byte, so it
    need not be valid UTF-8) and the raw bytes for a bytes stream.
    """

    path: str  # the overflow file
    kept: int  # bytes written to the file, from the start
    total: int  # bytes the stream produced

    @property
    def cut(self) -> bool:
        """Whether the file holds less than everything that was produced."""
        return self.kept < self.total


@dataclass(frozen=True)
class Scalar:
    """A complete ``None``, ``bool``, ``int``, ``float`` or short ``bytes`` value.

    ``value``: the value itself (exact built-in types only). A ``float`` is
    finite (``nan`` and infinities are ``Unavailable("opaque")``). Bytes are
    a ``Scalar`` only when kept whole; cut bytes are a bytes ``Text``.
    ``complete`` is always true and ``reason`` always ``None``.
    """

    value: None | bool | int | float | bytes

    @property
    def complete(self) -> bool:
        """Always true: a ``Scalar`` is never cut."""
        return True

    @property
    def reason(self) -> None:
        """Always ``None``."""
        return None


@dataclass(frozen=True, kw_only=True)
class Text:
    """A string, bytes or an output stream: the kept head and tail, and the original size.

    Fields (keyword-only):

    - ``head``: the kept start. ``str`` when ``unit="chars"``; the original
      ``bytes`` when ``unit="bytes"`` (never decoded).
    - ``tail``: the kept end, of the same type; empty when the text is
      complete. Defaults to empty (``head[:0]``).
    - ``length``: the size of the original, in ``unit``.
    - ``unit``: ``"chars"`` or ``"bytes"``.
    - ``reason``: ``None`` when ``head`` holds everything, else why it was
      cut: ``"max_length"`` (a limit or a stream's ``head`` / ``tail``) or
      ``"max_nodes"`` (capture had no inspection work left; nothing kept).
      ``merge`` keeps the first reason of its inputs.
    - ``full``: the ``Overflow`` file holding the whole content, or ``None``
      when nothing went to a file (no file requested, or the file could not be
      written). ``full.cut``: the file holds only the first ``full.kept``
      bytes.

    Derived: ``complete`` is ``reason is None`` (whether this text was
    cut; it says nothing of the node that holds it); ``omitted`` is
    ``length - len(head) - len(tail)``, the units left out between ``head``
    and ``tail``. Counts are exact in the text's own unit. A bytes cut made
    by this package ends and starts on UTF-8 code point boundaries.
    Renderers decode bytes (UTF-8, with replacement) only when they print.

    Raises ``ValueError`` when ``unit`` is neither ``"chars"`` nor
    ``"bytes"``, or when the parts are inconsistent: a part of the wrong type
    for ``unit``, more kept units than ``length``, or a complete text with a
    ``tail`` or omitted units.

    >>> t = Text(head="ab", tail="yz", length=10, unit="chars", reason="max_length")
    >>> t.omitted, t.complete
    (6, False)
    """

    head: str | bytes
    tail: str | bytes = ""
    length: int
    unit: Literal["chars", "bytes"]
    reason: Reason | None = None
    full: Overflow | None = None

    def __post_init__(self) -> None:
        """Check the fields (see the class docstring) and default ``tail`` to an empty part of ``head``'s type."""
        if self.unit != "chars" and self.unit != "bytes":
            raise ValueError(f'unit must be "chars" or "bytes", not {self.unit!r}')
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
    """A list, a tuple or an ``array_like`` value: the kept ``head`` and ``tail`` items, the original count.

    Fields:

    - ``type_name``: ``"list"`` or ``"tuple"``, the class name of a
      standard container read natively (``set``, ``frozenset``, a ``list``
      subclass...) or of a value registered as ``array_like`` without
      fields; ``"list"`` for the ``items``, ``data`` and ``labels`` parts of
      a shape.
    - ``length``: the number of items in the original.
    - ``head``: the items kept from the start, as nodes.
    - ``reason``: ``None`` when every item is kept, else ``"max_items"``,
      ``"max_length"``, ``"max_nodes"``, ``"max_depth"`` (no items kept) or
      ``"error"`` (a shape's ``items`` raised while being read). A kept item
      may itself be cut: look at its own ``reason``.
    - ``tail``: the items kept from the end, when ``Limits.tail_share``
      keeps both ends (empty otherwise).

    As in ``Text``: ``head`` is the kept start, ``tail`` the kept end and
    ``length`` the original count. ``complete`` is ``reason is None``.

    >>> s = Sequence("list", 5, (Scalar(1),), "max_items", (Scalar(5),))
    >>> s.complete, s.length - len(s.head) - len(s.tail)
    (False, 3)
    """

    type_name: str
    length: int
    head: tuple[Snapshot, ...]
    reason: Reason | None = None
    tail: tuple[Snapshot, ...] = ()

    @property
    def complete(self) -> bool:
        """Whether nothing was cut (no ``reason``)."""
        return self.reason is None


@dataclass(frozen=True)
class Mapping:
    """A dict, a ``mapping_like`` value or a ``capture_fields`` wrapper: kept ``head`` and ``tail`` entries, the count.

    Fields:

    - ``type_name``: ``"dict"``, the class name of a standard mapping read
      natively (``Counter``, ``OrderedDict``, ``defaultdict``, a ``dict``
      subclass...) or of a value registered as ``mapping_like`` without
      fields.
    - ``length``: the number of entries in the original.
    - ``head``: the ``(key, node)`` entries kept from the start. Keys are
      exact ``str``, ``int``, ``float``, ``bool`` or ``None``, or a small
      snapshot node of another key (see ``Key``).
    - ``reason``: ``None`` when every entry is kept, else ``"max_items"``,
      ``"max_length"``, ``"max_nodes"``, ``"max_depth"`` (no entries kept),
      ``"key"`` (it stopped at a key with no JSON form, or one whose JSON
      form an earlier kept key has, like ``1`` after ``"1"``) or ``"error"`` (a
      shape's ``items`` raised, or yielded something other than a pair). A
      kept entry's value may itself be cut: look at its own ``reason``.
    - ``tail``: the entries kept from the end, when ``Limits.tail_share``
      keeps both ends (empty otherwise).
    - ``wrapper``: true for the root of ``capture_fields``; ``render_json``
      then keeps every entry with at least a minimal value of its own type.

    As in ``Text``: ``head`` is the kept start, ``tail`` the kept end and
    ``length`` the original count. ``complete`` is ``reason is None``.
    """

    type_name: str
    length: int
    head: tuple[tuple[Key, Snapshot], ...]
    reason: Reason | None = None
    tail: tuple[tuple[Key, Snapshot], ...] = ()
    wrapper: bool = False

    @property
    def complete(self) -> bool:
        """Whether nothing was cut (no ``reason``)."""
        return self.reason is None


@dataclass(frozen=True)
class Object:
    """A value read as named fields: its kept fields, in order.

    A Pydantic model, a dataclass, a namedtuple, a plain object (its public
    attributes), a registered shape with fields, a standard type read
    natively (``kind`` ``"python.datetime"``, ``"python.enum"``...) or a
    bounded ``repr`` (``kind`` ``"python.repr"``, ``Limits(fallback="repr")``).

    Fields:

    - ``type_name``: the class name of the value (at most 200 characters).
    - ``fields``: the ``(name, node)`` fields kept, in declaration order.
      Hidden, excluded, ``repr=False``, private (``_name``) and
      ``ClassVar`` / ``InitVar`` fields are never present.
    - ``reason``: ``None`` when every visible field is present (a field's
      value may itself be cut: look at its own ``reason``), else
      ``"max_items"``, ``"max_length"``, ``"max_nodes"``, ``"max_depth"``
      (no fields kept) or ``"unreadable"`` (fields left out because they
      could not be read safely; no fields at all when the field list itself
      could not be read). A shape accessor that fails makes only its own
      field ``Unavailable("error")``.
    - ``kind``: the rendering key a shape set (``kind="numpy.ndarray"``), a
      standard ``python.*`` key, or ``None``. ``render_python`` looks up its
      hook by it.
    - ``length``: for a cut plain object, how many instance attributes it
      has (private ones included); ``None`` otherwise.

    ``complete`` is ``reason is None``. ``field(name)`` looks a field up.
    """

    type_name: str
    fields: tuple[tuple[str, Snapshot], ...]
    reason: Reason | None = None
    kind: str | None = None
    length: int | None = None

    @property
    def complete(self) -> bool:
        """Whether nothing was cut (no ``reason``)."""
        return self.reason is None

    def field(self, name: str) -> Snapshot | None:
        """The node of the kept field called ``name``, or ``None`` when it was not kept.

        >>> Object("Point", (("x", Scalar(1)),)).field("x")
        Scalar(value=1)
        """
        for field_name, node in self.fields:
            if field_name == name:
                return node
        return None


@dataclass(frozen=True)
class Unavailable:
    """A value with no content kept, and why.

    Fields:

    - ``reason``: ``"opaque"`` for a value of a type capture does not
      inspect (it renders as ``<opaque: type_name>``); ``"error"`` for a
      shape accessor that failed; ``"max_length"``, ``"max_nodes"`` or
      ``"max_depth"`` for a limit that left no room for even a minimal value
      of the type.
    - ``type_name``: the value's type name when it is known, else ``""``.
      For a built-in type it selects the minimal JSON fallback
      (``""``, ``0``, ``[]``, ...).
    - ``builtin``: true when the opaque value's type is an exact built-in
      one (a non-finite ``float``); where the ``"<opaque: float>"`` label
      does not fit, ``render_json`` falls back to ``0.0``.

    ``complete`` is always false.
    """

    reason: Reason
    type_name: str = ""
    builtin: bool = False

    @property
    def complete(self) -> bool:
        """Always false."""
        return False


@dataclass(frozen=True)
class Ref:
    """A reference back to an enclosing container: a cycle.

    Fields:

    - ``id``: the JSON Pointer (RFC 6901) of that container within the
      snapshot (``""`` is the root).
    - ``kind``: ``"sequence"``, ``"mapping"`` or ``"object"``, the node type
      it points at.

    ``complete`` is always false and ``reason`` always ``"cycle"``. A value
    that appears twice without a cycle is captured twice, not as a ``Ref``.
    """

    id: str
    kind: Literal["sequence", "mapping", "object"]

    @property
    def complete(self) -> bool:
        """Always false."""
        return False

    @property
    def reason(self) -> Reason:
        """Always ``"cycle"``."""
        return "cycle"


Snapshot = Scalar | Text | Sequence | Mapping | Object | Unavailable | Ref
"""One node of a snapshot tree: ``Scalar | Text | Sequence | Mapping | Object | Unavailable | Ref``."""

Key = str | int | float | bool | None | Snapshot
"""A mapping key kept in a snapshot: an exact ``str``, ``int``, ``float``, ``bool`` or ``None``,
or, for any other key (a tuple, an enum member...), its own small snapshot node.

A node key's JSON object key is its ``render_python`` text (``"(1, 2)"``), so
it collides like any other key whose JSON form an earlier kept key has.
"""
