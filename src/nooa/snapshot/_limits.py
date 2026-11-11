# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The budget of one capture: the public ``Limits`` and the private traversal ``_State``.

The lowest of the capture layers (``_limits`` <- ``_raw`` <- {``_fields``,
``_shapes``, ``_std``} <- ``_walk`` <- ``_capture``). Every layer charges the same
``_State``, so it lives below them all. Nothing here reads a live value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class Limits:
    """The one shared budget of a ``capture`` or ``capture_fields`` call.

    Every knob bounds the whole snapshot, not one value. The defaults are
    the bounds of #342's trace previews.

    - ``max_chars`` (default 50,000): the size of the snapshot as canonical
      JSON (``render_json``'s encoding), counted exactly, so
      ``render_json(capture(v, limits), max_chars=limits.max_chars)`` cuts
      nothing further. At least 4.
    - ``max_nodes`` (default 2,000): inspection work. Each value, container
      entry, MRO entry, stored instance field and visibility marker read
      costs one unit. Greater than 0.
    - ``max_depth`` (default 16): nesting. The root is depth 0; a container
      at ``max_depth`` keeps no entries (reason ``"max_depth"``).
      Non-negative.
    - ``max_items`` (default ``None``): the items kept per container, and
      the fields kept per object (the first ones). ``None``: only the shared
      budget limits them. Non-negative.
    - ``max_string`` (default ``None``): the units kept per string or bytes
      value, an upper bound (a cut byte value keeps only whole UTF-8 code
      points). ``None``: only the shared budget limits it. Non-negative.
    - ``tail_share`` (default 0.0): the part of a cut string or container
      kept from its end, whether ``max_items``, ``max_string`` or the shared
      ``max_chars`` budget cuts it. 0.0 keeps a prefix (trace previews);
      0.5 keeps both ends (agentdoc's split; the head takes the odd unit).
      A container the budget cuts keeps its head within the budget less
      ``tail_share`` of it, and its tail from the end with the rest; one
      that fits is kept whole. A shape read only forwards (``array_like``,
      ``mapping_like``, a ``set``) keeps a head only. From 0 to 1.
    - ``fallback`` (default ``"opaque"``): what a value capture cannot read
      becomes. ``"opaque"``: ``Unavailable("opaque")``, rendered
      ``<opaque: module.QualName>`` (traces: no code runs). ``"repr"``: the
      caller asks capture to run ``repr(value)``, and the result is kept as
      an ``Object`` with ``kind="python.repr"`` and one ``repr`` field, a
      ``Text`` cut like any string (``max_string``, ``tail_share``, the
      budget). This runs the value's own code, unbounded in time and memory
      before the cut: it is for model-facing previews, where a bounded view
      beats ``<opaque: T>`` (which a model answers with ``print(x)``). A
      ``repr`` that raises gives the opaque node.

    ``Limits`` does not check itself: ``capture`` and ``capture_fields``
    raise ``ValueError`` for values outside these ranges.

    >>> from nooa.snapshot import capture, render_python
    >>> render_python(capture(list(range(100)), Limits(max_items=4, tail_share=0.5)))
    'list(len=100, [:2]=[0, 1], [-2:]=[98, 99])'
    """

    max_chars: int = 50_000
    max_nodes: int = 2_000
    max_depth: int = 16
    max_items: int | None = None
    max_string: int | None = None
    tail_share: float = 0.0
    fallback: Literal["opaque", "repr"] = "opaque"


def _split(limits: Limits, kept: int) -> tuple[int, int]:
    """``kept`` units as (head, tail) under ``limits.tail_share``."""
    tail = int(kept * limits.tail_share)
    return kept - tail, tail


DEFAULT_LIMITS = Limits()


def _validate_limits(limits: Limits) -> None:
    """Reject configurations that cannot produce a useful valid root value."""
    if limits.max_chars < 4:
        raise ValueError("snapshot max_chars must be at least 4")
    if limits.max_nodes <= 0:
        raise ValueError("snapshot max_nodes must be greater than 0")
    if limits.max_depth < 0:
        raise ValueError("snapshot max_depth must be non-negative")
    if limits.max_items is not None and limits.max_items < 0:
        raise ValueError("snapshot max_items must be non-negative or None")
    if limits.max_string is not None and limits.max_string < 0:
        raise ValueError("snapshot max_string must be non-negative or None")
    if not 0.0 <= limits.tail_share <= 1.0:
        raise ValueError("snapshot tail_share must be between 0 and 1")
    if limits.fallback != "opaque" and limits.fallback != "repr":
        raise ValueError('snapshot fallback must be "opaque" or "repr"')


class _State:
    """Mutable bounded traversal state shared by one capture."""

    def __init__(self, limits: Limits) -> None:
        self.limits = limits
        self.nodes_left = limits.max_nodes
        # id(container) -> (JSON Pointer, kind) of the containers being built.
        self.active: dict[int, tuple[str, Literal["sequence", "mapping", "object"]]] = {}
        # id(owner) -> (owner, checked copy of its namespace or None); see ``_safe_dict``.
        self.namespaces: dict[int, tuple[Any, dict[str, Any] | None]] = {}
        # id(class) -> (class, the metadata of the hints that resolved one by one)
        # for a class whose ``get_type_hints`` failed in this capture; kept alive
        # for the call and retried by the next one (see ``_fields``).
        self.unreadable_hints: dict[int, tuple[type[Any], dict[str, tuple[Any, ...]]]] = {}

    def take_node(self) -> bool:
        """Consume one inspection unit, returning false when exhausted."""
        if self.nodes_left <= 0:
            return False
        self.nodes_left -= 1
        return True
