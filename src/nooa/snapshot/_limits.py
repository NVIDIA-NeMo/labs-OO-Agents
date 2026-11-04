# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The budget of one capture: the public ``Limits`` and the traversal ``_State``.

The leaf of the capture modules (``_limits`` <- ``_raw`` <- ``_visibility``
<- ``_shapes`` <- ``_walk`` <- ``_capture``): every layer charges the same
``_State``, so it sits below them all. Nothing here reads a live value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class Limits:
    """One shared budget for a capture.

    ``max_chars`` counts the snapshot's canonical JSON encoding; ``max_nodes``
    inspection work; ``max_depth`` nesting (the root is depth 0).
    ``max_items`` caps the items kept per container and ``max_string`` the
    characters kept per string (``None``: only the shared budget). When a
    string or container is cut, ``tail_share`` of what is kept comes from its
    end (0: a prefix, as in trace previews; 0.5: agentdoc's head/tail split,
    the head taking the odd one).
    """

    max_chars: int = 50_000
    max_nodes: int = 2_000
    max_depth: int = 16
    max_items: int | None = None
    max_string: int | None = None
    tail_share: float = 0.0


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


class _State:
    """Mutable bounded traversal state shared by one capture."""

    def __init__(self, limits: Limits) -> None:
        self.limits = limits
        self.nodes_left = limits.max_nodes
        # id(container) -> (JSON Pointer, kind) of the containers being built.
        self.active: dict[int, tuple[str, Literal["sequence", "mapping", "object"]]] = {}
        # id(owner) -> (owner, checked copy of its namespace or None); see ``_safe_dict``.
        self.namespaces: dict[int, tuple[Any, dict[str, Any] | None]] = {}

    def take_node(self) -> bool:
        """Consume one inspection unit, returning false when exhausted."""
        if self.nodes_left <= 0:
            return False
        self.nodes_left -= 1
        return True
