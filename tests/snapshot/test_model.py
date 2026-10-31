# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The snapshot model: frozen nodes that say whether they are complete, and why not."""

from __future__ import annotations

import dataclasses

import pytest

from nooa.snapshot import (
    REASONS,
    Mapping,
    Object,
    Opaque,
    Ref,
    Scalar,
    Sequence,
    Spill,
    Text,
    Unavailable,
)


def test_reason_names() -> None:
    assert REASONS == (
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


def test_every_node_says_whether_it_is_complete() -> None:
    nodes = [
        (Scalar(1), True, None),
        (Text("a", "", 1, "chars", True, None), True, None),
        (Text("a", "z", 9, "bytes", False, "max_length", omitted=7), False, "max_length"),
        (Sequence("list", 3, (), False, "max_items"), False, "max_items"),
        (Mapping("dict", 0, (), True, None), True, None),
        (Object("M", (), False, "opaque"), False, "opaque"),
        (Opaque("pkg", "Widget"), False, "opaque"),
        (Unavailable("error", "Widget"), False, "error"),
        (Ref("/a", "mapping"), False, "cycle"),
    ]
    for node, complete, reason in nodes:
        assert node.complete is complete
        assert node.reason == reason


def test_nodes_are_frozen_and_hashable() -> None:
    text = Text("a", "", 1, "chars", True, None, Spill("/tmp/x", 1, 1))
    with pytest.raises(dataclasses.FrozenInstanceError):
        text.head = "b"  # type: ignore[misc]
    assert hash(text) == hash(Text("a", "", 1, "chars", True, None, Spill("/tmp/x", 1, 1)))


def test_spill_cut() -> None:
    assert Spill("/tmp/x", 10, 20).cut
    assert not Spill("/tmp/x", 20, 20).cut
