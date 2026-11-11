# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The snapshot model: frozen nodes that say whether they are complete, and why not."""

from __future__ import annotations

import dataclasses
import random
import typing
from typing import Any

import pytest

import nooa.snapshot
from nooa.snapshot import (
    Limits,
    Mapping,
    Object,
    Reason,
    Ref,
    Scalar,
    Sequence,
    Spill,
    Text,
    Unavailable,
    capture,
)
from tests.snapshot._random_trees import random_tree


def test_reason_names() -> None:
    assert typing.get_args(Reason) == (
        "max_length",
        "max_items",
        "max_depth",
        "max_nodes",
        "cycle",
        "opaque",
        "unreadable",
        "key",
        "error",
    )


def test_removed_names_are_gone() -> None:
    # P3b: no consumer used them (principle 3). Spill state is read from
    # ``Text.full``; an uninspected value is ``Unavailable("opaque")``.
    assert not hasattr(nooa.snapshot, "Opaque")
    assert "Opaque" not in nooa.snapshot.__all__
    assert "replaced" not in {field.name for field in dataclasses.fields(Text)}
    assert not {"file_limit", "not_kept"} & set(typing.get_args(Reason))
    assert [field.name for field in dataclasses.fields(Unavailable)] == [
        "reason",
        "type_name",
        "builtin",
    ]


def test_every_node_says_whether_it_is_complete() -> None:
    nodes = [
        (Scalar(1), True, None),
        (Text(head="a", length=1, unit="chars"), True, None),
        (
            Text(head="a", tail="z", length=9, unit="chars", reason="max_length"),
            False,
            "max_length",
        ),
        (Sequence("list", 3, (), "max_items"), False, "max_items"),
        (Mapping("dict", 0, ()), True, None),
        (Object("M", (), "unreadable"), False, "unreadable"),
        (Unavailable("opaque", "Widget"), False, "opaque"),
        (Unavailable("error", "Widget"), False, "error"),
        (Ref("/a", "mapping"), False, "cycle"),
    ]
    for node, complete, reason in nodes:
        assert node.complete is complete
        assert node.reason == reason


def test_nodes_are_frozen_and_hashable() -> None:
    text = Text(
        head="a",
        tail="",
        length=1,
        unit="chars",
        reason=None,
        full=Spill("/tmp/x", 1, 1),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        text.head = "b"  # type: ignore[misc]
    assert hash(text) == hash(
        Text(
            head="a",
            tail="",
            length=1,
            unit="chars",
            reason=None,
            full=Spill("/tmp/x", 1, 1),
        )
    )


def test_containers_read_like_text_head_tail_length() -> None:
    # Paul and Severin (#515): all three node types say ``head``, ``tail`` and ``length``.
    assert [f.name for f in dataclasses.fields(Sequence)] == [
        "type_name",
        "length",
        "head",
        "reason",
        "tail",
    ]
    assert [f.name for f in dataclasses.fields(Mapping)][:5] == [
        "type_name",
        "length",
        "head",
        "reason",
        "tail",
    ]
    cut = capture(list(range(10)), Limits(max_items=4, tail_share=0.5))
    assert isinstance(cut, Sequence)
    assert (cut.head, cut.tail, cut.length) == ((Scalar(0), Scalar(1)), (Scalar(8), Scalar(9)), 10)
    assert not hasattr(cut, "items")


def test_text_fields_are_keyword_only() -> None:
    with pytest.raises(TypeError):
        Text("a", "", 1, "chars")  # type: ignore[misc]


def test_text_derives_complete_and_omitted() -> None:
    whole = Text(head=b"ab", length=2, unit="bytes")
    assert (whole.tail, whole.complete, whole.omitted) == (b"", True, 0)
    cut = Text(head="ab", tail="z", length=9, unit="chars", reason="max_length")
    assert (cut.complete, cut.reason, cut.omitted) == (False, "max_length", 6)


@pytest.mark.parametrize(
    "fields",
    [
        {"head": "ab", "length": 2, "unit": "bytes"},  # str parts in a bytes Text
        {"head": b"ab", "tail": "z", "length": 9, "unit": "bytes", "reason": "max_length"},
        {"head": "abc", "length": 2, "unit": "chars", "reason": "max_length"},  # keeps > length
        {"head": "ab", "tail": "z", "length": 3, "unit": "chars"},  # complete with a tail
        {"head": "ab", "length": 9, "unit": "chars"},  # complete with omitted units
        {"head": b"a", "length": 3, "unit": "octets", "reason": "max_length"},  # unknown unit
        {"head": "a", "length": 1, "unit": "lines"},  # unknown unit
    ],
)
def test_an_inconsistent_text_cannot_be_built(fields: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Text(**fields)


def test_random_trees_are_consistent() -> None:
    for seed in range(300):
        stack = [random_tree(random.Random(seed))]
        while stack:
            node = stack.pop()
            assert node.complete is (node.reason is None)
            if isinstance(node, Text):
                assert node.omitted >= 0
            elif isinstance(node, Sequence):
                stack += [*node.head, *node.tail]
            elif isinstance(node, Mapping):
                stack += [child for _key, child in (*node.head, *node.tail)]
            elif isinstance(node, Object):
                stack += [child for _name, child in node.fields]


def test_spill_cut() -> None:
    assert Spill("/tmp/x", 10, 20).cut
    assert not Spill("/tmp/x", 20, 20).cut
