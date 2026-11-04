# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The snapshot model: frozen nodes that say whether they are complete, and why not."""

from __future__ import annotations

import dataclasses
import typing

import pytest

import nooa.snapshot
from nooa.snapshot import (
    REASONS,
    Mapping,
    Object,
    Reason,
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
        "unreadable",
        "key",
        "error",
    )


def test_reasons_is_derived_from_the_reason_literal() -> None:
    assert REASONS == typing.get_args(Reason)


def test_removed_names_are_gone() -> None:
    # P3b: no consumer used them (principle 3). Spill state is read from
    # ``Text.full``; an uninspected value is ``Unavailable("opaque")``.
    assert not hasattr(nooa.snapshot, "Opaque")
    assert "Opaque" not in nooa.snapshot.__all__
    assert "replaced" not in {field.name for field in dataclasses.fields(Text)}
    assert not {"file_limit", "not_kept"} & set(REASONS)
    assert [field.name for field in dataclasses.fields(Unavailable)] == [
        "reason",
        "type_name",
        "module",
    ]


def test_every_node_says_whether_it_is_complete() -> None:
    nodes = [
        (Scalar(1), True, None),
        (Text(head="a", tail="", length=1, unit="chars", complete=True, reason=None), True, None),
        (
            Text(
                head="a",
                tail="z",
                length=9,
                unit="bytes",
                complete=False,
                reason="max_length",
                omitted=7,
            ),
            False,
            "max_length",
        ),
        (Sequence("list", 3, (), False, "max_items"), False, "max_items"),
        (Mapping("dict", 0, (), True, None), True, None),
        (Object("M", (), False, "unreadable"), False, "unreadable"),
        (Unavailable("opaque", "Widget", "pkg"), False, "opaque"),
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
        complete=True,
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
            complete=True,
            reason=None,
            full=Spill("/tmp/x", 1, 1),
        )
    )


def test_text_fields_are_keyword_only() -> None:
    with pytest.raises(TypeError):
        Text("a", "", 1, "chars", True, None)  # type: ignore[misc]


def test_producer_constructors_derive_the_accounting() -> None:
    from nooa.snapshot._model import cut_text, whole_text

    assert whole_text(b"ab", 2, "bytes") == Text(
        head=b"ab", tail=b"", length=2, unit="bytes", complete=True, reason=None
    )
    cut = cut_text("ab", "z", 9, "chars", "max_length")
    assert (cut.complete, cut.reason, cut.omitted) == (False, "max_length", 6)


def test_spill_cut() -> None:
    assert Spill("/tmp/x", 10, 20).cut
    assert not Spill("/tmp/x", 20, 20).cut
