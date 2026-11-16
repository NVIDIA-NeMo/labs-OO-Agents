# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``ndarray_like``: an ndarray-like class (no numpy), registered by type and rendered by a hook."""

from __future__ import annotations

import json
import sys
from typing import Any

import pytest

from nooa.snapshot import (
    Limits,
    Mapping,
    Object,
    Scalar,
    Sequence,
    Text,
    Unavailable,
    capture,
    ndarray_like,
    render_json,
    render_python,
)
from tests.snapshot._extend_support import (
    NDARRAY,
    _FakeNdarray,
    _HugeNdarray,
    _named,
    render_ndarray,
)


def test_ndarray_like_nodes(registered) -> None:
    registered(_FakeNdarray, NDARRAY)
    small = _FakeNdarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3))

    snapshot = capture(small)

    assert snapshot == Object(
        "ndarray",
        (
            (
                "dtype",
                Text(head="float64", tail="", length=7, unit="chars", reason=None),
            ),
            ("shape", Sequence("tuple", 2, (Scalar(2), Scalar(3)))),
            (
                "data",
                Sequence(
                    "list",
                    2,
                    (
                        Sequence("list", 3, (Scalar(1.0), Scalar(2.0), Scalar(3.0))),
                        Sequence("list", 3, (Scalar(4.0), Scalar(5.0), Scalar(6.0))),
                    ),
                ),
            ),
        ),
        None,
        "numpy.ndarray",
    )
    assert sorted(small.read) == [(i, j) for i in range(2) for j in range(3)]
    # The generic notations, no new render code.
    assert render_python(snapshot) == (
        "ndarray(dtype='float64', shape=(2, 3), data=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])"
    )
    assert json.loads(render_json(snapshot, max_chars=1_000)) == {
        "dtype": "float64",
        "shape": [2, 3],
        "data": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    }


def test_ndarray_like_render_hook(registered, hooks) -> None:
    before = set(sys.modules)
    registered(_FakeNdarray, NDARRAY)
    hooks(render_python, "numpy.ndarray", render_ndarray)

    snapshot = capture({"x": _FakeNdarray([1.0, 2.0, 3.0, 4.0], (2, 2))})
    assert render_python(snapshot) == (
        "{'x': array([[1.0, 2.0], [3.0, 4.0]], shape=(2, 2), dtype='float64')}"
    )
    assert set(sys.modules) - before == set()


def test_ndarray_like_failing_hook_falls_back(registered, hooks) -> None:
    registered(_FakeNdarray, NDARRAY)
    hooks(render_python, "numpy.ndarray", lambda node, r: node.fields["shape"])  # a TypeError

    snapshot = capture(_FakeNdarray([1.0], (1,)))
    assert render_python(snapshot) == "ndarray(dtype='float64', shape=(1,), data=[1.0])"


def test_ndarray_like_summarises_the_ends_of_every_axis(registered) -> None:
    registered(_FakeNdarray, NDARRAY)
    array = _HugeNdarray(10**6)  # (1000, 1000)

    snapshot = capture(array, Limits(max_items=4))

    assert isinstance(snapshot, Object)
    data = snapshot.field("data")
    assert isinstance(data, Sequence) and data.length == 1000 and data.reason == "max_items"
    assert len(data.head) == 2 and len(data.tail) == 2
    first, last = data.head[0], data.tail[-1]
    assert isinstance(first, Sequence) and isinstance(last, Sequence)
    assert first.head == (Scalar(0.0), Scalar(1.0)) and first.tail == (
        Scalar(998.0),
        Scalar(999.0),
    )
    assert last.tail[-1] == Scalar(999_999.0)
    # element ran only for the shown elements: 4 rows x 4 columns.
    shown = {(i, j) for i in (0, 1, 998, 999) for j in (0, 1, 998, 999)}
    assert len(array.read) == 16 and set(array.read) == shown
    assert render_python(snapshot).startswith(
        "ndarray(dtype='float64', shape=(1000, 1000), data=list(len=1000, [:2]=[list(len=1000,"
        " [:2]=[0.0, 1.0], [-2:]=[998.0, 999.0])"
    )


def test_ndarray_like_odd_count_gives_the_head_one_more(registered) -> None:
    registered(_FakeNdarray, NDARRAY)
    array = _FakeNdarray([float(i) for i in range(10)], (10,))

    data = capture(array, Limits(max_items=3))
    assert isinstance(data, Object)
    assert data.field("data") == Sequence(
        "list", 10, (Scalar(0.0), Scalar(1.0)), "max_items", (Scalar(9.0),)
    )
    assert sorted(array.read) == [(0,), (1,), (9,)]


@pytest.mark.parametrize("limits", [Limits(max_chars=400), Limits(max_chars=5_000), Limits()])
@pytest.mark.parametrize("size", [10**6, 10**9])
def test_ndarray_like_budget_picks_the_count(registered, limits, size) -> None:
    registered(_FakeNdarray, NDARRAY)
    array = _HugeNdarray(size)

    snapshot = capture(array, limits)

    assert isinstance(snapshot, Object) and snapshot.kind == "numpy.ndarray"
    assert [name for name, _node in snapshot.fields] == ["dtype", "shape", "data"]
    assert snapshot.field("shape") == Sequence("tuple", 2, (Scalar(size // 1000), Scalar(1000)))
    data = snapshot.field("data")
    assert isinstance(data, Sequence) and not data.complete and data.head and data.tail
    rows = [*data.head, *data.tail]
    shown = sum(len(row.head) + len(row.tail) for row in rows if isinstance(row, Sequence))
    assert len(array.read) == shown  # never an element that is not shown
    assert len(array.read) <= limits.max_nodes
    text = render_json(snapshot, max_chars=limits.max_chars)
    assert len(text) <= limits.max_chars
    assert json.loads(text)["shape"] == [size // 1000, 1000]


def test_ndarray_like_scalar_and_empty_axes(registered) -> None:
    registered(_FakeNdarray, NDARRAY)

    scalar = capture(_FakeNdarray([7.0], ()))
    assert isinstance(scalar, Object) and scalar.field("data") == Scalar(7.0)
    empty = _FakeNdarray([], (0, 5))
    node = capture(empty)
    assert isinstance(node, Object)
    assert node.field("data") == Sequence("list", 0, ())
    assert empty.read == []


def test_ndarray_like_labels_keep_only_the_shown_ones(registered) -> None:
    Frame = _named("Frame")
    reads: list[int] = []

    class Columns:
        def __getitem__(self, i: int) -> str:
            reads.append(i)
            return f"c{i}"

    shape = ndarray_like(
        lambda frame: (3, 1000),
        lambda frame, index: index[0] * 1000 + index[1],
        labels={1: lambda frame: Columns(), 0: lambda frame: ["a", "b", "c"]},
        kind="fake.frame",
    )
    registered(Frame, shape)

    snapshot = capture(Frame(), Limits(max_items=4))

    assert isinstance(snapshot, Object)
    assert [name for name, _node in snapshot.fields] == ["shape", "labels", "data"]
    assert snapshot.field("labels") == Mapping(
        "dict",
        2,
        (
            (
                1,
                Sequence(
                    "list",
                    1000,
                    (
                        Text(head="c0", tail="", length=2, unit="chars", reason=None),
                        Text(head="c1", tail="", length=2, unit="chars", reason=None),
                    ),
                    "max_items",
                    (
                        Text(head="c998", tail="", length=4, unit="chars", reason=None),
                        Text(head="c999", tail="", length=4, unit="chars", reason=None),
                    ),
                ),
            ),
            (
                0,
                Sequence(
                    "list",
                    3,
                    tuple(
                        Text(head=c, tail="", length=1, unit="chars", reason=None) for c in "abc"
                    ),
                ),
            ),
        ),
    )
    assert sorted(reads) == [0, 1, 998, 999]
    assert render_python(snapshot).startswith(
        "Frame(shape=(3, 1000), labels={1: list(len=1000, [:2]=['c0', 'c1'], [-2:]=['c998', 'c999']),"
        " 0: ['a', 'b', 'c']}, data=[list(len=1000, [:2]=[0, 1], [-2:]=[998, 999]),"
    )


def test_ndarray_like_errors(registered) -> None:
    Broken = _named("Broken")

    def element(value: Any, index: tuple[int, ...]) -> int:
        if index == (1,):
            raise RuntimeError("boom")
        return index[0]

    def fail(value: Any) -> Any:
        raise RuntimeError("boom")

    registered(Broken, ndarray_like(lambda v: (3,), element, labels={0: fail, 5: lambda v: []}))
    snapshot = capture(Broken())
    assert isinstance(snapshot, Object)
    assert snapshot.field("data") == Sequence(
        "list", 3, (Scalar(0), Unavailable("error"), Scalar(2))
    )
    labels = snapshot.field("labels")
    assert isinstance(labels, Mapping)
    assert labels.head == ((0, Unavailable("error")), (5, Unavailable("error")))  # 5: no such axis

    # A missing axis is an error at every budget, also when a limit cuts it.
    shallow = capture(Broken(), Limits(max_depth=2)).field("labels")  # type: ignore[union-attr]
    assert isinstance(shallow, Mapping)
    assert shallow.head[1] == (5, Unavailable("error"))

    for bad_shape in (fail, lambda v: [3], lambda v: (-1,), lambda v: (True, 2.0)):
        registered(Broken, ndarray_like(bad_shape, element))  # type: ignore[arg-type]
        assert capture(Broken()) == Unavailable("error", "Broken")


def test_ndarray_like_is_built_once_in_its_slot(registered) -> None:
    registered(_FakeNdarray, NDARRAY)
    nested = capture([[_FakeNdarray([1.0, 2.0], (2,))]], Limits(max_depth=3))
    assert render_python(nested) == (
        "[[ndarray(dtype='float64', shape=(tuple: 1 items), data=[list: 2 items])]]"
    )
