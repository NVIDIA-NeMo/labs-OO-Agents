# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A realistic extension: an ndarray-like class (no numpy), captured lazily and rendered by a hook."""

from __future__ import annotations

import json
import sys

import pytest

from nooa.snapshot import (
    Limits,
    Object,
    Scalar,
    Sequence,
    Text,
    capture,
    render_json,
    render_python,
)
from tests.snapshot._extend_support import (
    _FakeNdarray,
    _LazyNdarray,
    capture_ndarray,
    render_ndarray,
)

# --- a realistic example: an ndarray-like class, no numpy ------------------------------


def test_ndarray_like_example(lazy, hooks) -> None:
    before = set(sys.modules)
    lazy("fakenumpy_7c1e", "ndarray", capture_ndarray)
    hooks(render_python, "numpy.ndarray", render_ndarray)

    small = _FakeNdarray([1.0, 2.0, 3.0, 4.0], (2, 2))
    snapshot = capture({"x": small})
    assert render_python(snapshot) == (
        "{'x': array([1.0, 2.0, 3.0, 4.0], shape=(2, 2), dtype='float64')}"
    )
    # No json hook: the trace preview stays plain, generic JSON.
    assert json.loads(render_json(snapshot, max_chars=1_000)) == {
        "x": {"shape": [2, 2], "dtype": "float64", "sample": [1.0, 2.0, 3.0, 4.0]}
    }

    big = _FakeNdarray([float(i) for i in range(1_000_000)], (1000, 1000))
    cut = capture(big, Limits(max_chars=200, max_items=3))
    assert isinstance(cut, Object) and cut.kind == "numpy.ndarray"
    assert render_python(cut) == (
        "array(list(len=?, [:3]=[0.0, 1.0, 2.0]), shape=(1000, 1000), dtype='float64')"
    )
    assert big.pulled < 10  # the budget stopped the sample; the rest was never pulled
    assert len(render_json(cut, max_chars=200)) <= 200
    assert set(sys.modules) - before == set()


def test_ndarray_like_failing_hook_falls_back(lazy, hooks) -> None:
    lazy("fakenumpy_7c1e", "ndarray", capture_ndarray)
    hooks(render_python, "numpy.ndarray", lambda node, r: node.fields["shape"])  # a TypeError

    snapshot = capture(_FakeNdarray([1.0], (1,)))
    assert render_python(snapshot) == "ndarray(shape=(1,), dtype='float64', sample=[1.0])"


@pytest.mark.parametrize("limits", [Limits(max_chars=400), Limits(max_chars=5_000), Limits()])
@pytest.mark.parametrize("size", [10**6, 10**9])
def test_ndarray_like_large_keeps_shape_and_dtype(lazy, hooks, limits, size) -> None:
    lazy("fakenumpy_7c1e", "ndarray", capture_ndarray)
    hooks(render_python, "numpy.ndarray", render_ndarray)
    array = _LazyNdarray(size)

    snapshot = capture(array, limits)

    assert isinstance(snapshot, Object) and snapshot.kind == "numpy.ndarray"
    assert [name for name, _node in snapshot.fields] == ["shape", "dtype", "sample"]
    shape, dtype, sample = (snapshot.field(name) for name in ("shape", "dtype", "sample"))
    assert shape == Sequence("tuple", 2, (Scalar(size // 1000), Scalar(1000)), True, None)
    assert dtype == Text("float64", "", 7, "chars", True, None)
    assert isinstance(sample, Sequence) and not sample.complete
    expected = "max_nodes" if limits.max_chars == 50_000 else "max_length"
    assert sample.reason == expected
    assert 0 < len(sample.items) and array.pulled <= limits.max_nodes
    rendered = render_python(snapshot)
    assert rendered.startswith("array(") and f"shape=({size // 1000}, 1000)" in rendered
    text = render_json(snapshot, max_chars=limits.max_chars)
    assert len(text) <= limits.max_chars
    assert json.loads(text)["shape"] == [size // 1000, 1000]
