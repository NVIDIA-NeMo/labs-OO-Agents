# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The numpy adapter gives shape-aware pformat previews for large ndarrays.

Large arrays lead with shape/dtype and anchor a bounded head/tail sample to
axis 0; small arrays keep their plain repr (marker presence = truncation).
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from nooa.agentdoc import pformat  # noqa: E402
from nooa.agentdoc.adapters import register_all  # noqa: E402

# FormatConfig defaults used by the runtime's context rendering.
KW = {"max_length": 50, "max_string": 500}


@pytest.fixture(autouse=True)
def _register():
    # Reload to re-run the @spec.define_preview decorators even if a sibling
    # agentdoc test cleared the registry (register_all() alone is a no-op once
    # the module is imported).
    import importlib

    import nooa.agentdoc.adapters.numpy as _numpy_adapter

    importlib.reload(_numpy_adapter)


class TestRegistration:
    def test_register_all_includes_numpy(self):
        assert "numpy" in register_all()


class TestSmallArraysUnchanged:
    def test_small_array_is_plain_repr(self):
        assert pformat(np.array([1, 2, 3]), **KW) == "array([1, 2, 3])"

    def test_zero_d_array_is_plain_repr(self):
        assert pformat(np.array(7), **KW) == "array(7)"

    def test_unbudgeted_call_is_plain_repr(self):
        # No truncation budget → no preview, matching list behavior.
        assert pformat(np.arange(3)) == "array([0, 1, 2])"


class TestOneDim:
    def test_marker_leads_with_shape_and_dtype(self):
        result = pformat(np.arange(100) * 0.5, max_length=10, max_string=500)
        assert result.startswith("ndarray(shape=(100,), dtype=float64,")

    def test_head_tail_slice_keys(self):
        result = pformat(np.arange(100), max_length=10, max_string=500)
        assert "[:5]=[0, 1, 2, 3, 4]" in result
        assert "[-5:]=[95, 96, 97, 98, 99]" in result

    def test_middle_is_elided(self):
        result = pformat(np.arange(100), max_length=10, max_string=500)
        assert "42" not in result

    def test_object_dtype_keeps_quotes(self):
        result = pformat(np.array(["word"] * 5000, dtype=object), max_length=6, max_string=500)
        assert "'word'" in result
        assert "dtype=object" in result


class TestMultiDim:
    def test_2d_marker_and_row_sample(self):
        a = np.arange(1000).reshape(250, 4) * 1.5
        result = pformat(a, **KW)
        assert result.startswith("ndarray(shape=(250, 4), dtype=float64,")
        assert "[:3]=" in result and "[-3:]=" in result
        assert "1.5" in result and "1498.5" in result
        assert "\n" not in result

    def test_wide_2d_triggers_by_total_size(self):
        # 2 rows but 1M cells: numpy's own repr elides columns, so the
        # count trigger must use arr.size, not just axis-0 length.
        a = np.zeros((2, 1_000_000))
        result = pformat(a, **KW)
        assert result.startswith("ndarray(shape=(2, 1000000),")

    def test_3d_preview_stays_bounded(self):
        result = pformat(np.zeros((10, 20, 30)), **KW)
        assert result.startswith("ndarray(shape=(10, 20, 30), dtype=float64,")
        assert len(result) < 800

    def test_preview_never_materializes_huge_output(self):
        a = np.arange(3_000_000, dtype=np.int64).reshape(1000, 3000)
        result = pformat(a, **KW)
        assert result.startswith("ndarray(shape=(1000, 3000),")
        assert len(result) < 800
