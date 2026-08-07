# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pandas adapter gives shape-aware pformat previews for DataFrame/Series.

Large frames lead with shape and column dtypes and anchor a bounded record
sample to the ends; small frames keep their plain repr.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from nooa.agentdoc import pformat  # noqa: E402

# FormatConfig defaults used by the runtime's context rendering.
KW = {"max_length": 50, "max_string": 500}


@pytest.fixture(autouse=True)
def _register():
    # Reload to re-run the @spec.define_preview decorators even if a sibling
    # agentdoc test cleared the registry (register_all() alone is a no-op once
    # the module is imported).
    import importlib

    import nooa.agentdoc.adapters.pandas as _pandas_adapter

    importlib.reload(_pandas_adapter)


def _big_df(n: int = 500) -> pd.DataFrame:
    return pd.DataFrame({"price": [i / n for i in range(n)], "qty": range(n), "label": ["x"] * n})


class TestSmallValuesUnchanged:
    def test_small_dataframe_is_plain_repr(self):
        df = pd.DataFrame({"a": [1, 2]})
        assert pformat(df, **KW) == repr(df)

    def test_small_series_is_plain_repr(self):
        s = pd.Series([1, 2, 3])
        assert pformat(s, **KW) == repr(s)

    def test_unbudgeted_call_is_plain_repr(self):
        df = _big_df()
        assert pformat(df) == repr(df)

    def test_empty_dataframe_is_plain_repr(self):
        df = pd.DataFrame()
        assert pformat(df, **KW) == repr(df)


class TestDataFrame:
    def test_marker_leads_with_shape_and_column_dtypes(self):
        result = pformat(_big_df(), **KW)
        assert result.startswith("DataFrame(shape=(500, 3), columns=")
        assert "'price': 'float64'" in result
        assert "'qty': 'int64'" in result

    def test_head_tail_record_sample(self):
        result = pformat(_big_df(), **KW)
        assert "[:3]=" in result and "[-2:]=" in result
        assert "'qty': 0" in result  # first record
        assert "'qty': 499" in result  # last record

    def test_middle_rows_are_elided(self):
        result = pformat(_big_df(), **KW)
        assert "'qty': 250" not in result

    def test_wide_frame_triggers_by_cell_count(self):
        # 2 rows × 100 columns: pandas' own repr elides columns, so the
        # count trigger must use df.size, not just row count.
        df = pd.DataFrame([[0] * 100, [1] * 100])
        result = pformat(df, **KW)
        assert result.startswith("DataFrame(shape=(2, 100),")

    def test_preview_stays_bounded_for_huge_frames(self):
        df = pd.DataFrame({f"c{i}": range(10_000) for i in range(50)})
        result = pformat(df, **KW)
        assert result.startswith("DataFrame(shape=(10000, 50),")
        assert len(result) < 3000

    def test_long_cell_strings_are_truncated(self):
        df = pd.DataFrame({"doc": ["long text " * 100] * 200})
        result = pformat(df, **KW)
        assert result.startswith("DataFrame(shape=(200, 1),")
        assert len(result) < 1500


class TestSeries:
    def test_marker_leads_with_len_dtype_name(self):
        s = _big_df()["price"]
        result = pformat(s, **KW)
        assert result.startswith("Series(len=500, dtype=float64, name='price',")

    def test_head_tail_value_sample(self):
        s = pd.Series(range(300))
        result = pformat(s, **KW)
        assert "[:3]=[0, 1, 2]" in result
        assert "[-2:]=[298, 299]" in result

    def test_non_range_index_is_named(self):
        s = pd.Series(range(300), index=pd.date_range("2026-01-01", periods=300), name="hits")
        result = pformat(s, **KW)
        assert "index=DatetimeIndex" in result

    def test_range_index_is_not_named(self):
        result = pformat(pd.Series(range(300)), **KW)
        assert "index=" not in result
