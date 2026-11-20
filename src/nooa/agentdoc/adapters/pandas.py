# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""agentdoc adapter for pandas — concise doc() views for DataFrame and Series.

Import to register:

    import nooa.agentdoc.adapters.pandas

Without this adapter, ``doc(pd.DataFrame)`` is pandas' full ~50-line constructor
docstring. With it, it's a few lines focused on how to *construct* the object — the
information a CodeAct agent needs when a method returns ``pd.DataFrame`` (which falls
back to an ``Any`` tool schema, so the model has nothing structural to go on otherwise).
"""

import pandas as pd

from nooa.agentdoc import pformat, spec
from nooa.agentdoc.ext import CallableInfo, PreviewBudget, TypeInfo

_DATAFRAME_DOC = (
    "pandas DataFrame — 2-D labelled, column-oriented tabular data.\n\n"
    "Construct:\n"
    "    pd.DataFrame({'col_a': [1, 2], 'col_b': ['x', 'y']})   # dict of columns (most common)\n"
    "    pd.DataFrame([{'a': 1, 'b': 2}, {'a': 3, 'b': 4}])      # list of row dicts\n"
    "    pd.DataFrame(rows, columns=['a', 'b'])                  # 2-D array/list + column names\n\n"
    "Common ops: df['col'], df.loc[mask], df.groupby('k').agg(...), df.merge(other, on='k'), "
    "df.sort_values('col'), df.head(n), df.shape, df.columns, df.to_dict(orient='records')."
)

_SERIES_DOC = (
    "pandas Series — 1-D labelled array (a single DataFrame column).\n\n"
    "Construct:\n"
    "    pd.Series([1, 2, 3], name='col')\n"
    "    pd.Series({'a': 1, 'b': 2})            # index from dict keys\n\n"
    "Common ops: s.sum(), s.mean(), s.value_counts(), s.map(fn), s.to_list(), s.index, s.name."
)


@spec.define_doc(pd.DataFrame)
def _dataframe_doc(cls_or_instance) -> TypeInfo:
    return TypeInfo(
        name="DataFrame",
        base="pandas.DataFrame",
        fields=[],  # columns are dynamic, not fixed fields
        methods=[
            CallableInfo(
                name="from_records",
                signature="(data, columns=None)",
                return_type="DataFrame",
                docstring="Build a DataFrame from a list of row records (dicts or tuples).",
                is_classmethod=True,
            ),
            CallableInfo(
                name="to_dict",
                signature="(orient='records')",
                return_type="list[dict] | dict",
                docstring="Serialize to plain Python (e.g. orient='records' → list of row dicts).",
            ),
        ],
        docstring=_DATAFRAME_DOC,
    )


@spec.define_doc(pd.Series)
def _series_doc(cls_or_instance) -> TypeInfo:
    return TypeInfo(
        name="Series",
        base="pandas.Series",
        fields=[],
        methods=[
            CallableInfo(
                name="to_list",
                signature="()",
                return_type="list",
                docstring="Return the values as a plain Python list.",
            ),
        ],
        docstring=_SERIES_DOC,
    )


# --- pformat value previews -------------------------------------------------
#
# Large frames otherwise render as a truncated repr slice full of escaped
# newlines with the shape invisible; these previews lead with the structural
# facts (shape, column dtypes) and anchor a bounded row sample to the ends,
# reusing pformat itself so nested values inherit the marker family.

# Row sample budget: rows repeat every column name, so even a handful of
# records dominates the preview's size. Five rows split head/tail.
_ROW_SAMPLE = 5
# Column-dtype entries shown before the dict marker elides the rest.
_COL_SAMPLE = 10
# Per-cell string budget inside row samples: keeps one pathological cell
# (a long document in a column) from consuming the whole preview.
_CELL_STRING = 80


def _complete_within_budget(value, n_items: int, budget: PreviewBudget) -> bool:
    """True when the plain repr is a complete rendering within budget.

    Mirrors the sequence rule: an item count over ``max_length`` triggers the
    marker even when the repr text is short — pandas elides rows and columns
    with ``...`` in its own repr, so a short repr is not evidence of a
    complete value. ``n_items`` is the total cell count for a DataFrame and
    the length for a Series.
    """
    if budget.max_length is not None and n_items > budget.max_length:
        return False
    if budget.max_string is None:
        return True
    try:
        return len(repr(value)) <= budget.max_string
    except Exception:
        return False


def _bounded(value, budget: PreviewBudget, *, max_length: int) -> str:
    """Render a nested sample through pformat with a tightened budget."""
    max_string = _CELL_STRING if budget.max_string is None else min(budget.max_string, _CELL_STRING)
    return pformat(value, max_length=max_length, max_string=max_string, max_depth=3)


def _sample_split(n: int, budget: PreviewBudget) -> tuple[int, int]:
    """Head/tail row counts: ceiling half and floor half of the row budget."""
    k = _ROW_SAMPLE if budget.max_length is None else min(budget.max_length, _ROW_SAMPLE)
    k = max(1, min(k, n))
    n_head = (k + 1) // 2
    return n_head, k - n_head


@spec.define_preview(pd.DataFrame)
def _dataframe_preview(df: pd.DataFrame, budget: PreviewBudget) -> str | None:
    n = len(df)
    if df.empty or _complete_within_budget(df, df.size, budget):
        return None
    cols = {str(name): str(dtype) for name, dtype in df.dtypes.items()}
    header = (
        f"DataFrame(shape={df.shape!r}, columns={_bounded(cols, budget, max_length=_COL_SAMPLE)}"
    )

    n_head, n_tail = _sample_split(n, budget)
    head = _bounded(df.head(n_head).to_dict("records"), budget, max_length=_COL_SAMPLE)
    if n_tail > 0:
        tail = _bounded(df.tail(n_tail).to_dict("records"), budget, max_length=_COL_SAMPLE)
        return f"{header}, [:{n_head}]={head}, [-{n_tail}:]={tail})"
    return f"{header}, [:{n_head}]={head})"


@spec.define_preview(pd.Series)
def _series_preview(s: pd.Series, budget: PreviewBudget) -> str | None:
    n = len(s)
    if s.empty or _complete_within_budget(s, n, budget):
        return None
    header = f"Series(len={n}, dtype={s.dtype}"
    if s.name is not None:
        header += f", name={s.name!r}"
    if not isinstance(s.index, pd.RangeIndex):
        header += f", index={type(s.index).__name__}"

    n_head, n_tail = _sample_split(n, budget)
    head = _bounded(s.head(n_head).to_list(), budget, max_length=_ROW_SAMPLE)
    if n_tail > 0:
        tail = _bounded(s.tail(n_tail).to_list(), budget, max_length=_ROW_SAMPLE)
        return f"{header}, [:{n_head}]={head}, [-{n_tail}:]={tail})"
    return f"{header}, [:{n_head}]={head})"
