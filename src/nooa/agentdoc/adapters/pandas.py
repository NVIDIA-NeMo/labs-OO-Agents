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

from nooa.agentdoc import spec
from nooa.agentdoc.ext import CallableInfo, TypeInfo

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
