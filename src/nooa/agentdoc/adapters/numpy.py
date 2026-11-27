# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""agentdoc adapter for numpy — shape-aware pformat previews for ndarray.

Import to register:

    import nooa.agentdoc.adapters.numpy

Without this adapter, a large array renders as a truncated repr slice —
``ndarray(repr_len=10250, [:100]='array([[   0. , ...')`` — escaped newlines
and all, with the shape and dtype invisible. With it, the preview leads with
the structural facts and anchors a bounded head/tail sample to axis 0:

    ndarray(shape=(250, 4), dtype=float64, [:2]=[[0.0, 1.5, 3.0, 4.5], …], [-2:]=[…])

Small arrays (repr within budget) are unaffected: the extractor declines and
the plain repr renders, keeping the marker-presence-signals-truncation rule.
"""

from __future__ import annotations

import numpy as np

from nooa.agentdoc import spec
from nooa.agentdoc.ext import PreviewBudget

# Row/column sample budget for >=2-D arrays. Rows repeat per line, so the
# 1-D rule of "show max_length items" would be token-hungry; six per axis
# keeps the preview a few hundred characters no matter the array's size.
_MATRIX_SAMPLE = 6


def _one_line(text: str) -> str:
    """Collapse numpy's multi-line array text to a single normalized line."""
    return " ".join(text.split()).replace(" ,", ",")


def _subarray_str(elem: np.ndarray, inner_budget: int) -> str:
    """Bounded one-line rendering of one axis-0 element (itself an array).

    Deeper subarrays (>= 2-D) get one edge item per axis: their rendered
    size grows multiplicatively with each axis, so anything wider explodes
    the preview for 3-D+ arrays.
    """
    edgeitems = max(1, inner_budget // 2) if elem.ndim == 1 else 1
    text = np.array2string(
        elem,
        threshold=inner_budget,
        edgeitems=edgeitems,
        separator=", ",
    )
    return _one_line(text)


def _scalar_str(elem: object) -> str:
    """Render one axis-0 element of a 1-D array as a Python literal.

    Iterating a 1-D array yields numpy scalars (``np.float64`` …) for
    numeric dtypes but raw Python objects for ``dtype=object``; unwrap only
    the former so strings keep their quotes.
    """
    if isinstance(elem, np.generic):
        try:
            elem = elem.item()
        except Exception:
            pass
    return repr(elem)


def _sample(rows: np.ndarray, inner_budget: int) -> str:
    """Render a head or tail slice as a bracketed, comma-separated sample."""
    if rows.ndim == 1:
        parts = [_scalar_str(x) for x in rows]
    else:
        parts = [_subarray_str(x, inner_budget) for x in rows]
    return "[" + ", ".join(parts) + "]"


@spec.define_preview(np.ndarray)
def _ndarray_preview(arr: np.ndarray, budget: PreviewBudget) -> str | None:
    if arr.ndim == 0:
        return None  # scalars: repr is already compact

    max_length, max_string = budget
    n = arr.shape[0]

    # Decline when the value renders completely within budget — mirrors the
    # sequence rule (count over max_length OR repr over max_string triggers
    # the marker; otherwise the plain repr is the complete value). Element
    # count uses arr.size: numpy's own repr elides with '...' past its print
    # threshold, so a short repr is not evidence of a complete value.
    over_count = max_length is not None and arr.size > max_length
    if not over_count:
        r = repr(arr)
        if max_string is None or len(r) <= max_string:
            return None

    if arr.ndim == 1:
        k = max_length if max_length is not None else 10
    else:
        k = min(max_length if max_length is not None else _MATRIX_SAMPLE, _MATRIX_SAMPLE)
    k = max(1, min(k, n))
    inner_budget = _MATRIX_SAMPLE if max_length is None else min(max_length, _MATRIX_SAMPLE)

    header = f"ndarray(shape={arr.shape!r}, dtype={arr.dtype}"

    if n <= k:
        return f"{header}, values={_sample(arr, inner_budget)})"

    n_head = (k + 1) // 2
    n_tail = k - n_head
    head = _sample(arr[:n_head], inner_budget)
    if n_tail > 0:
        tail = _sample(arr[-n_tail:], inner_budget)
        return f"{header}, [:{n_head}]={head}, [-{n_tail}:]={tail})"
    return f"{header}, [:{n_head}]={head})"
