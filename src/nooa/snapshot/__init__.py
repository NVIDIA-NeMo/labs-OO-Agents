# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, detached snapshots of large values and output streams, and their renderers.

A snapshot is a small frozen tree of nodes (``Scalar``, ``Text``,
``Sequence``, ``Mapping``, ``Object``, ``Unavailable``, ``Ref``; their
union is ``Snapshot``). It holds no live objects. Every node says whether it
is complete and, if not, why (``Reason``); a ``Text`` also says how much was
left out and where the whole content is (``Overflow``).

Producers:

- ``capture(value, limits=Limits())``: a finished Python value. Capture
  reads values natively and runs none of their methods; the code that does
  run is class metadata (annotations), the accessors of a shape you
  registered, and, only with ``Limits(fallback="repr")``, the value's
  ``repr``. ``capture.register(T, shape)`` describes another type with one
  of the shapes ``array_like``, ``ndarray_like``, ``mapping_like``,
  ``object_like`` or ``text_like``. ``capture_fields(**fields)`` captures
  named parts (a call's arguments) so that every part survives.
- ``TextStream``: output that arrives over time, with an optional
  overflow file in an ``OverflowDir`` (``nooa.overflow``, which bounds the
  files of many streams), opened only once the output is cut;
  ``merge`` joins two ``Text``s (progress batches).
- ``cut(data, head=, tail=)``: a string or bytes that is already whole.

Renderers: ``render_output`` (command output), ``render_python``
(agentdoc's prefill notation; its hooks receive a ``PythonRenderer``),
``render_json`` (plain JSON within a size) and ``render_trace`` (that JSON
plus the trace span attributes saying what was cut).

Every public name's docstring is its documentation; ``docs/snapshot-model.md``
is the short contract.

>>> from nooa.snapshot import Limits, capture, render_python
>>> render_python(capture({"rows": list(range(1000))}, Limits(max_items=4, tail_share=0.5)))
"{'rows': list(len=1000, [:2]=[0, 1], [-2:]=[998, 999])}"
"""

from nooa.snapshot._capture import (
    Limits,
    array_like,
    capture,
    capture_fields,
    mapping_like,
    ndarray_like,
    object_like,
    text_like,
)
from nooa.snapshot._model import (
    Mapping,
    Object,
    Overflow,
    Reason,
    Ref,
    Scalar,
    Sequence,
    Snapshot,
    Text,
    Unavailable,
)
from nooa.snapshot._render import (
    PythonRenderer,
    render_json,
    render_output,
    render_python,
    render_trace,
)
from nooa.snapshot._stream import TextStream
from nooa.snapshot._text import cut, merge

__all__ = [
    "Limits",
    "Mapping",
    "Object",
    "PythonRenderer",
    "Reason",
    "Ref",
    "Scalar",
    "Sequence",
    "Overflow",
    "Snapshot",
    "Text",
    "TextStream",
    "Unavailable",
    "array_like",
    "capture",
    "capture_fields",
    "cut",
    "mapping_like",
    "merge",
    "ndarray_like",
    "object_like",
    "render_json",
    "render_output",
    "render_python",
    "render_trace",
    "text_like",
]
