# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, detached snapshots of large values and output streams, and their renderers.

A snapshot is a small frozen tree of nodes (``Scalar``, ``Text``,
``Sequence``, ``Mapping``, ``Object``, ``Unavailable``, ``Ref``; their
union is ``Snapshot``). It holds no live objects. Every node says whether it
is complete and, if not, why (``Reason``); a ``Text`` also says how much was
left out and where the whole content is (``Spill``).

Producers:

- ``capture(value, limits=Limits())``: a finished Python value, without
  running code on it. ``capture.register(T, shape)`` describes another type
  with one of the shapes ``array_like``, ``ndarray_like``,
  ``mapping_like``, ``object_like`` or ``text_like``.
  ``capture_fields(**fields)`` captures a framework wrapper.
- ``TextStream``: output that arrives over time, with an optional
  ``SpillTarget`` file; ``SpillDir`` bounds the files of many streams;
  ``merge`` joins two unread deltas.

Renderers: ``render_output`` (command output), ``render_python``
(agentdoc's prefill notation; its hooks receive a ``PythonRenderer``),
``render_json`` and ``render_trace`` (trace previews).

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
    Reason,
    Ref,
    Scalar,
    Sequence,
    Snapshot,
    Spill,
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
from nooa.snapshot._stream import SpillDir, SpillTarget, TextStream, merge

__all__ = [
    "Limits",
    "Mapping",
    "Object",
    "PythonRenderer",
    "Reason",
    "Ref",
    "Scalar",
    "Sequence",
    "Snapshot",
    "Spill",
    "SpillDir",
    "SpillTarget",
    "Text",
    "TextStream",
    "Unavailable",
    "array_like",
    "capture",
    "capture_fields",
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
