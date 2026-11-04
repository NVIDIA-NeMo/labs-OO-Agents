# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One snapshot model for values, output streams and spilled files.

A snapshot is a small, detached tree (``Scalar``, ``Text``, ``Sequence``,
``Mapping``, ``Object``, ``Unavailable``, ``Ref``) in which every
node says whether it is complete and, if not, why. Two producers build it:

- ``capture(value, limits)`` for finished Python values (no user code runs on
  types it does not know). ``capture.register(T, shape)``, process-wide and
  matched by class identity, extends it with a declarative shape that maps
  out the type's structure: ``array_like``, ``ndarray_like``,
  ``mapping_like``, ``object_like`` or ``text_like``. Capture builds the
  snapshot itself, with the builders and size rules of the built-in types;
- ``TextStream`` for output that arrives over time (head, rolling tail,
  counter and an optional spill file).

Renderers turn it into what a consumer shows: ``render_output`` (command
output), ``render_python`` (agentdoc's prefill notation, the one renderer with
per-kind hooks), ``render_json`` and ``render_trace`` (trace previews: the JSON
text and its span attributes). See ``docs/snapshot-model.md``.
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
from nooa.snapshot._stream import SpillTarget, TextStream, merge

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
