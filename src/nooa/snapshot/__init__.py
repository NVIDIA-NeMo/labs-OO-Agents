# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One snapshot model for values, output streams and spilled files.

A snapshot is a small, detached tree (``Scalar``, ``Text``, ``Sequence``,
``Mapping``, ``Object``, ``Opaque``, ``Unavailable``, ``Ref``) in which every
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
output), ``render_python`` (agentdoc's prefill notation), ``render_json`` and
``preview_attributes`` (trace previews). See ``docs/snapshot-model.md``.
"""

from nooa.snapshot._capture import (
    DEFAULT_LIMITS,
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
    REASONS,
    Key,
    Mapping,
    Object,
    Opaque,
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
    JsonRenderer,
    OutputRenderer,
    PythonRenderer,
    preview_attributes,
    render_json,
    render_output,
    render_python,
)
from nooa.snapshot._stream import DEFAULT_SPILL_BYTES, SpillTarget, TextStream, merge

__all__ = [
    "DEFAULT_LIMITS",
    "DEFAULT_SPILL_BYTES",
    "JsonRenderer",
    "REASONS",
    "Key",
    "Limits",
    "Mapping",
    "Object",
    "Opaque",
    "OutputRenderer",
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
    "preview_attributes",
    "render_json",
    "render_output",
    "render_python",
    "text_like",
]
