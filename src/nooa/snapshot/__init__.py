# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One snapshot model for values, output streams and spilled files.

A snapshot is a small, detached tree (``Scalar``, ``Text``, ``Sequence``,
``Mapping``, ``Object``, ``Opaque``, ``Unavailable``, ``Ref``) in which every
node says whether it is complete and, if not, why. Two producers build it:

- ``capture(value, limits)`` for finished Python values (no user code runs on
  types it does not know; ``capture.register(T)`` and, by name without
  importing, ``capture.register_lazy(module, qualname)`` extend it);
- ``TextStream`` for output that arrives over time (head, rolling tail,
  counter and an optional spill file).

Renderers turn it into what a consumer shows: ``render_output`` (command
output), ``render_python`` (agentdoc's prefill notation), ``render_json`` and
``preview_attributes`` (trace previews). See ``docs/snapshot-model.md``.
"""

from nooa.snapshot._capture import (
    DEFAULT_LIMITS,
    Deferred,
    Limits,
    Out,
    capture,
    capture_fields,
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
    "Deferred",
    "JsonRenderer",
    "REASONS",
    "Key",
    "Limits",
    "Mapping",
    "Object",
    "Opaque",
    "Out",
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
    "capture",
    "capture_fields",
    "merge",
    "preview_attributes",
    "render_json",
    "render_output",
    "render_python",
]
