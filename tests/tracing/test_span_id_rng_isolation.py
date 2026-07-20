# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Span-id generation must be immune to the global ``random`` module being reseeded.

Regression: user code executed via ``execute_python`` (e.g. a simulation that
calls ``random.seed(42)``) reseeds the process-global RNG. OpenTelemetry's
default ``RandomIdGenerator`` draws span/trace ids from that same global module,
so two executions that reseed to the same state were handed *identical* span
ids -> "Found N duplicate span_id(s) in trace" warnings in the trace explorer.

The fix: give the TracerProvider an id generator backed by its own
``random.Random`` instance, so ``random.seed()`` in user code cannot poison it.
"""

from __future__ import annotations

import random

from nooa.tracing import _IsolatedIdGenerator


def test_span_ids_unaffected_by_global_seed():
    """Re-seeding the global random module must not make two span ids identical."""
    gen = _IsolatedIdGenerator()

    random.seed(42)
    first = gen.generate_span_id()

    random.seed(42)
    second = gen.generate_span_id()

    assert first != second, (
        "span ids collided after re-seeding the global random module; "
        "the id generator must use its own RNG instance"
    )


def test_trace_ids_unaffected_by_global_seed():
    """Re-seeding the global random module must not make two trace ids identical."""
    gen = _IsolatedIdGenerator()

    random.seed(7)
    first = gen.generate_trace_id()
    random.seed(7)
    second = gen.generate_trace_id()

    assert first != second, (
        "trace ids collided after re-seeding the global random module; "
        "the id generator must use its own RNG instance"
    )


def test_collision_across_seeded_executions():
    """Two 'executions' that each seed(42) + draw the same values must still get
    distinct span ids from the tracer."""
    gen = _IsolatedIdGenerator()

    def one_execution() -> int:
        random.seed(42)
        for _ in range(100):
            random.random()  # noqa: S311 - intentional: simulate user reseeding/draws
        return gen.generate_span_id()

    assert one_execution() != one_execution()
