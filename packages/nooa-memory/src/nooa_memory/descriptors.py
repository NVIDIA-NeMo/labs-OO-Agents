# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verbal ordered descriptors — the single source of truth for the agent vocabulary.

The agent reads and writes ALL-CAPS ordered labels (e.g. ``importance=CRITICAL``);
the store and every scoring / decay / consolidation formula stay **numeric**. The
verbal<->numeric translation happens only at the agent-tool boundary
(``MemoryToolsMixin``) and when rendering a memory back to the agent — the memory
core never sees a label.

Each ladder is ordered high->low and chosen so the schema **defaults round-trip**:
importance 5.0=MEDIUM, salience 0.0=NONE, confidence 0.5=TENTATIVE, edge weight
1.0=STRONG — so existing behaviour (and the ``forgetting`` HIGH=8.0 protection
boundary) is unchanged.
"""

from __future__ import annotations

# kind -> ordered {LABEL: numeric}, highest band first.
_LADDERS: dict[str, dict[str, float]] = {
    "importance": {"CRITICAL": 10.0, "HIGH": 8.0, "MEDIUM": 5.0, "LOW": 3.0, "TRIVIAL": 1.0},
    "salience": {"PIVOTAL": 1.0, "NOTABLE": 0.5, "ROUTINE": 0.3, "NONE": 0.0},
    "confidence": {"CERTAIN": 1.0, "CONFIDENT": 0.75, "TENTATIVE": 0.5, "UNCERTAIN": 0.25},
    "edge_weight": {"STRONG": 1.0, "MODERATE": 0.6, "WEAK": 0.3},
}


def ladder(kind: str) -> tuple[str, ...]:
    """The ordered (highest-first) verbal labels for ``kind`` — e.g. for docstrings."""
    return tuple(_LADDERS[kind])


def to_numeric(kind: str, label: str) -> float:
    """Map a verbal label to its numeric value. Raise on an unknown label (no fallback)."""
    bands = _LADDERS[kind]
    if label not in bands:
        raise ValueError(f"unknown {kind} label {label!r}; expected one of {list(bands)}")
    return bands[label]


def to_label(kind: str, value: float) -> str:
    """Map a numeric value to its nearest band label (ties resolve to the higher band)."""
    bands = _LADDERS[kind]
    return min(bands, key=lambda lbl: (abs(bands[lbl] - value), -bands[lbl]))


# TODO lifecycle states — a state machine's states, not an ordered ladder.
# Agent-facing form is ALL-CAPS (OPEN | DONE | DROPPED); the stored form is lower.
TODO_STATUSES: tuple[str, ...] = ("open", "done", "dropped")


def to_status(label: str) -> str:
    """Map an agent-facing todo status to its stored form. Raise on unknown (no fallback)."""
    status = label.strip().lower()
    if status not in TODO_STATUSES:
        expected = " | ".join(s.upper() for s in TODO_STATUSES)
        raise ValueError(f"unknown todo status {label!r}; expected {expected}")
    return status
