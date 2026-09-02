# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validation for the per-change agent interface evaluation ledger."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nooa_bench.behavior_analyzer import RATE_DESCRIPTIONS, SIGNAL_DESCRIPTIONS

_DIRECTIONS = {"increase", "decrease", "non_decreasing", "non_increasing", "unchanged"}
_STATUSES = {"proposed", "implemented", "reverted"}
_REQUIRED = {"id", "status", "component", "hypothesis", "deterministic_checks", "trace_expectations", "benchmark_slices"}


def load_change_ledger(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate an interface-change evaluation ledger."""
    data = json.loads(Path(path).read_text())
    if data.get("schema_version") != 1:
        raise ValueError("unsupported change-ledger schema_version")
    changes = data.get("changes")
    if not isinstance(changes, list) or not changes:
        raise ValueError("change ledger must contain a non-empty changes list")
    seen: set[str] = set()
    for index, change in enumerate(changes):
        if not isinstance(change, dict):
            raise ValueError(f"change {index} must be an object")
        missing = _REQUIRED - change.keys()
        if missing:
            raise ValueError(f"change {index} is missing fields: {sorted(missing)}")
        change_id = change["id"]
        if not isinstance(change_id, str) or not change_id or change_id in seen:
            raise ValueError(f"change id must be a unique non-empty string: {change_id!r}")
        seen.add(change_id)
        if change["status"] not in _STATUSES:
            raise ValueError(f"change {change_id!r} has invalid status")
        if not change["deterministic_checks"]:
            raise ValueError(f"change {change_id!r} has no deterministic checks")
        if not change["benchmark_slices"]:
            raise ValueError(f"change {change_id!r} has no benchmark slices")
        for expectation in change["trace_expectations"]:
            signal = expectation.get("signal")
            if signal not in SIGNAL_DESCRIPTIONS and signal not in RATE_DESCRIPTIONS:
                raise ValueError(f"change {change_id!r} references unknown signal {signal!r}")
            if expectation.get("direction") not in _DIRECTIONS:
                raise ValueError(f"change {change_id!r} has invalid expected direction")
    return data
