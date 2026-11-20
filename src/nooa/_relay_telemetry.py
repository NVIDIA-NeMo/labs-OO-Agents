# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, detached JSON inspection values for Relay's native bridge."""

from __future__ import annotations

import json
from typing import Any

from nooa.snapshot import Limits, Unavailable, capture, render_trace


def _bridge_json(value: Any) -> Any:
    """Adapt only the already-bounded JSON copy to Rust's scalar limits."""
    if isinstance(value, str):
        return value.encode("utf-8", "backslashreplace").decode("utf-8")
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        # snapshot: allow Rust JSON integer range; normalize only the bounded protocol copy.
        return value if -(2**63) <= value < 2**64 else str(value)
    if isinstance(value, list):
        return [_bridge_json(item) for item in value]
    if isinstance(value, dict):
        return {_bridge_json(key): _bridge_json(item) for key, item in value.items()}
    return value


def inspection_snapshot(value: Any) -> tuple[Any, dict[str, Any]]:
    """Return bounded inspection JSON and standard snapshot preview metadata.

    Capture runs before encoding and never pickles or stringifies the original
    object. Small JSON values retain their shape; structured Python values use
    snapshot's visible fields and opaque fallback. The original is untouched.
    """
    limits = Limits()
    try:
        snapshot = capture(value, limits)
        text, attributes = render_trace(snapshot, direction="output", max_chars=limits.max_chars)
        return _bridge_json(json.loads(text)), attributes
    except Exception:
        # Telemetry runs after the method has finished; failure to inspect a
        # value must never replace that method's result or original exception.
        text, attributes = render_trace(
            Unavailable("error", "Python value"), direction="output", max_chars=limits.max_chars
        )
        return json.loads(text), attributes


def inspection_value(value: Any) -> Any:
    """Return the JSON portion of :func:`inspection_snapshot`."""
    return inspection_snapshot(value)[0]
