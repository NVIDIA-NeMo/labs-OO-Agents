# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The span attributes of ``render_trace``, for tests that check only them."""

from __future__ import annotations

import sys
from typing import Any

from nooa.snapshot import Snapshot, render_trace


def _attributes(direction: str, snapshot: Snapshot, max_chars: int = sys.maxsize) -> dict[str, Any]:
    """``render_trace``'s attributes (``sys.maxsize``: no cut made while rendering)."""
    return render_trace(snapshot, direction=direction, max_chars=max_chars)[1]
